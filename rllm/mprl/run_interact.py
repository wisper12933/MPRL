import argparse
import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer

from mprl.planned_interact_agent import PlannedInteractAgent
from mprl.task_specs import (
    DEFAULT_ALFWORLD_CONFIG,
    DEFAULT_ALFWORLD_DATA,
    DEFAULT_BASE_MODEL,
    DEFAULT_SCIWORLD_JAR,
    REPO_ROOT,
    RLLM_ROOT,
    TASK_SPECS,
    build_env_args,
    get_task_spec,
    load_task_dataset,
    resolve_env_class,
)
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k

DEFAULT_BASE_URL = "http://localhost:30000/v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Run local RL interaction sampling for ALFWorld / SciWorld / WebShop.")
    parser.add_argument("--task", choices=sorted(TASK_SPECS.keys()), required=True, help="Task/environment to sample.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible rollout endpoint.")
    parser.add_argument("--api-key", default="None", help="API key for the rollout endpoint.")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL), help="Base model path or identifier used for tokenizer loading.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path. Defaults to --base-model.")
    parser.add_argument("--model-alias", required=True, help="Served model / LoRA alias exposed by vLLM.")
    parser.add_argument("--test-index-path", default=None, help="Override the default test index JSON path for the selected task.")
    parser.add_argument("--prompt-path", default=None, help="Override the default prompt file path for the selected task.")
    parser.add_argument("--metaplan-prompt-path", default=None, help="Override the task-specific meta-plan template.")
    parser.add_argument(
        "--planning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate and inject a context-only meta-plan before interaction (default: enabled).",
    )
    parser.add_argument("--planning-max-tokens", type=int, default=1024)
    parser.add_argument("--planning-temperature", type=float, default=0.1)
    parser.add_argument("--planning-top-p", type=float, default=0.9)
    parser.add_argument("--repeat-k", type=int, default=2, help="How many times to repeat each unique task for pass@k estimation.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of unique tasks to sample. <=0 means all tasks.")
    parser.add_argument("--n-parallel-agents", type=int, default=None, help="Number of concurrent environment-agent workers.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override task-specific maximum steps / turns.")
    parser.add_argument("--max-prompt-length", type=int, default=6144, help="Maximum prompt length sent to the rollout engine.")
    parser.add_argument("--max-response-length", type=int, default=1024, help="Maximum response length from the rollout engine.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value.")
    parser.add_argument("--alfworld-data", default=str(Path(os.environ.get("ALFWORLD_DATA", DEFAULT_ALFWORLD_DATA))), help="ALFWorld data root used to rewrite task game_file paths.")
    parser.add_argument("--alfworld-config-path", default=str(DEFAULT_ALFWORLD_CONFIG), help="ALFWorld base config path.")
    parser.add_argument("--alfworld-split", default="eval_out_of_distribution", help="ALFWorld split passed to the env wrapper.")
    parser.add_argument("--sciworld-server-path", default=str(DEFAULT_SCIWORLD_JAR), help="SciWorld server jar path.")
    return parser.parse_args()


def prepare_tasks(args, task_name: str, index_path: Path):
    dataset = load_task_dataset(
        task_name,
        "test",
        index_path=index_path,
        alfworld_data=args.alfworld_data,
        limit=args.limit,
    )
    unique_tasks = dataset.get_data()

    repeated_tasks = []
    for _ in range(max(args.repeat_k, 1)):
        repeated_tasks.extend(deepcopy(task) for task in unique_tasks)

    return unique_tasks, repeated_tasks


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    task_name = args.task.lower()
    spec = get_task_spec(task_name)
    env_class = resolve_env_class(task_name)
    prompt_path = Path(args.prompt_path) if args.prompt_path else spec.instruction_path
    metaplan_prompt_path = Path(args.metaplan_prompt_path) if args.metaplan_prompt_path else spec.metaplan_prompt_path
    index_path = Path(args.test_index_path) if args.test_index_path else spec.test_index_path
    max_steps = args.max_steps if args.max_steps is not None else spec.max_steps
    n_parallel_agents = args.n_parallel_agents if args.n_parallel_agents is not None else spec.n_parallel_agents
    env_args = build_env_args(
        task_name,
        max_steps=max_steps,
        split="test",
        alfworld_config=args.alfworld_config_path,
        sciworld_jar=args.sciworld_server_path,
    )
    if task_name == "alfworld":
        env_args["split"] = args.alfworld_split
    tokenizer_path = args.tokenizer_path or args.base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    agent_args = {
        "base_prompt_path": str(prompt_path),
        "metaplan_prompt_path": str(metaplan_prompt_path),
        "planning_enabled": args.planning,
        "planning_max_tokens": args.planning_max_tokens,
        "planning_temperature": args.planning_temperature,
        "planning_top_p": args.planning_top_p,
    }
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "model": args.model_alias,
    }
    rollout_engine_args = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model_alias,
    }

    unique_tasks, tasks = prepare_tasks(args, task_name, index_path)

    summary = {
        "task": task_name,
        "repo_root": str(REPO_ROOT),
        "rllm_root": str(RLLM_ROOT),
        "env_class": env_class.__name__,
        "prompt_path": str(prompt_path),
        "metaplan_prompt_path": str(metaplan_prompt_path),
        "planning_enabled": args.planning,
        "test_index_path": str(index_path),
        "base_model": args.base_model,
        "tokenizer_path": tokenizer_path,
        "model_alias": args.model_alias,
        "base_url": args.base_url,
        "n_parallel_agents": n_parallel_agents,
        "max_steps": max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repeat_k": max(args.repeat_k, 1),
        "unique_task_count": len(unique_tasks),
        "scheduled_trajectories": len(tasks),
        "env_args": env_args,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    engine = AgentExecutionEngine(
        agent_class=PlannedInteractAgent,
        env_class=env_class,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args=rollout_engine_args,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        max_steps=max_steps,
        n_parallel_agents=n_parallel_agents,
    )

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results)


if __name__ == "__main__":
    main()
