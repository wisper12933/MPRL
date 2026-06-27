import argparse
import asyncio
import importlib
import json
import os
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer

from rllm.agents.interact_agent import InteractAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.utils import compute_pass_at_k


REPO_ROOT = Path(__file__).resolve().parents[2]
RLLM_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
INSTRUCTION_ROOT = DATA_ROOT / "instructions"
INDICES_ROOT = DATA_ROOT / "indices"
DEFAULT_ALFWORLD_DATA = REPO_ROOT / "data" / "alfworld_data"
DEFAULT_ALFWORLD_CONFIG = REPO_ROOT / "envs" / "alfworld" / "base_config.yaml"
DEFAULT_SCIWORLD_JAR = INDICES_ROOT / "sciworld" / "scienceworld.jar"
DEFAULT_BASE_URL = "http://localhost:30000/v1"
DEFAULT_BASE_MODEL = "/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct"


TASK_SPECS = {
    "webshop": {
        "env_module": "rllm.environments.interact.webshop_env",
        "env_class": "WebShopEnv",
        "prompt_path": INSTRUCTION_ROOT / "webshop_inst.txt",
        "test_index_path": INDICES_ROOT / "webshop" / "test_indices.json",
        "default_max_steps": 12,
        "default_n_parallel_agents": 2,
    },
    "alfworld": {
        "env_module": "rllm.environments.interact.alfworld_env",
        "env_class": "ALFWorldEnv",
        "prompt_path": INSTRUCTION_ROOT / "alfworld_inst.txt",
        "test_index_path": INDICES_ROOT / "alfworld" / "test_indices.json",
        "default_max_steps": 40,
        "default_n_parallel_agents": 2,
    },
    "sciworld": {
        "env_module": "rllm.environments.interact.sciworld_env",
        "env_class": "SciWorldEnv",
        "prompt_path": INSTRUCTION_ROOT / "sciworld_inst.txt",
        "test_index_path": INDICES_ROOT / "sciworld" / "test_indices.json",
        "default_max_steps": 60,
        "default_n_parallel_agents": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run local RL interaction sampling for ALFWorld / SciWorld / WebShop.")
    parser.add_argument("--task", choices=sorted(TASK_SPECS.keys()), required=True, help="Task/environment to sample.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible rollout endpoint.")
    parser.add_argument("--api-key", default="None", help="API key for the rollout endpoint.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model path or identifier used for tokenizer loading.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path. Defaults to --base-model.")
    parser.add_argument("--model-alias", required=True, help="Served model / LoRA alias exposed by vLLM.")
    parser.add_argument("--test-index-path", default=None, help="Override the default test index JSON path for the selected task.")
    parser.add_argument("--prompt-path", default=None, help="Override the default prompt file path for the selected task.")
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


def resolve_env_class(task_name: str):
    spec = TASK_SPECS[task_name]
    module = importlib.import_module(spec["env_module"])
    return getattr(module, spec["env_class"])


def load_raw_tasks(index_path: Path):
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_webshop_tasks(raw_tasks):
    tasks = []
    for item in raw_tasks:
        if isinstance(item, dict):
            if "id" not in item:
                raise ValueError(f"WebShop task is missing 'id': {item}")
            tasks.append({"id": int(item["id"])})
        else:
            tasks.append({"id": int(item)})
    return tasks


def normalize_alfworld_tasks(raw_tasks, alfworld_data_root: Path):
    tasks = []
    for item in raw_tasks:
        if not isinstance(item, dict) or "game_file" not in item:
            raise ValueError(f"ALFWorld task must be a dict with game_file: {item}")
        task = dict(item)
        game_file = str(task["game_file"])
        if "/alfworld_data/" in game_file:
            suffix = game_file.split("/alfworld_data/", 1)[1]
            task["game_file"] = str(alfworld_data_root / suffix)
        elif not os.path.isabs(game_file):
            task["game_file"] = str(alfworld_data_root / game_file)
        tasks.append(task)
    return tasks


def normalize_sciworld_tasks(raw_tasks):
    tasks = []
    for item in raw_tasks:
        if isinstance(item, dict):
            if "task_name" not in item or "variation_idx" not in item:
                raise ValueError(f"SciWorld task dict must contain task_name and variation_idx: {item}")
            tasks.append({"task_name": item["task_name"], "variation_idx": int(item["variation_idx"])})
            continue
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"SciWorld task must be a [task_name, variation_idx] pair: {item}")
        tasks.append({"task_name": item[0], "variation_idx": int(item[1])})
    return tasks


def build_env_args(args, task_name: str, max_steps: int):
    if task_name == "webshop":
        return {"max_turns": max_steps}
    if task_name == "alfworld":
        return {
            "max_turns": max_steps,
            "config_path": args.alfworld_config_path,
            "split": args.alfworld_split,
        }
    if task_name == "sciworld":
        return {
            "max_turns": max_steps,
            "server_path": args.sciworld_server_path,
        }
    raise ValueError(f"Unsupported task: {task_name}")


def prepare_tasks(args, task_name: str, index_path: Path):
    raw_tasks = load_raw_tasks(index_path)
    if task_name == "webshop":
        unique_tasks = normalize_webshop_tasks(raw_tasks)
    elif task_name == "alfworld":
        unique_tasks = normalize_alfworld_tasks(raw_tasks, Path(args.alfworld_data))
    elif task_name == "sciworld":
        unique_tasks = normalize_sciworld_tasks(raw_tasks)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    if args.limit > 0:
        unique_tasks = unique_tasks[: args.limit]

    repeated_tasks = []
    for _ in range(max(args.repeat_k, 1)):
        repeated_tasks.extend(deepcopy(task) for task in unique_tasks)

    return unique_tasks, repeated_tasks


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    task_name = args.task.lower()
    spec = TASK_SPECS[task_name]
    env_class = resolve_env_class(task_name)
    prompt_path = Path(args.prompt_path) if args.prompt_path else Path(spec["prompt_path"])
    index_path = Path(args.test_index_path) if args.test_index_path else Path(spec["test_index_path"])
    max_steps = args.max_steps if args.max_steps is not None else spec["default_max_steps"]
    n_parallel_agents = args.n_parallel_agents if args.n_parallel_agents is not None else spec["default_n_parallel_agents"]
    env_args = build_env_args(args, task_name, max_steps)
    tokenizer_path = args.tokenizer_path or args.base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    agent_args = {"base_prompt_path": str(prompt_path)}
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
        agent_class=InteractAgent,
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
