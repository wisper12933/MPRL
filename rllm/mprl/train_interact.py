from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from mprl.planned_interact_agent import PlannedInteractAgent
from mprl.task_specs import (
    DEFAULT_ALFWORLD_CONFIG,
    DEFAULT_ALFWORLD_DATA,
    DEFAULT_SCIWORLD_JAR,
    build_env_args,
    get_task_spec,
    load_task_dataset,
    resolve_env_class,
)
from rllm.trainer import AgentTrainer


def _configured_path(value, default: Path) -> Path:
    return Path(value) if value else default


@hydra.main(version_base=None, config_path="../rllm/trainer/config", config_name="mprl_swift_trainer")
def main(config: DictConfig) -> None:
    task = str(config.mprl.task).lower()
    spec = get_task_spec(task)
    env_class = resolve_env_class(task)

    alfworld_data = _configured_path(config.mprl.alfworld_data, DEFAULT_ALFWORLD_DATA)
    alfworld_config = _configured_path(config.mprl.alfworld_config, DEFAULT_ALFWORLD_CONFIG)
    sciworld_jar = _configured_path(config.mprl.sciworld_jar, DEFAULT_SCIWORLD_JAR)
    max_steps = int(config.mprl.max_steps or spec.max_steps)
    n_parallel_agents = int(config.mprl.n_parallel_agents or spec.n_parallel_agents)
    adapter_path = Path(config.model.adapter_path) if config.model.adapter_path else spec.adapter_path

    for required_path, description in (
        (Path(config.model.name), "base model"),
        (adapter_path, f"{task} SFT adapter"),
        (spec.instruction_path, f"{task} interaction prompt"),
        (spec.metaplan_prompt_path, f"{task} meta-plan prompt"),
        (spec.train_index_path, f"{task} train indices"),
        (spec.test_index_path, f"{task} test indices"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {description}: {required_path}")

    train_dataset = load_task_dataset(
        task,
        "train",
        alfworld_data=alfworld_data,
        limit=config.mprl.train_limit,
    )
    val_dataset = load_task_dataset(
        task,
        "test",
        alfworld_data=alfworld_data,
        limit=config.mprl.test_limit,
    )
    env_args = build_env_args(
        task,
        max_steps=max_steps,
        split="train",
        alfworld_config=alfworld_config,
        sciworld_jar=sciworld_jar,
    )
    agent_args = {
        "base_prompt_path": str(spec.instruction_path),
        "metaplan_prompt_path": str(spec.metaplan_prompt_path),
        "planning_enabled": bool(config.planning.enabled),
        "planning_max_tokens": int(config.planning.max_tokens),
        "planning_temperature": float(config.planning.temperature),
        "planning_top_p": float(config.planning.top_p),
    }

    OmegaConf.update(config, "model.adapter_path", str(adapter_path), force_add=True)
    OmegaConf.update(config, "model.use_lora", True, force_add=True)
    OmegaConf.update(config, "agent.max_steps", max_steps, force_add=True)
    OmegaConf.update(config, "agent.n_parallel_agents", n_parallel_agents, force_add=True)

    print(
        json.dumps(
            {
                "task": task,
                "base_model": config.model.name,
                "adapter_path": config.model.adapter_path,
                "train_tasks": len(train_dataset),
                "validation_tasks": len(val_dataset),
                "max_steps": max_steps,
                "n_parallel_agents": n_parallel_agents,
                "planning_enabled": bool(config.planning.enabled),
                "interaction_prompt": str(spec.instruction_path),
                "metaplan_prompt": str(spec.metaplan_prompt_path),
                "env_args": env_args,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    trainer = AgentTrainer(
        config=config,
        agent_class=PlannedInteractAgent,
        env_class=env_class,
        agent_args=agent_args,
        env_args=env_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="swift",
    )
    trainer.train()


if __name__ == "__main__":
    main()