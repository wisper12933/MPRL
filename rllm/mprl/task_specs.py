from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rllm.data.dataset import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
RLLM_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
INDICES_ROOT = DATA_ROOT / "indices"
INSTRUCTION_ROOT = DATA_ROOT / "instructions"
DEFAULT_BASE_MODEL = Path("/mnt/hdfs/lixingzuo/qwen_model/origin/Qwen3-4B-Instruct")
DEFAULT_ADAPTER_ROOT = Path("/mnt/hdfs/lixingzuo/qwen_model/sft/Qwen3-4B-Instruct/MPRL-lora")
DEFAULT_ALFWORLD_DATA = DATA_ROOT / "alfworld_data"
DEFAULT_ALFWORLD_CONFIG = REPO_ROOT / "envs" / "alfworld" / "base_config.yaml"
DEFAULT_SCIWORLD_JAR = INDICES_ROOT / "sciworld" / "scienceworld.jar"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_module: str
    env_class: str
    adapter_dirname: str
    instruction_path: Path
    metaplan_prompt_path: Path
    train_index_path: Path
    test_index_path: Path
    max_steps: int
    n_parallel_agents: int

    @property
    def adapter_path(self) -> Path:
        return DEFAULT_ADAPTER_ROOT / self.adapter_dirname


TASK_SPECS = {
    "webshop": TaskSpec(
        name="webshop",
        env_module="rllm.environments.interact.webshop_env",
        env_class="WebShopEnv",
        adapter_dirname="Qwen3-4B-Instruct-MAML-plan-sft-web",
        instruction_path=INSTRUCTION_ROOT / "webshop_inst.txt",
        metaplan_prompt_path=INSTRUCTION_ROOT / "metaplan" / "webshop.txt",
        train_index_path=INDICES_ROOT / "webshop" / "train_indices.json",
        test_index_path=INDICES_ROOT / "webshop" / "test_indices.json",
        max_steps=12,
        n_parallel_agents=2,
    ),
    "alfworld": TaskSpec(
        name="alfworld",
        env_module="rllm.environments.interact.alfworld_env",
        env_class="ALFWorldEnv",
        adapter_dirname="Qwen3-4B-Instruct-MAML-plan-sft-alf",
        instruction_path=INSTRUCTION_ROOT / "alfworld_inst.txt",
        metaplan_prompt_path=INSTRUCTION_ROOT / "metaplan" / "alfworld.txt",
        train_index_path=INDICES_ROOT / "alfworld" / "train_indices.json",
        test_index_path=INDICES_ROOT / "alfworld" / "test_indices.json",
        max_steps=40,
        n_parallel_agents=2,
    ),
    "sciworld": TaskSpec(
        name="sciworld",
        env_module="rllm.environments.interact.sciworld_env",
        env_class="SciWorldEnv",
        adapter_dirname="Qwen3-4B-Instruct-MAML-plan-sft-sci",
        instruction_path=INSTRUCTION_ROOT / "sciworld_inst.txt",
        metaplan_prompt_path=INSTRUCTION_ROOT / "metaplan" / "sciworld.txt",
        train_index_path=INDICES_ROOT / "sciworld" / "train_indices.json",
        test_index_path=INDICES_ROOT / "sciworld" / "test_indices.json",
        max_steps=60,
        n_parallel_agents=1,
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    task = task.lower()
    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported task {task!r}; choose from {sorted(TASK_SPECS)}")
    return TASK_SPECS[task]


def resolve_env_class(task: str):
    spec = get_task_spec(task)
    module = importlib.import_module(spec.env_module)
    return getattr(module, spec.env_class)


def _normalize_webshop(raw_tasks: list[Any]) -> list[dict[str, Any]]:
    tasks = []
    for item in raw_tasks:
        value = item.get("id") if isinstance(item, dict) else item
        if value is None:
            raise ValueError(f"WebShop task is missing id: {item!r}")
        tasks.append({"id": int(value)})
    return tasks


def _normalize_alfworld(raw_tasks: list[Any], data_root: Path) -> list[dict[str, Any]]:
    tasks = []
    for item in raw_tasks:
        if not isinstance(item, dict) or "game_file" not in item:
            raise ValueError(f"ALFWorld task must contain game_file: {item!r}")
        task = dict(item)
        game_file = str(task["game_file"])
        if "/alfworld_data/" in game_file:
            game_file = game_file.split("/alfworld_data/", 1)[1]
        path = Path(game_file)
        if not path.is_absolute():
            path = data_root / path
        task["game_file"] = str(path)
        tasks.append(task)
    return tasks


def _normalize_sciworld(raw_tasks: list[Any]) -> list[dict[str, Any]]:
    tasks = []
    for item in raw_tasks:
        if isinstance(item, dict):
            if "task_name" not in item or "variation_idx" not in item:
                raise ValueError(f"SciWorld task is missing task_name/variation_idx: {item!r}")
            task_name, variation_idx = item["task_name"], item["variation_idx"]
        elif isinstance(item, list) and len(item) == 2:
            task_name, variation_idx = item
        else:
            raise ValueError(f"SciWorld task must be [task_name, variation_idx]: {item!r}")
        tasks.append({"task_name": str(task_name), "variation_idx": int(variation_idx)})
    return tasks


def normalize_tasks(task: str, raw_tasks: list[Any], alfworld_data: Path | None = None) -> list[dict[str, Any]]:
    task = task.lower()
    if task == "webshop":
        return _normalize_webshop(raw_tasks)
    if task == "alfworld":
        return _normalize_alfworld(raw_tasks, Path(alfworld_data or DEFAULT_ALFWORLD_DATA))
    if task == "sciworld":
        return _normalize_sciworld(raw_tasks)
    raise ValueError(f"Unsupported task: {task}")


def load_task_dataset(
    task: str,
    split: str,
    *,
    index_path: str | Path | None = None,
    alfworld_data: str | Path | None = None,
    limit: int | None = None,
) -> Dataset:
    spec = get_task_spec(task)
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = Path(index_path) if index_path else (spec.train_index_path if split == "train" else spec.test_index_path)
    with path.open(encoding="utf-8") as handle:
        raw_tasks = json.load(handle)
    tasks = normalize_tasks(task, raw_tasks, Path(alfworld_data) if alfworld_data else None)
    if limit is not None and limit > 0:
        tasks = tasks[:limit]
    return Dataset(tasks, name=f"mprl-{task}", split=split)


def build_env_args(
    task: str,
    *,
    max_steps: int | None = None,
    split: str = "train",
    alfworld_config: str | Path = DEFAULT_ALFWORLD_CONFIG,
    sciworld_jar: str | Path = DEFAULT_SCIWORLD_JAR,
) -> dict[str, Any]:
    spec = get_task_spec(task)
    max_turns = max_steps or spec.max_steps
    if task == "webshop":
        return {"max_turns": max_turns}
    if task == "alfworld":
        alfworld_split = "train" if split == "train" else "eval_out_of_distribution"
        return {"max_turns": max_turns, "config_path": str(alfworld_config), "split": alfworld_split}
    if task == "sciworld":
        return {"max_turns": max_turns, "server_path": str(sciworld_jar)}
    raise ValueError(f"Unsupported task: {task}")
