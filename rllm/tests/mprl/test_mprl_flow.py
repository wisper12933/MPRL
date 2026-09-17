import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from mprl.planned_interact_agent import PlannedInteractAgent
from mprl.task_specs import TASK_SPECS, load_task_dataset, normalize_tasks
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout.swift_engine import SwiftEngine
from rllm.environments.interact.sciworld_env import SciWorldEnv, _ScienceWorldPortLogFilter
from rllm.trainer.swift.swift_policy_trainer import SwiftPolicyTrainer


def test_swift_minibatches_have_fixed_collective_count_across_ranks():
    rank_zero = SwiftPolicyTrainer._split_into_exact_minibatches(list(range(8)), 8)
    rank_one = SwiftPolicyTrainer._split_into_exact_minibatches(list(range(11)), 8)

    assert len(rank_zero) == len(rank_one) == 8
    assert all(rank_zero)
    assert all(rank_one)
    assert [item for chunk in rank_one for item in chunk] == list(range(11))


def _agent(tmp_path: Path) -> PlannedInteractAgent:
    interaction = tmp_path / "interaction.txt"
    metaplan = tmp_path / "metaplan.txt"
    interaction.write_text("Reply with Thought and Action.", encoding="utf-8")
    metaplan.write_text("Plan this task:\n<task>{{TASK}}</task>", encoding="utf-8")
    return PlannedInteractAgent(str(interaction), str(metaplan), planning_max_tokens=77)


def test_all_task_metaplan_templates_have_task_slot():
    for spec in TASK_SPECS.values():
        text = spec.metaplan_prompt_path.read_text(encoding="utf-8")
        assert "{{TASK}}" in text
        assert "<workflow>" in text


def test_context_only_plan_is_injected_once_and_not_added_as_step(tmp_path):
    agent = _agent(tmp_path)
    agent.update_from_env("Find the target.", reward=0.0, done=False, info={})
    engine = object.__new__(AgentExecutionEngine)
    observed = {}

    async def fake_response(messages, application_id, **kwargs):
        observed.update(messages=messages, application_id=application_id, kwargs=kwargs)
        return SimpleNamespace(text="<workflow>\nStep 1: inspect\n</workflow>")

    engine.get_model_response = fake_response
    elapsed = asyncio.run(engine._run_initial_planning(agent, "Find the target.", {}, "episode-1"))

    assert elapsed >= 0
    assert observed["kwargs"]["max_tokens"] == 77
    assert agent.generated_plan == "<workflow>\nStep 1: inspect\n</workflow>"
    assert agent.generated_plan in agent.chat_completions[-1]["content"]
    assert agent.trajectory.steps == []
    with pytest.raises(RuntimeError, match="already"):
        agent.inject_plan("another plan")


def test_explicitly_disabled_planning_skips_generation_and_injection(tmp_path):
    agent = _agent(tmp_path)
    agent.planning_enabled = False
    agent.update_from_env("Find the target.", reward=0.0, done=False, info={})
    initial_messages = list(agent.chat_completions)
    engine = object.__new__(AgentExecutionEngine)

    async def unexpected_response(*args, **kwargs):
        raise AssertionError("Planning generation must not run when explicitly disabled")

    engine.get_model_response = unexpected_response
    elapsed = asyncio.run(engine._run_initial_planning(agent, "Find the target.", {}, "episode-1"))

    assert elapsed == 0.0
    assert agent.generated_plan is None
    assert agent.chat_completions == initial_messages
    assert "metaplan" not in agent.trajectory.info


def test_planning_and_action_requests_get_different_batch_keys():
    engine = object.__new__(SwiftEngine)
    engine.sampling_params = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 512}
    engine.max_response_length = 4096
    action = SimpleNamespace(kwargs={})
    planning = SimpleNamespace(kwargs={"temperature": 0.1, "top_p": 0.9, "max_tokens": 1024})

    assert engine._request_batch_key(action) != engine._request_batch_key(planning)


def test_task_indices_are_normalized_and_alfworld_paths_are_rewritten(tmp_path):
    webshop = normalize_tasks("webshop", [3, {"id": "4"}])
    sciworld = normalize_tasks("sciworld", [["task-1-boil", 2]])
    alfworld = normalize_tasks(
        "alfworld",
        [{"game_file": "/old/root/alfworld_data/json_2.1.1/train/example/game.tw-pddl"}],
        tmp_path,
    )
    assert webshop == [{"id": 3}, {"id": 4}]
    assert sciworld == [{"task_name": "task-1-boil", "variation_idx": 2}]
    assert alfworld[0]["game_file"] == str(tmp_path / "json_2.1.1/train/example/game.tw-pddl")

    for task in TASK_SPECS:
        assert len(load_task_dataset(task, "train", limit=1)) == 1
        assert len(load_task_dataset(task, "test", limit=1)) == 1


def test_gradient_checkpointing_keeps_lora_graph_connected():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, LlamaConfig

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = get_peft_model(
        AutoModelForCausalLM.from_config(config),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )

    trainer = object.__new__(SwiftPolicyTrainer)
    trainer.model = model
    trainer.config = OmegaConf.create({"model": {"gradient_checkpointing": True}})
    trainer._enable_gradient_checkpointing()
    trainer.model.train()

    logits = trainer.model(input_ids=torch.randint(0, 64, (1, 8))).logits
    assert logits.requires_grad
    logits.sum().backward()
    assert any(p.grad is not None for p in trainer.model.parameters() if p.requires_grad)


def test_existing_adapter_is_loaded_trainable(monkeypatch):
    from peft import PeftModel

    adapter_path = TASK_SPECS["webshop"].adapter_path
    base_model = object()
    loaded_model = object()
    call = {}

    def fake_from_pretrained(model, path, is_trainable=False):
        call.update(model=model, path=path, is_trainable=is_trainable)
        return loaded_model

    monkeypatch.setattr(PeftModel, "from_pretrained", fake_from_pretrained)
    trainer = object.__new__(SwiftPolicyTrainer)
    trainer.config = OmegaConf.create({"model": {"adapter_path": str(adapter_path), "use_lora": True}})
    trainer.accelerator = None

    assert trainer._apply_lora(base_model) is loaded_model
    assert call == {"model": base_model, "path": str(adapter_path), "is_trainable": True}


class _FakeConnection:
    def __init__(self, result):
        self.result = result
        self.sent = []

    def send(self, value):
        self.sent.append(value)

    def recv(self):
        return self.result


def test_sciworld_returns_delta_reward_instead_of_cumulative_score():
    env = object.__new__(SciWorldEnv)
    env.parent_conn = _FakeConnection(("ok", 0.25, False, {"raw_score": 0.75}))
    env.lock = threading.Lock()
    env.current_turn = 0
    env.max_turns = 60
    env.done = False
    env.error_steps = 0
    env.max_error_steps = 5
    env.invalid_steps = 0
    env.max_invalid_steps = 5

    observation, reward, done, info = env.step("look around")

    assert observation == "Observation: ok"
    assert reward == 0.25
    assert info["raw_score"] == 0.75
    assert done is False


def test_sciworld_port_log_filter_repairs_third_party_format_string():
    record = logging.LogRecord(
        name="scienceworld.scienceworld",
        level=logging.INFO,
        pathname="scienceworld.py",
        lineno=51,
        msg="ScienceWorld server running on port",
        args=(43419,),
        exc_info=None,
    )

    assert _ScienceWorldPortLogFilter().filter(record)
    assert record.getMessage() == "ScienceWorld server running on port 43419"
