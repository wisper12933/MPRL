"""Train DeepScaler with the Swift backend (ms-swift style, no Ray).

Recommended rollout path: start a vLLM OpenAI server, then set
``rollout.mode=server`` so sampling is concurrent (not serial HF generate).
"""

import hydra
from omegaconf import DictConfig

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer import AgentTrainer


@hydra.main(version_base=None, config_path="../../rllm/trainer/config", config_name="swift_rl_trainer")
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

    if train_dataset is None or test_dataset is None:
        raise ValueError("Datasets not found. Run: python examples/deepscaler/prepare_math_data.py")

    trainer = AgentTrainer(
        config=config,
        agent_class=MathAgent,
        env_class=SingleTurnEnvironment,
        agent_args={},
        env_args={"reward_fn": math_reward_fn},
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        backend="swift",
    )
    trainer.train()


if __name__ == "__main__":
    main()
