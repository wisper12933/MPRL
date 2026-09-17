"""Data processing for TRL-style agent RL training."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from rllm.agents.agent import Step, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryGroup:
    trajectories: list[Trajectory]
    group_id: str | None = None


@dataclass
class TrlTrainingSample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    mask: torch.Tensor


class TrlAdvantageComputer:
    def __init__(self, algorithm_config):
        self.adv_estimator = algorithm_config.adv_estimator
        self.norm_by_std = algorithm_config.get("norm_adv_by_std_in_grpo", True)

    def compute_grpo_advantages(self, group_rewards: list[float]) -> list[float]:
        if not group_rewards:
            return []
        if len(group_rewards) == 1:
            return group_rewards
        mean_reward = sum(group_rewards) / len(group_rewards)
        advantages = [reward - mean_reward for reward in group_rewards]
        if self.norm_by_std and len(advantages) > 1:
            std = np.std(advantages)
            if std > 1e-8:
                advantages = [adv / std for adv in advantages]
        return advantages

    def compute(self, group_rewards: list[float]) -> list[float]:
        if self.adv_estimator == "grpo":
            return self.compute_grpo_advantages(group_rewards)
        if self.adv_estimator == "reinforce":
            return group_rewards
        logger.warning("Unknown advantage estimator %s, using GRPO", self.adv_estimator)
        return self.compute_grpo_advantages(group_rewards)


class TrlTrajectoryFilter:
    def __init__(self, algorithm_config):
        self.remove_constant_reward_groups = algorithm_config.get("remove_constant_reward_groups", False)

    @staticmethod
    def _all_same(values: list[float]) -> bool:
        if not values:
            return True
        first = values[0]
        return all(abs(value - first) < 1e-8 for value in values)

    def filter_groups(self, groups: list[TrajectoryGroup]) -> list[TrajectoryGroup]:
        if not self.remove_constant_reward_groups:
            return groups

        filtered = []
        for group in groups:
            rewards = [traj.reward for traj in group.trajectories]
            if not self._all_same(rewards):
                filtered.append(group)

        if not filtered:
            logger.warning("All groups have uniform rewards; keeping one group to avoid an empty batch.")
            return groups[:1]
        return filtered


class TrlSampleBuilder:
    @staticmethod
    def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
        return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1

    @staticmethod
    def build_from_trajectory(trajectory: Trajectory, advantage: float) -> list[TrlTrainingSample]:
        if not trajectory.steps:
            return []

        class SequenceAccumulator:
            def __init__(self):
                self.full_sequence: list[int] = []
                self.logprobs: list[float] = []
                self.advantages: list[float] = []
                self.mask: list[float] = []

            def is_empty(self) -> bool:
                return not self.full_sequence

            def clear(self) -> None:
                self.full_sequence = []
                self.logprobs = []
                self.advantages = []
                self.mask = []

            def add_step(self, step: Step, adv: float, is_extension: bool = False) -> None:
                if is_extension:
                    prev_len = len(self.full_sequence)
                    delta_prompt = step.prompt_ids[prev_len:]
                else:
                    delta_prompt = step.prompt_ids
                delta_prompt_len = len(delta_prompt)

                self.full_sequence.extend(delta_prompt)
                self.logprobs.extend([0.0] * delta_prompt_len)
                self.advantages.extend([0.0] * delta_prompt_len)
                self.mask.extend([0.0] * delta_prompt_len)

                response_logprobs = step.logprobs or [0.0] * len(step.response_ids)
                self.full_sequence.extend(step.response_ids)
                self.logprobs.extend(response_logprobs)
                self.advantages.extend([adv] * len(step.response_ids))
                self.mask.extend([1.0] * len(step.response_ids))

            def to_sample(self) -> TrlTrainingSample:
                input_tokens = self.full_sequence[:-1]
                target_tokens = self.full_sequence[1:]
                shifted_logprobs = self.logprobs[1:]
                shifted_advantages = self.advantages[1:]
                shifted_mask = self.mask[1:]
                return TrlTrainingSample(
                    input_ids=torch.tensor(input_tokens, dtype=torch.long),
                    target_ids=torch.tensor(target_tokens, dtype=torch.long),
                    old_logprobs=torch.tensor(shifted_logprobs, dtype=torch.float32),
                    advantages=torch.tensor(shifted_advantages, dtype=torch.float32),
                    mask=torch.tensor(shifted_mask, dtype=torch.float32),
                )

        datums: list[TrlTrainingSample] = []
        accumulator = SequenceAccumulator()

        for step in trajectory.steps:
            if accumulator.is_empty():
                accumulator.add_step(step, advantage, is_extension=False)
            elif TrlSampleBuilder._is_prefix(accumulator.full_sequence, step.prompt_ids):
                accumulator.add_step(step, advantage, is_extension=True)
            else:
                datums.append(accumulator.to_sample())
                accumulator.clear()
                accumulator.add_step(step, advantage, is_extension=False)

        if not accumulator.is_empty():
            datums.append(accumulator.to_sample())
        return datums


def _episode_group_id(episode, index: int) -> str:
    episode_id = getattr(episode, "id", "") or ""
    if episode_id:
        return episode_id

    task = getattr(episode, "task", None)
    if isinstance(task, dict):
        if task.get("uid"):
            return str(task["uid"])
        if task.get("question"):
            return f"question:{task['question'][:80]}"

    return f"fallback-episode-{index}"


def process_episodes(
    episodes: list,
    advantage_computer: TrlAdvantageComputer,
    trajectory_filter: TrlTrajectoryFilter,
    algorithm_config,
) -> tuple[list[TrlTrainingSample], dict]:
    grouping_level = algorithm_config.get("grouping_level", "episode")
    trajectory_groups_dict: dict = defaultdict(list)

    def get_task_id(episode, episode_index: int) -> str:
        episode_id = _episode_group_id(episode, episode_index)
        return ":".join(episode_id.split(":")[:-1]) if ":" in episode_id else episode_id

    for episode_index, episode in enumerate(episodes):
        task_id = get_task_id(episode, episode_index)
        if grouping_level == "trajectory":
            for traj_index, trajectory in enumerate(episode.trajectories):
                trajectory_name = trajectory.name or f"trajectory-{traj_index}"
                trajectory_groups_dict[(task_id, trajectory_name)].append(trajectory)
        elif grouping_level == "step":
            for traj_index, trajectory in enumerate(episode.trajectories):
                trajectory_name = trajectory.name or f"trajectory-{traj_index}"
                for step_idx, step in enumerate(trajectory.steps):
                    single_step_traj = Trajectory(name=trajectory_name, steps=[step], reward=trajectory.reward)
                    trajectory_groups_dict[(task_id, trajectory_name, step_idx)].append(single_step_traj)
        else:
            trajectory_groups_dict[task_id].extend(episode.trajectories)

    groups_before_filter = [TrajectoryGroup(trajectories=trajs, group_id=str(key)) for key, trajs in trajectory_groups_dict.items()]
    groups = trajectory_filter.filter_groups(groups_before_filter)

    samples: list[TrlTrainingSample] = []
    all_advantages: list[float] = []
    for group in groups:
        rewards = [traj.reward for traj in group.trajectories]
        advantages = advantage_computer.compute(rewards)
        all_advantages.extend(advantages)
        for traj, advantage in zip(group.trajectories, advantages, strict=False):
            samples.extend(TrlSampleBuilder.build_from_trajectory(traj, advantage))

    metrics = {
        "grouping/num_groups_before_filter": len(groups_before_filter),
        "grouping/num_groups": len(groups),
        "grouping/num_samples": len(samples),
        "advantage/mean": float(np.mean(all_advantages)) if all_advantages else 0.0,
        "advantage/std": float(np.std(all_advantages)) if all_advantages else 0.0,
    }
    return samples, metrics
