"""TRL-based trainer for rLLM agents (no Ray).

Uses AsyncAgentExecutionEngine for rollout and a local HF policy for updates.
Multi-GPU is supported via HuggingFace Accelerate (launch with `accelerate launch`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from rllm.trainer.trl.trl_policy_trainer import TrlPolicyTrainer

logger = logging.getLogger(__name__)


class TrlAgentTrainer:
    """Agent RL trainer backed by local HF rollout + policy updates."""

    def __init__(
        self,
        config,
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        train_dataset=None,
        val_dataset=None,
    ):
        self.config = config
        self.env_class = env_class
        self.agent_class = agent_class
        self.agent_args = agent_args or {}
        self.env_args = env_args or {}

        self.trainer = TrlPolicyTrainer(config=config)
        self.start_batch = self.trainer.initialize(resume_from_checkpoint=True)

        per_device_batch = self.config.data.train_batch_size
        per_device_val_batch = self.config.data.val_batch_size

        train_sampler = None
        val_sampler = None
        if self.trainer.num_processes > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.trainer.num_processes,
                rank=self.trainer.accelerator.process_index,
                shuffle=True,
            )
            if val_dataset is not None:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.trainer.num_processes,
                    rank=self.trainer.accelerator.process_index,
                    shuffle=False,
                )

        self.train_sampler = train_sampler
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=per_device_batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=lambda x: x,
        )
        self.val_dataloader = None
        if val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=per_device_val_batch,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=lambda x: x,
            )

        self.tokenizer = self.trainer.get_tokenizer()
        sampling_params = OmegaConf.to_container(self.config.sampling, resolve=True)
        group_size = self.config.training.group_size
        default_parallel = per_device_batch * group_size

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            config=self.config,
            engine_name="trl",
            tokenizer=self.tokenizer,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=agent_class,
            agent_args=agent_args,
            env_class=env_class,
            env_args=env_args,
            n_parallel_agents=self.config.agent.get("n_parallel_agents") or default_parallel,
            rollout_engine_args={
                "model": self.trainer.get_model(),
                "tokenizer": self.tokenizer,
                "max_prompt_length": self.config.data.max_prompt_length,
                "max_response_length": self.config.data.max_response_length,
                "sampling_params": sampling_params,
                "disable_thinking": self.config.get("disable_thinking", False),
                "device": self.trainer.device,
            },
        )
        self.num_train_batches = len(self.train_dataloader) if self.train_dataloader else None

    def fit_agent(self):
        asyncio.run(self._fit_agent_async())

    async def _fit_agent_async(self):
        from rllm.utils.tracking import Tracking

        if self.trainer.is_main_process:
            os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        self.trainer.wait_for_everyone()

        tracking_logger = None
        if self.trainer.is_main_process:
            logger_backend = self.config.trainer.logger
            if isinstance(logger_backend, str):
                logger_backend = [logger_backend]
            tracking_logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=logger_backend,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        batch_idx = self.start_batch
        learning_rate = self.config.training.learning_rate

        if self.config.trainer.get("val_before_train", False) and self.val_dataloader:
            self._sync_rollout_model()
            val_metrics = await self.validate_agent(self.val_dataloader)
            if val_metrics and tracking_logger is not None:
                tracking_logger.log(data=val_metrics, step=batch_idx)

        for epoch in range(self.config.trainer.total_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for batch_data in self.train_dataloader:
                if batch_idx < self.start_batch:
                    batch_idx += 1
                    continue

                t_start = time.time()
                time_metrics = {}
                group_size = self.config.training.group_size

                batch_data = self.build_interleave_batch(batch_data, group_size)
                self.init_envs_and_agents(batch_data)

                self._sync_rollout_model()
                t_sample_start = time.time()
                episodes = []
                async for episode_batch in self.generate_agent_episodes(
                    group_size=group_size,
                    minibatch_size=max(1, len(batch_data) // group_size),
                ):
                    episodes.extend(episode_batch)
                time_metrics["time/sample"] = time.time() - t_sample_start

                t_train_start = time.time()
                step_metrics = self.trainer.step(episodes, learning_rate=learning_rate)
                self._sync_rollout_model()
                time_metrics["time/train"] = time.time() - t_train_start
                time_metrics["time/total"] = time.time() - t_start

                rewards = [traj.reward for ep in episodes for traj in ep.trajectories]
                reward_mean = float(np.mean(rewards)) if rewards else 0.0
                metrics = {
                    **step_metrics,
                    **time_metrics,
                    "reward/mean": self.trainer.reduce_mean(reward_mean),
                    "reward/max": float(np.max(rewards)) if rewards else 0.0,
                    "reward/min": float(np.min(rewards)) if rewards else 0.0,
                    "batch/num_episodes": len(episodes),
                    "epoch": epoch,
                    "distributed/num_processes": self.trainer.num_processes,
                    "distributed/rank": self.trainer.accelerator.process_index,
                }
                if tracking_logger is not None:
                    tracking_logger.log(data=metrics, step=batch_idx)
                if self.trainer.is_main_process:
                    logger.info("batch=%s metrics=%s", batch_idx, metrics)

                if (
                    self.val_dataloader
                    and self.config.trainer.test_freq > 0
                    and batch_idx % self.config.trainer.test_freq == 0
                    and batch_idx > 0
                ):
                    val_metrics = await self.validate_agent(self.val_dataloader)
                    if val_metrics and tracking_logger is not None:
                        tracking_logger.log(data=val_metrics, step=batch_idx)

                if batch_idx % self.config.trainer.save_freq == 0:
                    self.trainer.save_checkpoint(batch_idx)
                self.trainer.wait_for_everyone()

                batch_idx += 1

        if batch_idx % self.config.trainer.save_freq != 0:
            self.trainer.save_checkpoint(batch_idx)
        self.trainer.wait_for_everyone()
        if tracking_logger is not None:
            del tracking_logger

    def _sync_rollout_model(self):
        self.agent_execution_engine.rollout_engine.set_model(self.trainer.get_model())

    def init_envs_and_agents(self, batch):
        env_args = batch
        full_agent_args = dict(self.config.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            item = env_args[i]
            if isinstance(item, str):
                item = json.loads(item)
            return i, self.env_class.from_dict({**item, **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            for idx, env in executor.map(_create_env, range(len(env_args))):
                envs[idx] = env

        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            for idx, agent in executor.map(_create_agent, range(len(envs))):
                agents[idx] = agent

        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    async def validate_agent(self, dataloader):
        episodes_ls = []
        for batch in dataloader:
            batch = self.build_interleave_batch(batch, 1)
            self.init_envs_and_agents(batch)
            async for episode_batch in self.generate_agent_episodes(group_size=1, minibatch_size=1):
                episodes_ls.extend(episode_batch)

        trajectories = [traj for episode in episodes_ls for traj in episode.trajectories]
        if not trajectories:
            return {}
        rewards = [traj.reward for traj in trajectories]
        turns = [len(traj.steps) for traj in trajectories]
        return {
            "val/reward_mean": self.trainer.reduce_mean(float(np.mean(rewards))),
            "val/reward_std": float(np.std(rewards)),
            "val/reward_min": float(np.min(rewards)),
            "val/reward_max": float(np.max(rewards)),
            "val/turns_mean": float(np.mean(turns)) if turns else 0.0,
        }

    async def generate_agent_episodes(self, timing_raw=None, meta_info=None, group_size: int = 1, minibatch_size: int = 1):
        if timing_raw is None:
            timing_raw = {}

        group_dict = defaultdict(list)
        episode_queue = asyncio.Queue()
        producer_exception = None

        async def produce_episodes():
            nonlocal producer_exception
            try:
                async for traj in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode="Step", meta_info=meta_info):
                    group_index = traj["idx"] // group_size
                    group_dict[group_index].append(traj)
                    if len(group_dict[group_index]) == group_size:
                        episode = self.convert_to_episode(group_dict[group_index])
                        await episode_queue.put(episode)
            except Exception as exc:
                producer_exception = exc
                logger.exception("Episode generation failed")
            finally:
                await episode_queue.put(None)

        producer_task = asyncio.create_task(produce_episodes())
        minibatch = []
        try:
            while True:
                if producer_exception is not None:
                    raise RuntimeError("Episode generation failed") from producer_exception
                episode = await episode_queue.get()
                if episode is None:
                    break
                minibatch.append(episode)
                if len(minibatch) == minibatch_size:
                    yield minibatch
                    minibatch = []
            if minibatch:
                yield minibatch
        finally:
            await producer_task
            if producer_exception is not None:
                raise RuntimeError("Episode generation failed") from producer_exception

    def convert_to_episode(self, group: list) -> Episode:
        trajectories = []
        episode_id = None
        episode_task = None
        for traj_idx, traj in enumerate(group):
            if episode_id is None:
                episode_id = traj.get("uid") or f"group-{traj['idx']}"
                episode_task = traj.get("task")
            steps = []
            for step in traj["steps"]:
                response_ids = step.get("completion_ids") or step.get("response_ids") or []
                steps.append(
                    Step(
                        prompt_ids=step["prompt_ids"],
                        response_ids=response_ids,
                        logprobs=step.get("logprobs") or [0.0] * len(response_ids),
                    )
                )
            trajectories.append(
                Trajectory(
                    name=f"trajectory-{traj_idx}",
                    task=traj.get("task"),
                    steps=steps,
                    reward=traj["trajectory_reward"],
                )
            )
        return Episode(id=episode_id or str(uuid.uuid4()), task=episode_task, trajectories=trajectories)

    def build_interleave_batch(self, batch: list, group_size: int):
        interleave_batch = []
        batch_with_uid = []
        for batch_item in batch:
            normalized_item = json.loads(batch_item) if isinstance(batch_item, str) else dict(batch_item)
            normalized_item["uid"] = str(uuid.uuid4())
            batch_with_uid.append(normalized_item)
        for batch_item in batch_with_uid:
            interleave_batch.extend([dict(batch_item) for _ in range(group_size)])
        return interleave_batch
