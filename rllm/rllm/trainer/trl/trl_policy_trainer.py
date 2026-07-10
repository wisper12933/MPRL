"""Policy training for TRL-based agent RL (no Ray)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from rllm.trainer.trl.trl_data_processor import (
    TrlAdvantageComputer,
    TrlTrainingSample,
    TrlTrajectoryFilter,
    process_episodes,
)

if TYPE_CHECKING:
    from peft import PeftModel

logger = logging.getLogger(__name__)


class TrlPolicyTrainer:
    """Loads a local HF policy and applies GRPO-style policy-gradient updates."""

    def __init__(self, config):
        self.config = config
        self.advantage_computer = TrlAdvantageComputer(config.algorithm)
        self.trajectory_filter = TrlTrajectoryFilter(config.algorithm)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, resume_from_checkpoint: bool = True) -> int:
        model_name = self.config.model.name
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            attn_implementation=self.config.model.get("attn_implementation", None),
        )

        if self.config.model.get("use_lora", False):
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.model.get("lora_rank", 32),
                lora_alpha=self.config.model.get("lora_alpha", 64),
                target_modules=self.config.model.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                lora_dropout=self.config.model.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        if self.config.model.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)
        lr = self.config.training.learning_rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps,
        )

        start_batch = 0
        if resume_from_checkpoint:
            start_batch = self._maybe_resume()
        return start_batch

    def _checkpoint_dir(self) -> str:
        return self.config.trainer.default_local_dir

    def _maybe_resume(self) -> int:
        checkpoint_dir = self._checkpoint_dir()
        marker_path = os.path.join(checkpoint_dir, "latest_batch.txt")
        if not os.path.exists(marker_path):
            return 0
        with open(marker_path) as f:
            batch = int(f.read().strip())
        model_path = os.path.join(checkpoint_dir, f"checkpoint-{batch}")
        if not os.path.isdir(model_path):
            return 0
        logger.info("Resuming from checkpoint %s", model_path)
        if self.config.model.get("use_lora", False):
            from peft import PeftModel

            base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            self.model = PeftModel.from_pretrained(base, model_path, is_trainable=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=self.model.dtype)
        self.model.to(self.device)
        optim_path = os.path.join(model_path, "optimizer.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(torch.load(optim_path, map_location=self.device))
        return batch + 1

    def save_checkpoint(self, batch_idx: int) -> None:
        checkpoint_dir = self._checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"checkpoint-{batch_idx}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        with open(os.path.join(checkpoint_dir, "latest_batch.txt"), "w") as f:
            f.write(str(batch_idx))

    def _collate_samples(self, samples: list[TrlTrainingSample]) -> dict[str, torch.Tensor]:
        max_len = max(sample.input_ids.numel() for sample in samples)
        pad_id = self.tokenizer.pad_token_id

        input_ids, target_ids, old_logprobs, advantages, mask = [], [], [], [], []
        for sample in samples:
            pad_len = max_len - sample.input_ids.numel()
            input_ids.append(F.pad(sample.input_ids, (0, pad_len), value=pad_id))
            target_ids.append(F.pad(sample.target_ids, (0, pad_len), value=pad_id))
            old_logprobs.append(F.pad(sample.old_logprobs, (0, pad_len), value=0.0))
            advantages.append(F.pad(sample.advantages, (0, pad_len), value=0.0))
            mask.append(F.pad(sample.mask, (0, pad_len), value=0.0))

        return {
            "input_ids": torch.stack(input_ids).to(self.device),
            "target_ids": torch.stack(target_ids).to(self.device),
            "old_logprobs": torch.stack(old_logprobs).to(self.device),
            "advantages": torch.stack(advantages).to(self.device),
            "mask": torch.stack(mask).to(self.device),
        }

    def _importance_sampling_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=(batch["input_ids"] != self.tokenizer.pad_token_id).long())
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, batch["target_ids"].unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(token_log_probs - batch["old_logprobs"])
        clip_eps = self.config.training.get("clip_ratio", 0.2)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -torch.min(ratio * batch["advantages"], clipped_ratio * batch["advantages"])
        masked_loss = policy_loss * batch["mask"]
        denom = batch["mask"].sum().clamp_min(1.0)
        loss = masked_loss.sum() / denom

        metrics = {
            "loss/policy": loss.detach().item(),
            "policy/ratio_mean": ((ratio * batch["mask"]).sum() / denom).detach().item(),
        }
        return loss, metrics

    def step(self, episodes: list, learning_rate: float | None = None) -> dict:
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

        samples, grouping_metrics = process_episodes(
            episodes,
            self.advantage_computer,
            self.trajectory_filter,
            self.config.algorithm,
        )
        if not samples:
            return grouping_metrics

        self.model.train()
        total_metrics = dict(grouping_metrics)
        num_minibatches = max(1, self.config.training.get("num_minibatches", 1))
        chunk_size = max(1, len(samples) // num_minibatches)

        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            batch = self._collate_samples(chunk)
            self.optimizer.zero_grad()
            loss, step_metrics = self._importance_sampling_loss(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.get("max_grad_norm", 1.0))
            self.optimizer.step()
            total_metrics.update(step_metrics)
            total_metrics["loss/grad_norm"] = float(grad_norm)

        return total_metrics

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
