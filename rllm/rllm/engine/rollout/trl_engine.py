"""RolloutEngine implementation using a local HuggingFace model for TRL-style training."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


class TrlEngine(RolloutEngine):
    """Local HF model rollout for agent training without Ray runtime."""

    def __init__(
        self,
        model,
        tokenizer,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        sampling_params: dict | None = None,
        device: str | torch.device | None = None,
        disable_thinking: bool = False,
        chat_parser=None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.sampling_params = sampling_params or {}
        self.device = device or (model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chat_parser = chat_parser or ChatTemplateParser.get_parser(tokenizer, disable_thinking=disable_thinking)
        self._gen_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def set_model(self, model) -> None:
        """Point rollout at the current policy weights."""
        self.model = model
        if hasattr(model, "device"):
            self.device = model.device

    def _resolve_sampling(self, kwargs: dict, validate: bool) -> dict:
        params = self.sampling_params.copy()
        params.update({k: v for k, v in kwargs.items() if k not in ("application_id", "validate", "enforce_max_prompt_length", "meta_info")})
        max_tokens = params.pop("max_tokens", params.pop("max_new_tokens", self.max_response_length))
        temperature = params.pop("temperature", 1.0)
        top_p = params.pop("top_p", 1.0)
        if validate:
            temperature = 0.0
            top_p = 1.0
        do_sample = temperature > 0
        return {
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature) if do_sample else 1.0,
            "top_p": float(top_p),
            "do_sample": do_sample,
        }

    def _generate_sync(self, messages: list[dict], kwargs: dict) -> ModelOutput:
        validate = kwargs.get("validate", False)
        enforce_max_prompt_length = kwargs.get("enforce_max_prompt_length", True)
        gen_kwargs = self._resolve_sampling(kwargs, validate)

        prompt_text = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_length = len(prompt_ids)

        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        input_ids = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_kwargs,
                )
                full_ids = generated[0]
                completion_ids = full_ids[prompt_length:].tolist()

                if completion_ids:
                    logits = self.model(full_ids.unsqueeze(0)).logits[0]
                    log_probs = F.log_softmax(logits[:-1], dim=-1)
                    token_logprobs = []
                    for idx in range(prompt_length - 1, len(full_ids) - 1):
                        token_logprobs.append(log_probs[idx, full_ids[idx + 1]].item())
                else:
                    token_logprobs = []
        finally:
            if was_training:
                self.model.train()

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        finish_reason = "length" if len(completion_ids) >= gen_kwargs["max_new_tokens"] else "stop"

        return ModelOutput(
            text=completion_text,
            content=completion_text,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=token_logprobs,
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        async with self._gen_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self._generate_sync, messages, kwargs)

    async def wake_up(self):
        return None

    async def sleep(self):
        return None
