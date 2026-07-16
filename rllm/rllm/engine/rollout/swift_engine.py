"""Rollout engines for the Swift backend (ms-swift style, no Ray).

Modes (inspired by ms-swift GRPO):
- ``server``: OpenAI-compatible vLLM HTTP. Concurrent async requests (true parallel).
- ``colocate``: In-process vLLM LLM. Requests are gathered into batches.
- ``transformers``: Local HF generate with request batching + output_scores (no 2nd forward).
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason

logger = logging.getLogger(__name__)


@dataclass
class _PendingRequest:
    messages: list[dict]
    kwargs: dict
    future: asyncio.Future
    prompt_text: str = ""
    prompt_ids: list[int] = field(default_factory=list)


class SwiftEngine(RolloutEngine):
    """ms-swift-style rollout: prefer vLLM batched/concurrent sampling."""

    def __init__(
        self,
        tokenizer,
        model=None,
        mode: str = "server",
        model_name: str | None = None,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
        sampling_params: dict | None = None,
        device: str | torch.device | None = None,
        disable_thinking: bool = False,
        chat_parser=None,
        batch_size: int = 8,
        batch_timeout_s: float = 0.05,
        vllm_gpu_memory_utilization: float = 0.85,
        vllm_tensor_parallel_size: int = 1,
        api_retries: int = 3,
        **kwargs,
    ):
        self.mode = mode
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or getattr(model, "name_or_path", None) or "default"
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.sampling_params = sampling_params or {}
        self.device = device or (getattr(model, "device", None) if model is not None else "cuda")
        self.chat_parser = chat_parser or ChatTemplateParser.get_parser(tokenizer, disable_thinking=disable_thinking)
        self.batch_size = max(1, int(batch_size))
        self.batch_timeout_s = float(batch_timeout_s)
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.api_retries = api_retries

        self._queue: asyncio.Queue[_PendingRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._vllm_llm = None
        self._openai_client = None

        if self.mode == "server":
            self._init_server_client()
        elif self.mode == "colocate":
            self._init_colocate_vllm()
        elif self.mode == "transformers":
            if self.model is None:
                raise ValueError("SwiftEngine mode=transformers requires a model")
        else:
            raise ValueError(f"Unsupported SwiftEngine mode: {mode}. Use server|colocate|transformers")

    def _init_server_client(self) -> None:
        import openai

        self._openai_client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        logger.info("SwiftEngine server mode → %s", self.base_url)

    def _init_colocate_vllm(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as exc:
            raise ImportError("mode=colocate requires vllm. pip install vllm or use mode=server/transformers") from exc

        model_path = self.model_name
        if self.model is not None and hasattr(self.model, "name_or_path"):
            model_path = self.model.name_or_path
        self._vllm_llm = LLM(
            model=model_path,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            max_model_len=self.max_prompt_length + self.max_response_length,
            trust_remote_code=True,
        )
        logger.info("SwiftEngine colocate vLLM loaded: %s", model_path)

    def set_model(self, model) -> None:
        """Update HF model pointer (transformers mode / weight source)."""
        self.model = model
        if self.mode == "transformers" and model is not None:
            self.device = getattr(model, "device", self.device)

    def sync_weights_from_hf(self, model) -> None:
        """Best-effort weight sync after a train step."""
        self.set_model(model)
        if self.mode == "colocate":
            logger.warning(
                "Colocate vLLM weight sync after train is best-effort. "
                "Prefer mode=server (train GPUs vs infer GPUs) like ms-swift, "
                "or mode=transformers for shared-weight correctness."
            )

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
            "max_tokens": int(max_tokens),
            "temperature": float(temperature) if do_sample else 0.0,
            "top_p": float(top_p),
            "do_sample": do_sample,
        }

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        if self.mode == "server":
            return await self._server_generate(messages, kwargs)

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        prompt_text = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        enforce = kwargs.get("enforce_max_prompt_length", True)
        if enforce and len(prompt_ids) > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        req = _PendingRequest(messages=messages, kwargs=kwargs, future=future, prompt_text=prompt_text, prompt_ids=prompt_ids)
        self._ensure_worker()
        await self._queue.put(req)
        return await future

    async def _batch_worker(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                return
            batch = [first]
            deadline = asyncio.get_event_loop().time() + self.batch_timeout_s
            while len(batch) < self.batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    await self._queue.put(None)
                    break
                batch.append(item)

            try:
                if self.mode == "colocate":
                    outputs = await self._colocate_generate_batch(batch)
                else:
                    outputs = await self._transformers_generate_batch(batch)
                for req, out in zip(batch, outputs, strict=False):
                    if not req.future.done():
                        req.future.set_result(out)
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    async def _server_generate(self, messages: list[dict], kwargs: dict) -> ModelOutput:
        validate = kwargs.get("validate", False)
        enforce = kwargs.get("enforce_max_prompt_length", True)
        gen = self._resolve_sampling(kwargs, validate)
        prompt_text = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if enforce and len(prompt_ids) > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        retries = self.api_retries
        last_err: Exception | None = None
        while retries > 0:
            try:
                response = await self._openai_client.completions.create(
                    model=self.model_name,
                    prompt=prompt_text,
                    max_tokens=gen["max_tokens"],
                    temperature=gen["temperature"] if gen["do_sample"] else 0.0,
                    top_p=gen["top_p"],
                    logprobs=1,
                    timeout=3600,
                )
                choice = response.choices[0]
                text = choice.text or ""
                completion_ids = self.tokenizer.encode(text, add_special_tokens=False)
                logprobs: list[float] = []
                if choice.logprobs is not None and choice.logprobs.token_logprobs is not None:
                    logprobs = [float(x) if x is not None else 0.0 for x in choice.logprobs.token_logprobs]
                if len(logprobs) != len(completion_ids):
                    if len(logprobs) > len(completion_ids):
                        logprobs = logprobs[: len(completion_ids)]
                    else:
                        logprobs = logprobs + [0.0] * (len(completion_ids) - len(logprobs))
                finish = choice.finish_reason or "stop"
                return ModelOutput(
                    text=text,
                    content=text,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    prompt_length=len(prompt_ids),
                    completion_length=len(completion_ids),
                    finish_reason=finish,
                )
            except Exception as exc:
                last_err = exc
                retries -= 1
                await asyncio.sleep(1)
        raise RuntimeError(f"SwiftEngine server generate failed: {last_err}") from last_err

    async def _colocate_generate_batch(self, batch: list[_PendingRequest]) -> list[ModelOutput]:
        from vllm import SamplingParams

        loop = asyncio.get_event_loop()

        def _run() -> list[ModelOutput]:
            prompts = [req.prompt_text for req in batch]
            gen = self._resolve_sampling(batch[0].kwargs, batch[0].kwargs.get("validate", False))
            sp = SamplingParams(
                max_tokens=gen["max_tokens"],
                temperature=gen["temperature"] if gen["do_sample"] else 0.0,
                top_p=gen["top_p"],
                logprobs=1,
            )
            results = self._vllm_llm.generate(prompts, sp)
            outputs: list[ModelOutput] = []
            for req, result in zip(batch, results, strict=False):
                out = result.outputs[0]
                completion_ids = list(out.token_ids)
                text = out.text
                logprobs: list[float] = []
                if out.logprobs:
                    for tok_lp in out.logprobs:
                        if tok_lp is None:
                            logprobs.append(0.0)
                            continue
                        if completion_ids and len(logprobs) < len(completion_ids):
                            tid = completion_ids[len(logprobs)]
                            if tid in tok_lp:
                                logprobs.append(float(tok_lp[tid].logprob))
                            else:
                                logprobs.append(float(next(iter(tok_lp.values())).logprob))
                        else:
                            logprobs.append(0.0)
                if len(logprobs) != len(completion_ids):
                    logprobs = (logprobs + [0.0] * len(completion_ids))[: len(completion_ids)]
                outputs.append(
                    ModelOutput(
                        text=text,
                        content=text,
                        prompt_ids=req.prompt_ids,
                        completion_ids=completion_ids,
                        logprobs=logprobs,
                        prompt_length=len(req.prompt_ids),
                        completion_length=len(completion_ids),
                        finish_reason=out.finish_reason or "stop",
                    )
                )
            return outputs

        return await loop.run_in_executor(self._executor, _run)

    async def _transformers_generate_batch(self, batch: list[_PendingRequest]) -> list[ModelOutput]:
        loop = asyncio.get_event_loop()

        def _run() -> list[ModelOutput]:
            gen = self._resolve_sampling(batch[0].kwargs, batch[0].kwargs.get("validate", False))
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            max_prompt = max(len(r.prompt_ids) for r in batch)
            input_rows = []
            attn_rows = []
            for req in batch:
                pad_len = max_prompt - len(req.prompt_ids)
                input_rows.append([pad_id] * pad_len + req.prompt_ids)
                attn_rows.append([0] * pad_len + [1] * len(req.prompt_ids))
            input_ids = torch.tensor(input_rows, device=self.device, dtype=torch.long)
            attention_mask = torch.tensor(attn_rows, device=self.device, dtype=torch.long)

            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen["max_tokens"],
                        do_sample=gen["do_sample"],
                        temperature=gen["temperature"] if gen["do_sample"] else None,
                        top_p=gen["top_p"] if gen["do_sample"] else None,
                        pad_token_id=pad_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    sequences = generated.sequences
                    scores = generated.scores
            finally:
                if was_training:
                    self.model.train()

            outputs: list[ModelOutput] = []
            for i, req in enumerate(batch):
                seq = sequences[i]
                prompt_len_padded = max_prompt
                completion_ids = seq[prompt_len_padded:].tolist()
                if pad_id is not None:
                    while completion_ids and completion_ids[-1] == pad_id:
                        completion_ids.pop()
                logprobs: list[float] = []
                for t, score_t in enumerate(scores):
                    if t >= len(completion_ids):
                        break
                    log_prob = torch.log_softmax(score_t[i], dim=-1)
                    logprobs.append(float(log_prob[completion_ids[t]].item()))
                text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                finish = "length" if len(completion_ids) >= gen["max_tokens"] else "stop"
                outputs.append(
                    ModelOutput(
                        text=text,
                        content=text,
                        prompt_ids=req.prompt_ids,
                        completion_ids=completion_ids,
                        logprobs=logprobs,
                        prompt_length=len(req.prompt_ids),
                        completion_length=len(completion_ids),
                        finish_reason=finish,
                    )
                )
            return outputs

        return await loop.run_in_executor(self._executor, _run)

    async def wake_up(self):
        return None

    async def sleep(self):
        return None

    async def close(self):
        if self._worker_task is not None:
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._worker_task, timeout=5)
            except Exception:
                self._worker_task.cancel()
            self._worker_task = None
        self._executor.shutdown(wait=False)
