"""Rollout engines for the Swift backend (no Ray).

Modes:
- ``server``: Official ms-swift ``swift rollout`` service with weight sync.
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
        server_timeout_s: float = 900,
        group_port: int = 51216,
        sync_weights: bool = True,
        weight_sync_mode: str = "auto",
        is_main_process: bool = True,
        sync_device: int | str | torch.device | None = None,
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
        self.server_timeout_s = float(server_timeout_s)
        self.group_port = int(group_port)
        self.sync_weights = bool(sync_weights)
        self.weight_sync_mode = weight_sync_mode
        self.is_main_process = bool(is_main_process)
        self.sync_device = sync_device
        self._last_synced_version: int | None = None
        self._communicator_initialized = False

        self._queue: asyncio.Queue[_PendingRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._vllm_llm = None
        self._swift_client = None

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
        try:
            from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
        except ImportError as exc:
            raise ImportError(
                "mode=server requires ms-swift==3.12.3. "
                "Install the rllm swift extra and launch the server with `swift rollout`."
            ) from exc

        # swift rollout uses root endpoints such as /health/ and /infer/, not
        # the OpenAI-compatible /v1 prefix.
        self.base_url = self.base_url.removesuffix("/v1").rstrip("/")
        self._swift_client = VLLMClient(
            base_urls=[self.base_url],
            group_ports=self.group_port,
            connection_timeout=self.server_timeout_s,
        )
        logger.info("SwiftEngine official rollout server mode → %s", self.base_url)

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

    def sync_weights_from_hf(self, model, policy_version: int | None = None) -> bool:
        """Synchronize the current policy through the official Swift protocol.

        Returns ``True`` only when this process transferred a new policy
        version. Distributed barriers are owned by ``SwiftAgentTrainer``.
        """
        self.set_model(model)
        if self.mode == "colocate":
            logger.warning(
                "Colocate vLLM weight sync after train is best-effort. "
                "Prefer mode=server with `swift rollout`, "
                "or mode=transformers for shared-weight correctness."
            )
            return False
        if self.mode != "server" or not self.sync_weights or not self.is_main_process:
            return False
        if policy_version is not None and policy_version == self._last_synced_version:
            return False

        self._ensure_communicator()
        sync_mode = self.weight_sync_mode
        if sync_mode == "auto":
            sync_mode = "adapter" if hasattr(model, "peft_config") else "full"
        if sync_mode == "adapter":
            self._sync_adapter_weights(model)
        elif sync_mode == "full":
            self._sync_full_weights(model)
        else:
            raise ValueError(f"Unsupported rollout.weight_sync_mode={sync_mode!r}; use auto|full|adapter")

        self._swift_client.reset_prefix_cache()
        self._last_synced_version = policy_version
        logger.info("Synchronized rollout policy version=%s mode=%s", policy_version, sync_mode)
        return True

    def _ensure_communicator(self) -> None:
        if self._communicator_initialized:
            return
        engine_info = self._swift_client.get_engine_type()
        if self.sync_device is None:
            sync_device = torch.cuda.current_device()
        elif isinstance(self.sync_device, torch.device):
            sync_device = self.sync_device.index
        else:
            sync_device = self.sync_device
        self._swift_client.init_communicator(device=sync_device)
        self._communicator_initialized = True
        logger.info(
            "Initialized Swift rollout communicator: group_port=%s engine=%s",
            self.group_port,
            engine_info.get("engine_type"),
        )

    @staticmethod
    def _clean_full_weight_name(name: str) -> str:
        return name.removeprefix("base_model.model.").replace(".base_layer", "")

    def _sync_full_weights(self, model) -> None:
        from swift.trainers.rlhf_trainer.utils import FlattenedTensorBucket, _create_parameter_buckets

        is_peft = hasattr(model, "peft_config")
        if is_peft:
            model.merge_adapter()
        try:
            named_weights = []
            for name, parameter in model.named_parameters():
                clean_name = self._clean_full_weight_name(name)
                if is_peft and ("lora_" in clean_name or "modules_to_save.original_module" in clean_name):
                    continue
                clean_name = clean_name.replace("modules_to_save.default.", "")
                named_weights.append((clean_name, parameter.detach()))

            bucket_size_mb = int(os.getenv("SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE", "512"))
            for named_bucket in _create_parameter_buckets(named_weights, bucket_size_mb=bucket_size_mb):
                bucket = FlattenedTensorBucket(named_tensors=named_bucket)
                self._swift_client.update_flattened_params(bucket.get_metadata(), bucket.get_flattened_tensor())
        finally:
            if is_peft:
                model.unmerge_adapter()

    def _sync_adapter_weights(self, model) -> None:
        if not hasattr(model, "peft_config"):
            raise ValueError("adapter weight sync requires model.use_lora=true")
        engine_info = self._swift_client.get_engine_type()
        if not engine_info.get("enable_lora", False):
            raise RuntimeError(
                "The rollout server was not started with --vllm_enable_lora true, "
                "but adapter-only synchronization was requested."
            )

        from peft import get_peft_model_state_dict
        from swift.trainers.rlhf_trainer.utils import FlattenedTensorBucket

        adapter_name = getattr(model, "active_adapter", "default")
        if isinstance(adapter_name, list):
            adapter_name = adapter_name[0]
        peft_config = model.peft_config[adapter_name]
        adapter_state = get_peft_model_state_dict(model, adapter_name=adapter_name)
        named_weights = [(name, weight.detach()) for name, weight in adapter_state.items()]
        bucket = FlattenedTensorBucket(named_tensors=named_weights)
        self._swift_client.update_adapter_flattened_param(
            peft_config,
            bucket.get_metadata(),
            bucket.get_flattened_tensor(),
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
                if self.mode == "server":
                    outputs = await self._server_generate_batch(batch)
                elif self.mode == "colocate":
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

    async def _server_generate_batch(self, batch: list[_PendingRequest]) -> list[ModelOutput]:
        loop = asyncio.get_event_loop()

        def _run() -> list[ModelOutput]:
            # A queue batch uses one sampling configuration. Agent batches in
            # this backend are homogeneous, as are validation batches.
            gen = self._resolve_sampling(batch[0].kwargs, batch[0].kwargs.get("validate", False))
            infer_requests = [{"messages": req.messages} for req in batch]
            request_config = {
                "max_tokens": gen["max_tokens"],
                "temperature": gen["temperature"] if gen["do_sample"] else 0.0,
                "top_p": gen["top_p"],
                "logprobs": True,
                "top_logprobs": 1,
                "return_details": True,
            }

            retries = self.api_retries
            last_err: Exception | None = None
            while retries > 0:
                try:
                    results = self._swift_client.infer(
                        infer_requests,
                        request_config,
                        use_tqdm=False,
                    )
                    if len(results) != len(batch):
                        raise RuntimeError(f"swift rollout returned {len(results)} results for {len(batch)} requests")
                    return [self._convert_server_output(req, result) for req, result in zip(batch, results, strict=True)]
                except Exception as exc:
                    last_err = exc
                    retries -= 1
                    if retries:
                        import time

                        time.sleep(1)
            raise RuntimeError(f"SwiftEngine server generate failed: {last_err}") from last_err

        return await loop.run_in_executor(self._executor, _run)

    def _convert_server_output(self, req: _PendingRequest, result) -> ModelOutput:
        rollout_response = getattr(result, "response", result)
        choices = getattr(rollout_response, "choices", None)
        if not choices:
            raise RuntimeError(f"swift rollout returned no choices: {result!r}")
        choice = choices[0]
        message = getattr(choice, "message", None)
        text = getattr(message, "content", None) if message is not None else getattr(choice, "text", "")
        text = text or ""

        response_token_ids = getattr(result, "response_token_ids", None) or []
        completion_ids = list(response_token_ids[-1]) if response_token_ids else list(getattr(choice, "token_ids", None) or [])
        if not completion_ids:
            completion_ids = self.tokenizer.encode(text, add_special_tokens=False)

        rollout_logprobs = getattr(result, "rollout_logprobs", None) or []
        logprobs = list(rollout_logprobs[-1]) if rollout_logprobs else self._choice_logprobs(choice)
        logprobs = (logprobs + [0.0] * len(completion_ids))[: len(completion_ids)]
        finish = getattr(choice, "finish_reason", None) or "stop"
        return ModelOutput(
            text=text,
            content=text,
            prompt_ids=req.prompt_ids,
            completion_ids=completion_ids,
            logprobs=logprobs,
            prompt_length=len(req.prompt_ids),
            completion_length=len(completion_ids),
            finish_reason=finish,
        )

    @staticmethod
    def _choice_logprobs(choice) -> list[float]:
        payload = getattr(choice, "logprobs", None)
        if not payload:
            return []
        content = payload.get("content", []) if isinstance(payload, dict) else getattr(payload, "content", [])
        values = []
        for item in content:
            value = item.get("logprob") if isinstance(item, dict) else getattr(item, "logprob", None)
            values.append(float(value) if value is not None else 0.0)
        return values

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
        if self._communicator_initialized and self._swift_client is not None:
            await asyncio.get_event_loop().run_in_executor(self._executor, self._swift_client.close_communicator)
            self._communicator_initialized = False
        self._executor.shutdown(wait=False)
