import asyncio
from types import SimpleNamespace

import pytest

from rllm.engine.rollout.swift_engine import SwiftEngine, _PendingRequest


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


def _sync_engine(*, main=True, mode="auto"):
    engine = SwiftEngine.__new__(SwiftEngine)
    engine.mode = "server"
    engine.model = None
    engine.sync_weights = True
    engine.is_main_process = main
    engine.weight_sync_mode = mode
    engine._last_synced_version = None
    engine._swift_client = SimpleNamespace(reset_prefix_cache=lambda: None)
    engine._ensure_communicator = lambda: None
    return engine


def test_sync_selects_full_and_deduplicates_policy_version():
    engine = _sync_engine()
    calls = []
    engine._sync_full_weights = lambda model: calls.append(("full", model))
    engine._sync_adapter_weights = lambda model: calls.append(("adapter", model))
    model = object()

    assert engine.sync_weights_from_hf(model, policy_version=3)
    assert not engine.sync_weights_from_hf(model, policy_version=3)
    assert calls == [("full", model)]


def test_sync_auto_selects_adapter_and_only_runs_on_main_process():
    adapter_model = SimpleNamespace(peft_config={"default": object()})
    engine = _sync_engine()
    calls = []
    engine._sync_full_weights = lambda model: calls.append("full")
    engine._sync_adapter_weights = lambda model: calls.append("adapter")

    assert engine.sync_weights_from_hf(adapter_model, policy_version=1)
    assert calls == ["adapter"]

    non_main = _sync_engine(main=False)
    non_main._ensure_communicator = lambda: pytest.fail("non-main rank initialized communicator")
    assert not non_main.sync_weights_from_hf(adapter_model, policy_version=1)


def test_sync_propagates_official_client_failure():
    engine = _sync_engine(mode="full")
    engine._sync_full_weights = lambda model: (_ for _ in ()).throw(RuntimeError("transfer failed"))

    with pytest.raises(RuntimeError, match="transfer failed"):
        engine.sync_weights_from_hf(object(), policy_version=1)


def test_convert_official_rollout_output_preserves_tokens_and_logprobs():
    engine = SwiftEngine.__new__(SwiftEngine)
    engine.tokenizer = _Tokenizer()
    req = _PendingRequest(
        messages=[{"role": "user", "content": "q"}],
        kwargs={},
        future=None,
        prompt_text="q",
        prompt_ids=[113],
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content="ok"),
        token_ids=[111, 107],
        finish_reason="stop",
        logprobs=None,
    )
    result = SimpleNamespace(
        response=SimpleNamespace(choices=[choice]),
        response_token_ids=[[111, 107]],
        rollout_logprobs=[[-0.1, -0.2]],
    )

    output = engine._convert_server_output(req, result)

    assert output.text == "ok"
    assert output.prompt_ids == [113]
    assert output.completion_ids == [111, 107]
    assert output.logprobs == [-0.1, -0.2]
    assert output.finish_reason == "stop"


def test_server_batch_uses_official_infer_endpoint():
    engine = SwiftEngine.__new__(SwiftEngine)
    engine.tokenizer = _Tokenizer()
    engine.api_retries = 1
    engine.max_response_length = 8
    engine.sampling_params = {}
    engine._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
    captured = {}

    choice = SimpleNamespace(
        message=SimpleNamespace(content="a"),
        token_ids=[97],
        finish_reason="length",
        logprobs=None,
    )
    result = SimpleNamespace(
        response=SimpleNamespace(choices=[choice]),
        response_token_ids=[[97]],
        rollout_logprobs=[[-0.5]],
    )

    def infer(requests, request_config, use_tqdm):
        captured["requests"] = requests
        captured["request_config"] = request_config
        return [result]

    engine._swift_client = SimpleNamespace(infer=infer)
    req = _PendingRequest(
        messages=[{"role": "user", "content": "q"}],
        kwargs={"temperature": 0.6},
        future=None,
        prompt_text="q",
        prompt_ids=[113],
    )
    try:
        outputs = asyncio.run(engine._server_generate_batch([req]))
    finally:
        engine._executor.shutdown()

    assert captured["requests"] == [{"messages": req.messages}]
    assert captured["request_config"]["return_details"] is True
    assert captured["request_config"]["logprobs"] is True
    assert outputs[0].completion_ids == [97]
