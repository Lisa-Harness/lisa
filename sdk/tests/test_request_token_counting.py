"""Request counting is local and includes every serialized input surface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kernel.context.request_budget import estimate_request_tokens, request_input_text
from kernel.provider_contracts import CompletionRequest
from sdk.providers.token_counting import count_request_tokens


def _request(model_id: str) -> CompletionRequest:
    return CompletionRequest(
        messages=[{"role": "user", "content": "def f(x):\n    return x + 1\n"}],
        model_def=SimpleNamespace(model_id=model_id),
        max_tokens=100,
        temperature=0.4,
    )


def test_unknown_tokenizer_is_explicitly_estimated() -> None:
    request = _request("ollama/unknown-model:cloud")
    count = count_request_tokens(request)
    assert count == estimate_request_tokens(request)
    assert "estimate" in count.method
    request.tools = [{"function": {"parameters": {"description": "中" * 1000}}}]
    assert count_request_tokens(request).tokens > count.tokens


def test_known_tokenizer_counts_code_unicode_arguments_and_schema() -> None:
    import tiktoken

    # The o200k_base encoder downloads from openaipublic.blob.core.windows.net
    # on first use — a network fetch no offline unit run may depend on (the
    # 2026-10-14 macOS CI runner hit a DNS failure for that host inside the
    # "not live" suite). Exercise the exact-tokenizer assertions when the
    # encoding resolves (already cached locally); otherwise keep the
    # estimation-path coverage so the suite stays offline-hermetic.
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        request = _request("openai/gpt-4o")
        count = count_request_tokens(request)
        assert "estimate" in count.method
        return
    request = _request("openai/gpt-4o")
    request.tools = [{"function": {"name": "read", "parameters": {"description": "中文"}}}]
    request.messages.append(
        {"role": "assistant", "tool_calls": [{"function": {"arguments": '{"path":"a"}'}}]}
    )
    count = count_request_tokens(request)
    assert count.tokens == len(encoding.encode(request_input_text(request)))
    assert count.method == "tiktoken_o200k_base_request_estimate"


def test_known_tokenizer_degrades_to_estimate_when_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A load failure for a known encoding degrades to the estimate.

    The 2026-11-19 offline run exposed the gap: ``encoding_for_model``
    raised ConnectionError while downloading o200k_base on a cold
    cache, and the loader caught only KeyError, so count_request_tokens
    crashed instead of estimating — taking the turn loop's per-request
    token accounting (and this suite's own offline fallback) down with
    it. Any load failure must degrade, never raise.
    """
    import tiktoken
    from sdk.providers.token_counting import _encoding

    def _unreachable(model: str) -> tiktoken.Encoding:
        raise ConnectionError("network is down")

    monkeypatch.setattr(tiktoken, "encoding_for_model", _unreachable)
    _encoding.cache_clear()
    try:
        count = count_request_tokens(_request("openai/gpt-4o"))
    finally:
        _encoding.cache_clear()

    assert "estimate" in count.method
