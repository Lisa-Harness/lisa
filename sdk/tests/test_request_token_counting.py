"""Request counting is local and includes every serialized input surface."""

from __future__ import annotations

from types import SimpleNamespace

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
