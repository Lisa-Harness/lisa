"""Local request accounting; unknown model tokenizers are explicitly estimated."""

from __future__ import annotations

from functools import lru_cache

import tiktoken
from kernel.context.request_budget import estimate_request_tokens, request_input_text
from kernel.provider_contracts import CompletionRequest, RequestTokenCount


@lru_cache(maxsize=32)
def _encoding(model_id: str) -> tiktoken.Encoding | None:
    # OpenRouter and Azure often prefix the provider or use a deployment alias.
    model = model_id.rsplit("/", 1)[-1]
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return None
    except Exception:  # pylint: disable=broad-except
        # A known encoding that cannot load right now — most commonly the
        # first-use o200k_base download failing offline (ConnectionError,
        # seen 2026-11-19). Token accounting must degrade to the estimate
        # instead of raising: the turn loop calls this hook on every model
        # request, and an offline host with a cold cache would otherwise
        # crash the turn at accounting time.
        return None


def count_request_tokens(request: CompletionRequest) -> RequestTokenCount:
    encoding = _encoding(request.model_def.model_id)
    if encoding is None:
        return estimate_request_tokens(request)
    # Content tokenization is model-specific; JSON framing and tool-schema
    # translation remain an estimate, covered by the explicit window margin.
    return RequestTokenCount(
        tokens=len(encoding.encode(request_input_text(request), disallowed_special=())),
        method=f"tiktoken_{encoding.name}_request_estimate",
    )
