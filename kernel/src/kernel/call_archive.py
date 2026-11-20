"""Shared projection helpers for the model-call archive response snapshots.

Spec: docs/SPEC/MODEL-CALL-ARCHIVE.md — one response snapshot per model
call, recorded verbatim from the normalized completion. The helpers here
own the field mapping both recording sites share: the turn loop's in-turn
recorder and ``OperationHost``'s off-turn recorder. Both produce the same
dict shape, so a reader dereferences either archive by its ``r<sha256>``
ref without knowing which site wrote it.

The module is a leaf (no kernel imports beyond the standard library), so
both the turn runner and the host adapter can import it without cycles.
"""

from __future__ import annotations

import copy
from typing import Any

#: Bounded exception message length (Spec §2: never a traceback or headers —
#: those can carry provider metadata the normalized form deliberately drops).
ERROR_MESSAGE_LIMIT = 400


def call_error_payload(error: BaseException) -> dict[str, str]:
    """Bound one failed call's error to the exception class and a short message."""
    return {
        "class": type(error).__name__,
        "message": str(error)[:ERROR_MESSAGE_LIMIT],
    }


def response_usage_snapshot(usage: Any) -> dict[str, Any] | None:
    """Project one ``CompletionUsage`` verbatim under unknown-is-not-zero.

        Every counter stays ``None`` when the provider did not surface it —
        unknown; a present ``0`` is a real measured zero (the same rules the
    token ledger records under). ``None`` input (no usage at all) returns
        ``None`` — the call's usage is unknown, not zero.
    """
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_tokens": getattr(usage, "cached_tokens", None),
        "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
        "cache_creation_tokens": getattr(usage, "cache_creation_tokens", None),
        "duration_ms": getattr(usage, "duration_ms", None),
        "provider_cost_usd": getattr(usage, "provider_cost_usd", None),
        "metric_origins": dict(getattr(usage, "metric_origins", None) or {}),
        "proxy_telemetry": dict(getattr(usage, "proxy_telemetry", None) or {}),
    }


def continuation_snapshot(continuation: Any) -> dict[str, Any] | None:
    """Project one ``ProviderContinuation`` presence verbatim.

    The items are provider-owned opaque context rehydrated only for the
    originating model; they are deep-copied so the archived snapshot cannot
    alias the live continuation the next call rehydrates.
    """
    if continuation is None:
        return None
    return {
        "provider": str(getattr(continuation, "provider", "") or ""),
        "model_id": str(getattr(continuation, "model_id", "") or ""),
        "items": copy.deepcopy(getattr(continuation, "items", None) or []),
    }
