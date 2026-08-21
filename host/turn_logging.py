"""Structured log-record assembly for the host.

This module centralises the *record assembly* logic for the JSONL turn log
and runtime event log.  It accepts a small ``LogContext`` dict of resolved
runtime values (log level, timezone, mode id, etc.) so the module is pure
and testable without importing the full host.

Assembled records are emitted through stdlib logging via
:func:`emit_log_record`; the shared :class:`sdk.logging_setup.JSONFormatter`
owns serialization and the file destination. The previous bespoke
file-append writer has been retired.

Dependency direction
--------------------
``turn_logging.py`` ← ``main.py``  (main builds the LogContext and calls the
assemblers; this module never imports main)

Public surface (re-exported by ``main.py`` for backward test compatibility)
---------------------------------------------------------------------------
- :func:`build_runtime_log_metadata`
- :func:`build_log_record`
- :func:`build_turn_log_record`
- :func:`build_turn_log_debug_payload`
- :func:`build_turn_completed_payload`
- :func:`emit_log_record`
- :func:`write_log_record` (deprecated wrapper around :func:`emit_log_record`)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── LogContext type ──────────────────────────────────────────────────

#: Keys that ``main.py`` must supply when calling the assemblers.
#: All values are already resolved (strings, ints, bools) — no state dict
#: access happens inside this module.
REQUIRED_LOG_CONTEXT_KEYS = (
    "log_level",
    "debug",
    "timezone",
    "timezone_source",
    "session_context",
    "mode_id",
    "stage_id",
    "next_model_provider",
    "next_model_id",
    "stream_enabled",
    "streaming_mode",
    "streaming_char_delay_ms",
)


def build_runtime_log_metadata(ctx: dict[str, Any]) -> dict[str, Any]:
    """Build the nested ``"runtime"`` metadata block for a log record.

    *ctx* must contain the keys listed in :data:`REQUIRED_LOG_CONTEXT_KEYS`.
    """
    return {
        "runtime": {
            "log_level": ctx.get("log_level", "info"),
            "debug": ctx.get("debug", False),
            "timezone": ctx.get("timezone", "UTC"),
            "timezone_source": ctx.get("timezone_source", "default"),
            "session_context": ctx.get("session_context") or "none",
            "mode_id": ctx.get("mode_id", "default"),
            "stage_id": ctx.get("stage_id", "-"),
            "next_model": {
                "provider": ctx.get("next_model_provider", "-"),
                "model_id": ctx.get("next_model_id", "-"),
            },
            "streaming": {
                "enabled": ctx.get("stream_enabled", False),
                "mode": ctx.get("streaming_mode", "none"),
                "char_delay_ms": ctx.get("streaming_char_delay_ms", 0),
            },
        }
    }


def build_log_record(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
    ctx: dict[str, Any],
    debug_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Assemble one JSONL log record, or return ``None`` if filtered out.

    The caller is responsible for level normalisation and threshold
    checking.  This function only assembles the record dict.

    When *ctx*["log_level"] is ``"debug"`` and *debug_payload* is available,
    the debug block is attached.
    """
    record: dict[str, Any] = {
        "logged_at_utc": datetime.now(UTC).isoformat(),
        "level": level,
        "event": event,
        **payload,
    }
    record.update(build_runtime_log_metadata(ctx))

    if ctx.get("log_level") == "debug":
        debug_data = dict(debug_payload or {})
        if debug_data:
            record["debug"] = debug_data

    return record


def build_turn_log_record(
    *,
    context: dict[str, Any],
    response_text: str,
    coerce_int: Any,
    preview_fn: Any,
) -> dict[str, Any]:
    """Assemble the ``turn.completed`` log payload from a turn context."""
    response_chars = len(response_text)
    model_info = _extract_model_info(context)
    completion_info = _extract_completion_info(context, coerce_int)

    mode_model = _format_model_label(model_info["model_id"], model_info["model_provider"])

    return {
        "type": "little_lisa_turn",
        "context": context,
        "mode_model": mode_model,
        "mode_model_id": model_info["model_id"],
        "mode_model_provider": model_info["model_provider"],
        "mode_reasoning_model": model_info["reasoning_model"],
        "mode_prompt_tokens": completion_info["prompt_tokens"],
        "mode_completion_tokens": completion_info["completion_tokens"],
        "mode_total_tokens": completion_info["total_tokens"],
        "response": {
            "text": response_text,
            "chars": response_chars,
        },
        "route": {
            "mode_id": str(context.get("mode_id") or ""),
            "mode_kind": str(context.get("mode_kind") or ""),
            "query_chars": len(str(context.get("query") or "")),
            "handoff": str(context.get("handoff") or ""),
        },
        "completion_summary": {
            "tool_call_count": completion_info["tool_call_count"],
            "iteration_count": completion_info["iteration_count"],
            "tokens": {
                "prompt_tokens": completion_info["prompt_tokens"],
                "completion_tokens": completion_info["completion_tokens"],
                "total_tokens": completion_info["total_tokens"],
            },
        },
        "response_preview": preview_fn(response_text, limit=320),
    }


def build_turn_completed_payload(
    *,
    mode_id: str,
    result: Any,
    coerce_int: Any,
) -> dict[str, Any]:
    """Assemble the ``turn.completed`` audit payload for a kernel turn.

    The effect-receipt ledger is **not** duplicated here — the thread entry
    (written by ``handle_message._execute_mode_turn``) is the canonical
    source of receipts and is addressable by turn number via
    ``thread_read_range`` / the markdown render operation. The event links
    to the entry by turn number, not by a second receipt copy.
    Hallucination signals (a mode with a completion contract whose ledger
    is empty) are queryable from the thread entry's ``effect_receipts``
    field without reconstructing individual dispatch entries.
    """
    return {
        "mode_id": mode_id,
        "handoff": str(getattr(result, "handoff_to", None) or ""),
        "tool_call_count": coerce_int(getattr(result, "tool_call_count", 0)),
        "iteration_count": coerce_int(getattr(result, "iteration_count", 0)),
        "blocked": bool(getattr(result, "blocked", False)),
        "block_origin": str(getattr(result, "block_origin", "") or ""),
    }


def _extract_model_info(context: dict[str, Any]) -> dict[str, Any]:
    """Extract model_id, provider, and reasoning flag from context."""
    mode_model_payload = context.get("resolved_model") if isinstance(context, dict) else None
    model_id = ""
    model_provider = ""
    reasoning_model = False
    if isinstance(mode_model_payload, dict):
        model_id = str(mode_model_payload.get("model_id") or "")
        model_provider = str(mode_model_payload.get("provider") or "")
        reasoning_model = bool(mode_model_payload.get("reasoning_model", False))
    return {
        "model_id": model_id,
        "model_provider": model_provider,
        "reasoning_model": reasoning_model,
    }


def _extract_completion_info(context: dict[str, Any], coerce_int: Any) -> dict[str, Any]:
    """Extract completion token and tool-call counts from context."""
    completion = context.get("completion") if isinstance(context, dict) else None
    tool_call_count = 0
    iteration_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if isinstance(completion, dict):
        tool_call_count = coerce_int(completion.get("tool_call_count"))
        iteration_count = coerce_int(completion.get("iteration_count"))
        prompt_tokens = coerce_int(completion.get("prompt_tokens"))
        completion_tokens = coerce_int(completion.get("completion_tokens"))
        total_tokens = coerce_int(completion.get("total_tokens"))
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
    return {
        "tool_call_count": tool_call_count,
        "iteration_count": iteration_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _format_model_label(model_id: str, model_provider: str) -> str:
    """Format a 'provider:model_id' label, or just model_id."""
    if model_provider and model_id:
        return f"{model_provider}:{model_id}"
    return model_id


def build_turn_log_debug_payload(
    *,
    context: dict[str, Any],
    response_text: str,
    state: dict[str, Any],
    coerce_int: Any,
    thread_dir: str,
    state_snapshot_loaded: bool,
) -> dict[str, Any]:
    """Assemble the debug payload for a ``turn.completed`` log record.

    Reads session token counters from *state* and includes them in the
    debug block when the runtime log level is ``debug``.
    """
    return {
        "context_keys": sorted(context.keys()),
        "turn_input": context.get("turn_input"),
        "completion": context.get("completion"),
        "response_full_text": response_text,
        "thread_dir": thread_dir,
        "state_snapshot_loaded": state_snapshot_loaded,
        "session_tokens": {
            "turn_prompt_tokens": coerce_int(state.get("turn_prompt_tokens")),
            "turn_completion_tokens": coerce_int(state.get("turn_completion_tokens")),
            "turn_total_tokens": coerce_int(state.get("turn_total_tokens")),
            "session_prompt_tokens": coerce_int(state.get("session_prompt_tokens")),
            "session_completion_tokens": coerce_int(state.get("session_completion_tokens")),
            "session_total_tokens": coerce_int(state.get("session_total_tokens")),
        },
    }


def emit_log_record(record: dict[str, Any], *, target: logging.Logger) -> None:
    """Emit one assembled record through *target* via stdlib ``extra=``.

    Replaces the previous bespoke file-append writer. The record dict
    (produced by :func:`build_log_record` or :func:`build_turn_log_record`)
    is split into the stdlib ``message`` + ``extra`` fields so the shared
    :class:`sdk.logging_setup.JSONFormatter` serialises it.

    The ``level`` key in *record* selects the stdlib level; ``event`` (when
    present) becomes the structured ``event`` field. The nested
    ``runtime`` block is preserved as a top-level ``runtime`` extra key so
    existing JSONL consumers keep the same field shape.
    """
    level_name = str(record.get("level", "info")).upper()
    if level_name == "WARN":
        level_name = "WARNING"
    level = getattr(logging, level_name, logging.INFO)
    event = record.get("event")
    message = record.get("type") or event or "turn_logging.event"
    extra: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"level", "event", "type"}:
            continue
        extra[key] = value
    if event:
        extra["event"] = event
    target.log(level, message, extra=extra)


# Retained for backward test compatibility — now a thin wrapper around
# :func:`emit_log_record`. New call sites should call ``emit_log_record``
# directly with a stdlib logger.
def write_log_record(record: dict[str, Any], log_path: Path | None) -> None:
    """Deprecated: route *record* through stdlib logging.

    The *log_path* argument is ignored — the shared handler in
    :mod:`sdk.logging_setup` owns the file destination. Kept only so
    existing tests/importers do not break during the cutover.
    """
    del log_path  # unused under the unified logging stack
    emit_log_record(record, target=logger)
