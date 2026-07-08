"""Structured JSONL logging and turn-log assembly for Little Lisa.

This module centralises the *record assembly* logic for the JSONL turn log
and runtime event log.  It accepts a small ``LogContext`` dict of resolved
runtime values (log level, timezone, mode id, etc.) so the module is pure
and testable without importing the full host.

The actual file I/O (``write_log_record``) is also here but called from
``main.py`` via a thin wrapper that supplies the runtime log-path accessor.

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
- :func:`write_log_record`
"""

from __future__ import annotations

import json
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
    "timeout_profile",
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
            "timeout_profile": ctx.get("timeout_profile", "balanced"),
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
    last_frontdoor_debug: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Assemble one JSONL log record, or return ``None`` if filtered out.

    The caller is responsible for level normalisation and threshold
    checking.  This function only assembles the record dict.

    When *ctx*["log_level"] is ``"debug"`` and *debug_payload* (or
    *last_frontdoor_debug*) is available, the debug block is attached.
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
        if "last_frontdoor_debug" not in debug_data and isinstance(last_frontdoor_debug, dict):
            debug_data["last_frontdoor_debug"] = last_frontdoor_debug
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
    frontdoor_info = _extract_frontdoor_info(context)
    completion_info = _extract_completion_info(context, coerce_int)

    mode_model = _format_model_label(model_info["model_id"], model_info["model_provider"])
    frontdoor_model = _format_model_label(
        frontdoor_info["model_id"], frontdoor_info["model_provider"]
    )

    return {
        "type": "little_lisa_turn",
        "context": context,
        "mode_model": mode_model,
        "mode_model_id": model_info["model_id"],
        "mode_model_provider": model_info["model_provider"],
        "mode_reasoning_model": model_info["reasoning_model"],
        "frontdoor_model": frontdoor_model,
        "frontdoor_model_id": frontdoor_info["model_id"],
        "frontdoor_model_provider": frontdoor_info["model_provider"],
        "frontdoor_reasoning_model": frontdoor_info["reasoning_model"],
        "mode_prompt_tokens": completion_info["prompt_tokens"],
        "mode_completion_tokens": completion_info["completion_tokens"],
        "mode_total_tokens": completion_info["total_tokens"],
        "frontdoor_prompt_tokens": frontdoor_info["prompt_tokens"],
        "frontdoor_completion_tokens": frontdoor_info["completion_tokens"],
        "frontdoor_total_tokens": frontdoor_info["total_tokens"],
        "frontdoor_completion_attempt_count": frontdoor_info["attempt_count"],
        "response": {
            "text": response_text,
            "chars": response_chars,
        },
        "route": {
            "mode_id": str(context.get("mode_id") or ""),
            "mode_kind": str(context.get("mode_kind") or ""),
            "query_chars": len(str(context.get("query") or "")),
            "handoff": str(context.get("handoff") or ""),
            "deep_route": bool(context.get("deep_route", False)),
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


def _extract_frontdoor_info(context: dict[str, Any]) -> dict[str, Any]:
    """Extract frontdoor model and token info from context."""
    frontdoor_debug = context.get("frontdoor_debug") if isinstance(context, dict) else None
    model_id = ""
    model_provider = ""
    reasoning_model = False
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    attempt_count = 0

    if isinstance(frontdoor_debug, dict):
        candidate_payload = frontdoor_debug.get("resolved_model")
        if isinstance(candidate_payload, dict):
            model_id = str(candidate_payload.get("model_id") or "")
            model_provider = str(candidate_payload.get("provider") or "")
            reasoning_model = bool(candidate_payload.get("reasoning_model", False))
        prompt_tokens = coerce_int_or_zero(frontdoor_debug, "planner_prompt_tokens")
        completion_tokens = coerce_int_or_zero(frontdoor_debug, "planner_completion_tokens")
        total_tokens = coerce_int_or_zero(frontdoor_debug, "planner_total_tokens")
        planner_attempts = frontdoor_debug.get("planner_completion_attempts")
        if isinstance(planner_attempts, list):
            attempt_count = len(planner_attempts)

    return {
        "model_id": model_id,
        "model_provider": model_provider,
        "reasoning_model": reasoning_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "attempt_count": attempt_count,
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


def coerce_int_or_zero(data: dict[str, Any], key: str) -> int:
    """Safely coerce a dict value to int, defaulting to 0."""
    value = data.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


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


def write_log_record(record: dict[str, Any], log_path: Path | None) -> None:
    """Write one JSON record as a single line to *log_path*.

    Does nothing if *log_path* is ``None``.  Parent directories are
    created lazily on first write.
    """
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.error("failed to write log record to %s: %s: %s", log_path, type(exc).__name__, exc)
