"""Token accounting and session usage tracking for the host.

This module centralises all token/latency ledger logic that was previously
inline in ``main.py``.  Every function here accepts the runtime state dict as
an explicit parameter so the module is pure (no global mutable state) and
independently testable.

The key data flows are:

1. **Per-turn recording** — :func:`apply_turn_usage` takes a completed
   ``TurnResult`` plus optional routing metadata, updates the
   per-turn counters, appends a ledger entry, and accumulates breakdowns
   by model / mode / stage.

2. **Session roll-up** — :func:`format_session_exit_summary_lines` reads
   the accumulated counters and breakdowns to render a Markdown summary
   table for the session exit banner and the ``/tokens`` command.

3. **Turn bookkeeping** — :func:`record_session_turn` increments
   success/error counters and mode-frequency counts.

Dependency direction
--------------------
``token_accounting.py`` ← ``main.py`` (main imports and wraps; module
never imports main so there is no circular dependency).

Re-exported names (wrapped in main.py for backward test compatibility)
---------------------------------------------------------------------
- :func:`apply_turn_usage`  ←  ``_apply_turn_usage``
- :func:`record_session_turn`  ←  ``_record_session_turn``
- :func:`top_session_mode`  ←  ``_top_session_mode``
- :func:`format_session_exit_summary_lines`  ←  ``_format_session_exit_summary_lines``
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Import coercion helpers from kernel
from kernel.token_usage import (
    coerce_non_negative_float,
    coerce_non_negative_int,
    ensure_usage_entry,
)

__all__ = [
    "apply_turn_usage",
    "coerce_non_negative_float",
    "coerce_non_negative_int",
    "ensure_usage_entry",
    "format_session_exit_summary_lines",
    "record_session_turn",
]


# ── Usage-entry accumulator ───────────────────────────────────────────

#: The 7 counter fields every breakdown entry tracks.
_USAGE_FIELDS: tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "duration_ms",
    "latency_ms",
)


def _new_usage_entry() -> dict[str, Any]:
    """Return a fresh usage-entry dict with all counters zeroed."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "duration_ms": 0.0,
        "latency_ms": 0,
    }


def _accumulate_entry(
    entry: dict[str, Any],
    *,
    prompt: int,
    completion: int,
    total: int,
    cached: int,
    reasoning: int,
    duration_ms: float,
    latency_ms: int,
) -> None:
    """Add one set of counters into a breakdown entry (in place)."""
    entry["prompt_tokens"] += prompt
    entry["completion_tokens"] += completion
    entry["total_tokens"] += total
    entry["cached_tokens"] += cached
    entry["reasoning_tokens"] += reasoning
    entry["duration_ms"] += duration_ms
    entry["latency_ms"] += latency_ms


def _accumulate_breakdown(
    state: dict[str, Any],
    breakdown_key: str,
    category_key: str,
    *,
    prompt: int,
    completion: int,
    total: int,
    cached: int,
    reasoning: int,
    duration_ms: float,
    latency_ms: int,
) -> None:
    """Accumulate one row into a named breakdown dict in ``state``."""
    breakdown: dict[str, dict[str, Any]] = state.setdefault(breakdown_key, {})
    entry = ensure_usage_entry(breakdown.setdefault(category_key, _new_usage_entry()))
    _accumulate_entry(
        entry,
        prompt=prompt,
        completion=completion,
        total=total,
        cached=cached,
        reasoning=reasoning,
        duration_ms=duration_ms,
        latency_ms=latency_ms,
    )


# ── Public accounting functions ──────────────────────────────────────


def apply_turn_usage(
    state: dict[str, Any],
    result: Any,
    *,
    model_id: str = "",
    mode_id: str = "",
    stage_id: str = "",
) -> None:
    """Persist turn and session token counters from one completed turn result."""
    main_tokens = _extract_main_tokens(result)

    turn_tokens = _combine_turn_tokens(main_tokens)
    _set_turn_state(state, turn_tokens)
    _accumulate_session_state(state, turn_tokens)

    ledger: list[dict[str, Any]] = state.setdefault("session_token_ledger", [])
    turn_index = len(ledger) + 1

    _record_main_ledger(state, ledger, turn_index, main_tokens, model_id, mode_id, stage_id)


def _extract_main_tokens(result: Any) -> dict[str, Any]:
    """Extract token counts from a TurnResult."""
    prompt = coerce_non_negative_int(result.prompt_tokens)
    completion = coerce_non_negative_int(result.completion_tokens)
    total = coerce_non_negative_int(result.total_tokens)
    if total <= 0:
        total = prompt + completion
    return {
        "prompt": prompt,
        "completion": completion,
        "total": total,
        "cached": coerce_non_negative_int(getattr(result, "cached_tokens", 0)),
        "reasoning": coerce_non_negative_int(getattr(result, "reasoning_tokens", 0)),
        "duration_ms": coerce_non_negative_float(getattr(result, "duration_ms", 0.0)),
        "latency_ms": coerce_non_negative_int(getattr(result, "latency_ms", 0)),
    }


def _combine_turn_tokens(main: dict[str, Any]) -> dict[str, Any]:
    """Return the turn token totals from the main completion counters."""
    return {
        "prompt": main["prompt"],
        "completion": main["completion"],
        "total": main["total"],
        "cached": main["cached"],
        "reasoning": main["reasoning"],
        "duration_ms": main["duration_ms"],
        "latency_ms": main["latency_ms"],
    }


def _set_turn_state(state: dict[str, Any], turn: dict[str, Any]) -> None:
    """Set the per-turn token counters in state."""
    state["turn_prompt_tokens"] = turn["prompt"]
    state["turn_completion_tokens"] = turn["completion"]
    state["turn_total_tokens"] = turn["total"]
    state["turn_cached_tokens"] = turn["cached"]
    state["turn_reasoning_tokens"] = turn["reasoning"]
    state["turn_duration_ms"] = turn["duration_ms"]
    state["turn_latency_ms"] = turn["latency_ms"]


def _accumulate_session_state(state: dict[str, Any], turn: dict[str, Any]) -> None:
    """Accumulate turn tokens into session totals."""
    state["session_prompt_tokens"] = (
        coerce_non_negative_int(state.get("session_prompt_tokens", 0)) + turn["prompt"]
    )
    state["session_completion_tokens"] = (
        coerce_non_negative_int(state.get("session_completion_tokens", 0)) + turn["completion"]
    )
    state["session_total_tokens"] = (
        coerce_non_negative_int(state.get("session_total_tokens", 0)) + turn["total"]
    )
    state["session_cached_tokens"] = (
        coerce_non_negative_int(state.get("session_cached_tokens", 0)) + turn["cached"]
    )
    state["session_reasoning_tokens"] = (
        coerce_non_negative_int(state.get("session_reasoning_tokens", 0)) + turn["reasoning"]
    )
    state["session_duration_ms"] = (
        coerce_non_negative_float(state.get("session_duration_ms", 0.0)) + turn["duration_ms"]
    )
    state["session_latency_ms"] = (
        coerce_non_negative_int(state.get("session_latency_ms", 0)) + turn["latency_ms"]
    )


def _record_main_ledger(
    state: dict[str, Any],
    ledger: list[dict[str, Any]],
    turn_index: int,
    main: dict[str, Any],
    model_id: str,
    mode_id: str,
    stage_id: str,
) -> None:
    """Record the main execution ledger entry and per-model/mode/stage breakdowns."""
    clean_model = (model_id or "-").strip()
    clean_mode = (mode_id or "-").strip()
    clean_stage = (stage_id or "-").strip()
    logger.debug(
        "recording turn ledger entry: model=%s mode=%s stage=%s total_tokens=%d",
        clean_model,
        clean_mode,
        clean_stage,
        main["total"],
    )
    ledger.append(
        {
            "turn": turn_index,
            "model_id": clean_model,
            "mode_id": clean_mode,
            "stage_id": clean_stage,
            "prompt_tokens": main["prompt"],
            "completion_tokens": main["completion"],
            "total_tokens": main["total"],
            "cached_tokens": main["cached"],
            "reasoning_tokens": main["reasoning"],
            "duration_ms": main["duration_ms"],
            "latency_ms": main["latency_ms"],
        }
    )
    main_kwargs = _main_breakdown_kwargs(main)
    _accumulate_breakdown(state, "model_token_totals", clean_model, **main_kwargs)
    _accumulate_breakdown(state, "mode_token_totals", clean_mode, **main_kwargs)
    _accumulate_breakdown(state, "stage_token_totals", clean_stage, **main_kwargs)


def _main_breakdown_kwargs(main: dict[str, Any]) -> dict[str, Any]:
    """Convert main token dict to _accumulate_breakdown kwargs."""
    return {
        "prompt": main["prompt"],
        "completion": main["completion"],
        "total": main["total"],
        "cached": main["cached"],
        "reasoning": main["reasoning"],
        "duration_ms": main["duration_ms"],
        "latency_ms": main["latency_ms"],
    }


def record_session_turn(
    state: dict[str, Any],
    *,
    succeeded: bool,
    latency_ms: float,
    mode_id: str | None = None,
) -> None:
    """Increment turn counters and mode-frequency counts in *state*."""
    state["session_turn_count"] = int(state.get("session_turn_count", 0)) + 1
    if succeeded:
        state["session_success_count"] = int(state.get("session_success_count", 0)) + 1
    else:
        state["session_error_count"] = int(state.get("session_error_count", 0)) + 1
        logger.warning("turn failed (mode=%s, latency_ms=%.0f)", mode_id or "-", latency_ms)
    total_latency = float(state.get("session_total_latency_ms", 0.0))
    state["session_total_latency_ms"] = total_latency + max(0.0, latency_ms)

    selected_mode = (mode_id or "").strip()
    if selected_mode:
        mode_counts = dict(state.get("session_mode_counts") or {})
        mode_counts[selected_mode] = int(mode_counts.get(selected_mode, 0)) + 1
        state["session_mode_counts"] = mode_counts
    logger.debug(
        "session turn recorded: total=%d ok=%d errors=%d latency_ms=%.0f",
        int(state.get("session_turn_count", 0)),
        int(state.get("session_success_count", 0)),
        int(state.get("session_error_count", 0)),
        latency_ms,
    )


def top_session_mode(state: dict[str, Any]) -> tuple[str, int] | None:
    """Return ``(mode_id, count)`` for the most-used mode, or ``None``."""
    mode_counts = state.get("session_mode_counts")
    if not isinstance(mode_counts, dict) or not mode_counts:
        return None
    best_mode = ""
    best_count = 0
    for mode, count in mode_counts.items():
        mode_name = str(mode).strip()
        if not mode_name:
            continue
        count_value = int(count)
        if count_value > best_count:
            best_mode = mode_name
            best_count = count_value
    if not best_mode:
        return None
    return best_mode, best_count


def format_session_exit_summary_lines(
    state: dict[str, Any],
    *,
    session_started_utc: datetime | None = None,
) -> list[str]:
    """Build the Markdown session-summary lines for the exit banner.

    Reads all counters and breakdown dicts from *state* and returns a list
    of Markdown lines.  The caller is responsible for rendering them.
    """
    now_utc = datetime.now(UTC)
    started_utc = session_started_utc or state.get("session_started_utc")
    if not isinstance(started_utc, datetime):
        started_utc = datetime.now(UTC)
    duration_seconds = max(0.0, (now_utc - started_utc).total_seconds())

    turn_count = int(state.get("session_turn_count", 0))
    success_count = int(state.get("session_success_count", 0))
    error_count = int(state.get("session_error_count", 0))
    cancelled_count = int(state.get("session_cancelled_count", 0))
    total_latency_ms = float(state.get("session_total_latency_ms", 0.0))
    avg_latency_ms = (total_latency_ms / turn_count) if turn_count else 0.0

    session_prompt = int(state.get("session_prompt_tokens", 0) or 0)
    session_completion = int(state.get("session_completion_tokens", 0) or 0)
    session_total = int(state.get("session_total_tokens", 0) or 0)
    session_cached = int(state.get("session_cached_tokens", 0) or 0)
    session_reasoning = int(state.get("session_reasoning_tokens", 0) or 0)
    session_duration_ms = float(state.get("session_duration_ms", 0.0) or 0.0)
    turn_prompt = int(state.get("turn_prompt_tokens", 0) or 0)
    turn_completion = int(state.get("turn_completion_tokens", 0) or 0)
    turn_total = int(state.get("turn_total_tokens", 0) or 0)
    turn_cached = int(state.get("turn_cached_tokens", 0) or 0)
    turn_reasoning = int(state.get("turn_reasoning_tokens", 0) or 0)

    avg_prompt = (session_prompt / turn_count) if turn_count else 0.0
    avg_completion = (session_completion / turn_count) if turn_count else 0.0
    avg_total_tokens = (session_total / turn_count) if turn_count else 0.0
    avg_duration_ms = (session_duration_ms / turn_count) if turn_count else 0.0

    cancelled_line = f" (cancelled: {cancelled_count})" if cancelled_count else ""
    lines = [
        "## Session Summary",
        f"- Duration: {duration_seconds:.1f}s",
        f"- Turns: {turn_count} (ok: {success_count}, errors: {error_count}{cancelled_line})",
        f"- Avg turn latency: {avg_latency_ms:.0f} ms",
        f"- Provider time: {session_duration_ms:.0f} ms (avg {avg_duration_ms:.0f} ms/turn)",
        "",
        "### Token Totals",
        "| Category | Prompt | Completion | Total | Cached | Reasoning |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Session | {session_prompt} | {session_completion} | {session_total} | "
        f"{session_cached} | {session_reasoning} |",
        f"| Last turn | {turn_prompt} | {turn_completion} | {turn_total} | "
        f"{turn_cached} | {turn_reasoning} |",
        f"| Avg/turn | {avg_prompt:.0f} | {avg_completion:.0f} | {avg_total_tokens:.0f} | "
        f"{(session_cached / turn_count) if turn_count else 0:.0f} | "
        f"{(session_reasoning / turn_count) if turn_count else 0:.0f} |",
    ]

    # ── Per-model breakdown ────────────────────────────────────────
    model_totals: dict[str, dict[str, Any]] = state.get("model_token_totals") or {}
    if model_totals:
        lines.extend(
            [
                "",
                "### By Model",
                "| Model | Prompt | Completion | Total | Cached | Reasoning | Duration ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for model_key in sorted(model_totals):
            entry = model_totals[model_key]
            lines.append(
                f"| {model_key} | {entry.get('prompt_tokens', 0)} | "
                f"{entry.get('completion_tokens', 0)} | {entry.get('total_tokens', 0)} | "
                f"{entry.get('cached_tokens', 0)} | {entry.get('reasoning_tokens', 0)} | "
                f"{entry.get('duration_ms', 0.0):.0f} |"
            )

    # ── Per-mode breakdown ─────────────────────────────────────────
    mode_totals: dict[str, dict[str, Any]] = state.get("mode_token_totals") or {}
    if mode_totals:
        lines.extend(
            [
                "",
                "### By Mode",
                "| Mode | Prompt | Completion | Total | Cached | Reasoning | Duration ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for mode_key in sorted(mode_totals):
            entry = mode_totals[mode_key]
            lines.append(
                f"| {mode_key} | {entry.get('prompt_tokens', 0)} | "
                f"{entry.get('completion_tokens', 0)} | {entry.get('total_tokens', 0)} | "
                f"{entry.get('cached_tokens', 0)} | {entry.get('reasoning_tokens', 0)} | "
                f"{entry.get('duration_ms', 0.0):.0f} |"
            )

    # ── Per-stage breakdown ────────────────────────────────────────
    stage_totals: dict[str, dict[str, Any]] = state.get("stage_token_totals") or {}
    if stage_totals:
        lines.extend(
            [
                "",
                "### By Stage",
                "| Stage | Prompt | Completion | Total | Cached | Reasoning | Duration ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for stage_key in sorted(stage_totals):
            entry = stage_totals[stage_key]
            lines.append(
                f"| {stage_key} | {entry.get('prompt_tokens', 0)} | "
                f"{entry.get('completion_tokens', 0)} | {entry.get('total_tokens', 0)} | "
                f"{entry.get('cached_tokens', 0)} | {entry.get('reasoning_tokens', 0)} | "
                f"{entry.get('duration_ms', 0.0):.0f} |"
            )

    top_mode = top_session_mode(state)
    if top_mode is not None:
        lines.append(f"- Most used mode: {top_mode[0]} ({top_mode[1]})")
    return lines