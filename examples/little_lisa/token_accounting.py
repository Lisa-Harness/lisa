"""Token accounting and session usage tracking for Little Lisa.

This module centralises all token/latency ledger logic that was previously
inline in ``main.py``.  Every function here accepts the runtime state dict as
an explicit parameter so the module is pure (no global mutable state) and
independently testable.

The key data flows are:

1. **Per-turn recording** — :func:`apply_turn_usage` takes a completed
   ``TurnResult`` plus optional front-door routing metadata, updates the
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

# ── Coercion helpers ──────────────────────────────────────────────────


def coerce_non_negative_int(value: Any) -> int:
    """Safely convert *value* to a non-negative int (0 on failure)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def coerce_non_negative_float(value: Any) -> float:
    """Safely convert *value* to a non-negative float (0.0 on failure)."""
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def ensure_usage_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Ensure one breakdown dict has every token/time field.

    Pre-existing entries (from earlier sessions or tests with the old
    3-key schema) are upgraded in place so the accumulator math never
    raises ``KeyError`` on a missing ``cached_tokens`` etc.
    """
    entry.setdefault("prompt_tokens", 0)
    entry.setdefault("completion_tokens", 0)
    entry.setdefault("total_tokens", 0)
    entry.setdefault("cached_tokens", 0)
    entry.setdefault("reasoning_tokens", 0)
    entry.setdefault("duration_ms", 0.0)
    entry.setdefault("latency_ms", 0)
    return entry


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
    frontdoor_routing: dict[str, Any] | None = None,
) -> None:
    """Persist turn and session token counters from one completed turn result.

    Records a per-turn ledger entry with model/mode/stage attribution and
    accumulates breakdown totals keyed by resolved model id (never designation),
    mode id, and stage id. Sub-counters (``cached_tokens``,
    ``reasoning_tokens``), ``duration_ms``, and ``latency_ms`` flow through
    so time-based providers (e.g. Ollama local GPU time) can be billed
    alongside token-based providers.
    """
    # ── Input extraction ───────────────────────────────────────────
    main_prompt = coerce_non_negative_int(result.prompt_tokens)
    main_completion = coerce_non_negative_int(result.completion_tokens)
    main_total = coerce_non_negative_int(result.total_tokens)
    if main_total <= 0:
        main_total = main_prompt + main_completion
    main_cached = coerce_non_negative_int(getattr(result, "cached_tokens", 0))
    main_reasoning = coerce_non_negative_int(getattr(result, "reasoning_tokens", 0))
    main_duration_ms = coerce_non_negative_float(getattr(result, "duration_ms", 0.0))
    main_latency_ms = coerce_non_negative_int(getattr(result, "latency_ms", 0))

    fd_prompt = 0
    fd_completion = 0
    fd_total = 0
    fd_cached = 0
    fd_reasoning = 0
    fd_duration_ms = 0.0
    fd_latency_ms = 0
    fd_model_id = ""
    if isinstance(frontdoor_routing, dict):
        fd_prompt = coerce_non_negative_int(frontdoor_routing.get("planner_prompt_tokens"))
        fd_completion = coerce_non_negative_int(frontdoor_routing.get("planner_completion_tokens"))
        fd_total = coerce_non_negative_int(frontdoor_routing.get("planner_total_tokens"))
        if fd_total <= 0:
            fd_total = fd_prompt + fd_completion
        fd_cached = coerce_non_negative_int(frontdoor_routing.get("planner_cached_tokens"))
        fd_reasoning = coerce_non_negative_int(frontdoor_routing.get("planner_reasoning_tokens"))
        fd_duration_ms = coerce_non_negative_float(frontdoor_routing.get("planner_duration_ms"))
        fd_latency_ms = coerce_non_negative_int(frontdoor_routing.get("planner_latency_ms"))
        fd_model_id = str(frontdoor_routing.get("resolved_model_id") or "").strip()

    # ── Turn totals ────────────────────────────────────────────────
    turn_prompt = main_prompt + fd_prompt
    turn_completion = main_completion + fd_completion
    turn_total = main_total + fd_total
    turn_cached = main_cached + fd_cached
    turn_reasoning = main_reasoning + fd_reasoning
    turn_duration_ms = main_duration_ms + fd_duration_ms
    turn_latency_ms = main_latency_ms + fd_latency_ms

    state["turn_prompt_tokens"] = turn_prompt
    state["turn_completion_tokens"] = turn_completion
    state["turn_total_tokens"] = turn_total
    state["turn_cached_tokens"] = turn_cached
    state["turn_reasoning_tokens"] = turn_reasoning
    state["turn_duration_ms"] = turn_duration_ms
    state["turn_latency_ms"] = turn_latency_ms

    state["session_prompt_tokens"] = (
        coerce_non_negative_int(state.get("session_prompt_tokens", 0)) + turn_prompt
    )
    state["session_completion_tokens"] = (
        coerce_non_negative_int(state.get("session_completion_tokens", 0)) + turn_completion
    )
    state["session_total_tokens"] = (
        coerce_non_negative_int(state.get("session_total_tokens", 0)) + turn_total
    )
    state["session_cached_tokens"] = (
        coerce_non_negative_int(state.get("session_cached_tokens", 0)) + turn_cached
    )
    state["session_reasoning_tokens"] = (
        coerce_non_negative_int(state.get("session_reasoning_tokens", 0)) + turn_reasoning
    )
    state["session_duration_ms"] = (
        coerce_non_negative_float(state.get("session_duration_ms", 0.0)) + turn_duration_ms
    )
    state["session_latency_ms"] = (
        coerce_non_negative_int(state.get("session_latency_ms", 0)) + turn_latency_ms
    )

    ledger: list[dict[str, Any]] = state.setdefault("session_token_ledger", [])
    turn_index = len(ledger) + 1

    # ── Front-door ledger and breakdowns (if applicable) ───────────
    if fd_total > 0 or fd_duration_ms > 0:
        logger.debug(
            "recording front-door ledger entry: model=%s prompt=%d completion=%d total=%d",
            fd_model_id or "front-door",
            fd_prompt,
            fd_completion,
            fd_total,
        )
        ledger.append(
            {
                "turn": turn_index,
                "model_id": fd_model_id or "front-door",
                "mode_id": "front-door",
                "stage_id": "routing",
                "prompt_tokens": fd_prompt,
                "completion_tokens": fd_completion,
                "total_tokens": fd_total,
                "cached_tokens": fd_cached,
                "reasoning_tokens": fd_reasoning,
                "duration_ms": fd_duration_ms,
                "latency_ms": fd_latency_ms,
            }
        )

        model_key = fd_model_id or "front-door"
        if model_key != "-":
            _accumulate_breakdown(
                state,
                "model_token_totals",
                model_key,
                prompt=fd_prompt,
                completion=fd_completion,
                total=fd_total,
                cached=fd_cached,
                reasoning=fd_reasoning,
                duration_ms=fd_duration_ms,
                latency_ms=fd_latency_ms,
            )

        _accumulate_breakdown(
            state,
            "mode_token_totals",
            "front-door",
            prompt=fd_prompt,
            completion=fd_completion,
            total=fd_total,
            cached=fd_cached,
            reasoning=fd_reasoning,
            duration_ms=fd_duration_ms,
            latency_ms=fd_latency_ms,
        )

    # ── Main execution ledger entry ────────────────────────────────
    logger.debug(
        "recording turn ledger entry: model=%s mode=%s stage=%s total_tokens=%d",
        (model_id or "-").strip(),
        (mode_id or "-").strip(),
        (stage_id or "-").strip(),
        main_total,
    )
    ledger.append(
        {
            "turn": turn_index,
            "model_id": (model_id or "-").strip(),
            "mode_id": (mode_id or "-").strip(),
            "stage_id": (stage_id or "-").strip(),
            "prompt_tokens": main_prompt,
            "completion_tokens": main_completion,
            "total_tokens": main_total,
            "cached_tokens": main_cached,
            "reasoning_tokens": main_reasoning,
            "duration_ms": main_duration_ms,
            "latency_ms": main_latency_ms,
        }
    )

    # ── Per-model / mode / stage breakdowns ────────────────────────
    _accumulate_breakdown(
        state,
        "model_token_totals",
        (model_id or "-").strip(),
        prompt=main_prompt,
        completion=main_completion,
        total=main_total,
        cached=main_cached,
        reasoning=main_reasoning,
        duration_ms=main_duration_ms,
        latency_ms=main_latency_ms,
    )
    _accumulate_breakdown(
        state,
        "mode_token_totals",
        (mode_id or "-").strip(),
        prompt=main_prompt,
        completion=main_completion,
        total=main_total,
        cached=main_cached,
        reasoning=main_reasoning,
        duration_ms=main_duration_ms,
        latency_ms=main_latency_ms,
    )
    _accumulate_breakdown(
        state,
        "stage_token_totals",
        (stage_id or "-").strip(),
        prompt=main_prompt,
        completion=main_completion,
        total=main_total,
        cached=main_cached,
        reasoning=main_reasoning,
        duration_ms=main_duration_ms,
        latency_ms=main_latency_ms,
    )


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

    lines = [
        "## Session Summary",
        f"- Duration: {duration_seconds:.1f}s",
        f"- Turns: {turn_count} (ok: {success_count}, errors: {error_count})",
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
