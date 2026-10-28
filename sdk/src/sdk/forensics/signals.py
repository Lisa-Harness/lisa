"""Behavioral signals over a recent window of Lisa's own turns.

The forensics reader answers "what happened in turn N?" on demand. This module
answers the question nobody asks explicitly: **"is something going wrong
repeatedly?"** It reads a bounded window of thread entries plus the dispatch
ledger and reduces them to a handful of countable signals, so the post-turn
slice can surface a pattern instead of waiting to be asked.

Design notes
------------
**Countable, not interpretive.** Every signal is arithmetic over records the
caller could recompute: a streak length, a ratio, a count. No model call, no
inference about *why*. The rendering says what is measurable; the model
decides what it means.

**Silent when nominal.** A signal is only returned when it crosses its own
threshold. The caller emits nothing when the list is empty, so an ordinary
turn produces no prompt tokens — the same ambient-silence contract the
self-model already holds.

**Independent of operational health.** A blocked streak is not a process-health
problem. `operational_health` answers "is the host degraded?" (memory, tasks,
MCP, log errors); a blocked streak with a healthy process is still worth
surfacing. Gating these behind health would hide exactly the case they exist
for — the 2026-10 ten-turn block streak the operator, not the harness, noticed.

**Bounded.** The window is capped by turns, by records, and by bytes, because
this runs inside a slice budget every turn. Measured on a live log: a 256 KiB
budget reads ~1,000 records in ~3 ms, which leaves the slice budget intact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sdk.forensics._join import dispatch_from_record
from sdk.forensics._logrec import filter_owned, read_jsonl_tail
from sdk.forensics._models import DispatchRecord
from sdk.thread import ThreadStore

__all__ = [
    "BehavioralSignals",
    "SignalThresholds",
    "compute_behavioral_signals",
    "render_signals",
]

#: Window of recent turns a slice-time scan examines.
#:
#: Deliberately small. ``ThreadStore.load_recent`` JSON-decodes each entry,
#: and entries carry full transcripts — a live store measured ~128 KiB per
#: entry, so 40 turns cost ~50 ms and would consume the whole slice budget.
#: Twelve turns is enough to detect a streak of ``SignalThresholds.blocked_streak``
#: consecutive failures and covers a typical mode dwell, at ~14 ms.
#: A caller wanting a wider behavioural view should use ``/forensics`` or the
#: ``turn_forensics`` operation, which are on-demand and not budget-bound.
DEFAULT_WINDOW_TURNS = 12
#: Byte budget per telemetry file for a slice-time scan. Chosen so the scan
#: stays in single-digit milliseconds on a multi-megabyte log.
DEFAULT_SCAN_BYTES = 256 * 1024
#: Cap on ledger rows examined, independent of the byte budget.
DEFAULT_SCAN_RECORDS = 2000


@dataclass(frozen=True, slots=True)
class SignalThresholds:
    """When a signal is worth surfacing.

    Defaults are conservative on purpose: a signal that fires on an ordinary
    turn is noise, and noise is what makes an ambient channel ignorable.
    """

    #: Consecutive blocked turns before a streak is reported.
    blocked_streak: int = 3
    #: Failed dispatches in one turn before it is reported.
    failed_dispatches_in_turn: int = 3
    #: Dispatch failure rate (0-1) across the window, once the window has at
    #: least ``min_dispatches_for_rate`` dispatches.
    dispatch_failure_rate: float = 0.25
    min_dispatches_for_rate: int = 10
    #: Turns of identical mode before dwell is reported.
    mode_dwell: int = 8
    #: Per-turn token total, as a multiple of the window median, before a
    #: spike is reported.
    cost_spike_multiple: float = 3.0


@dataclass
class BehavioralSignals:
    """Countable findings over the window. Empty means nominal."""

    window_turns: int = 0
    signals: list[str] = field(default_factory=list)
    #: Raw counters, always populated so a caller can log or assert without
    #: re-reading. Absent keys mean "not computed", not "zero".
    counters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True when nothing crossed its threshold."""
        return not self.signals


def _blocked(entry: dict[str, Any]) -> bool:
    """Return whether a thread entry records a blocked turn."""
    provenance = entry.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return bool(entry.get("blocked") or provenance.get("block_origin"))


def _block_origin(entry: dict[str, Any]) -> str:
    """Return the recorded block origin, or an empty string."""
    provenance = entry.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return str(entry.get("block_origin") or provenance.get("block_origin") or "")


def _receipt_count(entry: dict[str, Any]) -> int:
    """Return how many effect receipts the entry records."""
    receipts = entry.get("effect_receipts")
    return len(receipts) if isinstance(receipts, list) else 0


def _trailing_blocked_streak(entries: list[dict[str, Any]]) -> tuple[int, Counter[str]]:
    """Return the trailing consecutive blocked-turn count and origin histogram."""
    streak = 0
    origins: Counter[str] = Counter()
    for entry in reversed(entries):
        if not _blocked(entry):
            break
        streak += 1
        origins[_block_origin(entry) or "unknown"] += 1
    return streak, origins


def _mode_dwell(entries: list[dict[str, Any]]) -> tuple[str, int]:
    """Return the trailing run of identical modes as ``(mode_id, length)``."""
    if not entries:
        return "", 0
    last = str(entries[-1].get("mode_id") or "")
    length = 0
    for entry in reversed(entries):
        if str(entry.get("mode_id") or "") != last:
            break
        length += 1
    return last, length


def _median(values: list[int]) -> float:
    """Return the median of ``values`` (0.0 when empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_behavioral_signals(
    store: ThreadStore,
    *,
    decisions_path: Any = None,
    token_usage_path: Any = None,
    user_slug: str | None = None,
    thresholds: SignalThresholds | None = None,
    window_turns: int = DEFAULT_WINDOW_TURNS,
    scan_bytes: int = DEFAULT_SCAN_BYTES,
    scan_records: int = DEFAULT_SCAN_RECORDS,
) -> BehavioralSignals:
    """Scan a bounded recent window and return the findings that crossed threshold.

    ``user_slug`` scopes the shared ledgers to the thread's owner (see
    :func:`filter_owned`); ``None`` reads them unscoped.
    """
    limits = thresholds or SignalThresholds()
    entries = store.load_recent(limit=max(1, window_turns))
    result = BehavioralSignals(window_turns=len(entries))
    if not entries:
        return result

    # Ledger rows are scoped to the turns actually examined; a byte budget
    # alone would admit dispatches and cost from turns outside the window.
    window = {turn for entry in entries if isinstance(turn := entry.get("turn"), int)}
    dispatches = _recent_dispatches(decisions_path, scan_bytes, scan_records, window, user_slug)

    _check_blocked(entries, limits, result)
    _check_dispatches(dispatches, limits, result)
    _check_receipts(entries, dispatches, result)
    _check_mode_dwell(entries, limits, result)
    _check_cost_spike(
        _recent_token_totals(token_usage_path, scan_bytes, user_slug), window, limits, result
    )
    return result


def _check_blocked(
    entries: list[dict[str, Any]],
    limits: SignalThresholds,
    result: BehavioralSignals,
) -> None:
    """Report a trailing streak of blocked turns."""
    streak, origins = _trailing_blocked_streak(entries)
    result.counters["blocked_streak"] = streak
    result.counters["blocked_turns"] = sum(1 for entry in entries if _blocked(entry))
    if not streak:
        return
    result.counters["blocked_streak_origins"] = dict(origins)
    if streak >= limits.blocked_streak:
        origin_text = ", ".join(f"{name}×{count}" for name, count in origins.items())
        result.signals.append(
            f"{streak} consecutive turns ended blocked ({origin_text}); "
            "the same contract failed repeatedly, so the cause is structural "
            "rather than a one-off"
        )


def _check_dispatches(
    records: list[DispatchRecord],
    limits: SignalThresholds,
    result: BehavioralSignals,
) -> None:
    """Report the dispatch failure rate and any single turn with many failures."""
    if not records:
        return
    failed = [record for record in records if record.failed]
    result.counters["dispatches"] = len(records)
    result.counters["failed_dispatches"] = len(failed)

    rate = len(failed) / len(records)
    result.counters["dispatch_failure_rate"] = round(rate, 3)
    if len(records) >= limits.min_dispatches_for_rate and rate >= limits.dispatch_failure_rate:
        top = Counter(record.target_id or "unknown" for record in failed)
        names = ", ".join(f"{name}×{count}" for name, count in top.most_common(3))
        result.signals.append(
            f"{len(failed)} of {len(records)} dispatches in the window failed "
            f"({rate:.0%}); most frequent: {names}"
        )

    per_turn = Counter(record.turn_index for record in failed if record.turn_index is not None)
    worst = per_turn.most_common(1)
    if worst and worst[0][1] >= limits.failed_dispatches_in_turn:
        result.signals.append(f"turn {worst[0][0]} recorded {worst[0][1]} failed dispatches")


def _check_receipts(
    entries: list[dict[str, Any]],
    dispatches: list[DispatchRecord],
    result: BehavioralSignals,
) -> None:
    """Report turns that dispatched work but recorded no effect receipts.

    A dispatch with no receipt is the shape of a completion-contract gap: the
    call ran, the ledger did not see it. It is also the shape of a hallucinated
    success being *unblocked*, so it is worth naming even once.
    """
    if not dispatches:
        return
    dispatched_turns = {record.turn_index for record in dispatches}
    silent = [
        turn
        for entry in entries
        if isinstance(turn := entry.get("turn"), int)
        and turn in dispatched_turns
        and _receipt_count(entry) == 0
    ]
    if not silent:
        return
    result.counters["turns_without_receipts"] = silent[-5:]
    result.signals.append(
        f"{len(silent)} turn(s) dispatched work but recorded no effect receipts "
        f"(most recent: {', '.join(str(turn) for turn in silent[-3:])})"
    )


def _check_mode_dwell(
    entries: list[dict[str, Any]],
    limits: SignalThresholds,
    result: BehavioralSignals,
) -> None:
    """Report a long trailing run in one mode."""
    mode_id, dwell = _mode_dwell(entries)
    result.counters["mode_dwell"] = {"mode_id": mode_id, "turns": dwell}
    if dwell >= limits.mode_dwell:
        result.signals.append(
            f"{dwell} consecutive turns in mode '{mode_id}' — either the work "
            "genuinely needs it, or the mode is stuck"
        )


def _check_cost_spike(
    token_records: list[dict[str, Any]],
    window: set[int],
    limits: SignalThresholds,
    result: BehavioralSignals,
) -> None:
    """Report a turn whose token total dwarfs the window median."""
    per_turn_tokens: dict[int, int] = {}
    for record in token_records:
        turn = record.get("turn_index")
        if isinstance(turn, int) and turn in window:
            per_turn_tokens[turn] = per_turn_tokens.get(turn, 0) + int(
                record.get("total_tokens") or 0
            )
    if not per_turn_tokens:
        return
    totals = list(per_turn_tokens.values())
    median = _median(totals)
    peak_turn, peak = max(per_turn_tokens.items(), key=lambda item: item[1])
    result.counters["token_turns"] = len(totals)
    result.counters["token_median"] = int(median)
    result.counters["token_peak"] = {"turn": peak_turn, "tokens": peak}
    if median > 0 and peak >= median * limits.cost_spike_multiple:
        result.signals.append(
            f"turn {peak_turn} used {peak} tokens — {peak / median:.1f}× the "
            f"window median of {int(median)}"
        )


def _recent_dispatches(
    decisions_path: Any,
    scan_bytes: int,
    scan_records: int,
    window: set[int],
    user_slug: str | None,
) -> list[DispatchRecord]:
    """Return the recent dispatch records for the window's turns, oldest first."""
    if decisions_path is None:
        return []
    tail = read_jsonl_tail(decisions_path, max_bytes=scan_bytes, backup_count=None)
    raw = [
        record
        for record in filter_owned(tail.records, user_slug).records
        if record.get("topic") == "tool.telemetry" and record.get("turn_index") in window
    ]
    return [dispatch_from_record(record) for record in raw[-scan_records:]]


def _recent_token_totals(
    token_usage_path: Any, scan_bytes: int, user_slug: str | None
) -> list[dict[str, Any]]:
    """Return this user's recent token-usage payloads within the byte budget."""
    if token_usage_path is None:
        return []
    tail = read_jsonl_tail(token_usage_path, max_bytes=scan_bytes)
    return filter_owned(tail.records, user_slug).records


def render_signals(signals: BehavioralSignals) -> str:
    """Render the findings as a compact prompt block (empty when nominal)."""
    if signals.is_empty:
        return ""
    lines = [f"## Observed patterns (last {signals.window_turns} turns)"]
    lines.extend(f"- {finding}" for finding in signals.signals)
    lines.append(
        "Counted from recorded evidence. Verify with `thread_read_range` before "
        "treating any of this as a current problem."
    )
    return "\n".join(lines)
