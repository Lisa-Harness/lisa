"""Tests for behavioural signal computation.

Each signal is arithmetic over recorded evidence, so the tests pin both the
arithmetic and the threshold: a signal must fire on the pattern it names and
stay silent otherwise. Silence matters as much as detection — an ambient
channel that speaks on ordinary turns is one the model learns to ignore.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sdk.forensics import (
    BehavioralSignals,
    SignalThresholds,
    compute_behavioral_signals,
    render_signals,
)
from sdk.forensics.signals import DEFAULT_SCAN_BYTES
from sdk.thread import ThreadStore
from sdk.thread._backend import _segment_paths


def _store(tmp_path: Path) -> ThreadStore:
    return ThreadStore(tmp_path / "thread")


def _write_entries(
    tmp_path: Path,
    *,
    count: int,
    mode_id: str = "coding_execute",
    blocked_origins: list[str] | None = None,
    receipts: bool = True,
) -> ThreadStore:
    """Write ``count`` thread entries; ``blocked_origins`` marks trailing blocks."""
    paths = _segment_paths(tmp_path / "thread")
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    blocked_origins = blocked_origins or []
    now = datetime.now(UTC).isoformat()
    with paths.current.open("w", encoding="utf-8") as handle:
        for turn in range(count):
            blocked_from = count - len(blocked_origins)
            entry: dict[str, object] = {
                "timestamp_utc": now,
                "mode_id": mode_id,
                "query": f"q{turn}",
                "response_text": f"r{turn}",
                "turn": turn,
            }
            if receipts:
                entry["effect_receipts"] = [{"capability_id": "read_file", "effect": "read"}]
            if turn >= blocked_from:
                entry["provenance"] = {
                    "source": "observed",
                    "block_origin": blocked_origins[turn - blocked_from],
                }
            handle.write(json.dumps(entry) + "\n")
    return ThreadStore(tmp_path / "thread")


def _ledger(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "decisions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _tokens(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "token-usage.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _dispatches(*, turn: int, count: int, status: str = "success") -> list[dict]:
    return [
        {
            "topic": "tool.telemetry",
            "turn_index": turn,
            "target_id": "read_file",
            "target_type": "tool",
            "status": status,
        }
        for _ in range(count)
    ]


# ── Silence ───────────────────────────────────────────────────────────


def test_nominal_window_is_silent(tmp_path: Path) -> None:
    """An ordinary window produces no signals and therefore no prompt tokens."""
    store = _write_entries(tmp_path, count=5)
    signals = compute_behavioral_signals(store)

    assert isinstance(signals, BehavioralSignals)
    assert signals.is_empty
    assert render_signals(signals) == ""
    assert signals.window_turns == 5


def test_empty_store_is_silent(tmp_path: Path) -> None:
    """No turns recorded is not a finding."""
    signals = compute_behavioral_signals(_store(tmp_path))
    assert signals.is_empty
    assert signals.window_turns == 0


# ── Blocked streak ─────────────────────────────────────────────────────


def test_blocked_streak_fires_at_threshold(tmp_path: Path) -> None:
    """Three consecutive blocked turns is a structural failure, not a one-off."""
    store = _write_entries(
        tmp_path, count=6, blocked_origins=["task_completion", "task_completion", "task_completion"]
    )
    signals = compute_behavioral_signals(store)

    assert signals.counters["blocked_streak"] == 3
    assert any("3 consecutive turns ended blocked" in s for s in signals.signals)
    assert any("task_completion" in s for s in signals.signals)


def test_blocked_streak_below_threshold_is_silent(tmp_path: Path) -> None:
    """Two blocked turns is not yet a streak."""
    store = _write_entries(
        tmp_path, count=6, blocked_origins=["task_completion", "task_completion"]
    )
    signals = compute_behavioral_signals(store)

    assert signals.counters["blocked_streak"] == 2
    assert not any("consecutive turns ended blocked" in s for s in signals.signals)


def test_blocked_streak_counts_only_the_trailing_run(tmp_path: Path) -> None:
    """An older block broken by a clean turn is not a current streak."""
    paths = _segment_paths(tmp_path / "thread")
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    with paths.current.open("w", encoding="utf-8") as handle:
        for turn, blocked in enumerate([True, True, True, False]):
            entry: dict[str, object] = {
                "timestamp_utc": now,
                "mode_id": "m",
                "query": "q",
                "response_text": "r",
                "turn": turn,
            }
            if blocked:
                entry["provenance"] = {"block_origin": "task_completion"}
            handle.write(json.dumps(entry) + "\n")

    signals = compute_behavioral_signals(ThreadStore(tmp_path / "thread"))

    assert signals.counters["blocked_streak"] == 0
    assert signals.counters["blocked_turns"] == 3


def test_blocked_streak_threshold_is_configurable(tmp_path: Path) -> None:
    """A host can tune when a streak is worth surfacing."""
    store = _write_entries(tmp_path, count=6, blocked_origins=["guard", "guard"])
    strict = compute_behavioral_signals(store, thresholds=SignalThresholds(blocked_streak=2))

    assert any("2 consecutive turns ended blocked" in s for s in strict.signals)


# ── Dispatch failures ──────────────────────────────────────────────────


def test_dispatch_failure_rate_fires_above_threshold(tmp_path: Path) -> None:
    """A quarter of dispatches failing is a rate worth naming."""
    store = _write_entries(tmp_path, count=3)
    ledger = _ledger(
        tmp_path,
        _dispatches(turn=1, count=8, status="success")
        + _dispatches(turn=2, count=4, status="error"),
    )

    signals = compute_behavioral_signals(store, decisions_path=ledger)

    assert signals.counters["dispatches"] == 12
    assert signals.counters["failed_dispatches"] == 4
    assert any("dispatches in the window failed" in s for s in signals.signals)


def test_dispatch_failure_rate_needs_a_minimum_sample(tmp_path: Path) -> None:
    """Two failures out of three dispatches is not a rate."""
    store = _write_entries(tmp_path, count=2)
    ledger = _ledger(
        tmp_path,
        _dispatches(turn=1, count=1, status="success")
        + _dispatches(turn=1, count=2, status="error"),
    )

    signals = compute_behavioral_signals(store, decisions_path=ledger)

    assert not any("dispatches in the window failed" in s for s in signals.signals)


def test_concentrated_failures_in_one_turn_are_named(tmp_path: Path) -> None:
    """Three failures in a single turn is reported even below the rate."""
    store = _write_entries(tmp_path, count=1)
    ledger = _ledger(tmp_path, _dispatches(turn=0, count=3, status="timeout"))

    signals = compute_behavioral_signals(store, decisions_path=ledger)

    assert any("turn 0 recorded 3 failed dispatches" in s for s in signals.signals)


# ── Receipts vs dispatches ─────────────────────────────────────────────


def test_dispatch_without_receipt_is_named(tmp_path: Path) -> None:
    """A call that ran but left no receipt is a contract gap, not a clean turn."""
    store = _write_entries(tmp_path, count=1, receipts=False)
    ledger = _ledger(tmp_path, _dispatches(turn=0, count=1))

    signals = compute_behavioral_signals(store, decisions_path=ledger)

    assert signals.counters["turns_without_receipts"] == [0]
    assert any("recorded no effect receipts" in s for s in signals.signals)


def test_receipts_present_is_silent(tmp_path: Path) -> None:
    """A turn with receipts and dispatches is a healthy contract."""
    store = _write_entries(tmp_path, count=1)
    ledger = _ledger(tmp_path, _dispatches(turn=0, count=1))

    signals = compute_behavioral_signals(store, decisions_path=ledger)

    assert "turns_without_receipts" not in signals.counters


# ── Mode dwell ─────────────────────────────────────────────────────────


def test_mode_dwell_fires_at_threshold(tmp_path: Path) -> None:
    """A long run in one mode may mean the work needs it, or that it is stuck."""
    store = _write_entries(tmp_path, count=9, mode_id="coding_design")

    signals = compute_behavioral_signals(store)

    assert signals.counters["mode_dwell"] == {"mode_id": "coding_design", "turns": 9}
    assert any("consecutive turns in mode 'coding_design'" in s for s in signals.signals)


def test_mode_dwell_is_silent_when_mode_changes(tmp_path: Path) -> None:
    """Alternating modes do not accumulate dwell."""
    paths = _segment_paths(tmp_path / "thread")
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    with paths.current.open("w", encoding="utf-8") as handle:
        for turn in range(10):
            handle.write(
                json.dumps(
                    {
                        "timestamp_utc": now,
                        "mode_id": f"m{turn % 2}",
                        "query": "q",
                        "response_text": "r",
                        "turn": turn,
                    }
                )
                + "\n"
            )

    signals = compute_behavioral_signals(ThreadStore(tmp_path / "thread"))

    assert signals.counters["mode_dwell"]["turns"] == 1
    assert not any("consecutive turns in mode" in s for s in signals.signals)


# ── Cost spike ─────────────────────────────────────────────────────────


def test_cost_spike_fires_against_the_window_median(tmp_path: Path) -> None:
    """A turn several times the median is worth naming."""
    store = _write_entries(tmp_path, count=4)
    tokens = _tokens(
        tmp_path,
        [
            {"turn_index": 0, "total_tokens": 100},
            {"turn_index": 1, "total_tokens": 100},
            {"turn_index": 2, "total_tokens": 100},
            {"turn_index": 3, "total_tokens": 5000},
        ],
    )

    signals = compute_behavioral_signals(store, token_usage_path=tokens)

    assert signals.counters["token_peak"] == {"turn": 3, "tokens": 5000}
    assert any("turn 3 used 5000 tokens" in s for s in signals.signals)


def test_uniform_cost_is_silent(tmp_path: Path) -> None:
    """Identical turns produce no spike."""
    store = _write_entries(tmp_path, count=3)
    tokens = _tokens(
        tmp_path,
        [{"turn_index": turn, "total_tokens": 100} for turn in range(3)],
    )

    signals = compute_behavioral_signals(store, token_usage_path=tokens)

    assert not any("tokens —" in s for s in signals.signals)


# ── Contract ───────────────────────────────────────────────────────────


def test_counters_are_populated_even_when_silent(tmp_path: Path) -> None:
    """Counters are always available so a caller can log without re-reading."""
    store = _write_entries(tmp_path, count=4)
    signals = compute_behavioral_signals(store)

    assert signals.is_empty
    assert signals.counters["blocked_streak"] == 0
    assert signals.counters["mode_dwell"]["turns"] == 4


def test_scan_is_bounded_by_the_byte_budget(tmp_path: Path) -> None:
    """A slice-time scan must not read an unbounded ledger."""
    store = _write_entries(tmp_path, count=1)
    ledger = _ledger(tmp_path, _dispatches(turn=0, count=5000))

    wide = compute_behavioral_signals(store, decisions_path=ledger, scan_bytes=10_000_000)
    narrow = compute_behavioral_signals(store, decisions_path=ledger, scan_bytes=4096)

    assert wide.counters["dispatches"] > narrow.counters.get("dispatches", 0)
    assert narrow.counters["dispatches"] < 5000


def test_missing_ledger_does_not_raise(tmp_path: Path) -> None:
    """A host without a ledger still gets thread-derived signals."""
    store = _write_entries(tmp_path, count=5, blocked_origins=["guard"] * 4)

    signals = compute_behavioral_signals(store, decisions_path=tmp_path / "absent.jsonl")

    assert any("consecutive turns ended blocked" in s for s in signals.signals)


def test_rendered_block_states_how_to_verify(tmp_path: Path) -> None:
    """The prompt block always points back to the primary record."""
    store = _write_entries(tmp_path, count=4, blocked_origins=["guard", "guard", "guard"])
    rendered = render_signals(compute_behavioral_signals(store))

    assert rendered.startswith("## Observed patterns")
    assert "thread_read_range" in rendered


def test_default_scan_budget_stays_small() -> None:
    """The slice-time budget must remain slice-sized, not forensic-sized."""
    assert DEFAULT_SCAN_BYTES <= 512 * 1024


def test_ledger_rows_outside_the_turn_window_are_ignored(tmp_path: Path) -> None:
    """Dispatches and cost from turns outside the examined window do not count."""
    store = _write_entries(tmp_path, count=2)
    ledger = _ledger(
        tmp_path,
        _dispatches(turn=1, count=1) + _dispatches(turn=99, count=20, status="error"),
    )
    tokens = _tokens(
        tmp_path,
        [
            {"turn_index": 0, "total_tokens": 100},
            {"turn_index": 1, "total_tokens": 100},
            {"turn_index": 99, "total_tokens": 50_000},
        ],
    )

    signals = compute_behavioral_signals(store, decisions_path=ledger, token_usage_path=tokens)

    assert signals.counters["dispatches"] == 1
    assert signals.counters["failed_dispatches"] == 0
    assert signals.counters["token_peak"] == {"turn": 0, "tokens": 100}
    assert signals.is_empty


def test_dispatches_in_sealed_ledger_segments_are_read(tmp_path: Path) -> None:
    """Sealed ``decisions.jsonl.<n>`` segments are part of the scanned ledger."""
    store = _write_entries(tmp_path, count=1)
    ledger = _ledger(tmp_path, [])
    sealed = tmp_path / "decisions.jsonl.5"
    sealed.write_text(
        "".join(json.dumps(r) + "\n" for r in _dispatches(turn=0, count=3, status="error")),
        encoding="utf-8",
    )

    signals = compute_behavioral_signals(store, decisions_path=ledger)

    assert signals.counters["failed_dispatches"] == 3


def test_ledgers_are_scoped_to_the_thread_owner(tmp_path: Path) -> None:
    """Another user's failures and spend on colliding turn indices are ignored.

    Unowned (pre-attribution) rows are excluded too: with several users on a
    root they cannot be assigned to this thread.
    """
    store = _write_entries(tmp_path, count=4)
    alice = [{**row, "user_slug": "alice"} for row in _dispatches(turn=1, count=8)]
    bob = [{**row, "user_slug": "bob"} for row in _dispatches(turn=2, count=8, status="error")]
    legacy = _dispatches(turn=3, count=8, status="error")
    ledger = _ledger(tmp_path, alice + bob + legacy)
    tokens = _tokens(
        tmp_path,
        [{"user_slug": "alice", "turn_index": turn, "total_tokens": 100} for turn in range(4)]
        + [{"user_slug": "bob", "turn_index": 3, "total_tokens": 50000}],
    )

    signals = compute_behavioral_signals(
        store, decisions_path=ledger, token_usage_path=tokens, user_slug="alice"
    )

    assert signals.counters["dispatches"] == 8
    assert signals.counters["failed_dispatches"] == 0
    assert signals.counters["token_peak"] == {"turn": 0, "tokens": 100}
    assert signals.is_empty
