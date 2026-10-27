"""Tests for the behavioural-signals slice lane.

The self-model is gated on ``operational_health``: healthy means inject
nothing. Behavioural signals are gated on their *own* condition, because a
streak of blocked turns on a perfectly healthy process is exactly the case an
operator wants surfaced. These tests pin that independence, and the
ambient-silence default that keeps an ordinary turn free of prompt tokens.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from runtime.turn_hooks_builder import _run_behavioral_signals_slice_impl
from sdk.thread._backend import _segment_paths


def _write_entries(tmp_path: Path, entries: list[dict[str, Any]]) -> None:
    paths = _segment_paths(tmp_path / "thread" / "user")
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def _entry(turn: int, *, mode_id: str = "m", blocked: bool = False) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode_id": mode_id,
        "query": f"q{turn}",
        "response_text": f"r{turn}",
        "turn": turn,
        "effect_receipts": [{"capability_id": "read_file", "effect": "read"}],
    }
    if blocked:
        entry["provenance"] = {"source": "observed", "block_origin": "task_completion"}
    return entry


def _runtime(tmp_path: Path, *, thread_path: Path | None = None) -> Any:
    from runtime.testing import build_test_runtime

    # ``HostPaths`` is frozen, so the store location is set through config
    # rather than by assigning onto the runtime after construction.
    return build_test_runtime(
        tmp_path,
        user_slug="user",
        thread_dir=str(thread_path or (tmp_path / "thread")),
    )


def _items(runtime: Any) -> list[dict[str, Any]]:
    snapshot = runtime.session.state_snapshot or {}
    return [
        item
        for item in snapshot.get("context_items", [])
        if isinstance(item, dict) and item.get("namespace") == "behavioural_signals"
    ]


def test_nominal_window_emits_nothing(tmp_path: Path) -> None:
    """Silence is the default: an ordinary turn costs no prompt tokens."""
    _write_entries(tmp_path, [_entry(turn) for turn in range(5)])
    runtime = _runtime(tmp_path)

    work, last_item, note = _run_behavioral_signals_slice_impl(runtime)

    assert (work, last_item, note) == (0, "", "")
    assert _items(runtime) == []


def test_blocked_streak_emits_an_independent_item(tmp_path: Path) -> None:
    """A blocked streak surfaces even though the process is healthy.

    This is the whole point of the lane: ``operational_health`` answers "is the
    host degraded?", and a streak of contract failures on a healthy process
    would be silenced by reusing that gate.
    """
    _write_entries(
        tmp_path,
        [_entry(turn) for turn in range(3)] + [_entry(turn, blocked=True) for turn in range(3, 6)],
    )
    runtime = _runtime(tmp_path)

    work, last_item, note = _run_behavioral_signals_slice_impl(runtime)

    assert work == 1
    assert last_item == "behavioural_signals"
    assert "count=" in note
    items = _items(runtime)
    assert len(items) == 1
    assert "consecutive turns ended blocked" in items[0]["body"]


def test_emitted_item_is_fresh_on_each_run(tmp_path: Path) -> None:
    """Re-running replaces the prior observation rather than accumulating."""
    _write_entries(tmp_path, [_entry(turn, blocked=True) for turn in range(4)])
    runtime = _runtime(tmp_path)

    _run_behavioral_signals_slice_impl(runtime)
    _run_behavioral_signals_slice_impl(runtime)

    assert len(_items(runtime)) == 1


def test_item_is_marked_system_observed(tmp_path: Path) -> None:
    """The lane is machine-measured, and says so in its provenance."""
    _write_entries(tmp_path, [_entry(turn, blocked=True) for turn in range(4)])
    runtime = _runtime(tmp_path)

    _run_behavioral_signals_slice_impl(runtime)

    item = _items(runtime)[0]
    assert item["metadata"]["provenance"] == "system_observed"
    assert item["metadata"]["window_turns"] == 4
    assert "blocked_streak" in item["metadata"]["counters"]


@pytest.mark.parametrize("breakage", ["missing_thread", "corrupt_store"])
def test_scan_failure_never_breaks_the_slice(tmp_path: Path, breakage: str) -> None:
    """A failed scan is no signal, never an exception into the turn loop."""
    if breakage == "missing_thread":
        runtime = _runtime(tmp_path, thread_path=tmp_path / "does-not-exist")
    else:
        runtime = _runtime(tmp_path)
        paths = _segment_paths(tmp_path / "thread" / "user")
        paths.current.parent.mkdir(parents=True, exist_ok=True)
        paths.current.write_text("{not json\n", encoding="utf-8")

    work, last_item, note = _run_behavioral_signals_slice_impl(runtime)

    assert work == 0
    assert last_item == ""
    assert note == ""


def test_item_carries_verification_pointer(tmp_path: Path) -> None:
    """The injected block tells the model how to check the claim."""
    _write_entries(tmp_path, [_entry(turn, blocked=True) for turn in range(4)])
    runtime = _runtime(tmp_path)

    _run_behavioral_signals_slice_impl(runtime)

    assert "thread_read_range" in _items(runtime)[0]["body"]
