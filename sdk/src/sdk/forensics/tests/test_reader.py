"""Tests for :class:`sdk.forensics.ForensicReader` against real files.

These exercise the composition end-to-end: real JSONL on disk, a real
segmented thread store, real redaction. The contracts under test are the ones
that would make a forensic read dangerous if broken — path derivation,
degradation reporting, caps, and redaction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sdk.forensics import ForensicReader, render_bundle
from sdk.thread import ThreadRedactionPolicy
from sdk.thread._backend import _segment_paths


def _write_thread_entry(thread: Path, entry: dict) -> None:
    paths = _segment_paths(thread)
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _seed(tmp_path: Path, *, user: str = "u", turns: int = 3) -> tuple[Path, Path]:
    """Create a data root with a thread store and telemetry for ``turns`` turns."""
    data_dir = tmp_path / "data"
    thread_dir = data_dir / "data" / "threads"
    thread = thread_dir / user
    now = datetime.now(UTC).isoformat()

    for turn in range(turns):
        _write_thread_entry(
            thread,
            {
                "timestamp_utc": now,
                "mode_id": "coding_execute",
                "query": f"query {turn}",
                "response_text": f"response {turn}",
                "turn": turn,
                "provenance": {"source": "observed", "task_id": "t-1"},
            },
        )

    ledger = data_dir / "logs" / "decisions.jsonl"
    _write_jsonl(
        ledger,
        [
            record
            for turn in range(turns)
            for record in (
                {
                    "topic": "turn.decision",
                    "turn_index": turn,
                    "action_type": "ContinueAction",
                    "iteration": 1,
                },
                {
                    "topic": "tool.telemetry",
                    "turn_index": turn,
                    "target_id": "read_file",
                    "target_type": "tool",
                    "status": "success",
                },
            )
        ],
    )
    return data_dir, thread_dir


def test_reader_reads_the_latest_turn_by_default(tmp_path: Path) -> None:
    """``turns="last"`` selects the most recent recorded turn."""
    data_dir, thread_dir = _seed(tmp_path, turns=3)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u")

    bundle = reader.bundle()

    assert [turn.turn_index for turn in bundle.turns] == [2]
    assert bundle.turns[0].decisions[0].action_type == "ContinueAction"
    assert bundle.turns[0].dispatches[0].target_id == "read_file"
    assert bundle.sources.thread_store
    assert bundle.sources.decisions_log


def test_reader_parses_an_explicit_range(tmp_path: Path) -> None:
    """``turns="0-1"`` reads an inclusive range."""
    data_dir, thread_dir = _seed(tmp_path, turns=3)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u")

    bundle = reader.bundle(turns="0-1")

    assert [turn.turn_index for turn in bundle.turns] == [0, 1]


def test_reader_reports_missing_logs_rather_than_empty_ones(tmp_path: Path) -> None:
    """An absent ledger yields notes, because 'unknown' is not 'clean'."""
    thread_dir = tmp_path / "threads"
    _write_thread_entry(thread_dir / "u", {"mode_id": "default", "query": "q", "turn": 0})

    reader = ForensicReader(tmp_path, thread_dir=thread_dir, user_slug="u")
    bundle = reader.bundle()

    assert not bundle.sources.decisions_log
    assert any("decisions.jsonl" in note for note in bundle.notes)
    assert "unavailable" in render_bundle(bundle)


def test_reader_scopes_to_the_injected_user_slug(tmp_path: Path) -> None:
    """The store is selected by constructor state, never by a tool argument."""
    data_dir, thread_dir = _seed(tmp_path, user="alice", turns=1)
    _write_thread_entry(
        thread_dir / "bob", {"mode_id": "default", "query": "bob secret", "turn": 0}
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="alice").bundle()

    rendered = render_bundle(bundle)
    assert "query 0" in rendered
    assert "bob secret" not in rendered


def test_reader_applies_redaction_to_thread_text(tmp_path: Path) -> None:
    """Threads persist raw text; a forensic read must honor the policy."""
    thread_dir = tmp_path / "threads"
    _write_thread_entry(
        thread_dir / "u",
        {
            "mode_id": "default",
            "query": "why",
            "response_text": "I've hit my iteration budget in explainer (cap 12)",
            "turn": 0,
        },
    )
    policy = ThreadRedactionPolicy.from_patterns((r"I've hit my iteration budget[^\n]*",))
    reader = ForensicReader(tmp_path, thread_dir=thread_dir, user_slug="u", redaction=policy)

    rendered = render_bundle(reader.bundle())

    assert "hit my iteration budget" not in rendered


def test_reader_caps_turns_and_says_so(tmp_path: Path) -> None:
    """A cap that silently truncates would hide the reason for the read."""
    data_dir, thread_dir = _seed(tmp_path, turns=6)
    reader = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u", max_turns=2)

    bundle = reader.bundle(turns="0-5")

    assert len(bundle.turns) == 2
    assert any("read cap is 2" in note for note in bundle.notes)
    # The most recent turns are kept, not the oldest.
    assert [turn.turn_index for turn in bundle.turns] == [4, 5]


def test_reader_surfaces_log_errors_in_the_window(tmp_path: Path) -> None:
    """Warning/error lines accompany the bundle as evidence."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    now = datetime.now(UTC).isoformat()
    _write_jsonl(
        data_dir / "logs" / "runtime.jsonl",
        [
            {"ts": now, "level": "info", "msg": "noise"},
            {"ts": now, "level": "warning", "msg": "pack outdated"},
        ],
    )

    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle()

    assert [record["msg"] for record in bundle.log_errors] == ["pack outdated"]
    assert bundle.sources.runtime_log


def test_reader_handles_an_empty_data_root(tmp_path: Path) -> None:
    """Nothing to read is a stated empty answer, not a crash."""
    bundle = ForensicReader(tmp_path, user_slug="u").bundle()

    assert bundle.turns == []
    assert bundle.notes
    assert "No turn entries matched" in render_bundle(bundle)


@pytest.mark.parametrize("selection", ["nonsense", "5-1", "-2", "x-y"])
def test_reader_rejects_invalid_selections_with_a_note(tmp_path: Path, selection: str) -> None:
    """A bad range is reported; it never silently reads something else."""
    data_dir, thread_dir = _seed(tmp_path, turns=2)
    bundle = ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle(turns=selection)
    assert bundle.turns == []
    assert any("range" in note or "turn" in note for note in bundle.notes)


def test_reader_render_is_bounded_for_a_real_turn(tmp_path: Path) -> None:
    """A single busy turn stays a summary rather than a thread dump."""
    data_dir, thread_dir = _seed(tmp_path, turns=1)
    _write_jsonl(
        data_dir / "logs" / "decisions.jsonl",
        [
            {
                "topic": "turn.decision",
                "turn_index": 0,
                "action_type": "ContinueAction",
                "iteration": n,
            }
            for n in range(1, 41)
        ]
        + [
            {
                "topic": "tool.telemetry",
                "turn_index": 0,
                "target_id": "read_file",
                "target_type": "tool",
                "status": "success",
            }
            for _ in range(40)
        ],
    )

    rendered = render_bundle(
        ForensicReader(data_dir, thread_dir=thread_dir, user_slug="u").bundle()
    )

    assert len(rendered) < 6000
    assert "elided" in rendered
