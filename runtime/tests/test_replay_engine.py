"""Tests for the exact-mode replay engine core (Spec 3 slice 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime.replay import (
    ReplayDivergence,
    ReplayReport,
    ReplayToolGapError,
    TranscriptToolAnswerer,
)
from runtime.replay._engine import (
    _compare_receipts,
    _newest_scratch_entry,
    _replay_divergences_from_gaps,
    _replay_divergences_from_provider,
)
from runtime.testing.archive import ArchiveGatewayProvider

# ── TranscriptToolAnswerer ────────────────────────────────────────────────


def test_answerer_returns_recorded_output_for_known_id() -> None:
    answerer = TranscriptToolAnswerer(tool_results={"call_recorded_1": "ok"})
    assert answerer.answer("t", {"x": 1}, "call_recorded_1") == "ok"
    assert answerer.gaps == []


def test_answerer_records_gap_and_raises_for_unknown_id() -> None:
    answerer = TranscriptToolAnswerer(tool_results={})
    with pytest.raises(ReplayToolGapError, match="call_recorded_1"):
        answerer.answer("t", {"x": 1}, "call_recorded_1")
    assert answerer.gaps == [{"tool": "t", "tool_call_id": "call_recorded_1"}]


# ── Divergence projection ────────────────────────────────────────────────


def test_provider_divergences_project_onto_the_report() -> None:
    provider = ArchiveGatewayProvider(calls=[])
    provider.divergences.append(
        {"call_id": "call-1", "field": "model_id", "recorded": "a", "replayed": "b"}
    )
    divergences = _replay_divergences_from_provider(provider)
    assert len(divergences) == 1
    assert divergences[0].kind == "model_request"
    assert divergences[0].field == "model_id"
    assert divergences[0].call_id == "call-1"
    assert divergences[0].recorded == "a"
    assert divergences[0].replayed == "b"


def test_gap_divergences_project_onto_the_report() -> None:
    answerer = TranscriptToolAnswerer(tool_results={})
    answerer.gaps.append({"tool": "t", "tool_call_id": "call_recorded_1"})
    divergences = _replay_divergences_from_gaps(answerer)
    assert len(divergences) == 1
    assert divergences[0].kind == "tool_result"
    assert divergences[0].tool_call_id == "call_recorded_1"
    assert divergences[0].detail == "t"


# ── Receipts comparison ──────────────────────────────────────────────────


def test_receipts_match_structurally() -> None:
    recorded = [{"effect": "read", "capability_id": "t", "artifact_id": "", "mode_id": "default"}]
    replayed = [{"effect": "read", "capability_id": "t", "artifact_id": "", "mode_id": "replay"}]
    assert _compare_receipts(recorded, replayed) is True


def test_receipts_mismatch_on_capability_or_effect() -> None:
    recorded = [{"effect": "read", "capability_id": "t", "artifact_id": ""}]
    replayed = [{"effect": "write", "capability_id": "t", "artifact_id": ""}]
    assert _compare_receipts(recorded, replayed) is False
    replayed2 = [{"effect": "read", "capability_id": "other", "artifact_id": ""}]
    assert _compare_receipts(recorded, replayed2) is False


def test_receipts_mismatch_on_length() -> None:
    recorded = [{"effect": "read", "capability_id": "t", "artifact_id": ""}]
    assert _compare_receipts(recorded, []) is False


# ── Scratch entry discovery ──────────────────────────────────────────────


def test_newest_scratch_entry_reads_the_last_line(tmp_path: Path) -> None:
    threads = tmp_path / "threads" / "user"
    threads.mkdir(parents=True)
    active = threads / "thread.current.jsonl"
    active.write_text(
        json.dumps({"turn": 0, "text": "first"})
        + "\n"
        + json.dumps({"turn": 1, "text": "second"})
        + "\n",
        encoding="utf-8",
    )
    entry = _newest_scratch_entry(tmp_path)
    assert entry is not None
    assert entry["turn"] == 1
    assert entry["text"] == "second"


def test_newest_scratch_entry_returns_none_without_a_store(tmp_path: Path) -> None:
    assert _newest_scratch_entry(tmp_path) is None


# ── ReplayReport.ok ──────────────────────────────────────────────────────


def test_report_ok_requires_no_error_no_divergences_and_matches() -> None:
    report = ReplayReport(turn_index=0, turn_id="turn-abc")
    assert report.ok is True
    report.error = "boom"
    assert report.ok is False
    report.error = ""
    report.divergences.append(ReplayDivergence(kind="response"))
    assert report.ok is False
    report.divergences.clear()
    report.response_match = False
    assert report.ok is False
    report.response_match = True
    report.receipts_match = False
    assert report.ok is False


# ── replay_turn error paths (no runtime needed) ──────────────────────────


class _Store:
    """Minimal ThreadStore double: ``load_turns`` + ``base_path``."""

    def __init__(self, base: Path, entries: dict[int, dict[str, Any]]) -> None:
        self.base_path = base
        self._entries = entries

    def load_turns(self, turns: Any) -> dict[int, dict[str, Any]]:
        return {t: self._entries[t] for t in turns if t in self._entries}


@pytest.mark.asyncio
async def test_replay_turn_missing_turn_sets_report_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing recorded turn surfaces as the report's error, not a raise."""
    import runtime.replay._engine as engine

    monkeypatch.setattr(engine, "ThreadStore", lambda _path: _Store(tmp_path, {}))
    from runtime.replay import replay_turn

    report = await replay_turn(
        source_thread_path=tmp_path / "thread.jsonl",
        turn_index=7,
        scratch_data_dir=tmp_path / "scratch",
    )
    assert report.error == "no recorded turn at index 7"
    assert report.ok is False
