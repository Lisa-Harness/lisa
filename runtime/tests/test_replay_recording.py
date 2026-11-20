"""Tests for the replay recording loader (Spec 3 slice 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from runtime.replay import RecordedTurn, ReplayError, load_recorded_turn
from runtime.testing.archive import archive_loader


def _persist(base: Path, payload: dict[str, Any]) -> str:
    """Persist one snapshot through the production result-object store."""
    from runtime.thread._results import _persist_result

    return _persist_result(base, {"content": json.dumps(payload, ensure_ascii=False)})


def _snapshot_store(base: Path) -> dict[str, str]:
    """Persist the slice-2 test fixtures through the real store, return refs."""
    refs: dict[str, str] = {}
    refs["request_1"] = _persist(
        base,
        {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
    )
    refs["response_1"] = _persist(
        base,
        {
            "call_id": "call-1",
            "attempt": "primary",
            "iteration": 1,
            "outcome": "ok",
            "content": "first",
            "provider": "ollama",
            "model_id": "qwen3:8b",
            "provider_request_id": "req-1",
            "tool_calls": [
                {
                    "id": "call_recorded_1",
                    "type": "function",
                    "function": {"name": "t", "arguments": '{"x": 1}'},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "reasoning_content": "thinking",
            "finish_reason": "tool_calls",
            "truncated": False,
            "duration_ms": 12.5,
        },
    )
    refs["request_2"] = _persist(
        base,
        {
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "call_recorded_1", "content": "ok"},
            ],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
    )
    refs["response_2"] = _persist(
        base,
        {
            "call_id": "call-2",
            "attempt": "primary",
            "iteration": 2,
            "outcome": "ok",
            "content": "done",
            "provider": "ollama",
            "model_id": "qwen3:8b",
            "usage": None,
            "finish_reason": "stop",
            "truncated": None,
        },
    )
    return refs


def _entry(refs: dict[str, str], **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "turn_id": "turn-abc",
        "mode_id": "default",
        "query": "go",
        "response_text": "done",
        "turn_input": {
            "calls": [
                {
                    "call_id": "call-1",
                    "iteration": 1,
                    "attempt": "primary",
                    "ref": refs["request_1"],
                    "response_ref": refs["response_1"],
                },
                {
                    "call_id": "call-2",
                    "iteration": 2,
                    "attempt": "primary",
                    "ref": refs["request_2"],
                    "response_ref": refs["response_2"],
                },
            ]
        },
        "transcript": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_recorded_1",
                        "type": "function",
                        "function": {"name": "t", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_recorded_1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        "effect_receipts": [
            {"effect": "read", "capability_id": "t", "artifact_id": "", "mode_id": "default"}
        ],
        "run_ref": "run-1",
    }
    entry.update(overrides)
    return entry


class _Store:
    """Minimal ThreadStore double: ``load_turns`` + ``base_path``."""

    def __init__(self, base: Path, entries: dict[int, dict[str, Any]]) -> None:
        self.base_path = base
        self._entries = entries

    def load_turns(self, turns: Any) -> dict[int, dict[str, Any]]:
        return {t: self._entries[t] for t in turns if t in self._entries}


def _store(tmp_path: Path, **overrides: Any) -> _Store:
    base = tmp_path / "thread.jsonl"
    refs = _snapshot_store(base)
    return _Store(base, {0: _entry(refs, **overrides)})


def test_loads_calls_in_recorded_order_and_maps_tool_results(tmp_path: Path) -> None:
    recorded = load_recorded_turn(_store(tmp_path), 0)

    assert isinstance(recorded, RecordedTurn)
    assert recorded.turn_index == 0
    assert recorded.turn_id == "turn-abc"
    assert recorded.mode_id == "default"
    assert recorded.query == "go"
    assert recorded.response_text == "done"
    assert recorded.run_ref == "run-1"
    assert [c.call_id for c in recorded.calls] == ["call-1", "call-2"]
    assert recorded.calls[0].response["content"] == "first"
    assert recorded.calls[1].response["content"] == "done"
    assert recorded.tool_results == {"call_recorded_1": "ok"}
    assert recorded.effect_receipts[0]["capability_id"] == "t"


def test_missing_turn_raises_replay_error(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="no recorded turn at index 7"):
        load_recorded_turn(_store(tmp_path), 7)


def test_archive_gap_raises_replay_error_naming_the_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store._entries[0]["turn_input"]["calls"][0]["response_ref"] = None
    with pytest.raises(ReplayError, match="turn 0"):
        load_recorded_turn(store, 0)


def test_unloadable_ref_raises_replay_error_naming_the_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store._entries[0]["turn_input"]["calls"][0]["response_ref"] = "r" + "z" * 64
    with pytest.raises(ReplayError, match="turn 0"):
        load_recorded_turn(store, 0)


def test_list_content_tool_messages_flatten_to_text(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store._entries[0]["transcript"] = [
        {
            "role": "tool",
            "tool_call_id": "call_recorded_1",
            "content": [{"text": "part one "}, {"text": "part two"}],
        }
    ]
    recorded = load_recorded_turn(store, 0)
    assert recorded.tool_results == {"call_recorded_1": "part one part two"}


def test_archive_loader_reads_the_persisted_store(tmp_path: Path) -> None:
    """The production loader binding dereferences what the test persisted."""
    base = tmp_path / "thread.jsonl"
    refs = _snapshot_store(base)
    loaded = archive_loader(base)(refs["response_1"])
    # Production persists snapshots as {"content": json.dumps(snapshot)};
    # load_result_object returns that envelope as-is.
    assert json.loads(loaded["content"])["content"] == "first"


# ── Auxiliary-call ledger loading (FIX-1: numeric segment order) ──────────────


def _ledger_row(base: Path, call_id: str, *, user_slug: str = "local-user") -> dict[str, Any]:
    """One ``model.request`` ledger row with valid persisted refs."""
    request_ref = _persist(
        base,
        {
            "messages": [{"role": "user", "content": f"plan {call_id}"}],
            "tools": None,
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
    )
    response_ref = _persist(
        base,
        {
            "call_id": call_id,
            "outcome": "ok",
            "content": f"reply {call_id}",
            "provider": "ollama",
            "model_id": "qwen3:8b",
        },
    )
    return {
        "topic": "model.request",
        "user_slug": user_slug,
        "call_id": call_id,
        "request_ref": request_ref,
        "response_ref": response_ref,
    }


def _write_ledger_segment(logs_dir: Path, name: str, rows: list[dict[str, Any]]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (logs_dir / name).write_text(payload, encoding="utf-8")


def test_auxiliary_calls_load_in_numeric_segment_order_active_last(tmp_path: Path) -> None:
    """Sealed segments load oldest-first (numeric), then the active file.

    ``DecisionTelemetryHandler`` seals the active file to the next
    ``decisions.jsonl.<n>`` (``.1`` oldest, append-only), so the recorded
    order is numeric ascending with the active file LAST. A plain
    lexicographic sort misorders at 10+ segments (``.10`` before ``.2``)
    and puts the active file first (``decisions.jsonl`` sorts before
    ``decisions.jsonl.1``). The fixture mixes zero-padded and unpadded
    names to pin the numeric contract, and adds nonnumeric sidecars that
    must stay ignored.
    """
    from runtime.replay import load_auxiliary_calls

    base = tmp_path / "thread.jsonl"
    logs_dir = tmp_path / "data" / "logs"

    # 12 sealed segments: numeric order 1..12, with zero-padded and unpadded
    # names interleaved so a lexicographic sort cannot accidentally agree.
    names = [
        "decisions.jsonl.1",
        "decisions.jsonl.02",
        "decisions.jsonl.3",
        "decisions.jsonl.04",
        "decisions.jsonl.5",
        "decisions.jsonl.06",
        "decisions.jsonl.7",
        "decisions.jsonl.08",
        "decisions.jsonl.9",
        "decisions.jsonl.10",
        "decisions.jsonl.11",
        "decisions.jsonl.12",
    ]
    assert len(names) == 12
    for index, name in enumerate(names, start=1):
        _write_ledger_segment(logs_dir, name, [_ledger_row(base, f"call-{index:02d}")])

    # Nonnumeric sidecars: never part of the ledger's rotation set.
    _write_ledger_segment(logs_dir, "decisions.jsonl.tmp", [_ledger_row(base, "call-tmp")])
    _write_ledger_segment(logs_dir, "decisions.jsonl.bak", [_ledger_row(base, "call-bak")])

    # The active file is newest: it must load AFTER every sealed segment.
    _write_ledger_segment(logs_dir, "decisions.jsonl", [_ledger_row(base, "call-13")])

    calls = load_auxiliary_calls(
        source_data_dir=tmp_path / "data",
        thread_base_path=base,
        user_slug="local-user",
    )
    assert [c.call_id for c in calls] == [f"call-{i:02d}" for i in range(1, 14)]
    assert [c.response["content"] for c in calls] == [f"reply call-{i:02d}" for i in range(1, 14)]


def test_auxiliary_calls_skip_other_users_and_unscoped_rows(tmp_path: Path) -> None:
    """Only this thread's user's ``model.request`` rows load."""
    from runtime.replay import load_auxiliary_calls

    base = tmp_path / "thread.jsonl"
    logs_dir = tmp_path / "data" / "logs"
    _write_ledger_segment(
        logs_dir,
        "decisions.jsonl",
        [
            _ledger_row(base, "call-mine"),
            _ledger_row(base, "call-other", user_slug="someone-else"),
            {"topic": "tool.telemetry", "user_slug": "local-user"},
            {
                "topic": "model.request",
                "user_slug": "local-user",
                "call_id": "call-noref",
                "request_ref": None,
                "response_ref": None,
            },
        ],
    )
    calls = load_auxiliary_calls(
        source_data_dir=tmp_path / "data",
        thread_base_path=base,
        user_slug="local-user",
    )
    assert [c.call_id for c in calls] == ["call-mine"]


def test_auxiliary_calls_return_empty_without_a_logs_dir(tmp_path: Path) -> None:
    """A data root with no ``logs/`` directory has no auxiliary calls."""
    from runtime.replay import load_auxiliary_calls

    calls = load_auxiliary_calls(
        source_data_dir=tmp_path / "data",
        thread_base_path=tmp_path / "thread.jsonl",
        user_slug="local-user",
    )
    assert calls == []


def test_result_unwrap_covers_invalid_shapes() -> None:
    import runtime.replay._recording as recording

    assert recording._unwrap_envelope(None) == {}
    assert recording._unwrap_envelope({"content": 123}) == {"content": 123}
    assert recording._unwrap_envelope({"content": "{bad"}) == {"content": "{bad"}
    assert recording._unwrap_envelope({"content": "{}"}) == {}
    assert recording._text(None) == ""
    assert recording._text("plain") == "plain"


def test_load_recorded_turn_normalizes_non_dict_turn_input(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store._entries[0]["turn_input"] = []

    recorded = load_recorded_turn(store, 0)

    assert recorded.calls == []


def test_auxiliary_calls_skip_malformed_segments(tmp_path: Path) -> None:
    from runtime.replay import load_auxiliary_calls

    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "decisions.jsonl").write_text(
        "{bad json\n[]\n" + json.dumps({"topic": "other"}) + "\n",
        encoding="utf-8",
    )

    calls = load_auxiliary_calls(
        source_data_dir=tmp_path / "data",
        thread_base_path=tmp_path / "thread.jsonl",
        user_slug="local-user",
    )

    assert calls == []


def test_auxiliary_calls_wrap_unloadable_snapshot_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.replay._recording as recording
    from runtime.replay import load_auxiliary_calls

    logs = tmp_path / "data" / "logs"
    logs.mkdir(parents=True)
    (logs / "decisions.jsonl").write_text(
        json.dumps(
            {
                "topic": "model.request",
                "user_slug": "local-user",
                "call_id": "call-1",
                "request_ref": "request-ref",
                "response_ref": "response-ref",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _load_error(_ref: str) -> dict:
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(recording, "archive_loader", lambda _base: _load_error)

    with pytest.raises(ReplayError, match="auxiliary call request-ref"):
        load_auxiliary_calls(
            source_data_dir=tmp_path / "data",
            thread_base_path=tmp_path / "thread.jsonl",
            user_slug="local-user",
        )
