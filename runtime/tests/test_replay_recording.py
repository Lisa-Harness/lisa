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
