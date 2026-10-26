"""Regression tests for the ``turn.completed`` event's durability claim.

The event is an audit record that a turn finished. It used to be emitted
before the thread entry was appended, and its payload carried no turn
locator, so a consumer that observed ``turn.completed`` could not tell
whether the turn was durable and had no way to find the entry the payload's
receipts were supposed to live in. Both properties are asserted here against
the real ``_execute_mode_turn`` path rather than against the payload builder
alone, because the defect was the ordering between the emit and the append.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from kernel.turn_runner import TurnResult
from runtime.callbacks import NullTransport
from runtime.handle_message import handle_message
from runtime.testing import build_test_runtime
from runtime.thread import load_recent_thread_entries

from runtime import handle_message as handle_message_mod

_MODELS_YAML = "\n".join(
    [
        "version: 1",
        "mode_models: {default: gemma3:latest}",
        "models:",
        "  - model_id: gemma3:latest",
        "    provider: ollama",
    ]
)


def _make_runtime(tmp_path: Path) -> Any:
    return build_test_runtime(tmp_path, models_yaml=_MODELS_YAML)


def _completed_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if getattr(record, "event", "") == "turn.completed"]


@pytest.mark.asyncio
async def test_turn_completed_is_emitted_after_the_thread_entry_is_appended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The event must not be observable before the entry it points at exists.

    The fake ``run_turn`` records the thread length at the moment the turn
    finishes. The event handler then records the thread length at the moment
    the event is emitted. If the emit still precedes the append, the handler
    sees the pre-append length and this test fails.
    """
    runtime = _make_runtime(tmp_path)
    thread_path = runtime.paths.thread_path

    lengths_at_emit: list[int] = []

    def _record_length(*_args: Any, **_kwargs: Any) -> None:
        lengths_at_emit.append(len(load_recent_thread_entries(thread_path=thread_path, limit=100)))

    async def _fake_run_turn(**_kwargs: Any) -> TurnResult:
        return TurnResult(response_text="done")

    monkeypatch.setattr(handle_message_mod, "run_turn", _fake_run_turn)
    monkeypatch.setattr(handle_message_mod, "emit_structured_event", _record_length)

    with caplog.at_level(logging.INFO, logger="runtime.handle_message"):
        await handle_message(runtime, "do the thing", NullTransport())

    entries = load_recent_thread_entries(thread_path=thread_path, limit=100)
    assert len(entries) == 1, "the turn must persist exactly one thread entry"
    assert lengths_at_emit == [1], (
        "turn.completed must be emitted after the thread entry is appended; "
        f"observed thread length(s) at emit: {lengths_at_emit}"
    )


@pytest.mark.asyncio
async def test_turn_completed_payload_locates_the_thread_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The payload must name the turn and task so the entry is addressable."""
    runtime = _make_runtime(tmp_path)
    thread_path = runtime.paths.thread_path

    async def _fake_run_turn(**_kwargs: Any) -> TurnResult:
        return TurnResult(response_text="done")

    monkeypatch.setattr(handle_message_mod, "run_turn", _fake_run_turn)

    with caplog.at_level(logging.INFO, logger="runtime.handle_message"):
        await handle_message(runtime, "do the thing", NullTransport())

    events = _completed_events(caplog)
    assert len(events) == 1, "exactly one turn.completed event per finished turn"
    event = events[0]

    entries = load_recent_thread_entries(thread_path=thread_path, limit=100)
    assert len(entries) == 1
    entry = entries[0]

    assert getattr(event, "turn_index", None) == entry["turn"], (
        "the event's turn_index must address the appended entry"
    )
    assert getattr(event, "task_id", "") == entry["provenance"]["task_id"], (
        "the event's task_id must match the entry's provenance"
    )
