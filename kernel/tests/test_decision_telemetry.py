"""Tests for the durable decision/dispatch telemetry appender.

Covers ``kernel.decision_telemetry``: the JSONL append contract, topic
tagging, non-mapping rejection, filter hooks, and the ``turn_index`` stamp
that makes the record joinable to the thread and the cost log.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from kernel.decision_telemetry import (
    TOOL_TELEMETRY_TOPIC,
    TOPIC_FIELD,
    TURN_DECISION_TOPIC,
    DecisionTelemetryHandler,
)
from kernel.models import ToolExecutionTelemetry
from kernel.token_usage import build_token_usage_event


@pytest.mark.asyncio
async def test_turn_decision_appended_as_jsonl_with_topic() -> None:
    """A decision payload lands as one line tagged with its source topic."""
    with TemporaryDirectory() as td:
        path = Path(td) / "logs" / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_turn_decision(
            {
                "mode_id": "coding_design",
                "turn_index": 42,
                "iteration": 3,
                "action_type": "ContinueAction",
                "action_reason": "tool calls remain",
                "write_receipts": 1,
                "read_receipts": 2,
            }
        )

        assert path.exists()
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record[TOPIC_FIELD] == TURN_DECISION_TOPIC
        assert record["turn_index"] == 42
        assert record["mode_id"] == "coding_design"
        assert record["action_type"] == "ContinueAction"


@pytest.mark.asyncio
async def test_tool_telemetry_appended_and_keeps_turn_index() -> None:
    """A dispatch event keeps the turn key so the ledger joins to the thread."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        event = ToolExecutionTelemetry(
            target_id="fetch_page",
            target_type="tool",
            status="error",
            duration_ms=812,
            error_message="timeout",
            turn_index=17,
        )
        # The bus delivers serialized payloads; the handler must accept that
        # shape as well as the live event object.
        await handler.publish_tool_telemetry(event.model_dump(mode="json"))

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record[TOPIC_FIELD] == TOOL_TELEMETRY_TOPIC
        assert record["turn_index"] == 17
        assert record["target_id"] == "fetch_page"
        assert record["status"] == "error"
        assert record["error_message"] == "timeout"


@pytest.mark.asyncio
async def test_live_event_object_is_dumped_not_stringified() -> None:
    """An unserialized event is dumped into a mapping, not recorded as a repr."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_tool_telemetry(
            ToolExecutionTelemetry(
                target_id="self_model",
                target_type="operation",
                status="success",
                duration_ms=4,
                turn_index=5,
            )
        )

        record = json.loads(path.read_text(encoding="utf-8").strip())
        assert record["target_id"] == "self_model"
        assert record["turn_index"] == 5


@pytest.mark.asyncio
async def test_non_mapping_payload_is_rejected() -> None:
    """A payload that is not a mapping is dropped rather than wrapped."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_turn_decision("not-a-record")
        await handler.publish_turn_decision([1, 2, 3])

        assert handler.read_recent() == []


@pytest.mark.asyncio
async def test_filter_can_drop_a_record() -> None:
    """The filter hook sees the topic and can suppress persistence."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        seen: list[tuple[str, dict[str, Any]]] = []

        def _filter(topic: str, record: dict[str, Any]) -> bool:
            seen.append((topic, record))
            return record.get("mode_id") != "noisy"

        handler = DecisionTelemetryHandler(path, filter_fn=_filter)
        await handler.publish_turn_decision({"mode_id": "noisy", "turn_index": 1})
        await handler.publish_turn_decision({"mode_id": "quiet", "turn_index": 2})

        records = handler.read_recent()
        assert [r["mode_id"] for r in records] == ["quiet"]
        assert [topic for topic, _ in seen] == [TURN_DECISION_TOPIC, TURN_DECISION_TOPIC]


@pytest.mark.asyncio
async def test_raising_filter_still_persists() -> None:
    """A broken filter must not lose evidence; the record still lands."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"

        def _filter(_topic: str, _record: dict[str, Any]) -> bool:
            raise RuntimeError("filter exploded")

        handler = DecisionTelemetryHandler(path, filter_fn=_filter)
        await handler.publish_turn_decision({"mode_id": "kept", "turn_index": 9})

        assert [r["mode_id"] for r in handler.read_recent()] == ["kept"]


@pytest.mark.asyncio
async def test_sequence_distinguishes_same_clock_records() -> None:
    """Two records written in the same tick are still distinguishable."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        handler = DecisionTelemetryHandler(path)

        await handler.publish_turn_decision({"mode_id": "a", "turn_index": 1})
        await handler.publish_turn_decision({"mode_id": "a", "turn_index": 1})

        records = handler.read_recent()
        assert len(records) == 2
        assert records[0]["sequence"] != records[1]["sequence"]


def test_read_recent_tolerates_malformed_lines() -> None:
    """A truncated line is skipped rather than breaking the whole read."""
    with TemporaryDirectory() as td:
        path = Path(td) / "decisions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"mode_id": "ok", "turn_index": 1}\n{"mode_id": "truncated"\n',
            encoding="utf-8",
        )

        records = DecisionTelemetryHandler(path).read_recent()
        assert [r["mode_id"] for r in records] == ["ok"]


def test_read_recent_returns_empty_when_missing() -> None:
    """A missing file is 'no evidence yet', not an error."""
    with TemporaryDirectory() as td:
        handler = DecisionTelemetryHandler(Path(td) / "missing.jsonl")
        assert handler.read_recent() == []


def test_token_usage_event_carries_turn_index() -> None:
    """The cost log is per-turn attributable, not just per-mode."""
    event = build_token_usage_event(
        provider="ollama",
        model_id="llama3.2:latest",
        prompt_tokens=100,
        completion_tokens=20,
        mode_id="coding_design",
        turn_index=88,
    )
    assert event.turn_index == 88
    assert event.mode_id == "coding_design"
