"""Telemetry loss observability: publish warnings and ledger gap detection.

Regression tests for the two halves of loss observability:

1. A fire-and-forget ledger publish that fails — synchronously while
   scheduling, or asynchronously inside the task — logs a WARNING naming
   the source instead of vanishing. Previously only the synchronous shape
   was caught (at DEBUG), and an exception raised inside the created task
   was never retrieved at all.
2. Both durable handlers stamp each row with a per-instance ``sequence``
   and the writer's ``ledger_instance``; ``ForensicReader`` detects dropped
   rows by finding sequence gaps within one instance and reports the
   count as an honest read note (a lower bound — proof rows went missing,
   not which rows).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from kernel.decision_telemetry import TOOL_TELEMETRY_TOPIC, DecisionTelemetryHandler
from kernel.telemetry import PENDING_TELEMETRY_PUBLISH_TASKS, schedule_telemetry_publish
from kernel.token_usage import (
    TokenUsageTelemetryHandler,
    build_token_usage_event,
    publish_token_usage,
)


class _BrokenBus:
    """A bus whose publish raises inside the scheduled task."""

    async def publish(self, topic: str, payload: Any) -> None:
        raise RuntimeError("subscriber chain down")


async def _drain() -> None:
    await asyncio.sleep(0.05)


# ── Publish warnings ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_publish_failure_warns(caplog: pytest.LogCaptureFixture) -> None:
    """An exception raised inside the scheduled task logs a WARNING.

    Before the guard helper, this failure shape vanished: the per-site
    ``try`` only covered ``create_task`` and nothing retrieved
    ``task.exception()``.
    """
    with caplog.at_level(logging.WARNING, logger="kernel.telemetry"):
        schedule_telemetry_publish(_BrokenBus(), "tool.telemetry", {}, source="tool:read_file")
        await _drain()
    assert any(
        "telemetry publish failed for tool:read_file" in record.message
        for record in caplog.records
    ), [r.message for r in caplog.records]


@pytest.mark.asyncio
async def test_scheduling_failure_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kernel.telemetry"):
        schedule_telemetry_publish(
            object(), "tool.telemetry", {}, source="operation:mine_thread_segment"
        )
        await _drain()
    assert any("bus has no publish method" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_publish_never_raises_into_caller(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="kernel.telemetry"):
        schedule_telemetry_publish(_BrokenBus(), TOOL_TELEMETRY_TOPIC, {}, source="tool:x")
        await _drain()
    # The caller sees no exception; the failure is only a warning.
    assert not PENDING_TELEMETRY_PUBLISH_TASKS


@pytest.mark.asyncio
async def test_token_usage_publish_failure_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A failed durable token row is visible as a WARNING, not silent."""
    event = build_token_usage_event(provider="p", model_id="m", prompt_tokens=1)
    with caplog.at_level(logging.WARNING, logger="kernel.telemetry"):
        publish_token_usage(_BrokenBus(), event)
        await _drain()
    assert any(
        "telemetry publish failed for token_usage:" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_token_usage_no_bus_stays_silent(caplog: pytest.LogCaptureFixture) -> None:
    """The headless no-bus contract is unchanged: silent no-op."""
    event = build_token_usage_event(provider="p", model_id="m", prompt_tokens=1)
    with caplog.at_level(logging.WARNING, logger="kernel.telemetry"):
        publish_token_usage(None, event)
        publish_token_usage(object(), event)
        await _drain()
    assert not any("token_usage" in record.message for record in caplog.records)


# ── Handler instance + sequence stamps ────────────────────────────────


@pytest.mark.asyncio
async def test_token_usage_rows_carry_sequence_and_instance(tmp_path: Path) -> None:
    handler = TokenUsageTelemetryHandler(tmp_path / "token-usage.jsonl")
    for _ in range(3):
        await handler.publish(
            build_token_usage_event(provider="p", model_id="m", prompt_tokens=5)
        )
    rows = [
        json.loads(line) for line in (tmp_path / "token-usage.jsonl").read_text().splitlines()
    ]
    assert [r["sequence"] for r in rows] == [1, 2, 3]
    assert all(r["ledger_instance"] == handler.instance for r in rows)


@pytest.mark.asyncio
async def test_decision_rows_carry_sequence_and_instance(tmp_path: Path) -> None:
    handler = DecisionTelemetryHandler(tmp_path / "decisions.jsonl")
    await handler.publish_turn_decision({"turn_index": 0, "action_type": "ContinueAction"})
    await handler.publish_tool_telemetry({"turn_index": 0, "target_id": "t"})
    rows = [
        json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()
    ]
    assert [r["sequence"] for r in rows] == [1, 2]
    assert all(r["ledger_instance"] == handler.instance for r in rows)
    assert handler.instance != ""


@pytest.mark.asyncio
async def test_new_handler_instance_restarts_sequence(tmp_path: Path) -> None:
    """A restart begins a new instance at 1 — gaps never span instances."""
    first = TokenUsageTelemetryHandler(tmp_path / "token-usage.jsonl")
    await first.publish(build_token_usage_event(provider="p", model_id="m", prompt_tokens=1))
    second = TokenUsageTelemetryHandler(tmp_path / "token-usage.jsonl")
    await second.publish(build_token_usage_event(provider="p", model_id="m", prompt_tokens=1))
    rows = [
        json.loads(line) for line in (tmp_path / "token-usage.jsonl").read_text().splitlines()
    ]
    assert rows[0]["ledger_instance"] != rows[1]["ledger_instance"]
    assert rows[0]["sequence"] == rows[1]["sequence"] == 1
