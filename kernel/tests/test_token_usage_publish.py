"""Tests for the ``lisa.token_usage`` event-bus wiring.

Covers the full seam: the kernel's :func:`publish_token_usage` helper
(topic contract, best-effort swallowing, no-bus no-op) and the host's
subscriber wiring (``TokenUsageTelemetryHandler`` subscribed to the topic
in ``build_turn_hooks`` so every model call persists to
``token-usage.jsonl``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from kernel.context import InProcessEventBus
from kernel.token_usage import (
    TOKEN_USAGE_TOPIC,
    TokenUsageTelemetryHandler,
    build_token_usage_event,
    publish_token_usage,
)


class _BrokenBus:
    """A bus whose publish raises synchronously (the 2026-09-08 signature)."""

    async def publish(self, event) -> None:
        raise TypeError("publish() missing 1 required positional argument")


@pytest.mark.asyncio
async def test_publish_token_usage_delivers_payload_to_subscriber() -> None:
    """publish_token_usage routes a serialized event to a subscribed handler."""
    bus = InProcessEventBus()
    received: list[dict] = []

    async def subscriber(payload: dict) -> None:
        received.append(payload)

    bus.subscribe(TOKEN_USAGE_TOPIC, subscriber)

    event = build_token_usage_event(
        provider="ollama",
        model_id="kimi-k3:cloud",
        prompt_tokens=100,
        completion_tokens=20,
        call_site="turn_loop",
        mode_id="default",
    )
    publish_token_usage(bus, event)
    # Let the fire-and-forget task run.
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0]["event_type"] == "token_used"
    assert received[0]["model_id"] == "kimi-k3:cloud"
    assert received[0]["prompt_tokens"] == 100
    assert received[0]["call_site"] == "turn_loop"


@pytest.mark.asyncio
async def test_publish_token_usage_survives_synchronous_publish_raise() -> None:
    """A bus whose publish raises synchronously must not break the caller.

    Regression for the 2026-09-08 telemetry incident: a raise escaping the
    model-call path replaces the completion with a TypeError.
    """
    event = build_token_usage_event(provider="x", model_id="y", prompt_tokens=1)
    # Must not raise.
    publish_token_usage(_BrokenBus(), event)
    publish_token_usage(None, event)  # no bus: silent no-op
    publish_token_usage(object(), event)  # no publish attr: silent no-op


@pytest.mark.asyncio
async def test_host_handler_subscription_persists_events() -> None:
    """End-to-end: bus + handler subscription writes durable JSONL records.

    Mirrors the host wiring in ``turn_hooks_builder._wire_context_seams``:
    a fresh ``InProcessEventBus`` per hook build, with the runtime's
    ``TokenUsageTelemetryHandler.publish`` subscribed to the topic.
    """
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)
        bus = InProcessEventBus()
        bus.subscribe(TOKEN_USAGE_TOPIC, handler.publish)

        # Simulate the three kernel publish sites.
        for call_site in ("turn_loop", "forced_final", "terminal_stage"):
            publish_token_usage(
                bus,
                build_token_usage_event(
                    provider="ollama",
                    model_id="kimi-k3:cloud",
                    prompt_tokens=10,
                    completion_tokens=5,
                    call_site=call_site,
                ),
            )
        await asyncio.sleep(0.05)

        records = handler.read_recent(limit=10)
        assert [r["call_site"] for r in records] == [
            "turn_loop",
            "forced_final",
            "terminal_stage",
        ]
        assert all(r["event_type"] == "token_used" for r in records)
        # The lifetime /tokens view reads the same file shape.
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(raw) == 3
        assert sum(int(r["total_tokens"]) for r in raw) == 45
