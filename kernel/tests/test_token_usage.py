"""Tests for the durable TokenUsageTelemetryHandler."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kernel.models import TokenUsageEvent
from kernel.token_usage import TokenUsageTelemetryHandler, build_token_usage_event


@pytest.mark.asyncio
async def test_handler_appends_event_as_jsonl_line() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "lsa-state" / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        event = build_token_usage_event(
            provider="ollama",
            model_id="llama3.2:latest",
            prompt_tokens=120,
            completion_tokens=30,
            cached_tokens=10,
            reasoning_tokens=5,
            duration_ms=450.0,
            latency_ms=450,
            call_site="complete_live.openai_compatible",
        )
        await handler.publish(event)

        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "token_used"
        assert record["provider"] == "ollama"
        assert record["model_id"] == "llama3.2:latest"
        assert record["prompt_tokens"] == 120
        assert record["completion_tokens"] == 30
        assert record["total_tokens"] == 150
        assert record["cached_tokens"] == 10
        assert record["reasoning_tokens"] == 5
        assert record["duration_ms"] == 450.0
        assert record["latency_ms"] == 450
        assert record["call_site"] == "complete_live.openai_compatible"
        # Idempotency key is stamped when empty.
        assert record["idempotency_key"]


@pytest.mark.asyncio
async def test_handler_appends_multiple_events_without_collapsing() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        for i in range(3):
            event = build_token_usage_event(
                provider="openai",
                model_id="gpt-4o",
                prompt_tokens=i * 10,
                completion_tokens=5,
            )
            await handler.publish(event)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        # Each line gets a unique idempotency key.
        keys = {json.loads(line)["idempotency_key"] for line in lines}
        assert len(keys) == 3


@pytest.mark.asyncio
async def test_handler_ignores_non_token_usage_events() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)

        # Publish a bare dict (not a TokenUsageEvent).
        await handler.publish({"event_type": "something_else"})

        assert not path.exists()


@pytest.mark.asyncio
async def test_handler_filter_fn_skips_events() -> None:
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path, filter_fn=lambda e: e.prompt_tokens > 0)

        # Event with zero prompt_tokens is filtered out.
        await handler.publish(
            build_token_usage_event(provider="ollama", model_id="x", prompt_tokens=0)
        )
        # Event with prompt_tokens > 0 is persisted.
        await handler.publish(
            build_token_usage_event(provider="ollama", model_id="y", prompt_tokens=10)
        )

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["model_id"] == "y"


@pytest.mark.asyncio
async def test_handler_swallows_os_errors() -> None:
    # Point the handler at a path whose parent is a file (not a dir),
    # so mkdir fails. The handler must swallow the OSError and return.
    with TemporaryDirectory() as td:
        blocker = Path(td) / "blocker"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        path = blocker / "token-usage.jsonl"

        handler = TokenUsageTelemetryHandler(path)
        # Must not raise.
        await handler.publish(
            build_token_usage_event(provider="ollama", model_id="x", prompt_tokens=1)
        )


def test_build_token_usage_event_defaults_total_tokens() -> None:
    event = build_token_usage_event(
        provider="openai",
        model_id="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert event.total_tokens == 150
    assert event.kind == "chat_completion"
    assert event.duration_ms == 0.0
    assert event.time_cost_usd == 0.0


def test_build_token_usage_event_supports_time_billed_kind() -> None:
    event = build_token_usage_event(
        provider="ollama",
        model_id="llama3.2:latest",
        kind="time_billed",
        duration_ms=5000.0,
        time_cost_usd=0.001,
        prompt_tokens=200,  # still populated for observability
        completion_tokens=80,
    )
    assert event.kind == "time_billed"
    assert event.duration_ms == 5000.0
    assert event.time_cost_usd == 0.001
    assert event.prompt_tokens == 200