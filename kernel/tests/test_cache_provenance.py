"""Workstream 1: durable rows carry provenance, never fabricated zeros.

Extends the off-turn producer regression: counters the gateway did not
surface persist as absent (unknown), and the model-data source/version
provenance fields ride through when the model definition carries them.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from kernel.seams.host_adapter import CompletionAuditContext, build_callable_host
from kernel.token_usage import TokenUsageTelemetryHandler


class _ModelDef:
    model_id = "test-model"
    provider = "test"
    reasoning_model = False
    effective_model_data_source = None
    effective_model_data_version = None


class _SparseUsage:
    """A usage object where most counters are simply absent."""

    prompt_tokens = 12
    completion_tokens = 4


class _CompletionResult:
    content = "done"
    request_id = "req-7"
    duration_ms = 44.0
    usage = _SparseUsage()


class _Gateway:
    def __init__(self, result: Any = None) -> None:
        self.result = result or _CompletionResult()

    async def complete(self, request: Any) -> Any:
        return self.result


class _Gateways:
    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway

    def gateway_for(self, provider: str) -> Any:
        return self.gateway


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


def _audit(bus: Any) -> CompletionAuditContext:
    return CompletionAuditContext(
        record_request=lambda snapshot: "r" + "a" * 64,
        event_bus=bus,
        operation_id="mine_thread_segment",
        turn_index=None,
        user_slug="default",
        thread_store_slug="",
    )


async def _drain() -> None:
    import asyncio

    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_off_turn_missing_counters_stay_unknown() -> None:
    """An unsurfaced counter persists as absent, not as a fabricated zero."""
    bus = _Bus()
    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(bus),
    )
    await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
    )
    await _drain()
    usage = next(p for t, p in bus.published if t == "lisa.token_usage")
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 4
    # Never surfaced by the gateway: unknown, not a fabricated zero.
    assert usage["cached_tokens"] is None
    assert usage["cache_creation_tokens"] is None
    assert usage["reasoning_tokens"] is None


@pytest.mark.asyncio
async def test_off_turn_provenance_stamps_when_known() -> None:
    """Source/version ride through when the model definition carries them."""

    class _ProvenancedModel(_ModelDef):
        effective_model_data_source = "manifest"
        effective_model_data_version = "abc123"

    bus = _Bus()
    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(bus),
    )
    await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ProvenancedModel(),
        max_tokens=10,
        temperature=0.0,
    )
    await _drain()
    usage = next(p for t, p in bus.published if t == "lisa.token_usage")
    assert usage["effective_model_data_source"] == "manifest"
    assert usage["effective_model_data_version"] == "abc123"


@pytest.mark.asyncio
async def test_forced_final_forwards_metric_origins() -> None:
    """The forced-final producer forwards origins like the normal turn path."""
    from kernel.token_usage import build_token_usage_event

    event = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=10,
        completion_tokens=5,
        metric_origins={"prompt_tokens": "derived"},
    )
    # prompt+completion both known and total absent -> total is derived and
    # stamped; the explicit prompt origin survives beside it.
    assert event.metric_origins["prompt_tokens"] == "derived"
    assert event.metric_origins["total_tokens"] == "derived"
    with TemporaryDirectory() as td:
        path = Path(td) / "token-usage.jsonl"
        handler = TokenUsageTelemetryHandler(path)
        await handler.publish(event)
        row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["metric_origins"]["prompt_tokens"] == "derived"
    assert row["metric_origins"]["total_tokens"] == "derived"
