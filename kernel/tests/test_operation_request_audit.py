"""Durable off-turn model-request recording through the operation host adapter.

Regression tests for the audit seam added to close the recording gap: the
turn loop's own calls ride the thread entry's ``turn_input.calls``, but
completions made by operation dispatches (memory mining, consolidation,
distills) previously left no durable record of the exact request. The
``CompletionAuditContext`` archives each gateway completion content-addressed
and publishes a ``model.request`` decision-ledger row plus a token-usage
event, all best-effort so recording can never fail the completion.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from kernel.decision_telemetry import MODEL_REQUEST_TOPIC, DecisionTelemetryHandler
from kernel.models import OperationExecutionContext
from kernel.seams.host_adapter import CompletionAuditContext, build_callable_host


class _ModelDef:
    model_id = "test-model"
    provider = "test"
    reasoning_model = False


class _Usage:
    prompt_tokens = 12
    completion_tokens = 4
    total_tokens = 16
    cached_tokens = 2
    reasoning_tokens = 0
    cache_creation_tokens = 0
    latency_ms = 30


class _CompletionResult:
    content = "done"
    request_id = "req-42"
    duration_ms = 44.0
    usage = _Usage()


class _Gateway:
    def __init__(self, result: Any = None) -> None:
        self.result = result or _CompletionResult()
        self.requests: list[Any] = []

    async def complete(self, request: Any) -> Any:
        self.requests.append(request)
        return self.result


class _Gateways:
    def __init__(self, gateway: _Gateway) -> None:
        self.gateway = gateway

    def gateway_for(self, provider: str) -> Any:
        return self.gateway


class _Bus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, dict(payload)))


async def _drain() -> None:
    """Let fire-and-forget audit publishes complete."""
    await asyncio.sleep(0.05)


def _audit(
    *,
    recorder: Any,
    bus: Any | None = None,
    operation_id: str = "mine_thread_segment",
    turn_index: int | None = None,
) -> CompletionAuditContext:
    return CompletionAuditContext(
        record_request=recorder,
        event_bus=bus,
        operation_id=operation_id,
        turn_index=turn_index,
        user_slug="local-user",
        thread_store_slug="local-user",
    )


# ── Request archiving ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gateway_completion_records_verbatim_request() -> None:
    recorded: list[dict[str, Any]] = []

    def recorder(snapshot: dict[str, Any]) -> str:
        recorded.append(snapshot)
        return "r" + "a" * 64

    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(recorder=recorder, turn_index=3),
    )
    result = await host.complete(
        messages=[{"role": "user", "content": "distill this"}],
        model_def=_ModelDef(),
        max_tokens=640,
        temperature=0.1,
    )
    assert result.content == "done"
    assert len(recorded) == 1
    # Raw snapshot dict, same call shape as the turn loop's record_request —
    # the recorder owns the {"content": ...} wrapping, so a stored object
    # stays layout-identical to an in-turn snapshot's.
    stored = recorded[0]
    assert stored["messages"] == [{"role": "user", "content": "distill this"}]
    assert stored["model_id"] == "test-model"
    assert stored["operation_id"] == "mine_thread_segment"
    assert stored["turn_index"] == 3
    assert stored["user_slug"] == "local-user"
    assert stored["max_tokens"] == 640
    assert stored["temperature"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_recording_failure_never_fails_the_completion() -> None:
    def broken_recorder(snapshot: dict[str, Any]) -> str:
        raise OSError("disk full")

    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(recorder=broken_recorder),
    )
    result = await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
    )
    assert result.content == "done"


@pytest.mark.asyncio
async def test_structured_output_response_format_is_serializable() -> None:
    """A Pydantic model class in ``response_format`` records by name.

    Structured-output operations (memory mining, consolidation) pass
    ``response_format=MinedFacts`` — a class object, not JSON. The snapshot
    records its name; the schema itself is authored pack content.
    """

    class MinedFacts:  # stand-in for a Pydantic model class
        pass

    recorded: list[dict[str, Any]] = []

    def recorder(snapshot: dict[str, Any]) -> str:
        recorded.append(snapshot)
        return "r" + "9" * 64

    class _StructuredGateway(_Gateway):
        pass

    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_StructuredGateway()),
        audit=_audit(recorder=recorder),
    )
    await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
        response_format=MinedFacts,  # type: ignore[arg-type]
    )
    assert len(recorded) == 1
    assert recorded[0]["response_format"] == "MinedFacts"


@pytest.mark.asyncio
async def test_no_audit_context_keeps_previous_behavior() -> None:
    gateway = _Gateway()
    host = build_callable_host(gateway_provider_fn=lambda: _Gateways(gateway))
    await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
    )
    assert len(gateway.requests) == 1


# ── Ledger row and usage telemetry ─────────────────────────────────────


@pytest.mark.asyncio
async def test_model_request_row_and_usage_event_published() -> None:
    bus = _Bus()

    def recorder(snapshot: dict[str, Any]) -> str:
        return "r" + "b" * 64

    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(recorder=recorder, bus=bus, turn_index=None),
    )
    await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
    )
    await _drain()
    topics = [topic for topic, _ in bus.published]
    assert MODEL_REQUEST_TOPIC in topics
    assert "lisa.token_usage" in topics
    row = next(p for t, p in bus.published if t == MODEL_REQUEST_TOPIC)
    assert row["request_ref"] == "r" + "b" * 64
    # Off-turn work: null is meaningful and distinct from turn 0.
    assert row["turn_index"] is None
    assert row["operation_id"] == "mine_thread_segment"
    usage = next(p for t, p in bus.published if t == "lisa.token_usage")
    assert usage["prompt_tokens"] == 12
    assert usage["call_site"] == "operation"
    assert usage["turn_index"] is None


@pytest.mark.asyncio
async def test_no_event_bus_still_records_the_request() -> None:
    recorded: list[dict[str, Any]] = []

    def recorder(snapshot: dict[str, Any]) -> str:
        recorded.append(snapshot)
        return "r" + "c" * 64

    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(recorder=recorder, bus=None),
    )
    await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
    )
    await _drain()
    assert len(recorded) == 1


@pytest.mark.asyncio
async def test_raising_bus_publish_is_swallowed() -> None:
    class _BrokenBus:
        async def publish(self, topic: str, payload: dict[str, Any]) -> None:
            raise RuntimeError("bus down")

    def recorder(snapshot: dict[str, Any]) -> str:
        return "r" + "d" * 64

    host = build_callable_host(
        gateway_provider_fn=lambda: _Gateways(_Gateway()),
        audit=_audit(recorder=recorder, bus=_BrokenBus()),
    )
    result = await host.complete(
        messages=[{"role": "user", "content": "x"}],
        model_def=_ModelDef(),
        max_tokens=10,
        temperature=0.0,
    )
    await _drain()
    assert result.content == "done"


# ── Decision-telemetry handler persistence ─────────────────────────────


@pytest.mark.asyncio
async def test_model_request_row_persists_to_decisions_jsonl(tmp_path: Path) -> None:
    handler = DecisionTelemetryHandler(tmp_path / "decisions.jsonl")
    await handler.publish_model_request(
        {
            "operation_id": "mine_thread_segment",
            "request_ref": "r" + "e" * 64,
            "turn_index": None,
            "user_slug": "local-user",
            "thread_store_slug": "local-user",
        }
    )
    lines = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["topic"] == MODEL_REQUEST_TOPIC
    assert record["operation_id"] == "mine_thread_segment"
    assert record["turn_index"] is None


# ── Registry choke point ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_builds_audit_from_context_fields(tmp_path: Path) -> None:
    """The registry derives the audit context from the dispatch's own ctx."""
    from kernel.operations.registry import _build_completion_audit

    recorded: list[dict[str, Any]] = []

    def recorder(snapshot: dict[str, Any]) -> str:
        recorded.append(snapshot)
        return "r" + "f" * 64

    ctx = OperationExecutionContext(
        operation_id="mine_thread_segment",
        user_slug="local-user",
        thread_store_slug="local-user--conv1",
        turn_index=7,
        event_bus=None,
        request_recorder=recorder,
    )
    audit = _build_completion_audit("mine_thread_segment", ctx)
    assert audit is not None
    assert audit.operation_id == "mine_thread_segment"
    assert audit.turn_index == 7
    assert audit.thread_store_slug == "local-user--conv1"


@pytest.mark.asyncio
async def test_registry_returns_none_without_recorder() -> None:
    from kernel.operations.registry import _build_completion_audit

    ctx = OperationExecutionContext(
        operation_id="mine_thread_segment",
        event_bus=_Bus(),
    )
    assert _build_completion_audit("mine_thread_segment", ctx) is None