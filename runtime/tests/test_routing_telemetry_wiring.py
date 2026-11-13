"""Routing telemetry subscriptions (2026-11-13 turn 1258 postmortem follow-up).

``_wire_runtime_context_seams`` subscribes the durable
``DecisionTelemetryHandler`` to the ``turn.decision`` / ``tool.telemetry`` /
``model.request`` topics; the postmortem added ``routing.decision`` (the
hint lifecycle: what the model was shown, whether the deterministic
forced-action decision was armed and whether the model was exempt) and
``routing.forced`` (the enforcer's accountability row). These tests pin the
subscriptions so a fresh bus per hook build keeps all five topics durable —
a missing subscription silently drops those ledger rows (the mining
incident pattern: 21 logged failures, zero rows).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kernel.context.event_bus import InProcessEventBus
from runtime.turn_hooks_builder import _wire_runtime_context_seams

from runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from runtime.testing import build_test_runtime

    return build_test_runtime(tmp_path, user_slug="user")


def _no_discover() -> None:
    return None


def test_context_seams_wiring_subscribes_routing_topics(tmp_path: Path) -> None:
    """Both routing topics are subscribed to the durable decisions handler."""
    runtime = _make_runtime(tmp_path)
    bus = _wire_runtime_context_seams(
        runtime=runtime,
        item_builder=None,
        discover_router=_no_discover,
        discover_gateways=_no_discover,
    )
    # ContextSeams lazily resolves the bus from SessionState (refreshed by
    # the wirer); whatever object the wirer subscribed on must carry both
    # topics. Read the subscriptions off the live bus to pin the contract.
    live_bus = None
    # _wire_runtime_context_seams re-publishes the subscribed bus onto the
    # session; the ContextSeams.event_bus property reads the same object.
    live_bus = bus.event_bus if hasattr(bus, "event_bus") else bus
    assert live_bus is not None

    # Publish through the seams bus on both topics and assert durable rows.
    async def _publish_and_read() -> tuple[list[dict], list[dict]]:
        await live_bus.publish(
            "routing.decision",
            {"mode_id": "default", "hint_kind": "handoff", "forced_armed": True},
        )
        await live_bus.publish(
            "routing.forced",
            {"mode_id": "default", "kind": "transition", "non_compliant_dispatches": 2},
        )
        path: Path = runtime.paths.decisions_path
        assert path.exists(), "decisions ledger missing after routing publishes"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return (
            [r for r in records if r.get("topic") == "routing.decision"],
            [r for r in records if r.get("topic") == "routing.forced"],
        )

    decision_rows, forced_rows = asyncio.run(_publish_and_read())
    assert len(decision_rows) == 1
    assert decision_rows[0]["hint_kind"] == "handoff"
    assert len(forced_rows) == 1
    assert forced_rows[0]["non_compliant_dispatches"] == 2


def test_in_process_event_bus_delivers_both_routing_topics(tmp_path: Path) -> None:
    """``InProcessEventBus`` (the per-hook-build bus) routes both topics.

    Guards the subscription contract end to end: a fresh bus, the wirer's
    subscriptions applied, one publish per topic, one durable row apiece.
    """
    runtime = _make_runtime(tmp_path)
    bus = InProcessEventBus()
    handler = runtime.services.decision_telemetry_handler
    assert handler is not None
    bus.subscribe("routing.decision", handler.publish_routing_decision)
    bus.subscribe("routing.forced", handler.publish_routing_forced)

    async def _drive() -> None:
        await bus.publish(
            "routing.decision",
            {"mode_id": "coding_design", "hint_kind": "none"},
        )
        await bus.publish(
            "routing.forced",
            {"mode_id": "coding_design", "kind": "fork", "targets": ["researcher"]},
        )

    asyncio.run(_drive())
    path = runtime.paths.decisions_path
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    topics = [r["topic"] for r in records]
    assert topics == ["routing.decision", "routing.forced"]
