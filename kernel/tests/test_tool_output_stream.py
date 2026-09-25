"""Tests for incremental tool output streaming (``stream_output`` tools).

Covers:
* :func:`kernel.tools.output_stream.publish_tool_output` — gating on
  ``ctx.stream_output`` / bus presence, payload shape, failure isolation.
* The turn-state ``tool_output`` stage forwarding (no cap bookkeeping).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from kernel.models import ToolExecutionContext
from kernel.tools.output_stream import TOOL_OUTPUT_TOPIC, publish_tool_output


class _FakeBus:
    """Minimal EventBus double recording publishes and subscriptions."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.subscriptions: list[tuple[str, Any]] = []

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))

    def subscribe(self, topic: str, handler: Any) -> None:
        self.subscriptions.append((topic, handler))


def _ctx(**overrides: Any) -> ToolExecutionContext:
    fields: dict[str, Any] = {"tool_id": "bash_run"}
    fields.update(overrides)
    return ToolExecutionContext(**fields)


class TestPublishToolOutput:
    """The pack-facing publish helper."""

    def test_publishes_on_topic_when_gated_on(self) -> None:
        bus = _FakeBus()
        ctx = _ctx(event_bus=bus, stream_output=True)

        async def drive() -> None:
            publish_tool_output(ctx, text="line one", seq=0)
            # create_task needs a yielding loop to run the publish.
            await asyncio.sleep(0)

        asyncio.run(drive())
        assert len(bus.published) == 1
        topic, payload = bus.published[0]
        assert topic == TOOL_OUTPUT_TOPIC
        assert payload == {"tool_id": "bash_run", "text": "line one", "stream": "stdout", "seq": 0}

    def test_gated_off_publishes_nothing(self) -> None:
        bus = _FakeBus()
        ctx = _ctx(event_bus=bus, stream_output=False)

        async def drive() -> None:
            publish_tool_output(ctx, text="line")
            await asyncio.sleep(0)

        asyncio.run(drive())
        assert bus.published == []

    def test_no_bus_publishes_nothing(self) -> None:
        ctx = _ctx(event_bus=None, stream_output=True)

        # Must not raise even with no bus attached.
        async def drive() -> None:
            publish_tool_output(ctx, text="line")
            await asyncio.sleep(0)

        asyncio.run(drive())

    def test_empty_text_publishes_nothing(self) -> None:
        bus = _FakeBus()
        ctx = _ctx(event_bus=bus, stream_output=True)

        async def drive() -> None:
            publish_tool_output(ctx, text="")
            await asyncio.sleep(0)

        asyncio.run(drive())
        assert bus.published == []

    def test_stream_label_carried(self) -> None:
        bus = _FakeBus()
        ctx = _ctx(event_bus=bus, stream_output=True)

        async def drive() -> None:
            publish_tool_output(ctx, text="boom", stream="stderr")
            await asyncio.sleep(0)

        asyncio.run(drive())
        assert bus.published[0][1]["stream"] == "stderr"

    def test_publish_failure_is_swallowed(self) -> None:
        class _ExplodingBus:
            async def publish(self, topic: str, payload: dict[str, Any]) -> None:
                raise RuntimeError("bus down")

        ctx = _ctx(event_bus=_ExplodingBus(), stream_output=True)

        # Must not raise — streaming output never breaks the tool call.
        async def drive() -> None:
            publish_tool_output(ctx, text="line")
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        asyncio.run(drive())


class TestEmitToolOutput:
    """The turn-state forwarder emits a ``tool_output`` stage event."""

    def test_forwards_label_and_payload_without_caps(self) -> None:
        from kernel.turn_phases._turn_state import _TurnState, emit_tool_output

        received: list[tuple[str, dict[str, Any]]] = []

        async def hook(label: str, data: dict[str, Any]) -> None:
            received.append((label, dict(data)))

        state = _TurnState()
        state.max_iterations = 7
        state.max_tool_calls = 5
        state.max_handoffs = 3

        asyncio.run(
            emit_tool_output(state, hook, __import__("logging").getLogger("t"), {"text": "hi"})
        )
        assert len(received) == 1
        label, payload = received[0]
        assert label == "tool_output"
        # Deliberately NOT stamped with iteration/cap bookkeeping: an output
        # line is not a stage transition and must not touch the counters.
        assert "max_iterations" not in payload
        assert "max_tool_calls" not in payload
        assert payload["text"] == "hi"

    def test_hook_failure_is_swallowed(self) -> None:
        from kernel.turn_phases._turn_state import _TurnState, emit_tool_output

        async def hook(label: str, data: dict[str, Any]) -> None:
            raise RuntimeError("transport down")

        state = _TurnState()
        # Must not raise.
        asyncio.run(
            emit_tool_output(state, hook, __import__("logging").getLogger("t"), {"text": "hi"})
        )

    def test_none_hook_is_noop(self) -> None:
        from kernel.turn_phases._turn_state import _TurnState, emit_tool_output

        state = _TurnState()
        asyncio.run(
            emit_tool_output(state, None, __import__("logging").getLogger("t"), {"text": "hi"})
        )


class TestContextFlag:
    """``ToolExecutionContext.stream_output`` and registry propagation."""

    def test_default_false(self) -> None:
        assert _ctx().stream_output is False

    def test_registry_resolves_definition_flag(self) -> None:
        from kernel.models import ToolDefinition
        from kernel.tools.registry import _resolve_execution_configuration

        definition = ToolDefinition(tool_id="bash_run", stream_output=True)
        ctx = _ctx(tool_id="bash_run", pack_configuration_resolver=None)
        _resolve_execution_configuration(definition, ctx)
        assert ctx.stream_output is True

    def test_registry_flag_false_by_default(self) -> None:
        from kernel.models import ToolDefinition
        from kernel.tools.registry import _resolve_execution_configuration

        definition = ToolDefinition(tool_id="quiet_tool")
        ctx = _ctx(tool_id="quiet_tool", pack_configuration_resolver=None)
        _resolve_execution_configuration(definition, ctx)
        assert ctx.stream_output is False


@pytest.mark.parametrize("topic", [TOOL_OUTPUT_TOPIC])
def test_topic_name_is_stable(topic: str) -> None:
    """The bus topic is a cross-repo contract — lock it."""
    assert topic == "lisa.tool_output"
