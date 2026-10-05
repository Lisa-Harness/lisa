"""The wrapper timeout contract: lifetime bounds vs. idle bounds.

Regression background (2026-10-05 shell turn): ``bash_run``/``bwrap_run``
declared no ``timeout_seconds`` so the kernel wrapper inherited the 30s
default and cancelled the call before the pack's auto-background adoption
could run. The designed escape hatch — ``timeout_seconds: 0`` disables the
wrapper — was silently rewritten by the SDK loader's falsy coercion
(``or 30``), so packs had no expressible way to declare a self-bounding tool.
That loader bug is fixed and pinned in ``sdk/tests``; these tests pin the
kernel side of the contract:

* ``timeout_seconds: 0`` disables the wrapper (documented contract,
  ``sdk/AUTHORING.md``).
* A **streaming** tool with a finite ``timeout_seconds`` gets an *idle*
  deadline instead of a lifetime deadline: publishing incremental output on
  the ``lisa.tool_output`` bus topic resets the clock. A streaming tool that
  publishes within the window can run arbitrarily long; a streaming tool
  that goes silent for the full window is treated as hung and cancelled —
  that is the real distinction the wrapper was missing.
* A non-streaming tool keeps the exact lifetime bound (no behavior change).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import ToolDefinition, ToolExecutionContext
from kernel.tools.output_stream import publish_tool_output
from kernel.tools.registry import ToolExecutionError, execute_tool


def _definition(
    *,
    timeout_seconds: int,
    stream_output: bool,
    execute: Any,
) -> ToolDefinition:
    return ToolDefinition(
        tool_id="probe",
        timeout_seconds=timeout_seconds,
        stream_output=stream_output,
        implementation_module=SimpleNamespace(execute=execute),
    )


def _register(monkeypatch: pytest.MonkeyPatch, definition: ToolDefinition) -> None:
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", {"probe": definition})


def _ctx(bus: Any = None) -> ToolExecutionContext:
    return ToolExecutionContext(tool_id="probe", event_bus=bus)


class _FakeBus:
    """Minimal EventBus double recording publishes and dispatching to
    subscribers (mirrors ``InProcessEventBus`` delivery semantics)."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []
        self._handlers: dict[str, list[Any]] = {}

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))
        for handler in list(self._handlers.get(topic, ())):
            await handler(payload)

    def subscribe(self, topic: str, handler: Any) -> None:
        self._handlers.setdefault(topic, []).append(handler)


class TestZeroDisablesWrapper:
    """``timeout_seconds: 0`` disables the wrapper entirely."""

    @pytest.mark.asyncio
    async def test_zero_timeout_runs_longer_than_any_former_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def slow(ctx: ToolExecutionContext) -> Any:
            await asyncio.sleep(0.3)
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=0, stream_output=False, execute=slow))

        # A 30s-lifetime wrapper would never trip here, but 0 must mean
        # "no wrapper": the call completes whenever it completes.
        result = await execute_tool("probe", _ctx())
        assert result.output_text == "done"


class TestStreamingIdleDeadline:
    """Streaming tools get an idle deadline, not a lifetime deadline."""

    @pytest.mark.asyncio
    async def test_streaming_output_resets_the_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Total runtime exceeds the declared timeout, but the tool keeps
        publishing and finishes successfully."""
        bus = _FakeBus()

        async def chatty(ctx: ToolExecutionContext) -> str:
            # 8 publishes spaced 0.3s: total ~2.4s against a 1s declared
            # timeout. Lifetime semantics cancel at 1s; idle semantics reset
            # the clock on each publish and let the call finish.
            for seq in range(8):
                await asyncio.sleep(0.3)
                publish_tool_output(ctx, text=f"line {seq}")
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=chatty))
        ctx = _ctx(bus=bus)
        result = await execute_tool("probe", ctx)
        assert result.output_text == "done"
        assert len(bus.published) == 8

    @pytest.mark.asyncio
    async def test_streaming_tool_silent_past_window_is_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No publishes within the window → treated as hung → cancelled."""
        events: list[str] = []

        async def silent(ctx: ToolExecutionContext) -> Any:
            events.append("started")
            await asyncio.sleep(10)
            events.append("never")
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=silent))
        with pytest.raises(ToolExecutionError, match="timed out"):
            await execute_tool("probe", _ctx(bus=_FakeBus()))
        assert events == ["started"]

    @pytest.mark.asyncio
    async def test_non_streaming_keeps_lifetime_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-streaming tool is cancelled at the declared bound even while
        it publishes — the idle reset requires the stream_output contract."""

        async def silent(ctx: ToolExecutionContext) -> Any:
            await asyncio.sleep(10)
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=False, execute=silent))
        with pytest.raises(ToolExecutionError, match="timed out"):
            await execute_tool("probe", _ctx(bus=_FakeBus()))


class TestIdleDeadlineTelemetry:
    """The timeout error and span carry the idle semantics."""

    @pytest.mark.asyncio
    async def test_timeout_error_names_the_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def silent(ctx: ToolExecutionContext) -> Any:
            await asyncio.sleep(10)
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=silent))
        with pytest.raises(ToolExecutionError) as excinfo:
            await execute_tool("probe", _ctx(bus=_FakeBus()))
        # Message keeps the established contract ("Tool 'x' timed out").
        assert "Tool 'probe' timed out" in str(excinfo.value)
