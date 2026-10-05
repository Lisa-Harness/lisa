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
import contextlib
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


def _register_many(monkeypatch: pytest.MonkeyPatch, catalog: dict[str, ToolDefinition]) -> None:
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", catalog)


class _NoopBindingProvider:
    """Minimal capability-binding double: owned tools bind an empty config.

    Mirrors the shape ``_resolve_execution_configuration`` and
    ``dispatch_child`` expect — enough for nested dispatch without dragging
    in the real binding machinery.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, Any] = {}

    def binding_for(self, ref: Any) -> Any | None:
        key = getattr(ref, "owner_id", None) or str(ref)
        if key not in self._bindings:
            from kernel.capability_binding import CapabilityBinding

            self._bindings[key] = CapabilityBinding(key, {})
        return self._bindings[key]


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

    def unsubscribe(self, topic: str, handler: Any) -> None:
        handlers = self._handlers.get(topic)
        if handlers is None:
            return
        with contextlib.suppress(ValueError):
            handlers.remove(handler)

    def handler_count(self, topic: str) -> int:
        return len(self._handlers.get(topic, ()))


class TestZeroDisablesWrapper:
    """``timeout_seconds: 0`` disables the wrapper entirely."""

    @pytest.mark.asyncio
    async def test_zero_timeout_skips_the_deadline_path_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wrapper must not enter any deadline machinery at 0.

        A sleep-based test passes even if 0 is coerced to any default bound
        (2026-10-05 review, C4); pin the path instead: ``asyncio.wait_for``
        must never be called.
        """

        async def quick(ctx: ToolExecutionContext) -> str:
            return "done"

        called = []

        real_wait_for = asyncio.wait_for

        async def _spied_wait_for(*args: Any, **kwargs: Any) -> Any:
            called.append(True)
            return await real_wait_for(*args, **kwargs)

        monkeypatch.setattr("kernel.tools.registry.asyncio.wait_for", _spied_wait_for)
        _register(monkeypatch, _definition(timeout_seconds=0, stream_output=False, execute=quick))

        result = await execute_tool("probe", _ctx())
        assert result.output_text == "done"
        assert called == [], "timeout_seconds: 0 must not enter any wait_for deadline"


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
        it publishes on the topic — the idle reset requires the
        stream_output contract, not mere bus traffic (2026-10-05 review, C4)."""
        bus = _FakeBus()

        async def noisy_but_not_streaming(ctx: ToolExecutionContext) -> str:
            # Publishes raw topic events bypassing the stream_output gate:
            # if the wrapper applied idle semantics to a non-streaming tool,
            # these resets would keep it alive past the bound.
            for seq in range(8):
                await asyncio.sleep(0.3)
                await bus.publish(
                    "lisa.tool_output",
                    {"tool_id": ctx.tool_id, "text": f"line {seq}", "stream": "stdout"},
                )
            return "done"

        _register(
            monkeypatch,
            _definition(timeout_seconds=1, stream_output=False, execute=noisy_but_not_streaming),
        )
        with pytest.raises(ToolExecutionError, match="timed out"):
            await execute_tool("probe", _ctx(bus=bus))


class TestDegradeAndCleanup:
    """The no-bus degrade path and the subscription lifetime (C1/C5)."""

    @pytest.mark.asyncio
    async def test_streaming_without_bus_keeps_lifetime_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stream_output: true but no bus wired → degrade to lifetime bound."""

        async def silent(ctx: ToolExecutionContext) -> str:
            await asyncio.sleep(10)
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=silent))
        with pytest.raises(ToolExecutionError, match="timed out"):
            await execute_tool("probe", _ctx())  # no event_bus

    @pytest.mark.asyncio
    async def test_subscription_removed_after_successful_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The progress subscription is removed when the call completes.

        A leaked handler holds the ToolExecutionContext alive and makes every
        later publish on the topic pay an awaited no-op per dead handler
        (2026-10-05 review, C1).
        """
        bus = _FakeBus()

        async def chatty(ctx: ToolExecutionContext) -> str:
            for seq in range(3):
                await asyncio.sleep(0.05)
                publish_tool_output(ctx, text=f"line {seq}")
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=chatty))
        before = bus.handler_count("lisa.tool_output")
        await execute_tool("probe", _ctx(bus=bus))
        assert bus.handler_count("lisa.tool_output") == before

    @pytest.mark.asyncio
    async def test_subscription_removed_after_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The subscription is also removed on the cancellation path."""
        bus = _FakeBus()

        async def silent(ctx: ToolExecutionContext) -> str:
            await asyncio.sleep(10)
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=silent))
        before = bus.handler_count("lisa.tool_output")
        with pytest.raises(ToolExecutionError, match="timed out"):
            await execute_tool("probe", _ctx(bus=bus))
        assert bus.handler_count("lisa.tool_output") == before


class TestNestedDispatchRelay:
    """A composed child's progress resets the parent's idle clock (O2)."""

    @pytest.mark.asyncio
    async def test_child_stream_resets_parent_idle_clock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parent (finite bound, streaming) delegates to a slow streaming
        child; the child's publishes must keep the parent alive.

        This is the 2026-10-05 composition (bash_run → bwrap_run): without
        the relay, the child publishes under its own tool_id and the parent's
        filter never matches, so the parent's wrapper cancels mid-flight on
        legitimate work — the exact failure the redesign exists to fix.
        """
        bus = _FakeBus()

        async def child(ctx: ToolExecutionContext) -> str:
            for seq in range(8):
                await asyncio.sleep(0.3)
                publish_tool_output(ctx, text=f"child line {seq}")
            return "child done"

        async def parent(ctx: ToolExecutionContext) -> str:
            assert ctx.delegated_tool_dispatch is not None
            result = await ctx.delegated_tool_dispatch("child", {})
            return f"parent got: {result.output_text}"

        _register_many(
            monkeypatch,
            {
                "probe": ToolDefinition(
                    tool_id="probe",
                    timeout_seconds=1,
                    stream_output=True,
                    implementation_module=SimpleNamespace(execute=parent),
                ),
                "child": ToolDefinition(
                    tool_id="child",
                    timeout_seconds=0,
                    stream_output=True,
                    implementation_module=SimpleNamespace(execute=child),
                ),
            },
        )
        # The parent needs a capability binding provider for dispatch_child.
        ctx = ToolExecutionContext(
            tool_id="probe",
            event_bus=bus,
            stream_output=True,
            capability_binding_provider=_NoopBindingProvider(),
        )
        result = await execute_tool("probe", ctx)
        assert result.output_text == "parent got: child done"

    @pytest.mark.asyncio
    async def test_relay_cleared_after_parent_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The relay must not linger on the context after the call ends."""

        async def child(ctx: ToolExecutionContext) -> str:
            publish_tool_output(ctx, text="hi")
            return "done"

        async def parent(ctx: ToolExecutionContext) -> str:
            assert ctx.delegated_tool_dispatch is not None
            await ctx.delegated_tool_dispatch("child", {})
            assert ctx.idle_progress_relay is not None  # set during the call
            return "done"

        _register_many(
            monkeypatch,
            {
                "probe": ToolDefinition(
                    tool_id="probe",
                    timeout_seconds=1,
                    stream_output=True,
                    implementation_module=SimpleNamespace(execute=parent),
                ),
                "child": ToolDefinition(
                    tool_id="child",
                    timeout_seconds=0,
                    stream_output=True,
                    implementation_module=SimpleNamespace(execute=child),
                ),
            },
        )
        ctx = ToolExecutionContext(
            tool_id="probe",
            event_bus=_FakeBus(),
            stream_output=True,
            capability_binding_provider=_NoopBindingProvider(),
        )
        await execute_tool("probe", ctx)
        assert ctx.idle_progress_relay is None


class TestIdleDeadlineTelemetry:
    """The timeout error and span carry the idle semantics."""

    @pytest.mark.asyncio
    async def test_timeout_error_names_the_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def silent(ctx: ToolExecutionContext) -> str:
            await asyncio.sleep(10)
            return "done"

        _register(monkeypatch, _definition(timeout_seconds=1, stream_output=True, execute=silent))
        with pytest.raises(ToolExecutionError) as excinfo:
            await execute_tool("probe", _ctx(bus=_FakeBus()))
        # Message keeps the established contract ("Tool 'x' timed out").
        assert "Tool 'probe' timed out" in str(excinfo.value)
