"""Tests for ``HostRuntime.aclose`` cancellation during deferred teardown.

Reproduces the shutdown race where the ``aclose()`` caller is cancelled
while it is inside ``asyncio.shield(self._mcp_start_task)``. The start task
owns the MCP anyio cancel scopes and must finish teardown best-effort;
``aclose`` must NOT propagate a ``CancelledError`` out (that surfaces as
the ugly ``Cancelled via cancel scope ... by <Task _start_pool_impl>``
traceback on ``asyncio.run`` exit).

See ``deferred-start-owns-teardown.md`` and commit 868162a for background.
"""

from __future__ import annotations

# pylint: disable=protected-access
import asyncio
import contextlib
from pathlib import Path
from typing import Any, cast

import pytest
from kernel.tools.mcp_client import MCPToolPool
from runtime.runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path and no MCP configs."""
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, models_yaml="")
    # No warm/cold configs so _build_pool is a no-op, but the deferred
    # start-task lifecycle (started_event / close_signal / start_task) is
    # still exercised.
    runtime.resources._mcp_warm_configs = []
    runtime.resources._mcp_cold_configs = []
    return runtime


def test_aclose_suppresses_cancel_during_deferred_teardown(tmp_path: Path) -> None:
    """Cancelling the aclose() caller mid-teardown must not raise.

    Simulates asyncio.run shutdown: the main task calls aclose(), which is
    inside ``asyncio.shield(start_task)``; the main task is then cancelled.
    aclose() must return cleanly (suppress the CancelledError) so the
    asyncio.run exit path does not print a cancel-scope traceback. The
    start task is left to finish teardown best-effort.
    """

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.start_pool_deferred()
        # Let the build phase complete.
        await runtime.ensure_pool_ready()

        aclose_task = asyncio.create_task(runtime.aclose())
        # Give aclose a chance to reach the shield(await start_task) point.
        await asyncio.sleep(0)
        # Simulate asyncio.run cancelling the aclose caller mid-teardown.
        aclose_task.cancel()
        # Must NOT raise CancelledError out of aclose.
        await aclose_task

    asyncio.run(_scenario())


def test_aclose_clean_when_start_task_already_done(tmp_path: Path) -> None:
    """aclose() after the start task already finished (sync-style) is clean."""

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.start_pool_deferred()
        await runtime.ensure_pool_ready()
        # Force the start task to complete by setting the close signal
        # and awaiting it before calling aclose.
        assert runtime.resources._mcp_close_signal is not None
        runtime.resources._mcp_close_signal.set()
        assert runtime.resources._mcp_start_task is not None
        await runtime.resources._mcp_start_task
        # Now aclose should be a no-op (task already done) and not raise.
        await runtime.aclose()

    asyncio.run(_scenario())


def test_aclose_idempotent_after_cancelled_teardown(tmp_path: Path) -> None:
    """A second aclose() after a cancelled teardown is a safe no-op."""

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.start_pool_deferred()
        await runtime.ensure_pool_ready()

        aclose_task = asyncio.create_task(runtime.aclose())
        await asyncio.sleep(0)
        aclose_task.cancel()
        try:
            await aclose_task
        except asyncio.CancelledError:
            pytest.fail("aclose must not propagate CancelledError")
        # Second call must not raise and must not error.
        await runtime.aclose()

    asyncio.run(_scenario())


def test_aclose_suppresses_base_exception_group(tmp_path: Path) -> None:
    """Runtime shutdown suppresses anyio-style teardown exception groups."""

    class FakePool:
        async def aclose(self) -> None:
            raise BaseExceptionGroup(
                "MCP teardown",
                [RuntimeError("cancel scope exited in wrong task")],
            )

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.resources._mcp_tool_pool = cast(Any, FakePool())
        runtime.resources._mcp_started = True
        await runtime.aclose()
        assert runtime.resources._mcp_tool_pool is None

    asyncio.run(_scenario())


def test_mcp_pool_aclose_suppresses_child_stack_base_exception_group() -> None:
    class FakeStack:
        async def aclose(self) -> None:
            raise BaseExceptionGroup(
                "MCP child stack",
                [RuntimeError("cancel scope exited in wrong task")],
            )

    async def _scenario() -> None:
        pool = MCPToolPool([])
        pool._child_stacks["fake"] = cast(Any, FakeStack())
        await pool.aclose()
        assert pool._child_stacks == {}

    asyncio.run(_scenario())


def test_cancelled_aclose_keeps_deferred_state_until_task_finishes(tmp_path: Path) -> None:
    class BlockingPool:
        def __init__(self) -> None:
            self.closing = asyncio.Event()
            self.release = asyncio.Event()

        async def aclose(self) -> None:
            self.closing.set()
            await self.release.wait()

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.start_pool_deferred()
        await runtime.ensure_pool_ready()
        pool = BlockingPool()
        runtime.resources._mcp_tool_pool = cast(Any, pool)

        aclose_task = asyncio.create_task(runtime.aclose())
        await pool.closing.wait()
        aclose_task.cancel()
        await aclose_task

        assert runtime.resources._mcp_start_task is not None
        assert not runtime.resources._mcp_start_task.done()

        pool.release.set()
        await runtime.aclose()
        assert runtime.resources._mcp_tool_pool is None

    asyncio.run(_scenario())


def test_readiness_cancellation_does_not_cache_startup_error(tmp_path: Path) -> None:
    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.resources._mcp_started_event = asyncio.Event()
        readiness = asyncio.create_task(runtime.ensure_pool_ready())
        await asyncio.sleep(0)
        readiness.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await readiness
        assert runtime.resources._mcp_start_exc is None

    asyncio.run(_scenario())


def test_aclose_preserves_completed_deferred_start_error(tmp_path: Path) -> None:
    async def _fail_start() -> None:
        raise RuntimeError("deferred startup failed")

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        start_task = asyncio.create_task(_fail_start())
        with contextlib.suppress(RuntimeError):
            await start_task
        runtime.resources._mcp_start_task = start_task
        runtime.resources._mcp_started_event = asyncio.Event()
        runtime.resources._mcp_started_event.set()
        runtime.resources._mcp_close_signal = asyncio.Event()
        await runtime.aclose()
        assert isinstance(runtime.resources._mcp_start_exc, RuntimeError)
        assert str(runtime.resources._mcp_start_exc) == "deferred startup failed"

    asyncio.run(_scenario())


def test_aclose_suppresses_shared_session_lease_release_error(tmp_path: Path) -> None:
    class FakeLease:
        def release(self) -> None:
            raise RuntimeError("lease close failed")

    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        runtime.resources._shared_session_lease = cast(Any, FakeLease())
        await runtime.aclose()
        assert runtime.resources._shared_session_lease is None

    asyncio.run(_scenario())


def test_aclose_caches_cancelled_deferred_start(tmp_path: Path) -> None:
    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        start_task = asyncio.create_task(asyncio.sleep(10))
        start_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await start_task
        runtime.resources._mcp_start_task = start_task
        await runtime.aclose()
        assert isinstance(runtime.resources._mcp_start_exc, asyncio.CancelledError)

    asyncio.run(_scenario())


def test_cancelled_deferred_start_is_cached_by_readiness(tmp_path: Path) -> None:
    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        start_task = asyncio.create_task(asyncio.sleep(10))
        start_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await start_task
        runtime.resources._mcp_start_task = start_task
        runtime.resources._mcp_started_event = asyncio.Event()
        runtime.resources._mcp_started_event.set()

        with pytest.raises(asyncio.CancelledError):
            await runtime.resources.ensure_pool_ready()
        assert isinstance(runtime.resources._mcp_start_exc, asyncio.CancelledError)

    asyncio.run(_scenario())


def test_aclose_preserves_existing_start_error_over_cancellation(tmp_path: Path) -> None:
    async def _scenario() -> None:
        runtime = _make_runtime(tmp_path)
        start_task = asyncio.create_task(asyncio.sleep(10))
        start_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await start_task
        runtime.resources._mcp_start_task = start_task
        runtime.resources._mcp_start_exc = RuntimeError("existing startup failure")

        await runtime.aclose()

        assert isinstance(runtime.resources._mcp_start_exc, RuntimeError)
        assert str(runtime.resources._mcp_start_exc) == "existing startup failure"

    asyncio.run(_scenario())
