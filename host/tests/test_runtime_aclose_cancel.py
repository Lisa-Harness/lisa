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
from pathlib import Path

import pytest

from host.runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a HostRuntime whose data_dir is tmp_path and no MCP configs."""
    from host.testing import build_test_runtime

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
