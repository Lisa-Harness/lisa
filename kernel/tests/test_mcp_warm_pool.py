"""Tests for the warm-server pool lifecycle and the ``warm``/``required``
designations on :class:`MCPClientConfig`.

Covers:
- ``MCPClientConfig.warm`` / ``required`` round-trip through construction.
- ``MCPToolPool.start`` with ``warm=True, required=True`` raises
  :class:`MCPConnectionError` after bounded retry when ``_connect`` always
  fails.
- ``MCPToolPool.start`` with ``warm=True, required=False`` logs and proceeds
  with an empty tool list for the failed server.
- ``MCPToolPool.restart_warm`` re-establishes a dropped warm session.
- ``run_turn`` reuses ``hooks.tools.mcp_tool_pool`` when set (no per-turn
  ``start``/``aclose`` on the host pool).

These tests stub ``MCPToolPool._connect`` so no real subprocess is spawned.
The retry-with-backoff sleeps are patched to zero to keep the suite fast.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPConnectionError,
    MCPToolPool,
)
from kernel.turn_hooks import (
    ModelHooks,
    ToolHooks,
)
from kernel.turn_runner import TurnHooks, run_turn

# ── Helpers ─────────────────────────────────────────────────────────────────


def _cold_config(server_id: str = "cold") -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
        warm=False,
        required=False,
    )


def _warm_config(
    server_id: str = "warm",
    *,
    required: bool = False,
) -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
        warm=True,
        required=required,
    )


def _stub_connect(sessions: dict[str, Any], server_id: str) -> Any:
    """Return an async side-effect that records a session on the pool.

    The patched ``_connect`` becomes a bound method, so the effect receives
    ``(self, config)``. It writes into the pool's real ``_sessions`` dict so
    ``warm_server_ids()`` reflects the connection.
    """

    async def _effect(self: MCPToolPool, config: MCPClientConfig) -> None:
        sessions[config.server_id] = object()
        self._sessions[config.server_id] = sessions[config.server_id]

    return _effect


def _failing_connect() -> Any:
    """Return an async side-effect that always raises."""

    async def _effect(_self: MCPToolPool, _config: MCPClientConfig) -> None:
        raise RuntimeError("boom")

    return _effect


# ── Config round-trip ──────────────────────────────────────────────────────


def test_mcp_client_config_warm_required_defaults() -> None:
    cfg = MCPClientConfig(server_id="x", command="c", args=[])
    assert cfg.warm is False
    assert cfg.required is False


def test_mcp_client_config_warm_required_set() -> None:
    cfg = MCPClientConfig(server_id="x", command="c", args=[], warm=True, required=True)
    assert cfg.warm is True
    assert cfg.required is True


# ── start: warm + required raises after retry ──────────────────────────────


@pytest.mark.asyncio
async def test_start_warm_required_raises_on_connect_failure() -> None:
    pool = MCPToolPool([_warm_config("w", required=True)])
    # Patch sleep to zero so the backoff doesn't slow the test.
    with (
        patch("kernel.tools.mcp_client.asyncio.sleep", new=AsyncMock()),
        patch.object(MCPToolPool, "_connect", new=_failing_connect()),
        pytest.raises(MCPConnectionError, match="Required warm MCP server 'w'"),
    ):
        await pool.start()
    await pool.aclose()


# ── start: warm + not-required logs and proceeds ───────────────────────────


@pytest.mark.asyncio
async def test_start_warm_non_required_proceeds_without_server() -> None:
    pool = MCPToolPool([_warm_config("w", required=False)])
    with (
        patch("kernel.tools.mcp_client.asyncio.sleep", new=AsyncMock()),
        patch.object(MCPToolPool, "_connect", new=_failing_connect()),
    ):
        # Should not raise.
        await pool.start()
    assert pool.warm_failed_ids() == ["w"]
    assert "w" not in pool.warm_server_ids()
    tools = await pool.list_tools()
    assert tools == []
    await pool.aclose()


# ── start: cold server is NOT connected at start (lazy) ─────────────────────


@pytest.mark.asyncio
async def test_start_cold_server_not_connected_at_start() -> None:
    """Cold servers are not connected at pool.start() — they are lazily
    connected on first list_tools or call_tool. _connect should never be
    called for a cold server during start()."""
    pool = MCPToolPool([_cold_config("c")])
    # _connect should NOT be called during start for a cold server.
    mock_connect = AsyncMock(side_effect=RuntimeError("should not be called"))
    with patch.object(MCPToolPool, "_connect", new=mock_connect):
        await pool.start()  # no raise, no connect
        assert mock_connect.call_count == 0
    assert "c" not in pool._sessions
    await pool.aclose()


@pytest.mark.asyncio
async def test_list_tools_lazily_connects_cold_server() -> None:
    """list_tools lazily connects cold servers so the catalog reflects
    their tools. A cold server that fails to start is skipped."""
    sessions: dict[str, Any] = {}
    pool = MCPToolPool([_cold_config("c")])
    with patch.object(MCPToolPool, "_connect", new=_stub_connect(sessions, "c")):
        await pool.start()
        assert "c" not in pool._sessions  # not connected at start
        await pool.list_tools()  # triggers lazy connect
        assert "c" in pool._sessions  # connected on demand
    await pool.aclose()


@pytest.mark.asyncio
async def test_list_tools_skips_cold_server_that_fails_lazy_connect() -> None:
    """A cold server that fails to lazily connect is skipped — its tools
    are absent from the catalog but no error is raised."""
    pool = MCPToolPool([_cold_config("c")])
    with patch.object(MCPToolPool, "_connect", new=_failing_connect()):
        await pool.start()
        tools = await pool.list_tools()  # triggers lazy connect, fails, skips
    assert tools == []
    assert "c" not in pool._sessions
    await pool.aclose()


# ── restart_warm: re-establishes a dropped session ─────────────────────────


@pytest.mark.asyncio
async def test_restart_warm_reconnects() -> None:
    sessions: dict[str, Any] = {}
    pool = MCPToolPool([_warm_config("w", required=True)])
    with patch.object(MCPToolPool, "_connect", new=_stub_connect(sessions, "w")):
        await pool.start()
        assert "w" in pool.warm_server_ids()
        # Simulate a crash: drop the session.
        pool._sessions.pop("w")
        ok = await pool.restart_warm("w")
    assert ok is True
    assert "w" in pool.warm_server_ids()
    await pool.aclose()


@pytest.mark.asyncio
async def test_restart_warm_unknown_id_returns_false() -> None:
    pool = MCPToolPool([_warm_config("w")])
    with patch.object(MCPToolPool, "_connect", new=_stub_connect({}, "w")):
        await pool.start()
        ok = await pool.restart_warm("not-in-pool")
    assert ok is False
    await pool.aclose()


# ── warm/cold split on the pool ─────────────────────────────────────────────


def test_pool_splits_warm_and_cold_configs() -> None:
    warm = _warm_config("w")
    cold = _cold_config("c")
    pool = MCPToolPool([warm, cold])
    assert [c.server_id for c in pool.warm_configs] == ["w"]
    assert [c.server_id for c in pool.cold_configs] == ["c"]
    assert [c.server_id for c in pool.configs] == ["w", "c"]


# ── run_turn: prefers host-provided warm pool ───────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_uses_host_warm_pool(monkeypatch: MonkeyPatch) -> None:
    """When ``hooks.tools.mcp_tool_pool`` is set, run_turn must not start/aclose it."""
    # The shared conftest fixture handles all catalog wiring.

    async def stub_complete(**kwargs: Any) -> Any:
        class _R:
            content = "ok"
            tool_calls: list[dict[str, Any]] = []

        return _R()

    # A spy pool that records start/aclose calls but does no real work.
    started = []
    closed = []

    class _SpyPool(MCPToolPool):
        async def start(self) -> None:
            started.append(True)
            self._started = True

        async def aclose(self) -> None:
            closed.append(True)
            self._started = False

        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

    spy = _SpyPool([])
    spy._started = True  # host pre-starts the pool; run_turn must not re-start it
    hooks = TurnHooks(
    models=ModelHooks(
        complete=stub_complete,
    ),
    tools=ToolHooks(
        mcp_tool_pool=spy,
    ),
)
    await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert started == []  # run_turn did not start the host pool
    assert closed == []  # run_turn did not close the host pool
