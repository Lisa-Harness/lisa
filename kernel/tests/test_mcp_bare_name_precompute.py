"""Tests for the precomputed bare-name index in MCPToolPool.

Regression coverage for the ``*_id`` resolution audit: bare-name resolution
(``_resolve_server_for_bare_name``) must be an O(1) dict lookup over
connected servers via the precomputed ``_bare_to_server_index``, not a
linear scan that lazily spawns cold servers on every lookup. Cold-server
lazy connect still fires on first index build for an unknown bare name,
but only once per unknown name — never for names already resolvable from
warm/connected servers.

These tests stub ``_connect`` so no real subprocess is spawned.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPToolPool,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _config_named(server_id: str, *, warm: bool = False) -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
        warm=warm,
    )


class _FakeResult:
    def __init__(self, tools: list[Any] | None = None, text: str = "") -> None:
        self.tools = tools or []
        self.content = [type("C", (), {"text": text})()] if text else []
        self.is_error = False


class _FakeTool:
    def __init__(
        self, name: str, description: str = "", schema: dict[str, Any] | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {}}


class _FakeSession:
    def __init__(self, tools: list[_FakeTool] | None = None) -> None:
        self._tools = tools or []

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> _FakeResult:
        return _FakeResult(tools=self._tools)

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:
        return _FakeResult(text=f"server:{tool_id}")


def _patch_connect_with_session(
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, _FakeSession],
) -> None:
    """Stub _connect to install a prebuilt fake session for each server_id."""

    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        self._sessions[config.server_id] = sessions[config.server_id]
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bare_to_server_index_populated_on_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """_refresh_tool_index populates _bare_to_server_index for connected servers."""
    _patch_connect_with_session(
        monkeypatch,
        {
            "lisa-memory": _FakeSession([_FakeTool("add_memory"), _FakeTool("search_memory")]),
            "mem0": _FakeSession([_FakeTool("add_note")]),
        },
    )

    pool = MCPToolPool([_config_named("lisa-memory", warm=True), _config_named("mem0", warm=True)])
    await pool.start()
    # Both servers connected at start; their bare names are in the precomputed index.
    assert pool._bare_to_server_index.get("add_memory") == "lisa-memory"
    assert pool._bare_to_server_index.get("search_memory") == "lisa-memory"
    assert pool._bare_to_server_index.get("add_note") == "mem0"
    await pool.aclose()


@pytest.mark.asyncio
async def test_resolve_server_for_bare_name_uses_precomputed_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare-name resolution is an O(1) lookup; no cold connect for known names.

    A warm server exposes ``add_memory``. Resolving the bare name must return
    the server_id from the precomputed index WITHOUT touching _connect (which
    would spawn a subprocess). We allow _connect during start() (warm servers
    are connected at startup), then raise if _connect is invoked during the
    resolution call.
    """
    sessions = {"lisa-memory": _FakeSession([_FakeTool("add_memory")])}
    allow_connect = {"value": True}

    async def gated_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        if not allow_connect["value"]:
            raise AssertionError(
                f"_connect must not be called for bare-name resolution of a "
                f"known tool (server_id={config.server_id}); the precomputed "
                f"index should resolve it."
            )
        self._sessions[config.server_id] = sessions[config.server_id]
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "_connect", gated_connect)

    pool = MCPToolPool([_config_named("lisa-memory", warm=True)])
    await pool.start()
    # After start, the warm server is connected and indexed. Block any
    # further _connect; a bare-name resolution must hit the precomputed
    # index, not _connect.
    allow_connect["value"] = False
    resolved = await pool._resolve_server_for_bare_name("add_memory")
    assert resolved == "lisa-memory"
    await pool.aclose()


@pytest.mark.asyncio
async def test_resolve_server_for_bare_name_returns_none_for_unknown_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown bare name with no cold servers resolves to None without spawning.

    With only warm servers connected and no cold configs to lazily start,
    an unknown bare name resolves to None purely from the precomputed index.
    """
    sessions = {"lisa-memory": _FakeSession([_FakeTool("add_memory")])}
    _patch_connect_with_session(monkeypatch, sessions)

    # Ensure _connect is never called for the unknown name.
    async def _no_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        raise AssertionError("_connect must not be called when there are no cold configs.")

    monkeypatch.setattr(MCPToolPool, "_connect", _no_connect)

    pool = MCPToolPool([_config_named("lisa-memory", warm=True)])
    await pool.start()
    resolved = await pool._resolve_server_for_bare_name("does_not_exist")
    assert resolved is None
    await pool.aclose()


@pytest.mark.asyncio
async def test_resolve_server_for_bare_name_lazy_connects_cold_server_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown bare name triggers a cold-server connect once, then re-indexes.

    After the cold server connects and re-indexes, its bare names enter the
    precomputed index, so a second resolution of the same name is an O(1)
    lookup with no further _connect calls.
    """
    started: list[str] = []
    cold_session = _FakeSession([_FakeTool("cold_only_tool")])

    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        started.append(config.server_id)
        self._sessions[config.server_id] = cold_session
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)

    # Warm server with a different tool; cold server has cold_only_tool.
    warm_session = _FakeSession([_FakeTool("warm_tool")])

    async def fake_start(self: MCPToolPool) -> None:
        # Only connect the warm server at start; cold stays unconnected.
        self._started = True
        from contextlib import AsyncExitStack

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        self._sessions["warm"] = warm_session
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "start", fake_start)

    pool = MCPToolPool([_config_named("warm"), _config_named("cold")])
    await pool.start()
    assert started == []  # cold not connected at start

    # First resolution: unknown bare name triggers cold connect.
    resolved = await pool._resolve_server_for_bare_name("cold_only_tool")
    assert resolved == "cold"
    assert started == ["cold"]  # cold connected exactly once

    # Second resolution: now in the precomputed index; no further connect.
    resolved_again = await pool._resolve_server_for_bare_name("cold_only_tool")
    assert resolved_again == "cold"
    assert started == ["cold"]  # no additional connect
    await pool.aclose()


@pytest.mark.asyncio
async def test_call_tool_bare_name_uses_precomputed_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """call_tool with a bare MCP tool name dispatches via the precomputed index.

    End-to-end: the bare-name fallback in call_tool resolves through
    _resolve_server_for_bare_name (precomputed) and routes to the owning
    server's session.
    """
    sessions = {"lisa-storage": _FakeSession([_FakeTool("read_model_manifest")])}
    _patch_connect_with_session(monkeypatch, sessions)

    pool = MCPToolPool([_config_named("lisa-storage", warm=True)])
    await pool.start()
    # Bare-name call (internal-caller style) routes to lisa-storage.
    result = await pool.call_tool("read_model_manifest", {})
    assert result.is_error is False
    assert result.output_text == "server:read_model_manifest"
    await pool.aclose()
