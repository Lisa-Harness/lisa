"""Tests for the generic synthetic-tool mechanism in MCPToolPool.

Covers:
- SyntheticToolSpec registration via add_synthetic_tools().
- list_tools() includes synthetic tools alongside MCP-discovered tools.
- call_tool() dispatches synthetic tools to their in-process handler.
- Synthetic tools survive refresh_tool_index() (they are host-owned).
- aclose() clears synthetic handlers.
- Name collisions follow first-registered-wins (with a synthetic entry).
- mcp_alias_tool() redirects a synthetic name to an MCP server tool.
- Handler exceptions surface as error results.

These tests stub MCPToolPool._connect so no real subprocess is spawned.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import pytest

from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPToolPool,
    MCPToolSpec,
    SYNTHETIC_SERVER_ID,
    SyntheticToolSpec,
)
from kernel.tools.synthetic import mcp_alias_tool


# ── Helpers ─────────────────────────────────────────────────────────────────


def _config(server_id: str = "s") -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
    )


def _patch_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub _connect so no subprocess is spawned; just record a session."""

    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        self._sessions[config.server_id] = object()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)


def _patch_refresh_with(
    monkeypatch: pytest.MonkeyPatch,
    server_tools: dict[str, list[tuple[str, str, dict[str, Any]]]],
) -> None:
    """Replace _refresh_tool_index with one that seeds from a static map.

    server_tools maps server_id -> list of (name, description, input_schema).
    Synthetic tools are re-seeded after the MCP index is built.
    """

    async def fake_refresh(self: MCPToolPool) -> None:
        self._tools = []
        self._tool_index = {}
        self._tool_index_by_server = {}
        for config in self._configs:
            server_index: dict[str, MCPToolSpec] = {}
            for name, description, input_schema in server_tools.get(config.server_id, []):
                spec = MCPToolSpec(
                    server_id=config.server_id,
                    name=name,
                    description=description,
                    input_schema=input_schema,
                )
                server_index[name] = spec
                if name not in self._tool_index:
                    self._tool_index[name] = (config.server_id, spec)
                    self._tools.append(spec)
            self._tool_index_by_server[config.server_id] = server_index
        self._reindex_synthetic_tools()

    monkeypatch.setattr(MCPToolPool, "_refresh_tool_index", fake_refresh)


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthetic_tool_listed_and_dispatched(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"s": []})

    pool = MCPToolPool([_config()])
    await pool.start()

    async def handler(args: dict[str, Any]) -> str:
        return f"echo:{args.get('x', '')}"

    pool.add_synthetic_tools([
        SyntheticToolSpec(
            name="memory.search",
            description="Search memory.",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            handler=handler,
        )
    ])

    tools = await pool.list_tools()
    names = [t.name for t in tools]
    assert "memory.search" in names
    assert any(t.server_id == SYNTHETIC_SERVER_ID for t in tools)

    result = await pool.call_tool("memory.search", {"x": "hello"})
    assert result.is_error is False
    assert result.output_text == "echo:hello"
    await pool.aclose()


@pytest.mark.asyncio
async def test_synthetic_tool_survives_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"s": []})

    pool = MCPToolPool([_config()])
    await pool.start()

    async def handler(args: dict[str, Any]) -> str:
        return "ok"

    pool.add_synthetic_tools([
        SyntheticToolSpec(name="memory.add", description="d", input_schema={}, handler=handler)
    ])
    await pool.refresh_tool_index()

    tools = await pool.list_tools()
    assert "memory.add" in [t.name for t in tools]
    result = await pool.call_tool("memory.add", {})
    assert result.output_text == "ok"
    await pool.aclose()


@pytest.mark.asyncio
async def test_synthetic_aclose_clears_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"s": []})

    pool = MCPToolPool([_config()])
    await pool.start()

    async def handler(args: dict[str, Any]) -> str:
        return "x"

    pool.add_synthetic_tools([
        SyntheticToolSpec(name="t", description="d", input_schema={}, handler=handler)
    ])
    assert len(pool._synthetic_handlers) == 1
    await pool.aclose()
    assert pool._synthetic_handlers == {}


@pytest.mark.asyncio
async def test_synthetic_collision_first_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"s": [("shared", "from server", {})]})

    pool = MCPToolPool([_config()])
    await pool.start()

    async def synthetic_handler(args: dict[str, Any]) -> str:
        return "synthetic"

    # The server's "shared" tool is already in the index; synthetic loses.
    pool.add_synthetic_tools([
        SyntheticToolSpec(name="shared", description="d", input_schema={}, handler=synthetic_handler)
    ])
    # "shared" should route to the server, not the synthetic handler.
    assert "shared" not in pool._synthetic_handlers
    await pool.aclose()


@pytest.mark.asyncio
async def test_mcp_alias_tool_redirects_to_server_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"s": [("real_search", "server search", {})]})

    pool = MCPToolPool([_config()])
    await pool.start()

    # Inject a fake session so call_tool's MCP path doesn't hit a subprocess.

    class _FakeResult:
        content = [type("C", (), {"text": "server:real_search"})()]
        is_error = False

    class _FakeSession:
        async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:  # type: ignore[no-untyped-def]
            assert tool_id == "real_search"
            assert arguments.get("user_id") == "default"
            return _FakeResult()

    pool._sessions["s"] = _FakeSession()  # type: ignore[assignment]

    alias = mcp_alias_tool(
        name="memory.search",
        description="generic alias",
        pool=pool,
        target_tool="real_search",
        arg_map=lambda a: {**a, "user_id": "default"},
    )
    pool.add_synthetic_tools([alias])

    result = await pool.call_tool("memory.search", {"query": "hi"})
    assert result.is_error is False
    assert result.output_text == "server:real_search"
    await pool.aclose()


@pytest.mark.asyncio
async def test_synthetic_tool_handler_error_returns_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"s": []})

    pool = MCPToolPool([_config()])
    await pool.start()

    async def failing_handler(args: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    pool.add_synthetic_tools([
        SyntheticToolSpec(name="bad", description="d", input_schema={}, handler=failing_handler)
    ])
    result = await pool.call_tool("bad", {})
    assert result.is_error is True
    assert "boom" in result.output_text
    await pool.aclose()


# ── Server-scoped dispatch ─────────────────────────────────────────────────


def _config_named(server_id: str) -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
    )


@pytest.mark.asyncio
async def test_call_tool_scoped_routes_to_named_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """call_tool(server_id=...) bypasses bare-name collision and routes to
    the named server, even when another server registered the same tool name
    first in the merged index."""
    _patch_connect(monkeypatch)
    # Both servers expose "add_memory"; first-registered ("lisa-memory") wins
    # the bare-name index. Scoped dispatch must still reach "mem0".
    _patch_refresh_with(
        monkeypatch,
        {
            "lisa-memory": [("add_memory", "local add", {})],
            "mem0": [("add_memory", "cloud add", {})],
        },
    )

    pool = MCPToolPool([_config_named("lisa-memory"), _config_named("mem0")])
    await pool.start()

    seen: list[str] = []

    class _FakeResult:
        content = [type("C", (), {"text": "ok"})()]
        is_error = False

    class _FakeSession:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:  # type: ignore[no-untyped-def]
            seen.append(self.owner)
            return _FakeResult()

    pool._sessions["lisa-memory"] = _FakeSession("lisa-memory")  # type: ignore[assignment]
    pool._sessions["mem0"] = _FakeSession("mem0")  # type: ignore[assignment]

    # Bare-name call hits lisa-memory (first-registered-wins).
    await pool.call_tool("add_memory", {})
    assert seen == ["lisa-memory"]

    # Scoped call to mem0 bypasses the collision and routes there.
    await pool.call_tool("add_memory", {}, server_id="mem0")
    assert seen == ["lisa-memory", "mem0"]

    # Scoped call to lisa-memory also routes there despite losing nothing.
    await pool.call_tool("add_memory", {}, server_id="lisa-memory")
    assert seen == ["lisa-memory", "mem0", "lisa-memory"]
    await pool.aclose()


@pytest.mark.asyncio
async def test_call_tool_scoped_unknown_server_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_connect(monkeypatch)
    _patch_refresh_with(monkeypatch, {"lisa-memory": [("add_memory", "d", {})]})

    pool = MCPToolPool([_config_named("lisa-memory")])
    await pool.start()

    result = await pool.call_tool("add_memory", {}, server_id="nope")
    assert result.is_error is True
    assert "nope" in result.output_text
    await pool.aclose()


@pytest.mark.asyncio
async def test_call_tool_scoped_lazily_starts_cold_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scoped call to a cold server that isn't connected yet lazily starts
    it (via _connect) before dispatching. This is the mem0-as-default path:
    mem0 is cold and not started at pool.start() time; the scoped memory.add
    triggers _connect on first dispatch."""

    started: list[str] = []

    class _FakeResult:
        content = [type("C", (), {"text": "stored"})()]
        is_error = False

    class _FakeSession:
        async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:  # type: ignore[no-untyped-def]
            return _FakeResult()

        async def list_tools(self) -> Any:
            from types import SimpleNamespace
            return SimpleNamespace(tools=[
                type("T", (), {"name": "add_memory", "description": "cloud", "inputSchema": {}})()
            ])

    # A refresh that only indexes CONNECTED servers (mirrors real behavior),
    # so mem0 (not connected at start) has no tools until lazily started.
    async def fake_refresh(self: MCPToolPool) -> None:
        self._tools = []
        self._tool_index = {}
        self._tool_index_by_server = {}
        for config in self._configs:
            session = self._sessions.get(config.server_id)
            if session is None:
                continue
            result = await session.list_tools()
            server_index: dict[str, MCPToolSpec] = {}
            for tool in result.tools:
                spec = MCPToolSpec(
                    server_id=config.server_id,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {},
                )
                server_index[spec.name] = spec
                if spec.name not in self._tool_index:
                    self._tool_index[spec.name] = (config.server_id, spec)
                    self._tools.append(spec)
            self._tool_index_by_server[config.server_id] = server_index
        self._reindex_synthetic_tools()

    monkeypatch.setattr(MCPToolPool, "_refresh_tool_index", fake_refresh)

    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        started.append(config.server_id)
        self._sessions[config.server_id] = _FakeSession()
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)

    pool = MCPToolPool([_config_named("mem0")])
    # Override start so it does NOT eagerly connect cold servers — simulate the
    # "mem0 not started at init" reality. Warm path is empty here.
    async def fake_start(self: MCPToolPool) -> None:
        self._started = True
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "start", fake_start)
    await pool.start()
    assert "mem0" not in pool._sessions  # not connected at start
    assert pool._tool_index_by_server == {}  # no tools indexed yet

    result = await pool.call_tool("add_memory", {}, server_id="mem0")
    assert started == ["mem0"]  # lazily connected on first scoped call
    assert result.is_error is False
    assert result.output_text == "stored"
    await pool.aclose()


@pytest.mark.asyncio
async def test_mcp_alias_tool_target_server_id_routes_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mcp_alias_tool(target_server_id=...) forwards via scoped dispatch, so
    an alias can pin to mem0 even when lisa-memory owns the bare name."""
    _patch_connect(monkeypatch)
    _patch_refresh_with(
        monkeypatch,
        {
            "lisa-memory": [("add_memory", "local", {})],
            "mem0": [("add_memory", "cloud", {})],
        },
    )

    pool = MCPToolPool([_config_named("lisa-memory"), _config_named("mem0")])
    await pool.start()

    routed: list[str] = []

    class _FakeResult:
        content = [type("C", (), {"text": "cloud:added"})()]
        is_error = False

    class _FakeSession:
        def __init__(self, owner: str) -> None:
            self.owner = owner

        async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:  # type: ignore[no-untyped-def]
            routed.append(self.owner)
            return _FakeResult()

    pool._sessions["lisa-memory"] = _FakeSession("lisa-memory")  # type: ignore[assignment]
    pool._sessions["mem0"] = _FakeSession("mem0")  # type: ignore[assignment]

    alias = mcp_alias_tool(
        name="memory.add",
        description="generic alias",
        pool=pool,
        target_tool="add_memory",
        target_server_id="mem0",
    )
    pool.add_synthetic_tools([alias])

    result = await pool.call_tool("memory.add", {"content": "hi"})
    assert result.is_error is False
    assert result.output_text == "cloud:added"
    assert routed == ["mem0"]
    await pool.aclose()