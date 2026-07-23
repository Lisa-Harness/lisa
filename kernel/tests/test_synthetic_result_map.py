"""Tests for result_map support in mcp_alias_tool synthetic aliases.

Covers:
- mcp_alias_tool with result_map transforms the output_text before returning.
- mcp_alias_tool without result_map passes output_text through unchanged.
- result_map is recorded on the SyntheticToolSpec for diagnostics.
"""

from __future__ import annotations

from typing import Any

import pytest

from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPToolPool,
    MCPToolSpec,
    namespaced_tool_name,
)
from kernel.tools.synthetic import mcp_alias_tool


def _config(server_id: str = "s") -> MCPClientConfig:
    return MCPClientConfig(
        server_id=server_id,
        command="python",
        args=["-m", "nonexistent.server"],
    )


class _FakeResult:
    def __init__(self, text: str = "hello world") -> None:
        self.content = [type("C", (), {"text": text})()]
        self.is_error = False


class _FakeSession:
    """Fake MCP session that returns a fixed result for any tool call."""

    def __init__(self, text: str = "hello world") -> None:
        self._text = text

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> _FakeResult:
        return _FakeResult(self._text)


def _patch_connect(monkeypatch: pytest.MonkeyPatch, text: str = "hello world") -> None:
    """Stub _connect to inject a fake session that returns the given text."""

    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        self._sessions[config.server_id] = _FakeSession(text)

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)


def _patch_refresh_with(
    monkeypatch: pytest.MonkeyPatch,
    server_tools: dict[str, list[tuple[str, str, dict[str, Any]]]],
) -> None:
    async def fake_refresh(self: MCPToolPool) -> None:
        self._tools = []
        self._tool_index = {}
        self._tool_index_by_server = {}
        self._bare_to_server_index = {}
        for config in self._configs:
            server_index: dict[str, MCPToolSpec] = {}
            for name, description, input_schema in server_tools.get(config.server_id, []):
                bare_spec = MCPToolSpec(
                    server_id=config.server_id,
                    name=name,
                    description=description,
                    input_schema=input_schema,
                )
                server_index[name] = bare_spec
                namespaced = namespaced_tool_name(config.server_id, name)
                namespaced_spec = MCPToolSpec(
                    server_id=config.server_id,
                    name=namespaced,
                    description=description,
                    input_schema=input_schema,
                )
                if namespaced not in self._tool_index:
                    self._tool_index[namespaced] = (config.server_id, namespaced_spec)
                    self._tools.append(namespaced_spec)
                if name not in self._bare_to_server_index:
                    self._bare_to_server_index[name] = config.server_id
            self._tool_index_by_server[config.server_id] = server_index
        self._reindex_synthetic_tools()

    monkeypatch.setattr(MCPToolPool, "_refresh_tool_index", fake_refresh)


@pytest.mark.asyncio
async def test_result_map_transforms_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """mcp_alias_tool with result_map transforms the output_text."""
    _patch_connect(monkeypatch, text="hello world")
    _patch_refresh_with(monkeypatch, {"s": [("real_search", "server search", {})]})

    pool = MCPToolPool([_config()])
    await pool.start()

    def upper_reverse(raw: str) -> str:
        return raw.upper()[::-1]

    alias = mcp_alias_tool(
        name="memory.search",
        description="test alias",
        pool=pool,
        target_tool="real_search",
        target_server_id="s",
        result_map=upper_reverse,
    )
    pool.add_synthetic_tools([alias])

    result = await pool.call_tool("memory.search", {"query": "x"})
    assert result.is_error is False
    # "hello world" -> upper = "HELLO WORLD" -> reverse = "DLROW OLLEH"
    assert result.output_text == "DLROW OLLEH"
    assert alias.result_map is upper_reverse
    await pool.aclose()


@pytest.mark.asyncio
async def test_no_result_map_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """mcp_alias_tool without result_map returns output_text unchanged."""
    _patch_connect(monkeypatch, text="raw output")
    _patch_refresh_with(monkeypatch, {"s": [("real_list", "server list", {})]})

    pool = MCPToolPool([_config()])
    await pool.start()

    alias = mcp_alias_tool(
        name="memory.list",
        description="test alias",
        pool=pool,
        target_tool="real_list",
        target_server_id="s",
    )
    pool.add_synthetic_tools([alias])

    result = await pool.call_tool("memory.list", {})
    assert result.output_text == "raw output"
    assert alias.result_map is None
    await pool.aclose()
