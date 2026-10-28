"""Published-name bridge tests for capability servers in MCPToolPool.

A capability server (``config.capability`` set — the selected
implementation of e.g. ``storage``) publishes its tools under
``<capability>.<tool>`` instead of ``server_id.tool`` so the model-facing
ids are identical regardless of which implementation (Python, Go,
third-party) is live. ``tool_map`` maps published names to native wire
names for implementations whose native names differ.

Covers:

- catalog publishes ``storage.read_file`` for a winner named
  ``lisa-storage-go`` (implementation-independent ids);
- dispatch of a published id routes to the winner's session under the
  native wire name (tool_map translated back);
- a capability server WITH a tool_map publishes only the mapped
  contract; extra native tools stay internal;
- an ordinary server keeps ``server_id.tool`` unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPToolPool,
)


class _FakeResult:
    def __init__(self, tools: list[Any] | None = None, text: str = "") -> None:
        self.tools = tools or []
        self.content = [type("C", (), {"text": text})()] if text else []
        self.is_error = False


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.input_schema = {"type": "object", "properties": {}}
        self.annotations = None


class _FakeSession:
    def __init__(self, tools: list[_FakeTool]) -> None:
        self._tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> _FakeResult:
        return _FakeResult(tools=self._tools)

    async def call_tool(
        self, tool_id: str, arguments: dict[str, Any], **_kwargs: Any
    ) -> _FakeResult:
        self.calls.append((tool_id, arguments))
        return _FakeResult(text=f"server:{tool_id}")


def _capability_config(
    server_id: str,
    capability: str,
    capability_tools: tuple[str, ...],
    tool_map: dict[str, str] | None = None,
    capability_optional_tools: tuple[str, ...] = (),
):
    return MCPClientConfig(
        server_id=server_id,
        command="/bin/unused",
        args=[],
        capability=capability,
        capability_tools=capability_tools,
        capability_optional_tools=capability_optional_tools,
        tool_map=tool_map or {},
        warm=True,
    )


def _patch_connect(monkeypatch: pytest.MonkeyPatch, sessions: dict[str, _FakeSession]) -> None:
    async def fake_connect(self: MCPToolPool, config: MCPClientConfig) -> None:
        self._sessions[config.server_id] = sessions[config.server_id]
        await self._refresh_tool_index()

    monkeypatch.setattr(MCPToolPool, "_connect", fake_connect)


@pytest.mark.asyncio
async def test_capability_server_publishes_capability_prefixed_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Go winner publishes storage.* ids — same ids a Python winner would."""
    session = _FakeSession([_FakeTool("read_file"), _FakeTool("write_file")])
    _patch_connect(monkeypatch, {"lisa-storage-go": session})

    pool = MCPToolPool(
        [_capability_config("lisa-storage-go", "storage", ("read_file", "write_file"))],
    )
    await pool.start()

    tools = await pool.list_tools()
    published = sorted(tool.name for tool in tools)
    assert published == ["storage.read_file", "storage.write_file"]
    await pool.aclose()


@pytest.mark.asyncio
async def test_published_ids_dispatch_to_native_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatching storage.read_file calls the native read_file on the winner."""
    session = _FakeSession([_FakeTool("read_file")])
    _patch_connect(monkeypatch, {"lisa-storage-go": session})

    pool = MCPToolPool([_capability_config("lisa-storage-go", "storage", ("read_file",))])
    await pool.start()

    result = await pool.call_tool("storage.read_file", {})
    assert result.is_error is False
    assert session.calls == [("read_file", {})]
    await pool.aclose()


@pytest.mark.asyncio
async def test_tool_map_publishes_canonical_names_over_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """memory.add publishes over the native add_memory wire name."""
    session = _FakeSession([_FakeTool("add_memory"), _FakeTool("search_memory")])
    _patch_connect(
        monkeypatch,
        {"lisa-memory": session},
    )

    pool = MCPToolPool(
        [
            _capability_config(
                "lisa-memory",
                "memory",
                ("add", "search"),
                tool_map={"add": "add_memory", "search": "search_memory"},
            )
        ],
    )
    await pool.start()

    tools = await pool.list_tools()
    published = sorted(tool.name for tool in tools)
    assert published == ["memory.add", "memory.search"]

    result = await pool.call_tool("memory.add", {"content": "x"})
    assert result.is_error is False
    assert session.calls == [("add_memory", {"content": "x"})]
    await pool.aclose()


@pytest.mark.asyncio
async def test_tool_map_hides_unmapped_native_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capability server WITH a tool_map publishes only the mapped contract.

    ``retire_memory`` is a native tool with no published contract entry, so
    it must not leak into the model-facing catalog under the capability
    prefix (it stays reachable through the per-server index for internal
    scoped dispatch).
    """
    session = _FakeSession([_FakeTool("add_memory"), _FakeTool("retire_memory")])
    _patch_connect(
        monkeypatch,
        {"lisa-memory": session},
    )

    pool = MCPToolPool(
        [
            _capability_config(
                "lisa-memory",
                "memory",
                ("add",),
                tool_map={"add": "add_memory"},
            )
        ],
    )
    await pool.start()

    tools = await pool.list_tools()
    published = sorted(tool.name for tool in tools)
    assert published == ["memory.add"]
    await pool.aclose()


def test_capability_contract_hides_extra_and_rejects_missing_tools() -> None:
    pool = MCPToolPool([])
    config = _capability_config(
        "lisa-storage-go",
        "storage",
        ("read_file", "write_file"),
    )
    pool._merge_namespaced_catalog_tools(
        config,
        [_FakeToolSpec("read_file"), _FakeToolSpec("write_file"), _FakeToolSpec("debug")],
    )
    assert sorted(tool.name for tool in pool._tools) == [
        "storage.read_file",
        "storage.write_file",
    ]

    with pytest.raises(ValueError, match="missing required tools: write_file"):
        pool._merge_namespaced_catalog_tools(
            config,
            [_FakeToolSpec("read_file")],
        )


@pytest.mark.asyncio
async def test_plain_server_namespacing_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary servers keep server_id.tool ids (no capability set)."""
    session = _FakeSession([_FakeTool("check")])
    _patch_connect(monkeypatch, {"lisa-guard": session})

    pool = MCPToolPool([MCPClientConfig(server_id="lisa-guard", command="/bin/unused", args=[])])
    await pool.start()

    tools = await pool.list_tools()
    published = [tool.name for tool in tools]
    assert published == ["lisa-guard.check"]
    await pool.aclose()


class _FakeToolSpec:
    def __init__(self, name: str) -> None:
        self.server_id = "lisa-storage-go"
        self.name = name
        self.description = ""
        self.input_schema = {"type": "object", "properties": {}}
        self.read_only_hint = False


def _cold(config: MCPClientConfig) -> MCPClientConfig:
    return dataclasses.replace(config, warm=False)


@pytest.mark.asyncio
async def test_published_id_lazily_connects_cold_capability_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold owner with no ambient tools is reachable by its published id.

    ``fetch_page`` calls ``fetch.fetch_url`` on a cold server that is never
    connected while building a mode's tool list; the pool must connect the
    declared owner instead of reporting the tool as unregistered.
    """
    session = _FakeSession([_FakeTool("fetch_url"), _FakeTool("invalidate_fetch_cache")])
    _patch_connect(monkeypatch, {"lisa-fetch-go": session})
    pool = MCPToolPool(
        [
            _cold(
                _capability_config(
                    "lisa-fetch-go", "fetch", ("fetch_url", "invalidate_fetch_cache")
                )
            )
        ]
    )
    await pool.start()
    assert "lisa-fetch-go" not in pool._sessions

    result = await pool.call_tool("fetch.fetch_url", {"url": "https://example.com"})

    assert result.is_error is False
    assert session.calls == [("fetch_url", {"url": "https://example.com"})]
    await pool.aclose()


@pytest.mark.asyncio
async def test_undeclared_published_id_is_not_dispatched_to_capability_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An id outside the owner's declared contract is not dispatched to it."""
    session = _FakeSession([_FakeTool("fetch_url")])
    _patch_connect(monkeypatch, {"lisa-fetch-go": session})
    pool = MCPToolPool([_cold(_capability_config("lisa-fetch-go", "fetch", ("fetch_url",)))])
    await pool.start()

    result = await pool.call_tool("fetch.not_declared", {})

    assert result.is_error is True
    assert "not registered" in result.output_text
    assert session.calls == []
    await pool.aclose()
