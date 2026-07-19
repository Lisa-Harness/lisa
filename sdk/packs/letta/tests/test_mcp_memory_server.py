"""Tests for the Letta memory adapter MCP server."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

import sdk.packs.letta.mcp_memory_server as server_mod


def _mock_transport(routes: list[tuple[str, str, httpx.Response]]) -> httpx.MockTransport:
    """Return a MockTransport that dispatches by (method, path-substring).

    ``routes`` is a list of ``(method, needle, response)`` tuples. The
    first matching route wins. This is needed because the adapter makes
    two calls (create agent + create passage) whose URLs both contain
    ``/v1/agents`` but differ by method and tail path.
    """
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        for method, needle, resp in routes:
            if req.method == method and needle in url:
                return resp
        return httpx.Response(404, text=f"no mock for {req.method} {url}")
    return httpx.MockTransport(handler)


def _init(tmp_path: Path, transport: httpx.MockTransport) -> None:
    server_mod.init_server(
        tmp_path,
        base_url="http://letta.local",
        api_key="test-key",
        default_agent_name="lisa-agent",
        transport=transport,
    )


@pytest.mark.asyncio
async def test_add_memory_creates_agent_then_passage(tmp_path: Path) -> None:
    # Order matters: the archival-memory route must come before /v1/agents
    # because the archival URL (http://letta.local/v1/agents/agent-1/archival-memory)
    # contains "/v1/agents" as a substring.
    transport = _mock_transport([
        ("POST", "/archival-memory", httpx.Response(200, json={"id": "pass-1", "text": "durable fact"})),
        ("POST", "/v1/agents", httpx.Response(200, json={"id": "agent-1", "name": "lisa-agent"})),
    ])
    _init(tmp_path, transport)
    out = await server_mod.add_memory(user_slug="u", content="durable fact")
    data = json.loads(out)
    assert data["id"] == "pass-1"
    assert data["text"] == "durable fact"


@pytest.mark.asyncio
async def test_add_memory_reuses_mapped_agent(tmp_path: Path) -> None:
    # First call to seed the identity map
    transport = _mock_transport([
        ("POST", "/archival-memory", httpx.Response(200, json={"id": "pass-1", "text": "f"})),
        ("POST", "/v1/agents", httpx.Response(200, json={"id": "agent-1", "name": "lisa-agent"})),
    ])
    _init(tmp_path, transport)
    await server_mod.add_memory(user_slug="u", content="first")
    # Second call: no agent creation should happen (only archival posts)
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        if "archival-memory" in str(req.url):
            return httpx.Response(200, json={"id": "pass-2", "text": "second"})
        return httpx.Response(200, json={"id": "agent-1"})

    server_mod.init_server(
        tmp_path, base_url="http://letta.local", api_key="k",
        default_agent_name="lisa-agent", transport=httpx.MockTransport(handler),
    )
    out = await server_mod.add_memory(user_slug="u", content="second")
    data = json.loads(out)
    assert data["id"] == "pass-2"
    # No POST to /v1/agents (create) on the second call — only archival-memory
    create_calls = [c for c in calls if "/v1/agents" in c and "archival" not in c]
    assert not create_calls


@pytest.mark.asyncio
async def test_search_memory_returns_passages(tmp_path: Path) -> None:
    transport = _mock_transport([
        ("GET", "/archival-memory/search", httpx.Response(200, json=[{"id": "p1", "text": "found"}])),
        ("POST", "/v1/agents", httpx.Response(200, json={"id": "agent-1", "name": "lisa-agent"})),
    ])
    _init(tmp_path, transport)
    out = await server_mod.search_memory(query="test", user_slug="u")
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["text"] == "found"


@pytest.mark.asyncio
async def test_list_memories_returns_passages(tmp_path: Path) -> None:
    transport = _mock_transport([
        ("GET", "/archival-memory", httpx.Response(200, json=[{"id": "p1", "text": "recent"}])),
        ("POST", "/v1/agents", httpx.Response(200, json={"id": "agent-1", "name": "lisa-agent"})),
    ])
    _init(tmp_path, transport)
    out = await server_mod.list_memories(user_slug="u")
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["id"] == "p1"


@pytest.mark.asyncio
async def test_add_memory_surfaces_error_envelope(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="letta down"))
    _init(tmp_path, transport)
    out = await server_mod.add_memory(user_slug="u", content="x")
    data = json.loads(out)
    assert "error" in data


@pytest.mark.asyncio
async def test_not_initialized_raises(tmp_path: Path) -> None:
    # Reset module state
    server_mod._client = None
    server_mod._identity = None
    with pytest.raises(RuntimeError, match="not initialized"):
        await server_mod.add_memory(user_slug="u", content="x")