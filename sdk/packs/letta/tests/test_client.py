"""Tests for the Letta REST client wrapper using httpx.MockTransport."""

from __future__ import annotations

import pytest
import httpx

from sdk.packs.letta.utils.client import LettaClient, LettaError


def _client(transport: httpx.MockTransport, *, api_key: str = "test-key") -> LettaClient:
    return LettaClient(base_url="http://letta.local", api_key=api_key, transport=transport)


@pytest.mark.asyncio
async def test_create_agent_returns_json() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"id": "agent-1", "name": "lisa-agent"})
    )
    async with _client(transport) as client:
        agent = await client.create_agent(name="lisa-agent")
    assert agent["id"] == "agent-1"


@pytest.mark.asyncio
async def test_create_agent_raises_on_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(400, text="bad request"))
    async with _client(transport) as client:
        with pytest.raises(LettaError, match="HTTP 400"):
            await client.create_agent(name="x")


@pytest.mark.asyncio
async def test_send_message_posts_messages() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["json"] = req.read().decode()
        return httpx.Response(200, json={"messages": [{"message_type": "assistant_message", "content": "hi"}]})

    transport = httpx.MockTransport(handler)
    async with _client(transport) as client:
        resp = await client.send_message("agent-1", message="hello")
    assert resp["messages"][0]["content"] == "hi"
    assert "hello" in str(captured["json"])


@pytest.mark.asyncio
async def test_create_passage_posts_text() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["json"] = req.read().decode()
        return httpx.Response(200, json={"id": "pass-1", "text": "fact"})

    transport = httpx.MockTransport(handler)
    async with _client(transport) as client:
        passage = await client.create_passage("agent-1", text="fact")
    assert passage["id"] == "pass-1"
    assert "fact" in str(captured["json"])


@pytest.mark.asyncio
async def test_search_passages_returns_list() -> None:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=[{"id": "p1", "text": "a"}])
    )
    async with _client(transport) as client:
        results = await client.search_passages("agent-1", query="q")
    assert isinstance(results, list)
    assert results[0]["id"] == "p1"


@pytest.mark.asyncio
async def test_summarize_posts_to_summarize_endpoint() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        return httpx.Response(200, json={"num_messages_before": 10, "num_messages_after": 2})

    transport = httpx.MockTransport(handler)
    async with _client(transport) as client:
        result = await client.summarize("agent-1")
    assert result["num_messages_before"] == 10
    assert "summarize" in str(captured["url"])


@pytest.mark.asyncio
async def test_auth_header_set_when_api_key_provided() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("Authorization")
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async with _client(transport, api_key="secret") as client:
        await client.list_agents()
    assert captured["auth"] == "Bearer secret"


@pytest.mark.asyncio
async def test_no_auth_header_when_api_key_empty() -> None:
    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["auth"] = req.headers.get("Authorization")
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    async with _client(transport, api_key="") as client:
        await client.list_agents()
    assert captured["auth"] is None


@pytest.mark.asyncio
async def test_204_returns_none() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(204))
    async with _client(transport) as client:
        result = await client._get("/v1/agents/agent-1")
    assert result is None