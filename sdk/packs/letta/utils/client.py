"""Thin Letta REST client wrapper used by the adapter server and operations.

This is deliberately a small surface over Letta's REST API, not a full
client SDK. It keeps the dependency footprint to ``httpx`` and is
injectable for tests via ``transport`` (see ``httpx.MockTransport``).

All methods are async and return parsed JSON. Errors are surfaced as
:class:`LettaError` with the request URL and status so callers can fail
cleanly without inspecting the transport.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = ["LettaClient", "LettaError", "LettaAgent", "LettaPassage"]


class LettaError(RuntimeError):
    """Raised when a Letta REST call fails."""


@dataclass(slots=True)
class LettaAgent:
    """Minimal projection of a Letta agent state record."""

    id: str
    name: str = ""
    agent_type: str = "memgpt_agent"


@dataclass(slots=True)
class LettaPassage:
    """Minimal projection of a Letta archival memory passage."""

    id: str
    text: str


class LettaClient:
    """Async wrapper over the Letta REST API.

    Args:
        base_url: Letta server root, e.g. ``http://localhost:8283`` or
            ``https://api.letta.com``. No trailing slash required.
        api_key: Bearer token. Empty string is allowed for local
            unauthenticated Letta deployments.
        timeout: Per-request timeout in seconds.
        transport: Optional ``httpx`` transport for test injection.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> LettaClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Agents ───────────────────────────────────────────────────────

    async def list_agents(self, *, name: str | None = None) -> list[dict[str, Any]]:
        """List agents, optionally filtered by name."""
        params: dict[str, Any] = {}
        if name:
            params["name"] = name
        result = await self._get("/v1/agents", params=params)
        return result if isinstance(result, list) else []

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Retrieve one agent by id."""
        result = await self._get(f"/v1/agents/{agent_id}")
        return result if isinstance(result, dict) else {}

    async def find_agent_by_name(self, name: str) -> dict[str, Any] | None:
        """Find an existing agent by name. Returns ``None`` if not found.

        Letta Cloud's free plan allows up to 3 agents; creating a new
        agent when one already exists by name wastes a slot and may hit
        a 402. Callers should look up before creating.
        """
        try:
            results = await self.list_agents(name=name)
        except LettaError:
            return None
        if not isinstance(results, list):
            return None
        for agent in results:
            if isinstance(agent, dict) and str(agent.get("name") or "") == name:
                return agent
        return None

    async def create_agent(self, *, name: str, **body: Any) -> dict[str, Any]:
        """Create a new Letta agent.

        Callers may pass arbitrary Letta ``POST /v1/agents`` body fields
        (``memory``, ``llm_config``, ``tools``, etc.). Only ``name`` is
        required here.
        """
        payload = {"name": name, **body}
        result = await self._post("/v1/agents", json=payload)
        return result if isinstance(result, dict) else {}

    # ── Messages ─────────────────────────────────────────────────────

    async def send_message(
        self,
        agent_id: str,
        *,
        message: str,
        role: str = "user",
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Send a user message to an agent and return the response payload."""
        payload = {
            "messages": [{"role": role, "content": message}],
            "streaming": streaming,
        }
        result = await self._post(f"/v1/agents/{agent_id}/messages", json=payload)
        return result if isinstance(result, dict) else {}

    async def summarize(self, agent_id: str) -> dict[str, Any]:
        """Trigger Letta message compaction/summarization for an agent."""
        result = await self._post(f"/v1/agents/{agent_id}/summarize", json={})
        return result if isinstance(result, dict) else {}

    # ── Archival memory (passages) ────────────────────────────────────

    async def create_passage(self, agent_id: str, *, text: str) -> dict[str, Any]:
        """Create one archival memory passage attached to an agent."""
        result = await self._post(
            f"/v1/agents/{agent_id}/archival-memory",
            json={"text": text},
        )
        return result if isinstance(result, dict) else {}

    async def list_passages(self, agent_id: str) -> list[dict[str, Any]]:
        """List archival memory passages for an agent."""
        result = await self._get(f"/v1/agents/{agent_id}/archival-memory")
        return result if isinstance(result, list) else []

    async def search_passages(self, agent_id: str, *, query: str) -> list[dict[str, Any]]:
        """Search an agent's archival memory."""
        result = await self._get(
            f"/v1/agents/{agent_id}/archival-memory/search",
            params={"query": query},
        )
        return result if isinstance(result, list) else []

    # ── Core memory (blocks) ─────────────────────────────────────────

    async def list_blocks(self) -> list[dict[str, Any]]:
        """List all memory blocks (core memory)."""
        result = await self._get("/v1/blocks")
        return result if isinstance(result, list) else []

    async def update_block(self, block_id: str, *, value: str) -> dict[str, Any]:
        """Update one core memory block's value."""
        result = await self._patch(f"/v1/blocks/{block_id}", json={"value": value})
        return result if isinstance(result, dict) else {}

    # ── HTTP plumbing ─────────────────────────────────────────────────

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.get(path, params=params)
        return self._unwrap(path, resp)

    async def _post(self, path: str, *, json: Any) -> Any:
        resp = await self._client.post(path, json=json)
        return self._unwrap(path, resp)

    async def _patch(self, path: str, *, json: Any) -> Any:
        resp = await self._client.patch(path, json=json)
        return self._unwrap(path, resp)

    @staticmethod
    def _unwrap(path: str, resp: httpx.Response) -> Any:
        if resp.is_success:
            if resp.status_code == 204 or not resp.content:
                return None
            return resp.json()
        raise LettaError(
            f"Letta {path} failed: HTTP {resp.status_code}: {resp.text[:200]}"
        )