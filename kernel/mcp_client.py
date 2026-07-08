from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from kernel.models import MemoryRecord, MemoryWriteRequest


class MCPMemoryClient:
    """Uses MCP to communicate with the Context Server."""

    def __init__(
        self,
        command: str,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        # Hosts may pass env overrides for behavioral config (API keys,
        # feature flags). Path-like config is passed via CLI args
        # (e.g. --data-dir). When env is None the subprocess inherits the
        # parent env unchanged.
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )

    async def record_memory(self, request: MemoryWriteRequest | Any) -> MemoryRecord:
        if isinstance(request, MemoryWriteRequest):
            write = request
        else:
            write = MemoryWriteRequest.model_validate(request)

        async with stdio_client(self.server_params) as (read, write_stream):
            async with ClientSession(read, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "add_memory",
                    arguments={
                        "user_slug": write.user_slug,
                        "content": write.content,
                        "memory_scope": write.memory_scope,
                        "workspace_id": write.workspace_id,
                        "surface_id": write.surface_id,
                    },
                )
                # The MCP server right now returns a string "Memory stored with ID X".
                # For compatibility, we can mock the rest of the MemoryRecord if needed.
                return MemoryRecord(
                    id=1,
                    user_slug=write.user_slug,
                    content=write.content,
                    content_json={},
                    memory_scope=write.memory_scope,
                    workspace_id=write.workspace_id,
                    surface_id=write.surface_id,
                    created_at="now",
                    memory_tier="daily",
                    salience=0.5,
                    agent_id="mcp",
                    source_user_id="",
                    source_user_slug="",
                    operation_id="",
                    instruction_id="",
                    memory_kind="episodic",
                    confidence=1.0,
                    evidence_ids=[],
                    supersedes_id=0,
                    valid_until="",
                )

    async def recall(self, query: str, **kwargs: Any) -> str:
        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        workspace_id = str(kwargs.get("workspace_id") or "default").strip() or "default"
        memory_scope = str(kwargs.get("memory_scope") or "personal").strip() or "personal"
        surface_id = str(kwargs.get("surface_id") or "").strip()
        limit = int(kwargs.get("limit", 5) or 5)

        async with stdio_client(self.server_params) as (read, write_stream):
            async with ClientSession(read, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_memory",
                    arguments={
                        "user_slug": user_slug,
                        "query": query,
                        "memory_scope": memory_scope,
                        "workspace_id": workspace_id,
                        "surface_id": surface_id,
                        "limit": limit,
                    },
                )
                if result.content and len(result.content) > 0:
                    return result.content[0].text
                return ""

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        async with stdio_client(self.server_params) as (read, write_stream):
            async with ClientSession(read, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "list_memories",
                    arguments={
                        "user_slug": user_slug,
                        "memory_scope": memory_scope,
                        "workspace_id": workspace_id,
                        "surface_id": surface_id,
                        "limit": limit,
                    },
                )
                if result.content and len(result.content) > 0:
                    import json

                    text = result.content[0].text
                    data = json.loads(text)
                    return [MemoryRecord.model_validate(r) for r in data]
                return []

    async def list_recent_memories(self, **kwargs: Any) -> list[MemoryRecord]:
        return []

    async def store(self, query: str, response: str) -> None:
        """Compatibility helper used only by lightweight tests/examples."""
        content = f"Q: {query}\nA: {response}".strip()
        if not content:
            return
        await self.record_memory(
            MemoryWriteRequest(
                user_slug="default",
                memory_scope="personal",
                operation_id="lite_store",
                instruction_id="lite_store.v1",
                memory_tier="daily",
                content=content,
                content_json={"query": query},
                salience=0.2,
            )
        )


class MCPHttpMemoryClient:
    """Generic MemoryStore backed by any HTTP MCP memory server.

    Same :class:`~kernel.memory_protocol.MemoryStore` surface as
    :class:`MCPMemoryClient`, but speaks to a remote MCP endpoint over the
    streamable-HTTP transport. Backend-agnostic: the tool names for
    add/search/list are passed in by the caller (from the manifest), so this
    client works for Mem0 cloud, any future HTTP memory server, or a
    self-hosted endpoint without code changes. No backend-specific knowledge
    lives here.

    Each call opens a fresh MCP HTTP session; the host owns no long-lived
    subprocess (there is none — the endpoint is remote). Scoped by
    ``user_id`` (mapped from ``user_slug``) so multi-user isolation works
    the same way it does for lisa-context.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        tool_names: dict[str, str] | None = None,
        agent_id: str = "lisa",
    ) -> None:
        self._url = url
        self._agent_id = agent_id
        self._headers = dict(headers or {})
        # Tool names default to the generic aliases themselves so a manifest
        # that doesn't override them still works against a server whose tools
        # are literally named add/search/list.
        self._tool_names = {
            "add": "add_memory",
            "search": "search_memories",
            "list": "get_memories",
        }
        if tool_names:
            self._tool_names.update(tool_names)

    def _session(self):
        """Async context manager yielding an initialized streamable-HTTP MCP session.

        The streamable_http_client + ClientSession lifetimes must live in one
        cancel scope; returning a (session, stack) tuple and closing it later
        breaks anyio task/scope tracking and cancels the session mid-call. So
        this returns a proper async context manager — callers use
        ``async with self._session() as session:`` and the full lifecycle is
        scoped to the ``async with`` block.
        """
        from contextlib import asynccontextmanager

        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        @asynccontextmanager
        async def _cm():
            async with streamablehttp_client(self._url, headers=self._headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

        return _cm()

    async def record_memory(self, request: MemoryWriteRequest | Any) -> MemoryRecord:
        write = (
            request
            if isinstance(request, MemoryWriteRequest)
            else MemoryWriteRequest.model_validate(request)
        )
        async with self._session() as session:
            await session.call_tool(
                self._tool_names["add"],
                arguments={
                    "content": write.content,
                    "user_id": write.user_slug,
                    "agent_id": self._agent_id,
                    "metadata": {
                        "memory_scope": write.memory_scope,
                        "workspace_id": write.workspace_id,
                        "surface_id": write.surface_id,
                        "operation_id": write.operation_id,
                        "instruction_id": write.instruction_id,
                        "memory_tier": write.memory_tier,
                        "memory_kind": write.memory_kind,
                        "salience": write.salience,
                    },
                },
            )
        # The server stores its own IDs; echo a MemoryRecord back so the
        # kernel's downstream typing (MemoryStore Protocol) does not change.
        return MemoryRecord(
            id=0,
            user_slug=write.user_slug,
            content=write.content,
            content_json=dict(write.content_json or {}),
            memory_scope=write.memory_scope,
            workspace_id=write.workspace_id,
            surface_id=write.surface_id,
            created_at="now",
            memory_tier=write.memory_tier,
            salience=write.salience,
            agent_id=self._agent_id,
            source_user_id="",
            source_user_slug="",
            operation_id=write.operation_id,
            instruction_id=write.instruction_id,
            memory_kind=write.memory_kind,
            confidence=write.confidence,
            evidence_ids=[],
            supersedes_id=None,
            valid_until=None,
        )

    async def recall(self, query: str, **kwargs: Any) -> str:
        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        limit = int(kwargs.get("limit", 5) or 5)
        async with self._session() as session:
            result = await session.call_tool(
                self._tool_names["search"],
                arguments={
                    "query": query,
                    "user_id": user_slug,
                    "limit": limit,
                },
            )
        if result.content and len(result.content) > 0:
            return result.content[0].text
        return ""

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        async with self._session() as session:
            result = await session.call_tool(
                self._tool_names["list"],
                arguments={"user_id": user_slug, "limit": limit},
            )
        if not (result.content and len(result.content) > 0):
            return []
        import json as _json

        text = result.content[0].text
        try:
            data = _json.loads(text) if isinstance(text, str) else text
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, list):
            return []
        records: list[MemoryRecord] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") or {}
            records.append(
                MemoryRecord(
                    id=int(item.get("id") or index),
                    user_slug=user_slug,
                    content=str(item.get("memory") or item.get("content") or ""),
                    content_json=dict(metadata),
                    memory_scope=str(metadata.get("memory_scope", memory_scope)),
                    workspace_id=str(metadata.get("workspace_id", workspace_id)),
                    surface_id=str(metadata.get("surface_id", surface_id)),
                    created_at=str(item.get("created_at") or "now"),
                    memory_tier=str(metadata.get("memory_tier", "daily")),
                    salience=float(metadata.get("salience", 0.0)),
                    agent_id=str(item.get("agent_id") or self._agent_id),
                    source_user_id="",
                    source_user_slug="",
                    operation_id=str(metadata.get("operation_id", "")),
                    instruction_id=str(metadata.get("instruction_id", "")),
                    memory_kind="episodic",
                    confidence=1.0,
                    evidence_ids=[],
                    supersedes_id=None,
                    valid_until=None,
                )
            )
        return records
