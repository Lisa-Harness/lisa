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
                result = await session.call_tool("add_memory", arguments={
                    "user_slug": write.user_slug,
                    "content": write.content,
                    "memory_scope": write.memory_scope,
                    "workspace_id": write.workspace_id,
                    "surface_id": write.surface_id,
                })
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
                    valid_until=""
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
                result = await session.call_tool("search_memory", arguments={
                    "user_slug": user_slug,
                    "query": query,
                    "memory_scope": memory_scope,
                    "workspace_id": workspace_id,
                    "surface_id": surface_id,
                    "limit": limit
                })
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
                result = await session.call_tool("list_memories", arguments={
                    "user_slug": user_slug,
                    "memory_scope": memory_scope,
                    "workspace_id": workspace_id,
                    "surface_id": surface_id,
                    "limit": limit
                })
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
