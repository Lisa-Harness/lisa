"""Lisa Context MCP server exposing surface-scoped memory tools."""

import json
import os
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from kernel.lite_memory import LiteMemoryStore

# Initialize the MCP Server
mcp = FastMCP("Lisa Context Server")

# Global or injected memory store. We'll use a local LiteMemoryStore for standalone testing
# if the full Azure blob backed MemoryStore isn't provided.
local_db = Path(
    os.getenv("LISA_MEMORIES_JSON_PATH", str(Path(tempfile.gettempdir()) / "lisa_memories.json"))
)
store = LiteMemoryStore(local_db)


@mcp.tool()
async def add_memory(
    user_slug: str,
    content: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
) -> str:
    """
    Store a new memory snippet scoped to the provided user identity and surface.
    Enforces Lisa's surface-scoped privacy isolation.
    """
    record = await store.record_memory(
        {
            "user_slug": user_slug,
            "content": content,
            "memory_scope": memory_scope,
            "workspace_id": workspace_id,
            "surface_id": surface_id,
            "user_id": user_slug,
            "source_user_id": user_slug,
            "source_user_slug": user_slug,
            "agent_id": "mcp-client",
            "operation_id": "mcp.add_memory",
            "instruction_id": "mcp.v1",
            "memory_tier": "daily",
            "content_json": {"source": "mcp_tool"},
        }
    )
    return f"Memory stored with ID {record.id}"

@mcp.tool()
async def search_memory(
    user_slug: str,
    query: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 5,
) -> str:
    """
    Search for surface-scoped memories for a specific user.
    Records from other surfaces or users cannot be accessed.
    """
    result = await store.recall(
        query=query,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit
    )
    if not result:
        return "No relevant memories found."
    return result

@mcp.tool()
async def get_surface_context(
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 20,
) -> str:
    """
    Retrieve the most recent memories for a specific user and surface context.
    """
    records = await store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit
    )
    if not records:
        return "No recent context."
    return "\n".join(f"- {r.content}" for r in records)


@mcp.tool()
async def list_memories(
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 20,
) -> str:
    """
    Returns a raw JSON array string of memory records.
    """
    records = await store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit
    )
    return json.dumps([r.model_dump(mode="json") for r in records])

if __name__ == "__main__":
    # If run directly as a script, start stdio server.
    mcp.run(transport="stdio")
