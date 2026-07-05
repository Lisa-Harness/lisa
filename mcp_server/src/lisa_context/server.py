"""Lisa Context MCP server — surface-scoped memory tools.

Exposes the kernel's memory store (``SqliteMemoryStore`` by default,
``LiteMemoryStore`` fallback via env var) over MCP. Tools:

- ``add_memory`` — store a scoped memory snippet
- ``search_memory`` — semantic/keyword recall for one user
- ``list_memories`` — raw JSON array of recent memory records
- ``get_surface_context`` — newline-formatted recent context
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel.lite_memory import LiteMemoryStore
from kernel.sqlite_memory import SqliteMemoryStore

__all__ = ["mcp", "main", "get_store"]


def get_store() -> Any:
    """Build the memory store from environment configuration.

    Defaults to ``SqliteMemoryStore`` (production-grade). Set
    ``LISA_MEMORY_BACKEND=lite`` to use ``LiteMemoryStore`` for local testing.
    """
    backend = os.getenv("LISA_MEMORY_BACKEND", "sqlite").lower()
    if backend == "lite":
        db_path = Path(
            os.getenv("LISA_MEMORIES_JSON_PATH", str(Path(tempfile.gettempdir()) / "lisa_memories.json"))
        )
        return LiteMemoryStore(db_path)
    db_path = Path(os.getenv("LISA_MEMORY_DB_PATH", "lsa-state/memory.db"))
    return SqliteMemoryStore(db_path=db_path)


mcp = FastMCP("Lisa Context Server")
_store: Any | None = None


def _store_instance() -> Any:
    global _store
    if _store is None:
        _store = get_store()
    return _store


@mcp.tool()
async def add_memory(
    user_slug: str,
    content: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
) -> str:
    """Store a new memory snippet scoped to the provided user identity and surface.

    Enforces Lisa's surface-scoped privacy isolation.
    """
    from kernel.models import MemoryWriteRequest

    store = _store_instance()
    record = await store.record_memory(
        MemoryWriteRequest(
            user_slug=user_slug,
            content=content,
            memory_scope=memory_scope,  # type: ignore[arg-type]
            workspace_id=workspace_id,
            surface_id=surface_id,
            operation_id="mcp.add_memory",
            instruction_id="mcp.v1",
            memory_tier="daily",
            content_json={"source": "mcp_tool"},
        )
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
    """Search for surface-scoped memories for a specific user.

    Records from other surfaces or users cannot be accessed.
    """
    store = _store_instance()
    result = await store.recall(
        query=query,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
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
    """Retrieve the most recent memories for a specific user and surface context."""
    store = _store_instance()
    records = await store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
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
    """Returns a raw JSON array string of memory records."""
    store = _store_instance()
    records = await store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
    )
    return json.dumps([r.model_dump(mode="json") for r in records], default=str)


def main() -> None:
    """Entry point for the lisa-context MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()