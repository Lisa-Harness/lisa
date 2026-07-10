"""Lisa Memory MCP server — the minimal built-in memory backend.

Owns the SQLite-backed memory store (extracted from the former
``lisa_context`` server). Exposes four tools:

- ``add_memory`` — store a scoped memory snippet
- ``search_memory`` — keyword recall for one user
- ``list_memories`` — raw JSON array of recent memory records
- ``get_surface_context`` — newline-formatted recent context

This is the default backend behind the generic ``memory.add`` /
``memory.search`` / ``memory.list`` synthetic aliases. Mem0 (or any other
HTTP/stdio memory server) serves the same three operations and more via
the MCP tool pool; modes that want Mem0's full surface declare the
server's tools directly.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel.models import MemoryWriteRequest
from sdk.lisa_memory.store import SqliteMemoryStore

__all__ = ["get_store", "init_store", "main", "mcp"]

logger = logging.getLogger(__name__)

mcp = FastMCP("Lisa Memory Server")
_store: Any | None = None


def get_store(data_dir: Path) -> SqliteMemoryStore:
    """Build the memory store from a resolved data directory.

    The db path is derived exclusively from ``--data-dir``
    (via ``data_dir / "memory.db"``); no env-var fallback, no hardcoded default.
    """
    store: SqliteMemoryStore = SqliteMemoryStore(db_path=data_dir / "memory.db")
    return store


def _store_instance() -> SqliteMemoryStore:
    global _store
    if _store is None:
        raise RuntimeError(
            "lisa_memory store is not initialized. "
            "Call init_store(data_dir) from main() before serving tools."
        )
    store: SqliteMemoryStore = _store
    return store


def init_store(data_dir: Path) -> None:
    """Initialize the module-level store from a resolved data directory.

    Called by :func:`main` after parsing ``--data-dir``. Tests may call this
    directly with a ``tmp_path`` to pin the store without spawning the server.
    """
    global _store
    _store = get_store(data_dir)


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
    """Entry point for the lisa-memory MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--data-dir`` is required; the store's SQLite file is derived from it.
    """
    from sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-memory")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-memory")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-memory starting (data_dir=%s)", data_dir)

    init_store(data_dir)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
