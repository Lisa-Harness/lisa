"""Lisa Memory MCP server — the minimal built-in memory backend.

Owns the SQLite-backed memory store (extracted from the former
``lisa_context`` server). Exposes five tools:

- ``add_memory`` — store a scoped memory snippet
- ``search_memory`` — keyword recall for one user
- ``list_memories`` — raw JSON array of recent memory records
- ``get_memory`` — fetch a single record by id
- ``delete_memory`` — remove a single record by id

This is the default backend behind the generic ``memory.add`` /
``memory.search`` / ``memory.list`` / ``memory.get`` / ``memory.delete``
synthetic aliases. Mem0 (or any other HTTP/stdio memory server) serves the
same five operations and more via the MCP tool pool; modes that want Mem0's
full surface declare the server's tools directly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from kernel.models import MemoryWriteRequest
from sdk.memory.store import SqliteMemoryStore

__all__ = ["MemoryDeps", "init_store", "main", "mcp", "set_deps_for_test"]

logger = logging.getLogger(__name__)


# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class MemoryDeps:
    """Server-scoped dependencies owned by the lifespan."""

    store: SqliteMemoryStore


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: MemoryDeps | None = None


def set_deps_for_test(deps: MemoryDeps) -> None:
    """Test-only seam: inject ``MemoryDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> MemoryDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, MemoryDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise ToolError("lisa-memory is not initialized. Call set_deps_for_test(deps).")
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _MemoryConfig:
    data_dir: Path = Path()


_cfg = _MemoryConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[MemoryDeps]:
    """Construct the memory store from resolved CLI config and yield it.

    Tears down the SQLite connection on exit so the file handle is released
    cleanly when the stdio session ends.
    """
    deps = MemoryDeps(store=SqliteMemoryStore(db_path=_cfg.data_dir / "memory.db"))
    try:
        yield deps
    finally:
        deps.store.close()


mcp = MCPServer("Lisa Memory Server", lifespan=_lifespan)


def init_store(data_dir: Path) -> None:
    """Initialize the memory store from a resolved data directory.

    Kept as a thin back-compat shim for tests that call ``init_store``
    directly. Tests should prefer :func:`set_deps_for_test` with a
    :class:`MemoryDeps` instance.
    """
    set_deps_for_test(MemoryDeps(store=SqliteMemoryStore(db_path=data_dir / "memory.db")))


@mcp.tool(annotations=ToolAnnotations())
async def add_memory(
    user_slug: str,
    content: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    memory_tier: str = "daily",
    operation_id: str = "mcp.add_memory",
    instruction_id: str = "mcp.v1",
    content_json: dict[str, Any] | None = None,
    salience: float = 0.0,
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Store a new memory snippet scoped to the provided user identity and surface.

    Enforces Lisa's surface-scoped privacy isolation.

    Optional metadata fields (``memory_tier``, ``operation_id``,
    ``instruction_id``, ``content_json``, ``salience``) let callers control
    retention, provenance, and ranking. When omitted, sensible defaults
    are applied so the tool remains usable as a simple "store this" call.

    Args:
        user_slug: The user identity to scope the memory to.
        content: The memory text to store.
        memory_scope: Privacy scope (``personal``, ``shared``, ``channel``).
        workspace_id: Workspace isolation key.
        surface_id: Surface isolation key (e.g. channel id).
        memory_tier: Retention tier (``daily``, ``weekly``, ``long_term``,
            ``permanent``). Controls TTL expiry.
        operation_id: Provenance tag identifying the operation that
            created this memory.
        instruction_id: Provenance tag for the instruction version.
        content_json: Structured metadata dict stored alongside content.
        salience: Ranking weight (0.0-1.0) for recall ordering.
    """
    store = _resolve_deps(ctx).store
    record = await store.record_memory(
        MemoryWriteRequest(
            user_slug=user_slug,
            content=content,
            memory_scope=memory_scope,  # type: ignore[arg-type]
            workspace_id=workspace_id,
            surface_id=surface_id,
            operation_id=operation_id,
            instruction_id=instruction_id,
            memory_tier=memory_tier,  # type: ignore[arg-type]
            content_json=content_json or {"source": "mcp_tool"},
            salience=salience,
        )
    )
    return f"Memory stored with ID {record.id}"


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def search_memory(
    user_slug: str,
    query: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 5,
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Search for surface-scoped memories for a specific user.

    Records from other surfaces or users cannot be accessed.

    The recall engine uses lexical-overlap ranking with a salience boost.
    When no query tokens match any stored content, the most recent records
    are returned (ordered by id DESC) so the caller always gets context.
    Returns ``"No relevant memories found."`` when the scoped store is empty.
    """
    store = _resolve_deps(ctx).store
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


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def get_memory(
    memory_id: int | str,
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Fetch a single memory record by id, scoped to the caller's surface.

    Returns the record as a JSON object. Raises :class:`ToolError` when no
    record matches the id within the caller's surface scope. Cross-surface
    reads are impossible — the scope filters are applied server-side.
    """
    store = _resolve_deps(ctx).store
    record = await store.get_memory(
        memory_id=memory_id,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
    )
    if record is None:
        raise ToolError("No memory found.")
    return json.dumps(record.model_dump(mode="json"), default=str)


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def delete_memory(
    memory_id: int | str,
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Delete a single memory record by id, scoped to the caller's surface.

    Returns ``"Deleted memory <id>."`` on success. Raises :class:`ToolError`
    when no record matched the id within the caller's surface scope.
    Cross-surface deletion is impossible — the scope filters are applied
    server-side.
    """
    store = _resolve_deps(ctx).store
    deleted = await store.delete_memory(
        memory_id=memory_id,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
    )
    if not deleted:
        raise ToolError("No memory found.")
    return f"Deleted memory {memory_id}."


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_memories(
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    limit: int = 20,
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Returns a raw JSON array string of memory records."""
    store = _resolve_deps(ctx).store
    records = await store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
    )
    return json.dumps([r.model_dump(mode="json") for r in records], default=str)


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the store."""
    _cfg.data_dir = Path(args.data_dir)
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    set_deps_for_test(MemoryDeps(store=SqliteMemoryStore(db_path=_cfg.data_dir / "memory.db")))


def main() -> None:
    """Entry point for the lisa-memory MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper.
    ``--data-dir`` is required; the store's SQLite file is derived from it.
    """
    from mcp_server.config.cli import run

    run("lisa-memory", mcp, _init, supports_readonly=True)


if __name__ == "__main__":
    main()
