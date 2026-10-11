"""Lisa Memory MCP server — the minimal built-in memory backend.

Owns the SQLite-backed memory store (extracted from the former
``lisa_context`` server). Exposes five tools:

- ``add_memory`` — store a scoped memory snippet
- ``search_memory`` — keyword recall for one user
- ``list_memories`` — raw JSON array of recent memory records
- ``get_memory`` — fetch a single record by id
- ``delete_memory`` — remove a single record by id
- ``retire_memory`` — mark a memory retired
- ``count_memories`` — count memory records matching scope and filters

This is the default backend behind the generic ``memory.add`` /
``memory.search`` / ``memory.list`` / ``memory.get`` / ``memory.delete`` /
``memory.retire`` / ``memory.count`` synthetic aliases. Mem0 (or any other
HTTP/stdio memory server) serves the
same five operations and more via the MCP tool pool; modes that want Mem0's
full surface declare the server's tools directly.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.models import MemoryWriteRequest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from sdk.layout import DATA_DIR
from sdk.memory.backup import DEFAULT_KEEP_BACKUPS, backup_memory_db
from sdk.memory.embeddings import OpenAIEmbeddingProvider
from sdk.memory.store import SqliteMemoryStore

from mcp_server.memory.embeddings import resolve_embedding_config

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
    embedding_provider: OpenAIEmbeddingProvider | None = None
    embedding_dimension: int = 1536
    backup_keep: int = DEFAULT_KEEP_BACKUPS


_cfg = _MemoryConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[MemoryDeps]:
    """Construct the memory store from resolved CLI config and yield it.

    When an embedding model is configured (``--embedding-model`` /
    ``LISA_EMBEDDING_MODEL``), the store is constructed with an
    :class:`OpenAIEmbeddingProvider` so semantic recall populates ``memory_vec``.
    When absent, the store stays lexical-only (no vector chunks) — the
    previous default behavior.

    Tears down the SQLite connection on exit so the file handle is released
    cleanly when the stdio session ends.
    """
    deps = MemoryDeps(
        store=SqliteMemoryStore(
            db_path=_cfg.data_dir / DATA_DIR / "memory.db",
            embedding_provider=_cfg.embedding_provider,
            embedding_dimension=_cfg.embedding_dimension,
        )
    )
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
    set_deps_for_test(
        MemoryDeps(
            store=SqliteMemoryStore(
                db_path=data_dir / DATA_DIR / "memory.db",
                embedding_provider=_cfg.embedding_provider,
                embedding_dimension=_cfg.embedding_dimension,
            )
        )
    )


@mcp.tool(annotations=ToolAnnotations())
async def add_memory(
    user_slug: str,
    content: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    *,
    memory_tier: str = "daily",
    memory_kind: str = "episodic",
    operation_id: str = "mcp.add_memory",
    instruction_id: str = "mcp.v1",
    content_json: dict[str, Any] | None = None,
    salience: float = 0.0,
    supersedes_id: int | None = None,
    item_ids: list[int] | None = None,
    confidence: float = 1.0,
    valid_until: str | None = None,
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Store a new memory snippet scoped to the provided user identity and surface.

    Enforces Lisa's surface-scoped privacy isolation.

    Curation contract — memory is ONLY what is true beyond this thread
    (the thread log already records what happened). Do NOT store:
    - ephemeral tool/MCP errors, registration gaps, session drops;
    - iteration-budget pauses or execution-flow state;
    - point-in-time volatile infrastructure status (query it live instead).
    DO store durable user preferences, stable architectural invariants,
    verified problem/solution pairs, and permanent project configuration.

    Optional metadata fields (``memory_tier``, ``memory_kind``,
    ``operation_id``, ``instruction_id``, ``content_json``, ``salience``,
    ``supersedes_id``, ``item_ids``, ``confidence``, ``valid_until``)
    let callers control retention, provenance, ranking, and consolidation
    supersession. When omitted, sensible defaults are applied so the tool
    remains usable as a simple "store this" call.

    Args:
        user_slug: The user identity to scope the memory to.
        content: The memory text to store.
        memory_scope: Privacy scope (``personal``, ``shared``, ``channel``).
        workspace_id: Workspace isolation key.
        surface_id: Surface isolation key (e.g. channel id).
        memory_tier: Retention tier (``daily``, ``weekly``, ``long_term``,
            ``permanent``). Controls TTL expiry.
        memory_kind: Cognitive kind (``episodic``, ``semantic``,
            ``procedural``, ``reflection``, ``preference``).
        operation_id: Provenance tag identifying the operation that
            created this memory.
        instruction_id: Provenance tag for the instruction version.
        content_json: Structured metadata dict stored alongside content.
        salience: Ranking weight (0.0-1.0) for recall ordering.
        supersedes_id: Id of the record this one supersedes (consolidation).
        item_ids: Ids of the records this one was derived from.
        confidence: Confidence in the memory (0.0-1.0).
        valid_until: Optional semantic-validity bound (ISO 8601); overrides
            the tier TTL when set.
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
            memory_kind=memory_kind,  # type: ignore[arg-type]
            content_json=content_json or {"source": "mcp_tool"},
            salience=salience,
            supersedes_id=supersedes_id,
            item_ids=item_ids or [],
            confidence=confidence,
            valid_until=valid_until,
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
    *,
    limit: int = 5,
    memory_kind: str | None = None,
    status: str | None = "active",
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Search for surface-scoped memories for a specific user.

    Records from other surfaces or users cannot be accessed.

    The recall engine uses lexical-overlap ranking with a salience boost,
    falling back to semantic KNN when embeddings are available. When no
    query tokens match any stored content, the most recent records are
    returned (ordered by id DESC) so the caller always gets context.
    Returns a JSON array of memory records (the canonical
    ``list[MemoryRecord]`` shape, matching ``list_memories`` /
    ``get_memory``) so the ``recall_memory`` operation can ``json.loads``
    the result. Returns ``"[]"`` when the scoped store is empty.

    ``status`` defaults to ``"active"`` so retired/superseded records
    drop out of recall automatically; pass ``None`` to include all.
    """
    store = _resolve_deps(ctx).store
    records = await store.recall_records(
        query=query,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
        memory_kind=memory_kind,
        status=status,
    )
    if not records:
        return "[]"
    return json.dumps([r.model_dump(mode="json") for r in records], default=str)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def get_memory(
    memory_id: int | str,
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    *,
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
    *,
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
    *,
    limit: int = 20,
    memory_kind: str | None = None,
    status: str | None = "active",
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Returns a raw JSON array string of memory records.

    ``status`` defaults to ``"active"`` so retired records are excluded;
    pass ``None`` to list all statuses. ``memory_kind`` filters by
    ``episodic``/``semantic``/… when given.
    """
    store = _resolve_deps(ctx).store
    records = await store.list_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        limit=limit,
        memory_kind=memory_kind,
        status=status,
    )
    return json.dumps([r.model_dump(mode="json") for r in records], default=str)


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True))
async def retire_memory(
    memory_id: int | str,
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    *,
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Retire a memory record by id, scoped to the caller's surface.

    Retirement (not deletion) preserves the audit trail — the record
    stays queryable with ``status=None`` but drops out of the default
    ``status="active"`` recall pool. Used by consolidation to retire
    episodic originals after a semantic successor is written.

    Returns ``"Retired memory <id>."`` on success. Raises
    :class:`ToolError` when no active record matched the id within the
    caller's surface scope.
    """
    store = _resolve_deps(ctx).store
    retired = await store.retire_memory(
        memory_id=memory_id,
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
    )
    if not retired:
        raise ToolError("No active memory found.")
    return f"Retired memory {memory_id}."


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def count_memories(
    user_slug: str,
    memory_scope: str = "personal",
    workspace_id: str = "default",
    surface_id: str = "",
    *,
    memory_kind: str | None = None,
    status: str | None = "active",
    ctx: Context[MemoryDeps, Any] | None = None,
) -> str:
    """Return the count of memory records matching the scope and filters.

    ``status`` defaults to ``"active"`` so retired records are excluded;
    pass ``None`` to count all statuses. ``memory_kind`` filters by
    ``episodic``/``semantic``/… when given.
    """
    store = _resolve_deps(ctx).store
    count = await store.count_memories(
        user_slug=user_slug,
        workspace_id=workspace_id,
        memory_scope=memory_scope,
        surface_id=surface_id,
        memory_kind=memory_kind,
        status=status,
    )
    return str(count)


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the store."""
    _cfg.data_dir = Path(args.data_dir)
    provider, dimension = resolve_embedding_config(args)
    _cfg.embedding_provider = provider
    _cfg.embedding_dimension = dimension
    _cfg.backup_keep = args.backup_keep
    # Point-in-time backup before the store opens the DB: dump memory.db to a
    # timestamped SQL file under <data_dir>/backups/ and prune beyond keep.
    backup_memory_db(data_dir=_cfg.data_dir, keep=_cfg.backup_keep)
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    set_deps_for_test(
        MemoryDeps(
            store=SqliteMemoryStore(
                db_path=_cfg.data_dir / DATA_DIR / "memory.db",
                embedding_provider=_cfg.embedding_provider,
                embedding_dimension=_cfg.embedding_dimension,
            )
        )
    )


def _add_memory_args(parser: Any) -> None:
    """Add lisa-memory-specific CLI args for embedding configuration."""
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Embedding model id for semantic recall (e.g. "
            "'nomic-embed-text'). When omitted, the store stays lexical-only. "
            "Also settable via LISA_EMBEDDING_MODEL."
        ),
    )
    parser.add_argument(
        "--embedding-endpoint",
        type=str,
        default=None,
        help=(
            "OpenAI-compatible embeddings base URL (e.g. "
            "'http://localhost:11434/v1'). Required when --embedding-model "
            "is set. Also settable via LISA_EMBEDDING_ENDPOINT."
        ),
    )
    parser.add_argument(
        "--backup-keep",
        type=int,
        default=_backup_keep_default(),
        help=(
            "Number of timestamped memory.db dumps to retain under "
            "<data-dir>/backups/ (default 10). Also settable via "
            "LISA_MEMORY_BACKUP_KEEP."
        ),
    )


def _backup_keep_default() -> int:
    """Resolve the backup retention count from env, defaulting to 10."""
    raw = os.getenv("LISA_MEMORY_BACKUP_KEEP", "").strip()
    if not raw:
        return DEFAULT_KEEP_BACKUPS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("invalid LISA_MEMORY_BACKUP_KEEP %r; using %d", raw, DEFAULT_KEEP_BACKUPS)
        return DEFAULT_KEEP_BACKUPS


def main() -> None:
    """Entry point for the lisa-memory MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper.
    ``--data-dir`` is required; the store's SQLite file is derived from it.
    ``--embedding-model`` / ``--embedding-endpoint`` optionally wire an
    :class:`OpenAIEmbeddingProvider` for semantic recall.
    """
    from mcp_server.config.cli import run

    run("lisa-memory", mcp, _init, supports_readonly=True, add_args=_add_memory_args)


if __name__ == "__main__":
    main()
