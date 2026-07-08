"""Lisa Context MCP server — surface-scoped memory + context brokerage.

Owns the SQLite-backed memory store and the context broker logic. Tools:

- ``add_memory`` — store a scoped memory snippet
- ``search_memory`` — semantic/keyword recall for one user
- ``list_memories`` — raw JSON array of recent memory records
- ``get_surface_context`` — newline-formatted recent context
- ``broker_context`` — query-aware context brokerage
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mcp.server.fastmcp import FastMCP

from kernel.models import (
    ContextBrokerRequest,
    ContextBrokerResponse,
    MemoryRecord,
    MemoryWriteRequest,
    UnifiedState,
)
from lisa_sdk.lisa_context.context_broker import (
    render_context_packet_markdown,
    select_ranked_context,
)

__all__ = ["get_store", "main", "mcp"]

logger = logging.getLogger(__name__)

DEFAULT_TTL_DAYS: dict[str, int | None] = {
    "daily": 7,
    "weekly": 30,
    "long_term": 365,
    "permanent": None,
}


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generate embeddings for text content."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _entry_id() -> str:
    return uuid.uuid4().hex[:12]


def _days(n: int) -> Any:
    from datetime import timedelta

    return timedelta(days=n)


def _expires_at(tier: str, ttl_days: int | None) -> str | None:
    if ttl_days is None:
        return None
    return (datetime.now(UTC).replace(microsecond=0) + _days(ttl_days)).isoformat()


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    raw_json = row["content_json"] or "{}"
    try:
        content_json = json.loads(raw_json) if isinstance(raw_json, str) else dict(raw_json or {})
    except (json.JSONDecodeError, TypeError):
        content_json = {}
    user_slug = row["user_slug"]
    return MemoryRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        user_id=user_slug,
        user_slug=user_slug,
        memory_scope=row["memory_scope"],
        surface_id=row["surface_id"],
        source_user_id=user_slug,
        source_user_slug=user_slug,
        agent_id=row["agent_id"],
        operation_id=row["operation_id"],
        instruction_id=row["instruction_id"],
        memory_tier=row["memory_tier"],
        content=row["content"],
        content_json=content_json,
        salience=row["salience"],
        created_at=row["created_at"],
        access_count=row["access_count"],
        last_accessed_at=row["last_accessed_at"],
        valid_until=row["expires_at"],
    )


@dataclass
class SqliteMemoryStore:
    """SQLite-backed memory store owned by the MCP context server."""

    db_path: str | Path
    embedding_provider: EmbeddingProvider | None = None
    embedding_dimension: int = 1536
    default_ttl_days: dict[str, int | None] = field(default_factory=lambda: dict(DEFAULT_TTL_DAYS))

    _conn: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._conn = conn
        self._init_schema()

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL UNIQUE,
                user_slug TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT 'default',
                memory_scope TEXT NOT NULL DEFAULT 'personal',
                surface_id TEXT NOT NULL DEFAULT '',
                agent_id TEXT NOT NULL DEFAULT 'core',
                operation_id TEXT NOT NULL DEFAULT '',
                instruction_id TEXT NOT NULL DEFAULT '',
                memory_tier TEXT NOT NULL DEFAULT 'long_term',
                content TEXT NOT NULL,
                content_json TEXT DEFAULT '{}',
                salience REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(user_slug, workspace_id, memory_scope, surface_id);

            CREATE INDEX IF NOT EXISTS idx_memories_operation
                ON memories(user_slug, operation_id);

            CREATE INDEX IF NOT EXISTS idx_memories_expires
                ON memories(expires_at) WHERE expires_at IS NOT NULL;
            """
        )

    async def record_memory(self, request: MemoryWriteRequest | Any) -> MemoryRecord:
        write = (
            request
            if isinstance(request, MemoryWriteRequest)
            else MemoryWriteRequest.model_validate(request)
        )
        assert self._conn is not None
        entry_id = _entry_id()
        created_at = _now_iso()
        ttl_days = self.default_ttl_days.get(write.memory_tier, DEFAULT_TTL_DAYS["daily"])
        expires_at = _expires_at(write.memory_tier, ttl_days)
        self._conn.execute(
            """
            INSERT INTO memories (
                entry_id, user_slug, workspace_id, memory_scope, surface_id, agent_id,
                operation_id, instruction_id, memory_tier, content, content_json,
                salience, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                write.user_slug,
                write.workspace_id,
                write.memory_scope,
                write.surface_id,
                write.agent_id,
                write.operation_id,
                write.instruction_id,
                write.memory_tier,
                write.content,
                json.dumps(write.content_json or {}, default=str),
                write.salience,
                created_at,
                expires_at,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM memories WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to persist memory record")
        return _row_to_record(row)

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[MemoryRecord]:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_slug = ? AND workspace_id = ? AND memory_scope = ? AND surface_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_slug, workspace_id, memory_scope, surface_id, limit),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    async def recall(self, query: str, **kwargs: Any) -> str:
        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        workspace_id = str(kwargs.get("workspace_id") or "default").strip() or "default"
        memory_scope = str(kwargs.get("memory_scope") or "personal").strip() or "personal"
        surface_id = str(kwargs.get("surface_id") or "").strip()
        limit = int(kwargs.get("limit", 5) or 5)

        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_slug = ? AND workspace_id = ? AND memory_scope = ? AND surface_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_slug, workspace_id, memory_scope, surface_id, limit),
        ).fetchall()
        if not rows:
            return ""
        tokens = {token.strip().lower() for token in query.split() if token.strip()}
        ranked: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _row_to_record(row)
            haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
            overlap = len(tokens.intersection({tok.lower() for tok in haystack.split()}))
            lexical = overlap / max(len(tokens), 1) if tokens else 1.0
            ranked.append((lexical + max(record.salience, 0.0) * 0.2, record))
        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        top = ranked[: max(1, min(limit, 8))]
        lines = [f"- {record.content}" for _, record in top]
        return "\n".join(lines)


def get_store(data_dir: Path) -> Any:
    """Build the memory store from a resolved data directory.

    The context server always owns the SQLite-backed store. The db path is
    derived exclusively from ``--data-dir`` (via ``data_dir / "memory.db"``);
    no env-var fallback, no hardcoded default.
    """
    return SqliteMemoryStore(db_path=data_dir / "memory.db")


mcp = FastMCP("Lisa Context Server")
_store: Any | None = None


def _store_instance() -> Any:
    global _store
    if _store is None:
        raise RuntimeError(
            "lisa_context store is not initialized. "
            "Call init_store(data_dir) from main() before serving tools."
        )
    return _store


def init_store(data_dir: Path) -> None:
    """Initialize the module-level store from a resolved data directory.

    Called by :func:`main` after parsing ``--data-dir``. Tests may call this
    directly with a ``tmp_path`` to pin the store without spawning the server.
    """
    global _store
    _store = get_store(data_dir)


def _state_from_request(payload: dict[str, Any]) -> UnifiedState:
    state_payload = payload.get("state")
    if isinstance(state_payload, UnifiedState):
        return state_payload
    if isinstance(state_payload, dict):
        return UnifiedState.model_validate(state_payload)
    return UnifiedState(compiled_at="", context_markdown="")


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


@mcp.tool()
async def broker_context(payload: dict[str, Any]) -> str:
    """Return a brokered context packet and markdown for the current turn."""
    request = ContextBrokerRequest.model_validate(payload)
    state = _state_from_request(payload)
    packet = select_ranked_context(
        state=state,
        query=request.query,
        mode_id=request.mode_id,
        stage_id=request.stage_id,
        connectors=request.connectors or None,
    )
    response = ContextBrokerResponse(
        packet=packet,
        packet_markdown=render_context_packet_markdown(packet),
        context_tag=request.context_budget.get("context_tag", ""),
    )
    return response.model_dump_json()


def main() -> None:
    """Entry point for the lisa-context MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper, derives
    ``memory.db`` from ``--data-dir``, initializes the store, and runs the
    server. No env-var path fallback.
    """
    from lisa_sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-context", supports_readonly=True)
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-context")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-context starting (data_dir=%s, readonly=%s)", data_dir, args.readonly)

    init_store(data_dir)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
