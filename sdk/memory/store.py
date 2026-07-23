"""SQLite-backed memory store for the lisa-memory MCP server.

Extracted verbatim from the former ``lisa_context.server`` module: the
``SqliteMemoryStore`` plus its TTL helpers and row mapper. This is the
minimal built-in memory backend — three operations (add / search / list)
backed by a local SQLite file. Mem0 (or any other HTTP/stdio memory server)
serves the same three operations and more via the MCP tool pool.
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

from kernel.models import MemoryRecord, MemoryWriteRequest

__all__ = ["DEFAULT_TTL_DAYS", "EmbeddingProvider", "SqliteMemoryStore"]

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


def _expires_at(ttl_days: int | None) -> str | None:
    if ttl_days is None:
        return None
    return str((datetime.now(UTC).replace(microsecond=0) + _days(ttl_days)).isoformat())


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
    """SQLite-backed memory store owned by the lisa-memory MCP server."""

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
        expires_at = _expires_at(ttl_days)
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
        recall_pool = int(kwargs.get("recall_pool", 40) or 40)

        assert self._conn is not None

        tokens = {token.strip().lower() for token in query.split() if token.strip()}
        now_iso = _now_iso()

        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_slug = ? AND workspace_id = ? AND memory_scope = ? AND surface_id = ?
              AND (expires_at IS NULL OR expires_at >= ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_slug, workspace_id, memory_scope, surface_id, now_iso, recall_pool),
        ).fetchall()
        if not rows:
            return ""

        ranked: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _row_to_record(row)
            haystack_lower = (
                f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
            ).lower()
            if tokens:
                haystack_tokens = {tok for tok in haystack_lower.split()}
                overlap = len(tokens & haystack_tokens)
                # Normalize by both query size and content length to avoid
                # rewarding short content with a single matching keyword.
                content_word_count = max(len(haystack_tokens), 1)
                lexical = overlap / max(len(tokens), 1) * (1.0 - 1.0 / content_word_count)
            else:
                lexical = 1.0
            salience_boost = max(record.salience, 0.0) * 0.2
            # Small recency bias: newer records (higher id) get a slight edge,
            # bounded so it never overrides a strong lexical match.
            ranked.append((lexical + salience_boost, record))

        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        top = ranked[: max(1, min(limit, 8))]

        # Opportunistic access tracking: bump access_count and last_accessed_at
        # for the records we actually surfaced. Piggybacks on the recall call
        # — no separate maintenance pass needed.
        top_ids = [record.id for _, record in top]
        if top_ids:
            placeholders = ",".join("?" * len(top_ids))
            self._conn.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                (now_iso, *top_ids),
            )
            self._conn.commit()

        lines = [f"- {record.content}" for _, record in top]
        return "\n".join(lines)
