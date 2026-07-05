"""Production-grade memory store backed by SQLite + sqlite-vec.

Implements the ``MemoryStore`` Protocol from ``kernel.seams.protocols``. Ships
as the default ``ContextSeams.memory_store``; ``LiteMemoryStore`` stays as the
zero-dependency fallback for local testing.

Features
--------
- Vector embeddings for semantic search (via the sqlite-vec extension)
- Keyword fallback for when embeddings are unavailable
- TTL-based expiration (daily, weekly, long_term, permanent tiers)
- Multi-tenant isolation (user_slug x workspace_id x memory_scope x surface_id)
- Salience-weighted recall ordering
- Automatic embedding generation via a pluggable ``EmbeddingProvider``
- JSON content column for structured memory payloads

The store degrades gracefully: if the sqlite-vec extension cannot be loaded
(or no ``embedding_provider`` is supplied), recall falls back to keyword search
over ``memories.content``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kernel.models import MemoryRecord, MemoryWriteRequest

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProvider",
    "SqliteMemoryStore",
    "DEFAULT_TTL_DAYS",
    "load_sqlite_vec",
]


DEFAULT_TTL_DAYS: dict[str, int | None] = {
    "daily": 7,
    "weekly": 30,
    "long_term": 365,
    "permanent": None,
}
"""Days until expiration per memory tier. ``None`` means never expires."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generate embeddings for text content."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


def load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into ``conn``.

    Returns ``True`` when the extension is available and loaded, ``False``
    when it is missing. The store uses the return value to decide whether
    semantic search is available or whether to fall back to keyword mode.
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError:
        logger.info("sqlite-vec not installed; memory store runs in keyword-only mode")
        return False
    try:
        conn.enable_load_extension(True)  # type: ignore[attr-defined]
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)  # type: ignore[attr-defined]
        return True
    except (AttributeError, sqlite3.OperationalError) as exc:
        logger.warning("sqlite-vec extension load failed: %s", exc)
        return False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _entry_id() -> str:
    return uuid.uuid4().hex[:12]


def _expires_at(tier: str, ttl_days: int | None) -> str | None:
    if ttl_days is None:
        return None
    return (datetime.now(UTC).replace(microsecond=0) + _days(ttl_days)).isoformat()


def _days(n: int) -> Any:
    from datetime import timedelta

    return timedelta(days=n)


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
    """Production-grade memory store backed by SQLite + sqlite-vec."""

    db_path: str | Path
    embedding_provider: EmbeddingProvider | None = None
    embedding_dimension: int = 1536
    default_ttl_days: dict[str, int | None] = field(default_factory=lambda: dict(DEFAULT_TTL_DAYS))

    _conn: sqlite3.Connection | None = field(default=None, init=False, repr=False)
    _vec_available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._vec_available = load_sqlite_vec(conn)
        self._conn = conn
        self._init_schema()

    # ── Schema ──────────────────────────────────────────────────────────

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
        if self._vec_available:
            dim = int(self.embedding_dimension)
            self._conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_embeddings
                    USING vec0(
                        memory_id INTEGER PRIMARY KEY,
                        embedding FLOAT[{dim}]
                    )
                """
            )
        self._conn.commit()

    # ── MemoryStore Protocol ────────────────────────────────────────────

    async def record_memory(self, request: MemoryWriteRequest | Any) -> MemoryRecord:
        if isinstance(request, MemoryWriteRequest):
            write = request
        else:
            write = MemoryWriteRequest.model_validate(request)

        tier = write.memory_tier if write.memory_tier in self.default_ttl_days else "long_term"
        ttl = self.default_ttl_days.get(tier)
        expires_at = _expires_at(tier, ttl)
        created = _now_iso()
        entry_id = _entry_id()
        content_json_str = json.dumps(write.content_json, ensure_ascii=False)

        assert self._conn is not None
        cur = self._conn.execute(
            """
            INSERT INTO memories (
                entry_id, user_slug, workspace_id, memory_scope, surface_id,
                agent_id, operation_id, instruction_id, memory_tier,
                content, content_json, salience, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                write.user_slug,
                write.workspace_id,
                write.memory_scope,
                write.surface_id,
                write.agent_id or "core",
                write.operation_id,
                write.instruction_id,
                tier,
                write.content,
                content_json_str,
                float(write.salience),
                created,
                expires_at,
            ),
        )
        memory_id = int(cur.lastrowid)
        self._conn.commit()

        if self._vec_available and self.embedding_provider is not None and write.content:
            await self._store_embedding(memory_id, write.content)

        return MemoryRecord(
            id=memory_id,
            workspace_id=write.workspace_id,
            user_id=write.user_id,
            user_slug=write.user_slug,
            memory_scope=write.memory_scope,
            surface_id=write.surface_id,
            source_user_id=write.source_user_id,
            source_user_slug=write.source_user_slug,
            agent_id=write.agent_id or "core",
            operation_id=write.operation_id,
            instruction_id=write.instruction_id,
            memory_tier=tier,
            content=write.content,
            content_json=write.content_json,
            salience=float(write.salience),
            created_at=created,
            valid_until=expires_at,
        )

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
            WHERE user_slug = ? AND workspace_id = ? AND memory_scope = ?
              AND (? = '' OR surface_id = ?)
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_slug, workspace_id, memory_scope, surface_id, surface_id, _now_iso(), int(limit)),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    async def list_recent_memories(
        self,
        *,
        user_slug: str,
        operation_id: str = "",
        limit: int = 20,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> list[MemoryRecord]:
        records = await self.list_memories(
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            limit=limit,
        )
        if not operation_id:
            return records
        return [record for record in records if record.operation_id == operation_id]

    async def recall(self, query: str, **kwargs: Any) -> str:
        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        workspace_id = str(kwargs.get("workspace_id") or "default").strip() or "default"
        memory_scope = str(kwargs.get("memory_scope") or "personal").strip() or "personal"
        surface_id = str(kwargs.get("surface_id") or "").strip()
        limit = int(kwargs.get("limit", 5) or 5)
        budget_ms = float(kwargs.get("budget_ms", 200) or 200)

        if self._vec_available and self.embedding_provider is not None and query.strip():
            records = await self._semantic_recall(
                query=query,
                user_slug=user_slug,
                workspace_id=workspace_id,
                memory_scope=memory_scope,
                surface_id=surface_id,
                limit=limit,
                budget_ms=budget_ms,
            )
        else:
            records = await self._keyword_recall(
                query=query,
                user_slug=user_slug,
                workspace_id=workspace_id,
                memory_scope=memory_scope,
                surface_id=surface_id,
                limit=limit,
            )
        if not records:
            return ""
        lines = [f"- {record.content}" for record in records if record.content]
        return "\n".join(lines)

    # ── Maintenance ──────────────────────────────────────────────────────

    async def evict_expired(self) -> int:
        assert self._conn is not None
        now = _now_iso()
        cur = self._conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        deleted = int(cur.rowcount or 0)
        if self._vec_available and deleted:
            self._conn.execute(
                """
                DELETE FROM memory_embeddings
                WHERE memory_id NOT IN (SELECT id FROM memories)
                """
            )
        self._conn.commit()
        return deleted

    async def vacuum(self) -> None:
        assert self._conn is not None
        await asyncio.to_thread(self._conn.execute, "VACUUM")
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Internal: semantic recall ────────────────────────────────────────

    async def _store_embedding(self, memory_id: int, content: str) -> None:
        assert self._conn is not None
        try:
            vectors = await self.embedding_provider.embed([content])  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding generation failed for memory %s: %s", memory_id, exc)
            return
        if not vectors:
            return
        vector = vectors[0]
        blob = _vector_to_blob(vector)
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
            (memory_id, blob),
        )
        self._conn.commit()

    async def _semantic_recall(
        self,
        *,
        query: str,
        user_slug: str,
        workspace_id: str,
        memory_scope: str,
        surface_id: str,
        limit: int,
        budget_ms: float,
    ) -> list[MemoryRecord]:
        assert self._conn is not None
        try:
            vectors = await self.embedding_provider.embed([query])  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("recall embedding failed: %s", exc)
            return await self._keyword_recall(
                query=query,
                user_slug=user_slug,
                workspace_id=workspace_id,
                memory_scope=memory_scope,
                surface_id=surface_id,
                limit=limit,
            )
        if not vectors:
            return []
        query_blob = _vector_to_blob(vectors[0])
        now = _now_iso()
        rows = self._conn.execute(
            """
            SELECT m.*, e.distance AS distance
            FROM memory_embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE m.user_slug = ? AND m.workspace_id = ? AND m.memory_scope = ?
              AND (? = '' OR m.surface_id = ?)
              AND (m.expires_at IS NULL OR m.expires_at > ?)
            ORDER BY e.distance ASC
            LIMIT ?
            """,
            (user_slug, workspace_id, memory_scope, surface_id, surface_id, now, int(limit)),
        ).fetchall()
        records: list[MemoryRecord] = []
        for row in rows:
            record = _row_to_record(row)
            # Salience-weighted ordering: lower distance = higher similarity.
            distance = float(row["distance"] or 0.0)
            similarity = max(0.0, 1.0 - distance)
            record.salience = float(record.salience) * similarity
            records.append(record)
        records.sort(key=lambda r: r.salience, reverse=True)
        await self._touch_records(records)
        return records

    # ── Internal: keyword recall ─────────────────────────────────────────

    async def _keyword_recall(
        self,
        *,
        query: str,
        user_slug: str,
        workspace_id: str,
        memory_scope: str,
        surface_id: str,
        limit: int,
    ) -> list[MemoryRecord]:
        assert self._conn is not None
        keywords = [w.lower() for w in query.split() if len(w) >= 3]
        if not keywords:
            return []
        now = _now_iso()
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_slug = ? AND workspace_id = ? AND memory_scope = ?
              AND (? = '' OR surface_id = ?)
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_slug, workspace_id, memory_scope, surface_id, surface_id, now, max(limit * 4, 50)),
        ).fetchall()
        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _row_to_record(row)
            haystack = f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}".lower()
            hits = sum(1 for kw in keywords if kw in haystack)
            if hits == 0:
                continue
            score = (hits / len(keywords)) * float(record.salience)
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        records = [record for _, record in scored[:limit]]
        await self._touch_records(records)
        return records

    async def _touch_records(self, records: Iterable[MemoryRecord]) -> None:
        assert self._conn is not None
        now = _now_iso()
        for record in records:
            self._conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = ?
                WHERE id = ?
                """,
                (now, record.id),
            )
        self._conn.commit()


def _vector_to_blob(vector: list[float]) -> bytes:
    """Serialize a float vector into the sqlite-vec blob layout (little-endian)."""
    import struct

    return struct.pack(f"<{len(vector)}f", *vector)