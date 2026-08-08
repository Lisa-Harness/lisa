"""SQLite-backed memory store for the lisa-memory MCP server.

Extracted verbatim from the former ``lisa_context.server`` module: the
``SqliteMemoryStore`` plus its TTL helpers and row mapper. This is the
minimal built-in memory backend — five operations (add / search / list /
get / delete) backed by a local SQLite file. Mem0 (or any other HTTP/stdio
memory server) serves the same five operations and more via the MCP tool
pool.
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

try:  # optional: semantic recall via sqlite-vec. Absent → lexical-only.
    import sqlite_vec as _sqlite_vec
except Exception:  # pragma: no cover - import-time guard  # pylint: disable=broad-except
    _sqlite_vec = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_TTL_DAYS",
    "SCHEMA_VERSION",
    "EmbeddingProvider",
    "SqliteMemoryStore",
]

logger = logging.getLogger(__name__)

DEFAULT_TTL_DAYS: dict[str, int | None] = {
    "daily": 7,
    "weekly": 30,
    "long_term": 365,
    "permanent": None,
}

#: Schema version recorded in ``PRAGMA user_version``. Bumping this triggers a
#: hard recreate of the ``memories`` table (drop + create) per the project's
#: no-compat/migration rule — dev DBs are expendable, and external backends
#: (Mem0) own their own schemas. Add one for every material schema change.
SCHEMA_VERSION: int = 2


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
    raw_evidence = row["item_ids"] or "[]"
    try:
        item_ids = [int(eid) for eid in json.loads(raw_evidence)] if isinstance(
            raw_evidence, str
        ) else list(raw_evidence or [])
    except (json.JSONDecodeError, TypeError, ValueError):
        item_ids = []
    supersedes_id = row["supersedes_id"]
    valid_until = row["valid_until"]
    if valid_until is None:
        # TTL expiry is the default validity bound. An explicit valid_until
        # (set on write) overrides it for semantic-validity windows that
        # differ from the tier TTL.
        valid_until = row["expires_at"]
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
        valid_until=valid_until,
        memory_kind=row["memory_kind"],
        confidence=row["confidence"],
        item_ids=item_ids,
        supersedes_id=int(supersedes_id) if supersedes_id is not None else None,
        status=row["status"],
    )


@dataclass
class SqliteMemoryStore:
    """SQLite-backed memory store owned by the lisa-memory MCP server."""

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
        self._conn = conn
        self._init_schema()

    def _load_vec_extension(self) -> bool:
        """Load the ``sqlite-vec`` extension and return True if available.

        Wrapped in try/except so hosts without the wheel (or where the
        SQLite build lacks extension loading) degrade to lexical-only
        recall. A warning is logged once at construction; the
        ``_vec_available`` flag gates every downstream vec0 path.
        """
        assert self._conn is not None
        if _sqlite_vec is None:
            return False
        try:
            self._conn.enable_load_extension(True)
            self._conn.load_extension(_sqlite_vec.loadable_path())
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "sqlite-vec extension unavailable; semantic recall disabled: %s", exc
            )
            return False

    def _init_schema(self) -> None:
        assert self._conn is not None
        # Hard cutover on schema version bump: drop and recreate. Per the
        # project contract there are no compatibility shims — dev DBs are
        # expendable and external backends own their schemas. The
        # user_version pragma is the single source of truth.
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current != SCHEMA_VERSION:
            self._conn.executescript("DROP TABLE IF EXISTS memories;")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
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
                memory_kind TEXT NOT NULL DEFAULT 'episodic',
                status TEXT NOT NULL DEFAULT 'active',
                supersedes_id INTEGER,
                item_ids TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 1.0,
                merge_count INTEGER NOT NULL DEFAULT 0,
                content TEXT NOT NULL,
                content_json TEXT DEFAULT '{}',
                salience REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                valid_until TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(user_slug, workspace_id, memory_scope, surface_id);

            CREATE INDEX IF NOT EXISTS idx_memories_operation
                ON memories(user_slug, operation_id);

            CREATE INDEX IF NOT EXISTS idx_memories_expires
                ON memories(expires_at) WHERE expires_at IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_memories_kind_status
                ON memories(user_slug, workspace_id, memory_scope, surface_id,
                            memory_kind, status);
            """
        )
        self._vec_available = self._load_vec_extension()
        if self._vec_available:
            dim = int(self.embedding_dimension)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
                f"memory_id INTEGER PRIMARY KEY, embedding float[{dim}])"
            )
            self._conn.commit()

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
                operation_id, instruction_id, memory_tier, memory_kind, status,
                supersedes_id, item_ids, confidence, merge_count, content,
                content_json, salience, created_at, expires_at, valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                write.memory_kind,
                "active",
                write.supersedes_id,
                json.dumps([int(eid) for eid in write.item_ids], default=str),
                float(write.confidence),
                0,
                write.content,
                json.dumps(write.content_json or {}, default=str),
                write.salience,
                created_at,
                expires_at,
                write.valid_until,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM memories WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Failed to persist memory record")
        record = _row_to_record(row)
        # Opportunistic embedding write: best-effort, never blocks the
        # memory record. An embedding failure (provider down, dimension
        # mismatch) logs and continues — the row is already durable in
        # ``memories``; semantic recall simply won't find it until the
        # embedding is backfilled.
        await self._write_embedding(record.id, write.content)
        return record

    async def _write_embedding(self, memory_id: int, content: str) -> None:
        """Persist an embedding for ``content`` into the ``memory_vec`` table.

        Best-effort: no-op when no provider or when ``sqlite-vec`` is
        unavailable. Failures are logged and swallowed so a memory write
        never fails because of the embedding sidecar.
        """
        if self.embedding_provider is None or not self._vec_available:
            return
        assert self._conn is not None
        try:
            vectors = await self.embedding_provider.embed([content])
            if not vectors:
                return
            vec = vectors[0]
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                (int(memory_id), json.dumps([float(x) for x in vec])),
            )
            self._conn.commit()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "embedding write failed for memory_id=%s: %s", memory_id, exc
            )

    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
        memory_kind: str | None = None,
        status: str | None = "active",
    ) -> list[MemoryRecord]:
        """List memories for a surface, optionally filtered by kind and status.

        ``status`` defaults to ``"active"`` so retired/superseded records
        drop out of the common listing automatically. Pass ``None`` to
        include all statuses (used by consolidation and admin views).
        ``memory_kind`` filters by ``episodic``/``semantic``/… when given.
        """
        assert self._conn is not None
        where = (
            "user_slug = ? AND workspace_id = ? AND memory_scope = ? AND surface_id = ?"
        )
        params: list[Any] = [user_slug, workspace_id, memory_scope, surface_id]
        if memory_kind is not None:
            where += " AND memory_kind = ?"
            params.append(memory_kind)
        if status is not None:
            where += " AND status = ?"
            params.append(status)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    async def get_memory(
        self,
        *,
        memory_id: int | str,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> MemoryRecord | None:
        """Fetch a single memory record by id, scoped to the caller's surface.

        Returns ``None`` when no row matches — the caller decides whether to
        treat that as "not found" vs "deleted". Surface-scope filters
        (``user_slug`` / ``workspace_id`` / ``memory_scope`` / ``surface_id``)
        are applied so a caller cannot read across user or surface
        boundaries by guessing an id.
        """
        assert self._conn is not None
        row = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE id = ? AND user_slug = ? AND workspace_id = ?
              AND memory_scope = ? AND surface_id = ?
            """,
            (int(memory_id), user_slug, workspace_id, memory_scope, surface_id),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    async def delete_memory(
        self,
        *,
        memory_id: int | str,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> bool:
        """Delete a single memory record by id, scoped to the caller's surface.

        Returns ``True`` when a row was deleted, ``False`` when no matching
        record existed (wrong id or wrong scope). Surface-scope filters
        prevent cross-user / cross-surface deletion by id guessing.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            """
            DELETE FROM memories
            WHERE id = ? AND user_slug = ? AND workspace_id = ?
              AND memory_scope = ? AND surface_id = ?
            """,
            (int(memory_id), user_slug, workspace_id, memory_scope, surface_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    async def retire_memory(
        self,
        *,
        memory_id: int | str,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> bool:
        """Mark a memory record retired, scoped to the caller's surface.

        Retirement (not deletion) preserves the audit trail — the record
        stays queryable with ``status=None`` but drops out of the default
        ``status='active'`` recall pool. Used by consolidation to retire
        episodic originals after a semantic successor is written. Returns
        ``True`` when a row was updated.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            """
            UPDATE memories
            SET status = 'retired'
            WHERE id = ? AND user_slug = ? AND workspace_id = ?
              AND memory_scope = ? AND surface_id = ? AND status = 'active'
            """,
            (int(memory_id), user_slug, workspace_id, memory_scope, surface_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    async def update_salience(
        self,
        *,
        memory_id: int | str,
        salience: float,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> bool:
        """Update a record's salience, scoped to the caller's surface.

        Hook for decay-aware dynamic salience (the consolidation slice
        boosts the semantic successor's salience; a future enhancement
        can decay on recall access). Returns ``True`` when a row was
        updated.
        """
        assert self._conn is not None
        cursor = self._conn.execute(
            """
            UPDATE memories
            SET salience = ?
            WHERE id = ? AND user_slug = ? AND workspace_id = ?
              AND memory_scope = ? AND surface_id = ?
            """,
            (
                float(max(0.0, min(1.0, salience))),
                int(memory_id),
                user_slug,
                workspace_id,
                memory_scope,
                surface_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    async def count_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        memory_kind: str | None = None,
        status: str | None = "active",
    ) -> int:
        """Count memories for a surface, optionally filtered by kind and status.

        Used by the consolidation slice to compute episodic pressure
        (episodic-active count / ``memory_max_records``) without pulling
        rows. Defaults to ``status='active'`` to match ``list_memories``.
        """
        assert self._conn is not None
        where = (
            "user_slug = ? AND workspace_id = ? AND memory_scope = ? AND surface_id = ?"
        )
        params: list[Any] = [user_slug, workspace_id, memory_scope, surface_id]
        if memory_kind is not None:
            where += " AND memory_kind = ?"
            params.append(memory_kind)
        if status is not None:
            where += " AND status = ?"
            params.append(status)
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM memories WHERE {where}", tuple(params)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    async def _semantic_pool(
        self,
        *,
        query: str,
        user_slug: str,
        workspace_id: str,
        memory_scope: str,
        surface_id: str,
        memory_kind: str | None,
        status: str | None,
        now_iso: str,
        recall_pool: int,
    ) -> list[sqlite3.Row]:
        """KNN candidate pool over ``memory_vec``, re-ranked by the caller.

        Returns an empty list when embeddings are unavailable or the
        query is empty — the caller falls back to the lexical pool.
        """
        assert self._conn is not None
        if self.embedding_provider is None or not self._vec_available or not query.strip():
            return []
        try:
            vectors = await self.embedding_provider.embed([query])
            if not vectors:
                return []
            where = (
                "m.user_slug = ? AND m.workspace_id = ? "
                "AND m.memory_scope = ? AND m.surface_id = ?"
            )
            params: list[Any] = [user_slug, workspace_id, memory_scope, surface_id]
            if memory_kind is not None:
                where += " AND m.memory_kind = ?"
                params.append(memory_kind)
            if status is not None:
                where += " AND m.status = ?"
                params.append(status)
            params.append(now_iso)
            params.append(json.dumps([float(x) for x in vectors[0]]))
            params.append(recall_pool)
            return self._conn.execute(
                f"""
                SELECT m.* FROM memories m
                JOIN memory_vec v ON v.memory_id = m.id
                WHERE {where}
                  AND (m.expires_at IS NULL OR m.expires_at >= ?)
                ORDER BY v.embedding MATCH ? KNN ?
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("semantic recall failed; falling back to lexical: %s", exc)
            return []

    def _lexical_pool(
        self,
        *,
        user_slug: str,
        workspace_id: str,
        memory_scope: str,
        surface_id: str,
        memory_kind: str | None,
        status: str | None,
        now_iso: str,
        recall_pool: int,
    ) -> list[sqlite3.Row]:
        """Recent-rows candidate pool for lexical recall ranking."""
        assert self._conn is not None
        where = (
            "user_slug = ? AND workspace_id = ? AND memory_scope = ? "
            "AND surface_id = ? AND (expires_at IS NULL OR expires_at >= ?)"
        )
        params: list[Any] = [user_slug, workspace_id, memory_scope, surface_id, now_iso]
        if memory_kind is not None:
            where += " AND memory_kind = ?"
            params.append(memory_kind)
        if status is not None:
            where += " AND status = ?"
            params.append(status)
        params.append(recall_pool)
        return self._conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    async def recall(self, query: str, **kwargs: Any) -> str:
        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        workspace_id = str(kwargs.get("workspace_id") or "default").strip() or "default"
        memory_scope = str(kwargs.get("memory_scope") or "personal").strip() or "personal"
        surface_id = str(kwargs.get("surface_id") or "").strip()
        limit = int(kwargs.get("limit", 5) or 5)
        recall_pool = int(kwargs.get("recall_pool", 40) or 40)
        memory_kind = kwargs.get("memory_kind")
        status = kwargs.get("status", "active")

        assert self._conn is not None

        tokens = {token.strip().lower() for token in query.split() if token.strip()}
        now_iso = _now_iso()

        rows = await self._semantic_pool(
            query=query,
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            memory_kind=memory_kind,
            status=status,
            now_iso=now_iso,
            recall_pool=recall_pool,
        )
        if not rows:
            rows = self._lexical_pool(
                user_slug=user_slug,
                workspace_id=workspace_id,
                memory_scope=memory_scope,
                surface_id=surface_id,
                memory_kind=memory_kind,
                status=status,
                now_iso=now_iso,
                recall_pool=recall_pool,
            )
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

    async def recall_records(self, query: str, **kwargs: Any) -> list[MemoryRecord]:
        """Rank and return the matching memory records (no prose rendering).

        Same ranking and opportunistic access tracking as :meth:`recall`,
        but returns the structured records instead of a bullet-list string.
        The MCP ``search_memory`` tool serializes this to a JSON array so
        the ``recall_memory`` operation (which does ``json.loads`` on the
        tool result) can validate the records — :meth:`recall`'s prose
        output is not JSON-parseable and left ``recall_memory`` reporting
        "No memory matches found" against a full store.
        """
        user_slug = str(kwargs.get("user_slug") or "").strip() or "default"
        workspace_id = str(kwargs.get("workspace_id") or "default").strip() or "default"
        memory_scope = str(kwargs.get("memory_scope") or "personal").strip() or "personal"
        surface_id = str(kwargs.get("surface_id") or "").strip()
        limit = int(kwargs.get("limit", 5) or 5)
        recall_pool = int(kwargs.get("recall_pool", 40) or 40)
        memory_kind = kwargs.get("memory_kind")
        status = kwargs.get("status", "active")

        assert self._conn is not None

        tokens = {token.strip().lower() for token in query.split() if token.strip()}
        now_iso = _now_iso()

        rows = await self._semantic_pool(
            query=query,
            user_slug=user_slug,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
            memory_kind=memory_kind,
            status=status,
            now_iso=now_iso,
            recall_pool=recall_pool,
        )
        if not rows:
            rows = self._lexical_pool(
                user_slug=user_slug,
                workspace_id=workspace_id,
                memory_scope=memory_scope,
                surface_id=surface_id,
                memory_kind=memory_kind,
                status=status,
                now_iso=now_iso,
                recall_pool=recall_pool,
            )
        if not rows:
            return []

        ranked: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            record = _row_to_record(row)
            haystack_lower = (
                f"{record.content} {json.dumps(record.content_json, ensure_ascii=True)}"
            ).lower()
            if tokens:
                haystack_tokens = {tok for tok in haystack_lower.split()}
                overlap = len(tokens & haystack_tokens)
                content_word_count = max(len(haystack_tokens), 1)
                lexical = overlap / max(len(tokens), 1) * (1.0 - 1.0 / content_word_count)
            else:
                lexical = 1.0
            salience_boost = max(record.salience, 0.0) * 0.2
            ranked.append((lexical + salience_boost, record))

        ranked.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        top = ranked[: max(1, min(limit, 8))]

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

        return [record for _, record in top]

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Called by the lisa-memory MCP server's lifespan teardown so the
        connection is released cleanly when the stdio session ends. Safe
        to call when the store was never initialized (no-op).
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None
