"""IPSAR local SQLite memory store with eager blob sync and optional embeddings.

Called by note capture, post-turn learning synthesis, and the memory recall
engine. The SQLite file is the durable local source of truth; blob snapshotting
and sync events keep that state recoverable across restarts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import sqlite_vec

from kernel.models import (
    MemoryEmbeddingRecord,
    MemoryRecord,
    MemorySalienceEvent,
    MemoryScope,
    MemoryTier,
    MemoryWriteRequest,
)

from app import blob_client, providers
from app.config import settings
from app.data_sanitizer import apply_redaction_pipeline
from app.event_bus import BusBackend
from kernel.seams.protocols import MemoryStore as MemoryStoreProtocol

logger = logging.getLogger(__name__)

# Default ANN dimension when the provider returns a list of unknown length.
# The embed path is expected to report the actual dim after first use.
_DEFAULT_VEC_DIMENSION = 1536
# Cosine is the recall contract: distance in [0, 2], lower is more similar.
_VEC_METRIC = "cosine"

MemoryKindValue = Literal["episodic", "semantic", "procedural", "reflection", "preference"]
MemoryStatusValue = Literal["active", "stale", "disputed", "retired"]

# Module-level bus reference injected at startup via set_memory_salience_bus().
_salience_bus: BusBackend | None = None


def set_memory_salience_bus(bus: BusBackend) -> None:
    """Inject the running event bus so record_memory() can publish MemorySalienceEvent."""
    global _salience_bus  # pylint: disable=global-statement
    _salience_bus = bus


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL DEFAULT 'default',
        user_id TEXT NOT NULL,
        user_slug TEXT NOT NULL,
        memory_scope TEXT NOT NULL DEFAULT 'personal',
        surface_id TEXT NOT NULL DEFAULT '',
        source_user_id TEXT NOT NULL DEFAULT '',
        source_user_slug TEXT NOT NULL DEFAULT '',
        agent_id TEXT NOT NULL,
        operation_id TEXT NOT NULL,
        instruction_id TEXT NOT NULL,
        memory_tier TEXT NOT NULL,
        content TEXT NOT NULL,
        content_json TEXT NOT NULL,
        salience REAL NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_operation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        op TEXT NOT NULL,
        memory_ids TEXT NOT NULL DEFAULT '[]',
        query_text TEXT NOT NULL DEFAULT '',
        injected_tokens INTEGER NOT NULL DEFAULT 0,
        latency_ms INTEGER NOT NULL DEFAULT 0,
        trigger_operation TEXT NOT NULL DEFAULT '',
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_queue (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        payload_jsonl TEXT NOT NULL,
        created_at TEXT NOT NULL,
        synced_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_embeddings (
        memory_id INTEGER PRIMARY KEY,
        workspace_id TEXT NOT NULL DEFAULT 'default',
        user_slug TEXT NOT NULL,
        provider TEXT NOT NULL,
        model_id TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        dimensions INTEGER NOT NULL DEFAULT 0,
        vector_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
    )
    """,
)


def _hash_memory_content(content: str) -> str:
    """Return a stable content hash used to detect stale embeddings."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _vec_dim_from_settings() -> int:
    """Return the configured ANN dimension, falling back to a safe default."""
    dim = getattr(settings, "memory_vec_dimension", 0) or 0
    if dim > 0:
        return int(dim)
    return _DEFAULT_VEC_DIMENSION


def _enable_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension on one connection."""
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())


def _create_vec_table_sql(dimension: int) -> str:
    """Return the vec0 CREATE VIRTUAL TABLE statement for ``memory_vec``."""
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
        f"memory_id INTEGER PRIMARY KEY, "
        f"embedding float[{int(dimension)}] distance_metric={_VEC_METRIC})"
    )


def _serialize_vec(vector: list[float]) -> bytes:
    """Pack a Python float list into the sqlite-vec binary blob format."""
    return bytes(sqlite_vec.serialize_float32(vector))


def _read_vec_table_dimension(conn: sqlite3.Connection) -> int:
    """Return the float[] dimension of an existing memory_vec table, or 0."""
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_vec'"
        ).fetchall()
    except sqlite3.DatabaseError:
        return 0
    if not rows:
        return 0
    create_sql = str(rows[0][0] or "")
    # Parse "float[<N>]" out of the column list.
    marker = "float["
    start = create_sql.find(marker)
    if start < 0:
        return 0
    end = create_sql.find("]", start)
    if end < 0:
        return 0
    try:
        return int(create_sql[start + len(marker) : end])
    except ValueError:
        return 0


def _bulk_upsert_vec_rows(conn: sqlite3.Connection, rows: list[tuple[int, list[float]]]) -> None:
    """Insert or replace (memory_id, embedding) rows in bulk."""
    if not rows:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO memory_vec(memory_id, embedding) VALUES (?, ?)",
        [(memory_id, _serialize_vec(vector)) for memory_id, vector in rows],
    )


def _delete_vec_rows(conn: sqlite3.Connection, memory_ids: list[int]) -> None:
    if not memory_ids:
        return
    placeholders = ", ".join("?" for _ in memory_ids)
    conn.execute(
        f"DELETE FROM memory_vec WHERE memory_id IN ({placeholders})",
        memory_ids,
    )


def _knn_memory_ids(
    conn: sqlite3.Connection,
    query_vector: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Return up to ``k`` (memory_id, cosine_distance) pairs nearest to ``query_vector``."""
    if k <= 0 or not query_vector:
        return []
    rows = conn.execute(
        """
        SELECT memory_id, distance
        FROM memory_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        (_serialize_vec(query_vector), int(k)),
    ).fetchall()
    return [(int(memory_id), float(distance)) for memory_id, distance in rows]


def _knn_memory_ids_self(
    db_path: Path,
    query_vector: list[float],
    k: int,
) -> list[tuple[int, float]]:
    """Open a fresh connection, load sqlite-vec, and run a KNN scan."""
    if k <= 0 or not query_vector:
        return []
    with sqlite3.connect(db_path) as conn:
        _enable_sqlite_vec(conn)
        return _knn_memory_ids(conn, query_vector, k)


class IPSARMemoryStore(MemoryStoreProtocol):
    """Manages local memory writes and eager sync to blob storage."""

    def __init__(self) -> None:
        self._db_path = Path(settings.memory_sqlite_path)
        self._sync_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._snapshot_due_at = 0.0
        self._initialized = False
        self._ann_conn: sqlite3.Connection | None = None

    async def start(self) -> None:
        self._db_path = Path(settings.memory_sqlite_path)
        if not settings.memory_enabled or self._initialized:
            return
        await asyncio.to_thread(self._initialize_db)
        self._initialized = True
        self._snapshot_due_at = asyncio.get_running_loop().time() + max(
            1, settings.memory_snapshot_interval_seconds
        )
        self._stop_event.clear()
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("IPSAR memory store ready at %s", self._db_path)

    async def shutdown(self, timeout_seconds: float | None = None) -> None:
        if not self._initialized:
            self._ann_read_conn_reset()
            return

        self._stop_event.set()
        if self._sync_task is not None:
            try:
                await asyncio.wait_for(
                    self._sync_task,
                    timeout=max(1.0, timeout_seconds or settings.memory_final_sync_timeout_seconds),
                )
            except TimeoutError:
                logger.warning("memory sync loop stop timed out")
            self._sync_task = None

        try:
            await asyncio.wait_for(
                self.flush_pending_sync(),
                timeout=max(1.0, timeout_seconds or settings.memory_final_sync_timeout_seconds),
            )
            await asyncio.wait_for(
                self.write_snapshot(),
                timeout=max(1.0, timeout_seconds or settings.memory_final_sync_timeout_seconds),
            )
        except TimeoutError:
            logger.warning("memory final sync timed out")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory final sync failed: %s", exc)

        self._ann_read_conn_reset()

    async def record_memory(self, request: MemoryWriteRequest) -> int:
        """Persist one memory write and best-effort index it for semantic recall."""
        if not settings.memory_enabled:
            return 0

        if not self._initialized:
            await self.start()

        # Redact PII from content before persisting.
        redacted_content = apply_redaction_pipeline(request.content)
        if redacted_content != request.content:
            request = request.model_copy(update={"content": redacted_content})

        memory_id = await asyncio.to_thread(self._insert_memory, request)
        await asyncio.to_thread(
            self._log_memory_op,
            op="write",
            memory_ids=[memory_id],
            trigger_operation=request.operation_id,
        )
        # Phase 6: emit salience event when the memory crosses the configured threshold
        if (
            _salience_bus is not None
            and request.salience >= settings.memory_salience_event_threshold
        ):
            try:
                await _salience_bus.publish(
                    MemorySalienceEvent(
                        idempotency_key=str(memory_id),
                        workspace_id=request.workspace_id,
                        memory_id=memory_id,
                        user_slug=request.user_slug,
                        memory_scope=request.memory_scope,
                        surface_id=request.surface_id,
                        operation_id=request.operation_id,
                        salience_score=request.salience,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("memory salience event publish failed (non-fatal): %s", exc)

        if settings.memory_sync_enabled and settings.memory_eager_sync_on_write:
            try:
                await self.flush_pending_sync(max_rows=settings.memory_sync_batch_size)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("eager memory sync failed: %s", exc)

        if settings.memory_embedding_enabled and request.content.strip():
            try:
                records = await asyncio.to_thread(self._get_memory_records_by_ids, [memory_id])
                await self.ensure_embeddings_for_records(records)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("memory embedding update failed: %s", exc)
        return memory_id

    async def delete_memories(
        self,
        *,
        workspace_id: str,
        memory_scope: MemoryScope | None = None,
        surface_id: str | None = None,
        user_slug: str | None = None,
    ) -> int:
        """Delete memory rows matching the supplied scope filters.

        All provided kwargs are AND-combined.  At least ``workspace_id`` is
        required so that a bare call never accidentally deletes across workspaces.
        Returns the number of rows deleted.
        """
        if not settings.memory_enabled:
            return 0

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return 0

        return await asyncio.to_thread(
            self._delete_memories,
            workspace_id,
            memory_scope,
            surface_id,
            user_slug,
        )

    async def retire_memories(
        self,
        *,
        workspace_id: str,
        memory_scope: MemoryScope,
        surface_id: str,
        user_slug: str,
        content_query: str = "",
    ) -> int:
        """Retire active memories in one privacy scope with optional content filtering.

        This keeps rows auditable in SQLite while removing them from normal
        retrieval paths that exclude retired status.
        """
        if not settings.memory_enabled:
            return 0

        if memory_scope != "personal" and not surface_id.strip():
            return 0

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return 0

        retired = await asyncio.to_thread(
            self._retire_memories,
            workspace_id,
            memory_scope,
            surface_id,
            user_slug,
            content_query,
        )
        if retired:
            await asyncio.to_thread(
                self._log_memory_op,
                op="retire",
                memory_ids=[],
                query_text=content_query,
                trigger_operation="default_chat_forget",
            )
        return retired

    async def retire_memories_by_ids(
        self,
        *,
        workspace_id: str,
        memory_scope: MemoryScope,
        surface_id: str,
        user_slug: str,
        memory_ids: list[int],
    ) -> int:
        """Retire exact memory IDs constrained to one privacy scope."""
        if not settings.memory_enabled:
            return 0

        if memory_scope != "personal" and not surface_id.strip():
            return 0

        scoped_ids = [
            memory_id for memory_id in memory_ids if isinstance(memory_id, int) and memory_id > 0
        ]
        if not scoped_ids:
            return 0

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return 0

        retired = await asyncio.to_thread(
            self._retire_memories_by_ids,
            workspace_id,
            memory_scope,
            surface_id,
            user_slug,
            scoped_ids,
        )
        if retired:
            await asyncio.to_thread(
                self._log_memory_op,
                op="retire",
                memory_ids=scoped_ids,
                trigger_operation="default_chat_forget",
            )
        return retired

    async def dispute_memories_by_ids(
        self,
        *,
        workspace_id: str,
        memory_scope: MemoryScope,
        surface_id: str,
        user_slug: str,
        memory_ids: list[int],
        dispute_reason: str = "",
    ) -> int:
        """Mark exact memory IDs as disputed inside one privacy scope."""
        if not settings.memory_enabled:
            return 0

        if memory_scope != "personal" and not surface_id.strip():
            return 0

        scoped_ids = [
            memory_id for memory_id in memory_ids if isinstance(memory_id, int) and memory_id > 0
        ]
        if not scoped_ids:
            return 0

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return 0

        disputed = await asyncio.to_thread(
            self._update_memory_status_by_ids,
            workspace_id,
            memory_scope,
            surface_id,
            user_slug,
            scoped_ids,
            "disputed",
            "memory_dispute",
            dispute_reason,
        )
        if disputed:
            await asyncio.to_thread(
                self._log_memory_op,
                op="dispute",
                memory_ids=scoped_ids,
                query_text=dispute_reason,
                trigger_operation="default_chat_memory_correction",
            )
        return disputed

    async def replace_memory_by_id(
        self,
        *,
        replaced_memory_id: int,
        replacement_request: MemoryWriteRequest,
    ) -> int:
        """Replace one memory row with a successor linked by supersedes_id."""
        if not settings.memory_enabled or replaced_memory_id <= 0:
            return 0

        if (
            replacement_request.memory_scope != "personal"
            and not replacement_request.surface_id.strip()
        ):
            return 0

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return 0

        replacement_request = replacement_request.model_copy(
            update={"supersedes_id": replaced_memory_id}
        )
        new_memory_id = await self.record_memory(replacement_request)
        if new_memory_id <= 0:
            return 0

        replaced_count = await asyncio.to_thread(
            self._update_memory_status_by_ids,
            replacement_request.workspace_id,
            replacement_request.memory_scope,
            replacement_request.surface_id,
            replacement_request.user_slug,
            [replaced_memory_id],
            "disputed",
            "memory_replace",
            replacement_request.content,
            new_memory_id,
        )
        if replaced_count <= 0:
            return 0

        await asyncio.to_thread(
            self._log_memory_op,
            op="replace",
            memory_ids=[replaced_memory_id, new_memory_id],
            query_text=replacement_request.content,
            trigger_operation="default_chat_memory_correction",
        )
        return new_memory_id

    def _delete_memories(
        self,
        workspace_id: str,
        memory_scope: MemoryScope | None,
        surface_id: str | None,
        user_slug: str | None,
    ) -> int:
        where_clauses = ["workspace_id = ?"]
        params: list[object] = [workspace_id]
        if memory_scope is not None:
            where_clauses.append("memory_scope = ?")
            params.append(memory_scope)
        if surface_id is not None:
            where_clauses.append("surface_id = ?")
            params.append(surface_id)
        if user_slug is not None:
            where_clauses.append("user_slug = ?")
            params.append(user_slug)

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE {' AND '.join(where_clauses)}",
                params,
            )
            conn.commit()
        return cursor.rowcount

    def _retire_memories(
        self,
        workspace_id: str,
        memory_scope: MemoryScope,
        surface_id: str,
        user_slug: str,
        content_query: str,
    ) -> int:
        where_clauses = [
            "workspace_id = ?",
            "memory_scope = ?",
            "status != 'retired'",
        ]
        params: list[object] = [workspace_id, memory_scope]

        if memory_scope == "personal":
            where_clauses.append("user_slug = ?")
            params.append(user_slug)
        else:
            where_clauses.append("surface_id = ?")
            params.append(surface_id)

        normalized_query = content_query.strip().lower()
        if normalized_query:
            where_clauses.append("LOWER(content) LIKE ?")
            params.append(f"%{normalized_query}%")

        retired_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            id_cursor = conn.execute(
                (f"SELECT id FROM memories WHERE {' AND '.join(where_clauses)}"),
                params,
            )
            memory_ids = [int(row[0]) for row in id_cursor.fetchall()]
            if not memory_ids:
                return 0

            placeholders = ", ".join("?" for _ in memory_ids)
            cursor = conn.execute(
                (
                    "UPDATE memories "
                    "SET status = 'retired', valid_until = ? "
                    f"WHERE id IN ({placeholders})"
                ),
                [retired_at, *memory_ids],
            )
            payload = {
                "op": "memory_retire",
                "ts": retired_at,
                "workspace_id": workspace_id,
                "memory_scope": memory_scope,
                "surface_id": surface_id,
                "user_slug": user_slug,
                "content_query": content_query,
                "memory_ids": memory_ids,
                "retired_count": int(cursor.rowcount),
            }
            conn.execute(
                "INSERT INTO sync_queue (payload_jsonl, created_at, synced_at) VALUES (?, ?, NULL)",
                (json.dumps(payload, separators=(",", ":")), retired_at),
            )
            conn.commit()
        return cursor.rowcount

    def _retire_memories_by_ids(
        self,
        workspace_id: str,
        memory_scope: MemoryScope,
        surface_id: str,
        user_slug: str,
        memory_ids: list[int],
    ) -> int:
        where_clauses = [
            "workspace_id = ?",
            "memory_scope = ?",
            "status != 'retired'",
        ]
        params: list[object] = [workspace_id, memory_scope]
        if memory_scope == "personal":
            where_clauses.append("user_slug = ?")
            params.append(user_slug)
        else:
            where_clauses.append("surface_id = ?")
            params.append(surface_id)

        placeholders = ", ".join("?" for _ in memory_ids)
        where_clauses.append(f"id IN ({placeholders})")
        params.extend(memory_ids)

        retired_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            id_cursor = conn.execute(
                f"SELECT id FROM memories WHERE {' AND '.join(where_clauses)}",
                params,
            )
            scoped_ids = [int(row[0]) for row in id_cursor.fetchall()]
            if not scoped_ids:
                return 0

            scoped_placeholders = ", ".join("?" for _ in scoped_ids)
            cursor = conn.execute(
                (
                    "UPDATE memories "
                    "SET status = 'retired', valid_until = ? "
                    f"WHERE id IN ({scoped_placeholders})"
                ),
                [retired_at, *scoped_ids],
            )
            payload = {
                "op": "memory_retire",
                "ts": retired_at,
                "workspace_id": workspace_id,
                "memory_scope": memory_scope,
                "surface_id": surface_id,
                "user_slug": user_slug,
                "memory_ids": scoped_ids,
                "retired_count": int(cursor.rowcount),
            }
            conn.execute(
                "INSERT INTO sync_queue (payload_jsonl, created_at, synced_at) VALUES (?, ?, NULL)",
                (json.dumps(payload, separators=(",", ":")), retired_at),
            )
            conn.commit()
        return cursor.rowcount

    def _update_memory_status_by_ids(
        self,
        workspace_id: str,
        memory_scope: MemoryScope,
        surface_id: str,
        user_slug: str,
        memory_ids: list[int],
        status: str,
        sync_op: str,
        detail: str = "",
        successor_memory_id: int | None = None,
    ) -> int:
        where_clauses = [
            "workspace_id = ?",
            "memory_scope = ?",
            "status != 'retired'",
        ]
        params: list[object] = [workspace_id, memory_scope]
        if memory_scope == "personal":
            where_clauses.append("user_slug = ?")
            params.append(user_slug)
        else:
            where_clauses.append("surface_id = ?")
            params.append(surface_id)

        placeholders = ", ".join("?" for _ in memory_ids)
        where_clauses.append(f"id IN ({placeholders})")
        params.extend(memory_ids)

        changed_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            id_cursor = conn.execute(
                f"SELECT id FROM memories WHERE {' AND '.join(where_clauses)}",
                params,
            )
            scoped_ids = [int(row[0]) for row in id_cursor.fetchall()]
            if not scoped_ids:
                return 0

            scoped_placeholders = ", ".join("?" for _ in scoped_ids)
            cursor = conn.execute(
                (
                    "UPDATE memories "
                    "SET status = ?, valid_until = ? "
                    f"WHERE id IN ({scoped_placeholders})"
                ),
                [status, changed_at, *scoped_ids],
            )
            payload = {
                "op": sync_op,
                "ts": changed_at,
                "workspace_id": workspace_id,
                "memory_scope": memory_scope,
                "surface_id": surface_id,
                "user_slug": user_slug,
                "memory_ids": scoped_ids,
                "status": status,
                "detail": detail,
                "changed_count": int(cursor.rowcount),
            }
            if successor_memory_id is not None:
                payload["successor_memory_id"] = successor_memory_id
            conn.execute(
                "INSERT INTO sync_queue (payload_jsonl, created_at, synced_at) VALUES (?, ?, NULL)",
                (json.dumps(payload, separators=(",", ":")), changed_at),
            )
            conn.commit()
        return cursor.rowcount

    async def list_recent_memories(
        self,
        user_slug: str,
        operation_id: str,
        limit: int = 20,
        workspace_id: str = "default",
        memory_scope: MemoryScope = "personal",
        surface_id: str = "",
    ) -> list[MemoryRecord]:
        """Return recent memories for one exact operation identifier."""
        return await self.list_memories(
            user_slug=user_slug,
            operation_ids=[operation_id],
            limit=limit,
            workspace_id=workspace_id,
            memory_scope=memory_scope,
            surface_id=surface_id,
        )

    async def list_memories(
        self,
        *,
        user_slug: str,
        limit: int = 20,
        workspace_id: str = "default",
        memory_scope: MemoryScope = "personal",
        surface_id: str = "",
        operation_ids: list[str] | None = None,
        memory_tiers: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Return recent memories for one privacy scope with optional operation filters."""
        if not settings.memory_enabled:
            return []

        if memory_scope != "personal" and not surface_id.strip():
            return []

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return []

        return await asyncio.to_thread(
            self._list_memories,
            user_slug,
            operation_ids or [],
            memory_tiers or [],
            limit,
            workspace_id,
            memory_scope,
            surface_id,
        )

    async def get_memory_records_by_ids(self, memory_ids: list[int]) -> list[MemoryRecord]:
        """Return specific memory records by id for indexing and recall."""
        if not settings.memory_enabled or not memory_ids:
            return []

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return []

        return await asyncio.to_thread(self._get_memory_records_by_ids, memory_ids)

    async def get_memory_embeddings(
        self,
        memory_ids: list[int],
    ) -> dict[int, MemoryEmbeddingRecord]:
        """Return stored embeddings for the requested memory ids."""
        if not settings.memory_enabled or not memory_ids:
            return {}

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return {}

        return await asyncio.to_thread(self._get_memory_embeddings, memory_ids)

    async def ensure_embeddings_for_records(
        self,
        records: list[MemoryRecord],
    ) -> dict[int, MemoryEmbeddingRecord]:
        """Ensure the given records have current embeddings when embedding support is enabled."""
        if not settings.memory_enabled or not records:
            return {}

        if not self._initialized:
            await self.start()

        if not self._initialized:
            return {}

        memory_ids = [record.id for record in records]
        stored_embeddings = await asyncio.to_thread(self._get_memory_embeddings, memory_ids)

        if not settings.memory_embedding_enabled or not settings.memory_embedding_model.strip():
            return stored_embeddings

        provider_name = settings.memory_embedding_provider.strip() or "azure_openai"
        model_id = settings.memory_embedding_model.strip()
        pending_records: list[MemoryRecord] = []
        pending_texts: list[str] = []

        for record in records:
            content_hash = _hash_memory_content(record.content)
            existing = stored_embeddings.get(record.id)
            if (
                existing is not None
                and existing.provider == provider_name
                and existing.model_id == model_id
                and existing.content_hash == content_hash
            ):
                continue
            pending_records.append(record)
            pending_texts.append(record.content)

        if not pending_records:
            return stored_embeddings

        try:
            result = await providers.embed_texts(
                pending_texts,
                provider=provider_name,
                model_id=model_id,
                dimensions=settings.memory_embedding_dimensions,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory embedding request failed: %s", exc)
            return stored_embeddings

        if len(result.embeddings) != len(pending_records):
            logger.warning(
                "memory embedding count mismatch: expected=%s actual=%s",
                len(pending_records),
                len(result.embeddings),
            )

        upserts: list[MemoryEmbeddingRecord] = []
        now_iso = datetime.now(UTC).isoformat()
        for record, vector in zip(pending_records, result.embeddings):
            existing = stored_embeddings.get(record.id)
            embedding_record = MemoryEmbeddingRecord(
                memory_id=record.id,
                workspace_id=record.workspace_id,
                user_slug=record.user_slug,
                provider=provider_name,
                model_id=model_id,
                content_hash=_hash_memory_content(record.content),
                dimensions=len(vector),
                vector=[float(value) for value in vector],
                created_at=existing.created_at if existing is not None else now_iso,
                updated_at=now_iso,
            )
            upserts.append(embedding_record)

        await asyncio.to_thread(self._upsert_memory_embeddings, upserts)
        stored_embeddings.update({record.memory_id: record for record in upserts})

        # Mirror the same vectors into the sqlite-vec ANN index so the
        # recall ranker can use a KNN scan instead of a linear cosine loop.
        if settings.memory_ann_enabled:
            try:
                vec_payloads: list[tuple[int, list[float]]] = []
                for record, vector in zip(pending_records, result.embeddings):
                    if len(vector) != _vec_dim_from_settings():
                        continue
                    vec_payloads.append((record.id, [float(v) for v in vector]))
                if vec_payloads:
                    await asyncio.to_thread(self._upsert_memory_vec_rows, vec_payloads)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("memory vec upsert failed: %s", exc)

        return stored_embeddings

    async def reindex_memory_vec(
        self,
        *,
        force: bool = False,
    ) -> int:
        """Rebuild the ``memory_vec`` table from the current ``memory_embeddings`` table.

        Returns the number of vectors (re)inserted. Idempotent: on a clean
        ANN table the call is a no-op unless ``force`` is set.
        """
        if not settings.memory_enabled:
            return 0
        if not self._initialized:
            await self.start()
        if not self._initialized:
            return 0
        return await asyncio.to_thread(self._reindex_memory_vec_sync, bool(force))

    async def ann_search_memory_ids(
        self,
        query_vector: list[float],
        *,
        k: int,
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` (memory_id, cosine_distance) pairs via sqlite-vec."""
        if not settings.memory_enabled or not query_vector:
            return []
        if not self._initialized:
            await self.start()
        if not self._initialized:
            return []
        return await asyncio.to_thread(self._ann_search_sync, list(query_vector), int(k))

    def _ann_search_sync(self, query_vector: list[float], k: int) -> list[tuple[int, float]]:
        """Synchronous ANN search using a cached sqlite-vec connection.

        The connection lives for the lifetime of the store so the cost of
        ``load_extension`` is paid once. We only use this connection for
        read-only KNN queries; writes go through their own short-lived
        connections.
        """
        if k <= 0 or not query_vector:
            return []
        conn = self._ann_read_conn()
        try:
            return _knn_memory_ids(conn, query_vector, k)
        except sqlite3.DatabaseError as exc:
            logger.warning("memory ann search failed, dropping cached conn: %s", exc)
            self._ann_read_conn_reset()
            return []

    def _ann_read_conn(self) -> sqlite3.Connection:
        """Return the cached ANN read connection, creating it on first use."""
        conn: sqlite3.Connection | None = self._ann_conn
        if conn is not None:
            return conn
        new_conn = sqlite3.connect(self._db_path, check_same_thread=False)
        _enable_sqlite_vec(new_conn)
        self._ann_conn = new_conn
        return new_conn

    def _ann_read_conn_reset(self) -> None:
        """Drop the cached read connection (called when sqlite-vec raises)."""
        conn = getattr(self, "_ann_conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # pylint: disable=broad-except
            pass
        self._ann_conn = None

    def _upsert_memory_vec_rows(self, rows: list[tuple[int, list[float]]]) -> None:
        """Sync upsert helper used by ``ensure_embeddings_for_records`` and reindex."""
        if not rows:
            return
        with sqlite3.connect(self._db_path) as conn:
            _enable_sqlite_vec(conn)
            target_dim = _vec_dim_from_settings()
            filtered = [
                (memory_id, vector) for memory_id, vector in rows if len(vector) == target_dim
            ]
            _bulk_upsert_vec_rows(conn, filtered)
            conn.commit()

    def _reindex_memory_vec_sync(self, force: bool) -> int:
        target_dim = _vec_dim_from_settings()
        with sqlite3.connect(self._db_path) as conn:
            _enable_sqlite_vec(conn)
            current_dim = _read_vec_table_dimension(conn)
            if current_dim != target_dim:
                if current_dim > 0:
                    conn.execute("DROP TABLE IF EXISTS memory_vec")
                conn.execute(_create_vec_table_sql(target_dim))
            existing = int(conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0] or 0)
            stored = int(conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] or 0)
            if not force and existing >= stored and existing > 0:
                return 0
            conn.execute("DELETE FROM memory_vec")
            rows = conn.execute(
                """
                SELECT memory_id, vector_json FROM memory_embeddings
                WHERE dimensions = ?
                """,
                (target_dim,),
            ).fetchall()
            payloads: list[tuple[int, list[float]]] = []
            for memory_id, vector_json in rows:
                try:
                    vector = [float(v) for v in json.loads(str(vector_json))]
                except (TypeError, ValueError):
                    continue
                if len(vector) != target_dim:
                    continue
                payloads.append((int(memory_id), vector))
            _bulk_upsert_vec_rows(conn, payloads)
            conn.commit()
            return len(payloads)

    async def flush_pending_sync(self, max_rows: int | None = None) -> int:
        if not settings.memory_sync_enabled or not self._initialized:
            return 0

        rows = await asyncio.to_thread(
            self._read_pending_rows,
            max_rows or settings.memory_sync_batch_size,
        )
        synced_rows: list[int] = []
        for seq, payload_jsonl in rows:
            blob_client.append_memory_sync_event(payload_jsonl)
            synced_rows.append(seq)

        if synced_rows:
            await asyncio.to_thread(self._mark_rows_synced, synced_rows)
        return len(synced_rows)

    async def write_snapshot(self) -> bool:
        if not settings.memory_sync_enabled or not self._initialized:
            return False

        snapshot = await asyncio.to_thread(self._db_path.read_bytes)
        return blob_client.write_memory_snapshot(snapshot)

    async def _sync_loop(self) -> None:
        interval = max(1, settings.memory_flush_interval_seconds)
        while not self._stop_event.is_set():
            try:
                await self.flush_pending_sync(max_rows=settings.memory_sync_batch_size)
                now = asyncio.get_running_loop().time()
                if now >= self._snapshot_due_at:
                    await self.write_snapshot()
                    self._snapshot_due_at = now + max(1, settings.memory_snapshot_interval_seconds)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("memory sync loop iteration failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=float(interval))
            except TimeoutError:
                continue

    def _initialize_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._db_path.exists() or self._db_path.stat().st_size == 0:
            snapshot = blob_client.read_memory_snapshot()
            if snapshot:
                self._db_path.write_bytes(snapshot)
                logger.info("restored local memory db from blob snapshot")

        with sqlite3.connect(self._db_path) as conn:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)

            # sqlite-vec ANN table. Created lazily with the configured dim
            # so a brand-new db and a bootstrapped db both converge on the
            # same shape. The vec0 table is dropped and recreated only when
            # the stored dim differs from the configured dim.
            _enable_sqlite_vec(conn)
            current_dim = _read_vec_table_dimension(conn)
            target_dim = _vec_dim_from_settings()
            if current_dim != target_dim:
                if current_dim > 0:
                    logger.info(
                        "recreating memory_vec (dim %s -> %s)",
                        current_dim,
                        target_dim,
                    )
                    conn.execute("DROP TABLE IF EXISTS memory_vec")
                conn.execute(_create_vec_table_sql(target_dim))

            # Migration: Add workspace_id if missing in existing databases
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = [info[1] for info in cursor.fetchall()]
            if "workspace_id" not in columns:
                logger.info("migrating memory store: adding workspace_id column")
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'"
                )
            if "operation_id" not in columns:
                if "skill_id" in columns:
                    logger.info("migrating memory store: renaming skill_id column to operation_id")
                    conn.execute("ALTER TABLE memories RENAME COLUMN skill_id TO operation_id")
                else:
                    logger.info("migrating memory store: adding operation_id column")
                    conn.execute(
                        "ALTER TABLE memories ADD COLUMN operation_id TEXT NOT NULL DEFAULT ''"
                    )

            migrations = {
                "memory_scope": (
                    "ALTER TABLE memories ADD COLUMN memory_scope TEXT NOT NULL DEFAULT 'personal'"
                ),
                "surface_id": (
                    "ALTER TABLE memories ADD COLUMN surface_id TEXT NOT NULL DEFAULT ''"
                ),
                "source_user_id": (
                    "ALTER TABLE memories ADD COLUMN source_user_id TEXT NOT NULL DEFAULT ''"
                ),
                "source_user_slug": (
                    "ALTER TABLE memories ADD COLUMN source_user_slug TEXT NOT NULL DEFAULT ''"
                ),
            }
            for column_name, statement in migrations.items():
                if column_name not in columns:
                    logger.info("migrating memory store: adding %s column", column_name)
                    conn.execute(statement)

            log_cursor = conn.execute("PRAGMA table_info(memory_operation_log)")
            log_columns = {info[1] for info in log_cursor.fetchall()}
            if "trigger_operation" not in log_columns:
                if "trigger_skill" in log_columns:
                    logger.info(
                        "migrating memory store: renaming trigger_skill column to trigger_operation"
                    )
                    conn.execute(
                        "ALTER TABLE memory_operation_log "
                        "RENAME COLUMN trigger_skill TO trigger_operation"
                    )
                else:
                    logger.info("migrating memory store: adding trigger_operation column")
                    conn.execute(
                        "ALTER TABLE memory_operation_log "
                        "ADD COLUMN trigger_operation TEXT NOT NULL DEFAULT ''"
                    )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories (workspace_id, memory_scope, surface_id, user_slug, id)
                """
            )

            # Phase 2 uplift: new lifecycle columns
            lifecycle_migrations = {
                "memory_kind": (
                    "ALTER TABLE memories ADD COLUMN memory_kind TEXT NOT NULL DEFAULT 'episodic'"
                ),
                "confidence": (
                    "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0"
                ),
                "evidence_ids": (
                    "ALTER TABLE memories ADD COLUMN evidence_ids TEXT NOT NULL DEFAULT '[]'"
                ),
                "supersedes_id": "ALTER TABLE memories ADD COLUMN supersedes_id INTEGER",
                "valid_until": "ALTER TABLE memories ADD COLUMN valid_until TEXT",
                "last_accessed_at": "ALTER TABLE memories ADD COLUMN last_accessed_at TEXT",
                "access_count": (
                    "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
                ),
                "status": ("ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"),
            }
            cursor2 = conn.execute("PRAGMA table_info(memories)")
            existing_cols = {info[1] for info in cursor2.fetchall()}
            for col, stmt in lifecycle_migrations.items():
                if col not in existing_cols:
                    logger.info("migrating memory store: adding %s column", col)
                    try:
                        conn.execute(stmt)
                    except Exception:  # pylint: disable=broad-except
                        pass

            conn.commit()

    def _insert_memory(self, request: MemoryWriteRequest) -> int:
        """Write one memory row plus its sync-queue payload to SQLite."""
        created_at = datetime.now(UTC).isoformat()
        content_json = request.content_json or {}
        source_user_id = request.source_user_id or request.user_id
        source_user_slug = request.source_user_slug or request.user_slug
        payload = {
            "op": "memory_write",
            "ts": created_at,
            "workspace_id": request.workspace_id,
            "user_id": request.user_id,
            "user_slug": request.user_slug,
            "memory_scope": request.memory_scope,
            "surface_id": request.surface_id,
            "source_user_id": source_user_id,
            "source_user_slug": source_user_slug,
            "agent_id": request.agent_id,
            "operation_id": request.operation_id,
            "instruction_id": request.instruction_id,
            "memory_tier": request.memory_tier,
            "content": request.content,
            "content_json": content_json,
            "salience": request.salience,
        }

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (
                    workspace_id,
                    user_id,
                    user_slug,
                    memory_scope,
                    surface_id,
                    source_user_id,
                    source_user_slug,
                    agent_id,
                    operation_id,
                    instruction_id,
                    memory_tier,
                    content,
                    content_json,
                    salience,
                    created_at,
                    memory_kind,
                    confidence,
                    evidence_ids,
                    supersedes_id,
                    valid_until
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.workspace_id,
                    request.user_id,
                    request.user_slug,
                    request.memory_scope,
                    request.surface_id,
                    source_user_id,
                    source_user_slug,
                    request.agent_id,
                    request.operation_id,
                    request.instruction_id,
                    request.memory_tier,
                    request.content,
                    json.dumps(content_json, separators=(",", ":")),
                    request.salience,
                    created_at,
                    request.memory_kind,
                    request.confidence,
                    json.dumps(request.evidence_ids, separators=(",", ":")),
                    request.supersedes_id,
                    request.valid_until,
                ),
            )
            memory_id = int(cursor.lastrowid or 0)
            payload["memory_id"] = memory_id
            cursor.execute(
                "INSERT INTO sync_queue (payload_jsonl, created_at, synced_at) VALUES (?, ?, NULL)",
                (json.dumps(payload, separators=(",", ":")), created_at),
            )
            conn.commit()
        return memory_id

    def _read_pending_rows(self, max_rows: int) -> list[tuple[int, str]]:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT seq, payload_jsonl
                FROM sync_queue
                WHERE synced_at IS NULL
                ORDER BY seq ASC
                LIMIT ?
                """,
                (max_rows,),
            )
            return [(int(row[0]), str(row[1])) for row in cursor.fetchall()]

    def _mark_rows_synced(self, seqs: list[int]) -> None:
        synced_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "UPDATE sync_queue SET synced_at = ? WHERE seq = ?",
                [(synced_at, seq) for seq in seqs],
            )
            conn.commit()

    def _list_memories(
        self,
        user_slug: str,
        operation_ids: list[str],
        memory_tiers: list[str],
        limit: int,
        workspace_id: str = "default",
        memory_scope: MemoryScope = "personal",
        surface_id: str = "",
    ) -> list[MemoryRecord]:
        """Read recent memory rows with optional operation and tier filters."""
        where_clauses = ["workspace_id = ?", "memory_scope = ?"]
        params: list[object] = [workspace_id, memory_scope]

        if memory_scope == "personal":
            where_clauses.append("user_slug = ?")
            params.append(user_slug)
        else:
            where_clauses.append("surface_id = ?")
            params.append(surface_id)

        if operation_ids:
            placeholders = ", ".join("?" for _ in operation_ids)
            where_clauses.append(f"operation_id IN ({placeholders})")
            params.extend(operation_ids)

        if memory_tiers:
            placeholders = ", ".join("?" for _ in memory_tiers)
            where_clauses.append(f"memory_tier IN ({placeholders})")
            params.extend(memory_tiers)

        params.append(limit)
        query = (
            "SELECT id, workspace_id, user_id, user_slug, memory_scope, surface_id, "
            "source_user_id, source_user_slug, agent_id, operation_id, instruction_id, "
            "memory_tier, content, content_json, salience, created_at, "
            "memory_kind, confidence, evidence_ids, supersedes_id, valid_until, "
            "last_accessed_at, access_count, status "
            f"FROM memories WHERE {' AND '.join(where_clauses)}"
            " AND status NOT IN ('retired', 'disputed') ORDER BY id DESC LIMIT ?"
        )

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return self._rows_to_memory_records(rows)

    def _get_memory_records_by_ids(self, memory_ids: list[int]) -> list[MemoryRecord]:
        """Read specific memory rows by id."""
        if not memory_ids:
            return []

        placeholders = ", ".join("?" for _ in memory_ids)
        query = (
            "SELECT id, workspace_id, user_id, user_slug, memory_scope, surface_id, "
            "source_user_id, source_user_slug, agent_id, operation_id, instruction_id, "
            "memory_tier, content, content_json, salience, created_at, "
            "memory_kind, confidence, evidence_ids, supersedes_id, valid_until, "
            "last_accessed_at, access_count, status "
            f"FROM memories WHERE id IN ({placeholders}) ORDER BY id DESC"
        )

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(query, memory_ids)
            rows = cursor.fetchall()

        return self._rows_to_memory_records(rows)

    def _get_memory_embeddings(self, memory_ids: list[int]) -> dict[int, MemoryEmbeddingRecord]:
        """Read persisted memory embeddings for the given memory ids."""
        if not memory_ids:
            return {}

        placeholders = ", ".join("?" for _ in memory_ids)
        query = (
            "SELECT memory_id, workspace_id, user_slug, provider, model_id, content_hash, "
            "dimensions, vector_json, created_at, updated_at "
            f"FROM memory_embeddings WHERE memory_id IN ({placeholders})"
        )

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(query, memory_ids)
            rows = cursor.fetchall()

        embeddings: dict[int, MemoryEmbeddingRecord] = {}
        for row in rows:
            embeddings[int(row[0])] = MemoryEmbeddingRecord(
                memory_id=int(row[0]),
                workspace_id=str(row[1]),
                user_slug=str(row[2]),
                provider=str(row[3]),
                model_id=str(row[4]),
                content_hash=str(row[5]),
                dimensions=int(row[6]),
                vector=[float(value) for value in json.loads(str(row[7]) or "[]")],
                created_at=str(row[8]),
                updated_at=str(row[9]),
            )
        return embeddings

    def _upsert_memory_embeddings(self, rows: list[MemoryEmbeddingRecord]) -> None:
        """Insert or update persisted embedding rows for semantic recall."""
        if not rows:
            return

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                """
                INSERT INTO memory_embeddings (
                    memory_id,
                    workspace_id,
                    user_slug,
                    provider,
                    model_id,
                    content_hash,
                    dimensions,
                    vector_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    workspace_id = excluded.workspace_id,
                    user_slug = excluded.user_slug,
                    provider = excluded.provider,
                    model_id = excluded.model_id,
                    content_hash = excluded.content_hash,
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        row.memory_id,
                        row.workspace_id,
                        row.user_slug,
                        row.provider,
                        row.model_id,
                        row.content_hash,
                        row.dimensions,
                        json.dumps(row.vector, separators=(",", ":")),
                        row.created_at,
                        row.updated_at,
                    )
                    for row in rows
                ],
            )
            conn.commit()

    def evict_memories_for_consolidation(
        self,
        *,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        user_slug: str = "",
    ) -> dict[str, int]:
        """Retire low-value memories under the configured caps and floors.

        Returns counts of the actions taken, suitable for logging.
        The job is non-destructive: rows are marked ``status='retired'``
        and a sync_queue event is emitted so blob snapshots converge.
        Active records cannot be unretired without an explicit
        ``replace_memory_by_id`` call.

        Policy, in order:

        1. Age cap: any active memory older than ``memory_eviction_default_ttl_days``
           is retired unless it is pinned by salience or access count.
        2. Salience floor: anything below ``memory_eviction_min_salience`` and older
           than ``memory_eviction_min_age_days`` is retired.
        3. Per-scope cap: per-user (personal) or per-surface (group/chat) active count
           is held under ``memory_eviction_max_active_per_scope``.
        4. Per-kind cap: episodic, semantic, procedural, reflection, and preference
           memories each have an independent cap from ``memory_eviction_max_kind_per_user``.
        """
        stats: dict[str, int] = {
            "age_retired": 0,
            "salience_retired": 0,
            "scope_cap_retired": 0,
            "kind_cap_retired": 0,
        }
        if not settings.memory_eviction_enabled:
            return stats
        cutoff_iso = (
            datetime.now(UTC) - timedelta(days=int(settings.memory_eviction_min_age_days))
        ).isoformat()
        ttl_cutoff_iso = (
            datetime.now(UTC) - timedelta(days=int(settings.memory_eviction_default_ttl_days))
        ).isoformat()
        min_salience = float(settings.memory_eviction_min_salience)
        min_access = int(settings.memory_eviction_min_access_count)
        max_active = int(settings.memory_eviction_max_active_per_scope)
        per_kind_caps_raw = settings.memory_eviction_max_kind_per_user or {}
        per_kind_caps: dict[str, int] = {
            str(k): int(v) for k, v in per_kind_caps_raw.items() if int(v) > 0
        }

        with sqlite3.connect(self._db_path) as conn:
            # 1) Age cap: any active memory older than the TTL floor is retired
            #    unless it is pinned by salience or access count.
            rows = conn.execute(
                """
                SELECT id FROM memories
                WHERE workspace_id = ? AND memory_scope = ? AND status = 'active'
                  AND created_at < ?
                  AND (salience < ? OR access_count < ?)
                """,
                (
                    workspace_id,
                    memory_scope,
                    ttl_cutoff_iso,
                    min_salience,
                    min_access,
                ),
            ).fetchall()
            age_ids = [int(r[0]) for r in rows]
            if age_ids:
                self._bulk_retire_sync(
                    conn,
                    memory_ids=age_ids,
                    sync_op="memory_evict_age",
                    workspace_id=workspace_id,
                    memory_scope=memory_scope,
                    surface_id=surface_id,
                    user_slug=user_slug,
                    detail=f"age>{int(settings.memory_eviction_default_ttl_days)}d",
                )
                stats["age_retired"] = len(age_ids)

            # 2) Salience floor: anything with salience below the floor and
            #    below the min-age cutoff is retired (do not touch fresh items).
            rows = conn.execute(
                """
                SELECT id FROM memories
                WHERE workspace_id = ? AND memory_scope = ? AND status = 'active'
                  AND created_at < ?
                  AND salience < ?
                """,
                (
                    workspace_id,
                    memory_scope,
                    cutoff_iso,
                    min_salience,
                ),
            ).fetchall()
            salience_ids = [int(r[0]) for r in rows]
            if salience_ids:
                self._bulk_retire_sync(
                    conn,
                    memory_ids=salience_ids,
                    sync_op="memory_evict_salience",
                    workspace_id=workspace_id,
                    memory_scope=memory_scope,
                    surface_id=surface_id,
                    user_slug=user_slug,
                    detail=f"salience<{min_salience}",
                )
                stats["salience_retired"] = len(salience_ids)

            # 3) Per-scope active cap. For personal scope, cap per user_slug.
            if memory_scope == "personal":
                if not user_slug:
                    # When called without a slug, sum across users and apply the cap globally.
                    count_row = conn.execute(
                        """
                        SELECT COUNT(*) FROM memories
                        WHERE workspace_id = ? AND memory_scope = ? AND status = 'active'
                        """,
                        (workspace_id, memory_scope),
                    ).fetchone()
                    active_total = int(count_row[0] or 0)
                    if active_total > max_active:
                        overflow = active_total - max_active
                        rows = conn.execute(
                            """
                            SELECT id FROM memories
                            WHERE workspace_id = ? AND memory_scope = ? AND status = 'active'
                            ORDER BY salience ASC, access_count ASC, created_at ASC
                            LIMIT ?
                            """,
                            (workspace_id, memory_scope, int(overflow)),
                        ).fetchall()
                        overflow_ids = [int(r[0]) for r in rows]
                        if overflow_ids:
                            self._bulk_retire_sync(
                                conn,
                                memory_ids=overflow_ids,
                                sync_op="memory_evict_scope_cap",
                                workspace_id=workspace_id,
                                memory_scope=memory_scope,
                                surface_id=surface_id,
                                user_slug="",
                                detail=f"scope>cap({max_active})",
                            )
                            stats["scope_cap_retired"] = len(overflow_ids)
                else:
                    count_row = conn.execute(
                        """
                        SELECT COUNT(*) FROM memories
                        WHERE workspace_id = ? AND memory_scope = ? AND user_slug = ?
                          AND status = 'active'
                        """,
                        (workspace_id, memory_scope, user_slug),
                    ).fetchone()
                    active_user = int(count_row[0] or 0)
                    if active_user > max_active:
                        overflow = active_user - max_active
                        rows = conn.execute(
                            """
                            SELECT id FROM memories
                            WHERE workspace_id = ? AND memory_scope = ? AND user_slug = ?
                              AND status = 'active'
                            ORDER BY salience ASC, access_count ASC, created_at ASC
                            LIMIT ?
                            """,
                            (
                                workspace_id,
                                memory_scope,
                                user_slug,
                                int(overflow),
                            ),
                        ).fetchall()
                        overflow_ids = [int(r[0]) for r in rows]
                        if overflow_ids:
                            self._bulk_retire_sync(
                                conn,
                                memory_ids=overflow_ids,
                                sync_op="memory_evict_scope_cap",
                                workspace_id=workspace_id,
                                memory_scope=memory_scope,
                                surface_id=surface_id,
                                user_slug=user_slug,
                                detail=f"user>cap({max_active})",
                            )
                            stats["scope_cap_retired"] = len(overflow_ids)
            else:
                if not surface_id:
                    return stats
                count_row = conn.execute(
                    """
                    SELECT COUNT(*) FROM memories
                    WHERE workspace_id = ? AND memory_scope = ? AND surface_id = ?
                      AND status = 'active'
                    """,
                    (workspace_id, memory_scope, surface_id),
                ).fetchone()
                active_surface = int(count_row[0] or 0)
                if active_surface > max_active:
                    overflow = active_surface - max_active
                    rows = conn.execute(
                        """
                        SELECT id FROM memories
                        WHERE workspace_id = ? AND memory_scope = ? AND surface_id = ?
                          AND status = 'active'
                        ORDER BY salience ASC, access_count ASC, created_at ASC
                        LIMIT ?
                        """,
                        (
                            workspace_id,
                            memory_scope,
                            surface_id,
                            int(overflow),
                        ),
                    ).fetchall()
                    overflow_ids = [int(r[0]) for r in rows]
                    if overflow_ids:
                        self._bulk_retire_sync(
                            conn,
                            memory_ids=overflow_ids,
                            sync_op="memory_evict_scope_cap",
                            workspace_id=workspace_id,
                            memory_scope=memory_scope,
                            surface_id=surface_id,
                            user_slug=user_slug,
                            detail=f"surface>cap({max_active})",
                        )
                        stats["scope_cap_retired"] = len(overflow_ids)

            # 4) Per-kind cap. Retired after the scope cap so we never evict
            #    before knowing the user really has more than the cap.
            for kind, cap in per_kind_caps.items():
                where_clauses = [
                    "workspace_id = ?",
                    "memory_scope = ?",
                    "memory_kind = ?",
                    "status = 'active'",
                ]
                params: list[object] = [
                    workspace_id,
                    memory_scope,
                    kind,
                ]
                if memory_scope == "personal":
                    if not user_slug:
                        continue
                    where_clauses.append("user_slug = ?")
                    params.append(user_slug)
                else:
                    if not surface_id:
                        continue
                    where_clauses.append("surface_id = ?")
                    params.append(surface_id)

                count_sql = f"SELECT COUNT(*) FROM memories WHERE {' AND '.join(where_clauses)}"
                active_kind = int(conn.execute(count_sql, params).fetchone()[0] or 0)
                if active_kind <= cap:
                    continue
                overflow = active_kind - cap
                rows = conn.execute(
                    f"""
                    SELECT id FROM memories
                    WHERE {" AND ".join(where_clauses)}
                    ORDER BY salience ASC, access_count ASC, created_at ASC
                    LIMIT ?
                    """,
                    [*params, int(overflow)],
                ).fetchall()
                kind_ids = [int(r[0]) for r in rows]
                if kind_ids:
                    self._bulk_retire_sync(
                        conn,
                        memory_ids=kind_ids,
                        sync_op="memory_evict_kind_cap",
                        workspace_id=workspace_id,
                        memory_scope=memory_scope,
                        surface_id=surface_id,
                        user_slug=user_slug,
                        detail=f"{kind}>cap({cap})",
                    )
                    stats["kind_cap_retired"] = len(kind_ids)

            conn.commit()
        return stats

    def _bulk_retire_sync(
        self,
        conn: sqlite3.Connection,
        *,
        memory_ids: list[int],
        sync_op: str,
        workspace_id: str,
        memory_scope: str,
        surface_id: str,
        user_slug: str,
        detail: str,
    ) -> None:
        """Retire a list of memory ids in one transaction + emit one sync event."""
        if not memory_ids:
            return
        now_iso = datetime.now(UTC).isoformat()
        placeholders = ", ".join("?" for _ in memory_ids)
        conn.execute(
            f"""
            UPDATE memories
            SET status = 'retired', valid_until = ?
            WHERE id IN ({placeholders}) AND status = 'active'
            """,
            [now_iso, *memory_ids],
        )
        payload = {
            "op": sync_op,
            "ts": now_iso,
            "workspace_id": workspace_id,
            "memory_scope": memory_scope,
            "surface_id": surface_id,
            "user_slug": user_slug,
            "memory_ids": memory_ids,
            "detail": detail,
            "retired_count": len(memory_ids),
        }
        conn.execute(
            "INSERT INTO sync_queue (payload_jsonl, created_at, synced_at) VALUES (?, ?, NULL)",
            (json.dumps(payload, separators=(",", ":")), now_iso),
        )

    def list_active_scopes(self) -> list[dict[str, str]]:
        """Return one row per (memory_scope, surface_id, user_slug) with at least one active memory.

        The consolidation job uses this to discover which scopes need an
        eviction pass without having to walk every user/surface.
        """
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT memory_scope, surface_id, user_slug
                FROM memories
                WHERE status = 'active'
                """
            ).fetchall()
        return [
            {
                "memory_scope": str(scope or "personal"),
                "surface_id": str(surface or ""),
                "user_slug": str(user or ""),
            }
            for scope, surface, user in rows
        ]

    def _log_memory_op(
        self,
        *,
        op: str,
        memory_ids: list[int],
        query_text: str = "",
        injected_tokens: int = 0,
        latency_ms: int = 0,
        trigger_operation: str = "",
    ) -> None:
        """Append one row to the memory_operation_log table (non-fatal)."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO memory_operation_log "
                    "(op, memory_ids, query_text, injected_tokens,"
                    " latency_ms, trigger_operation, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        op,
                        json.dumps(memory_ids, separators=(",", ":")),
                        query_text,
                        injected_tokens,
                        latency_ms,
                        trigger_operation,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                conn.commit()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("memory op log write failed (non-fatal): %s", exc)

    def _update_memory_access(self, memory_ids: list[int]) -> None:
        """Increment access_count and set last_accessed_at for the given memory IDs."""
        if not memory_ids:
            return
        now_iso = datetime.now(UTC).isoformat()
        placeholders = ", ".join("?" for _ in memory_ids)
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    f"UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? "
                    f"WHERE id IN ({placeholders})",
                    [now_iso, *memory_ids],
                )
                conn.commit()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory access update failed: %s", exc)

    def _rows_to_memory_records(self, rows: list[tuple[object, ...]]) -> list[MemoryRecord]:
        """Convert raw SQLite rows into typed memory records."""
        records: list[MemoryRecord] = []
        for row in rows:
            memory_scope = cast(MemoryScope, str(row[4]) or "personal")
            memory_tier = cast(MemoryTier, str(row[11]) or "daily")
            memory_kind = cast(
                MemoryKindValue,
                str(row[16]) if row[16] else "episodic",
            )
            memory_status = cast(MemoryStatusValue, str(row[23]) if row[23] else "active")
            salience_value = float(row[14] or 0.0)
            confidence_value = float(row[17] or 1.0)
            supersedes_id = None
            if row[19] is not None:
                supersedes_id = int(cast(Literal[0] | int | str, row[19]))
            access_count = int(row[22] or 0)
            records.append(
                MemoryRecord(
                    id=int(row[0]),
                    workspace_id=str(row[1]),
                    user_id=str(row[2]),
                    user_slug=str(row[3]),
                    memory_scope=memory_scope,
                    surface_id=str(row[5]),
                    source_user_id=str(row[6]),
                    source_user_slug=str(row[7]),
                    agent_id=str(row[8]),
                    operation_id=str(row[9]),
                    instruction_id=str(row[10]),
                    memory_tier=memory_tier,
                    content=str(row[12]),
                    content_json=json.loads(str(row[13]) or "{}"),
                    salience=salience_value,
                    created_at=str(row[15]),
                    memory_kind=memory_kind,
                    confidence=confidence_value,
                    evidence_ids=json.loads(str(row[18]) or "[]"),
                    supersedes_id=supersedes_id,
                    valid_until=str(row[20]) if row[20] else None,
                    last_accessed_at=str(row[21]) if row[21] else None,
                    access_count=access_count,
                    status=memory_status,
                )
            )
        return records


_default_memory_store: IPSARMemoryStore | None = None


def get_memory_store() -> IPSARMemoryStore:
    global _default_memory_store  # pylint: disable=global-statement
    if _default_memory_store is None:
        _default_memory_store = IPSARMemoryStore()
    return _default_memory_store


def reset_memory_store_for_testing() -> None:
    global _default_memory_store  # pylint: disable=global-statement
    _default_memory_store = None
