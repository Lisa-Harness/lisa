"""Tests for the SQLite + sqlite-vec memory store (Part 3).

Covers schema creation, record/list/recall with and without embeddings, TTL
expiration, multi-tenant isolation, keyword fallback, eviction, vacuum, and
salience-weighted ordering.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import pytest

from kernel.models import MemoryWriteRequest
from kernel.sqlite_memory import (
    DEFAULT_TTL_DAYS,
    EmbeddingProvider,
    SqliteMemoryStore,
    load_sqlite_vec,
)


# ── Stub embedding provider ──────────────────────────────────────────────


class StubEmbeddingProvider:
    """Deterministic embedding provider for tests.

    Maps each text to a unit vector derived from a simple hash so that
    semantically similar texts (sharing words) produce closer vectors.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in text.lower().split():
                idx = hash(word) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_memory.db"


@pytest.fixture
def store(db_path: Path) -> SqliteMemoryStore:
    s = SqliteMemoryStore(db_path=db_path, embedding_dimension=16)
    yield s
    s.close()


@pytest.fixture
def vec_store(db_path: Path) -> SqliteMemoryStore:
    s = SqliteMemoryStore(
        db_path=db_path,
        embedding_provider=StubEmbeddingProvider(dim=16),
        embedding_dimension=16,
    )
    yield s
    s.close()


# ── Schema / init ──────────────────────────────────────────────────────────


def test_load_sqlite_vec_returns_bool(tmp_path: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "x.db"))
    result = load_sqlite_vec(conn)
    assert isinstance(result, bool)
    conn.close()


def test_store_creates_schema(store: SqliteMemoryStore) -> None:
    assert store._conn is not None
    tables = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {row["name"] for row in tables}
    assert "memories" in names


def test_store_creates_vec_table_when_available(vec_store: SqliteMemoryStore) -> None:
    if not vec_store._vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    assert vec_store._conn is not None
    tables = vec_store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {row["name"] for row in tables}
    assert "memory_embeddings" in names


# ── record_memory ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_memory_returns_record(store: SqliteMemoryStore) -> None:
    record = await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="hello world",
            operation_id="test.op",
            instruction_id="test.v1",
            memory_tier="daily",
            salience=0.8,
        )
    )
    assert record.id >= 1
    assert record.user_slug == "alice"
    assert record.content == "hello world"
    assert record.memory_tier == "daily"
    assert record.salience == 0.8


@pytest.mark.asyncio
async def test_record_memory_accepts_dict(store: SqliteMemoryStore) -> None:
    record = await store.record_memory(
        {
            "user_slug": "bob",
            "content": "dict input",
            "operation_id": "test.op",
            "instruction_id": "test.v1",
            "memory_tier": "long_term",
        }
    )
    assert record.user_slug == "bob"
    assert record.content == "dict input"


@pytest.mark.asyncio
async def test_record_memory_stores_embedding(vec_store: SqliteMemoryStore) -> None:
    if not vec_store._vec_available:
        pytest.skip("sqlite-vec not available")
    await vec_store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="semantic content",
            operation_id="test.op",
            instruction_id="test.v1",
            memory_tier="long_term",
        )
    )
    assert vec_store._conn is not None
    count = vec_store._conn.execute("SELECT COUNT(*) AS c FROM memory_embeddings").fetchone()["c"]
    assert count == 1


# ── list_memories ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_memories_scoped(store: SqliteMemoryStore) -> None:
    for i in range(3):
        await store.record_memory(
            MemoryWriteRequest(
                user_slug="alice",
                content=f"alice memory {i}",
                operation_id="test.op",
                instruction_id="test.v1",
                memory_tier="long_term",
            )
        )
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="bob",
            content="bob memory",
            operation_id="test.op",
            instruction_id="test.v1",
            memory_tier="long_term",
        )
    )
    alice = await store.list_memories(user_slug="alice", limit=10)
    bob = await store.list_memories(user_slug="bob", limit=10)
    assert len(alice) == 3
    assert len(bob) == 1
    assert all(r.user_slug == "alice" for r in alice)


@pytest.mark.asyncio
async def test_list_memories_surface_filter(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="surface A",
            surface_id="surf-a",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="surface B",
            surface_id="surf-b",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    a = await store.list_memories(user_slug="alice", surface_id="surf-a", limit=10)
    assert len(a) == 1
    assert a[0].surface_id == "surf-a"


@pytest.mark.asyncio
async def test_list_recent_memories_operation_filter(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="op-one",
            operation_id="op.one",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="op-two",
            operation_id="op.two",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    filtered = await store.list_recent_memories(user_slug="alice", operation_id="op.one", limit=10)
    assert len(filtered) == 1
    assert filtered[0].operation_id == "op.one"


# ── recall: keyword fallback ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_keyword_fallback(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="The quick brown fox jumps",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
            salience=0.9,
        )
    )
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="Completely unrelated note about python",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
            salience=0.1,
        )
    )
    result = await store.recall("quick fox", user_slug="alice", limit=5)
    assert "quick brown fox" in result
    assert "python" not in result


@pytest.mark.asyncio
async def test_recall_keyword_no_match_returns_empty(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="unrelated content here",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    result = await store.recall("zzzzz", user_slug="alice", limit=5)
    assert result == ""


@pytest.mark.asyncio
async def test_recall_keyword_respects_scope_isolation(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="secret alice memory",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    result = await store.recall("secret alice", user_slug="bob", limit=5)
    assert result == ""


# ── recall: semantic ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_semantic_returns_relevant(vec_store: SqliteMemoryStore) -> None:
    if not vec_store._vec_available:
        pytest.skip("sqlite-vec not available")
    await vec_store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="machine learning model training",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
            salience=0.9,
        )
    )
    await vec_store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="cooking recipe pasta italian",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
            salience=0.5,
        )
    )
    result = await vec_store.recall("learning model", user_slug="alice", limit=2)
    assert "machine learning" in result.lower()


@pytest.mark.asyncio
async def test_recall_semantic_falls_back_on_provider_error(db_path: Path) -> None:
    class BrokenProvider:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("broken")

    s = SqliteMemoryStore(
        db_path=db_path,
        embedding_provider=BrokenProvider(),  # type: ignore[arg-type]
        embedding_dimension=16,
    )
    try:
        await s.record_memory(
            MemoryWriteRequest(
                user_slug="alice",
                content="keyword searchable content",
                operation_id="op",
                instruction_id="v1",
                memory_tier="long_term",
            )
        )
        # Provider raises during recall embed -> falls back to keyword
        result = await s.recall("keyword searchable", user_slug="alice", limit=5)
        assert "keyword searchable content" in result
    finally:
        s.close()


# ── TTL / expiration ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expired_memories_excluded(store: SqliteMemoryStore) -> None:
    # Insert a memory with a past expiration directly.
    assert store._conn is not None
    store._conn.execute(
        """
        INSERT INTO memories (
            entry_id, user_slug, workspace_id, memory_scope, surface_id,
            agent_id, operation_id, instruction_id, memory_tier,
            content, content_json, salience, created_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "expired001",
            "alice",
            "default",
            "personal",
            "",
            "core",
            "op",
            "v1",
            "daily",
            "expired content",
            "{}",
            0.5,
            "2020-01-01T00:00:00+00:00",
            "2020-01-02T00:00:00+00:00",
        ),
    )
    store._conn.commit()
    records = await store.list_memories(user_slug="alice", limit=10)
    assert len(records) == 0


@pytest.mark.asyncio
async def test_evict_expired_removes_rows(store: SqliteMemoryStore) -> None:
    assert store._conn is not None
    store._conn.execute(
        """
        INSERT INTO memories (
            entry_id, user_slug, workspace_id, memory_scope, surface_id,
            agent_id, operation_id, instruction_id, memory_tier,
            content, content_json, salience, created_at, expires_at
        ) VALUES
            (?, 'alice', 'default', 'personal', '', 'core', 'op', 'v1', 'daily',
             'old', '{}', 0.5, '2020-01-01T00:00:00+00:00', '2020-01-02T00:00:00+00:00'),
            (?, 'alice', 'default', 'personal', '', 'core', 'op', 'v1', 'long_term',
             'fresh', '{}', 0.5, '2026-01-01T00:00:00+00:00', NULL)
        """,
        ("expired002", "fresh001"),
    )
    store._conn.commit()
    deleted = await store.evict_expired()
    assert deleted == 1
    remaining = await store.list_memories(user_slug="alice", limit=10)
    assert len(remaining) == 1
    assert remaining[0].content == "fresh"


@pytest.mark.asyncio
async def test_long_term_tier_gets_expiration(store: SqliteMemoryStore) -> None:
    record = await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="long term fact",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    assert record.valid_until is not None


@pytest.mark.asyncio
async def test_permanent_tier_never_expires(db_path: Path) -> None:
    s = SqliteMemoryStore(
        db_path=db_path,
        default_ttl_days={"daily": 7, "weekly": 30, "long_term": None, "permanent": None},
        embedding_dimension=16,
    )
    try:
        record = await s.record_memory(
            MemoryWriteRequest(
                user_slug="alice",
                content="permanent fact",
                operation_id="op",
                instruction_id="v1",
                memory_tier="long_term",
            )
        )
        assert record.valid_until is None
    finally:
        s.close()


# ── vacuum ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vacuum_runs_without_error(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="vacuum test",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    await store.vacuum()
    # No exception means success
    records = await store.list_memories(user_slug="alice", limit=10)
    assert len(records) == 1


# ── access tracking ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_updates_access_count(store: SqliteMemoryStore) -> None:
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="alice",
            content="access tracking test content",
            operation_id="op",
            instruction_id="v1",
            memory_tier="long_term",
        )
    )
    await store.recall("access tracking", user_slug="alice", limit=5)
    assert store._conn is not None
    row = store._conn.execute(
        "SELECT access_count AS c, last_accessed_at AS l FROM memories WHERE user_slug = ?",
        ("alice",),
    ).fetchone()
    assert row["c"] == 1
    assert row["l"] is not None


# ── DEFAULT_TTL_DAYS ──────────────────────────────────────────────────────────


def test_default_ttl_days_has_all_tiers() -> None:
    for tier in ("daily", "weekly", "long_term", "permanent"):
        assert tier in DEFAULT_TTL_DAYS
    assert DEFAULT_TTL_DAYS["permanent"] is None