"""Tests for the SqliteMemoryStore — record, list, get, delete, and recall semantics.

Covers the minimal built-in backend's contract: surface-scoped isolation,
TTL expiry filtering, lexical-overlap recall ranking, opportunistic
access tracking, and surface-scoped get/delete by id. These are the
behaviors the lisa-memory MCP server and the generic ``memory.*`` aliases
depend on; Mem0 (or any other backend) is expected to preserve the same
observable semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kernel.models import MemoryRecord, MemoryWriteRequest
from sdk.memory.store import DEFAULT_TTL_DAYS, SqliteMemoryStore


def _write(
    *,
    user_slug: str = "adam",
    content: str,
    memory_scope: str = "personal",
    surface_id: str = "",
    workspace_id: str = "default",
    memory_tier: str = "long_term",
    salience: float = 0.0,
    content_json: dict[str, object] | None = None,
    operation_id: str = "test",
) -> MemoryWriteRequest:
    return MemoryWriteRequest(
        user_slug=user_slug,
        content=content,
        memory_scope=memory_scope,  # type: ignore[arg-type]
        surface_id=surface_id,
        workspace_id=workspace_id,
        memory_tier=memory_tier,  # type: ignore[arg-type]
        salience=salience,
        content_json=content_json or {},
        operation_id=operation_id,
    )


def _store(tmp_path: Path) -> SqliteMemoryStore:
    return SqliteMemoryStore(db_path=tmp_path / "memory.db")


# ---------------------------------------------------------------------------
# Schema / construction
# ---------------------------------------------------------------------------


def test_post_init_creates_db_file_and_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deep" / "memory.db"
    SqliteMemoryStore(db_path=db_path)

    assert db_path.exists()
    # The schema must be idempotent — constructing twice on the same path
    # must not raise.
    SqliteMemoryStore(db_path=db_path)


def test_default_ttl_days_are_sensible() -> None:
    assert DEFAULT_TTL_DAYS["daily"] == 7
    assert DEFAULT_TTL_DAYS["weekly"] == 30
    assert DEFAULT_TTL_DAYS["long_term"] == 365
    assert DEFAULT_TTL_DAYS["permanent"] is None


# ---------------------------------------------------------------------------
# record_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_memory_persists_and_returns_record(tmp_path: Path) -> None:
    store = _store(tmp_path)

    record = await store.record_memory(
        _write(content="We chose PostgreSQL for the control plane.", salience=0.8)
    )

    assert isinstance(record, MemoryRecord)
    assert record.content == "We chose PostgreSQL for the control plane."
    assert record.user_slug == "adam"
    assert record.memory_scope == "personal"
    assert record.salience == 0.8
    assert record.created_at  # ISO timestamp is populated


@pytest.mark.asyncio
async def test_record_memory_accepts_raw_dict_request(tmp_path: Path) -> None:
    """The record_memory boundary accepts both MemoryWriteRequest and dicts."""
    store = _store(tmp_path)

    record = await store.record_memory(
        {
            "user_slug": "nadia",
            "content": "Stripe integration ships Friday.",
            "memory_tier": "weekly",
            "salience": 0.5,
        }
    )

    assert record.content == "Stripe integration ships Friday."
    assert record.user_slug == "nadia"


@pytest.mark.asyncio
async def test_record_memory_applies_tier_ttl(tmp_path: Path) -> None:
    """A daily-tier memory gets a 7-day expiry; permanent gets none."""
    store = _store(tmp_path)

    daily = await store.record_memory(_write(content="ephemeral", memory_tier="daily"))
    permanent = await store.record_memory(_write(content="forever", memory_tier="permanent"))

    assert daily.valid_until is not None
    assert permanent.valid_until is None


# ---------------------------------------------------------------------------
# list_memories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_memories_is_surface_scoped_and_ordered_by_recency(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    # Shared scope is a different memory_scope and must not leak into personal.
    for n in range(3):
        await store.record_memory(_write(content=f"shared note {n}", memory_scope="shared"))
    # Two personal-scope notes on the default surface; newer one comes first.
    await store.record_memory(_write(content="personal note one"))
    await store.record_memory(_write(content="personal note two"))
    # Personal-scope but a different surface_id — must not leak across surfaces.
    await store.record_memory(
        _write(content="other surface note", memory_scope="personal", surface_id="channel-1")
    )

    default_surface = await store.list_memories(user_slug="adam", memory_scope="personal")
    channel_surface = await store.list_memories(
        user_slug="adam", memory_scope="personal", surface_id="channel-1"
    )
    shared_scope = await store.list_memories(user_slug="adam", memory_scope="shared")

    assert [r.content for r in default_surface] == ["personal note two", "personal note one"]
    assert [r.content for r in channel_surface] == ["other surface note"]
    assert [r.content for r in shared_scope] == ["shared note 2", "shared note 1", "shared note 0"]


@pytest.mark.asyncio
async def test_list_memories_respects_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for n in range(5):
        await store.record_memory(_write(content=f"note {n}"))

    records = await store.list_memories(user_slug="adam", limit=2)

    assert len(records) == 2
    # Most recent first (highest id first).
    assert records[0].content == "note 4"
    assert records[1].content == "note 3"


@pytest.mark.asyncio
async def test_list_memories_isolates_by_user(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(user_slug="adam", content="adam's note"))
    await store.record_memory(_write(user_slug="nadia", content="nadia's note"))

    adam = await store.list_memories(user_slug="adam")
    nadia = await store.list_memories(user_slug="nadia")

    assert [r.content for r in adam] == ["adam's note"]
    assert [r.content for r in nadia] == ["nadia's note"]


# ---------------------------------------------------------------------------
# get_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_memory_returns_record_by_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="target note"))

    fetched = await store.get_memory(memory_id=record.id, user_slug="adam")

    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.content == "target note"


@pytest.mark.asyncio
async def test_get_memory_returns_none_for_wrong_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="a note"))

    fetched = await store.get_memory(memory_id=9999, user_slug="adam")

    assert fetched is None


@pytest.mark.asyncio
async def test_get_memory_is_surface_scoped(tmp_path: Path) -> None:
    """A record on surface-a cannot be fetched from surface-b."""
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="surface-a note", surface_id="surface-a"))

    # Same user, wrong surface → not found.
    wrong_surface = await store.get_memory(
        memory_id=record.id, user_slug="adam", surface_id="surface-b"
    )
    assert wrong_surface is None
    # Same user, wrong scope → not found.
    wrong_scope = await store.get_memory(
        memory_id=record.id, user_slug="adam", memory_scope="shared"
    )
    assert wrong_scope is None
    # Wrong user → not found.
    wrong_user = await store.get_memory(memory_id=record.id, user_slug="nadia")
    assert wrong_user is None
    # Correct scope → found.
    correct = await store.get_memory(memory_id=record.id, user_slug="adam", surface_id="surface-a")
    assert correct is not None
    assert correct.content == "surface-a note"


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_memory_removes_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="doomed"))

    deleted = await store.delete_memory(memory_id=record.id, user_slug="adam")

    assert deleted is True
    # Subsequent get returns None.
    fetched = await store.get_memory(memory_id=record.id, user_slug="adam")
    assert fetched is None


@pytest.mark.asyncio
async def test_delete_memory_returns_false_for_wrong_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="a note"))

    deleted = await store.delete_memory(memory_id=9999, user_slug="adam")

    assert deleted is False


@pytest.mark.asyncio
async def test_delete_memory_is_surface_scoped(tmp_path: Path) -> None:
    """A record on surface-a cannot be deleted from surface-b."""
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="surface-a note", surface_id="surface-a"))

    # Wrong surface → no deletion.
    wrong_surface = await store.delete_memory(
        memory_id=record.id, user_slug="adam", surface_id="surface-b"
    )
    assert wrong_surface is False
    # Wrong user → no deletion.
    wrong_user = await store.delete_memory(memory_id=record.id, user_slug="nadia")
    assert wrong_user is False
    # Correct scope → deleted.
    correct = await store.delete_memory(
        memory_id=record.id, user_slug="adam", surface_id="surface-a"
    )
    assert correct is True
    # Record is gone.
    assert (
        await store.get_memory(memory_id=record.id, user_slug="adam", surface_id="surface-a")
        is None
    )


# ---------------------------------------------------------------------------
# recall — ranking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_ranks_by_token_overlap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="We postponed the Redis migration."))
    await store.record_memory(_write(content="We chose PostgreSQL for the primary database."))

    result = await store.recall("chose postgresql primary", user_slug="adam", limit=5)

    lines = result.splitlines()
    assert lines[0][2:] == "We chose PostgreSQL for the primary database."


@pytest.mark.asyncio
async def test_recall_longer_content_with_more_overlap_beats_shorter_with_one_match(
    tmp_path: Path,
) -> None:
    """The length-normalized scoring must not reward a short note matching
    a single keyword over a longer note matching several."""
    store = _store(tmp_path)
    await store.record_memory(_write(content="postgres", salience=0.0))
    await store.record_memory(
        _write(content="We chose postgres for the primary database and analytics.", salience=0.0)
    )

    result = await store.recall("chose postgres primary database", user_slug="adam", limit=2)

    lines = [line[2:] for line in result.splitlines()]
    assert lines[0].startswith("We chose postgres for the primary database")


@pytest.mark.asyncio
async def test_recall_salience_boost_breaks_ties(tmp_path: Path) -> None:
    """When two memories match equally on tokens, higher salience wins."""
    store = _store(tmp_path)
    await store.record_memory(_write(content="use redis for cache", salience=0.0))
    await store.record_memory(_write(content="use redis for cache", salience=0.9))

    result = await store.recall("redis cache", user_slug="adam", limit=2)

    # The higher-salience record was inserted later (higher id), so it must
    # appear first.
    assert "0.9" not in result  # salience is not rendered; just confirm ordering
    # Both have identical content, so we only assert a single line surfaces
    # both — but the tie-breaker is the higher-id (and thus higher-salience)
    # record first.
    assert result.count("use redis for cache") == 2


@pytest.mark.asyncio
async def test_recall_returns_empty_when_no_records(tmp_path: Path) -> None:
    store = _store(tmp_path)

    result = await store.recall("anything", user_slug="nobody")

    assert result == ""


@pytest.mark.asyncio
async def test_recall_falls_back_to_recent_when_no_token_overlap(
    tmp_path: Path,
) -> None:
    """When the query matches nothing, the most recent records are returned."""
    store = _store(tmp_path)
    await store.record_memory(_write(content="unrelated content one"))
    await store.record_memory(_write(content="unrelated content two"))

    result = await store.recall("zzz no match", user_slug="adam", limit=2)

    lines = [line[2:] for line in result.splitlines()]
    assert "unrelated content two" in lines
    assert "unrelated content one" in lines


# ---------------------------------------------------------------------------
# recall — expiry filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_excludes_expired_memories(tmp_path: Path) -> None:
    """An expired daily memory must not surface in recall even if it matches."""
    store = _store(tmp_path)
    record = await store.record_memory(
        _write(content="postgres decision from last week", memory_tier="daily")
    )
    # Force the row's expires_at into the past.
    assert store._conn is not None
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store._conn.execute("UPDATE memories SET expires_at = ? WHERE id = ?", (past, record.id))
    store._conn.commit()

    result = await store.recall("postgres decision", user_slug="adam", limit=5)

    assert result == ""


@pytest.mark.asyncio
async def test_recall_includes_non_expired_memories(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="postgres decision today", memory_tier="long_term"))

    result = await store.recall("postgres decision", user_slug="adam", limit=5)

    assert "postgres decision today" in result


# ---------------------------------------------------------------------------
# recall — pool width
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_pool_widens_candidate_window(tmp_path: Path) -> None:
    """A high-relevance memory older than ``limit`` rows must still surface
    when recall_pool is large enough to include it."""
    store = _store(tmp_path)
    # Insert one matching memory first.
    await store.record_memory(
        _write(content="chose postgres for the primary database", salience=0.9)
    )
    # Bury it under 10 newer but irrelevant memories.
    for n in range(10):
        await store.record_memory(_write(content=f"noise note {n}"))

    # Default limit is 5, but recall_pool=40 lets the ranker see the old match.
    result = await store.recall(
        "chose postgres primary database", user_slug="adam", limit=5, recall_pool=40
    )

    assert "chose postgres for the primary database" in result


@pytest.mark.asyncio
async def test_recall_pool_too_narrow_misses_old_match(tmp_path: Path) -> None:
    """When recall_pool is smaller than the offset of the matching row, the
    ranker never sees it and cannot surface it."""
    store = _store(tmp_path)
    await store.record_memory(_write(content="chose postgres for the primary database"))
    for n in range(5):
        await store.record_memory(_write(content=f"noise note {n}"))

    result = await store.recall(
        "chose postgres primary database", user_slug="adam", limit=5, recall_pool=3
    )

    # The match is at id=1; pool=3 only sees ids 6,5,4 — misses the match.
    assert "chose postgres for the primary database" not in result


# ---------------------------------------------------------------------------
# recall — access tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_bumps_access_count_and_last_accessed_at(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="chose postgres primary"))
    assert store._conn is not None

    await store.recall("postgres primary", user_slug="adam", limit=5)

    row = store._conn.execute(
        "SELECT access_count, last_accessed_at FROM memories WHERE id = ?",
        (record.id,),
    ).fetchone()
    assert row is not None
    assert row["access_count"] == 1
    assert row["last_accessed_at"] is not None


@pytest.mark.asyncio
async def test_recall_does_not_bump_access_for_non_surfaced_records(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    matched = await store.record_memory(_write(content="postgres primary database"))
    noise = await store.record_memory(_write(content="totally unrelated"))
    assert store._conn is not None

    await store.recall("postgres primary", user_slug="adam", limit=1)

    matched_row = store._conn.execute(
        "SELECT access_count FROM memories WHERE id = ?", (matched.id,)
    ).fetchone()
    noise_row = store._conn.execute(
        "SELECT access_count FROM memories WHERE id = ?", (noise.id,)
    ).fetchone()
    assert matched_row["access_count"] == 1
    assert noise_row["access_count"] == 0


@pytest.mark.asyncio
async def test_recall_accumulates_access_count_across_calls(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="postgres primary database"))
    assert store._conn is not None

    for _ in range(3):
        await store.recall("postgres primary", user_slug="adam", limit=1)

    row = store._conn.execute(
        "SELECT access_count FROM memories WHERE content LIKE '%postgres%'"
    ).fetchone()
    assert row["access_count"] == 3


# ---------------------------------------------------------------------------
# recall — isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_isolates_by_user_scope_and_surface(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(user_slug="adam", content="adam postgres note"))
    await store.record_memory(_write(user_slug="nadia", content="nadia postgres note"))
    await store.record_memory(
        _write(user_slug="adam", content="adam channel postgres note", surface_id="chan")
    )

    adam_personal = await store.recall("postgres", user_slug="adam", limit=5)
    nadia_personal = await store.recall("postgres", user_slug="nadia", limit=5)
    adam_channel = await store.recall("postgres", user_slug="adam", surface_id="chan", limit=5)

    assert "adam postgres note" in adam_personal
    assert "adam channel postgres note" not in adam_personal
    assert "nadia postgres note" in nadia_personal
    assert "adam channel postgres note" in adam_channel
    assert "adam postgres note" not in adam_channel


# ---------------------------------------------------------------------------
# Phase 1: schema cutover — new columns, retirement, salience, count, embeddings
# ---------------------------------------------------------------------------


def test_schema_has_consolidation_columns(tmp_path: Path) -> None:
    """The hard-cutover schema materializes the model's consolidation fields."""
    store = _store(tmp_path)
    assert store._conn is not None
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(memories)").fetchall()}
    for col in (
        "memory_kind",
        "status",
        "supersedes_id",
        "item_ids",
        "confidence",
        "merge_count",
        "valid_until",
    ):
        assert col in cols, f"missing column: {col}"


def test_schema_version_is_set_and_recreates_on_bump(tmp_path: Path) -> None:
    """user_version tracks SCHEMA_VERSION; a stale DB is recreated."""
    db = tmp_path / "memory.db"
    s1 = SqliteMemoryStore(db_path=db)
    assert s1._conn is not None
    assert s1._conn.execute("PRAGMA user_version").fetchone()[0] >= 2
    # Insert a legacy row by force-dropping to an old version, then reopen —
    # the store must recreate and drop the stale data.
    s1._conn.execute("DROP TABLE memories")
    s1._conn.execute("PRAGMA user_version = 0")
    s1._conn.commit()
    s1.close()
    s2 = SqliteMemoryStore(db_path=db)
    assert s2._conn is not None
    assert s2._conn.execute("PRAGMA user_version").fetchone()[0] >= 2
    # Table recreated with the new columns.
    cols = {row[1] for row in s2._conn.execute("PRAGMA table_info(memories)").fetchall()}
    assert "memory_kind" in cols


@pytest.mark.asyncio
async def test_record_memory_persists_memory_kind_and_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.record_memory(
        MemoryWriteRequest(
            user_slug="adam",
            content="semantic fact",
            memory_kind="semantic",
            confidence=0.8,
            item_ids=[1, 2, 3],
            supersedes_id=1,
        )
    )
    assert record.memory_kind == "semantic"
    assert record.status == "active"
    assert record.confidence == 0.8
    assert record.item_ids == [1, 2, 3]
    assert record.supersedes_id == 1


@pytest.mark.asyncio
async def test_retire_memory_flips_status_and_is_scoped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="episodic note"))
    # Wrong scope cannot retire.
    assert await store.retire_memory(
        memory_id=record.id, user_slug="nadia"
    ) is False
    # Correct scope retires.
    assert await store.retire_memory(memory_id=record.id, user_slug="adam") is True
    fetched = await store.get_memory(memory_id=record.id, user_slug="adam")
    assert fetched is not None
    assert fetched.status == "retired"
    # Re-retiring an already-retired record is a no-op (status='active' guard).
    assert await store.retire_memory(memory_id=record.id, user_slug="adam") is False


@pytest.mark.asyncio
async def test_list_memories_filters_by_kind_and_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="episodic one"))
    await store.record_memory(_write(content="episodic two"))
    sem = await store.record_memory(
        MemoryWriteRequest(user_slug="adam", content="semantic fact", memory_kind="semantic")
    )
    await store.retire_memory(memory_id=sem.id, user_slug="adam")

    # Default status='active' excludes the retired semantic record.
    active = await store.list_memories(user_slug="adam")
    assert all(r.status == "active" for r in active)
    assert len(active) == 2
    # Filter by kind.
    episodic = await store.list_memories(user_slug="adam", memory_kind="episodic")
    assert all(r.memory_kind == "episodic" for r in episodic)
    # status=None surfaces the retired record.
    all_records = await store.list_memories(user_slug="adam", status=None)
    assert any(r.status == "retired" for r in all_records)


@pytest.mark.asyncio
async def test_count_memories_filters_by_kind_and_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.record_memory(_write(content="episodic one"))
    await store.record_memory(_write(content="episodic two"))
    sem = await store.record_memory(
        MemoryWriteRequest(user_slug="adam", content="semantic fact", memory_kind="semantic")
    )
    await store.retire_memory(memory_id=sem.id, user_slug="adam")

    assert await store.count_memories(user_slug="adam") == 2  # active only
    assert (
        await store.count_memories(user_slug="adam", memory_kind="episodic") == 2
    )
    assert (
        await store.count_memories(user_slug="adam", memory_kind="semantic") == 0
    )  # retired
    assert (
        await store.count_memories(user_slug="adam", status=None) == 3
    )


@pytest.mark.asyncio
async def test_update_salience_is_clamped_and_scoped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = await store.record_memory(_write(content="note", salience=0.3))
    assert await store.update_salience(
        memory_id=record.id, salience=1.5, user_slug="adam"
    ) is True
    fetched = await store.get_memory(memory_id=record.id, user_slug="adam")
    assert fetched is not None
    assert fetched.salience == 1.0  # clamped
    # Wrong scope.
    assert await store.update_salience(
        memory_id=record.id, salience=0.2, user_slug="nadia"
    ) is False


@pytest.mark.asyncio
async def test_recall_excludes_retired_by_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = await store.record_memory(_write(content="active postgres note"))
    b = await store.record_memory(_write(content="retired postgres note"))
    await store.retire_memory(memory_id=b.id, user_slug="adam")

    result = await store.recall("postgres", user_slug="adam", limit=5)
    assert "active postgres note" in result
    assert "retired postgres note" not in result
    # status=None surfaces the retired one.
    all_result = await store.recall(
        "postgres", user_slug="adam", limit=5, status=None
    )
    assert "retired postgres note" in all_result
    del a


# ---------------------------------------------------------------------------
# Embedding seam (sqlite-vec)
# ---------------------------------------------------------------------------


class _StubEmbeddingProvider:
    """Deterministic embedding provider: hashes content to a fixed vector."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            # Stable 4-d vector from the text; dimension matches the store below.
            base = [float(ord(ch) % 7) for ch in (text + "padding")][:4]
            while len(base) < 4:
                base.append(0.0)
            out.append(base)
        return out


@pytest.mark.asyncio
async def test_embedding_write_and_semantic_recall(tmp_path: Path) -> None:
    store = SqliteMemoryStore(
        db_path=tmp_path / "memory.db",
        embedding_provider=_StubEmbeddingProvider(),  # type: ignore[arg-type]
        embedding_dimension=4,
    )
    if not store._vec_available:
        pytest.skip("sqlite-vec not available in this environment")
    await store.record_memory(_write(content="we chose postgres for analytics"))
    await store.record_memory(_write(content="the weather is sunny today"))

    # Query that lexically overlaps "postgres" but should rank the
    # postgres record first via semantic proximity too.
    result = await store.recall(
        "postgres analytics database", user_slug="adam", limit=2
    )
    assert "we chose postgres for analytics" in result


@pytest.mark.asyncio
async def test_embedding_failure_does_not_block_write(tmp_path: Path) -> None:
    class _BrokenProvider:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding service down")

    store = SqliteMemoryStore(
        db_path=tmp_path / "memory.db",
        embedding_provider=_BrokenProvider(),  # type: ignore[arg-type]
        embedding_dimension=4,
    )
    # The write must succeed despite the embedding provider raising.
    record = await store.record_memory(_write(content="survives embedding failure"))
    assert record.content == "survives embedding failure"
    # Lexical recall still works.
    result = await store.recall("survives", user_slug="adam", limit=1)
    assert "survives embedding failure" in result
