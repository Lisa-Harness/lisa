"""Tests for kernel/lite_memory.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.models import MemoryWriteRequest

from kernel.lite_memory import LiteMemoryStore, _make_entry_id, _synthesize_content

# ── LiteMemoryStore tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_and_recall_single_entry(tmp_path: Path) -> None:
    """Store one entry, then recall it by keyword match."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    await store.store("what is my favorite color", "Your favorite color is blue.")

    result = await store.recall("what is my favorite color")
    assert "Your favorite color is blue" in result


@pytest.mark.asyncio
async def test_record_memory_and_list_memories_round_trip(tmp_path: Path) -> None:
    """Operation-facing write/read API persists canonical records."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    created = await store.record_memory(
        MemoryWriteRequest(
            workspace_id="little-lisa",
            user_slug="founder",
            memory_scope="personal",
            operation_id="save_learning",
            instruction_id="save_learning.v1",
            memory_tier="long_term",
            content="Founders should keep launch messaging anchored to conversion telemetry.",
            content_json={"source": "coach"},
            salience=0.9,
        )
    )

    assert created.id == 1
    rows = await store.list_memories(
        user_slug="founder",
        workspace_id="little-lisa",
        memory_scope="personal",
        limit=5,
    )
    assert len(rows) == 1
    assert rows[0].operation_id == "save_learning"
    assert rows[0].content_json["source"] == "coach"


@pytest.mark.asyncio
async def test_list_recent_memories_filters_by_operation_id(tmp_path: Path) -> None:
    """Recent-memory listing honors operation-id filtering."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    await store.record_memory(
        MemoryWriteRequest(
            user_slug="founder",
            memory_scope="personal",
            operation_id="save_learning",
            instruction_id="save_learning.v1",
            memory_tier="long_term",
            content="Durable decision A",
        )
    )
    await store.record_memory(
        MemoryWriteRequest(
            user_slug="founder",
            memory_scope="personal",
            operation_id="notes",
            instruction_id="notes.v1",
            memory_tier="daily",
            content="Durable decision B",
        )
    )

    filtered = await store.list_recent_memories(
        user_slug="founder",
        operation_id="save_learning",
        limit=10,
    )
    assert len(filtered) == 1
    assert filtered[0].operation_id == "save_learning"


@pytest.mark.asyncio
async def test_recall_returns_empty_when_no_match(tmp_path: Path) -> None:
    """recall returns empty string when no entries match."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    await store.store("favorite color", "blue")
    result = await store.recall("pizza toppings")

    assert result == ""


@pytest.mark.asyncio
async def test_recall_ignores_short_words(tmp_path: Path) -> None:
    """recall skips keywords of 2 chars or fewer."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    await store.store("color preference", "I like blue")
    # "my" (2 chars) and "is" (2 chars) are too short to be keywords
    result = await store.recall("my is")

    assert result == ""


@pytest.mark.asyncio
async def test_recall_returns_most_recent_first(tmp_path: Path) -> None:
    """Multiple matches are ordered by recency (newest first)."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    await store.store("favorite color", "blue")
    await store.store("favorite food", "pizza")
    await store.store("favorite animal", "dog")

    result = await store.recall("favorite")
    lines = result.split("\n")

    # Most recent first
    assert "dog" in lines[0]
    assert "pizza" in lines[1]
    assert "blue" in lines[2]


@pytest.mark.asyncio
async def test_recall_capped_at_five_entries(tmp_path: Path) -> None:
    """recall returns at most 5 entries."""
    store = LiteMemoryStore(tmp_path / "memory.json")

    for i in range(10):
        await store.store(f"item {i}", f"content {i}")

    result = await store.recall("item")
    lines = [line for line in result.split("\n") if line.strip()]

    assert len(lines) == 5


@pytest.mark.asyncio
async def test_store_persists_across_instances(tmp_path: Path) -> None:
    """Entries survive across separate LiteMemoryStore instances."""
    path = tmp_path / "memory.json"

    store1 = LiteMemoryStore(path)
    await store1.store("favorite color", "blue")

    store2 = LiteMemoryStore(path)
    result = await store2.recall("favorite color")

    assert "blue" in result


@pytest.mark.asyncio
async def test_new_file_starts_empty(tmp_path: Path) -> None:
    """A new store with no file starts with empty entries."""
    store = LiteMemoryStore(tmp_path / "nonexistent.json")

    result = await store.recall("anything")
    assert result == ""


@pytest.mark.asyncio
async def test_corrupt_json_is_recovered(tmp_path: Path) -> None:
    """Corrupt JSON is treated as empty (no crash)."""
    path = tmp_path / "memory.json"
    path.write_text("not valid json {{{", encoding="utf-8")

    store = LiteMemoryStore(path)
    result = await store.recall("anything")

    assert result == ""

    # Should still be able to write new entries
    await store.store("new entry", "still works")
    result2 = await store.recall("new entry")
    assert "still works" in result2


# ── Helper function tests ─────────────────────────────────────────────


def test_make_entry_id_is_unique() -> None:
    """_make_entry_id produces unique 8-char IDs."""
    ids = {_make_entry_id() for _ in range(100)}
    assert len(ids) == 100


def test_synthesize_content_short_response() -> None:
    """Short responses are kept intact."""
    result = _synthesize_content("hello", "hi there")
    assert result == "hi there"


def test_synthesize_content_long_response_is_truncated() -> None:
    """Long responses are truncated to ~200 chars with ellipsis."""
    long_response = "x " * 150
    result = _synthesize_content("test", long_response)

    assert len(result) <= 210  # query + colon + space + ~200 chars
    assert result.endswith("...")
    assert result.startswith("test: ")