"""Tests for sdk.memory.backup — point-in-time dumps and retention pruning."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sdk.memory.backup import DEFAULT_KEEP_BACKUPS, backup_memory_db


def _make_db(path: Path, rows: int = 2) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
        for idx in range(rows):
            conn.execute("INSERT INTO memories (content) VALUES (?)", (f"note-{idx}",))
        conn.commit()
    finally:
        conn.close()


def _read_dump(path: Path) -> str:
    assert path is not None
    return path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_backup_skips_when_db_missing(tmp_path: Path) -> None:
    result = backup_memory_db(data_dir=tmp_path)
    assert result is None
    assert not (tmp_path / "backups").exists()


@pytest.mark.asyncio
async def test_backup_writes_timestamped_dump(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    _make_db(db_path)

    result = backup_memory_db(data_dir=tmp_path)

    assert result is not None
    assert result.parent == tmp_path / "backups"
    assert result.name.startswith("memory-")
    assert result.suffix == ".sql"
    dump = _read_dump(result)
    assert "CREATE TABLE memories" in dump
    assert "note-0" in dump
    assert "note-1" in dump


@pytest.mark.asyncio
async def test_backup_prunes_to_keep(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    _make_db(db_path)
    backup_dir = tmp_path / "backups"

    keep = 3
    produced: list[Path] = []
    for _ in range(keep + 2):
        produced.append(backup_memory_db(data_dir=tmp_path, keep=keep))  # type: ignore[arg-type]

    remaining = sorted(backup_dir.glob("memory-*.sql"))
    assert len(remaining) == keep
    # The most recent dump always survives pruning.
    assert produced[-1].name in {path.name for path in remaining}
    # Every surviving file is one of the produced dumps.
    produced_names = {path.name for path in produced}
    assert {path.name for path in remaining} <= produced_names


@pytest.mark.asyncio
async def test_backup_default_keep_is_ten(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    _make_db(db_path)

    for _ in range(DEFAULT_KEEP_BACKUPS + 3):
        backup_memory_db(data_dir=tmp_path)

    remaining = sorted((tmp_path / "backups").glob("memory-*.sql"))
    assert len(remaining) == DEFAULT_KEEP_BACKUPS


@pytest.mark.asyncio
async def test_backup_respects_custom_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db_path = data_dir / "custom.db"
    backup_dir = tmp_path / "elsewhere"
    data_dir.mkdir(parents=True)
    _make_db(db_path)

    result = backup_memory_db(data_dir=data_dir, db_path=db_path, backup_dir=backup_dir, keep=5)

    assert result is not None
    assert result.parent == backup_dir
