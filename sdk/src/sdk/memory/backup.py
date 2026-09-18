"""Point-in-time SQLite dump backups for the memory store.

On memory-server boot, dump ``memory.db`` (via the SQLite backup API +
``.dump``-equivalent serialization) to a timestamped SQL file under
``<data_dir>/backups/`` and prune older dumps so only the newest ``keep``
snapshots remain.

The dump uses :meth:`sqlite3.Connection.iterdump` — the same text format
as the ``sqlite3 memory.db .dump`` CLI shell — so backups are restorable
with ``sqlite3 memory.db < dump.sql``. A consistent source snapshot is
taken first via the SQLite backup API so the dump is point-in-time even
while the live store is open.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

try:  # optional: required only when the DB contains a vec0 virtual table.
    import sqlite_vec as _sqlite_vec
except Exception:  # pragma: no cover - import-time guard  # pylint: disable=broad-except
    _sqlite_vec = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_KEEP_BACKUPS",
    "backup_memory_db",
    "default_backup_dir",
]

logger = logging.getLogger(__name__)

# Default retention: keep the last 10 point-in-time dumps.
DEFAULT_KEEP_BACKUPS = 10

_BACKUP_PREFIX = "memory"
_BACKUP_SUFFIX = ".sql"
_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
_BACKUP_NAME_RE = re.compile(
    rf"^{_BACKUP_PREFIX}-\d{{8}}T\d{{6}}(-\d+)?{re.escape(_BACKUP_SUFFIX)}$"
)


def default_backup_dir(data_dir: Path) -> Path:
    """Return the default backup directory for a memory data dir."""
    from sdk.layout import DATA_DIR  # noqa: PLC0415

    return Path(data_dir) / DATA_DIR / "backups"


def _resolved_db_path(data_dir: Path, db_path: Path | None) -> Path:
    if db_path is not None:
        return Path(db_path)
    from sdk.layout import DATA_DIR  # noqa: PLC0415

    return Path(data_dir) / DATA_DIR / "memory.db"


def _timestamp() -> str:
    return datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)


def _unique_backup_path(backup_dir: Path, timestamp: str) -> Path:
    """Return a collision-free backup path for the given timestamp."""
    candidate = backup_dir / f"{_BACKUP_PREFIX}-{timestamp}{_BACKUP_SUFFIX}"
    sequence = 1
    while candidate.exists():
        candidate = backup_dir / f"{_BACKUP_PREFIX}-{timestamp}-{sequence}{_BACKUP_SUFFIX}"
        sequence += 1
    return candidate


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load the ``sqlite-vec`` extension into ``conn``; return True if loaded.

    Mirrors :meth:`SqliteMemoryStore._load_vec_extension` so the snapshot
    connection can read ``vec0`` virtual tables. Fails soft: when the wheel
    is missing or the SQLite build blocks extension loading, returns False
    and the dump proceeds extension-less (a DB with a vec0 table will then
    raise from ``iterdump``).
    """
    if _sqlite_vec is None:
        return False
    try:
        conn.enable_load_extension(True)
        conn.load_extension(_sqlite_vec.loadable_path())
        return True
    except Exception:  # pragma: no cover - depends on build  # pylint: disable=broad-except
        return False


def _dump_snapshot(db_path: Path, target: Path) -> None:
    """Write a consistent ``.dump`` SQL snapshot of ``db_path`` to ``target``.

    The live database is first copied into an in-memory snapshot via the
    SQLite backup API so the dump reflects one point in time even if the
    source store is mid-write. The snapshot is then serialized with
    ``iterdump`` — byte-compatible with the ``sqlite3 .dump`` shell.
    The sqlite-vec extension is loaded into both connections when
    available so databases containing ``vec0`` virtual tables (created by
    ``SqliteMemoryStore``) dump cleanly.
    """
    source = sqlite3.connect(str(db_path))
    try:
        _load_vec_extension(source)
        snapshot = sqlite3.connect(":memory:")
        try:
            # The snapshot needs vec0 too: iterdump reads the virtual table's
            # schema through the snapshot connection.
            _load_vec_extension(snapshot)
            source.backup(snapshot)
            dump_text = "\n".join(snapshot.iterdump())
        finally:
            snapshot.close()
    finally:
        source.close()
    target.write_text(dump_text + "\n", encoding="utf-8")


def _list_backups(backup_dir: Path) -> list[Path]:
    """Return existing backup dumps, oldest first.

    Ordered by modification time (name as a stable tiebreak) rather than
    filename, so same-second dumps with collision suffixes still prune
    oldest-first — lexicographic name order breaks once a suffix reaches
    two digits (``-19`` sorts before ``-2``).
    """
    if not backup_dir.is_dir():
        return []
    backups = [path for path in backup_dir.iterdir() if _BACKUP_NAME_RE.match(path.name)]
    return sorted(backups, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _prune_backups(backup_dir: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` backups; return removed paths."""
    backups = _list_backups(backup_dir)
    excess = len(backups) - keep
    if excess <= 0:
        return []
    removed: list[Path] = []
    for path in backups[:excess]:
        path.unlink()
        removed.append(path)
    return removed


def backup_memory_db(
    *,
    data_dir: Path,
    db_path: Path | None = None,
    backup_dir: Path | None = None,
    keep: int = DEFAULT_KEEP_BACKUPS,
) -> Path | None:
    """Dump ``memory.db`` to a timestamped SQL backup and prune old dumps.

    Writes one ``memory-<UTC timestamp>.sql`` file under ``backup_dir``
    (default ``<data_dir>/backups/``) and deletes older dumps beyond the
    ``keep`` limit. Returns the new backup path, or ``None`` when the
    source database does not exist yet (first boot — nothing to back up).

    ``keep`` is clamped to at least 1 so the fresh dump always survives.
    """
    source_path = _resolved_db_path(data_dir, db_path)
    if not source_path.exists():
        logger.info("memory backup skipped; no database at %s", source_path)
        return None
    keep = max(1, int(keep))
    target_dir = Path(backup_dir) if backup_dir is not None else default_backup_dir(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = _unique_backup_path(target_dir, _timestamp())
    _dump_snapshot(source_path, target_path)
    removed = _prune_backups(target_dir, keep)
    if removed:
        logger.info(
            "memory backup wrote %s; pruned %d old backup(s) (keep=%d)",
            target_path,
            len(removed),
            keep,
        )
    else:
        logger.info("memory backup wrote %s (keep=%d)", target_path, keep)
    return target_path
