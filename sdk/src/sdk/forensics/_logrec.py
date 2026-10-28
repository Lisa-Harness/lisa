"""Rotation-aware, bounded JSONL reading for forensic evidence.

The durable telemetry files this package reads share one property: they are
append-only JSONL that can rotate underneath a reader.

- ``logs/runtime.jsonl`` is written by ``logging.handlers.RotatingFileHandler``
  (5 MiB x 3 backups), so historical records live in ``runtime.jsonl.1`` ..
  ``runtime.jsonl.3``. A reader that opens only the active file silently loses
  whatever was rotated out — after a burst of verbose logging, that can be the
  entire window worth investigating.
- ``logs/decisions.jsonl`` is segmented independently by its handler: the
  active file is sealed to ``decisions.jsonl.<n>`` once it reaches the size
  threshold, and sealed segments are never deleted. The full segment set is
  read when ``backup_count`` is ``None``.
- ``logs/token-usage.jsonl`` is appended by its own handler and does not
  segment today, but it is read through the same primitive so adding
  segmentation later cannot silently truncate a window.

Two bounds apply, and they are deliberately different:

- **Byte budget.** A line count is not a memory bound: one serialized record
  can carry a full model request or a traceback. The budget is spent from the
  newest segment backwards, so an oversized history degrades to "the most
  recent window" rather than to "the whole file in memory".
- **Time window.** Callers filter on the record's own timestamp; this module
  supplies only the parsing, so the semantics live in one place.

Malformed lines are skipped, never raised: a torn write is evidence too, and
one bad line must not hide the records around it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_BACKUP_SEGMENTS",
    "DEFAULT_MAX_BYTES",
    "filter_window",
    "in_window",
    "iter_jsonl_tail",
    "parse_ts",
    "record_ts",
    "rotated_segments",
]

#: ``RotatingFileHandler`` keeps ``backupCount`` older segments as
#: ``<name>.1`` .. ``<name>.N``. ``sdk.logging_setup`` uses 3.
DEFAULT_BACKUP_SEGMENTS = 3

#: Default read ceiling: spans the active runtime log plus its full rotation
#: set with margin, while remaining a hard memory bound.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024


def rotated_segments(path: Path, backup_count: int | None = DEFAULT_BACKUP_SEGMENTS) -> list[Path]:
    """Return the existing rotation set for ``path``, oldest file first.

    When ``backup_count`` is ``None``, every sealed segment (``<name>.<n>``)
    is discovered, sorted descending by ``n``, and the active file is appended
    last — preserving the oldest-first contract. When it is an int, only the
    newest ``backup_count`` segments are returned (byte-identical to the
    previous behaviour).
    """
    if backup_count is None:
        parent = path.parent
        name = path.name
        prefix = name + "."
        numbered: list[tuple[int, Path]] = []
        if parent.is_dir():
            for candidate in parent.iterdir():
                if candidate.name.startswith(prefix):
                    suffix = candidate.name[len(prefix) :]
                    if suffix.isdigit():
                        numbered.append((int(suffix), candidate))
        numbered.sort(key=lambda item: item[0], reverse=True)
        discovered = [p for _, p in numbered]
        if path.is_file():
            discovered.append(path)
        return discovered
    segments: list[Path] = []
    for index in range(backup_count, 0, -1):
        candidate = path.with_name(f"{path.name}.{index}")
        if candidate.is_file():
            segments.append(candidate)
    if path.is_file():
        segments.append(path)
    return segments


def _read_segment_tail(segment: Path, budget: int) -> tuple[str, int]:
    """Read one segment within ``budget`` bytes; return ``(text, spent)``.

    A segment larger than the remaining budget is read from its tail and its
    leading partial line is discarded, so a half-written record is never
    parsed as if it were evidence.
    """
    size = segment.stat().st_size
    if size <= budget:
        return segment.read_text(encoding="utf-8", errors="replace"), size
    with segment.open("rb") as handle:
        handle.seek(size - budget)
        text = handle.read().decode("utf-8", errors="replace")
    newline = text.find("\n")
    return (text[newline + 1 :] if newline >= 0 else ""), budget


def _parse_lines(lines: list[str]) -> Iterator[dict[str, Any]]:
    """Yield the mapping records from raw JSONL lines, skipping anything else."""
    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(record, dict):
            yield record


def iter_jsonl_tail(
    path: Path | None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int | None = DEFAULT_BACKUP_SEGMENTS,
) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a JSONL file's rotation set, oldest first.

    Yields nothing when the path is ``None`` or no segment exists. Parse
    failures and non-mapping payloads are skipped.
    """
    if path is None:
        return
    segments = rotated_segments(path, backup_count)
    if not segments:
        return

    budget = max_bytes
    chunks: list[list[str]] = []
    for segment in reversed(segments):
        if budget <= 0:
            break
        try:
            text, spent = _read_segment_tail(segment, budget)
        except OSError:
            continue
        budget -= spent
        chunks.append(text.splitlines())

    for lines in reversed(chunks):
        yield from _parse_lines(lines)


def parse_ts(raw: Any) -> datetime | None:
    """Parse a telemetry timestamp into an aware UTC datetime, or ``None``.

    Accepts the shapes the writers emit: ISO-8601 (``logging_setup``'s ``ts``,
    ``TokenUsageEvent.emitted_at``) and a numeric epoch. Anything else is
    unknown rather than "now", so a malformed record cannot manufacture
    recency.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            return datetime.fromtimestamp(float(raw), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def record_ts(record: dict[str, Any]) -> datetime | None:
    """Return a record's timestamp from whichever field its writer uses."""
    return parse_ts(
        record.get("ts")
        or record.get("emitted_at")
        or record.get("timestamp_utc")
        or record.get("logged_at")
    )


def in_window(ts: datetime | None, cutoff: datetime) -> bool:
    """Return whether ``ts`` falls inside the window ending at ``cutoff``.

    An unparseable timestamp is admitted: the record was recent enough to be
    in the segment just read, and dropping unknown-time evidence would hide
    exactly the malformed writes worth investigating.
    """
    return ts is None or ts >= cutoff


def filter_window(
    records: Iterable[dict[str, Any]],
    *,
    now: datetime,
    window_seconds: float,
) -> list[dict[str, Any]]:
    """Return the records whose own timestamp falls within the trailing window."""
    cutoff = now - timedelta(seconds=window_seconds)
    return [record for record in records if in_window(record_ts(record), cutoff)]
