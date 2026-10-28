"""Rotation-aware, bounded JSONL reading for forensic evidence.

The durable telemetry files this package reads share one property: they are
append-only JSONL that can rotate underneath a reader.

- ``logs/runtime.jsonl`` is written by ``logging.handlers.RotatingFileHandler``
  (5 MiB x 3 backups), so historical records live in ``runtime.jsonl.1`` ..
  ``runtime.jsonl.3``. A reader that opens only the active file silently loses
  whatever was rotated out — after a burst of verbose logging, that can be the
  entire window worth investigating.
- ``logs/decisions.jsonl`` is segmented independently by its handler: the
  active file is sealed to the next ``decisions.jsonl.<n>`` once it reaches
  the size threshold, and sealed segments are never deleted. Its numbering is
  therefore append-only (``.1`` oldest), the reverse of rotation backups. The
  full segment set is read when ``backup_count`` is ``None``.
- ``logs/token-usage.jsonl`` is appended by its own handler and does not
  segment today, but it is read through the same primitive so adding
  segmentation later cannot silently truncate a window.

Two bounds apply, and they are deliberately different:

- **Byte budget.** A line count is not a memory bound: one serialized record
  can carry a full model request or a traceback. The budget is spent from the
  newest segment backwards, so an oversized history degrades to "the most
  recent window" rather than to "the whole file in memory" — and the result
  says so (``JsonlTail.truncated``), so a partial read is never presented as
  a complete one.
- **Time window.** Files are read backwards in blocks and, given ``since``,
  the read stops at the first record older than the window. A 24h summary
  therefore costs the size of 24h of records, not of the whole history.
  Callers still filter on each record's own timestamp; this module supplies
  the parsing, so the semantics live in one place.

Malformed lines are skipped, never raised: a torn write is evidence too, and
one bad line must not hide the records around it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_BACKUP_SEGMENTS",
    "DEFAULT_MAX_BYTES",
    "JsonlTail",
    "OwnedRecords",
    "filter_owned",
    "filter_window",
    "in_window",
    "parse_ts",
    "read_jsonl_tail",
    "record_ts",
    "rotated_segments",
]

#: ``RotatingFileHandler`` keeps ``backupCount`` older segments as
#: ``<name>.1`` .. ``<name>.N``. ``sdk.logging_setup`` uses 3.
DEFAULT_BACKUP_SEGMENTS = 3

#: Default read ceiling. Generous on purpose: forensic and introspection reads
#: exist so the model can examine its own behaviour, and a budget that cuts a
#: window short hides exactly the evidence that matters. It spans the active
#: runtime log plus its full rotation set (5 MiB x 4) with headroom, and stays
#: a hard memory bound. Time-windowed reads usually stop far earlier.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024

#: Unit of a backwards read. Small enough that a windowed read past its
#: window start wastes little; large enough that syscalls stay negligible.
_READ_BLOCK_BYTES = 256 * 1024


def rotated_segments(path: Path, backup_count: int | None = DEFAULT_BACKUP_SEGMENTS) -> list[Path]:
    """Return the existing rotation set for ``path``, oldest file first.

    The two numbering schemes run in opposite directions:

    - ``backup_count`` an int: ``RotatingFileHandler`` backups, where ``.1``
      is the *newest* backup, so ``.N`` .. ``.1`` are returned.
    - ``backup_count`` ``None``: append-only sealed segments (as written by
      ``DecisionTelemetryHandler``), where each seal takes the next index, so
      ``.1`` is the *oldest*. Every numbered segment is discovered and
      returned ascending.

    In both cases the active file is last.
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
        numbered.sort(key=lambda item: item[0])
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


@dataclass(frozen=True, slots=True)
class JsonlTail:
    """The records read from a JSONL rotation set, and whether the read was cut short.

    ``truncated`` is True only when the byte budget stopped the read before
    the requested span was covered: before the oldest segment's start, or —
    when ``since`` was given — before reaching a record older than ``since``.
    A caller summarising a window must report a truncated read as a lower
    bound, never as the window's total.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


def _older_than(line: bytes | str, since: datetime) -> bool:
    """Return whether one raw JSONL line is a record dated before ``since``."""
    try:
        record = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    ts = record_ts(record)
    return ts is not None and ts < since


def _first_full_line(buffer: bytes) -> bytes:
    """Return the first complete line after the leading partial one."""
    start = buffer.find(b"\n")
    if start < 0:
        return b""
    end = buffer.find(b"\n", start + 1)
    return buffer[start + 1 : end] if end >= 0 else b""


def _read_segment_tail(
    segment: Path,
    budget: int,
    since: datetime | None,
) -> tuple[list[str], int, str]:
    """Read one segment backwards within ``budget`` bytes.

    Returns ``(lines, spent, stop)`` where ``stop`` is ``"complete"`` (the
    whole segment was read), ``"window"`` (a record older than ``since`` was
    reached, so older bytes are not needed), or ``"budget"``. A read that
    starts mid-file discards its leading partial line, so a half-written
    record is never parsed as evidence.
    """
    size = segment.stat().st_size
    position = size
    blocks: list[bytes] = []
    stop = "complete"
    with segment.open("rb") as handle:
        while position > 0:
            if size - position >= budget:
                stop = "budget"
                break
            step = min(_READ_BLOCK_BYTES, position, budget - (size - position))
            position -= step
            handle.seek(position)
            blocks.insert(0, handle.read(step))
            if since is not None and position > 0:
                probe = blocks[0] + (blocks[1] if len(blocks) > 1 else b"")
                if _older_than(_first_full_line(probe), since):
                    stop = "window"
                    break
    text = b"".join(blocks).decode("utf-8", errors="replace")
    if position > 0:
        newline = text.find("\n")
        text = text[newline + 1 :] if newline >= 0 else ""
    return text.splitlines(), size - position, stop


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


def read_jsonl_tail(
    path: Path | None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int | None = DEFAULT_BACKUP_SEGMENTS,
    since: datetime | None = None,
) -> JsonlTail:
    """Read a JSONL rotation set newest-first and return its records oldest-first.

    With ``since``, reading stops once a record older than ``since`` is
    reached, so a time-windowed caller pays only for its window however large
    the history grows; records older than ``since`` may still be returned and
    are the caller's to filter. ``max_bytes`` is a memory safety rail, and
    hitting it before the span is covered sets ``truncated``.

    Returns an empty result when the path is ``None`` or no segment exists.
    Parse failures and non-mapping payloads are skipped.
    """
    if path is None:
        return JsonlTail()
    segments = rotated_segments(path, backup_count)
    budget = max_bytes
    chunks: list[list[str]] = []
    truncated = False
    for segment in reversed(segments):
        if budget <= 0:
            truncated = True
            break
        try:
            lines, spent, stop = _read_segment_tail(segment, budget, since)
        except OSError:
            continue
        budget -= spent
        chunks.append(lines)
        if stop == "budget":
            truncated = True
            break
        if stop == "window" or (since is not None and lines and _older_than(lines[0], since)):
            break
    records = [record for lines in reversed(chunks) for record in _parse_lines(lines)]
    return JsonlTail(records=records, truncated=truncated)


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


@dataclass(frozen=True, slots=True)
class OwnedRecords:
    """Records attributed to one user, plus how many lacked any attribution."""

    records: list[dict[str, Any]]
    #: Records with no ``user_slug`` (written before attribution existed).
    unowned: int = 0


def filter_owned(records: list[dict[str, Any]], user_slug: str | None) -> OwnedRecords:
    """Keep only records whose ``user_slug`` is ``user_slug``.

    The telemetry logs are shared by every user of a data root, so a per-user
    read must not join another user's records onto its own turns (turn indices
    are per-thread and collide across users). Records without a ``user_slug``
    cannot be attributed and are excluded too; the count is returned so the
    caller can say evidence is missing rather than absent. A falsy
    ``user_slug`` means an unscoped (whole-root) read and keeps everything.
    """
    if not user_slug:
        return OwnedRecords(records=records)
    owned: list[dict[str, Any]] = []
    unowned = 0
    for record in records:
        owner = record.get("user_slug")
        if owner == user_slug:
            owned.append(record)
        elif not owner:
            unowned += 1
    return OwnedRecords(records=owned, unowned=unowned)
