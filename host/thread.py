"""Transport-neutral thread persistence (segmented + zstd-sealed).

This module owns the pure, stateless logic for the per-user thread store.
The store is a three-tier lossless layout keyed by a *base path*
(``thread_dir / <user_slug>``):

- **Active segment** — ``<base>.current.jsonl``: plain JSONL, append-only.
  Bounded by ``segment_turns`` entries. This is the only file the
  recent-N read ever touches, so the read is O(``limit``) in practice
  rather than O(history).
- **Sealed segments** — ``<base>.<seg_id>.jsonl.zst``: zstd-compressed
  JSONL, one per sealed chunk. ``seg_id`` is a zero-padded 4-digit
  counter (``0001``, ``0002``, ...). Sealing happens at append time when
  the active segment crosses ``segment_turns`` — no background scheduler.
- **Sparse index** — ``<base>.index.jsonl``: one JSON line per sealed
  segment recording ``seg_id`` and the turn/timestamp range it covers.
  O(segments) = O(turns / segment_turns), i.e. tiny forever.

Sidecars:

- ``<base>.digest.md`` — lossy digest (per-user). The
  ``FileBackedStateProvider`` slice summarizes sealed segments into it.
- ``<base>.digest.marker`` — the last digested ``seg_id``.
- ``<base>.dict.zstd`` — optional trained zstd dictionary for better
  compression on the homogeneous JSON-key vocabulary.

The one-time rename guard in :func:`append_thread_entry` migrates a
legacy ``<base>.jsonl`` to ``<base>.current.jsonl`` on first write. The
old path is never read again after the rename (hard cutover, no shims).

See ``docs/SPEC/THREAD_STORE.md`` for the full on-disk format spec.

Dependency direction
--------------------
``thread.py`` ← ``main.py`` / ``handle_message.py``  (callers import from
thread; thread never imports main or any transport code)

``thread.py`` → ``host.host_commands`` (leaf: command-prefix registry),
``host.lisa_config`` (legacy digest constant). Neither imports thread back.

Public surface
--------------
- :func:`load_recent_thread_entries`
- :func:`load_thread_range`
- :func:`append_thread_entry`
- :func:`write_turn_history`
- :func:`load_thread_digest`
- :func:`build_recent_thread_excerpt`
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from host.host_commands import is_host_command_query as _is_host_command_query

if TYPE_CHECKING:
    import zstandard  # type-check-only; imported lazily at runtime

logger = logging.getLogger(__name__)

# ``_is_host_command_query`` is re-exported here for backward compatibility
# only — callers should import directly from ``host.host_commands``.
# The persistence layer (this module) must not own command-name vocabulary;
# the registry lives in ``host_commands`` (a leaf module with no
# ``host`` imports) so both the command dispatch layer and the
# persistence layer can share it without a cycle.

__all__ = [
    "append_thread_entry",
    "build_recent_thread_excerpt",
    "load_recent_thread_entries",
    "load_thread_digest",
    "load_thread_range",
    "write_turn_history",
]

# ── Defaults ───────────────────────────────────────────────────────────

_DEFAULT_SEGMENT_TURNS = 256
_ZSTD_LEVEL = 5
_DICT_RETRAIN_EVERY = 10
_DICT_MIN_SAMPLES = 8


# ── Path derivation ────────────────────────────────────────────────────


class _SegmentPaths(NamedTuple):
    """All file paths derived from a thread store base path."""

    base: Path
    current: Path  # <base>.current.jsonl
    index_path: Path  # <base>.index.jsonl
    digest: Path  # <base>.digest.md
    marker: Path  # <base>.digest.marker
    dictionary: Path  # <base>.dict.zstd
    legacy: Path  # <base>.jsonl (pre-segmentation, for the rename guard)


def _segment_paths(base: Path) -> _SegmentPaths:
    return _SegmentPaths(
        base=base,
        current=Path(f"{base}.current.jsonl"),
        index_path=Path(f"{base}.index.jsonl"),
        digest=Path(f"{base}.digest.md"),
        marker=Path(f"{base}.digest.marker"),
        dictionary=Path(f"{base}.dict.zstd"),
        legacy=Path(f"{base}.jsonl"),
    )


def _sealed_segment_path(base: Path, seg_id: str) -> Path:
    return Path(f"{base}.{seg_id}.jsonl.zst")


# ── Index records ──────────────────────────────────────────────────────


class SegmentRecord(NamedTuple):
    """One line of ``<base>.index.jsonl`` describing a sealed segment."""

    seg_id: str
    turn_lo: int
    turn_hi: int
    ts_lo: str
    ts_hi: str
    bytes_raw: int
    bytes_zst: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seg_id": self.seg_id,
            "turn_lo": self.turn_lo,
            "turn_hi": self.turn_hi,
            "ts_lo": self.ts_lo,
            "ts_hi": self.ts_hi,
            "bytes_raw": self.bytes_raw,
            "bytes_zst": self.bytes_zst,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SegmentRecord:
        return cls(
            seg_id=str(raw["seg_id"]),
            turn_lo=int(raw["turn_lo"]),
            turn_hi=int(raw["turn_hi"]),
            ts_lo=str(raw["ts_lo"]),
            ts_hi=str(raw["ts_hi"]),
            bytes_raw=int(raw["bytes_raw"]),
            bytes_zst=int(raw["bytes_zst"]),
        )


def _read_index(index_path: Path) -> list[SegmentRecord]:
    """Parse the sparse segment index. Returns ``[]`` when absent/unreadable."""
    if not index_path.exists():
        return []
    records: list[SegmentRecord] = []
    for raw_line in index_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            try:
                records.append(SegmentRecord.from_dict(decoded))
            except (KeyError, TypeError, ValueError):
                continue
    return records


def _append_index(index_path: Path, record: SegmentRecord) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _next_seg_id(index_path: Path) -> str:
    """Return the next zero-padded 4-digit segment id (starts at ``0001``)."""
    records = _read_index(index_path)
    if not records:
        return "0001"
    last_id = records[-1].seg_id
    try:
        n = int(last_id) + 1
    except ValueError:
        n = len(records) + 1
    return f"{n:04d}"


# ── zstd compression ──────────────────────────────────────────────────


def _load_zstd_dictionary(
    dict_path: Path,
) -> zstandard.ZstdCompressionDict | None:
    """Load a trained zstd dictionary, or ``None`` when absent."""
    if not dict_path.exists():
        return None
    try:
        import zstandard  # noqa: PLC0415  # lazy: defers optional dep startup cost

        return zstandard.ZstdCompressionDict(dict_path.read_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to load zstd dictionary %s: %s", dict_path, exc)
        return None


def _compress_bytes(data: bytes, dict_path: Path) -> bytes:
    """Compress ``data`` with zstd, using a trained dictionary when available."""
    import zstandard  # noqa: PLC0415  # lazy: defers optional dep startup cost

    dictionary = _load_zstd_dictionary(dict_path)
    params = zstandard.ZstdCompressionParameters.from_level(_ZSTD_LEVEL)
    if dictionary is not None:
        cctx = zstandard.ZstdCompressor(dict_data=dictionary, compression_params=params)
    else:
        cctx = zstandard.ZstdCompressor(compression_params=params)
    # TODO(seekable-frames): revisit when segment_turns > ~1000 or decompress
    # latency exceeds ~50ms; switch to zstd seekable frame format (frame_dlog)
    # for byte-offset range reads without full decompress.
    return bytes(cctx.compress(data))


def _decompress_bytes(data: bytes, dict_path: Path) -> bytes:
    """Decompress a sealed segment, using the trained dictionary when present."""
    import zstandard  # noqa: PLC0415  # lazy: defers optional dep startup cost

    dictionary = _load_zstd_dictionary(dict_path)
    if dictionary is not None:
        dctx = zstandard.ZstdDecompressor(dict_data=dictionary)
    else:
        dctx = zstandard.ZstdDecompressor()
    # TODO(seekable-frames): revisit when segment_turns > ~1000 or decompress
    # latency exceeds ~50ms; switch to zstd seekable frame format (frame_dlog)
    # for byte-offset range reads without full decompress.
    return bytes(dctx.decompress(data))


def _train_dictionary(samples: Sequence[bytes | bytearray], dict_path: Path) -> None:
    """Train and persist a zstd dictionary from sample documents."""
    if len(samples) < _DICT_MIN_SAMPLES:
        return  # too few samples to train meaningfully
    try:
        import zstandard  # noqa: PLC0415  # lazy: defers optional dep startup cost

        dict_data = zstandard.train_dictionary(_DEFAULT_SEGMENT_TURNS, list(samples))
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dict_path.write_bytes(dict_data.as_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to train zstd dictionary: %s", exc)


def _maybe_retrain_dictionary(paths: _SegmentPaths, sealed_segment_paths: list[Path]) -> None:
    """Retrain the dictionary every ``_DICT_RETRAIN_EVERY`` seals."""
    if len(sealed_segment_paths) % _DICT_RETRAIN_EVERY != 0:
        return
    samples: list[bytes] = []
    for seg_path in sealed_segment_paths[-_DICT_RETRAIN_EVERY:]:
        try:
            samples.append(seg_path.read_bytes())
        except OSError:
            continue
    if samples:
        _train_dictionary(samples, paths.dictionary)


# ── Sealing ────────────────────────────────────────────────────────────


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file (best-effort)."""
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _parse_entries(raw_lines: list[str]) -> list[dict[str, Any]]:
    """Parse JSONL lines into dicts, skipping malformed/non-dict lines."""
    entries: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            entries.append(decoded)
    return entries


def _read_active_entries(current_path: Path) -> list[dict[str, Any]]:
    if not current_path.exists():
        return []
    return _parse_entries(current_path.read_text(encoding="utf-8").splitlines())


def _seal_segment(paths: _SegmentPaths, seg_id: str | None = None) -> SegmentRecord | None:
    """Seal the active segment to a compressed sidecar.

    Reads ``current.jsonl``, compresses it with zstd (optional trained
    dictionary), writes ``<base>.<seg_id>.jsonl.zst``, appends an index
    line, then deletes the active segment. Returns the index record, or
    ``None`` when the active segment is empty/missing.
    """
    if not paths.current.exists():
        return None
    raw_bytes = paths.current.read_bytes()
    if not raw_bytes.strip():
        return None

    if seg_id is None:
        seg_id = _next_seg_id(paths.index_path)
    sealed_path = _sealed_segment_path(paths.base, seg_id)

    entries = _parse_entries(paths.current.read_text(encoding="utf-8").splitlines())
    if not entries:
        return None

    compressed = _compress_bytes(raw_bytes, paths.dictionary)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_bytes(compressed)

    record = SegmentRecord(
        seg_id=seg_id,
        turn_lo=0,  # filled below from the first entry's global turn index
        turn_hi=0,
        ts_lo=str(entries[0].get("timestamp_utc", "")),
        ts_hi=str(entries[-1].get("timestamp_utc", "")),
        bytes_raw=len(raw_bytes),
        bytes_zst=len(compressed),
    )
    # Compute absolute turn range from the index (sum of prior sealed
    # segment turn counts + this segment's own count).
    prior_records = _read_index(paths.index_path)
    turn_lo = sum(r.turn_hi - r.turn_lo + 1 for r in prior_records)
    turn_hi = turn_lo + len(entries) - 1
    record = record._replace(turn_lo=turn_lo, turn_hi=turn_hi)

    _append_index(paths.index_path, record)

    # Opportunistic dictionary retraining.
    sealed = sorted(paths.base.parent.glob(f"{paths.base.name}.*.jsonl.zst"))
    _maybe_retrain_dictionary(paths, sealed)

    paths.current.unlink()
    return record


# ── Legacy migration ───────────────────────────────────────────────────


def _maybe_rename_legacy(paths: _SegmentPaths) -> None:
    """One-time rename of a legacy ``<base>.jsonl`` to the active segment.

    Hard cutover: after the rename, the legacy path is never read again.
    Not a compat shim — there is no dual-path read code.
    """
    if not paths.current.exists() and paths.legacy.exists():
        paths.legacy.rename(paths.current)


# ── Public read API ────────────────────────────────────────────────────


def load_recent_thread_entries(
    *,
    thread_path: Path | None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Load up to ``limit`` recent thread entries (oldest-first in the tail).

    Reads only the active segment. If it has fewer than ``limit`` entries,
    falls back to the most recent sealed segment for the remaining tail.
    Two file touches max. Returns ``[]`` when ``thread_path`` is ``None``
    or no store exists.
    """
    if thread_path is None:
        return []
    paths = _segment_paths(thread_path)
    active = _read_active_entries(paths.current)
    if len(active) >= limit > 0:
        return active[-limit:]
    # Spill into the most recent sealed segment when active is short.
    if limit <= 0 or len(active) >= limit:
        return active
    records = _read_index(paths.index_path)
    if not records:
        return active
    last = records[-1]
    sealed_path = _sealed_segment_path(paths.base, last.seg_id)
    sealed_entries = _load_sealed_entries(sealed_path, paths.dictionary, last.seg_id)
    if not sealed_entries:
        return active
    needed = limit - len(active)
    tail = sealed_entries[-needed:] if needed < len(sealed_entries) else sealed_entries
    return tail + active


def _load_sealed_entries(
    sealed_path: Path,
    dict_path: Path,
    seg_id: str,
) -> list[dict[str, Any]]:
    """Decompress and parse a sealed segment. Returns [] on any failure."""
    if not sealed_path.exists():
        return []
    try:
        decompressed = _decompress_bytes(sealed_path.read_bytes(), dict_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to decompress sealed segment %s: %s", seg_id, exc)
        return []
    return _parse_entries(decompressed.decode("utf-8").splitlines())


def load_thread_range(
    *,
    thread_path: Path | None,
    start_turn: int,
    end_turn: int,
) -> list[dict[str, Any]]:
    """Load entries whose global turn index falls in ``[start_turn, end_turn]``.

    Global turn index is the 0-based ordinal of a turn across the whole
    store (sealed segments + active). The index is consulted to find
    overlapping segments; only those are decompressed. The active segment
    is read directly for its in-range tail.
    """
    if thread_path is None or start_turn > end_turn:
        return []
    paths = _segment_paths(thread_path)
    records = _read_index(paths.index_path)
    results: list[dict[str, Any]] = []
    for record in records:
        if record.turn_hi < start_turn or record.turn_lo > end_turn:
            continue
        sealed_path = _sealed_segment_path(paths.base, record.seg_id)
        if not sealed_path.exists():
            continue
        try:
            decompressed = _decompress_bytes(sealed_path.read_bytes(), paths.dictionary)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("thread: range decompress failed for %s: %s", record.seg_id, exc)
            continue
        entries = _parse_entries(decompressed.decode("utf-8").splitlines())
        for offset, entry in enumerate(entries):
            global_turn = record.turn_lo + offset
            if start_turn <= global_turn <= end_turn:
                results.append(entry)
    # Active (unsealed) tail: its turn range starts after the last sealed.
    active_turn_lo = sum(r.turn_hi - r.turn_lo + 1 for r in records)
    for offset, entry in enumerate(_read_active_entries(paths.current)):
        global_turn = active_turn_lo + offset
        if start_turn <= global_turn <= end_turn:
            results.append(entry)
    return results


def load_thread_digest(*, thread_path: Path | None) -> str:
    """Load the per-user thread digest sidecar (``<base>.digest.md``).

    Returns an empty string when ``thread_path`` is ``None`` or the digest
    file is absent or unreadable.
    """
    if thread_path is None:
        return ""
    paths = _segment_paths(thread_path)
    if not paths.digest.is_file():
        return ""
    try:
        return paths.digest.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


# ── Public write API ───────────────────────────────────────────────────


def append_thread_entry(
    *,
    thread_path: Path | None,
    user_slug: str,
    mode_id: str,
    query: str,
    response_text: str,
    provenance: dict[str, Any] | None = None,
    segment_turns: int = _DEFAULT_SEGMENT_TURNS,
) -> None:
    """Append a new thread entry to the active segment.

    Seals the active segment first when it has reached ``segment_turns``
    entries (pressure-driven, synchronous, sub-10ms for 256 lines). No-ops
    when ``thread_path`` is ``None`` or ``user_slug`` is empty.
    """
    if thread_path is None or not user_slug:
        return
    paths = _segment_paths(thread_path)
    _maybe_rename_legacy(paths)

    thread_path.parent.mkdir(parents=True, exist_ok=True)
    if paths.current.exists() and _count_lines(paths.current) >= segment_turns:
        _seal_segment(paths)

    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "user_slug": user_slug,
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
    }
    if provenance:
        entry["provenance"] = provenance

    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_turn_history(
    *,
    thread_path: Path | None,
    new_history: list[dict[str, Any]],
) -> None:
    """Destructive compaction write-back (used by compaction /summarize-history).

    Deletes all sealed segments, the index, the dictionary, and the digest
    marker, then writes ``new_history`` as the fresh active segment and
    clears the digest sidecar. Old turns are gone — this is the existing
    compaction semantic (compacted history replaces everything).
    """
    if thread_path is None:
        raise RuntimeError("write_turn_history: no thread path configured")
    paths = _segment_paths(thread_path)
    # Remove all sealed segments + index + dictionary + marker.
    for sealed in paths.base.parent.glob(f"{paths.base.name}.*.jsonl.zst"):
        try:
            sealed.unlink()
        except OSError as exc:
            logger.warning("write_turn_history: failed to remove %s: %s", sealed, exc)
    for sidecar in (paths.index_path, paths.dictionary, paths.marker):
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError as exc:
                logger.warning("write_turn_history: failed to remove %s: %s", sidecar, exc)
    # Clear the digest.
    if paths.digest.exists():
        try:
            paths.digest.write_text("", encoding="utf-8")
        except OSError as exc:
            logger.warning("write_turn_history: failed to clear digest: %s", exc)
    # Write the new active segment.
    paths.current.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(entry, ensure_ascii=False) for entry in new_history]
    paths.current.write_text(
        ("\n".join(lines) + "\n") if lines else "",
        encoding="utf-8",
    )


# ── Excerpt (prompt injection) ─────────────────────────────────────────


def build_recent_thread_excerpt(
    *,
    thread_path: Path | None,
    limit: int = 3,
) -> str:
    """Build a short human-readable excerpt of recent thread entries.

    Returns an empty string when there are no entries. Each line shows the
    query and a preview of the response (or the query itself if the
    response is empty).

    Host-only command entries (``/pack disable github``, ``/help``,
    ``/tokens``, ...) are stripped: they are operator UI gestures, not
    dialogue, and replaying them as "recent conversation" makes the model
    treat a UI gesture as something the user said. The model already
    observes their effect via the live capability manifest / pack lock.

    The loader reads up to ``limit * 4`` raw entries (a bounded over-read)
    so that after stripping host commands the excerpt can still yield up
    to ``limit`` genuine dialogue turns in normal sessions.
    """
    # Bounded over-read so we can filter host commands and still fill the
    # excerpt with real dialogue. ``* 4`` is a heuristic: most sessions are
    # dominated by dialogue, with occasional ``/pack`` or ``/help`` runs.
    # If a user fires many host commands in a row the excerpt may run
    # short, which is the correct outcome (there is no fresh dialogue to
    # surface). Capped to avoid loading a pathological thread in full.
    raw_limit = max(limit, min(limit * 4, 200))
    entries = load_recent_thread_entries(thread_path=thread_path, limit=raw_limit)
    if not entries:
        return ""
    lines = ["Recent thread context:"]
    emitted = 0
    for entry in reversed(entries):
        if emitted >= limit:
            break
        query = str(entry.get("query") or "").strip()
        response = str(entry.get("response_text") or "").strip()
        if not query and not response:
            continue
        # Strip host-only commands (``/pack``, ``/help``, ``/tokens`` ...)
        # from the conversation transcript fed back to the model.
        if _is_host_command_query(query):
            continue
        body = response or query
        prov = entry.get("provenance")
        if prov:
            body = (
                f"{body} [system_provenance: {prov.get('source', 'observed')}, "
                f"model_id: {prov.get('model_id', 'unknown')}]"
            )
        lines.append(f"- {query or 'unknown'}: {body}")
        emitted += 1
    if emitted == 0:
        return ""
    # Restore chronological order (oldest first) for readability.
    return "\n".join([lines[0], *reversed(lines[1:])])