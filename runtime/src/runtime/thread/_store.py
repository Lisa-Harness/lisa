"""Thread store mechanics — paths, segment index, zstd compression, sealing.

The low-level persistence internals for the per-user segmented thread store.
No public surface here: callers use :mod:`runtime.thread` (the package), which
re-exports the read/write API that delegates into :meth:`ThreadStore`.

See ``docs/SPEC/THREAD_STORE.md`` for the on-disk format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import zstandard  # type-check-only; imported lazily at runtime

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────

#: Entries per active segment before a synchronous seal at append time.
_DEFAULT_SEGMENT_TURNS = 256
#: zstd compression level for sealed segments.
_ZSTD_LEVEL = 5
#: Retrain the shared dictionary every N sealed segments.
_DICT_RETRAIN_EVERY = 10
#: Minimum samples before a dictionary is trained at all.
_DICT_MIN_SAMPLES = 8


# ── Path derivation ────────────────────────────────────────────────────


class _SegmentPaths(NamedTuple):
    """All file paths derived from a thread store base path."""

    base: Path
    current: Path  # <base>.current.jsonl
    index_path: Path  # <base>.index.jsonl
    dictionary: Path  # <base>.dict.zstd
    legacy: Path  # <base>.jsonl (pre-segmentation, for the rename guard)


def _segment_paths(base: Path) -> _SegmentPaths:
    return _SegmentPaths(
        base=base,
        current=Path(f"{base}.current.jsonl"),
        index_path=Path(f"{base}.index.jsonl"),
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
        import zstandard  # lazy: defers optional dep startup cost

        return zstandard.ZstdCompressionDict(dict_path.read_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to load zstd dictionary %s: %s", dict_path, exc)
        return None


def _compress_bytes(data: bytes, dict_path: Path) -> bytes:
    """Compress ``data`` with zstd, using a trained dictionary when available."""
    import zstandard  # lazy: defers optional dep startup cost

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
    import zstandard  # lazy: defers optional dep startup cost

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
        import zstandard  # lazy: defers optional dep startup cost

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


def _read_active_entries_count(current_path: Path) -> int:
    """Count entries in the active segment without materializing them.

    Used at append time to derive the next global turn index cheaply —
    line count is the entry count because each entry is one JSONL line.
    """
    return _count_lines(current_path)


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
