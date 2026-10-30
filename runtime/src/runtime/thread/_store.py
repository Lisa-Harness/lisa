"""Thread store mechanics — paths, segment index, zstd compression, sealing.

The low-level persistence internals for the per-user segmented thread store.
No public surface here: callers use :mod:`runtime.thread` (the package), which
re-exports the read/write API that delegates into :meth:`ThreadStore`.

See ``docs/SPEC/THREAD_STORE.md`` for the on-disk format.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import zstandard  # type-check-only; imported lazily at runtime

from sdk.filelock import atomic_write_bytes

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
    lock: Path  # <base>.lock (cross-process advisory lock)


def _segment_paths(base: Path) -> _SegmentPaths:
    return _SegmentPaths(
        base=base,
        current=Path(f"{base}.current.jsonl"),
        index_path=Path(f"{base}.index.jsonl"),
        dictionary=Path(f"{base}.dict.zstd"),
        legacy=Path(f"{base}.jsonl"),
        lock=Path(f"{base}.lock"),
    )


def _sealed_segment_path(base: Path, seg_id: str) -> Path:
    return Path(f"{base}.{seg_id}.jsonl.zst")


# ── Hash chain ─────────────────────────────────────────────────────────

#: Hash the entry *line content* (UTF-8 bytes, no newline terminator) so the
#: value is stable across the newline conventions of any transport and
#: identical for the active JSONL and the sealed (compressed) copy.
def _entry_sha256(raw_line: str) -> str:
    return hashlib.sha256(raw_line.encode("utf-8")).hexdigest()


def _tail_line_sha(path: Path, *, block_bytes: int = 65536) -> str | None:
    """Return the SHA-256 of the last non-empty raw line of ``path``.

    Streams from the end in blocks, so the cost is O(tail bytes), not
    O(file). Returns ``None`` when the file is absent or has no
    non-empty line.
    """
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if size == 0:
            return None
        tail = b""
        while size > 0:
            step = min(block_bytes, size)
            size -= step
            handle.seek(size)
            tail = handle.read(step) + tail
            lines = tail.split(b"\n")
            # The writer terminates every entry with a newline, so the
            # sought line is the last non-empty element; the trailing empty
            # element (and any blank lines) are skipped.
            for candidate in reversed(lines):
                if candidate.strip():
                    return _entry_sha256(candidate.decode("utf-8", errors="replace"))
            # No newline in this window yet; the last line may span blocks,
            # so keep the whole accumulation (bounded by that line's size).
        # The whole file is one unterminated line.
        text = tail.decode("utf-8", errors="replace").strip()
        return _entry_sha256(text) if text else None


def _prev_entry_sha(paths: _SegmentPaths) -> str:
    """Return the chain link for the next appended entry, or ``""`` for the head.

    The previous entry's hash comes from the active segment's tail when one
    exists; after a seal (or a fresh store with sealed history), the index's
    newest ``tail_sha256`` serves the link without decompressing anything.
    ``""`` names the chain head — the first chained entry in this store —
    so the verifier can state where chaining begins rather than assuming
    history it cannot check.
    """
    tail = _tail_line_sha(paths.current)
    if tail is not None:
        return tail
    records = _read_index(paths.index_path)
    for record in reversed(records):
        if record.tail_sha256:
            return record.tail_sha256
        # A pre-chain sealed segment: the entries after it cannot link to a
        # verifiable predecessor, so chaining restarts at the next append.
        return ""
    return ""


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
    #: SHA-256 of the segment's terminal raw entry line (no newline). The
    #: hash chain continues across the seal boundary: the first entry
    #: appended after this seal links to this value. Empty on index lines
    #: written before the chain existed — the verifier treats those
    #: segments as unchained history, honestly, not as tampered.
    tail_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        record = {
            "seg_id": self.seg_id,
            "turn_lo": self.turn_lo,
            "turn_hi": self.turn_hi,
            "ts_lo": self.ts_lo,
            "ts_hi": self.ts_hi,
            "bytes_raw": self.bytes_raw,
            "bytes_zst": self.bytes_zst,
        }
        if self.tail_sha256:
            record["tail_sha256"] = self.tail_sha256
        return record

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
            tail_sha256=str(raw.get("tail_sha256") or ""),
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


def _train_dictionary(samples: Sequence[bytes | bytearray], dict_path: Path) -> None:
    """Train and persist a zstd dictionary from sample documents."""
    if len(samples) < _DICT_MIN_SAMPLES:
        return  # too few samples to train meaningfully
    try:
        import zstandard  # lazy: defers optional dep startup cost

        dict_data = zstandard.train_dictionary(_DEFAULT_SEGMENT_TURNS, list(samples))
        atomic_write_bytes(dict_path, dict_data.as_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to train zstd dictionary: %s", exc)


def _maybe_retrain_dictionary(paths: _SegmentPaths, sealed_segment_paths: list[Path]) -> None:
    """Train the dictionary once ``_DICT_RETRAIN_EVERY`` segments have been sealed.

    Once ``paths.dictionary`` exists on disk, it is kept immutable so earlier
    dictionary-compressed segments remain decodable with the same dictionary ID.
    """
    if paths.dictionary.exists():
        return
    if not sealed_segment_paths or len(sealed_segment_paths) % _DICT_RETRAIN_EVERY != 0:
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

    entries = _parse_entries(raw_bytes.decode("utf-8").splitlines())
    if not entries:
        return None

    compressed = _compress_bytes(raw_bytes, paths.dictionary)
    atomic_write_bytes(sealed_path, compressed)

    # The seal's terminal raw line is the chain link for the first entry
    # appended after this seal; hash it here (the raw bytes are already in
    # hand) so the next append needs no decompression.
    raw_lines = raw_bytes.decode("utf-8").splitlines()
    tail_sha = _entry_sha256(raw_lines[-1]) if raw_lines and raw_lines[-1].strip() else ""

    record = SegmentRecord(
        seg_id=seg_id,
        turn_lo=0,  # filled below from the first entry's global turn index
        turn_hi=0,
        ts_lo=str(entries[0].get("timestamp_utc", "")),
        ts_hi=str(entries[-1].get("timestamp_utc", "")),
        bytes_raw=len(raw_bytes),
        bytes_zst=len(compressed),
        tail_sha256=tail_sha,
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

    paths.current.unlink(missing_ok=True)
    return record


# ── Legacy migration ───────────────────────────────────────────────────


def _maybe_rename_legacy(paths: _SegmentPaths) -> None:
    """One-time rename of a legacy ``<base>.jsonl`` to the active segment.

    Hard cutover: after the rename, the legacy path is never read again.
    Not a compat shim — there is no dual-path read code.
    """
    if not paths.current.exists() and paths.legacy.exists():
        paths.legacy.rename(paths.current)
