"""Lossless segmented thread-store read backend.

Extracted from ``runtime.thread``: the segment-path derivation, sparse
index parsing, zstd decompression, and the public read operations
(``load_recent``, ``load_range``, ``build_recent_excerpt``,
``build_manifest``). The write path stays in ``runtime.thread`` because it
owns the per-turn response and effect-receipt persistence and the
host-command query check used by ``build_recent_excerpt``.

The MCP server (``mcp_server.thread.server``) wraps this; the host
imports it directly for the context-assembly excerpt/manifest builders.
Both use the same SDK contract — the thread is the single source of
truth for workflow state.
"""

from __future__ import annotations

import contextlib
import heapq
import io
import json
import logging
import sys
from collections import deque
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from sdk.filelock import file_lock
from sdk.thread.redaction import ThreadRedactionPolicy

if TYPE_CHECKING:
    from sdk.thread._verify import ChainVerification

logger = logging.getLogger(__name__)

#: Query preview width in the thread manifest table rows.
_MANIFEST_QUERY_PREVIEW_CHARS = 80

__all__ = ["SegmentRecord", "ThreadStore"]

# ── Defaults ───────────────────────────────────────────────────────────

_DEFAULT_SEGMENT_TURNS = 256
_ZSTD_LEVEL = 5
_DICT_RETRAIN_EVERY = 10
_DICT_MIN_SAMPLES = 8
#: Block size for streaming reads of active and sealed segments.
_READ_BLOCK_BYTES = 65536


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
    lock: Path  # <base>.lock (cross-process advisory lock)


def _segment_paths(base: Path) -> _SegmentPaths:
    return _SegmentPaths(
        base=base,
        current=Path(f"{base}.current.jsonl"),
        index_path=Path(f"{base}.index.jsonl"),
        digest=Path(f"{base}.digest.md"),
        marker=Path(f"{base}.digest.marker"),
        dictionary=Path(f"{base}.dict.zstd"),
        legacy=Path(f"{base}.jsonl"),
        lock=Path(f"{base}.lock"),
    )


def _read_lock_context(paths: _SegmentPaths) -> contextlib.AbstractContextManager[None]:
    """Acquire ``paths.lock`` when any thread-store artifact exists on disk."""
    if (
        paths.lock.exists()
        or paths.current.exists()
        or paths.index_path.exists()
        or paths.legacy.exists()
    ):
        return file_lock(paths.lock)
    return contextlib.nullcontext()


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
    #: SHA-256 of the segment's terminal raw entry line (no newline); the
    #: hash chain continues across the seal boundary through this value.
    #: Empty on index lines written before the chain existed.
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


# ── Index / parse helpers ───────────────────────────────────────────────


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


# ── zstd decompression (read-only — compression stays in the host) ─────


def _load_zstd_dictionary(dict_path: Path) -> Any:
    """Load a trained zstd dictionary, or ``None`` when absent."""
    if not dict_path.exists():
        return None
    try:
        import zstandard  # lazy: optional dep startup cost

        return zstandard.ZstdCompressionDict(dict_path.read_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to load zstd dictionary %s: %s", dict_path, exc)
        return None


def _sealed_lines(sealed_path: Path, dict_path: Path) -> Iterator[bytes]:
    """Stream the non-empty lines of a sealed segment through a bounded decoder.

    Only one decoder block and one line are resident at a time. A missing or
    undecodable segment ends the stream after the lines already decoded.
    """
    if not sealed_path.exists():
        return
    try:
        import zstandard  # lazy: optional dep startup cost

        dictionary = _load_zstd_dictionary(dict_path)
        dctx = (
            zstandard.ZstdDecompressor(dict_data=dictionary)
            if dictionary is not None
            else zstandard.ZstdDecompressor()
        )
        with sealed_path.open("rb") as source, dctx.stream_reader(source) as reader:
            for line in io.BufferedReader(reader, _READ_BLOCK_BYTES):
                if line.strip():
                    yield line
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to decompress sealed segment %s: %s", sealed_path, exc)


def _parse_line(line: bytes) -> dict[str, Any] | None:
    """Parse one JSONL line; ``None`` for malformed or non-dict content.

    Every archive record decode goes through here, so it is the single seam
    that bounds how many bytes a read parses.
    """
    try:
        decoded = json.loads(line.decode("utf-8"))
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _active_lines(current_path: Path) -> Iterator[bytes]:
    """Stream non-empty lines of the active segment without parsing them."""
    try:
        handle = current_path.open("rb")
    except OSError:
        return
    with handle:
        for line in handle:
            if line.strip():
                yield line


def _reverse_active_lines(current_path: Path) -> Iterator[bytes]:
    """Stream non-empty active-segment lines newest-first, reading blocks from the end."""
    try:
        handle = current_path.open("rb")
    except OSError:
        return
    with handle:
        position = handle.seek(0, io.SEEK_END)
        pending: list[bytes] = []  # pieces of the current line, latest piece first
        while position > 0:
            size = min(_READ_BLOCK_BYTES, position)
            position -= size
            handle.seek(position)
            block = handle.read(size)
            pieces = block.split(b"\n")
            if len(pieces) == 1:
                pending.append(block)
                continue
            pending.append(pieces[-1])
            line = b"".join(reversed(pending))
            if line.strip():
                yield line
            for piece in reversed(pieces[1:-1]):
                if piece.strip():
                    yield piece
            pending = [pieces[0]]
        line = b"".join(reversed(pending))
        if line.strip():
            yield line


def _newest_entries(lines: Iterator[bytes], limit: int) -> list[dict[str, Any]]:
    """Parse newest-first lines until ``limit`` entries are found; return oldest-first."""
    entries: list[dict[str, Any]] = []
    for line in lines:
        entry = _parse_line(line)
        if entry is not None:
            entries.append(entry)
            if len(entries) >= limit:
                break
    entries.reverse()
    return entries


def _read_active_entries(current_path: Path) -> list[dict[str, Any]]:
    entries = (_parse_line(line) for line in _active_lines(current_path))
    return [entry for entry in entries if entry is not None]


class _ActiveScan:
    """Malformed-line map of the append-only active segment, extended incrementally.

    Turn ordinals count *valid* entries, so a malformed line shifts every later
    ordinal. One streaming pass finds the malformed line slots (parsing each
    line transiently); afterwards only newly appended complete lines are
    parsed, and reads parse just the lines they return.
    """

    def __init__(self, fingerprint: bytes) -> None:
        self.fingerprint = fingerprint
        self.offset = 0  # bytes scanned, always at a line boundary
        self.slots = 0  # non-empty lines scanned
        self.malformed: set[int] = set()


_ACTIVE_SCANS: dict[Path, _ActiveScan] = {}
_MAX_ACTIVE_SCANS = 64
_FINGERPRINT_BYTES = 256


def _active_scan(current_path: Path) -> _ActiveScan | None:
    """Bring the cached scan of ``current_path`` up to its last complete line."""
    try:
        with current_path.open("rb") as handle:
            fingerprint = handle.read(_FINGERPRINT_BYTES)
            size = handle.seek(0, io.SEEK_END)
            scan = _ACTIVE_SCANS.get(current_path)
            if scan is not None and (
                size < scan.offset or not fingerprint.startswith(scan.fingerprint[:size])
            ):
                scan = None
            if scan is None:
                scan = _ActiveScan(fingerprint)
                if len(_ACTIVE_SCANS) >= _MAX_ACTIVE_SCANS:
                    _ACTIVE_SCANS.clear()
                _ACTIVE_SCANS[current_path] = scan
            handle.seek(scan.offset)
            for line in handle:
                if not line.endswith(b"\n"):
                    break
                scan.offset += len(line)
                if not line.strip():
                    continue
                if _parse_line(line) is None:
                    scan.malformed.add(scan.slots)
                scan.slots += 1
    except OSError:
        _ACTIVE_SCANS.pop(current_path, None)
        return None
    return scan


def _active_entries(
    current_path: Path,
) -> Iterator[tuple[bytes, dict[str, Any] | None]]:
    """Yield ``(line, parsed_or_None)`` for every valid entry line, oldest first.

    Lines inside the scanned prefix are classified from the malformed map
    without parsing (``parsed`` is ``None``; the caller parses what it returns).
    Lines in the unscanned tail (a partial or just-appended line) are parsed here.
    """
    scan = _active_scan(current_path)
    if scan is None:
        return
    for slot, line in enumerate(_active_lines(current_path)):
        if slot < scan.slots:
            if slot not in scan.malformed:
                yield line, None
        else:
            entry = _parse_line(line)
            if entry is not None:
                yield line, entry


def _count_active_entries(current_path: Path) -> int:
    """Count valid active-segment entries; only newly appended lines are parsed."""
    scan = _active_scan(current_path)
    if scan is None:
        return 0
    tail = 0
    with current_path.open("rb") as handle:
        handle.seek(scan.offset)
        for line in handle:
            if line.strip() and _parse_line(line) is not None:
                tail += 1
    return scan.slots - len(scan.malformed) + tail


# ── ThreadStore ─────────────────────────────────────────────────────────


class ThreadStore:
    """Read-only access to a lossless segmented thread store.

    Owns the three-tier layout keyed by a *base path*
    (``thread_dir / <user_slug>``):

    - **Active segment** — ``<base>.current.jsonl``: plain JSONL, append-only.
    - **Sealed segments** — ``<base>.<seg_id>.jsonl.zst``: zstd-compressed
      JSONL, one per sealed chunk.
    - **Sparse index** — ``<base>.index.jsonl``: one JSON line per sealed
      segment recording ``seg_id`` and the turn/timestamp range it covers.

    The write path (append, seal, dictionary training) stays in
    ``runtime.thread``; this class is the read-only contract shared by the
    MCP server and the host's context-assembly layer.
    """

    def __init__(
        self,
        base_path: Path,
        *,
        redaction: ThreadRedactionPolicy | None = None,
    ) -> None:
        self._base = base_path
        self._paths = _segment_paths(base_path)
        self._redaction = redaction

    @property
    def base_path(self) -> Path:
        return self._base

    @property
    def paths(self) -> _SegmentPaths:
        return self._paths

    def verify_chain(self) -> ChainVerification:
        """Verify this store's tamper-evidence chain (on-demand integrity audit).

        Walks the whole store — every sealed segment is decompressed — so
        it is never part of a per-read path. See
        :func:`sdk.thread.verify_chain` for the result contract and the
        honest threat model.
        """
        from sdk.thread._verify import verify_chain as _verify

        return _verify(self._base)

    def load_recent(self, *, limit: int = 30) -> list[dict[str, Any]]:
        """Load up to ``limit`` recent thread entries (oldest-first in the tail).

        The active segment is read from its end, parsing only as many lines as
        needed. If it holds fewer than ``limit`` entries, the remainder comes
        from the tail of the most recent sealed segment, which is streamed
        through a bounded decoder keeping only the newest raw lines resident.
        ``limit <= 0`` returns every active entry. Returns ``[]`` when no
        store exists.
        """
        with _read_lock_context(self._paths):
            if limit <= 0:
                return _read_active_entries(self._paths.current)
            active = _newest_entries(_reverse_active_lines(self._paths.current), limit)
            if len(active) >= limit:
                return active
            records = _read_index(self._paths.index_path)
            if not records:
                return active
            sealed_path = _sealed_segment_path(self._paths.base, records[-1].seg_id)
            return self._sealed_tail(sealed_path, limit - len(active)) + active

    def _sealed_tail(self, sealed_path: Path, needed: int) -> list[dict[str, Any]]:
        """Return the newest ``needed`` valid entries of one sealed segment, oldest-first."""
        keep = max(needed * 2, 16)
        while True:
            window: deque[bytes] = deque(maxlen=keep)
            total = 0
            for line in _sealed_lines(sealed_path, self._paths.dictionary):
                window.append(line)
                total += 1
            entries = _newest_entries(iter(reversed(window)), needed)
            if len(entries) >= needed or total <= keep:
                return entries
            keep *= 4

    def next_turn_index(self) -> int:
        """Return the 0-based global turn index the next appended entry will receive.

        Same derivation ``runtime.thread.append_thread_entry`` uses to stamp the
        ``turn`` field at append time: the sum of all sealed-segment turn
        spans plus the current active-segment entry count. Computing this at
        turn start (before the turn runs) lets the host annotate the model
        briefing with the turn number and thread the pre-computed value
        through to the append, so the number the model saw and the number
        stamped on the entry are one source of truth — no off-by-one.
        The active segment is counted by streaming lines, never parsed.
        """
        with _read_lock_context(self._paths):
            records = _read_index(self._paths.index_path)
            sealed_spans = sum(r.turn_hi - r.turn_lo + 1 for r in records)
            return sealed_spans + _count_active_entries(self._paths.current)

    def load_range(self, *, start_turn: int, end_turn: int) -> list[dict[str, Any]]:
        """Load entries whose global turn index falls in ``[start_turn, end_turn]``.

        Global turn index is the 0-based ordinal of a turn across the whole
        store (sealed segments + active). The index is consulted to find
        overlapping segments; only those are streamed.
        """
        return [entry for _, entry in self.iter_range(start_turn=start_turn, end_turn=end_turn)]

    def load_turns(self, turns: Collection[int]) -> dict[int, dict[str, Any]]:
        """Load the entries at specific global turns in one pass, omitting absent turns."""
        wanted = {turn for turn in turns if turn >= 0}
        if not wanted:
            return {}
        return dict(self._iter(min(wanted), max(wanted), wanted))

    def iter_range(self, *, start_turn: int, end_turn: int) -> Iterator[tuple[int, dict[str, Any]]]:
        """Yield entries with their stable global ordinals without building a range list."""
        return self._iter(start_turn, end_turn, None)

    def _iter(
        self,
        start_turn: int,
        end_turn: int,
        wanted: Collection[int] | None,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        """Stream ``(turn, entry)`` in ``[start_turn, end_turn]``, optionally only ``wanted``.

        Sealed ordinals count valid entries, so lines of an overlapping sealed
        segment before the window are parsed one at a time and discarded; the
        decoder stops once the window ends. Active ordinals count valid entries too,
        taken from the cached malformed-line map, so lines outside the window
        are never re-parsed.
        """
        if start_turn > end_turn:
            return
        with _read_lock_context(self._paths):
            records = _read_index(self._paths.index_path)
            for record in records:
                if record.turn_hi < start_turn or record.turn_lo > end_turn:
                    continue
                yield from self._iter_sealed(record, start_turn, end_turn, wanted)
            active_lo = sum(r.turn_hi - r.turn_lo + 1 for r in records)
            yield from self._iter_active(active_lo, start_turn, end_turn, wanted)

    def _iter_sealed(
        self,
        record: SegmentRecord,
        start_turn: int,
        end_turn: int,
        wanted: Collection[int] | None,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        sealed_path = _sealed_segment_path(self._paths.base, record.seg_id)
        turn = record.turn_lo
        for line in _sealed_lines(sealed_path, self._paths.dictionary):
            if turn > end_turn:
                return
            entry = _parse_line(line)
            if entry is None:
                continue
            if turn >= start_turn and (wanted is None or turn in wanted):
                yield turn, entry
            turn += 1

    def _iter_active(
        self,
        first_turn: int,
        start_turn: int,
        end_turn: int,
        wanted: Collection[int] | None,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        turn = first_turn
        for line, entry in _active_entries(self._paths.current):
            if turn > end_turn:
                return
            if turn >= start_turn and (wanted is None or turn in wanted):
                parsed = entry or _parse_line(line)
                if parsed is not None:
                    yield turn, parsed
            turn += 1

    def build_recent_excerpt(
        self,
        *,
        limit: int = 3,
        is_host_command: Any | None = None,
    ) -> str:
        """Build a short human-readable excerpt of recent thread entries.

        Returns an empty string when there are no entries. Each line shows
        the query and a preview of the response (or the query itself if the
        response is empty).

        ``is_host_command`` is an optional callable ``(query: str) -> bool``
        used to strip host-only command entries (``/pack disable github``,
        ``/help``, ``/tokens``, ...) — operator UI gestures, not dialogue.
        The host passes ``runtime.host_commands.is_host_command_query``; the
        MCP server passes ``None`` (keep all entries).
        """
        raw_limit = max(limit, min(limit * 4, 200))
        entries = self.load_recent(limit=raw_limit)
        if not entries:
            return ""
        # Post-hoc sectionality: the entries are listed first, then a
        # footer note tells the model how to interpret what it just
        # read. Post-hoc framing is more effective than pre-hoc for
        # small fast models because the model anchors on recency — a
        # closing note after the data lands gets read as a definitive
        # summary, while a preamble before the data gets blurred into
        # the data itself (2026-08-22 "dogmatic about old history"
        # symptom).
        lines = ["Recorded thread context:"]
        emitted = 0
        for entry in reversed(entries):
            if emitted >= limit:
                break
            query = str(entry.get("query") or "").strip()
            response = str(entry.get("response_text") or "").strip()
            if not query and not response:
                continue
            if is_host_command is not None and is_host_command(query):
                continue
            if self._redaction is not None:
                query = self._redaction.redact(query)
                response = self._redaction.redact(response)
            body = response or query
            prov = entry.get("provenance")
            if prov:
                body = (
                    f"{body} [system_provenance: {prov.get('source', 'observed')}, "
                    f"model_id: {prov.get('model_id', 'unknown')}]"
                )
            timestamp = str(entry.get("timestamp_utc") or "unknown")
            if self._redaction is not None:
                timestamp = self._redaction.redact(timestamp)
                body = self._redaction.redact(body)
            lines.append(f"- [recorded at {timestamp}] {query or 'unknown'}: {body}")
            emitted += 1
        if emitted == 0:
            return ""
        # Post-hoc footer: lands after the data so the model treats it
        # as a closing directive, not a preamble that gets absorbed.
        footer = (
            "These are recorded past exchanges, not the active request; "
            "they do not establish current task status."
        )
        return "\n".join([lines[0], *reversed(lines[1:]), "", footer])

    def build_manifest(self, *, limit: int = 30) -> str:
        """Build a compact structured manifest of recent thread entries.

        This is the lossless replacement for the lossy prose digest. Instead
        of a summarized narrative, the model gets a "table of contents" — a
        structured index of what's in the thread — and can pull individual
        entries on demand via ``thread_read_range``.

        Each manifest line shows: turn index, mode_id, query preview
        (truncated), write-receipt capability ids, and handoff target.

        The final (most recent) row is annotated ``← latest recorded`` so the model
        can anchor on the latest recorded exchange instead of treating the flat
        table as equal-weight history (the 2026-08-14 "bringing back old
        ideas" symptom: every row looked identical, so the model re-raised
        stale queries).

        Returns an empty string when there are no entries.
        """
        entries = self.load_recent(limit=limit)
        if not entries:
            return ""
        lines = ["## Thread Manifest (Recorded History)", ""]
        lines.append("| Turn | Recorded At (UTC) | Mode | Query | Writes | Handoff |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        # ``load_recent`` returns oldest-first; the last entry is the most
        # recent turn. Derive a stable 1-based turn index from position so
        # the column is meaningful even when the stored entry lacks a
        # ``turn`` field (host-side ``append_thread_entry`` does not set
        # one). The total count anchors the index so the model sees
        # "turn 47 of 47" rather than a reset per-manifest counter.
        total = len(entries)
        for offset, entry in enumerate(entries):
            stored_turn = entry.get("turn")
            # 1-based ordinal within the manifest window (when the stored
            # entry lacks a ``turn`` field); the total suffix disambiguates
            # the window from the global sequence.
            turn = str(stored_turn) if isinstance(stored_turn, int) else f"{offset + 1}/{total}"
            mode_id = str(entry.get("mode_id") or "-")
            query = str(entry.get("query") or "").strip()
            if self._redaction is not None:
                query = self._redaction.redact(query)
            timestamp = str(entry.get("timestamp_utc") or "unknown")
            if self._redaction is not None:
                timestamp = self._redaction.redact(timestamp)
            timestamp = timestamp.replace("|", "\\|")
            if len(query) > _MANIFEST_QUERY_PREVIEW_CHARS:
                query = query[: _MANIFEST_QUERY_PREVIEW_CHARS - 3] + "…"
            # Escape pipes in query for markdown table safety.
            query = query.replace("|", "\\|")
            receipts = entry.get("effect_receipts") or []
            write_caps = [
                str(r.get("capability_id") or "")
                for r in receipts
                if isinstance(r, dict) and str(r.get("effect", "")) in ("write", "external_write")
            ]
            writes = ", ".join(sorted(set(write_caps))) or "-"
            handoff = str(entry.get("handoff_to") or entry.get("transitioned_to") or "-")
            is_latest = offset == total - 1
            row_marker = " ← latest recorded" if is_latest else ""
            lines.append(
                f"| {turn} | {timestamp} | {mode_id} | {query} | {writes} | {handoff}{row_marker} |"
            )
        return "\n".join(lines)

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        is_host_command: Any | None = None,
        end_turn: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search all thread entries by content and return ranked matches.

        Unlike ``load_recent`` / ``load_range`` (which address by position),
        this scans every entry across sealed and active segments and ranks
        them by lexical overlap with the search query. It is the finding
        aid that makes ``thread_read_range`` useful for deep history beyond
        the manifest window — the model searches to discover the right
        turn numbers, then pulls those turns for full content.

        Ranking mirrors ``SqliteMemoryStore.recall_records``
        (``sdk/memory/store.py``): whitespace-tokenized overlap scored as
        ``overlap / max(len(tokens), 1) * (1 - 1/content_word_count)``,
        with a recency tiebreak on the global turn index (descending).
        An empty query returns the most recent entries (recency fallback),
        matching the memory store's behavior.

        Each result dict carries: ``turn`` (global index), ``mode_id``,
        ``query`` (full, not truncated — the model needs the complete query
        to judge relevance), ``response_preview`` (first 200 chars of
        ``response_text``), ``handoff_to``, ``transitioned_to``, and
        ``score`` (the lexical match score, so the model can see
        confidence). Entries with no query and no response are skipped.
        Host-only command entries are stripped when ``is_host_command``
        is provided (mirrors ``build_recent_excerpt``).

        The scan streams every segment on each call with bounded memory: one
        decoded line at a time plus the top ``limit`` results.
        """
        capped_limit = max(1, min(limit, 20))
        tokens = {tok.strip().lower() for tok in query.split() if tok.strip()}

        def scored_entries() -> Iterator[tuple[float, int, dict[str, Any]]]:
            stop = sys.maxsize if end_turn is None else end_turn
            for global_turn, entry in self.iter_range(start_turn=0, end_turn=stop):
                scored = self._score_entry(entry, global_turn, tokens, is_host_command)
                if scored is not None:
                    yield scored

        # Streaming top-k: at most ``capped_limit`` scored results are resident.
        best = heapq.nlargest(capped_limit, scored_entries(), key=lambda item: (item[0], item[1]))
        return [entry for _, _, entry in best]

    def _score_entry(
        self,
        entry: dict[str, Any],
        global_turn: int,
        tokens: set[str],
        is_host_command: Any | None,
    ) -> tuple[float, int, dict[str, Any]] | None:
        """Score one entry for search ranking. Returns ``(score, turn, result)`` or ``None``.

        ``None`` means the entry should be skipped (empty, or a host-command
        entry when the filter is wired). When a :class:`ThreadRedactionPolicy`
        is wired, ``query`` and ``response_preview`` in the result are
        redacted before return — the raw entry is never mutated.
        """
        query_text = str(entry.get("query") or "").strip()
        response_text = str(entry.get("response_text") or "").strip()
        if not query_text and not response_text:
            return None
        if is_host_command is not None and is_host_command(query_text):
            return None

        mode_id = str(entry.get("mode_id") or "")
        haystack = f"{query_text} {response_text} {mode_id}".lower()
        haystack_tokens = {tok for tok in haystack.split()}
        content_word_count = max(len(haystack_tokens), 1)

        if tokens:
            overlap = len(tokens & haystack_tokens)
            lexical = overlap / max(len(tokens), 1) * (1.0 - 1.0 / content_word_count)
        else:
            # Empty query → recency fallback: every entry scores equally,
            # so the (score, turn) sort reduces to turn-descending.
            lexical = 1.0

        response_preview = response_text[:200] if response_text else ""
        if self._redaction is not None:
            query_text = self._redaction.redact(query_text)
            response_preview = self._redaction.redact(response_preview)
        result = {
            "turn": global_turn,
            "mode_id": entry.get("mode_id"),
            "query": query_text,
            "response_preview": response_preview,
            "handoff_to": entry.get("handoff_to"),
            "transitioned_to": entry.get("transitioned_to"),
            "score": round(lexical, 4),
        }
        return (lexical, global_turn, result)
