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

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

from sdk.thread.redaction import ThreadRedactionPolicy

logger = logging.getLogger(__name__)

#: Query preview width in the thread manifest table rows.
_MANIFEST_QUERY_PREVIEW_CHARS = 80

__all__ = ["SegmentRecord", "ThreadStore"]

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


def _decompress_bytes(data: bytes, dict_path: Path) -> bytes:
    """Decompress a sealed segment, using the trained dictionary when present."""
    import zstandard  # lazy: optional dep startup cost

    dictionary = _load_zstd_dictionary(dict_path)
    if dictionary is not None:
        dctx = zstandard.ZstdDecompressor(dict_data=dictionary)
    else:
        dctx = zstandard.ZstdDecompressor()
    return bytes(dctx.decompress(data))


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

    def load_recent(self, *, limit: int = 30) -> list[dict[str, Any]]:
        """Load up to ``limit`` recent thread entries (oldest-first in the tail).

        Reads only the active segment. If it has fewer than ``limit``
        entries, falls back to the most recent sealed segment for the
        remaining tail. Two file touches max. Returns ``[]`` when no
        store exists.
        """
        active = _read_active_entries(self._paths.current)
        if len(active) >= limit > 0:
            return active[-limit:]
        if limit <= 0 or len(active) >= limit:
            return active
        records = _read_index(self._paths.index_path)
        if not records:
            return active
        last = records[-1]
        sealed_path = _sealed_segment_path(self._paths.base, last.seg_id)
        sealed_entries = self._load_sealed_entries(sealed_path, last.seg_id)
        if not sealed_entries:
            return active
        needed = limit - len(active)
        tail = sealed_entries[-needed:] if needed < len(sealed_entries) else sealed_entries
        return tail + active

    def _load_sealed_entries(
        self,
        sealed_path: Path,
        seg_id: str,
    ) -> list[dict[str, Any]]:
        """Decompress and parse a sealed segment. Returns [] on any failure."""
        if not sealed_path.exists():
            return []
        try:
            decompressed = _decompress_bytes(sealed_path.read_bytes(), self._paths.dictionary)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("thread: failed to decompress sealed segment %s: %s", seg_id, exc)
            return []
        return _parse_entries(decompressed.decode("utf-8").splitlines())

    def next_turn_index(self) -> int:
        """Return the 0-based global turn index the next appended entry will receive.

        Same derivation ``runtime.thread.append_thread_entry`` uses to stamp the
        ``turn`` field at append time: the sum of all sealed-segment turn
        spans plus the current active-segment entry count. Computing this at
        turn start (before the turn runs) lets the host annotate the model
        briefing with the turn number and thread the pre-computed value
        through to the append, so the number the model saw and the number
        stamped on the entry are one source of truth — no off-by-one.
        """
        records = _read_index(self._paths.index_path)
        sealed_spans = sum(r.turn_hi - r.turn_lo + 1 for r in records)
        active_count = len(_read_active_entries(self._paths.current))
        return sealed_spans + active_count

    def load_range(self, *, start_turn: int, end_turn: int) -> list[dict[str, Any]]:
        """Load entries whose global turn index falls in ``[start_turn, end_turn]``.

        Global turn index is the 0-based ordinal of a turn across the whole
        store (sealed segments + active). The index is consulted to find
        overlapping segments; only those are decompressed. The active segment
        is read directly for its in-range tail.
        """
        if start_turn > end_turn:
            return []
        records = _read_index(self._paths.index_path)
        results: list[dict[str, Any]] = []
        for record in records:
            if record.turn_hi < start_turn or record.turn_lo > end_turn:
                continue
            sealed_path = _sealed_segment_path(self._paths.base, record.seg_id)
            if not sealed_path.exists():
                continue
            try:
                decompressed = _decompress_bytes(sealed_path.read_bytes(), self._paths.dictionary)
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
        for offset, entry in enumerate(_read_active_entries(self._paths.current)):
            global_turn = active_turn_lo + offset
            if start_turn <= global_turn <= end_turn:
                results.append(entry)
        return results

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
        lines = ["Recent thread context:"]
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
            lines.append(f"- {query or 'unknown'}: {body}")
            emitted += 1
        if emitted == 0:
            return ""
        # Post-hoc footer: lands after the data so the model treats it
        # as a closing directive, not a preamble that gets absorbed.
        lines.append("")
        lines.append(
            "The entries above are resolved past exchanges — the user is "
            "now responding to the most recent one. Do not re-raise or "
            "re-narrate resolved issues from earlier rows."
        )
        return "\n".join([lines[0], *reversed(lines[1:])])

    def build_manifest(self, *, limit: int = 30) -> str:
        """Build a compact structured manifest of recent thread entries.

        This is the lossless replacement for the lossy prose digest. Instead
        of a summarized narrative, the model gets a "table of contents" — a
        structured index of what's in the thread — and can pull individual
        entries on demand via ``thread_read_range``.

        Each manifest line shows: turn index, mode_id, query preview
        (truncated), write-receipt capability ids, and handoff target.

        The final (most recent) row is annotated ``← current`` so the model
        can anchor on the latest exchange instead of treating the flat
        table as equal-weight history (the 2026-08-14 "bringing back old
        ideas" symptom: every row looked identical, so the model re-raised
        stale queries).

        Returns an empty string when there are no entries.
        """
        entries = self.load_recent(limit=limit)
        if not entries:
            return ""
        lines = ["## Thread Manifest", ""]
        lines.append("| Turn | Mode | Query | Writes | Handoff |")
        lines.append("| --- | --- | --- | --- | --- |")
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
            is_current = offset == total - 1
            row_marker = " ← current" if is_current else ""
            lines.append(f"| {turn} | {mode_id} | {query} | {writes} | {handoff}{row_marker} |")
        return "\n".join(lines)

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        is_host_command: Any | None = None,
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

        The scan decompresses every sealed segment on each call. This is
        acceptable for a tool the model calls occasionally; for
        high-frequency search an in-memory entry cache (rebuilt on seal)
        would be the next optimization.
        """
        capped_limit = max(1, min(limit, 20))
        tokens = {tok.strip().lower() for tok in query.split() if tok.strip()}

        # Gather every entry with its global turn index.
        records = _read_index(self._paths.index_path)
        candidates: list[tuple[float, int, dict[str, Any]]] = []

        for record in records:
            sealed_path = _sealed_segment_path(self._paths.base, record.seg_id)
            sealed_entries = self._load_sealed_entries(sealed_path, record.seg_id)
            for offset, entry in enumerate(sealed_entries):
                global_turn = record.turn_lo + offset
                scored = self._score_entry(entry, global_turn, tokens, is_host_command)
                if scored is not None:
                    candidates.append(scored)

        active_turn_lo = sum(r.turn_hi - r.turn_lo + 1 for r in records)
        for offset, entry in enumerate(_read_active_entries(self._paths.current)):
            global_turn = active_turn_lo + offset
            scored = self._score_entry(entry, global_turn, tokens, is_host_command)
            if scored is not None:
                candidates.append(scored)

        if not candidates:
            return []

        # Sort by (score desc, turn desc) — recency tiebreak.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in candidates[:capped_limit]]

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
