"""Lossless segmented thread-store read backend.

Extracted from ``host.thread``: the segment-path derivation, sparse
index parsing, zstd decompression, and the public read operations
(``load_recent``, ``load_range``, ``build_recent_excerpt``,
``build_manifest``). The write path (``append`` / ``write_turn_history`` /
sealing) stays in ``host.thread`` because it owns the per-turn response
and effect-receipt persistence and the host-command query check used
by ``build_recent_excerpt``.

The MCP server (``mcp_server.thread.server``) wraps this; the host
imports it directly for the context-assembly excerpt/manifest builders.
Both use the same SDK contract — the thread is the single source of
truth for workflow state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

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
    def from_dict(cls, raw: dict[str, Any]) -> "SegmentRecord":
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
        import zstandard  # noqa: PLC0415  # lazy: optional dep startup cost

        return zstandard.ZstdCompressionDict(dict_path.read_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("thread: failed to load zstd dictionary %s: %s", dict_path, exc)
        return None


def _decompress_bytes(data: bytes, dict_path: Path) -> bytes:
    """Decompress a sealed segment, using the trained dictionary when present."""
    import zstandard  # noqa: PLC0415  # lazy: optional dep startup cost

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
    ``host.thread``; this class is the read-only contract shared by the
    MCP server and the host's context-assembly layer.
    """

    def __init__(self, base_path: Path) -> None:
        self._base = base_path
        self._paths = _segment_paths(base_path)

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
                decompressed = _decompress_bytes(
                    sealed_path.read_bytes(), self._paths.dictionary
                )
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
        is_host_command: "Any" | None = None,
    ) -> str:
        """Build a short human-readable excerpt of recent thread entries.

        Returns an empty string when there are no entries. Each line shows
        the query and a preview of the response (or the query itself if the
        response is empty).

        ``is_host_command`` is an optional callable ``(query: str) -> bool``
        used to strip host-only command entries (``/pack disable github``,
        ``/help``, ``/tokens``, ...) — operator UI gestures, not dialogue.
        The host passes ``host.host_commands.is_host_command_query``; the
        MCP server passes ``None`` (keep all entries).
        """
        raw_limit = max(limit, min(limit * 4, 200))
        entries = self.load_recent(limit=raw_limit)
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
            if is_host_command is not None and is_host_command(query):
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
            if isinstance(stored_turn, int):
                turn = str(stored_turn)
            else:
                # 1-based ordinal within the manifest window; the total
                # suffix disambiguates the window from the global sequence.
                turn = f"{offset + 1}/{total}"
            mode_id = str(entry.get("mode_id") or "-")
            query = str(entry.get("query") or "").strip()
            if len(query) > 80:
                query = query[:77] + "…"
            # Escape pipes in query for markdown table safety.
            query = query.replace("|", "\\|")
            receipts = entry.get("effect_receipts") or []
            write_caps = [
                str(r.get("capability_id") or "")
                for r in receipts
                if isinstance(r, dict)
                and str(r.get("effect", "")) in ("write", "external_write")
            ]
            writes = ", ".join(sorted(set(write_caps))) or "-"
            handoff = str(entry.get("handoff_to") or entry.get("transitioned_to") or "-")
            is_current = offset == total - 1
            row_marker = " ← current" if is_current else ""
            lines.append(
                f"| {turn} | {mode_id} | {query} | {writes} | {handoff}{row_marker} |"
            )
        return "\n".join(lines)