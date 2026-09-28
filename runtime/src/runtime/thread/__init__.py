"""Transport-neutral thread persistence (segmented + zstd-sealed).

This package owns the pure, stateless logic for the per-user thread store.
The store is a three-tier lossless layout keyed by a *base path*
(``thread_dir / <user_slug>``):

- **Active segment** — ``<base>.current.jsonl``: plain JSONL, append-only.
  Bounded by ``segment_turns`` entries. This is the only file the
  recent-N read ever touches, so the read is O(``limit``) in practice
  rather than O(history).
- **Sealed segments** — ``<base>.<seg_id>.jsonl.zst``: zstd-compressed
  JSONL, one per sealed chunk. Sealing happens at append time when the
  active segment crosses ``segment_turns`` — no background scheduler.
- **Sparse index** — ``<base>.index.jsonl``: one JSON line per sealed
  segment recording ``seg_id`` and the turn/timestamp range it covers.

Modules
-------
- :mod:`runtime.thread._store`  — mechanics: paths, index, zstd, sealing
- :mod:`runtime.thread._excerpt` — prompt-injection reads (excerpt + manifest)

The public read/write surface lives here, re-exported from the submodules
so the historical ``from runtime.thread import X`` namespace is unchanged.

See ``docs/SPEC/THREAD_STORE.md`` for the full on-disk format spec.

Dependency direction: ``thread`` ← ``main`` / ``handle_message``; the
package never imports transport code.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdk.thread import ThreadRedactionPolicy, ThreadStore

from runtime.thread._excerpt import build_recent_thread_excerpt, build_thread_manifest
from runtime.thread._replay import build_open_task_messages
from runtime.thread._store import (
    _DEFAULT_SEGMENT_TURNS,
    SegmentRecord,
    _count_lines,
    _maybe_rename_legacy,
    _read_active_entries_count,
    _read_index,
    _seal_segment,
    _segment_paths,
    _SegmentPaths,
)

logger = logging.getLogger(__name__)

# ── Thread redaction policy (module-level, set once at startup) ────────
# The read-time transform applied by ``ThreadStore`` when rendering thread
# entries into model context. Injected once at host startup via
# ``set_thread_redaction_policy`` so the pure factory ``_thread_store`` can
# attach it to every store it creates. ``None`` = no redaction.
_thread_redaction: ThreadRedactionPolicy | None = None


def set_thread_redaction_policy(policy: ThreadRedactionPolicy | None) -> None:
    """Set the module-level thread redaction policy.

    Called once at host startup from ``runtime.bootstrap`` (or equivalent)
    after ``HostConfig`` is resolved. Pass ``None`` to disable redaction.
    """
    global _thread_redaction  # noqa: PLW0603 — module-level singleton seam
    _thread_redaction = policy


def _thread_store(thread_path: Path | None) -> ThreadStore | None:
    """Return a ``ThreadStore`` for ``thread_path``, or ``None``.

    Attaches the module-level redaction policy so every read applies the
    configured strikethrough transform.
    """
    if thread_path is None:
        return None
    return ThreadStore(thread_path, redaction=_thread_redaction)


def load_recent_thread_entries(
    *,
    thread_path: Path | None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Load up to ``limit`` recent thread entries (oldest-first in the tail).

    Delegates to :meth:`sdk.thread.ThreadStore.load_recent`.
    Returns ``[]`` when ``thread_path`` is ``None`` or no store exists.
    """
    store = _thread_store(thread_path)
    if store is None:
        return []
    return store.load_recent(limit=limit)


def load_thread_range(
    *,
    thread_path: Path | None,
    start_turn: int,
    end_turn: int,
) -> list[dict[str, Any]]:
    """Load entries whose global turn index falls in ``[start_turn, end_turn]``.

    Delegates to :meth:`sdk.thread.ThreadStore.load_range`.
    """
    store = _thread_store(thread_path)
    if store is None:
        return []
    return store.load_range(start_turn=start_turn, end_turn=end_turn)


def paused_turn_owner_mode(*, thread_path: Path | None) -> str:
    """Return the mode that owned the latest turn if it ended paused, else ``""``."""
    entries = load_recent_thread_entries(thread_path=thread_path, limit=1)
    if not entries:
        return ""
    last = entries[-1]
    if (last.get("provenance") or {}).get("operator_exit") != "paused":
        return ""
    return str(last.get("handoff_to") or last.get("mode_id") or "")


def next_turn_index(*, thread_path: Path | None) -> int:
    """Return the 0-based global turn index the next appended entry will receive.

    Delegates to :meth:`sdk.thread.ThreadStore.next_turn_index`. Returns ``0``
    when ``thread_path`` is ``None`` or no store exists. The host computes
    this at turn start so it can annotate the model briefing with the turn
    number and thread the pre-computed value through to ``append_thread_entry``.
    """
    store = _thread_store(thread_path)
    if store is None:
        return 0
    return store.next_turn_index()


def _resolve_thread_turn(
    paths: _SegmentPaths,
    turn: int | None,
) -> int:
    """Resolve the global turn number for a new thread entry.

    Uses the caller-provided ``turn`` when set; otherwise computes it from
    the sealed-segment index plus the active segment's entry count.
    """
    if turn is not None:
        return turn
    sealed_records = _read_index(paths.index_path)
    active_count = _read_active_entries_count(paths.current)
    return sum(r.turn_hi - r.turn_lo + 1 for r in sealed_records) + active_count


def _build_thread_entry(
    *,
    user_slug: str,
    mode_id: str,
    query: str,
    response_text: str,
    global_turn: int,
    content_source: str,
    reasoning_text: str,
    provenance: dict[str, Any] | None,
    effect_receipts: list[dict[str, Any]] | None,
    handoff_to: str,
    transitioned_to: str,
    yielded: bool,
    yield_question: str,
    turn_input: dict[str, Any] | None,
    memory_ids: list[int] | None,
    transcript: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one thread-entry JSON dict, omitting empty optional fields."""
    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "user_slug": user_slug,
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
        "turn": global_turn,
    }
    _apply_thread_metadata_fields(
        entry,
        provenance=provenance,
        effect_receipts=effect_receipts,
        handoff_to=handoff_to,
        transitioned_to=transitioned_to,
        yielded=yielded,
        yield_question=yield_question,
    )
    _apply_thread_content_fields(
        entry,
        turn_input=turn_input,
        memory_ids=memory_ids,
        content_source=content_source,
        reasoning_text=reasoning_text,
    )
    if transcript:
        entry["transcript"] = transcript
    return entry


def _apply_thread_metadata_fields(
    entry: dict[str, Any],
    *,
    provenance: dict[str, Any] | None,
    effect_receipts: list[dict[str, Any]] | None,
    handoff_to: str,
    transitioned_to: str,
    yielded: bool,
    yield_question: str,
) -> None:
    """Write the provenance / receipt / handoff / yield metadata fields (in place)."""
    if provenance:
        entry["provenance"] = provenance
    if effect_receipts:
        entry["effect_receipts"] = effect_receipts
    if handoff_to:
        entry["handoff_to"] = handoff_to
    if transitioned_to:
        entry["transitioned_to"] = transitioned_to
    if yielded:
        entry["yielded"] = True
        if yield_question:
            entry["yield_question"] = yield_question


def _apply_thread_content_fields(
    entry: dict[str, Any],
    *,
    turn_input: dict[str, Any] | None,
    memory_ids: list[int] | None,
    content_source: str,
    reasoning_text: str,
) -> None:
    """Write the turn-input / memory / content-source fields (in place)."""
    if turn_input:
        entry["turn_input"] = turn_input
    if memory_ids:
        entry["memory_ids"] = memory_ids
    if content_source:
        # "reasoning_tail": ``response_text`` is a labeled quote of the
        # tail of the model's reasoning (no visible answer was produced).
        # The FULL reasoning rides ``reasoning_text`` so the log stays
        # exhaustive at rest; playback surfaces redact/bound that field.
        entry["content_source"] = content_source
    if reasoning_text:
        entry["reasoning_text"] = reasoning_text


def append_thread_entry(
    *,
    thread_path: Path | None,
    user_slug: str,
    mode_id: str,
    query: str,
    response_text: str,
    content_source: str = "",
    reasoning_text: str = "",
    provenance: dict[str, Any] | None = None,
    effect_receipts: list[dict[str, Any]] | None = None,
    handoff_to: str = "",
    transitioned_to: str = "",
    segment_turns: int = _DEFAULT_SEGMENT_TURNS,
    turn: int | None = None,
    yielded: bool = False,
    yield_question: str = "",
    turn_input: dict[str, Any] | None = None,
    memory_ids: list[int] | None = None,
    transcript: list[dict[str, Any]] | None = None,
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

    global_turn = _resolve_thread_turn(paths, turn)
    entry = _build_thread_entry(
        user_slug=user_slug,
        mode_id=mode_id,
        query=query,
        response_text=response_text,
        global_turn=global_turn,
        content_source=content_source,
        reasoning_text=reasoning_text,
        provenance=provenance,
        effect_receipts=effect_receipts,
        handoff_to=handoff_to,
        transitioned_to=transitioned_to,
        yielded=yielded,
        yield_question=yield_question,
        turn_input=turn_input,
        memory_ids=memory_ids,
        transcript=transcript,
    )

    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


__all__ = [
    "SegmentRecord",
    "append_thread_entry",
    "build_open_task_messages",
    "build_recent_thread_excerpt",
    "build_thread_manifest",
    "load_recent_thread_entries",
    "load_thread_range",
    "next_turn_index",
    "paused_turn_owner_mode",
    "set_thread_redaction_policy",
]
