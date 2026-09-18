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
- :mod:`host.thread._store`  — mechanics: paths, index, zstd, sealing
- :mod:`host.thread._excerpt` — prompt-injection reads (excerpt + manifest)

The public read/write surface lives here, re-exported from the submodules
so the historical ``from host.thread import X`` namespace is unchanged.

See ``docs/SPEC/THREAD_STORE.md`` for the full on-disk format spec.

Dependency direction: ``thread`` ← ``main`` / ``handle_message``; the
package never imports transport code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sdk.thread import ThreadRedactionPolicy, ThreadStore

from host.thread._store import (
    SegmentRecord,
    _read_index,
    _seal_segment,
    _segment_paths,
    _maybe_rename_legacy,
    _count_lines,
    _read_active_entries_count,
)
from host.thread._excerpt import build_recent_thread_excerpt, build_thread_manifest

import json
import logging
from datetime import UTC, datetime

from host.thread._store import _DEFAULT_SEGMENT_TURNS

logger = logging.getLogger(__name__)

# ── Thread redaction policy (module-level, set once at startup) ────────
# The read-time transform applied by ``ThreadStore`` when rendering thread
# entries into model context. Injected once at host startup via
# ``set_thread_redaction_policy`` so the pure factory ``_thread_store`` can
# attach it to every store it creates. ``None`` = no redaction.
_thread_redaction: ThreadRedactionPolicy | None = None


def set_thread_redaction_policy(policy: ThreadRedactionPolicy | None) -> None:
    """Set the module-level thread redaction policy.

    Called once at host startup from ``host.bootstrap`` (or equivalent)
    after ``HostConfig`` is resolved. Pass ``None`` to disable redaction.
    """
    global _thread_redaction
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

    if turn is None:
        sealed_records = _read_index(paths.index_path)
        active_count = _read_active_entries_count(paths.current)
        global_turn = sum(r.turn_hi - r.turn_lo + 1 for r in sealed_records) + active_count
    else:
        global_turn = turn

    entry: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "user_slug": user_slug,
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
        "turn": global_turn,
    }
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

    paths.current.parent.mkdir(parents=True, exist_ok=True)
    with paths.current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


__all__ = [
    "SegmentRecord",
    "append_thread_entry",
    "build_recent_thread_excerpt",
    "build_thread_manifest",
    "load_recent_thread_entries",
    "load_thread_range",
    "next_turn_index",
    "set_thread_redaction_policy",
]
