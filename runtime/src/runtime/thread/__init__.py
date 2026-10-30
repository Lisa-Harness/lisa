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
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

from sdk.filelock import file_lock
from sdk.thread import ThreadRedactionPolicy, ThreadStore, load_result_object, project_tool_messages

from runtime.thread import _reservations
from runtime.thread._excerpt import build_recent_thread_excerpt, build_thread_manifest
from runtime.thread._replay import build_open_task_messages
from runtime.thread._results import _persist_result
from runtime.thread._store import (
    _DEFAULT_SEGMENT_TURNS,
    SegmentRecord,
    _count_lines,
    _maybe_rename_legacy,
    _prev_entry_sha,
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


def get_thread_redaction_policy() -> ThreadRedactionPolicy | None:
    """Return the module-level thread redaction policy (``None`` = disabled)."""
    return _thread_redaction


def project_thread_messages(
    messages: list[dict[str, Any]],
    task_id: str = "",
    *,
    thread_path: Path | None,
    budget_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Persist large raw results before sending recoverable previews to a model.

    ``budget_bytes`` bounds the resident tool-result bytes in this projection
    (``None`` keeps the sdk default ceiling); the turn-start replay path uses
    it to recompute the resident working set under the session's replay lane.
    """
    if thread_path is None:
        raise ValueError("Tool-result projection requires a thread storage path")
    return project_tool_messages(
        messages,
        task_id,
        lambda result: _persist_result(thread_path, result),
        _thread_redaction,
        lambda ref: load_result_object(thread_path, ref),
        budget_bytes=budget_bytes,
    )


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
    provenance = last.get("provenance") or {}
    source = provenance.get("source")
    if source in {"host-approved", "host-approved-failed"}:
        return str(last.get("mode_id") or "")
    if not (
        provenance.get("operator_exit") == "paused" or provenance.get("pending_approval_request_id")
    ):
        return ""
    return str(last.get("handoff_to") or last.get("mode_id") or "")


def latest_turn_owner_mode(*, thread_path: Path | None) -> str:
    """Return the mode the latest turn left active, or ``""`` for an empty thread.

    A long-lived host keeps this as ``current_mode_id``; a host started per
    message recovers it here so the next turn does not re-enter the start
    mode, re-route, and miss the owner mode's provider cache prefix.
    """
    entries = load_recent_thread_entries(thread_path=thread_path, limit=1)
    if not entries:
        return ""
    last = entries[-1]
    if (last.get("provenance") or {}).get("source") in {"host-approved", "host-approved-failed"}:
        return str(last.get("mode_id") or "")
    return str(last.get("handoff_to") or last.get("mode_id") or "")


def next_turn_index(*, thread_path: Path | None) -> int:
    """Return the number of committed entries — the next entry's index if no turn overlaps.

    Delegates to :meth:`sdk.thread.ThreadStore.next_turn_index`. Returns ``0``
    when ``thread_path`` is ``None`` or no store exists. A turn starting
    while others are in flight takes its provisional index from
    :func:`reserved_turn` instead, so overlapping turns do not share one.
    """
    store = _thread_store(thread_path)
    if store is None:
        return 0
    return store.next_turn_index()


class TurnReservation(NamedTuple):
    """A turn's provisional index and the token that releases its reservation."""

    turn: int
    token: str


@contextmanager
def reserved_turn(*, thread_path: Path | None) -> Iterator[TurnReservation]:
    """Reserve a provisional turn index for the duration of one turn.

    The index is ``max(committed entries, highest live reservation + 1)``,
    so overlapping turns on one thread get distinct numbers for the briefing,
    banner and telemetry. Pass ``reservation.token`` to
    :func:`append_thread_entry`, which releases it atomically with the commit;
    leaving the block releases it on every other exit path. The committed
    index is still the log position and can differ — the entry then records
    the provisional one as ``provisional_turn``. With ``thread_path=None`` the
    reservation is ``(0, "")`` and nothing is persisted.
    """
    if thread_path is None:
        yield TurnReservation(turn=0, token="")
        return
    paths = _segment_paths(thread_path)
    token = uuid4().hex
    thread_path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(paths.lock):
        _maybe_rename_legacy(paths)
        turn = _reservations.reserve(
            _reservations.reservations_path(thread_path),
            entry_count=_resolve_thread_turn(paths),
            token=token,
        )
    try:
        yield TurnReservation(turn=turn, token=token)
    finally:
        with file_lock(paths.lock):
            _reservations.release(_reservations.reservations_path(thread_path), token=token)


def _resolve_thread_turn(paths: _SegmentPaths) -> int:
    """Resolve the global turn number for a new thread entry under the store lock.

    The turn number is the entry's position in the log: the sealed-segment
    turn count plus the active segment's entry count. It is never taken from
    the caller's provisional index — overlapping turns commit in completion
    order, so a provisional index may already be taken or may run ahead of
    the log, and honouring it would duplicate or skip a turn number.
    """
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
    verified_facts: list[dict[str, Any]] | None = None,
    departed_modes: list[str] | None = None,
    provisional_turn: int | None = None,
    prev_entry_sha256: str = "",
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
    if provisional_turn is not None and provisional_turn != global_turn:
        # The turn ran (and stamped its telemetry) under a different index
        # than the one it committed at; forensics remaps through this.
        entry["provisional_turn"] = provisional_turn
    if prev_entry_sha256:
        # Tamper-evidence chain: SHA-256 of the previous stored entry's
        # raw line, "" (omitted) only for the first chained entry.
        entry["prev_entry_sha256"] = prev_entry_sha256
    _apply_thread_metadata_fields(
        entry,
        provenance=provenance,
        effect_receipts=effect_receipts,
        handoff_to=handoff_to,
        transitioned_to=transitioned_to,
        yielded=yielded,
        yield_question=yield_question,
        verified_facts=verified_facts,
    )
    if departed_modes:
        entry["departed_modes"] = departed_modes
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
    verified_facts: list[dict[str, Any]] | None,
) -> None:
    """Write the provenance / receipt / handoff / yield metadata fields (in place)."""
    if provenance:
        entry["provenance"] = provenance
    if effect_receipts:
        entry["effect_receipts"] = effect_receipts
    if verified_facts:
        entry["verified_facts"] = verified_facts
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
    verified_facts: list[dict[str, Any]] | None = None,
    departed_modes: list[str] | None = None,
    reservation: str = "",
) -> int | None:
    """Append a new thread entry to the active segment under the thread store lock.

    Seals the active segment first when it has reached ``segment_turns``
    entries (pressure-driven, synchronous, sub-10ms for 256 lines). No-ops
    and returns ``None`` when ``thread_path`` is ``None`` or ``user_slug`` is
    empty; otherwise returns the committed 0-based ``global_turn``. ``turn``
    is the provisional index the turn ran under; the committed index is the
    entry's log position, and a mismatch is recorded as ``provisional_turn``.
    ``reservation`` is the :func:`reserved_turn` token, released under the
    same lock so no new turn can observe the entry without the reservation
    being gone (or vice versa).
    """
    if thread_path is None or not user_slug:
        return None
    paths = _segment_paths(thread_path)
    with file_lock(paths.lock):
        _maybe_rename_legacy(paths)

        thread_path.parent.mkdir(parents=True, exist_ok=True)
        if paths.current.exists() and _count_lines(paths.current) >= segment_turns:
            _seal_segment(paths)

        global_turn = _resolve_thread_turn(paths)
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
            verified_facts=verified_facts,
            departed_modes=departed_modes,
            provisional_turn=turn,
            prev_entry_sha256=_prev_entry_sha(paths),
        )

        paths.current.parent.mkdir(parents=True, exist_ok=True)
        with paths.current.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        _reservations.release(_reservations.reservations_path(thread_path), token=reservation)
    return global_turn


__all__ = [
    "SegmentRecord",
    "TurnReservation",
    "append_thread_entry",
    "build_open_task_messages",
    "build_recent_thread_excerpt",
    "build_thread_manifest",
    "latest_turn_owner_mode",
    "load_recent_thread_entries",
    "load_thread_range",
    "next_turn_index",
    "paused_turn_owner_mode",
    "reserved_turn",
    "set_thread_redaction_policy",
]
