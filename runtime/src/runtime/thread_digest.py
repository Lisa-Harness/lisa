"""Thread-derived context items pushed into the retrieval lane each turn.

The thread store is the lossless ground truth; the read tools address it on
demand. The 2026-10-02 reviewer burn (turn 455: ~500k tokens, 23
``thread_read_content`` calls reconstructing state the harness already held)
showed the cost of making tools the *primary* knowledge channel: every page
read lives in the transcript that each iteration re-sends (97% of spend is
prompt re-send, per docs/TOKEN_BUDGET.md).

This module is the push side of "push, don't pull": it derives two or three
bounded :class:`~kernel.models.ContextItem` records from the store tail using
the existing readers (``build_task_view``, recent-entry loads) and feeds
them through the single ``context_items`` candidate feed so the
session-owned ContextMapper renders them into the budgeted retrieval lane
at turn start. The model begins each turn already knowing its recorded task
state and recent history; ``thread_read_content`` remains the depth
fallback, not the primary channel.

Pure reads, no new ranking: selection stays owned by the ContextMapper and
the session plan's ``context`` block, exactly like every other candidate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel.models import ContextItem
from sdk.thread import ThreadStore, build_task_view

logger = logging.getLogger(__name__)

__all__ = ["build_thread_context_items"]

_DIGEST_NAMESPACE = "thread_history"
_DIGEST_SOURCE = "lisa_thread_digest"
_RECENT_ENTRY_LIMIT = 10
_RECENT_BODY_CHARS = 1800
_TASK_VIEW_BUDGET_BYTES = 4096
_MAX_OPEN_OBJECTIVE_LINES = 3
_SALIENCE_TASK_STATE = 0.9
_SALIENCE_RECENT_HISTORY = 0.7
_SALIENCE_OPEN_OBJECTIVES = 0.8
_CONFIDENCE = 0.9


def build_thread_context_items(
    thread_path: Path | None,
    *,
    redaction: Any = None,
) -> list[ContextItem]:
    """Derive retrieval-lane context items from the lossless thread tail.

    Returns up to three items — recorded task state, recent history, and
    open objectives — or an empty list when no store exists. Redaction is
    applied to the recorded text via the same policy the read tools use.
    """
    if thread_path is None:
        return []
    try:
        store = ThreadStore(thread_path, redaction=redaction)
        entries = [
            (turn, entry)
            for turn, entry in enumerate(store.load_recent(limit=_RECENT_ENTRY_LIMIT * 4))
        ]
    except (OSError, ValueError):
        logger.debug("thread digest: could not read recent entries", exc_info=True)
        return []
    if not entries:
        return []
    items: list[ContextItem] = []
    _append_task_state_item(items, entries, redaction)
    _append_recent_history_item(items, entries, redaction)
    _append_open_objectives_item(items, entries, redaction)
    return items


def _turn_of(entry: dict[str, Any]) -> int:
    turn = entry.get("turn")
    return int(turn) if isinstance(turn, int) else 0


def _append_task_state_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
) -> None:
    view = build_task_view(
        entries,
        redaction=redaction,
        budget_bytes=_TASK_VIEW_BUDGET_BYTES,
    )
    if view.get("status") == "empty" and not str(view.get("task_id") or ""):
        return
    body = _render_task_state(view)
    items.append(
        ContextItem(
            item_id="thread:task_state",
            namespace=_DIGEST_NAMESPACE,
            source=_DIGEST_SOURCE,
            source_type="general",
            evidence_type="task_state",
            title="Recorded task state (latest recorded task)",
            body=body,
            collected_at=str(entries[-1][1].get("timestamp_utc") or ""),
            salience_score=_SALIENCE_TASK_STATE,
            confidence=_CONFIDENCE,
            tags=["thread_digest", "task_state"],
            metadata={
                "provenance": "system_observed",
                "scope": "recorded_history",
                "selection": view.get("selection", "latest_recorded"),
            },
        )
    )


def _render_task_state(view: dict[str, Any]) -> str:
    objective = view.get("objective") or {}
    latest = view.get("latest") or {}
    lines = [
        f"task_id: {view.get('task_id', '')}",
        f"status: {view.get('status', '')} (recorded observation; not verified live state)",
        f"turns: {view.get('turns', 0)} (turn range {view.get('turn_range', [])})",
        f"objective: {str(objective.get('text') or '')[:400]}",
        f"latest response: {str(latest.get('text') or '')[:400]}",
    ]
    blocker = view.get("blocker")
    if blocker:
        lines.append(f"blocker: {blocker}")
    lines.append("Use thread tools for depth; this digest is the recorded state at last turn.")
    return "\n".join(lines)


def _append_recent_history_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
) -> None:
    body = _render_recent_history(entries, redaction)
    if not body:
        return
    items.append(
        ContextItem(
            item_id="thread:recent_history",
            namespace=_DIGEST_NAMESPACE,
            source=_DIGEST_SOURCE,
            source_type="general",
            evidence_type="recent_history",
            title="Recent recorded exchanges (oldest to newest)",
            body=body,
            collected_at=str(entries[-1][1].get("timestamp_utc") or ""),
            salience_score=_SALIENCE_RECENT_HISTORY,
            confidence=_CONFIDENCE,
            tags=["thread_digest", "history"],
            metadata={"provenance": "system_observed", "scope": "recorded_history"},
        )
    )


def _render_recent_history(entries: list[tuple[int, dict[str, Any]]], redaction: Any) -> str:
    lines: list[str] = []
    budget = _RECENT_BODY_CHARS
    for _, entry in entries:
        query = str(entry.get("query") or "").strip()
        if not query:
            continue
        response = str(entry.get("response_text") or "").strip()
        if redaction is not None:
            query = redaction.redact(query)
            response = redaction.redact(response)
        preview = response[:160] if response else ""
        line = f"- {query[:200]}: {preview}".rstrip()
        if budget - len(line) < 0:
            break
        lines.append(line)
        budget -= len(line) + 1
        if len(lines) >= _RECENT_ENTRY_LIMIT:
            break
    return "\n".join(lines)


def _append_open_objectives_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
) -> None:
    lines: list[str] = []
    for _, entry in reversed(entries):
        provenance = entry.get("provenance") or {}
        blocked = provenance.get("block_origin") or entry.get("block_origin")
        pending = provenance.get("pending_approval_request_id") or entry.get("yielded")
        if not (blocked or pending):
            continue
        query = str(entry.get("query") or "").strip()
        response = str(entry.get("response_text") or "").strip()
        if redaction is not None:
            query = redaction.redact(query)
            response = redaction.redact(response)
        if blocked:
            lines.append(f"- blocked turn ({blocked}): {query[:160]} → {response[:200]}")
        else:
            lines.append(f"- awaiting user: {query[:160]} → {response[:200]}")
        if len(lines) >= _MAX_OPEN_OBJECTIVE_LINES:
            break
    if not lines:
        return
    items.append(
        ContextItem(
            item_id="thread:open_objectives",
            namespace=_DIGEST_NAMESPACE,
            source=_DIGEST_SOURCE,
            source_type="general",
            evidence_type="open_objectives",
            title="Unfinished turns (blocked or awaiting approval/user)",
            body="\n".join(lines),
            collected_at=str(entries[-1][1].get("timestamp_utc") or ""),
            salience_score=_SALIENCE_OPEN_OBJECTIVES,
            confidence=_CONFIDENCE,
            tags=["thread_digest", "open_objectives"],
            metadata={"provenance": "system_observed", "scope": "recorded_history"},
        )
    )
