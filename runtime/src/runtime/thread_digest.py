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

__all__ = [
    "build_thread_context_items",
    "recorded_approval_request_ids",
    "resolve_live_workflow_state",
]

_DIGEST_NAMESPACE = "thread_history"
_DIGEST_SOURCE = "lisa_thread_digest"
#: Defaults mirror ``kernel.session_plan.ThreadDigestPlan``; the session
#: plan's ``context.history.digest`` block overrides them.
_RECENT_ENTRY_LIMIT = 10
_RECENT_BODY_CHARS = 1800
_TASK_VIEW_BUDGET_BYTES = 4096
_MAX_OPEN_OBJECTIVE_LINES = 3
_SALIENCE_TASK_STATE = 0.9
_SALIENCE_VERIFIED_FACTS = 0.95
_SALIENCE_RECENT_HISTORY = 0.7
_SALIENCE_OPEN_OBJECTIVES = 0.8
_CONFIDENCE = 0.9
#: Per-fact claim bound inside the digest item — long enough for one real
#: conclusion, short enough that a dozen verdicts stay cheap to render.
_FACT_CLAIM_CHARS = 200
#: Rendering cap for the item: newest facts win when the budget runs out.
_FACT_ITEM_BUDGET_CHARS = 2048
#: Shared past/present framing carried once per digest assembly by the
#: ``thread:epistemics`` item (F4) — the same framing the
#: recent_conversation lane and readout pages carry. Post-hoc (after the
#: data) so a small fast model reads it as a closing directive, not a
#: preamble it blurs into the data (2026-08-22 "dogmatic about old
#: history" finding; re-confirmed by the 2026-10-05 turn-510 stale-yield
#: loop where "status: awaiting_approval" was read as the live present).
_PAST_PRESENT_DIRECTIVE = (
    "This is the recorded past as of the timestamped turns above — "
    "not the live present. Live state is what tools return in the current turn; "
    "verify any recorded blocker against the approval/recovery state before "
    "asking the operator to decide it."
)


def build_thread_context_items(
    thread_path: Path | None,
    *,
    redaction: Any = None,
    settings: Any = None,
    live_state: Any = None,
) -> list[ContextItem]:
    """Derive retrieval-lane context items from the lossless thread tail.

    Returns up to three items — recorded task state, recent history, and
    open objectives — or an empty list when no store exists. ``settings``
    is the session plan's ``context.history.digest`` block (None →
    defaults). Redaction is applied to the recorded text via the same
    policy the read tools use.

    ``live_state`` is the host's pre-turn snapshot of durable workflow
    state (``resolve_live_workflow_state``): recorded approval blockers are
    reconciled against it so a blocker the store already settled is
    annotated as superseded instead of being replayed as a live prompt —
    the failure mode that trapped the 2026-10-05 thread in a stale-yield
    loop after a crash lost the continuation turn.
    """
    if thread_path is None:
        return []
    if settings is not None and not getattr(settings, "enabled", True):
        return []
    recent_limit = _setting(settings, "recent_entries", _RECENT_ENTRY_LIMIT)
    body_chars = _setting(settings, "recent_body_chars", _RECENT_BODY_CHARS)
    view_budget = _setting(settings, "task_view_budget_bytes", _TASK_VIEW_BUDGET_BYTES)
    objective_lines = _setting(settings, "max_open_objective_lines", _MAX_OPEN_OBJECTIVE_LINES)
    try:
        store = ThreadStore(thread_path, redaction=redaction)
        entries = [
            (turn, entry) for turn, entry in enumerate(store.load_recent(limit=recent_limit * 4))
        ]
    except (OSError, ValueError):
        logger.debug("thread digest: could not read recent entries", exc_info=True)
        return []
    if not entries:
        return []
    items: list[ContextItem] = []
    _append_task_state_item(items, entries, redaction, view_budget, live_state)
    _append_verified_facts_item(items, entries, redaction)
    _append_recent_history_item(items, entries, redaction, recent_limit, body_chars)
    _append_open_objectives_item(items, entries, redaction, objective_lines, live_state)
    # F4: one shared epistemics footer instead of a copy in every body —
    # emitted only when at least one other digest item exists.
    if items:
        _append_epistemics_item(items, entries)
    return items


def _setting(settings: Any, name: str, default: int) -> int:
    value = getattr(settings, name, None) if settings is not None else None
    return int(value) if isinstance(value, int) and value > 0 else default


def _turn_of(entry: dict[str, Any]) -> int:
    turn = entry.get("turn")
    return int(turn) if isinstance(turn, int) else 0


def recorded_approval_request_ids(
    thread_path: Path | None,
    *,
    limit: int = 40,
) -> list[str]:
    """Return approval request ids named by recorded blockers in the tail.

    The host calls this before the turn (async context) and feeds the ids
    to :func:`resolve_live_workflow_state` so the digest can reconcile the
    recorded blockers against the live approval store without the digest
    itself doing any I/O.
    """
    if thread_path is None:
        return []
    try:
        store = ThreadStore(thread_path)
        entries = store.load_recent(limit=limit)
    except (OSError, ValueError):
        return []
    request_ids: list[str] = []
    for entry in reversed(entries):
        request_id = str((entry.get("provenance") or {}).get("pending_approval_request_id") or "")
        if request_id and request_id not in request_ids:
            request_ids.append(request_id)
    return request_ids


def _superseded_note(
    entry: dict[str, Any],
    live_state: Any,
) -> str:
    """Return the live-state reconciliation note for a recorded approval blocker.

    Empty when the recorded entry has no approval blocker, when no live
    snapshot was supplied, or when the live store has no record for the
    request id. The note names the store outcome so the model can see the
    recorded "awaiting your decision" is history, not a live prompt.
    """
    request_id = str((entry.get("provenance") or {}).get("pending_approval_request_id") or "")
    if not request_id or live_state is None:
        return ""
    resolved = getattr(live_state, "resolved_approvals", None) or {}
    outcome = str(resolved.get(request_id) or "").strip()
    # "pending" is the store confirming the decision is still outstanding —
    # the live state, not the recorded past. Annotating it as superseded
    # would present an undecided approval as settled and suppress the yield.
    if not outcome or outcome == "pending":
        return ""
    return f" [superseded (live approval store: {outcome})]"


def _resolved_on_evidence(
    entry: dict[str, Any],
    later_entries: list[tuple[int, dict[str, Any]]],
) -> bool:
    """Return True when a later same-task entry resolves this blocker.

    F3 (2026-10-14 self-review): recorded blockers and yielded questions
    re-rendered as open every turn although later manifest turns showed
    the work landed — stale blockers re-billed each turn and nudged the
    model to re-litigate finished work. A later entry carrying the same
    task id and no blocker/yield marker is resolution evidence: the
    objective proceeded past the blocker. Entries without a task id
    never resolve on evidence (a later turn cannot be attributed to
    them); live-approval reconciliation stays in ``_superseded_note``.
    """
    task_id = str((entry.get("provenance") or {}).get("task_id") or "")
    if not task_id:
        return False
    for _, later in later_entries:
        later_provenance = later.get("provenance") or {}
        if str(later_provenance.get("task_id") or "") != task_id:
            continue
        later_blocked = later_provenance.get("block_origin") or later.get("block_origin")
        later_pending = later_provenance.get("pending_approval_request_id") or later.get("yielded")
        if not (later_blocked or later_pending):
            return True
    return False


def _append_task_state_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
    view_budget: int,
    live_state: Any = None,
) -> None:
    view = build_task_view(
        entries,
        redaction=redaction,
        budget_bytes=view_budget,
    )
    if view.get("status") == "empty" and not str(view.get("task_id") or ""):
        return
    body = _render_task_state(view, entries, live_state)
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


def _render_task_state(
    view: dict[str, Any],
    entries: list[tuple[int, dict[str, Any]]],
    live_state: Any = None,
) -> str:
    objective = view.get("objective") or {}
    latest = view.get("latest") or {}
    turn_range = view.get("turn_range") or [0, 0]
    latest_entry = entries[-1][1]
    superseded = _superseded_note(latest_entry, live_state)
    lines = [
        f"task_id: {view.get('task_id', '')}",
        f"status: {view.get('status', '')} (recorded observation; not verified live state)",
        f"recorded at turn {latest_entry.get('turn', turn_range[1])} "
        f"(turn range {turn_range}); as_of: {view.get('as_of_utc') or 'unknown'})",
        f"objective: {str(objective.get('text') or '')[:400]}",
        f"latest response: {str(latest.get('text') or '')[:400]}{superseded}",
    ]
    blocker = view.get("blocker")
    if blocker:
        lines.append(f"blocker: {blocker}")
    return "\n".join(lines)


def _append_verified_facts_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
) -> None:
    """Push verified conclusions as a durable, retrieval-lane item.

    Facts ride the thread record (``verified_facts`` on each entry), so they
    survive turns, worktree replacement, and preview collapse. The render is
    newest-first and budget-capped: a fact is one conclusion plus its
    ``at`` anchor (``file:line``) and the ``revision`` it was verified
    against, so a later turn can cite evidence without re-deriving it from
    a receipt teaser (the gap behind the 2026-10-05 audit re-mining).
    """
    lines: list[str] = []
    budget = _FACT_ITEM_BUDGET_CHARS
    for _, entry in reversed(entries):
        facts = entry.get("verified_facts") or []
        if not isinstance(facts, list):
            continue
        timestamp = str(entry.get("timestamp_utc") or "unknown")
        for fact in reversed(facts):
            if not isinstance(fact, dict):
                continue
            claim = str(fact.get("claim") or "").strip()[:_FACT_CLAIM_CHARS]
            if not claim:
                continue
            if redaction is not None:
                claim = redaction.redact(claim)
            at = str(fact.get("at") or "").strip()
            revision = str(fact.get("revision") or "").strip()
            suffix = f" ({at}" + (f" @ {revision}" if revision else "") + ")" if at else ""
            line = f"- [recorded {timestamp}] {claim}{suffix}"
            if budget - len(line) <= 0:
                break
            lines.append(line)
            budget -= len(line) + 1
        if budget <= 0:
            break
    if not lines:
        return
    items.append(
        ContextItem(
            item_id="thread:verified_facts",
            namespace=_DIGEST_NAMESPACE,
            source=_DIGEST_SOURCE,
            source_type="general",
            evidence_type="verified_facts",
            title="Verified facts (recorded conclusions, newest first)",
            body="\n".join(lines),
            collected_at=str(entries[-1][1].get("timestamp_utc") or ""),
            salience_score=_SALIENCE_VERIFIED_FACTS,
            confidence=_CONFIDENCE,
            tags=["thread_digest", "verified_facts"],
            metadata={"provenance": "system_observed", "scope": "recorded_history"},
        )
    )


def _append_recent_history_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
    recent_limit: int,
    body_chars: int,
) -> None:
    body = _render_recent_history(entries, redaction, recent_limit, body_chars)
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


def _render_recent_history(
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
    recent_limit: int,
    body_chars: int,
) -> str:
    lines: list[str] = []
    budget = body_chars
    # Newest-first truncation: ``recent_entries`` bounds how many of the
    # most recent exchanges appear, not the oldest ones in the window.
    window = entries[-recent_limit:]
    window_start = len(entries) - len(window)
    for window_pos in range(len(window) - 1, -1, -1):
        _, entry = window[window_pos]
        query = str(entry.get("query") or "").strip()
        if not query:
            continue
        timestamp = str(entry.get("timestamp_utc") or "unknown")
        provenance = entry.get("provenance") or {}
        blocked = provenance.get("block_origin") or entry.get("block_origin")
        pending = provenance.get("pending_approval_request_id") or entry.get("yielded")
        if (blocked or pending) and not _resolved_on_evidence(
            entry, entries[window_start + window_pos + 1 :]
        ):
            # F4: open objectives owns the canonical copy of a still-open
            # turn; history keeps a ref instead of re-billing the content.
            line = f"- [recorded {timestamp}] canonical copy in open objectives"
        else:
            response = str(entry.get("response_text") or "").strip()
            if redaction is not None:
                query = redaction.redact(query)
                response = redaction.redact(response)
            preview = response[:160] if response else ""
            line = f"- [recorded {timestamp}] {query[:200]}: {preview}".rstrip()
        if budget - len(line) < 0:
            break
        lines.append(line)
        budget -= len(line) + 1
        if len(lines) >= recent_limit:
            break
    if not lines:
        return ""
    return "\n".join(lines)


def _append_open_objectives_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
    redaction: Any,
    objective_lines: int,
    live_state: Any = None,
) -> None:
    lines: list[str] = []
    for pos in range(len(entries) - 1, -1, -1):
        _, entry = entries[pos]
        provenance = entry.get("provenance") or {}
        blocked = provenance.get("block_origin") or entry.get("block_origin")
        pending = provenance.get("pending_approval_request_id") or entry.get("yielded")
        if not (blocked or pending):
            continue
        # F3: a later same-task entry that proceeded without a blocker
        # resolves this objective — drop it instead of re-listing the
        # finished work as open every turn.
        if _resolved_on_evidence(entry, entries[pos + 1 :]):
            continue
        query = str(entry.get("query") or "").strip()
        response = str(entry.get("response_text") or "").strip()
        if redaction is not None:
            query = redaction.redact(query)
            response = redaction.redact(response)
        note = _superseded_note(entry, live_state)
        if blocked:
            lines.append(
                f"- blocked turn ({blocked}) [recorded]: {query[:160]} → {response[:200]}{note}"
            )
        else:
            lines.append(f"- awaiting user [recorded]: {query[:160]} → {response[:200]}{note}")
        if len(lines) >= objective_lines:
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


def _append_epistemics_item(
    items: list[ContextItem],
    entries: list[tuple[int, dict[str, Any]]],
) -> None:
    """Carry the shared past/present directive once per digest assembly.

    F4 (2026-10-14 self-review): the directive repeated in every digest
    body, billing the same framing tokens N times per turn. One item owns
    it; the per-body appends were removed. Emitted only when at least one
    other digest item exists — a lone footer with no data is noise.
    """
    items.append(
        ContextItem(
            item_id="thread:epistemics",
            namespace=_DIGEST_NAMESPACE,
            source=_DIGEST_SOURCE,
            source_type="general",
            evidence_type="epistemics",
            title="Epistemics framing (recorded past, not live present)",
            body=_PAST_PRESENT_DIRECTIVE,
            collected_at=str(entries[-1][1].get("timestamp_utc") or ""),
            salience_score=_SALIENCE_TASK_STATE,
            confidence=_CONFIDENCE,
            tags=["thread_digest", "epistemics"],
            metadata={"provenance": "system_observed", "scope": "recorded_history"},
        )
    )


async def resolve_live_workflow_state(
    runtime: Any,
    *,
    request_ids: list[str],
) -> Any | None:
    """Snapshot the durable approval state for the recorded blockers.

    The digest builders are synchronous, but the approval store is async;
    the host resolves this snapshot before the turn starts (in the
    async pre-turn path) and hands it to ``build_thread_context_items``.
    Reads only the request ids the recorded thread tail actually names so
    a long approval history costs nothing.

    Returns ``None`` when no request ids are named or the store cannot be
    read — the digest then preserves its unannotated behavior exactly.
    """
    if not request_ids:
        return None
    service = getattr(getattr(runtime, "services", None), "approval_service", None)
    if service is None or not hasattr(service, "get"):
        return None
    from types import SimpleNamespace

    resolved: dict[str, str] = {}
    for request_id in request_ids:
        try:
            record = await service.get(request_id)
        except Exception:  # pylint: disable=broad-except
            logger.debug("live workflow state: get(%s) failed", request_id, exc_info=True)
            continue
        if record is not None:
            resolved[str(request_id)] = str(record.state or "")
    if not resolved:
        return None
    return SimpleNamespace(resolved_approvals=resolved)
