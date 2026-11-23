"""Correlate a thread turn with the telemetry that belongs to it.

The join key is ``turn_index``: the global thread ordinal computed once at turn
start, stamped on the thread entry, and carried into every decision, dispatch,
and token record. This module is the only place that decides how those records
find their turn, so the rule cannot drift between callers.

Attribution rules, in order of importance:

1. **A record joins only on an exact index match.** No "nearest turn"
   fallback: a dispatch 200 ms before turn N+1 started is not evidence about
   turn N, and guessing would fabricate a timeline.
2. **``None`` never joins.** Off-turn slice work has no turn, and treating
   ``None`` as ``0`` would attribute background processing to the first turn.
3. **Unmatched records are surfaced, not dropped.** A decision or dispatch
   that names an index outside the requested range is a finding in its own
   right (the caller asked about turns 3-7 and there is activity at turn 12
   that did not attach to anything).

Everything here is a pure function over already-parsed records: no I/O, so it
is trivially testable and cannot introduce a second read path.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sdk.forensics._logrec import record_ts
from sdk.forensics._models import (
    ApprovalEvidence,
    CallCost,
    CostRecord,
    DecisionRecord,
    DelegationNode,
    DispatchRecord,
    ForensicBundle,
    RunManifestChange,
    TurnEvidence,
)

__all__ = [
    "PREVIEW_CHARS",
    "approval_from_record",
    "build_bundle",
    "cost_from_record",
    "decision_from_record",
    "dispatch_from_record",
    "turn_from_entry",
]

#: Preview width for query/response in a forensic read. Long enough to
#: recognize the turn, short enough that a multi-turn bundle stays a summary
#: rather than a second copy of the thread. Callers expand via thread reads.
PREVIEW_CHARS = 280


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    """Return a bounded single-line preview of a thread field."""
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _int_or_none(value: Any) -> int | None:
    """Return ``value`` as a non-negative int, or ``None`` when unusable.

    Mirrors ``kernel.models.resolve_turn_index`` without importing kernel: the
    rule is "a number or nothing", never "guess".
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _str_or_none_for_attempt(value: Any) -> str | None:
    """Return the attempt vocabulary token as ``str`` (or ``None`` when unknown).

    The writers emit attempt as a string token (``primary", the streaming
    tool-call fallback, the truncation continuation). Legacy rows may carry
    an int (e.g. 0) from the pre-vocabulary era: coerce through ``str()``
    rather than reject, since the reader renders the value verbatim and the
    vocabulary is the writer's contract.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value or None
    try:
        return str(value)
    except (TypeError, ValueError):
        return None


def turn_from_entry(turn: int, entry: dict[str, Any]) -> TurnEvidence:
    """Project one raw thread entry into a :class:`TurnEvidence` shell.

    The evidence records (decisions/dispatches/costs) are attached by
    :func:`build_bundle`; this function reads only the thread's own fields.
    """
    provenance = entry.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    receipts = entry.get("effect_receipts")
    receipt_ids = [
        str(receipt.get("capability_id") or "")
        for receipt in (receipts or [])
        if isinstance(receipt, dict) and receipt.get("capability_id")
    ]
    return TurnEvidence(
        turn_index=int(turn),
        provisional_turn=_int_or_none(entry.get("provisional_turn")),
        timestamp_utc=str(entry.get("timestamp_utc") or ""),
        mode_id=str(entry.get("mode_id") or ""),
        query_preview=_preview(entry.get("query")),
        response_preview=_preview(entry.get("response_text")),
        handoff_to=str(entry.get("handoff_to") or ""),
        transitioned_to=str(entry.get("transitioned_to") or ""),
        blocked=bool(
            entry.get("blocked")
            or provenance.get("blocked")
            or entry.get("block_origin")
            or provenance.get("block_origin")
        ),
        block_origin=str(entry.get("block_origin") or provenance.get("block_origin") or ""),
        content_source=str(entry.get("content_source") or ""),
        provenance_source=str(provenance.get("source") or ""),
        task_id=str(entry.get("task_id") or provenance.get("task_id") or ""),
        # The thread entry is the lineage record: turn_id names the
        # execution, the parent keys name the spawning turn and dispatch.
        # Omitted keys on a top-level entry (and explicit None, which the
        # envelope stamps on rows) both mean unknown here.
        turn_id=str(entry.get("turn_id") or ""),
        parent_turn_id=str(entry.get("parent_turn_id") or ""),
        parent_dispatch_id=str(entry.get("parent_dispatch_id") or ""),
        run_ref=str(entry.get("run_ref") or ""),
        mode_digest=str(entry.get("mode_digest") or ""),
        receipt_capability_ids=receipt_ids,
    )


def decision_from_record(record: dict[str, Any]) -> DecisionRecord:
    """Project one ``turn.decision`` payload."""
    return DecisionRecord(
        turn_index=_int_or_none(record.get("turn_index")),
        iteration=int(record.get("iteration") or 0),
        turns_in_stage=int(record.get("turns_in_stage") or 0),
        mode_id=str(record.get("mode_id") or ""),
        action_type=str(record.get("action_type") or ""),
        action_reason=str(record.get("action_reason") or ""),
        total_tool_calls=int(record.get("total_tool_calls") or 0),
        write_receipts=int(record.get("write_receipts") or 0),
        read_receipts=int(record.get("read_receipts") or 0),
        logged_at=_ts_text(record),
        call_id=str(record.get("call_id") or ""),
        attempt=_str_or_none_for_attempt(record.get("attempt")),
        request_ref=str(record.get("request_ref") or ""),
    )


def dispatch_from_record(record: dict[str, Any]) -> DispatchRecord:
    """Project one ``tool.telemetry`` payload."""
    return DispatchRecord(
        turn_index=_int_or_none(record.get("turn_index")),
        target_id=str(record.get("target_id") or ""),
        target_type=str(record.get("target_type") or ""),
        status=str(record.get("status") or ""),
        duration_ms=int(record.get("duration_ms") or 0),
        error_message=str(record.get("error_message") or ""),
        logged_at=_ts_text(record),
        dispatch_id=str(record.get("dispatch_id") or ""),
        tool_call_id=str(record.get("tool_call_id") or ""),
    )


def cost_from_record(record: dict[str, Any]) -> CostRecord:
    """Project one ``token_usage`` payload."""
    return CostRecord(
        turn_index=_int_or_none(record.get("turn_index")),
        mode_id=str(record.get("mode_id") or ""),
        model_id=str(record.get("model_id") or ""),
        call_site=str(record.get("call_site") or ""),
        prompt_tokens=int(record.get("prompt_tokens") or 0),
        completion_tokens=int(record.get("completion_tokens") or 0),
        total_tokens=int(record.get("total_tokens") or 0),
        cached_tokens=int(record.get("cached_tokens") or 0),
        cost_usd=float(record.get("cost_usd") or 0.0),
        emitted_at=str(record.get("emitted_at") or ""),
        call_id=str(record.get("call_id") or ""),
        attempt=_str_or_none_for_attempt(record.get("attempt")),
        iteration=_int_or_none(record.get("iteration")),
        request_ref=str(record.get("request_ref") or ""),
    )


def approval_from_record(record: dict[str, Any]) -> ApprovalEvidence:
    """Project one approval-ledger transition row."""
    return ApprovalEvidence(
        turn_index=_int_or_none(record.get("turn_index")),
        request_id=str(record.get("request_id") or ""),
        target_id=str(record.get("target_id") or ""),
        target_kind=str(record.get("target_kind") or ""),
        effect=str(record.get("effect") or ""),
        state=str(record.get("state") or ""),
        decision_by=str(record.get("decision_by") or ""),
        decision_source=str(record.get("decision_source") or ""),
        reason=str(record.get("reason") or ""),
        logged_at=_ts_text(record),
    )


def _ts_text(record: dict[str, Any]) -> str:
    """Return a record's ISO timestamp for display, or ``""``."""
    parsed = record_ts(record)
    return parsed.isoformat() if parsed is not None else ""


def _group_call_costs(costs: list[CostRecord]) -> list[CallCost]:
    """Reduce a turn's token rows to one spend per named call.

    Rows without a ``call_id`` never group: they stay in the turn aggregate
    and the renderer keeps the aggregate line for them. Grouping preserves
    row order, so two calls in one iteration read in the order they were
    made.
    """
    grouped: dict[str, CallCost] = {}
    order: list[str] = []
    for record in costs:
        if not record.call_id:
            continue
        existing = grouped.get(record.call_id)
        if existing is None:
            grouped[record.call_id] = CallCost(
                call_id=record.call_id,
                attempt=record.attempt,
                iteration=record.iteration,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                total_tokens=record.total_tokens,
                cached_tokens=record.cached_tokens,
                cost_usd=round(record.cost_usd, 6),
                model=record.model_id,
                request_ref=record.request_ref,
            )
            order.append(record.call_id)
        else:
            merged = CallCost(
                call_id=existing.call_id,
                attempt=existing.attempt if existing.attempt is not None else record.attempt,
                iteration=existing.iteration
                if existing.iteration is not None
                else record.iteration,
                prompt_tokens=existing.prompt_tokens + record.prompt_tokens,
                completion_tokens=existing.completion_tokens + record.completion_tokens,
                total_tokens=existing.total_tokens + record.total_tokens,
                cached_tokens=existing.cached_tokens + record.cached_tokens,
                cost_usd=round(existing.cost_usd + record.cost_usd, 6),
                model=existing.model or record.model_id,
                request_ref=existing.request_ref or record.request_ref,
            )
            grouped[record.call_id] = merged
    return [grouped[call_id] for call_id in order]


def _build_delegation_tree(turns: list[TurnEvidence]) -> tuple[DelegationNode, ...]:
    """Assemble the delegation tree from the turns' lineage keys.

    Every turn is a node; a turn naming a parent turn id attaches under the
    node carrying that id when the parent is inside the bundle, and stays a
    root (edges kept on the node) when it is not. Turns without lineage keys
    stay roots, which keeps the tree honest for rows written before the
    lineage existed.
    """
    nodes: dict[int, DelegationNode] = {}
    parents: dict[int, str] = {}
    for turn in turns:
        nodes[turn.turn_index] = DelegationNode(
            turn_index=turn.turn_index,
            turn_id=turn.turn_id,
            parent_turn_id=turn.parent_turn_id,
            parent_dispatch_id=turn.parent_dispatch_id,
        )
        if turn.parent_turn_id:
            parents[turn.turn_index] = turn.parent_turn_id
    child_of: dict[int, list[int]] = defaultdict(list)
    by_id: dict[str, int] = {turn.turn_id: turn.turn_index for turn in turns if turn.turn_id}
    for child_index, parent_id in parents.items():
        parent_index = by_id.get(parent_id)
        if parent_index is None or parent_index == child_index:
            continue
        child_of[parent_index].append(child_index)
    attached: set[int] = set()
    for child_indices in child_of.values():
        attached.update(child_indices)

    def _finish(node_index: int) -> DelegationNode:
        node = nodes[node_index]
        children = tuple(_finish(child) for child in child_of.get(node_index, []) if child in nodes)
        return DelegationNode(
            turn_index=node.turn_index,
            turn_id=node.turn_id,
            parent_turn_id=node.parent_turn_id,
            parent_dispatch_id=node.parent_dispatch_id,
            children=children,
        )

    ordered_roots = [
        _finish(turn.turn_index)
        for turn in turns
        if turn.turn_index not in attached and turn.turn_index in nodes
    ]
    return tuple(ordered_roots)


def _detect_run_manifest_changes(turns: list[TurnEvidence]) -> tuple[RunManifestChange, ...]:
    """Find consecutive-turn boundaries where the manifest or the resolved
    mode digest differs.

    Turns lacking the keys (legacy rows) participate: an unknown next to a
    known value is exactly the boundary where a pack reload happened (or the
    stamping began), so the change is reported either way.
    """
    changes: list[RunManifestChange] = []
    previous: TurnEvidence | None = None
    for turn in turns:
        if previous is not None and (
            turn.run_ref != previous.run_ref or turn.mode_digest != previous.mode_digest
        ):
            changes.append(
                RunManifestChange(
                    turn_index=turn.turn_index,
                    previous_run_ref=previous.run_ref,
                    previous_mode_digest=previous.mode_digest,
                    current_run_ref=turn.run_ref,
                    current_mode_digest=turn.mode_digest,
                )
            )
        previous = turn
    return tuple(changes)


def build_bundle(
    *,
    generated_at: str,
    turns: list[TurnEvidence],
    decisions: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    costs: list[dict[str, Any]],
    log_errors: list[dict[str, Any]],
    sources: Any,
    notes: list[str],
    approvals: list[dict[str, Any]] | None = None,
) -> ForensicBundle:
    """Join raw telemetry records onto ``turns`` and return the bundle.

    ``turns`` is consumed for its indices and returned with evidence attached
    (model copies — the inputs are not mutated). A turn that committed under
    a different index than it ran under is matched by its
    ``provisional_turn``: the decision/dispatch/cost records were emitted
    mid-turn, before the commit fixed the final index.
    """
    wanted = {turn.telemetry_turn_index for turn in turns}
    decisions_by_turn: dict[int, list[DecisionRecord]] = defaultdict(list)
    dispatches_by_turn: dict[int, list[DispatchRecord]] = defaultdict(list)
    costs_by_turn: dict[int, list[CostRecord]] = defaultdict(list)
    approvals_by_turn: dict[int, list[ApprovalEvidence]] = defaultdict(list)
    stray_decisions: list[DecisionRecord] = []
    stray_dispatches: list[DispatchRecord] = []
    stray_approvals: list[ApprovalEvidence] = []

    for raw in decisions:
        decision = decision_from_record(raw)
        if decision.turn_index is not None and decision.turn_index in wanted:
            decisions_by_turn[decision.turn_index].append(decision)
        else:
            stray_decisions.append(decision)

    for raw in dispatches:
        dispatch = dispatch_from_record(raw)
        if dispatch.turn_index is not None and dispatch.turn_index in wanted:
            dispatches_by_turn[dispatch.turn_index].append(dispatch)
        else:
            stray_dispatches.append(dispatch)

    for raw in costs:
        cost = cost_from_record(raw)
        # Costs outside the requested range are simply not part of the answer;
        # unlike decisions/dispatches they are not a finding, because the
        # caller asked about specific turns and every model call in the window
        # is already attributed to its own turn.
        if cost.turn_index is not None and cost.turn_index in wanted:
            costs_by_turn[cost.turn_index].append(cost)

    for raw in approvals or []:
        approval = approval_from_record(raw)
        if approval.turn_index is not None and approval.turn_index in wanted:
            approvals_by_turn[approval.turn_index].append(approval)
        else:
            stray_approvals.append(approval)

    joined = [
        turn.model_copy(
            update={
                "decisions": decisions_by_turn.get(turn.telemetry_turn_index, []),
                "dispatches": dispatches_by_turn.get(turn.telemetry_turn_index, []),
                "costs": costs_by_turn.get(turn.telemetry_turn_index, []),
                "approvals": approvals_by_turn.get(turn.telemetry_turn_index, []),
                "call_costs": _group_call_costs(costs_by_turn.get(turn.telemetry_turn_index, [])),
            }
        )
        for turn in turns
    ]

    return ForensicBundle(
        generated_at=generated_at,
        turns=joined,
        sources=sources,
        unattributed_decisions=stray_decisions,
        unattributed_dispatches=stray_dispatches,
        unattributed_approvals=stray_approvals,
        log_errors=log_errors,
        notes=notes,
        delegation_tree=_build_delegation_tree(joined),
        run_manifest_changes=_detect_run_manifest_changes(joined),
    )
