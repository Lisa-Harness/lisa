"""Render a :class:`~sdk.forensics._models.ForensicBundle` as Markdown.

This is the **only** formatter. Both consumers — the pack operation the model
calls, and the operator-facing ``/forensics`` host command — render through
here. Two formatters is how the introspection path ended up reading
``message`` while the writer emitted ``msg``: the same data described twice
drifts, and the drift is invisible until someone compares them.

Design rules, in order:

1. **Turn-specific claims name their turn.** A line that says "blocked by
   task_completion" is followed by ``(turn 1057)`` so a reader can check it
   with ``thread_read_range`` instead of trusting the summary. Totals,
   process logs, and unattributed rows are explicitly labelled as such;
   missing attribution stays missing — the renderer never fabricates a
   turn ID or forces a fake join.
2. **Absence is stated, not implied.** A missing file, an empty join, and a
   zeroed window are three different answers; the renderer prints which one
   applies rather than emitting a tidy table that hides it.
3. **No model call, no inference.** The renderer formats; it never decides
   what the evidence means beyond arithmetic the reader could redo.
4. **Bounded output.** A bundle is a summary with pointers, not a second copy
   of the thread.
"""

from __future__ import annotations

from sdk.forensics._models import (
    ApprovalEvidence,
    CallCost,
    DecisionRecord,
    DelegationNode,
    ForensicBundle,
    TurnEvidence,
)

__all__ = ["render_bundle", "render_turn_line"]

#: Turns rendered in full before the remainder collapse to one-line summaries.
#: Keeps a wide range readable; the count of elided turns is always stated.
DETAILED_TURN_LIMIT = 8
#: Decision rows shown from each end of a turn's iteration list. A 40-iteration
#: turn is mostly repetition; the framing and the ending carry the signal.
_MAX_DECISION_ROWS = 6

_MAX_REASON_CHARS = 160
_MAX_ERROR_CHARS = 200

#: Origin-aware classification for blocked turns. ``block_origin`` names the
#: gate that produced the block; the classification restates it as a cause so
#: guard blocks are not universally described as routing or contract failures.
_BLOCK_ORIGIN_CLASSIFICATIONS = {
    "capability_gate": "the capability gate refused the turn's target",
    "input_guard": "an input guard blocked the request before the model call",
    "output_guard": "an output guard replaced the response with a safe fallback",
    "task_completion": "the turn did not satisfy its task completion contract",
    "no_tools": "the turn made no verified action-tool write required by its completion contract",
}


def _blocked_classification(origin: str) -> str:
    """Classify the cause of a blocked turn from its recorded origin."""
    if origin in _BLOCK_ORIGIN_CLASSIFICATIONS:
        return _BLOCK_ORIGIN_CLASSIFICATIONS[origin]
    if origin:
        return f"a non-guard lifecycle failure named {origin}"
    return "a non-guard lifecycle failure (origin not recorded)"


def _clip(text: str, limit: int) -> str:
    """Return ``text`` bounded to ``limit`` characters."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_turn_line(turn: TurnEvidence) -> str:
    """Render one compact turn summary line (used when eliding detail)."""
    flags: list[str] = []
    if turn.blocked:
        flags.append(f"blocked:{turn.block_origin or 'unknown'}")
    if turn.failed_dispatches:
        flags.append(f"{len(turn.failed_dispatches)} failed dispatch(es)")
    if turn.handoff_to:
        flags.append(f"handoff→{turn.handoff_to}")
    suffix = f" [{', '.join(flags)}]" if flags else ""
    return (
        f"- Turn {turn.turn_index} · {turn.mode_id or 'unknown mode'} · "
        f"{len(turn.dispatches)} call(s) · {turn.total_tokens} tokens{suffix}"
    )


def _render_sources(bundle: ForensicBundle) -> list[str]:
    lines = ["### Sources"]
    sources = bundle.sources
    lines.append(f"- thread store: {'read' if sources.thread_store else 'unavailable'}")
    lines.append(f"- decisions log: {'read' if sources.decisions_log else 'unavailable'}")
    lines.append(f"- token usage log: {'read' if sources.token_usage_log else 'unavailable'}")
    lines.append(f"- approvals log: {'read' if sources.approvals_log else 'unavailable'}")
    lines.append(f"- errors ledger: {'read' if sources.errors_log else 'unavailable'}")
    lines.append(f"- runtime log: {'read' if sources.runtime_log else 'unavailable'}")
    if sources.missing:
        lines.append(f"- **missing:** {', '.join(sources.missing)}")
        lines.append(
            "\nMissing sources mean the corresponding evidence is unknown, "
            "not absent. Do not read a blank section as a clean result."
        )
    if sources.legacy_row_count or sources.unreadable_row_count:
        lines.append(
            f"- schema reads: {sources.legacy_row_count} legacy rows read "
            f"best-effort, {sources.unreadable_row_count} newer-version rows "
            "excluded from the join and counted"
        )
    return lines


def _render_receipts(receipts: list[str]) -> str:
    """Return receipts as ``name×count`` pairs in first-seen order.

    A turn that reads one file forty times produces forty identical ids; the
    count is the information, the repetition is noise.
    """
    counts: dict[str, int] = {}
    for receipt in receipts:
        counts[receipt] = counts.get(receipt, 0) + 1
    return ", ".join(f"{name}×{count}" if count > 1 else name for name, count in counts.items())


def _decision_line(decision: DecisionRecord) -> str:
    """Render one decision row."""
    reason = _clip(decision.action_reason, _MAX_REASON_CHARS)
    detail = f" — {reason}" if reason else ""
    return (
        f"  - iter {decision.iteration}: {decision.action_type or 'unknown'}"
        f"{detail} (writes={decision.write_receipts}, reads={decision.read_receipts})"
    )


def _render_routing(turn: TurnEvidence) -> str:
    """Return the routing line for one turn, or an empty string when unrouted."""
    route: list[str] = []
    if turn.handoff_to:
        route.append(f"handoff→{turn.handoff_to}")
    if turn.transitioned_to:
        route.append(f"transition→{turn.transitioned_to}")
    return f"- routing: {', '.join(route)}" if route else ""


def _render_turn_header(turn: TurnEvidence) -> list[str]:
    """Render the identity/attribution lines of one turn."""
    lines = [
        f"### Turn {turn.turn_index} · {turn.mode_id or 'unknown mode'}",
        f"- recorded: {turn.timestamp_utc or 'unknown'}",
    ]
    if turn.provisional_turn is not None:
        lines.append(f"- ran as turn {turn.provisional_turn} (committed out of start order)")
    for label, value in (
        ("task", turn.task_id),
        ("provenance", turn.provenance_source),
        ("query", _clip(turn.query_preview, 200) if turn.query_preview else ""),
        ("response", _clip(turn.response_preview, 200) if turn.response_preview else ""),
    ):
        if value:
            lines.append(f"- {label}: {value}")
    lines.extend(_render_execution_identity(turn))
    routing = _render_routing(turn)
    if routing:
        lines.append(routing)
    if turn.blocked:
        origin = turn.block_origin or ""
        lines.append(
            f"- **blocked:** origin={origin or 'unknown'} — {_blocked_classification(origin)}"
        )
    if turn.content_source:
        lines.append(
            f"- content_source: {turn.content_source} "
            "(response may be a reasoning tail, not a visible answer)"
        )
    if turn.receipt_capability_ids:
        lines.append(f"- effect receipts: {_render_receipts(turn.receipt_capability_ids)}")
    else:
        lines.append("- effect receipts: none recorded")
    return lines


def _render_execution_identity(turn: TurnEvidence) -> list[str]:
    """Render the execution-identity and lineage lines (Evidence Envelope S3/S4).

    Split out of the turn header so the header stays under its complexity
    budget; the lines name the manifest the turn ran under, the execution
    id, and the delegation edges a child turn records.
    """
    lines: list[str] = []
    if turn.run_ref or turn.mode_digest:
        # The manifest this turn ran under. Named per turn so a pack reload
        # is visible at the turn where it landed.
        lines.append(
            f"- run: {turn.run_ref or 'unknown'} / mode digest: {turn.mode_digest or 'unknown'}"
        )
    if turn.turn_id:
        lines.append(f"- turn_id: {turn.turn_id}")
    if turn.parent_turn_id or turn.parent_dispatch_id:
        lines.append(
            f"- delegations from: turn {turn.parent_turn_id or 'unknown'}"
            f" via dispatch {turn.parent_dispatch_id or 'unknown'}"
        )
    return lines


def _render_decisions(turn: TurnEvidence) -> list[str]:
    """Render the iteration decisions, keeping the framing and the ending."""
    if not turn.decisions:
        return []
    lines = [f"- decisions ({len(turn.decisions)} iteration(s)):"]
    # The opening iterations show how the turn was framed, the closing ones
    # show why it ended. The middle is repetition.
    head = turn.decisions[:_MAX_DECISION_ROWS]
    has_elided = len(turn.decisions) > _MAX_DECISION_ROWS * 2
    tail = turn.decisions[-_MAX_DECISION_ROWS:] if has_elided else []
    lines.extend(_decision_line(decision) for decision in head)
    if has_elided:
        lines.append(f"  - … {len(turn.decisions) - len(head) - len(tail)} elided …")
        lines.extend(_decision_line(decision) for decision in tail)
    else:
        lines.extend(_decision_line(decision) for decision in turn.decisions[_MAX_DECISION_ROWS:])
    return lines


def _render_dispatches(turn: TurnEvidence) -> list[str]:
    """Render the dispatch ledger for one turn."""
    if not turn.dispatches:
        return []
    lines = [f"- dispatches ({len(turn.dispatches)}, {len(turn.failed_dispatches)} failed):"]
    for dispatch in turn.dispatches:
        marker = "✗" if dispatch.failed else "✓"
        error = ""
        if dispatch.error_message:
            error = f" — {_clip(dispatch.error_message, _MAX_ERROR_CHARS)}"
        join = f" -> tool_call {dispatch.tool_call_id}" if dispatch.tool_call_id else ""
        lines.append(
            f"  - {marker} {dispatch.target_id or 'unknown'} "
            f"({dispatch.target_type or '?'}, {dispatch.status or '?'}, "
            f"{dispatch.duration_ms} ms){error}{join}"
        )
    return lines


def _render_cost(turn: TurnEvidence) -> list[str]:
    """Render cost, stating explicitly when no record joined."""
    if not turn.costs:
        return [
            "- cost: no token record joined this turn "
            "(the cost log is missing, offline, or predates the turn key)"
        ]
    lines: list[str] = []
    if turn.call_costs:
        # Per-call spend (Evidence Envelope S5): rows that named their call
        # render per call; rows without a call id stay in the aggregate.
        lines.append("- cost (per call):")
        lines.extend(f"  - {_cost_line(cost)}" for cost in turn.call_costs)
        legacy = [record for record in turn.costs if not record.call_id]
        if legacy:
            agg_total = sum(record.total_tokens for record in legacy)
            agg_cost = round(sum(record.cost_usd for record in legacy), 6)
            lines.append(f"  - legacy rows without a call id: {agg_total} tokens, ${agg_cost:.4f}")
    else:
        top = max(turn.costs, key=lambda record: record.total_tokens)
        lines.append(
            f"- cost: {turn.total_tokens} tokens, ${turn.total_cost_usd:.4f} "
            f"over {len(turn.costs)} call(s); heaviest model: {top.model_id or 'unknown'}"
        )
    return lines


def _cost_line(cost: CallCost) -> str:
    """Render one per-call spend line."""
    attempt = f" attempt {cost.attempt}" if cost.attempt else ""
    iteration = f" iter {cost.iteration}" if cost.iteration is not None else ""
    model = f" model {cost.model}" if cost.model else ""
    return (
        f"call {cost.call_id}{iteration}{attempt}: "
        f"{cost.total_tokens} tokens, ${cost.cost_usd:.4f}{model}"
    )


def _render_approvals(turn: TurnEvidence) -> list[str]:
    """Render the approval decisions that touched this turn.

    Rows are transitions, so one request can appear several times (requested
    → approved → consumed). Grouped by ``request_id`` in first-seen order so
    the operator reads the lifecycle of each decision, not a flat event log.
    """
    if not turn.approvals:
        return []
    by_request: dict[str, list[ApprovalEvidence]] = {}
    for row in turn.approvals:
        by_request.setdefault(row.request_id or row.target_id, []).append(row)
    lines = [f"- approvals ({len(by_request)} request(s)):"]
    for request_id, rows in by_request.items():
        target = rows[0].target_id or "unknown"
        # The latest transition names the current state.
        latest = rows[-1]
        who = latest.decision_by or latest.decision_source or "unknown"
        lines.append(f"  - {target} → {latest.state} (by {who}, request {request_id[:8]})")
    return lines


def _render_turn_detail(turn: TurnEvidence) -> list[str]:
    """Render one turn in full."""
    lines = _render_turn_header(turn)
    lines.extend(_render_decisions(turn))
    lines.extend(_render_dispatches(turn))
    lines.extend(_render_cost(turn))
    lines.extend(_render_approvals(turn))
    return lines


def _render_totals(bundle: ForensicBundle) -> list[str]:
    """Render the aggregate totals for the requested turns."""
    lines = ["### Totals", f"- turns: {len(bundle.turns)}", f"- tokens: {bundle.total_tokens}"]
    blocked = [turn.turn_index for turn in bundle.turns if turn.blocked]
    if blocked:
        lines.append(f"- blocked turns: {', '.join(str(turn) for turn in blocked)}")
    failed = sum(len(turn.failed_dispatches) for turn in bundle.turns)
    if failed:
        lines.append(f"- failed dispatches: {failed}")
    return lines


def _render_unattributed(bundle: ForensicBundle) -> list[str]:
    """Render records that could not join a requested turn."""
    if not (
        bundle.unattributed_decisions
        or bundle.unattributed_dispatches
        or bundle.unattributed_approvals
    ):
        return []
    lines = [
        "### Unattributed records",
        "These records named a turn outside the requested range, so they "
        "could not be joined. They are listed rather than dropped: activity "
        "that attached to no requested turn is itself a finding.",
    ]
    for decision in bundle.unattributed_decisions:
        turn = decision.turn_index if decision.turn_index is not None else "no turn"
        lines.append(
            f"- decision (turn {turn}): {decision.action_type or 'unknown'}"
            f" — {_clip(decision.action_reason, _MAX_REASON_CHARS)}"
        )
    for dispatch in bundle.unattributed_dispatches:
        turn = dispatch.turn_index if dispatch.turn_index is not None else "no turn"
        lines.append(
            f"- dispatch (turn {turn}): {dispatch.target_id or 'unknown'} "
            f"({dispatch.status or '?'})"
        )
    for approval in bundle.unattributed_approvals:
        turn = approval.turn_index if approval.turn_index is not None else "no turn"
        who = approval.decision_by or approval.decision_source or "unknown"
        lines.append(
            f"- approval (turn {turn}): {approval.target_id or 'unknown'} "
            f"→ {approval.state or '?'} (by {who})"
        )
    return lines


def _render_log_errors(bundle: ForensicBundle) -> list[str]:
    """Render warning/error evidence, collapsing repeated messages.

    A startup that logs the same pack warning 33 times is one finding, not
    thirty-three. Grouping by (level, message) keeps the count and the time
    span — which is the actual information — instead of spending the output
    budget on repetition.
    """
    if not bundle.log_errors:
        return []
    grouped: dict[tuple[str, str], list[str]] = {}
    for record in bundle.log_errors:
        level = str(record.get("level") or "?")
        message = str(record.get("msg") or record.get("message") or "")
        grouped.setdefault((level, message), []).append(str(record.get("ts") or ""))

    lines = [
        f"### Log warnings and errors ({len(bundle.log_errors)} line(s), {len(grouped)} distinct)"
    ]
    for (level, message), stamps in grouped.items():
        first = stamps[0] or "unknown"
        count = f" ×{len(stamps)}" if len(stamps) > 1 else ""
        span = f" (last {stamps[-1] or 'unknown'})" if len(stamps) > 1 else ""
        lines.append(f"- {level}{count} @ {first}{span}: {_clip(message, _MAX_ERROR_CHARS)}")
    return lines


def _render_delegation_tree(bundle: ForensicBundle) -> list[str]:
    """Render the delegation tree built from the turns' lineage keys."""
    if not bundle.delegation_tree:
        return []
    lines = ["### Delegation tree"]

    def _node(node: DelegationNode, depth: int) -> None:
        indent = "  " * depth
        tid = node.turn_id or "unknown"
        if node.parent_turn_id or node.parent_dispatch_id:
            lines.append(
                f"{indent}- turn {node.turn_index} (turn_id={tid}) -> "
                f"parent {node.parent_turn_id or 'unknown'} "
                f"via dispatch {node.parent_dispatch_id or 'unknown'}"
            )
        else:
            lines.append(f"{indent}- turn {node.turn_index} (turn_id={tid})")
        for child in node.children:
            _node(child, depth + 1)

    for node in bundle.delegation_tree:
        _node(node, 0)
    return lines


def _render_manifest_changes(bundle: ForensicBundle) -> list[str]:
    """Render consecutive-turn manifest changes (pack reloads)."""
    if not bundle.run_manifest_changes:
        return []
    lines = ["### Run manifest changes"]
    lines.extend(
        f"- at turn {change.turn_index}: "
        f"{change.previous_run_ref or 'unknown'} / "
        f"{change.previous_mode_digest or 'unknown'} -> "
        f"{change.current_run_ref or 'unknown'} / "
        f"{change.current_mode_digest or 'unknown'}"
        for change in bundle.run_manifest_changes
    )
    return lines


def _render_turns_section(bundle: ForensicBundle) -> list[str]:
    """Render the per-turn sections plus totals."""
    if not bundle.turns:
        return [
            "### Turns",
            "No turn entries matched the requested range.",
        ]
    lines = ["### Turns"]
    detailed = bundle.turns[-DETAILED_TURN_LIMIT:]
    elided = bundle.turns[:-DETAILED_TURN_LIMIT]
    if elided:
        lines.append(
            f"{len(elided)} earlier turn(s) summarized; the most recent "
            f"{len(detailed)} are detailed below."
        )
        lines.extend(render_turn_line(turn) for turn in elided)
        lines.append("")
    for turn in detailed:
        lines.extend(_render_turn_detail(turn))
        lines.append("")
    lines.extend(_render_totals(bundle))
    return lines


def render_bundle(bundle: ForensicBundle) -> str:
    """Render ``bundle`` as Markdown for a model or an operator."""
    lines: list[str] = [f"## Turn forensics ({bundle.generated_at})", ""]
    lines.extend(_render_sources(bundle))
    lines.append("")
    lines.extend(_render_turns_section(bundle))
    lines.append("")
    for section in (
        _render_unattributed(bundle),
        _render_log_errors(bundle),
        _render_manifest_changes(bundle),
        _render_delegation_tree(bundle),
    ):
        if section:
            lines.extend(section)
            lines.append("")
    if bundle.notes:
        lines.append("### Read notes")
        lines.extend(f"- {note}" for note in bundle.notes)
        lines.append("")
    lines.append(
        "Recorded evidence, not live state. Verify any claim with "
        "`thread_read_range` on the cited turn before acting on it."
    )
    return "\n".join(lines)
