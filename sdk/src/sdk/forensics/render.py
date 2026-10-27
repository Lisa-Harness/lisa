"""Render a :class:`~sdk.forensics._models.ForensicBundle` as Markdown.

This is the **only** formatter. Both consumers — the pack operation the model
calls, and the operator-facing ``/forensics`` host command — render through
here. Two formatters is how the introspection path ended up reading
``message`` while the writer emitted ``msg``: the same data described twice
drifts, and the drift is invisible until someone compares them.

Design rules, in order:

1. **Every claim names its turn.** A line that says "blocked by task_completion"
   is followed by ``(turn 1057)`` so a reader can check it with
   ``thread_read_range`` instead of trusting the summary.
2. **Absence is stated, not implied.** A missing file, an empty join, and a
   zeroed window are three different answers; the renderer prints which one
   applies rather than emitting a tidy table that hides it.
3. **No model call, no inference.** The renderer formats; it never decides
   what the evidence means beyond arithmetic the reader could redo.
4. **Bounded output.** A bundle is a summary with pointers, not a second copy
   of the thread.
"""

from __future__ import annotations

from sdk.forensics._models import DecisionRecord, ForensicBundle, TurnEvidence

__all__ = ["render_bundle", "render_turn_line"]

#: Turns rendered in full before the remainder collapse to one-line summaries.
#: Keeps a wide range readable; the count of elided turns is always stated.
DETAILED_TURN_LIMIT = 8
#: Decision rows shown from each end of a turn's iteration list. A 40-iteration
#: turn is mostly repetition; the framing and the ending carry the signal.
_MAX_DECISION_ROWS = 6

_MAX_REASON_CHARS = 160
_MAX_ERROR_CHARS = 200


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
    lines.append(f"- runtime log: {'read' if sources.runtime_log else 'unavailable'}")
    if sources.missing:
        lines.append(f"- **missing:** {', '.join(sources.missing)}")
        lines.append(
            "\nMissing sources mean the corresponding evidence is unknown, "
            "not absent. Do not read a blank section as a clean result."
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
    for label, value in (
        ("task", turn.task_id),
        ("provenance", turn.provenance_source),
        ("query", _clip(turn.query_preview, 200) if turn.query_preview else ""),
        ("response", _clip(turn.response_preview, 200) if turn.response_preview else ""),
    ):
        if value:
            lines.append(f"- {label}: {value}")
    routing = _render_routing(turn)
    if routing:
        lines.append(routing)
    if turn.blocked:
        lines.append(
            f"- **blocked:** origin={turn.block_origin or 'unknown'} "
            "— the turn ended without satisfying its contract"
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
        lines.append(
            f"  - {marker} {dispatch.target_id or 'unknown'} "
            f"({dispatch.target_type or '?'}, {dispatch.status or '?'}, "
            f"{dispatch.duration_ms} ms){error}"
        )
    return lines


def _render_cost(turn: TurnEvidence) -> list[str]:
    """Render cost, stating explicitly when no record joined."""
    if not turn.costs:
        return [
            "- cost: no token record joined this turn "
            "(the cost log is missing, offline, or predates the turn key)"
        ]
    top = max(turn.costs, key=lambda record: record.total_tokens)
    return [
        f"- cost: {turn.total_tokens} tokens, ${turn.total_cost_usd:.4f} "
        f"over {len(turn.costs)} call(s); heaviest model: {top.model_id or 'unknown'}"
    ]


def _render_turn_detail(turn: TurnEvidence) -> list[str]:
    """Render one turn in full."""
    lines = _render_turn_header(turn)
    lines.extend(_render_decisions(turn))
    lines.extend(_render_dispatches(turn))
    lines.extend(_render_cost(turn))
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
    if not (bundle.unattributed_decisions or bundle.unattributed_dispatches):
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
