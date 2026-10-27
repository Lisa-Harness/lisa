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
    CostRecord,
    DecisionRecord,
    DispatchRecord,
    ForensicBundle,
    TurnEvidence,
)

__all__ = [
    "PREVIEW_CHARS",
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
        timestamp_utc=str(entry.get("timestamp_utc") or ""),
        mode_id=str(entry.get("mode_id") or ""),
        query_preview=_preview(entry.get("query")),
        response_preview=_preview(entry.get("response_text")),
        handoff_to=str(entry.get("handoff_to") or ""),
        transitioned_to=str(entry.get("transitioned_to") or ""),
        blocked=bool(entry.get("blocked") or provenance.get("blocked")),
        block_origin=str(entry.get("block_origin") or provenance.get("block_origin") or ""),
        content_source=str(entry.get("content_source") or ""),
        provenance_source=str(provenance.get("source") or ""),
        task_id=str(entry.get("task_id") or provenance.get("task_id") or ""),
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
    )


def _ts_text(record: dict[str, Any]) -> str:
    """Return a record's ISO timestamp for display, or ``""``."""
    parsed = record_ts(record)
    return parsed.isoformat() if parsed is not None else ""


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
) -> ForensicBundle:
    """Join raw telemetry records onto ``turns`` and return the bundle.

    ``turns`` is consumed for its indices and returned with evidence attached
    (model copies — the inputs are not mutated).
    """
    wanted = {turn.turn_index for turn in turns}
    decisions_by_turn: dict[int, list[DecisionRecord]] = defaultdict(list)
    dispatches_by_turn: dict[int, list[DispatchRecord]] = defaultdict(list)
    costs_by_turn: dict[int, list[CostRecord]] = defaultdict(list)
    stray_decisions: list[DecisionRecord] = []
    stray_dispatches: list[DispatchRecord] = []

    for raw in decisions:
        record = decision_from_record(raw)
        if record.turn_index is not None and record.turn_index in wanted:
            decisions_by_turn[record.turn_index].append(record)
        else:
            stray_decisions.append(record)

    for raw in dispatches:
        record = dispatch_from_record(raw)
        if record.turn_index is not None and record.turn_index in wanted:
            dispatches_by_turn[record.turn_index].append(record)
        else:
            stray_dispatches.append(record)

    for raw in costs:
        record = cost_from_record(raw)
        # Costs outside the requested range are simply not part of the answer;
        # unlike decisions/dispatches they are not a finding, because the
        # caller asked about specific turns and every model call in the window
        # is already attributed to its own turn.
        if record.turn_index is not None and record.turn_index in wanted:
            costs_by_turn[record.turn_index].append(record)

    joined = [
        turn.model_copy(
            update={
                "decisions": decisions_by_turn.get(turn.turn_index, []),
                "dispatches": dispatches_by_turn.get(turn.turn_index, []),
                "costs": costs_by_turn.get(turn.turn_index, []),
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
        log_errors=log_errors,
        notes=notes,
    )
