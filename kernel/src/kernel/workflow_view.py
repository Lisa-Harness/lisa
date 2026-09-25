"""Derived workflow view from the lossless thread store.

The thread is the single source of truth for workflow state. This module
derives a ``WorkflowView`` — a frozen snapshot of the active workflow —
from the thread tail at turn start. The view is never persisted; it is
rebuilt every turn from the append-only thread log.

The plan lives as a ``## Plan`` block in a turn's ``response_text``.
Completion is inferred from ``effect_receipts`` in later thread entries
matching each step's ``write_capability`` (for write steps) or the first
handoff (for read steps). When a match is found, the completing entry's
global turn index is recorded on the step as ``completing_turn`` — the
completion is turn-addressed and verifiable via ``thread_read_range`` or
the markdown render operation. The string match is the *discovery*; the
addressed turn is the *conclusion*. Mode/stage pointers are derived from
the most recent thread entry's ``mode_id`` and ``transitioned_to``.

This replaces the mutable ``WorkflowPosition`` + ``FileWorkflowPositionStore``
pair that drifted from the thread. One store, append-only, lossless.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kernel.workflow_position import WorkflowStep

__all__ = [
    "WorkflowView",
    "derive_workflow_view",
    "is_complete",
    "remaining_steps",
    "render_workflow_progress",
]


# ── Plan block parser ────────────────────────────────────────────────

_PLAN_HEADER_RE = re.compile(r"^##\s+Plan\s*$", re.MULTILINE)
# Matches:  1. **search** — Search for the latest rust async runtime news
# Captures: mode_id="search", description="Search for the latest..."
_PLAN_STEP_RE = re.compile(r"^\s*(?:\d+\.|[-*])\s+\*{0,2}([a-zA-Z0-9_-]+)\*{0,2}\s*[—–-]\s*(.+)$")
# Optional rationale line:  _Provides web_search_
_RATIONALE_RE = re.compile(r"^\s*_\s*(.+?)\s*_\s*$")
# Optional deliverable line:  _writes: markdown_forge_document_  or  _reads_
_WRITES_RE = re.compile(r"^\s*_\s*writes:\s*([a-zA-Z0-9_.-]+)\s*_\s*$")
_READS_RE = re.compile(r"^\s*_\s*reads_\s*$")


def _parse_step_deliverable_lines(
    lines: list[str],
    step_index: int,
) -> tuple[str, str | None, int]:
    """Consume the optional rationale / deliverable lines following one plan step.

    Both the rationale and the ``_writes_`` / ``_reads_`` line use the ``_..._``
    wrapper; they are distinguished by content, and the deliverable line takes
    precedence for capability extraction. Returns ``(rationale, write_capability,
    next_index)`` where ``next_index`` is the line index after the consumed
    lookahead lines.
    """
    rationale = ""
    write_capability: str | None = None
    i = step_index
    if i + 1 >= len(lines):
        return rationale, write_capability, i

    writes_match = _WRITES_RE.match(lines[i + 1])
    reads_match = _READS_RE.match(lines[i + 1])
    if writes_match:
        write_capability = writes_match.group(1).strip()
        i += 1
        if i + 1 < len(lines):
            rat_match = _RATIONALE_RE.match(lines[i + 1])
            if rat_match:
                rationale = rat_match.group(1).strip()
                i += 1
    elif reads_match:
        i += 1
        if i + 1 < len(lines):
            rat_match = _RATIONALE_RE.match(lines[i + 1])
            if rat_match:
                rationale = rat_match.group(1).strip()
                i += 1
    else:
        rat_match = _RATIONALE_RE.match(lines[i + 1])
        if rat_match:
            rationale = rat_match.group(1).strip()
            i += 1
    return rationale, write_capability, i


def _parse_plan_block(text: str) -> tuple[str, list[tuple[str, str, str, str | None]]] | None:
    """Extract the ``## Plan`` block from ``text``.

    Returns ``(plan_text, steps)`` where ``steps`` is a list of
    ``(mode_id, description, rationale, write_capability)`` tuples.
    ``write_capability`` is the capability id parsed from a ``_writes: <id>_``
    line, ``None`` for read steps (``_reads_`` or no deliverable line), or
    ``None`` when no ``## Plan`` header is found.

    The parser is best-effort: it handles the standard format produced by
    ``_render_plan_markdown`` and common prose variants. Steps with
    unparseable mode_ids are skipped (not silently accepted with empty
    mode_id — the FSM theme: if preconditions are not met, stop).
    """
    match = _PLAN_HEADER_RE.search(text)
    if not match:
        return None

    # Extract from the header to the next ## heading or end of text.
    block_start = match.start()
    after = text[match.end() :]
    next_header = re.search(r"^##\s+", after, re.MULTILINE)
    if next_header:
        block_text = text[block_start : match.end() + next_header.start()]
    else:
        block_text = text[block_start:]

    lines = block_text.splitlines()
    steps: list[tuple[str, str, str, str | None]] = []
    i = 1  # skip the header line
    while i < len(lines):
        step_match = _PLAN_STEP_RE.match(lines[i])
        if step_match:
            mode_id = step_match.group(1).strip()
            description = step_match.group(2).strip()
            # Check next lines for rationale and/or deliverable line.
            rationale, write_capability, i = _parse_step_deliverable_lines(lines, i)
            # Skip steps with empty mode_id or description (hollow steps)
            if mode_id and description:
                steps.append((mode_id, description, rationale, write_capability))
        i += 1

    if not steps:
        # A ## Plan header with no parseable steps is not a plan.
        return None

    return block_text.strip(), steps


# ── WorkflowView model ────────────────────────────────────────────────


class WorkflowView(BaseModel):
    """A derived, immutable snapshot of the active workflow.

    Drop-in compatible with ``WorkflowPosition``: same field names,
    same ``step_by_id`` method. Built from the thread tail by
    ``derive_workflow_view`` — never persisted, never mutated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    # ── Identity & provenance ────────────────────────────────────────
    position_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    plan_digest: str = ""
    spawned_from: str = "thread_derived"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_progressed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── Inter-mode plan ───────────────────────────────────────────────
    declared_steps: tuple[WorkflowStep, ...] = ()
    completed_steps: tuple[str, ...] = ()
    current_step_id: str = ""
    plan_text: str = ""
    step_results: dict[str, str] = Field(default_factory=dict)

    # ── Intra-mode FSM pointer ────────────────────────────────────────
    mode_id: str = ""
    stage_id: str = ""
    turn_count: int = 0
    terminal_reached: bool = False

    def step_by_id(self, step_id: str) -> WorkflowStep | None:
        """Return the declared step with the given id, or ``None``."""
        for step in self.declared_steps:
            if step.step_id == step_id:
                return step
        return None


# ── Derivation ────────────────────────────────────────────────────────


def _step_id_for(mode_id: str, index: int) -> str:
    """Build a stable step id from a mode id and its position in the chain."""
    slug = "".join(c if c.isalnum() else "_" for c in mode_id.lower())
    return f"{slug}_{index}"


def _has_write_receipt(entry: dict[str, Any]) -> bool:
    """Check if a thread entry has at least one write/external_write receipt."""
    receipts = entry.get("effect_receipts")
    if not receipts or not isinstance(receipts, list):
        return False
    return any(
        str(r.get("effect", "")) in ("write", "external_write")
        for r in receipts
        if isinstance(r, dict)
    )


def _receipt_capability_ids(entry: dict[str, Any]) -> set[str]:
    """Return the set of capability ids (raw + bare form) from write receipts."""
    receipts = entry.get("effect_receipts")
    if not receipts or not isinstance(receipts, list):
        return set()
    ids: set[str] = set()
    for r in receipts:
        if not isinstance(r, dict):
            continue
        if str(r.get("effect", "")) not in ("write", "external_write"):
            continue
        cap = str(r.get("capability_id") or "")
        if not cap:
            continue
        ids.add(cap)
        # Bare form: strip a ``provider.`` namespace prefix (e.g.
        # ``lisa-storage.read_file`` -> ``read_file``).
        if "." in cap:
            ids.add(cap.rsplit(".", 1)[-1])
    return ids


def _step_satisfied(
    entries: list[dict[str, Any]],
    step: WorkflowStep,
    after_index: int,
) -> dict[str, Any] | None:
    """Return the thread entry that satisfies ``step``, or ``None``.

    - **Write step** (``step.write_capability`` set): the first later
      entry with ``mode_id == step.mode_id`` whose write receipts include
      ``step.write_capability`` (raw or bare form). No dense-content
      fallback — the receipt is the honest signal.
    - **Read step** (``step.write_capability`` is None): the first later
      entry with ``mode_id == step.mode_id`` and a non-empty
      ``handoff_to`` or ``transitioned_to``. The reads and prose are
      already in the lossless thread; the handoff is the completion
      signal.
    """
    target = step.write_capability
    for entry in entries[after_index + 1 :]:
        if entry.get("mode_id") != step.mode_id:
            continue
        if target is not None:
            if target in _receipt_capability_ids(entry):
                return entry
        else:
            handoff = str(entry.get("handoff_to") or "")
            transitioned = str(entry.get("transitioned_to") or "")
            if handoff or transitioned:
                return entry
    return None


def _receipts_for_mode(entries: list[dict[str, Any]], mode_id: str, after_index: int) -> bool:
    """Check if any entry after ``after_index`` has a write receipt while
    running in ``mode_id``."""
    for entry in entries[after_index + 1 :]:
        if entry.get("mode_id") == mode_id and _has_write_receipt(entry):
            return True
    return False


def _latest_plan_entry(
    thread_entries: list[dict[str, Any]],
) -> tuple[int, str, list[tuple[str, str, str, str | None]]] | None:
    """Find the most recent ``## Plan`` block, scanning backwards.

    Returns ``(entry_index, plan_text, parsed_steps)``, or None when no
    entry carries a plan block (no active workflow).
    """
    for i in range(len(thread_entries) - 1, -1, -1):
        entry = thread_entries[i]
        response = str(entry.get("response_text") or "")
        result = _parse_plan_block(response)
        if result is not None:
            plan_text, parsed_steps = result
            return i, plan_text, parsed_steps
    return None


def _build_view_steps(
    parsed_steps: list[tuple[str, str, str, str | None]],
) -> list[WorkflowStep]:
    """Build the declared WorkflowStep chain from one parsed plan."""
    steps: list[WorkflowStep] = []
    for idx, (mode_id, description, rationale, write_capability) in enumerate(parsed_steps):
        step_id = _step_id_for(mode_id, idx)
        steps.append(
            WorkflowStep(
                step_id=step_id,
                mode_id=mode_id,
                step_query=description,
                plan_text=rationale,
                is_terminal=(idx == len(parsed_steps) - 1),
                spawned_from="thread_derived",
                write_capability=write_capability,
            )
        )
    return steps


def _match_step_completions(
    steps: list[WorkflowStep],
    thread_entries: list[dict[str, Any]],
    plan_entry_index: int,
) -> tuple[list[str], dict[str, str], list[WorkflowStep]]:
    """Determine step completion via the deliverable contract.

    - Write step (write_capability set): complete iff a later thread entry
      has a receipt whose capability_id matches write_capability (raw or
      bare form). No dense-content fallback — the receipt is the honest
      signal. The matching entry's global turn index is recorded on the
      step as ``completing_turn`` so the completion is turn-addressed and
      verifiable via ``thread_read_range``.
    - Read step (write_capability None): complete on the first handoff_to or
      transitioned_to in a later entry with matching mode_id. The reads
      and prose are already in the lossless thread; the handoff is the
      completion signal.

    Returns ``(completed_ids, step_results, updated_steps)`` where the
    updated steps carry ``completing_turn``.
    """
    completed: list[str] = []
    step_results: dict[str, str] = {}
    updated_steps = list(steps)

    for idx, step in enumerate(updated_steps):
        completing_entry = _step_satisfied(thread_entries, step, plan_entry_index + idx)
        if completing_entry is None:
            continue
        completed.append(step.step_id)
        preview = str(completing_entry.get("response_text") or "").strip()[:300]
        if preview:
            step_results[step.step_id] = preview
        # Record the completing turn index on the step so the
        # completion is turn-addressed. The entry's ``turn`` field is
        # the global 0-based turn index stamped at append time.
        completing_turn = completing_entry.get("turn")
        if isinstance(completing_turn, int):
            updated_steps[idx] = step.model_copy(update={"completing_turn": completing_turn})

    return completed, step_results, updated_steps


def _count_consecutive_same_mode_turns(
    thread_entries: list[dict[str, Any]],
    current_mode: str,
    current_stage: str,
) -> int:
    """Count consecutive tail entries sharing the same mode_id + stage_id."""
    turn_count = 1
    for j in range(len(thread_entries) - 2, -1, -1):
        entry = thread_entries[j]
        if (
            entry.get("mode_id") == current_mode
            and str(entry.get("transitioned_to") or "") == current_stage
        ):
            turn_count += 1
        else:
            break
    return turn_count


def derive_workflow_view(
    thread_entries: list[dict[str, Any]],
    catalog: dict[str, Any] | None = None,
) -> WorkflowView | None:
    """Derive the active workflow view from the thread tail.

    Scans the thread entries (chronological order) for the most recent
    ``## Plan`` block, parses its steps, and determines completion by
    matching ``effect_receipts`` in later entries against each step's
    ``mode_id``.

    Returns ``None`` when:
    - No ``## Plan`` block is found in any entry (no active workflow).
    - The plan is found but all steps are complete (the workflow is done;
      the view should not be injected — "if I finished a plan then quit,
      there shouldn't be a plan").

    The ``catalog`` parameter is optional and currently unused — it's
    reserved for future enrichment (e.g. resolving ``satisfies_intents``
    from the mode catalog).
    """
    if not thread_entries:
        return None

    # Scan from most recent backwards for the latest ## Plan block.
    latest_plan = _latest_plan_entry(thread_entries)
    if latest_plan is None:
        # No plan block found — no active workflow.
        return None
    plan_entry_index, plan_text, parsed_steps = latest_plan

    # Build WorkflowStep objects from the parsed plan.
    steps = _build_view_steps(parsed_steps)

    completed, step_results, steps = _match_step_completions(
        steps, thread_entries, plan_entry_index
    )

    # Derive mode/stage pointer from the most recent entry.
    last_entry = thread_entries[-1]
    current_mode = str(last_entry.get("mode_id") or "")
    current_stage = str(last_entry.get("transitioned_to") or "")

    # Derive turn_count: consecutive entries with same mode_id + stage_id.
    turn_count = _count_consecutive_same_mode_turns(thread_entries, current_mode, current_stage)

    # Determine current step: first non-completed step.
    current_step_id = ""
    remaining = [s for s in steps if s.step_id not in completed]
    if not remaining:
        # All steps complete — the workflow is done. Return None so the
        # caller doesn't inject a stale plan ("if I finished a plan then
        # quit, there shouldn't be a plan").
        return None
    current_step_id = remaining[0].step_id

    # Deterministic position_id from plan_text hash (stable across turns
    # for the same plan, so recovery record correlation works).
    plan_digest = hashlib.sha256(plan_text.encode()).hexdigest()[:16]
    position_id = hashlib.sha256(
        f"{plan_digest}:{current_mode}:{current_stage}".encode()
    ).hexdigest()[:32]

    now = datetime.now(UTC)
    terminal = len(completed) == len(steps)

    return WorkflowView(
        position_id=position_id,
        plan_digest=plan_digest,
        spawned_from="thread_derived",
        created_at=now,
        last_progressed_at=now,
        declared_steps=tuple(steps),
        completed_steps=tuple(completed),
        current_step_id=current_step_id,
        plan_text=plan_text,
        step_results=step_results,
        mode_id=current_mode,
        stage_id=current_stage,
        turn_count=turn_count,
        terminal_reached=terminal,
    )


# ── Helpers (moved from workflow_position.py, adapted for WorkflowView) ──


def remaining_steps(view: WorkflowView) -> tuple[WorkflowStep, ...]:
    """Return the declared steps not yet completed, in declared order."""
    completed = set(view.completed_steps)
    return tuple(s for s in view.declared_steps if s.step_id not in completed)


def is_complete(view: WorkflowView) -> bool:
    """Return True when the view has no remaining work.

    Complete means: every declared step is completed AND the intra-mode
    pointer is either absent or parked on a terminal stage.
    """
    steps_open = bool(view.declared_steps) and len(view.completed_steps) < len(view.declared_steps)
    stage_open = bool(view.stage_id) and not view.terminal_reached
    return not steps_open and not stage_open


def render_workflow_progress(view: WorkflowView) -> str:
    """Render a human-readable summary of completed and remaining steps.

    Returns an empty string when the view has no declared steps or is
    complete (no progress to report). Otherwise emits a checklist of
    all declared steps (with a 300-char result preview and the
    completing turn index for completed steps), followed by a
    ``## Current step`` block giving the current step's full description
    and rationale so the model can orient narratively — not just
    structurally.
    """
    if not view.declared_steps:
        return ""
    if is_complete(view):
        return ""
    completed = set(view.completed_steps)
    lines: list[str] = ["## Workflow progress"]
    for step in view.declared_steps:
        mark = "[x]" if step.step_id in completed else "[ ]"
        label = f"{mark} {step.mode_id}"
        preview = view.step_results.get(step.step_id, "")
        if preview:
            label += f": {preview[:300]}"
        if step.step_id in completed and step.completing_turn is not None:
            label += f" ← turn {step.completing_turn}"
        lines.append(label)
    # ── Current step orientation block ───────────────────────────────
    # Give the model the full description and rationale for the step it
    # is about to run (or is running), not just a checkbox label. This
    # is the narrative orientation that lets the model "pick up where we
    # left off" rather than just seeing a structural checklist.
    current_step = view.step_by_id(view.current_step_id) if view.current_step_id else None
    if current_step is not None:
        lines.append("")
        lines.append("## Current step")
        lines.append("")
        lines.append(f"**{current_step.mode_id}**: {current_step.step_query}")
        if current_step.plan_text:
            lines.append("")
            lines.append(current_step.plan_text)
    return "\n".join(lines)
