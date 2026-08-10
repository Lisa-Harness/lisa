"""Derived workflow view from the lossless thread store.

The thread is the single source of truth for workflow state. This module
derives a ``WorkflowView`` — a frozen snapshot of the active workflow —
from the thread tail at turn start. The view is never persisted; it is
rebuilt every turn from the append-only thread log.

The plan lives as a ``## Plan`` block in a turn's ``response_text``.
Completion is inferred from ``effect_receipts`` in later thread entries
matching each step's ``mode_id``. Mode/stage pointers are derived from
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
_PLAN_STEP_RE = re.compile(
    r"^\s*(?:\d+\.|[-*])\s+\*{0,2}([a-zA-Z0-9_-]+)\*{0,2}\s*[—–-]\s*(.+)$"
)
# Optional rationale line:  _Provides web_search_
_RATIONALE_RE = re.compile(r"^\s*_\s*(.+?)\s*_\s*$")


def _parse_plan_block(text: str) -> tuple[str, list[tuple[str, str, str]]] | None:
    """Extract the ``## Plan`` block from ``text``.

    Returns ``(plan_text, steps)`` where ``steps`` is a list of
    ``(mode_id, description, rationale)`` tuples, or ``None`` when no
    ``## Plan`` header is found.

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
    after = text[match.end():]
    next_header = re.search(r"^##\s+", after, re.MULTILINE)
    if next_header:
        block_text = text[block_start : match.end() + next_header.start()]
    else:
        block_text = text[block_start:]

    lines = block_text.splitlines()
    steps: list[tuple[str, str, str]] = []
    i = 1  # skip the header line
    while i < len(lines):
        line = lines[i]
        step_match = _PLAN_STEP_RE.match(line)
        if step_match:
            mode_id = step_match.group(1).strip()
            description = step_match.group(2).strip()
            rationale = ""
            # Check next line for rationale
            if i + 1 < len(lines):
                rat_match = _RATIONALE_RE.match(lines[i + 1])
                if rat_match:
                    rationale = rat_match.group(1).strip()
                    i += 1
            # Skip steps with empty mode_id or description (hollow steps)
            if mode_id and description:
                steps.append((mode_id, description, rationale))
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


def _receipts_for_mode(
    entries: list[dict[str, Any]], mode_id: str, after_index: int
) -> bool:
    """Check if any entry after ``after_index`` has a write receipt while
    running in ``mode_id``."""
    for entry in entries[after_index + 1 :]:
        if entry.get("mode_id") == mode_id and _has_write_receipt(entry):
            return True
    return False


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
    plan_entry_index = -1
    plan_text = ""
    parsed_steps: list[tuple[str, str, str]] = []

    for i in range(len(thread_entries) - 1, -1, -1):
        entry = thread_entries[i]
        response = str(entry.get("response_text") or "")
        result = _parse_plan_block(response)
        if result is not None:
            plan_text, parsed_steps = result
            plan_entry_index = i
            break

    if plan_entry_index == -1:
        # No plan block found — no active workflow.
        return None

    # Build WorkflowStep objects from the parsed plan.
    steps: list[WorkflowStep] = []
    for idx, (mode_id, description, rationale) in enumerate(parsed_steps):
        step_id = _step_id_for(mode_id, idx)
        steps.append(
            WorkflowStep(
                step_id=step_id,
                mode_id=mode_id,
                step_query=description,
                plan_text=rationale,
                is_terminal=(idx == len(parsed_steps) - 1),
                spawned_from="thread_derived",
            )
        )

    # Determine completion: a step is complete when a later thread entry
    # has the same mode_id AND at least one write effect receipt.
    completed: list[str] = []
    step_results: dict[str, str] = {}

    for idx, step in enumerate(steps):
        if _receipts_for_mode(thread_entries, step.mode_id, plan_entry_index + idx):
            completed.append(step.step_id)
            # Use the response_text preview from the completing entry.
            for entry in thread_entries[plan_entry_index + idx + 1 :]:
                if entry.get("mode_id") == step.mode_id and _has_write_receipt(entry):
                    preview = str(entry.get("response_text") or "").strip()[:240]
                    if preview:
                        step_results[step.step_id] = preview
                    break

    # Derive mode/stage pointer from the most recent entry.
    last_entry = thread_entries[-1]
    current_mode = str(last_entry.get("mode_id") or "")
    current_stage = str(last_entry.get("transitioned_to") or "")

    # Derive turn_count: consecutive entries with same mode_id + stage_id.
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

    # Determine current step: first non-completed step.
    current_step_id = ""
    remaining = [s for s in steps if s.step_id not in completed]
    if remaining:
        current_step_id = remaining[0].step_id
    else:
        # All steps complete — the workflow is done. Return None so the
        # caller doesn't inject a stale plan ("if I finished a plan then
        # quit, there shouldn't be a plan").
        return None

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
    steps_open = bool(view.declared_steps) and len(view.completed_steps) < len(
        view.declared_steps
    )
    stage_open = bool(view.stage_id) and not view.terminal_reached
    return not steps_open and not stage_open


def render_workflow_progress(view: WorkflowView) -> str:
    """Render a short human-readable summary of completed and remaining steps.

    Returns an empty string when the view has no declared steps or is
    complete (no progress to report).
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
            label += f": {preview[:120]}"
        lines.append(label)
    return "\n".join(lines)