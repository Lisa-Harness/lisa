"""The single durable workflow object: inter-mode plan + intra-mode pointer.

The kernel's durability primitives, scoped by boundary:

- ``RecoveryRecord`` (``kernel/recovery.py``) — one work item (tool dispatch).
- ``chain_satisfied`` (``kernel/turn_phases/_turn_state.py``) — one NL turn's
  ephemeral intent-chain accumulator (a projection of the position's
  ``completed_steps``).
- **``WorkflowPosition``** (this module) — the workflow itself: one object
  that survives across modes, stages, turns, and host restarts and that
  routing, recovery, and the iteration budget all reference.

``WorkflowPosition`` merges what used to be two objects that drifted apart:
the inter-mode ``WorkflowInstance`` (declared steps, completed steps) and
the intra-mode ``WorkflowSessionState`` (mode/stage pointer, per-stage turn
count). The split produced the drift class of bugs: a mode handoff advanced
the instance but left the session pointing at the old mode's stage; the
resolver's stall counter was fed from two different units depending on mode
kind. One position object makes those splits unrepresentable.

The thread-derived ``WorkflowView`` (``kernel/workflow_view.py``) is the
host-facing projection: it derives the active workflow from the append-only
thread log at turn start, with no separate mutable position store. The
model and pure helpers are here.

Design constraints (mirroring the codebase conventions):

- Tag-blind and prose-blind: this module names no intent. Steps arrive fully
  formed from the planner operation, which owns the catalog vocabulary.
- Frozen and immutable: every advance helper returns a new position,
  mirroring ``RecoveryRecord`` immutability.
- One origin: a workflow comes from ``PlanWorkflowOutput.steps`` (the
  ``plan_workflow`` operation). The ad-hoc compiler that derived steps from
  routing intents was removed — it could not run in any real install, because
  it required a host-supplied updater (never wired) and the absence of
  ``planner`` from the catalog (always installed).
- Catalog-free by design, so it is importable before the catalog is wired.
  Catalog-aware work lives in ``kernel/ipsar_fsm.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "WorkflowPosition",
    "WorkflowPositionOrigin",
    "WorkflowStep",
    "advance_step",
    "is_complete",
    "next_step_after_pointer",
    "note_turn",
    "remaining_steps",
    "render_workflow_progress",
    "resumed_step",
]


WorkflowPositionOrigin = Literal["adhoc", "session_plan", "plan_workflow_op", "thread_derived"]
"""How a position was spawned.

``adhoc``: retained for compatibility with positions persisted before ad-hoc
compilation was removed. Nothing produces it now — a workflow originates from
a ``## Plan`` block (``plan_workflow_op``). It stays in the literal because
reading an older position must not fail validation.

``session_plan``: the session manifest declared an explicit workflow.

``plan_workflow_op``: the ``plan_workflow`` operation returned a step list
(the model-supplied plan path). The only live origin.

``thread_derived``: the ``WorkflowView`` was derived from the thread tail
by ``kernel.workflow_view.derive_workflow_view`` — the thread-as-store
architecture where no separate mutable position exists.
"""


class WorkflowStep(BaseModel):
    """One declared step in a multi-stage workflow.

    A step is the *inter-mode* unit of progress: one step ≈ one mode
    handoff. The intra-mode stage pointer (``WorkflowPosition.stage_id``)
    is orthogonal — a step may span multiple stages within its mode, but
    the position's step pointer only advances when the model hands off to
    the next step's mode (``transition_stage`` with a new ``mode_id``
    target).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    step_id: str = Field(min_length=1)
    mode_id: str = Field(min_length=1)
    step_query: str = ""
    satisfies_intents: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    is_terminal: bool = False
    spawned_from: WorkflowPositionOrigin = "adhoc"
    # The capability id whose write/external_write effect_receipt completes
    # this step (e.g. "markdown_forge_document"). None for read/gather
    # steps, which complete on the next handoff (``handoff_to`` /
    # ``transitioned_to``) rather than a write receipt. Bound by the
    # planner from the mode's write contract (``mode_completion_effects``)
    # so the view's step-completion check matches the kernel's turn-level
    # write-receipt contract — no false completes from stray receipts, no
    # incomplete-forever read steps.
    write_capability: str | None = None
    # The global turn index whose write receipt completed this step, set
    # by ``WorkflowView`` when it matches ``write_capability`` against a
    # later thread entry's ``effect_receipts``. ``None`` until the match
    # is found. This makes the completion turn-addressed: the receipt
    # string match is the *discovery* mechanism, and this field is the
    # *conclusion* — the step is complete because turn N wrote, and turn
    # N is addressable via ``thread_read_range`` / the markdown render
    # operation for verification.
    completing_turn: int | None = None
    # The full rendered ``## Plan`` block from ``plan_workflow``'s
    # ``output_text``. Carried on the position (not just the turn's
    # ephemeral ``_state.plan_prefix``) so the handoff context builder
    # can inject it into each step's target-mode system prompt. Without
    # this, the plan content dies with the planner turn and the executor
    # sees only a short ``step_query`` (the 2026-08-09 "admitted to not
    # knowing the plan" incident).
    plan_text: str = ""


class WorkflowPosition(BaseModel):
    """The durable, cross-mode, cross-turn execution of one workflow.

    One object carries both axes of workflow state:

    - the **inter-mode plan**: ``declared_steps`` / ``completed_steps`` /
      ``current_step_id`` — which mode handoffs remain. The in-memory
      ``chain_satisfied`` accumulator on ``_TurnState`` is a *projection*
      of ``completed_steps`` — hydrated at turn start and written back on
      each successful handoff.
    - the **intra-mode FSM pointer**: ``mode_id`` / ``stage_id`` /
      ``turn_count`` / ``stage_data`` — where inside the current mode's
      stage graph the workflow sits. ``turn_count`` is the number of turns
      spent in the current stage (the resolver's unified stall counter);
      it resets on every stage or mode advance.

    Identity is stable across turns, modes, and restarts: ``position_id``
    is a uuid hex assigned at creation and never changed. The store keys
    active positions by ``RecoveryIdentity`` (the same identity scope as
    recovery records) so one user's active workflow is resumable per
    surface. ``terminal_reached`` records that the pointer parked on a
    terminal stage; a resume after exit starts a fresh position.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    # ── Identity & provenance ────────────────────────────────────────
    position_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    plan_digest: str = ""
    spawned_from: WorkflowPositionOrigin = "adhoc"
    resumed_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_progressed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # ── Inter-mode plan ──────────────────────────────────────────────
    declared_steps: tuple[WorkflowStep, ...] = ()
    completed_steps: tuple[str, ...] = ()
    current_step_id: str = ""
    # The full rendered ``## Plan`` block from ``plan_workflow``'s
    # ``output_text``. Stored on the position so the handoff context
    # builder can inject it into each step's target-mode system prompt
    # — the executor needs the plan it is executing, not just a mode id.
    plan_text: str = ""
    # Per-step result previews (step_id -> short text). Populated from
    # ``RecoveryRecord.result_preview`` as steps complete; used by the
    # compaction/context layer as the "what we've done so far" summary
    # instead of re-reading the thread tail.
    step_results: dict[str, str] = Field(default_factory=dict)
    # ── Intra-mode FSM pointer (absorbed WorkflowSessionState) ───────
    mode_id: str = ""
    stage_id: str = ""
    entered_at: str = ""
    turn_count: int = 0
    stage_data: dict[str, Any] = Field(default_factory=dict)
    terminal_reached: bool = False

    def step_by_id(self, step_id: str) -> WorkflowStep | None:
        """Return the declared step with the given id, or ``None``."""
        for step in self.declared_steps:
            if step.step_id == step_id:
                return step
        return None


def remaining_steps(position: WorkflowPosition) -> tuple[WorkflowStep, ...]:
    """Return the declared steps not yet completed, in declared order.

    A step is remaining when its ``step_id`` is not in
    ``completed_steps``. ``is_terminal`` does not affect this — a terminal
    step is remaining until it completes.
    """
    completed = set(position.completed_steps)
    return tuple(s for s in position.declared_steps if s.step_id not in completed)


def next_step_after_pointer(position: Any) -> WorkflowStep | None:
    """Return the first uncompleted step strictly after the position's pointer.

    The forward-only routing rule, shared by every routing consumer
    (post-handoff hint, resume-of-completed-pointer, yield fallback,
    per-step budget transition, planned-writer rescue):

    - Pointer **on-plan** (``current_step_id`` names a declared step): the
      first uncompleted step whose declaration index is greater than the
      pointer's. ``None`` when no uncompleted step follows. A step declared
      *before* the pointer is never named: steps can complete out of
      declared order (step 2 runs while step 1 is abandoned), and the
      naive "first remaining" fallback then names the abandoned step
      every turn — the hint and the forced-action enforcer obey it and
      drag the turn back on each handoff (the 2026-10-30 t1107/t1108
      loop: pointer on the tail, every hint said `thinker`, the enforcer
      synthesized the backward transition over the model's refusal twice
      per turn).
    - Pointer **absent** (``current_step_id`` empty): ``None`` — routing
      stays silent. An empty pointer with open steps means those steps
      precede the completion frontier: abandoned, not resumable.
    - Pointer **off-plan** (names no declared step, e.g. a recompile
      dropped the step — positions only; derived views always name a
      declared step or none): the first remaining step — there is no
      in-plan front to be backward from, so the earliest open step is
      the best available guess.

    This rule never *resumes* the pointer step itself — the resume
    consumer (``_active_position_resume_target``) checks the open
    pointer before calling, because "resume the step in flight" and
    "route to the next step" are different intents.

    Duck-typed over any position-like (``declared_steps`` /
    ``current_step_id`` / ``completed_steps``): both ``WorkflowPosition``
    and the thread-derived ``WorkflowView`` carry the three fields.
    """
    steps = tuple(getattr(position, "declared_steps", ()) or ())
    current_id = str(getattr(position, "current_step_id", "") or "")
    completed = set(getattr(position, "completed_steps", ()) or ())
    if not steps or not current_id:
        return None
    current_index = next(
        (idx for idx, step in enumerate(steps) if step.step_id == current_id),
        None,
    )
    if current_index is None:
        return next((step for step in steps if step.step_id not in completed), None)
    return next(
        (step for step in steps[current_index + 1 :] if step.step_id not in completed),
        None,
    )


def resumed_step(position: Any) -> WorkflowStep | None:
    """Return the step a resume should continue: the open pointer, else forward.

    ``next_step_after_pointer`` refuses to name the pointer itself (correct
    for routing), but a resume continues work in flight: when the pointer
    names a declared step that is still open, that step IS the resume
    target (re-entering the chain's originating mode must not re-run the
    planner against a continuation query — the 2026-08-01
    plan→markdown→learn→re-plan failure). A completed pointer falls
    through to the forward rule.
    """
    steps = tuple(getattr(position, "declared_steps", ()) or ())
    current_id = str(getattr(position, "current_step_id", "") or "")
    completed = set(getattr(position, "completed_steps", ()) or ())
    if steps and current_id:
        current = next((step for step in steps if step.step_id == current_id), None)
        if current is not None and current.step_id not in completed:
            return current
    return next_step_after_pointer(position)


def is_complete(position: WorkflowPosition) -> bool:
    """Return True when the position has no remaining work.

    Complete means: every declared step is completed AND the intra-mode
    pointer is either absent (no stage graph relevance) or parked on a
    terminal stage (``terminal_reached``). A position with a live
    non-terminal stage pointer stays active even with no declared steps —
    that is the normal state of a single-mode turn.
    """
    steps_open = bool(position.declared_steps) and len(position.completed_steps) < len(
        position.declared_steps
    )
    stage_open = bool(position.stage_id) and not position.terminal_reached
    return not steps_open and not stage_open


def note_turn(position: WorkflowPosition) -> WorkflowPosition:
    """Return a new position with ``turn_count`` incremented by one.

    Called once per turn at turn start by the turn loop. ``turn_count``
    counts turns spent in the current stage — the lifecycle resolver's
    unified stall counter. It resets on every stage or mode advance
    (``apply_stage_transition`` / ``enter_mode`` in ``kernel/ipsar_fsm.py``).
    """
    return position.model_copy(update={"turn_count": position.turn_count + 1})


def advance_step(
    position: WorkflowPosition,
    step_id: str,
    *,
    result_preview: str = "",
) -> WorkflowPosition:
    """Return a new position with ``step_id`` marked completed and the
    current pointer advanced to the next remaining step.

    Idempotent: re-advancing an already-completed step is a no-op (the
    step set and pointer do not change), so recovery resume can call this
    safely. The pointer advances to the first uncompleted step *after* the
    completed one in declaration order and clears when none follows — it
    never moves backward to an abandoned earlier step, even when steps
    complete out of order (completing the last step clears the pointer;
    ``mode_id`` keeps pointing at the mode the position is parked
    in — mode identity is the stage pointer's concern, not the step
    pointer's — the caller's ``enter_mode`` applies the mode move).
    Completing a step declared ``is_terminal`` (the chain's last step)
    sets ``terminal_reached`` so the host store clears the finished
    position instead of leaving an all-done zombie active.

    ``result_preview`` (truncated to 240 chars, mirroring
    ``RecoveryRecord.result_preview``) is stored under ``step_results`` so
    the context layer can summarize what each step produced without
    re-reading the full turn history.
    """
    step = position.step_by_id(step_id)
    if step is None:
        return position
    completed = list(position.completed_steps)
    if step_id not in completed:
        completed.append(step_id)
    # Advance the pointer to the first remaining step *after* the completed
    # one in declaration order — never backward. Steps can complete out of
    # order (step 2 runs while step 1 is abandoned), and a
    # ``remaining[0]`` fallback would then park the pointer on the
    # abandoned earlier step, making every later hint name it (the
    # 2026-10-30 backward-routing loop). When nothing uncompleted follows,
    # the pointer clears — the same terminal handling as completing the
    # last step.
    steps = list(position.declared_steps)
    done_index = steps.index(step)
    next_step = next(
        (s for s in steps[done_index + 1 :] if s.step_id not in completed),
        None,
    )
    step_results = dict(position.step_results)
    preview = (result_preview or "").strip()
    if preview:
        step_results[step_id] = preview[:240]
    now = datetime.now(UTC)
    return position.model_copy(
        update={
            "completed_steps": tuple(completed),
            "current_step_id": next_step.step_id if next_step else "",
            "last_progressed_at": now,
            "step_results": step_results,
            "terminal_reached": position.terminal_reached or step.is_terminal,
        }
    )


def _step_id_for(mode_id: str, index: int) -> str:
    """Build a stable step id from a mode id and its position in the chain.

    Step ids are deterministic so a recompiled plan (e.g. after a session
    plan edit) preserves ``completed_steps`` by id intersection. The mode
    id is slugified (lowercased, non-alphanumerics → ``_``) so it is safe
    as a dict key and stable across recompiles.
    """
    slug = "".join(c if c.isalnum() else "_" for c in mode_id.lower())
    return f"{slug}_{index}"


def render_workflow_progress(position: WorkflowPosition) -> str:
    """Render a short human-readable summary of completed and remaining steps.

    Used by the host's context assembler as the "work done so far"
    summary — replacing re-reading the thread tail (which
    is host-owned UI/replay state the kernel never reads for routing). The
    summary lists each declared step with a checkmark (done) or a dash
    (remaining), plus the result preview when available.

    Returns an empty string when the position has no declared steps or is
    complete (no progress to report).
    """
    if not position.declared_steps:
        return ""
    if is_complete(position):
        return ""
    completed = set(position.completed_steps)
    lines: list[str] = ["## Workflow progress"]
    for step in position.declared_steps:
        mark = "[x]" if step.step_id in completed else "[ ]"
        label = f"{mark} {step.mode_id}"
        preview = position.step_results.get(step.step_id, "")
        if preview:
            # Truncate the preview to keep the summary compact.
            label += f": {preview[:120]}"
        lines.append(label)
    return "\n".join(lines)
