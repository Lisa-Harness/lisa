"""First-class durable workflow instance for multi-stage workflows.

The kernel already has four durability primitives, each scoped to a
different boundary:

- ``RecoveryRecord`` (``kernel/recovery.py``) — one work item (tool dispatch).
- ``CheckpointBundle`` (``kernel/models.py``) — one turn snapshot.
- ``WorkflowSessionState`` (``kernel/models.py``) — one mode's intra-mode FSM.
- ``chain_satisfied`` (``kernel/turn_phases/_turn_state.py``) — one NL turn's
  ephemeral intent-chain accumulator.

None of them is a *first-class object representing the multi-stage workflow
this turn is executing*. There is no identity that survives across modes,
turns, and host restarts and that routing, checkpoint, recovery, and the
iteration budget all reference. The result (proven by the 2026-07-28
glm-5.2 self-assessment run): a 6-stage chain starves its final step on a
shared iteration budget, a stalled turn can't resume the *plan* (only the
mode), recovery records can't block or skip workflow steps, and the
canonical ``host`` host doesn't even wire the existing FSM/checkpoint
hooks.

This module owns that missing object — **``WorkflowInstance``** — plus the
pure helpers that advance it and the step compiler that derives steps from
routing targets for ad-hoc workflows. The durable store contract lives in
``kernel/workflow_instance_store.py`` (Protocol only); the file-backed
implementation lives in ``host/workflow_instance_store.py``. The
model and helpers are here.

Design constraints (mirroring the codebase conventions):

- Tag-blind and prose-blind: the step compiler reuses the existing routing
  helpers (``_first_target_for_earliest_intent``, ``_query_is_sequential``,
  ``_detect_query_intents``) so it never duplicates intent vocabulary or
  connector lists. It reads opaque pack-authored intent tags and routing
  target ids — it names no intent in code.
- Frozen and immutable: ``advance_step`` returns a new instance, mirroring
  ``RecoveryRecord`` / ``WorkflowSessionState`` immutability.
- Ad-hoc and specified workflows share the shape: ``compile_steps_from_routing_targets``
  produces steps for an ad-hoc instance at turn 1; ``PlanWorkflowOutput.steps``
  produces steps for a specified instance. Both resume/checkpoint/recover
  identically because they share the same ``WorkflowInstance`` model.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kernel.routing_vocabulary import get_consumer_intents, get_producer_intents

__all__ = [
    "WorkflowInstance",
    "WorkflowInstanceOrigin",
    "WorkflowStep",
    "advance_step",
    "compile_steps_from_routing_targets",
    "is_complete",
    "remaining_steps",
    "render_workflow_progress",
]


WorkflowInstanceOrigin = Literal["adhoc", "session_plan", "plan_workflow_op"]
"""How an instance was spawned.

``adhoc``: the host built the instance at turn 1 from the routing hint
when no explicit plan was supplied (the common case for natural-language
multi-step requests).

``session_plan``: the session manifest declared an explicit workflow.

``plan_workflow_op``: the ``plan_workflow`` operation returned a step list
(the model-supplied plan path).
"""


class WorkflowStep(BaseModel):
    """One declared step in a multi-stage workflow.

    A step is the *inter-mode* unit of progress: one step ≈ one mode
    handoff. The intra-mode FSM stage pointer (``WorkflowSessionState``)
    is orthogonal — a step may span multiple stages within its mode, but
    the instance only advances when the model hands off to the next step's
    mode (``transition_stage`` with a new ``mode_id`` target).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    step_id: str = Field(min_length=1)
    mode_id: str = Field(min_length=1)
    satisfies_intents: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    is_terminal: bool = False
    spawned_from: WorkflowInstanceOrigin = "adhoc"


class WorkflowInstance(BaseModel):
    """The durable, cross-mode, cross-turn execution of one workflow.

    ``WorkflowInstance`` is the single durable inter-mode object.
    ``WorkflowSessionState`` (per-mode FSM) stays as the intra-mode stage
    pointer; they cross-reference by ``instance_id`` (added to the session
    in a later phase). The instance survives host restart via
    ``FileWorkflowInstanceStore``; the in-memory ``chain_satisfied``
    accumulator on ``_TurnState`` is a *projection* of ``completed_steps`` —
    hydrated at turn start and written back on each successful handoff.

    Identity is stable across turns, modes, and restarts: ``instance_id``
    is a uuid hex assigned at creation and never changed. The store keys
    active instances by ``RecoveryIdentity`` (the same identity scope as
    recovery records) so one user's active workflow is resumable per surface.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    instance_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    plan_digest: str = ""
    declared_steps: tuple[WorkflowStep, ...] = ()
    completed_steps: tuple[str, ...] = ()
    current_step_id: str = ""
    current_mode_id: str = ""
    resumed_count: int = 0
    spawned_from: WorkflowInstanceOrigin = "adhoc"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_progressed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Per-step result previews (step_id -> short text). Populated from
    # ``RecoveryRecord.result_preview`` as steps complete; used by the
    # compaction/context layer as the "what we've done so far" summary
    # instead of re-reading ``CheckpointBundle.turn_history``.
    step_results: dict[str, str] = Field(default_factory=dict)

    def step_by_id(self, step_id: str) -> WorkflowStep | None:
        """Return the declared step with the given id, or ``None``."""
        for step in self.declared_steps:
            if step.step_id == step_id:
                return step
        return None


def remaining_steps(instance: WorkflowInstance) -> tuple[WorkflowStep, ...]:
    """Return the declared steps not yet completed, in declared order.

    A step is remaining when its ``step_id`` is not in
    ``completed_steps``. ``is_terminal`` does not affect this — a terminal
    step is remaining until it completes.
    """
    completed = set(instance.completed_steps)
    return tuple(s for s in instance.declared_steps if s.step_id not in completed)


def is_complete(instance: WorkflowInstance) -> bool:
    """Return True when every declared step is completed."""
    if not instance.declared_steps:
        return True
    return len(instance.completed_steps) >= len(instance.declared_steps)


def advance_step(
    instance: WorkflowInstance,
    step_id: str,
    *,
    result_preview: str = "",
) -> WorkflowInstance:
    """Return a new instance with ``step_id`` marked completed and the
    current pointer advanced to the next remaining step.

    Idempotent: re-advancing an already-completed step is a no-op (the
    step set and pointer do not change), so recovery resume can call this
    safely. When advancing the last remaining step, ``current_step_id`` is
    cleared (the workflow is complete; ``is_complete`` returns True).

    ``result_preview`` (truncated to 240 chars, mirroring
    ``RecoveryRecord.result_preview``) is stored under ``step_results`` so
    the context layer can summarize what each step produced without
    re-reading the full turn history.
    """
    step = instance.step_by_id(step_id)
    if step is None:
        return instance
    completed = list(instance.completed_steps)
    if step_id not in completed:
        completed.append(step_id)
    remaining = [s for s in instance.declared_steps if s.step_id not in completed]
    next_step = remaining[0] if remaining else None
    step_results = dict(instance.step_results)
    preview = (result_preview or "").strip()
    if preview:
        step_results[step_id] = preview[:240]
    now = datetime.now(UTC)
    return instance.model_copy(
        update={
            "completed_steps": tuple(completed),
            "current_step_id": next_step.step_id if next_step else "",
            "current_mode_id": next_step.mode_id if next_step else "",
            "last_progressed_at": now,
            "step_results": step_results,
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


def compile_steps_from_routing_targets(
    *,
    query: str,
    routing_targets: list[str],
    catalog: dict[str, Any],
    spawned_from: WorkflowInstanceOrigin = "adhoc",
) -> tuple[WorkflowStep, ...]:
    """Derive a producer-then-consumer step chain from the routing catalog.

    This is the ad-hoc path's step-derivation (and the plan-failure
    fallback): when no explicit ``PlanWorkflowOutput.steps`` is supplied,
    the instance is built deterministically from the routing targets that
    cover the query's detected intents. The ordering reuses the existing
    routing helpers so there is one truth for producer→consumer sequencing:

    1. Detect the query's intents (``_detect_query_intents``).
    2. For a sequential query, walk producer intents first then consumer
       intents, picking the earliest-keyword target for each remaining gap
       (``_first_target_for_earliest_intent``). This matches the
       post-handoff hint chain order so the instance and the router agree.
    3. For a non-sequential multi-intent query that forks, flatten the fork
       into one step per fork child (the instance does not model
       parallelism yet; child instances are future work).
    4. Mark the last step terminal.

    Tag-blind: the compiler reads opaque intent tags and never names them.
    The producer/consumer role sets come from
    :func:`kernel.routing_vocabulary.get_producer_intents` /
    :func:`kernel.routing_vocabulary.get_consumer_intents`, injected by
    the host at boot. When no session vocabulary is wired, the historical
    defaults apply (mirroring the routing hint path).
    """
    # Local import to avoid a kernel-boot-time cycle: _routing_hint imports
    # from this module's neighbors but not from here, so the deferred
    # import keeps workflow_instance importable before the catalog is wired.
    from kernel.turn_phases._routing_hint import (  # noqa: PLC0415
        _detect_query_intents,
        _first_target_for_earliest_intent,
        _query_is_sequential,
    )

    if not routing_targets:
        return ()
    detected = _detect_query_intents(query=query, routing_targets=routing_targets)
    if not detected:
        return ()

    producers = get_producer_intents()
    consumers = get_consumer_intents()
    producer_intents = detected & producers
    consumer_intents = detected & consumers
    other_intents = detected - producers - consumers

    # Ordered intent groups: producers first (evidence gathering), then
    # any non-role intents, then consumers (artifact creation). This is the
    # sequential chain order the post-handoff hint recomputation follows;
    # mirroring it here keeps the instance and the router in lockstep.
    ordered_intent_groups: list[set[str]] = []
    if producer_intents:
        ordered_intent_groups.append(producer_intents)
    if other_intents:
        ordered_intent_groups.append(other_intents)
    if consumer_intents:
        ordered_intent_groups.append(consumer_intents)

    sequential = _query_is_sequential(query)
    seen_targets: set[str] = set()
    steps: list[WorkflowStep] = []
    step_index = 0

    for group in ordered_intent_groups:
        if sequential:
            # For a sequential query, emit one step per intent in the group,
            # picking the earliest-keyword target for each intent. A group
            # like {coding, write} (two consumer intents) must produce two
            # steps (coding then markdown), not one — otherwise the second
            # consumer is dropped and never reached (the glm-5.2
            # self-assessment bug). Intents within a group are ordered by
            # their earliest keyword position in the query so the chain
            # follows the user's phrasing.
            intent_order = _intents_by_earliest_keyword(
                query=query,
                routing_targets=routing_targets,
                catalog=catalog,
                gap_intents=group,
            )
            for intent in intent_order:
                target = _first_target_for_earliest_intent(
                    query=query,
                    routing_targets=routing_targets,
                    catalog=catalog,
                    gap_intents={intent},
                )
                if not target or target in seen_targets:
                    continue
                seen_targets.add(target)
                mode_cfg = catalog.get(target)
                satisfies = tuple(getattr(mode_cfg, "satisfies_intents", []) or [])
                steps.append(
                    WorkflowStep(
                        step_id=_step_id_for(target, step_index),
                        mode_id=target,
                        satisfies_intents=satisfies,
                        depends_on=tuple(s.step_id for s in steps),
                        is_terminal=False,
                        spawned_from=spawned_from,
                    )
                )
                step_index += 1
        else:
            # For a non-sequential query, a group may fork across multiple
            # targets — flatten them so the instance tracks each child as
            # its own step (parallelism is not modeled yet).
            targets = _targets_covering_intents(
                routing_targets=routing_targets,
                catalog=catalog,
                gap_intents=group,
            )
            for target_id in targets:
                if not target_id or target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                mode_cfg = catalog.get(target_id)
                satisfies = tuple(getattr(mode_cfg, "satisfies_intents", []) or [])
                steps.append(
                    WorkflowStep(
                        step_id=_step_id_for(target_id, step_index),
                        mode_id=target_id,
                        satisfies_intents=satisfies,
                        depends_on=tuple(s.step_id for s in steps),
                        is_terminal=False,
                        spawned_from=spawned_from,
                    )
                )
                step_index += 1

    if not steps:
        return ()
    # Mark the last step terminal so the turn loop knows the chain ends.
    last = steps[-1]
    steps[-1] = last.model_copy(update={"is_terminal": True})
    return tuple(steps)


def _targets_covering_intents(
    *,
    routing_targets: list[str],
    catalog: dict[str, Any],
    gap_intents: set[str],
) -> list[str]:
    """Return routing targets whose intent tags overlap ``gap_intents``.

    Used by the non-sequential branch of ``compile_steps_from_routing_targets``
    to flatten a fork's children into one step each. A target is included
    when any of its ``routing_intent_tags`` is in the gap. Order follows the
    ``routing_targets`` list (catalog priority order) so the step chain is
    stable across recompiles.
    """
    out: list[str] = []
    for target_id in routing_targets:
        mode_cfg = catalog.get(target_id)
        if mode_cfg is None:
            continue
        tags = set(getattr(mode_cfg, "routing_intent_tags", []) or [])
        if tags & gap_intents:
            out.append(target_id)
    return out


def _intents_by_earliest_keyword(
    *,
    query: str,
    routing_targets: list[str],
    catalog: dict[str, Any],
    gap_intents: set[str],
) -> list[str]:
    """Order intents by the earliest position of any of their keywords.

    For a sequential chain, intents within a group (e.g. two consumer
    intents ``{coding, write}``) must be emitted as separate steps in the
    order the user phrased them. Each intent is mapped to the earliest
    position of any keyword attributed to it across all routing targets'
    ``routing_intent_keywords``. Intents with no matching keyword sort last
    (stable order by first-seen), so a detected-but-keywordless intent is
    still emitted (it was detected via a fast-path pattern).
    """
    query_lower = query.lower()
    positions: list[tuple[int, str]] = []
    for intent in gap_intents:
        best = -1
        for target_id in routing_targets:
            mode_cfg = catalog.get(target_id)
            if mode_cfg is None:
                continue
            keywords = (getattr(mode_cfg, "routing_intent_keywords", {}) or {}).get(intent, [])
            for keyword in keywords:
                pos = query_lower.find(keyword)
                if pos != -1 and (best == -1 or pos < best):
                    best = pos
        positions.append((best, intent))
    # Sort by position; -1 (no keyword found) sorts after all real positions.
    # Use a stable sort so ties keep first-seen order.
    positions.sort(key=lambda item: (1 if item[0] == -1 else 0, item[0]))
    return [intent for _, intent in positions]


def render_workflow_progress(instance: WorkflowInstance) -> str:
    """Render a short human-readable summary of completed and remaining steps.

    Used by the host's context assembler (Phase 7) as the "work done so far"
    summary — replacing re-reading ``CheckpointBundle.turn_history`` (which
    is host-owned UI/replay state the kernel never reads for routing). The
    summary lists each declared step with a checkmark (done) or a dash
    (remaining), plus the result preview when available.

    Returns an empty string when the instance has no declared steps or is
    complete (no progress to report).
    """
    if not instance.declared_steps:
        return ""
    if is_complete(instance):
        return ""
    completed = set(instance.completed_steps)
    lines: list[str] = ["## Workflow progress"]
    for step in instance.declared_steps:
        mark = "[x]" if step.step_id in completed else "[ ]"
        label = f"{mark} {step.mode_id}"
        preview = instance.step_results.get(step.step_id, "")
        if preview:
            # Truncate the preview to keep the summary compact.
            label += f": {preview[:120]}"
        lines.append(label)
    return "\n".join(lines)
