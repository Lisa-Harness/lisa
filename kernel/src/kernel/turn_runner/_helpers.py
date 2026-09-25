"""Helpers used by the kernel turn loop (_fire_slice, plan→workflow
step compilation, recovery identity, routing/coordinator helpers).

These were extracted from the former single-file ``turn_runner``; the engine
imports them back so ``kernel.turn_runner`` keeps its historical module
namespace (``mock.patch("kernel.turn_runner.<name>")``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from kernel.models import OperationResult
from kernel.slice import SliceMode, SliceOutcome, build_policy_for_trigger, resolve_mode_for_trigger
from kernel.turn_hooks import TurnHooks
from kernel.workflow_position import WorkflowStep, _step_id_for

logger = logging.getLogger(__name__)


def _engine_catalog() -> dict[str, Any]:
    """Resolve the workflow catalog through the engine's module namespace.

    The helpers here must honor the historical patch contract:
    ``mock.patch("kernel.turn_runner.get_workflow_catalog")`` patches the
    *engine's* binding, so the helper reads the catalog attribute off the
    ``kernel.turn_runner`` package (which is the engine module after the
    split) at call time rather than binding a private copy at import.
    """
    import kernel.turn_runner as _engine  # package namespace

    return _engine.get_workflow_catalog()


# ── Internal helpers ────────────────────────────────────────────────────────


async def _fire_slice(
    *,
    hooks: TurnHooks,
    trigger: SliceMode,
    slice_outcomes: list[SliceOutcome],
) -> SliceOutcome | None:
    """Dispatch one opportunistic house-keeping slice.

    Returns the outcome if the slice actually ran, ``None`` otherwise.
    The hook is invoked only when the host has both wired the
    corresponding slot AND enabled the trigger in its ``SliceTuning``.

    The caller is responsible for updating
    ``hooks.lifecycle.last_slice_finished_at`` after the call returns; that
    field is on the host's ``TurnHooks`` instance so it survives across
    turns of the same session.
    """
    slot = {
        SliceMode.PRE_TURN: hooks.lifecycle.pre_turn_slice,
        SliceMode.POST_TURN: hooks.lifecycle.post_turn_slice,
        SliceMode.EXPLICIT: hooks.lifecycle.explicit_slice,
    }.get(trigger)
    if slot is None:
        return None

    tuning = hooks.lifecycle.slice_tuning
    if tuning is None:
        return None
    if resolve_mode_for_trigger(tuning=tuning, trigger=trigger) == SliceMode.NEVER:
        return None

    has_pre_turn = (
        hooks.lifecycle.pre_turn_slice is not None
        and resolve_mode_for_trigger(tuning=tuning, trigger=SliceMode.PRE_TURN) != SliceMode.NEVER
    )
    has_post_turn = (
        hooks.lifecycle.post_turn_slice is not None
        and resolve_mode_for_trigger(tuning=tuning, trigger=SliceMode.POST_TURN) != SliceMode.NEVER
    )
    policy = build_policy_for_trigger(
        trigger=trigger,
        tuning=tuning,
        last_slice_finished_at=hooks.lifecycle.last_slice_finished_at,
        has_pre_turn=has_pre_turn,
        has_post_turn=has_post_turn,
    )
    if policy.budget_ms <= 0:
        return None

    started = time.monotonic()
    try:
        outcome = await slot(policy)
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.warning(
            "%s slice failed: %s: %s",
            trigger.value,
            type(exc).__name__,
            exc,
        )
        outcome = SliceOutcome(
            work_items_completed=0,
            elapsed_ms=elapsed_ms,
            last_item="<error>",
            notes=(f"{type(exc).__name__}: {exc}",),
        )
    if not isinstance(outcome, SliceOutcome):
        # Defensive: a host that returns a non-SliceOutcome should not
        # break the turn. Coerce silently and log once.
        logger.debug(
            "%s slice returned %s instead of SliceOutcome; coercing",
            trigger.value,
            type(outcome).__name__,
        )
        outcome = SliceOutcome(
            work_items_completed=0,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )
    # Backfill elapsed_ms if the host didn't measure.
    elif outcome.elapsed_ms <= 0:
        outcome = SliceOutcome(
            work_items_completed=outcome.work_items_completed,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=outcome.last_item,
            notes=outcome.notes,
        )
    slice_outcomes.append(outcome)
    return outcome


@dataclass(frozen=True, slots=True)
class _StubModelDef:
    model_id: str
    reasoning_model: bool = False


def _extract_pre_turn_briefing(result: OperationResult) -> str:
    """Return the briefing text that should be injected into the user payload.

    Declarative operations may fall back to rendering their full structured payload
    as JSON when `output_text` is empty. That is useful for direct operation
    callers, but pre-turn prompt assembly should only inject the authored briefing
    text field itself. When the structured payload explicitly says `output_text`
    is empty, treat the pre-turn briefing as absent.
    """

    payload = result.output_json
    if isinstance(payload, dict):
        for field_name in ("output_text", "body_markdown", "markdown", "answer_summary", "summary"):
            candidate = payload.get(field_name)
            if isinstance(candidate, str):
                return candidate.strip()

    return result.output_text.strip()


def _plan_steps_to_workflow_steps(
    plan_steps: list[dict[str, Any]],
) -> tuple[WorkflowStep, ...]:
    """Convert plan_workflow's ``steps`` dicts into ``WorkflowStep`` objects.

    Each plan step has ``mode_id``, ``description``, and ``rationale``. The
    ``WorkflowStep`` carries ``mode_id`` and ``step_id`` (the
    description/rationale are for the user-visible plan block, not the
    instance). To keep the planned path shape-identical to the ad-hoc
    compiler's output (``compile_steps_from_routing_targets``), the
    converter enriches each step with ``satisfies_intents`` and
    ``depends_on`` looked up from the workflow catalog. This makes every
    downstream consumer (``chain_satisfied`` hydration, resume hints,
    progress rendering) origin-blind — no branch needs to special-case
    ``spawned_from == "plan_workflow_op"`` to compensate for missing
    fields. The last step is marked terminal. Step ids are deterministic
    via ``_step_id_for`` so a recompiled plan preserves ``completed_steps``
    by id intersection.
    """
    if not plan_steps:
        return ()
    catalog = _engine_catalog()
    steps: list[WorkflowStep] = []
    for idx, step in enumerate(plan_steps):
        mode_id = str(step.get("mode_id", "")).strip()
        if not mode_id:
            continue
        mode_cfg = catalog.get(mode_id)
        satisfies = tuple(getattr(mode_cfg, "satisfies_intents", []) or []) if mode_cfg else ()

        # Extract the step-specific query from the plan's description or rationale.
        # This becomes the authoritative task payload when this step executes.
        step_query = str(step.get("description", "")).strip()
        if not step_query:
            step_query = str(step.get("rationale", "")).strip()

        # Carry the step's rationale onto the WorkflowStep so the handoff
        # context builder can surface why this step exists, not just the
        # short description. The rationale was previously dropped here
        # (the 2026-08-09 plan-amnesia incident: the executor saw a bare
        # mode id + generic description with no plan context).
        step_rationale = str(step.get("rationale", "")).strip()

        # Carry the step's deliverable capability so the WorkflowView's
        # completion check matches the kernel's write-receipt contract.
        # None for read/gather steps (complete on handoff).
        write_capability = step.get("write_capability")
        if not isinstance(write_capability, str) or not write_capability.strip():
            write_capability = None
        else:
            write_capability = write_capability.strip()

        steps.append(
            WorkflowStep(
                step_id=_step_id_for(mode_id, idx),
                mode_id=mode_id,
                step_query=step_query,
                satisfies_intents=satisfies,
                depends_on=tuple(s.step_id for s in steps),
                is_terminal=(idx == len(plan_steps) - 1),
                spawned_from="plan_workflow_op",
                plan_text=step_rationale,
                write_capability=write_capability,
            )
        )
    return tuple(steps)
