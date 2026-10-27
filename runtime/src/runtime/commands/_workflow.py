"""Operator commands for the active multi-step workflow.

``/workflow`` exposes the plan a ``plan_workflow`` turn left behind: what is
done, what is left, and a way to retire it deliberately. Before this command
the only endings were finishing every step or superseding the plan with a new
one — an abandoned plan stayed active forever, and because the derived
``WorkflowView`` is injected into every turn's briefing and steers routing, it
kept costing the operator context and misdirected handoffs.

Concluding is durable because the marker is a thread entry, not in-process
state: ``derive_workflow_view`` stops at a conclusion recorded after the plan,
so the decision survives restarts and stays auditable in the thread.

Dependency direction: ``_workflow`` → ``_log``, ``_render``, ``kernel.*``,
``runtime.thread``. ``_simple`` imports the handler, so this module must not
import ``_simple``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from kernel.workflow_view import (
    WORKFLOW_CONCLUSION_KEY,
    WorkflowView,
    derive_workflow_view,
)

from runtime.commands._log import _append_thread_entry
from runtime.commands._render import render_workflow_status
from runtime.runtime import HostRuntime
from runtime.thread import load_thread_range, next_turn_index

logger = logging.getLogger(__name__)

#: Mode id stamped on the thread entry recording a conclusion. It is not a
#: routable mode: the entry exists to carry the marker and the reason, and
#: nothing routes from it because the plan it retires is no longer active.
_CONCLUSION_MODE_ID = "workflow"

#: How many turns back to look for the active plan. ``_latest_plan_entry``
#: scans backwards from the newest entry, so the window only has to reach the
#: plan; a generous bound keeps a long-running plan visible without reading
#: the whole history on every invocation.
_STATUS_WINDOW_TURNS = 2048


def _load_window(runtime: HostRuntime) -> list[dict[str, object]]:
    """Return the recent thread entries a workflow command inspects."""
    end_turn = next_turn_index(thread_path=runtime.paths.thread_path) - 1
    if end_turn < 0:
        return []
    return load_thread_range(
        thread_path=runtime.paths.thread_path,
        start_turn=max(0, end_turn - _STATUS_WINDOW_TURNS + 1),
        end_turn=end_turn,
    )


def _active_view(runtime: HostRuntime) -> WorkflowView | None:
    """Return the active workflow view, or ``None`` when nothing is in flight."""
    entries = _load_window(runtime)
    if not entries:
        return None
    return derive_workflow_view(entries)


def _render_no_workflow() -> str:
    return (
        "## Workflow\n\n"
        "No workflow is active. A workflow begins when the `planner` mode "
        "writes a `## Plan` block; it retires automatically once every declared "
        "step has its completion evidence, or deliberately via "
        "`/workflow conclude`."
    )


def _conclude(runtime: HostRuntime, reason: str) -> str:
    """Record a durable conclusion for the active workflow.

    Returns the operator-facing result. Appends exactly one thread entry
    carrying ``provenance[workflow_conclusion]``; the append-only thread means
    that entry alone retires the plan for every later derivation.
    """
    view = _active_view(runtime)
    if view is None:
        return _render_no_workflow()

    completed_ids = set(view.completed_steps)
    remaining = [step for step in view.declared_steps if step.step_id not in completed_ids]
    completed = len(view.declared_steps) - len(remaining)
    detail = (
        f"Concluded the workflow with {completed} of {len(view.declared_steps)} step(s) complete."
    )
    if remaining:
        pending = ", ".join(step.mode_id for step in remaining)
        detail += f" Not run: {pending}."

    provenance: dict[str, object] = {
        "source": "operator",
        WORKFLOW_CONCLUSION_KEY: reason,
    }
    _append_thread_entry(
        runtime,
        mode_id=_CONCLUSION_MODE_ID,
        query=reason or "/workflow conclude",
        response_text=detail,
        provenance=provenance,
    )
    logger.info(
        "workflow concluded by operator: completed=%d/%d remaining=%r reason=%r",
        completed,
        len(view.declared_steps),
        [step.mode_id for step in remaining],
        reason,
    )
    return "\n\n".join(part for part in ("## Workflow", detail, _reopen_hint(reason)) if part)


def _reopen_hint(reason: str) -> str:
    """Explain how the retired plan is recorded, so the decision is auditable."""
    stamp = datetime.now(UTC).isoformat()
    reason_text = f" Reason recorded: {reason}" if reason else ""
    return (
        f"Recorded in the thread at {stamp}.{reason_text}\n"
        "The plan is no longer injected into turn briefings and no longer "
        "steers routing. A new `plan` turn starts a fresh workflow."
    )


def handle_workflow_command(runtime: HostRuntime, stripped: str, message: str) -> str:
    """Handle ``/workflow [status|conclude [reason]]``.

    ``status`` (the default) renders the active plan's steps and the current
    step. ``conclude`` retires it durably, optionally recording the operator's
    reason. Unknown subcommands report usage rather than silently defaulting,
    so a typo cannot look like it concluded a workflow.
    """
    arg = stripped[len("/workflow") :].strip()
    subcommand, _, remainder = arg.partition(" ")
    subcommand = subcommand.lower()
    reason = remainder.strip()

    if subcommand in ("", "status", "show"):
        view = _active_view(runtime)
        if view is None:
            return _render_no_workflow()
        return render_workflow_status(view)

    if subcommand in ("conclude", "close", "abandon", "cancel", "done"):
        return _conclude(runtime, reason)

    return (
        "Usage: `/workflow [status | conclude [reason]]`\n\n"
        f"Unrecognized subcommand: {subcommand!r}"
    )


__all__ = ["handle_workflow_command"]
