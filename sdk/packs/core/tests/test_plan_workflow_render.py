"""Plan-block rendering: no internal template tokens in operator-facing text.

The rendered ``## Plan`` block is both the visible artifact and the input to
``kernel.workflow_view``'s parser, so these tests pin both halves: the
operator-facing text must not carry template syntax, and the round-trip
through the parser must still bind each step's completion contract.

Regression for the 2026-10-27T18:12Z turn, where a read step rendered as
``_reads_`` — the literal template token — in the plan shown to the user.
"""

from __future__ import annotations

from kernel.workflow_view import WorkflowView, _parse_plan_block, derive_workflow_view
from sdk.packs.core.operations.planning.plan_workflow import (
    PlanStep,
    PlanWorkflowOutput,
    _render_plan_markdown,
)


def _read_step() -> PlanStep:
    return PlanStep(
        mode_id="default",
        description="Talk through where the planning system stands",
        rationale="The user is saying they have run out of problems to pose.",
    )


def _write_step() -> PlanStep:
    return PlanStep(
        mode_id="markdown",
        description="Write the guide",
        rationale="The markdown mode owns the forge deliverable.",
        write_capability="markdown_forge_document",
    )


def test_render_plan_markdown_omits_read_step_template_token() -> None:
    """A read step must not render the bare ``_reads_`` template token.

    ``_reads_`` is parser syntax, not prose. The parser already treats an
    absent deliverable line as ``write_capability is None``, so rendering
    the token adds nothing to the contract and leaks internals to the
    operator.
    """
    rendered = _render_plan_markdown(PlanWorkflowOutput(steps=[_read_step()]))

    assert "_reads_" not in rendered
    assert "1. **default**" in rendered
    # The rationale is a schema-declared, user-facing field: keep it, on its
    # own line and not folded into the description.
    assert "   _The user is saying they have run out of problems to pose._" in rendered


def test_render_plan_markdown_keeps_write_step_deliverable_line() -> None:
    """A write step still renders its ``_writes: <capability>_`` line.

    That line is what binds step completion to a write receipt; omitting
    unset fields must not drop the field that is set.
    """
    rendered = _render_plan_markdown(PlanWorkflowOutput(steps=[_write_step()]))

    assert "_writes: markdown_forge_document_" in rendered
    assert "run out of problems" not in rendered  # not this step's rationale


def test_rendered_plan_round_trips_through_the_kernel_parser() -> None:
    """The rendered block must still parse into the same completion contract.

    Dropping the ``_reads_`` token is only safe if ``derive_workflow_view``
    reads the resulting step as a read step (``write_capability is None``)
    and the write step as its bound capability. Without this, a cosmetic
    fix would silently break plan progress tracking.
    """
    rendered = _render_plan_markdown(PlanWorkflowOutput(steps=[_read_step(), _write_step()]))
    parsed = _parse_plan_block(rendered)
    assert parsed is not None
    _plan_text, steps = parsed

    assert len(steps) == 2
    read_mode, read_desc, read_rationale, read_cap = steps[0]
    assert read_mode == "default"
    assert read_desc == "Talk through where the planning system stands"
    # The rationale lands in its own field, which becomes the step's plan_text.
    assert read_rationale == "The user is saying they have run out of problems to pose."
    assert read_cap is None

    write_mode, _write_desc, _write_rationale, write_cap = steps[1]
    assert write_mode == "markdown"
    assert write_cap == "markdown_forge_document"

    # The view derives the same contract from a thread entry carrying it.
    view = _derive_view_from(rendered)
    assert view is not None
    by_mode = {s.mode_id: s for s in view.declared_steps}
    assert by_mode["default"].write_capability is None
    assert by_mode["markdown"].write_capability == "markdown_forge_document"
    assert isinstance(view, WorkflowView)


def _derive_view_from(rendered: str) -> WorkflowView | None:
    """Derive a WorkflowView from one synthetic thread entry carrying the plan."""
    return derive_workflow_view(
        [
            {
                "turn": 0,
                "mode_id": "planner",
                "response_text": rendered,
                "handoff_to": "",
                "transitioned_to": "",
            }
        ]
    )
