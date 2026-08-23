"""Phase 7 tests: workflow progress rendering.

Covers ``render_workflow_progress`` (the "work done so far" summary).
"""

from __future__ import annotations

from kernel.workflow_position import (
    WorkflowPosition,
    WorkflowStep,
    render_workflow_progress,
)


def _instance() -> WorkflowPosition:
    return WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search", satisfies_intents=("research",)),
            WorkflowStep(
                step_id="coding_1",
                mode_id="coding",
                satisfies_intents=("coding",),
                depends_on=("search_0",),
                is_terminal=True,
            ),
        ),
        completed_steps=("search_0",),
        current_step_id="coding_1",
        mode_id="coding",
        step_results={"search_0": "found 3 frameworks"},
    )


def test_render_workflow_progress_marks_completed_and_remaining() -> None:
    """The summary shows [x] for completed steps and [ ] for remaining."""
    progress = render_workflow_progress(_instance())
    assert "## Workflow progress" in progress
    assert "[x] search: found 3 frameworks" in progress
    assert "[ ] coding" in progress


def test_render_workflow_progress_empty_when_no_steps() -> None:
    """An instance with no declared steps renders an empty summary."""
    assert render_workflow_progress(WorkflowPosition()) == ""


def test_render_workflow_progress_empty_when_complete() -> None:
    """A complete instance renders an empty summary (nothing to report)."""
    instance = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="s_0", mode_id="search"),),
        completed_steps=("s_0",),
    )
    assert render_workflow_progress(instance) == ""


def test_render_workflow_progress_truncates_long_preview() -> None:
    """The result preview is truncated to keep the summary compact."""
    long_preview = "x" * 300
    instance = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="s_0", mode_id="search"),
            WorkflowStep(step_id="c_1", mode_id="coding", is_terminal=True),
        ),
        completed_steps=("s_0",),
        current_step_id="c_1",
        step_results={"s_0": long_preview},
    )
    progress = render_workflow_progress(instance)
    # The preview line is at most 120 chars after the mode label.
    line = [ln for ln in progress.splitlines() if "search" in ln][0]
    assert len(line) < 200  # not the full 300-char preview
