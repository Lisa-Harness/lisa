"""Unit tests for the derived WorkflowView.

Tests the pure derivation function with synthetic thread tails — no
disk I/O, no host wiring. Covers:
- Plan with 0/1/all receipts → correct completion state
- No plan block → None
- Resume across entries (plan in older entry, receipts in newer)
- Prose plan without **mode_id** tags → skipped (not hollow)
- Progress rendering
"""

from __future__ import annotations

from kernel.workflow_view import (
    WorkflowView,
    derive_workflow_view,
    is_complete,
    remaining_steps,
    render_workflow_progress,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _entry(
    mode_id: str = "default",
    response_text: str = "",
    effect_receipts: list[dict] | None = None,
    handoff_to: str = "",
    transitioned_to: str = "",
) -> dict:
    """Build a synthetic thread entry."""
    entry: dict = {
        "timestamp_utc": "2026-08-10T12:00:00+00:00",
        "user_slug": "test",
        "mode_id": mode_id,
        "query": "test query",
        "response_text": response_text,
    }
    if effect_receipts:
        entry["effect_receipts"] = effect_receipts
    if handoff_to:
        entry["handoff_to"] = handoff_to
    if transitioned_to:
        entry["transitioned_to"] = transitioned_to
    return entry


def _receipt(capability_id: str = "write_file", effect: str = "write") -> dict:
    return {"capability_id": capability_id, "effect": effect, "artifact_id": ""}


_PLAN_3_STEP = """\
## Plan

1. **search** — Search for the latest rust async runtime news
   _reads_
   _Provides web_search_
2. **coding** — Implement the client
   _writes: coding_apply_patch_
   _Provides code generation_
3. **markdown** — Document usage
   _writes: markdown_generate_
   _Provides document drafting_
"""


# ── Tests ──────────────────────────────────────────────────────────────


class TestDeriveWorkflowView:
    """Tests for derive_workflow_view."""

    def test_no_entries_returns_none(self):
        assert derive_workflow_view([]) is None

    def test_no_plan_block_returns_none(self):
        entries = [
            _entry(mode_id="default", response_text="Just a chat response."),
        ]
        assert derive_workflow_view(entries) is None

    def test_plan_with_zero_receipts_all_remaining(self):
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert len(view.declared_steps) == 3
        assert len(view.completed_steps) == 0
        assert view.current_step_id == "search_0"
        assert not is_complete(view)
        assert len(remaining_steps(view)) == 3

    def test_plan_with_one_receipt_first_step_complete(self):
        # search is a read step (_reads_); it completes on handoff_to.
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(
                mode_id="search",
                response_text="Found 3 results.",
                handoff_to="coding",
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert len(view.completed_steps) == 1
        assert view.completed_steps[0] == "search_0"
        assert view.current_step_id == "coding_1"
        assert len(remaining_steps(view)) == 2
        assert not is_complete(view)

    def test_write_step_completes_on_matching_receipt(self):
        # coding is a write step (_writes: coding_apply_patch_);
        # it completes on a matching write receipt, not on handoff.
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="Implemented.",
                effect_receipts=[_receipt("coding_apply_patch", "write")],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "search_0" in view.completed_steps
        assert "coding_1" in view.completed_steps
        assert view.current_step_id == "markdown_2"

    def test_write_step_not_completed_by_non_matching_receipt(self):
        # A write step must match its specific write_capability — a stray
        # receipt for a different capability does not complete it.
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="Read-only work.",
                effect_receipts=[_receipt("coding_read_file", "write")],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "search_0" in view.completed_steps
        assert "coding_1" not in view.completed_steps
        assert view.current_step_id == "coding_1"

    def test_plan_with_all_receipts_returns_none(self):
        """When all steps are complete, the workflow is done — return None."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="Implemented.",
                effect_receipts=[_receipt("coding_apply_patch", "write")],
            ),
            _entry(
                mode_id="markdown",
                response_text="Documented.",
                effect_receipts=[_receipt("markdown_generate", "write")],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is None

    def test_resume_across_entries(self):
        """Plan in an older entry, completion in newer entries — correctly derived."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searching..."),
            _entry(mode_id="search", response_text="Done.", handoff_to="coding"),
            _entry(mode_id="default", response_text="Continue."),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert len(view.completed_steps) == 1
        assert view.current_step_id == "coding_1"

    def test_mode_stage_derived_from_last_entry(self):
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(
                mode_id="search",
                response_text="Found results.",
                effect_receipts=[_receipt("web_search", "write")],
                transitioned_to="reason",
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert view.mode_id == "search"
        assert view.stage_id == "reason"

    def test_turn_count_consecutive_same_mode(self):
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Turn 1"),
            _entry(mode_id="search", response_text="Turn 2"),
            _entry(mode_id="search", response_text="Turn 3"),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert view.turn_count == 3

    def test_turn_count_resets_on_mode_change(self):
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Turn 1"),
            _entry(mode_id="coding", response_text="Turn 2"),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert view.turn_count == 1

    def test_read_receipts_do_not_complete_steps(self):
        """Read-effect receipts (effect=read) should not complete a write step."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Read-only results.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="Read-only work.",
                effect_receipts=[_receipt("coding_apply_patch", "read")],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        # search (read step) completed on handoff; coding (write step) did
        # not — a read-effect receipt does not satisfy the write contract.
        assert "search_0" in view.completed_steps
        assert "coding_1" not in view.completed_steps
        assert view.current_step_id == "coding_1"

    def test_prose_plan_without_mode_tags_returns_none(self):
        """A prose plan without **mode_id** tags is not parseable — return None."""
        prose = """\
## Plan

First I'll search for the topic, then code the implementation,
and finally write the documentation.
"""
        entries = [_entry(mode_id="default", response_text=prose)]
        view = derive_workflow_view(entries)
        assert view is None

    def test_step_results_populated_from_completing_entry(self):
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(
                mode_id="search",
                response_text="Found 3 results about rust async runtimes.",
                handoff_to="coding",
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "search_0" in view.step_results
        assert "rust async" in view.step_results["search_0"]

    def test_position_id_stable_for_same_plan(self):
        """Same plan text + same mode → same position_id across calls."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
        ]
        view1 = derive_workflow_view(entries)
        view2 = derive_workflow_view(entries)
        assert view1 is not None
        assert view2 is not None
        assert view1.position_id == view2.position_id

    def test_handoff_to_tracked_in_entries(self):
        """handoff_to field in thread entries completes a read step."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(
                mode_id="search",
                response_text="Done.",
                handoff_to="coding",
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        # search (read step) completed via handoff_to; mode derived from
        # the last entry's mode_id.
        assert "search_0" in view.completed_steps
        assert view.mode_id == "search"


class TestRenderWorkflowProgress:
    """Tests for render_workflow_progress."""

    def test_empty_view_returns_empty(self):
        view = WorkflowView()
        assert render_workflow_progress(view) == ""

    def test_complete_view_returns_empty(self):
        from kernel.workflow_position import WorkflowStep

        step = WorkflowStep(
            step_id="search_0",
            mode_id="search",
            is_terminal=True,
        )
        view = WorkflowView(
            declared_steps=(step,),
            completed_steps=("search_0",),
            terminal_reached=True,
        )
        assert render_workflow_progress(view) == ""

    def test_incomplete_view_shows_progress(self):
        from kernel.workflow_position import WorkflowStep

        steps = (
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding"),
        )
        view = WorkflowView(
            declared_steps=steps,
            completed_steps=("search_0",),
            current_step_id="coding_1",
        )
        progress = render_workflow_progress(view)
        assert "## Workflow progress" in progress
        assert "[x] search" in progress
        assert "[ ] coding" in progress

    def test_progress_with_step_results(self):
        from kernel.workflow_position import WorkflowStep

        steps = (
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding"),
        )
        view = WorkflowView(
            declared_steps=steps,
            completed_steps=("search_0",),
            current_step_id="coding_1",
            step_results={"search_0": "Found 3 results about rust async runtimes."},
        )
        progress = render_workflow_progress(view)
        assert "[x] search: Found 3 results" in progress