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

import pytest
from kernel.workflow_view import (
    WorkflowView,
    derive_workflow_view,
    has_workflow_plan,
    is_complete,
    remaining_steps,
    render_workflow_progress,
)

# ── Helpers ────────────────────────────────────────────────────────────


def _entry(  # noqa: PLR0917 — one default per synthetic-entry field
    mode_id: str = "default",
    response_text: str = "",
    effect_receipts: list[dict] | None = None,
    handoff_to: str = "",
    transitioned_to: str = "",
    turn: int | None = None,
    departed_modes: list[str] | None = None,
) -> dict:
    """Build a synthetic thread entry."""
    entry: dict = {
        "timestamp_utc": "2026-08-10T12:00:00+00:00",
        "user_slug": "test",
        "mode_id": mode_id,
        "query": "test query",
        "response_text": response_text,
    }
    if departed_modes is not None:
        entry["departed_modes"] = departed_modes
    if effect_receipts:
        entry["effect_receipts"] = effect_receipts
    if handoff_to:
        entry["handoff_to"] = handoff_to
    if transitioned_to:
        entry["transitioned_to"] = transitioned_to
    if turn is not None:
        entry["turn"] = turn
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("## Plan\nNo parseable steps", False),
        ("### Plan\n1. executor - Edit the worktree", False),
        ("## Plan\n1. executor - Edit the worktree\n_writes: coding_edit_files_", True),
        (_PLAN_3_STEP, True),
    ],
)
def test_has_workflow_plan_requires_parseable_steps(text: str, expected: bool) -> None:
    assert has_workflow_plan(text) is expected


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

    def test_completing_turn_recorded_on_write_step(self):
        # When a write step's receipt match is found, the completing entry's
        # global turn index is recorded on the step as ``completing_turn``.
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP, turn=0),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding", turn=1),
            _entry(
                mode_id="coding",
                response_text="Implemented.",
                effect_receipts=[_receipt("coding_apply_patch", "write")],
                turn=2,
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        coding_step = view.step_by_id("coding_1")
        assert coding_step is not None
        assert coding_step.completing_turn == 2
        search_step = view.step_by_id("search_0")
        assert search_step is not None
        assert search_step.completing_turn == 1

    def test_completing_turn_none_when_step_not_complete(self):
        # An incomplete step has completing_turn = None.
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP, turn=0),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding", turn=1),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        coding_step = view.step_by_id("coding_1")
        assert coding_step is not None
        assert coding_step.completing_turn is None

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
        view = derive_workflow_view(entries, idle_turn_limit=0)
        assert view is not None
        assert view.turn_count == 3

    def test_turn_count_resets_on_mode_change(self):
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Turn 1"),
            _entry(mode_id="coding", response_text="Turn 2"),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=0)
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


class TestOutOfOrderCompletion:
    """Steps completing out of declared order must never route backward.

    Regression lock for the 2026-10-30 t1107/t1108 loop: a plan declared
    ``thinker → researcher`` where only ``researcher`` completed. The naive
    ``remaining[0]`` pointer parked on the abandoned ``thinker`` step
    forever, so every post-handoff routing hint said "go to thinker" and
    the forced-action enforcer synthesized that backward transition over
    the model's refusal — twice per turn, with the loop feeding itself
    because each hijacked turn ran in ``thinker`` and reset the idle
    retirement clock.
    """

    _PLAN_TWO_STEP = """\
## Plan

1. **thinker** — Analyze the landscape
   _reads_
2. **researcher** — Investigate the trends
   _writes: markdown_forge_document_
"""

    def _completed_second_step_entries(self) -> list[dict]:
        """A plan whose second step completed; first step never did."""
        return [
            _entry(mode_id="planner", response_text=self._PLAN_TWO_STEP, turn=0),
            _entry(
                mode_id="researcher",
                response_text="Research done.",
                effect_receipts=[
                    {
                        "capability_id": "markdown_forge_document",
                        "effect": "write",
                        "mode_id": "researcher",
                    }
                ],
                turn=1,
            ),
        ]

    def test_pointer_never_parks_on_abandoned_earlier_step(self):
        """With the tail step complete, the pointer is empty — not backward.

        The chain has no forward step to name: ``researcher`` is done and
        ``thinker`` was abandoned (it precedes the completion frontier).
        The view stays alive so the checklist renders, but routing
        (hint, resume, budget transition) goes silent.
        """
        view = derive_workflow_view(self._completed_second_step_entries(), idle_turn_limit=0)
        assert view is not None
        assert view.completed_steps == ("researcher_1",)
        assert view.current_step_id == ""

    def test_abandoned_pre_frontier_mode_does_not_keep_plan_alive(self):
        """A turn in the abandoned step's mode must not reset the idle clock.

        Each hijacked turn of the 2026-10-30 loop ended in `thinker` —
        the abandoned pre-frontier step's mode — which reset the idle
        retirement count under the old rule and immortalized the zombie
        plan. Only post-frontier open steps keep a plan alive now.
        """
        entries = [
            *self._completed_second_step_entries(),
            _entry(mode_id="thinker", response_text="Looped back.", turn=2),
            _entry(mode_id="default", response_text="Unrelated.", turn=3),
        ]
        # The thinker turn would reset the idle count under the old rule;
        # with the front-aware open-mode set, turns 2 and 3 are both idle
        # and the plan retires at limit 2.
        assert derive_workflow_view(entries, idle_turn_limit=2) is None

    def test_pointer_resumes_on_first_open_step_after_frontier(self):
        """With a post-frontier step open, the pointer names it — never earlier.

        ``search → coding → markdown`` with ``coding`` completed but
        ``search`` abandoned: the pointer is ``markdown_2`` (first open
        step after the frontier), not the abandoned ``search_0``.
        """
        plan = """\
## Plan

1. **search** — Search for the news
   _reads_
2. **coding** — Implement it
   _writes: coding_apply_patch_
3. **markdown** — Document it
   _writes: markdown_generate_
"""
        entries = [
            _entry(mode_id="planner", response_text=plan, turn=0),
            _entry(
                mode_id="coding",
                response_text="Implemented.",
                effect_receipts=[
                    {"capability_id": "coding_apply_patch", "effect": "write", "mode_id": "coding"}
                ],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=0)
        assert view is not None
        assert view.completed_steps == ("coding_1",)
        assert view.current_step_id == "markdown_2"


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

    def test_progress_includes_completing_turn_annotation(self):
        """Completed steps show ``← turn N`` when ``completing_turn`` is set."""
        from kernel.workflow_position import WorkflowStep

        steps = (
            WorkflowStep(
                step_id="search_0",
                mode_id="search",
                completing_turn=5,
            ),
            WorkflowStep(step_id="coding_1", mode_id="coding"),
        )
        view = WorkflowView(
            declared_steps=steps,
            completed_steps=("search_0",),
            current_step_id="coding_1",
        )
        progress = render_workflow_progress(view)
        assert "[x] search" in progress
        assert "← turn 5" in progress

    def test_progress_includes_current_step_block(self):
        """The ``## Current step`` block gives the full description + rationale."""
        from kernel.workflow_position import WorkflowStep

        steps = (
            WorkflowStep(
                step_id="search_0",
                mode_id="search",
                step_query="Search for rust async runtime benchmarks",
                plan_text="_Provides web_search_",
                completing_turn=3,
            ),
            WorkflowStep(
                step_id="coding_1",
                mode_id="coding",
                step_query="Implement the async client based on search findings",
                plan_text="_writes: coding_apply_patch_",
            ),
        )
        view = WorkflowView(
            declared_steps=steps,
            completed_steps=("search_0",),
            current_step_id="coding_1",
        )
        progress = render_workflow_progress(view)
        assert "## Current step" in progress
        assert "**coding**: Implement the async client based on search findings" in progress
        assert "_writes: coding_apply_patch_" in progress

    def test_progress_preview_uses_300_char_limit(self):
        """The step result preview is truncated at 300 chars, not 120."""
        from kernel.workflow_position import WorkflowStep

        long_preview = "A" * 400
        steps = (
            WorkflowStep(step_id="search_0", mode_id="search", completing_turn=1),
            WorkflowStep(step_id="coding_1", mode_id="coding"),
        )
        view = WorkflowView(
            declared_steps=steps,
            completed_steps=("search_0",),
            current_step_id="coding_1",
            step_results={"search_0": long_preview},
        )
        progress = render_workflow_progress(view)
        # 300 chars of the preview should be present, but not 400.
        assert "A" * 300 in progress
        assert "A" * 301 not in progress


class TestWriteReceiptContract:
    """Lock in the terminal-write-mode receipt contract from the execution plan.

    A write-oriented step is terminal only when the runtime observes a
    matching ``effect_receipt``. Model assertions ("I am done writing")
    are never sufficient — the receipt is the honest signal.
    """

    def test_model_assertion_without_receipt_does_not_complete(self):
        """A write step whose mode produced content but no receipt stays incomplete."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="I have finished implementing the client.",
                # No effect_receipts — the model claims completion but
                # produced no durable proof.
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "search_0" in view.completed_steps
        assert "coding_1" not in view.completed_steps
        assert view.current_step_id == "coding_1"

    def test_bare_form_capability_match_completes(self):
        """A receipt with a namespaced capability id matches the bare form."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="Implemented.",
                # The plan says _writes: coding_apply_patch_ but the
                # receipt carries the namespaced form. The bare-form
                # match should still complete the step.
                effect_receipts=[
                    _receipt("lisa-storage.coding_apply_patch", "write"),
                ],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "coding_1" in view.completed_steps

    def test_external_write_receipt_completes_write_step(self):
        """An ``external_write`` effect satisfies the write contract."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            _entry(
                mode_id="coding",
                response_text="Implemented via external tool.",
                effect_receipts=[
                    _receipt("coding_apply_patch", "external_write"),
                ],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "coding_1" in view.completed_steps

    def test_write_step_in_wrong_mode_does_not_complete(self):
        """A receipt for the right capability but in the wrong mode does not complete."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP),
            _entry(mode_id="search", response_text="Searched.", handoff_to="coding"),
            # The receipt is in 'markdown' mode, not 'coding' mode.
            _entry(
                mode_id="markdown",
                response_text="Oops, wrong mode.",
                effect_receipts=[_receipt("coding_apply_patch", "write")],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "coding_1" not in view.completed_steps

    def test_all_write_steps_complete_workflow_returns_none(self):
        """When every step (including terminal write steps) is satisfied,
        the workflow is done and ``derive_workflow_view`` returns ``None``."""
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
                effect_receipts=[_receipt("markdown_generate", "external_write")],
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is None


# ── Regression: a turn can span several modes ─────────────────────────


class TestReceiptModeAttribution:
    """Completion is scoped by the mode that *dispatched* each receipt.

    A single thread entry can span several modes: a turn enters in one mode,
    hands off mid-turn, and hands off again. The entry records only the entry
    mode, so scoping a receipt by the entry's ``mode_id`` mislabels work done
    by a later mode in the same turn. In the operator's own history, 184 of
    203 handoff-bearing turns carried receipts and 115 of those carried write
    receipts — all attributable to the entry mode under the old rule.
    """

    _PLAN = """\
## Plan

1. **coding_review** — Review the change
   _reads_
2. **coding_execute** — Commit it
   _writes: coding_commit_
"""

    def test_receipt_from_a_later_mode_in_the_same_turn_completes_the_step(self):
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            _entry(
                mode_id="coding_review",
                response_text="Reviewed.",
                handoff_to="thinker",
                turn=1,
            ),
            _entry(
                mode_id="thinker",
                response_text="Did the whole chain in one turn.",
                effect_receipts=[
                    {
                        "capability_id": "coding_commit",
                        "effect": "write",
                        "mode_id": "coding_execute",
                    }
                ],
                turn=2,
            ),
        ]
        # Both steps satisfied -> the workflow is complete.
        assert derive_workflow_view(entries) is None

    def test_receipt_from_a_different_mode_does_not_complete_the_step(self):
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            _entry(
                mode_id="thinker",
                response_text="Committed from the wrong mode.",
                effect_receipts=[
                    {"capability_id": "coding_commit", "effect": "write", "mode_id": "thinker"}
                ],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "coding_execute_1" not in view.completed_steps

    def test_mode_and_capability_must_come_from_the_same_receipt(self):
        """One mode's receipt and another mode's capability must not combine."""
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            _entry(
                mode_id="thinker",
                response_text="Two unrelated writes.",
                effect_receipts=[
                    {"capability_id": "save_notes", "effect": "write", "mode_id": "coding_execute"},
                    {"capability_id": "coding_commit", "effect": "write", "mode_id": "thinker"},
                ],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries)
        assert view is not None
        assert "coding_execute_1" not in view.completed_steps

    def test_legacy_receipt_without_mode_falls_back_to_entry_mode(self):
        """Receipts persisted before the field existed still attribute."""
        plan = """\
## Plan

1. **coding_execute** — Commit it
   _writes: coding_commit_
"""
        entries = [
            _entry(mode_id="planner", response_text=plan, turn=0),
            _entry(
                mode_id="coding_execute",
                response_text="Committed.",
                effect_receipts=[{"capability_id": "coding_commit", "effect": "write"}],
                turn=1,
            ),
        ]
        assert derive_workflow_view(entries) is None


# ── Regression: the 2026-10-28 zombie plan ────────────────────────────


class TestPlanTurnCompletion:
    """Work done inside the plan-emitting turn completes its steps.

    Turn 1094 opened with a two-step plan (``thinker`` reads, then
    ``researcher`` writes) and executed both steps in the same turn. The
    view only scanned *later* entries, so neither step was credited and the
    plan re-attached itself to every turn after it.
    """

    _PLAN = """\
## Plan

1. **thinker** — Analyze the competitive landscape
   _reads_
2. **researcher** — Research incumbent strategies
   _writes: markdown_forge_document_

Interpretation: a competitive threat analysis.
"""

    def _plan_turn(self, **extra: object) -> dict:
        entry = _entry(
            mode_id="default",
            response_text=self._PLAN + "\nResearch complete.",
            handoff_to="researcher",
            effect_receipts=[
                {
                    "capability_id": "markdown_forge_document",
                    "effect": "write",
                    "mode_id": "researcher",
                }
            ],
            turn=1094,
        )
        entry.update(extra)
        return entry

    def test_plan_turn_that_runs_every_step_retires_the_plan(self):
        entry = self._plan_turn(departed_modes=["default", "planner", "thinker"])
        assert derive_workflow_view([entry]) is None

    def test_write_receipt_in_the_plan_turn_completes_its_step(self):
        view = derive_workflow_view([self._plan_turn()], idle_turn_limit=0)
        assert view is not None
        assert view.completed_steps == ("researcher_1",)
        assert view.declared_steps[1].completing_turn == 1094

    def test_departed_mode_mid_turn_completes_a_read_step(self):
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            _entry(
                mode_id="default",
                response_text="Thought it through.",
                handoff_to="coding",
                departed_modes=["default", "thinker"],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=0)
        assert view is not None
        assert view.completed_steps == ("thinker_0",)

    def test_departed_modes_replace_the_entry_mode_fallback(self):
        """A recorded trail is authoritative: the entry mode alone is not a departure."""
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            _entry(
                mode_id="thinker",
                response_text="Stayed in thinker.",
                handoff_to="",
                transitioned_to="",
                departed_modes=[],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=0)
        assert view is not None
        assert view.completed_steps == ()

    def test_one_receipt_cannot_complete_two_steps(self):
        plan = """\
## Plan

1. **coding** — First change
   _writes: coding_commit_
2. **coding** — Second change
   _writes: coding_commit_
"""
        entries = [
            _entry(mode_id="planner", response_text=plan, turn=0),
            _entry(
                mode_id="coding",
                response_text="Committed once.",
                effect_receipts=[_receipt("coding_commit")],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=0)
        assert view is not None
        assert view.completed_steps == ("coding_0",)

    def test_one_turn_can_complete_steps_in_several_modes(self):
        """Evidence is claimed per unit, so a multi-mode turn credits each mode."""
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            self._plan_turn(
                response_text="Did both.",
                departed_modes=["thinker"],
                turn=1,
            ),
        ]
        assert derive_workflow_view(entries) is None


class TestStepResultPreview:
    """A step's result preview is a single line, free of response structure."""

    _PLAN = """\
## Plan

1. **researcher** — Research it
   _writes: markdown_forge_document_
2. **thinker** — Reflect
   _reads_
"""

    def _view(self, response_text: str) -> WorkflowView:
        entries = [
            _entry(mode_id="planner", response_text=self._PLAN, turn=0),
            _entry(
                mode_id="researcher",
                response_text=response_text,
                effect_receipts=[_receipt("markdown_forge_document")],
                turn=1,
            ),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=0)
        assert view is not None
        return view

    def test_multiline_markdown_is_flattened(self):
        view = self._view("## Findings\n\n- one\n- two\n\n**Bold** close.")
        preview = view.step_results["researcher_0"]
        assert "\n" not in preview
        assert preview == "## Findings - one - two **Bold** close."
        rendered = render_workflow_progress(view)
        assert rendered.splitlines()[1] == f"[x] researcher: {preview} ← turn 1"

    def test_prior_progress_footer_is_dropped(self):
        view = self._view("Wrote it.\n\n## Workflow progress\n[ ] stale: old footer")
        assert view.step_results["researcher_0"] == "Wrote it."

    def test_long_preview_is_truncated_with_ellipsis(self):
        view = self._view("word " * 200)
        preview = view.step_results["researcher_0"]
        assert len(preview) == 300
        assert preview.endswith("…")

    def test_plan_block_of_the_plan_turn_is_dropped(self):
        entry = _entry(
            mode_id="default",
            response_text=self._PLAN + "\nResearch summary here.",
            effect_receipts=[
                {
                    "capability_id": "markdown_forge_document",
                    "effect": "write",
                    "mode_id": "researcher",
                }
            ],
            turn=7,
        )
        view = derive_workflow_view([entry], idle_turn_limit=0)
        assert view is not None
        assert view.step_results["researcher_0"] == "Research summary here."


class TestIdlePlanRetirement:
    """A plan that stops making progress retires after the idle limit."""

    def _entries(self, *responses: str) -> list[dict]:
        return [_entry(mode_id="planner", response_text=_PLAN_3_STEP, turn=0)] + [
            _entry(mode_id="default", response_text=text, turn=i + 1)
            for i, text in enumerate(responses)
        ]

    def test_plan_stays_active_below_the_limit(self):
        view = derive_workflow_view(self._entries("unrelated"), idle_turn_limit=2)
        assert view is not None

    def test_plan_retires_at_the_limit(self):
        entries = self._entries("unrelated", "also unrelated")
        assert derive_workflow_view(entries, idle_turn_limit=2) is None

    def test_idle_turns_count_from_the_latest_progress(self):
        entries = [
            *self._entries("unrelated"),
            _entry(mode_id="search", response_text="Found it.", handoff_to="coding", turn=2),
            _entry(mode_id="default", response_text="unrelated", turn=3),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=2)
        assert view is not None
        assert view.completed_steps == ("search_0",)

    def test_zero_limit_disables_retirement(self):
        entries = self._entries("a", "b", "c", "d")
        assert derive_workflow_view(entries, idle_turn_limit=0) is not None

    def test_default_limit_comes_from_settings(self, monkeypatch):
        from kernel.config import settings

        monkeypatch.setattr(settings, "workflow_plan_idle_turn_limit", 3)
        entries = self._entries("a", "b")
        assert derive_workflow_view(entries) is not None
        assert derive_workflow_view([*entries, _entry(response_text="c", turn=3)]) is None

    def test_turns_in_an_open_steps_mode_are_not_idle(self):
        """A step still being worked on keeps the plan alive.

        Regression: the idle rule counted every turn since the last
        completion, so a plan whose open step kept being worked on still
        retired once enough turns had passed. The rule is "idle turns" —
        turns that neither complete a step nor run in an open step's mode.
        """
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP, turn=0),
            _entry(mode_id="search", response_text="Still searching.", turn=1),
            _entry(mode_id="search", response_text="Still searching.", turn=2),
            _entry(mode_id="search", response_text="Still searching.", turn=3),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=2)
        assert view is not None
        assert view.completed_steps == ()

    def test_idle_run_breaks_when_an_open_steps_mode_returns(self):
        """An open step's mode resets the idle run rather than ending it."""
        entries = [
            _entry(mode_id="planner", response_text=_PLAN_3_STEP, turn=0),
            _entry(mode_id="default", response_text="unrelated", turn=1),
            _entry(mode_id="coding", response_text="Implementing.", turn=2),
            _entry(mode_id="default", response_text="unrelated", turn=3),
        ]
        view = derive_workflow_view(entries, idle_turn_limit=2)
        assert view is not None
