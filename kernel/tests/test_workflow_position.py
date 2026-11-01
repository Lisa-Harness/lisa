"""Tests for the durable ``WorkflowPosition`` model, helpers, and store.

Covers Phase 1 of the durable-workflow-instance plan: model round-trip,
step ordering (producer before consumer), ``advance_step`` immutability,
``is_complete``, the ad-hoc step compiler, and the file-backed store's
save/load/load_active/clear semantics.
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.models import WorkflowModeConfig
from kernel.recovery import RecoveryIdentity
from kernel.routing_vocabulary import reset_routing_vocabulary
from kernel.workflow_position import (
    WorkflowPosition,
    WorkflowStep,
    advance_step,
    is_complete,
    next_step_after_pointer,
    remaining_steps,
    resumed_step,
)


def _identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope="personal",
        session_plan_digest="digest",
    )


def _mode_config(
    *,
    mode_id: str,
    satisfies_intents: list[str] | None = None,
    routing_intent_tags: list[str] | None = None,
    routing_intent_keywords: dict[str, list[str]] | None = None,
    routable: bool = True,
) -> WorkflowModeConfig:
    """A minimal WorkflowModeConfig for step-compiler tests (mirrors test_turn_runner)."""
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=routable,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def _catalog() -> dict[str, WorkflowModeConfig]:
    """A catalog with producer (search/ponder) and consumer (coding/markdown) modes."""
    return {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search", "latest", "web"]},
        ),
        "ponder": _mode_config(
            mode_id="ponder",
            satisfies_intents=["research", "reasoning"],
            routing_intent_tags=["research", "reasoning"],
            routing_intent_keywords={"research": ["framework"], "reasoning": ["analyze", "swot"]},
        ),
        "coding": _mode_config(
            mode_id="coding",
            satisfies_intents=["coding", "review"],
            routing_intent_tags=["coding", "review"],
            routing_intent_keywords={"coding": ["code", "read", "implement"], "review": ["review"]},
        ),
        "markdown": _mode_config(
            mode_id="markdown",
            satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["write", "markdown", "document"]},
        ),
    }


@pytest.fixture(autouse=True)
def _reset_vocabulary() -> Any:
    """Reset and re-register routing vocabulary around each test.

    The kernel has no hardcoded fallbacks (2026-08-11 cutover); the
    workflow-position helpers read producer/consumer intents, so a
    representative vocabulary must be wired or the reader raises.
    """
    from kernel.routing_vocabulary import register_routing_vocabulary

    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by"],
        producer_intents=["research", "gather", "learn", "read", "analyze", "review", "architect"],
        consumer_intents=["code", "write", "implement", "review"],
        resume_phrases=["proceed", "continue", "go ahead"],
    )
    yield
    reset_routing_vocabulary()


# ── Model + helpers ─────────────────────────────────────────────────────────


def test_workflow_instance_model_round_trip() -> None:
    """An instance serializes and re-validates without loss."""
    instance = WorkflowPosition(
        plan_digest="abc",
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
        current_step_id="search_0",
        mode_id="search",
        spawned_from="adhoc",
    )
    data = instance.model_dump(mode="json")
    restored = WorkflowPosition.model_validate(data)
    assert restored == instance
    assert restored.position_id == instance.position_id


def test_workflow_instance_is_frozen() -> None:
    """The instance is immutable; model_copy is the only mutation path."""
    instance = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="s_0", mode_id="search"),),
        current_step_id="s_0",
    )
    with pytest.raises(Exception):  # noqa: B017
        instance.current_step_id = "other"  # type: ignore[misc]


def test_advance_step_marks_completed_and_advances_pointer() -> None:
    """Advancing a step adds it to completed_steps and moves the step pointer.

    The mode pointer (``mode_id``) does NOT move — mode moves are
    ``enter_mode``'s concern (kernel/ipsar_fsm.py), applied by the turn
    loop's handoff continuation after the step advance.
    """
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding", is_terminal=True),
        ),
        current_step_id="search_0",
        mode_id="search",
    )
    advanced = advance_step(position, "search_0", result_preview="found 3 frameworks")
    assert "search_0" in advanced.completed_steps
    assert advanced.current_step_id == "coding_1"
    assert advanced.mode_id == "search"  # mode moves via enter_mode, not advance_step
    assert not advanced.terminal_reached
    assert advanced.step_results["search_0"] == "found 3 frameworks"
    # Immutability: the original is unchanged.
    assert position.completed_steps == ()
    assert position.current_step_id == "search_0"


def test_advance_step_is_idempotent() -> None:
    """Re-advancing an already-completed step is a no-op for the pointer."""
    instance = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding", is_terminal=True),
        ),
        current_step_id="coding_1",
        mode_id="coding",
        completed_steps=("search_0",),
    )
    advanced = advance_step(instance, "search_0")
    # search_0 is already completed; pointer stays on coding_1.
    assert advanced.completed_steps == ("search_0",)
    assert advanced.current_step_id == "coding_1"


def test_advance_step_last_step_clears_pointer() -> None:
    """Advancing the terminal step clears the step pointer and marks the chain done.

    Completing a step declared ``is_terminal`` sets ``terminal_reached`` so
    the host store clears the finished position instead of leaving an
    all-done zombie active. The mode pointer survives (the position is
    still parked in the last mode until the host clears it).
    """
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding", is_terminal=True),
        ),
        current_step_id="coding_1",
        mode_id="coding",
        completed_steps=("search_0",),
    )
    advanced = advance_step(position, "coding_1")
    assert is_complete(advanced)
    assert advanced.current_step_id == ""
    assert advanced.terminal_reached
    assert advanced.mode_id == "coding"


def test_advance_step_unknown_step_is_noop() -> None:
    """Advancing a step id not in declared_steps returns the instance unchanged."""
    instance = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="search_0", mode_id="search"),),
        current_step_id="search_0",
    )
    advanced = advance_step(instance, "nonexistent")
    assert advanced == instance


def test_remaining_steps_excludes_completed() -> None:
    """remaining_steps returns only steps not in completed_steps."""
    instance = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding"),
            WorkflowStep(step_id="markdown_2", mode_id="markdown", is_terminal=True),
        ),
        completed_steps=("search_0",),
    )
    remaining = remaining_steps(instance)
    assert [s.step_id for s in remaining] == ["coding_1", "markdown_2"]


def test_is_complete_when_all_steps_completed() -> None:
    """is_complete is True when completed_steps covers all declared steps."""
    instance = WorkflowPosition(
        declared_steps=(WorkflowStep(step_id="a_0", mode_id="a"),),
        completed_steps=("a_0",),
    )
    assert is_complete(instance)


def test_is_complete_false_when_steps_remain() -> None:
    instance = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b"),
        ),
        completed_steps=("a_0",),
    )
    assert not is_complete(instance)


def test_is_complete_true_for_empty_declared_steps() -> None:
    """An instance with no declared steps is trivially complete."""
    assert is_complete(WorkflowPosition())


# ── Front-aware next-step selection ──────────────────────────────────────


def test_advance_step_out_of_order_completion_clears_pointer() -> None:
    """Completing a step when only earlier steps remain open clears the pointer.

    Regression lock for the 2026-10-30 backward-routing loop: completing
    the tail step while an earlier step was abandoned must not park the
    pointer on the abandoned step — the old ``remaining[0]`` fallback did
    exactly that, and every hint then routed a later turn backward to it.
    """
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="thinker_0", mode_id="thinker"),
            WorkflowStep(step_id="researcher_1", mode_id="researcher", is_terminal=True),
        ),
        current_step_id="researcher_1",
        mode_id="researcher",
        completed_steps=(),
    )
    advanced = advance_step(position, "researcher_1")
    assert "researcher_1" in advanced.completed_steps
    # thinker_0 is open but declared before the completed step: the
    # pointer clears instead of moving backward.
    assert advanced.current_step_id == ""
    assert "thinker_0" not in advanced.completed_steps


def test_next_step_after_pointer_skips_abandoned_earlier_step() -> None:
    """On-plan pointer at the tail yields no next step — never the abandoned one.

    The pointer names the last declared step; the only open step precedes
    it. Routing must go silent (None) rather than name the abandoned
    step (the 2026-10-30 t1107/t1108 loop: the hint said `thinker` and
    the enforcer forced it twice per turn over the model's refusal).
    """
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="thinker_0", mode_id="thinker"),
            WorkflowStep(step_id="researcher_1", mode_id="researcher"),
        ),
        current_step_id="researcher_1",
        completed_steps=("researcher_1",),
    )
    assert next_step_after_pointer(position) is None


def test_next_step_after_pointer_names_first_forward_open_step() -> None:
    """The step after the pointer wins even when an earlier step is open."""
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b"),
            WorkflowStep(step_id="c_2", mode_id="c"),
        ),
        current_step_id="a_0",
        completed_steps=("b_1",),
    )
    assert next_step_after_pointer(position) is not None
    assert next_step_after_pointer(position).step_id == "c_2"


def test_next_step_after_pointer_empty_pointer_is_none() -> None:
    """An absent pointer with open steps means they were abandoned."""
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b"),
        ),
        current_step_id="",
        completed_steps=("b_1",),
    )
    assert next_step_after_pointer(position) is None


def test_next_step_after_pointer_off_plan_falls_back_to_first_remaining() -> None:
    """A pointer naming no declared step has no front — first open step is the guess."""
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b"),
        ),
        current_step_id="other_9",
        completed_steps=(),
    )
    assert next_step_after_pointer(position) is not None
    assert next_step_after_pointer(position).step_id == "a_0"


def test_resumed_step_returns_open_pointer() -> None:
    """A resume continues the step in flight — the open pointer step itself.

    Routing must never name the pointer step (forward-only), but resuming
    re-enters the work in flight: the pointer names an open step and that
    step is the resume target (re-entering the chain's originating mode
    must not re-run the planner against a continuation query).
    """
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="markdown_1", mode_id="markdown", is_terminal=True),
        ),
        completed_steps=("search_0",),
        current_step_id="markdown_1",
    )
    step = resumed_step(position)
    assert step is not None
    assert step.step_id == "markdown_1"


def test_resumed_step_completed_pointer_falls_forward() -> None:
    """A completed pointer resumes at the first forward open step."""
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding"),
            WorkflowStep(step_id="markdown_2", mode_id="markdown"),
        ),
        completed_steps=("search_0", "coding_1"),
        current_step_id="coding_1",
    )
    step = resumed_step(position)
    assert step is not None
    assert step.step_id == "markdown_2"


def test_resumed_step_spent_chain_is_none() -> None:
    """An empty pointer with only pre-frontier steps open resumes nothing.

    Regression lock for the 2026-10-30 t1107/t1108 loop's spent position:
    researcher completed, thinker abandoned, pointer empty. A resume must
    not re-open the abandoned step.
    """
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="thinker_0", mode_id="thinker"),
            WorkflowStep(step_id="researcher_1", mode_id="researcher"),
        ),
        completed_steps=("researcher_1",),
        current_step_id="",
    )
    assert resumed_step(position) is None
