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
    compile_steps_from_routing_targets,
    is_complete,
    remaining_steps,
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


# ── Step compiler ───────────────────────────────────────────────────────────


def test_compile_steps_sequential_producer_before_consumer() -> None:
    """A sequential query orders producer intents before consumer intents.

    This is the glm-5.2 self-assessment regression case: "search ... then
    read your code ... then write a markdown" must compile to
    search → coding → markdown, with producers first so the consumer
    (coding/markdown) is reachable before the iteration budget runs out.
    """
    from unittest.mock import patch

    catalog = _catalog()
    query = "search the web for frameworks then read your code then write a markdown doc"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        steps = compile_steps_from_routing_targets(
            query=query,
            routing_targets=["search", "ponder", "coding", "markdown"],
            catalog=catalog,
        )
    mode_ids = [s.mode_id for s in steps]
    # Producers (search) before consumers (coding, markdown).
    assert mode_ids[0] == "search"
    assert "coding" in mode_ids
    assert "markdown" in mode_ids
    assert mode_ids[-1] == "markdown"  # terminal step is the last consumer
    assert steps[-1].is_terminal is True
    # Each step depends on the one before it (sequential chain).
    for i, step in enumerate(steps[1:], start=1):
        assert steps[i - 1].step_id in step.depends_on


def test_compile_steps_returns_empty_when_no_intents_detected() -> None:
    """A query with no detectable routing intents compiles to no steps."""
    from unittest.mock import patch

    catalog = _catalog()
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        steps = compile_steps_from_routing_targets(
            query="hello",
            routing_targets=["search", "coding"],
            catalog=catalog,
        )
    assert steps == ()


def test_compile_returns_empty_when_no_routing_targets() -> None:
    steps = compile_steps_from_routing_targets(
        query="search for frameworks",
        routing_targets=[],
        catalog={
            "search": _mode_config(
                mode_id="search",
                routing_intent_tags=["research"],
                routing_intent_keywords={"research": ["search"]},
            )
        },
    )
    assert steps == ()


def test_compile_steps_step_ids_are_stable_across_recompiles() -> None:
    """Recompiling the same query produces the same step ids (resume stability)."""
    from unittest.mock import patch

    catalog = _catalog()
    query = "search the web then write a markdown"
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        steps1 = compile_steps_from_routing_targets(
            query=query,
            routing_targets=["search", "markdown"],
            catalog=catalog,
        )
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        steps2 = compile_steps_from_routing_targets(
            query=query,
            routing_targets=["search", "markdown"],
            catalog=catalog,
        )
    assert [s.step_id for s in steps1] == [s.step_id for s in steps2]


def test_compile_steps_spawned_from_propagated() -> None:
    """The spawned_from origin is set on every compiled step."""
    from unittest.mock import patch

    catalog = _catalog()
    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        steps = compile_steps_from_routing_targets(
            query="search the web then write a markdown",
            routing_targets=["search", "markdown"],
            catalog=catalog,
            spawned_from="plan_workflow_op",
        )
    assert all(s.spawned_from == "plan_workflow_op" for s in steps)


def test_plan_steps_to_workflow_steps_shape_matches_ad_hoc_compiler() -> None:
    """``_plan_steps_to_workflow_steps`` produces the same field shape as
    ``compile_steps_from_routing_targets`` so downstream consumers can be
    origin-blind.

    Both paths must populate ``satisfies_intents`` (from the catalog) and
    ``depends_on`` (the linear prefix of prior step ids), mark the last
    step ``is_terminal``, and use the same ``_step_id_for`` scheme. The
    only legitimate difference is ``spawned_from``. This is the
    foundation for dropping the ``spawned_from`` gates in the turn loop
    (Phase 2 of the unification plan).
    """
    from unittest.mock import patch

    from kernel.turn_runner import _plan_steps_to_workflow_steps

    catalog = _catalog()
    plan_steps = [
        {"mode_id": "search", "description": "Search the web", "rationale": "research"},
        {"mode_id": "markdown", "description": "Write a summary", "rationale": "write"},
    ]
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog):
        planned = _plan_steps_to_workflow_steps(plan_steps)

    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog):
        adhoc = compile_steps_from_routing_targets(
            query="search the web then write a markdown",
            routing_targets=["search", "markdown"],
            catalog=catalog,
            spawned_from="adhoc",
        )

    assert len(planned) == len(adhoc) == 2
    for planned_step, adhoc_step in zip(planned, adhoc, strict=True):
        # step_id uses the same _step_id_for scheme (mode + index).
        assert planned_step.step_id == adhoc_step.step_id, (
            f"step_id mismatch: {planned_step.step_id} vs {adhoc_step.step_id}"
        )
        assert planned_step.mode_id == adhoc_step.mode_id
        # satisfies_intents populated from the catalog in both paths.
        assert planned_step.satisfies_intents == adhoc_step.satisfies_intents, (
            f"satisfies_intents mismatch for {planned_step.mode_id}: "
            f"{planned_step.satisfies_intents} vs {adhoc_step.satisfies_intents}"
        )
        # depends_on is the linear prefix of prior step ids in both paths.
        assert planned_step.depends_on == adhoc_step.depends_on, (
            f"depends_on mismatch for {planned_step.mode_id}: "
            f"{planned_step.depends_on} vs {adhoc_step.depends_on}"
        )
        assert planned_step.is_terminal == adhoc_step.is_terminal
    # The only legitimate difference is spawned_from.
    assert all(s.spawned_from == "plan_workflow_op" for s in planned)
    assert all(s.spawned_from == "adhoc" for s in adhoc)


def test_is_terminal_stage_returns_false_for_empty_stage_id() -> None:
    """``is_terminal_stage`` returns False (not raises ``KeyError: ''``)
    when the position has an empty ``stage_id``.

    Regression for the 2026-08-10 ``KeyError: ''`` crash: an action/chat
    mode (e.g. ``researcher``) synthesized from a thread-derived view
    with no ``transitioned_to`` carries ``stage_id=""``. The old
    implementation did ``mode.stages[position.stage_id].terminal`` and
    raised ``KeyError: ''`` because ``""`` is not a key in
    ``mode.stages``. The guard returns False — an absent intra-mode
    pointer is not a terminal stage.
    """
    from unittest.mock import patch

    from kernel.ipsar_fsm import is_terminal_stage
    from kernel.models import WorkflowStageConfig

    # A synthesized action mode: one stage keyed by the mode_id, NOT "".
    mode_cfg = WorkflowModeConfig(
        mode_id="researcher",
        start_stage_id="researcher",
        mode_kind="action",
        routable=True,
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        stages={"researcher": WorkflowStageConfig(stage_id="researcher", terminal=False)},
    )
    catalog = {"researcher": mode_cfg}

    # Position parked in researcher with no intra-mode stage pointer.
    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="researcher_1", mode_id="researcher", is_terminal=True),
        ),
        current_step_id="researcher_1",
        mode_id="researcher",
        stage_id="",
        spawned_from="thread_derived",
    )

    with patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog):
        result = is_terminal_stage(position)

    assert result is False


def test_is_terminal_stage_returns_false_for_unknown_stage_id() -> None:
    """``is_terminal_stage`` returns False when ``stage_id`` names a stage
    not present in the mode's stage graph (a stale/derived pointer),
    instead of raising ``KeyError``.
    """
    from unittest.mock import patch

    from kernel.ipsar_fsm import is_terminal_stage
    from kernel.models import WorkflowStageConfig

    mode_cfg = WorkflowModeConfig(
        mode_id="ponder",
        start_stage_id="ponder",
        mode_kind="workflow",
        routable=True,
        satisfies_intents=["reasoning"],
        routing_intent_tags=["reasoning"],
        stages={
            "ponder": WorkflowStageConfig(stage_id="ponder", terminal=False),
            "exit": WorkflowStageConfig(stage_id="exit", terminal=True),
        },
    )
    catalog = {"ponder": mode_cfg}

    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="ponder_1", mode_id="ponder", is_terminal=True),
        ),
        current_step_id="ponder_1",
        mode_id="ponder",
        stage_id="stale_stage_not_in_graph",
        spawned_from="thread_derived",
    )

    with patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog):
        result = is_terminal_stage(position)

    assert result is False


def test_is_terminal_stage_returns_true_for_terminal_stage() -> None:
    """``is_terminal_stage`` returns True when the position points at a
    real terminal stage (the happy path preserved by the guard fix).
    """
    from unittest.mock import patch

    from kernel.ipsar_fsm import is_terminal_stage
    from kernel.models import WorkflowStageConfig

    mode_cfg = WorkflowModeConfig(
        mode_id="ponder",
        start_stage_id="ponder",
        mode_kind="workflow",
        routable=True,
        satisfies_intents=["reasoning"],
        routing_intent_tags=["reasoning"],
        stages={
            "ponder": WorkflowStageConfig(stage_id="ponder", terminal=False),
            "exit": WorkflowStageConfig(stage_id="exit", terminal=True),
        },
    )
    catalog = {"ponder": mode_cfg}

    position = WorkflowPosition(
        declared_steps=(
            WorkflowStep(step_id="ponder_1", mode_id="ponder", is_terminal=True),
        ),
        current_step_id="ponder_1",
        mode_id="ponder",
        stage_id="exit",
        spawned_from="thread_derived",
    )

    with patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog):
        result = is_terminal_stage(position)

    assert result is True
