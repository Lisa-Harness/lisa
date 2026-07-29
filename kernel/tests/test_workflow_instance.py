"""Tests for the durable ``WorkflowInstance`` model, helpers, and store.

Covers Phase 1 of the durable-workflow-instance plan: model round-trip,
step ordering (producer before consumer), ``advance_step`` immutability,
``is_complete``, the ad-hoc step compiler, and the file-backed store's
save/load/load_active/clear semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel.recovery import RecoveryIdentity
from kernel.routing_vocabulary import reset_routing_vocabulary
from kernel.workflow_instance import (
    WorkflowInstance,
    WorkflowStep,
    advance_step,
    compile_steps_from_routing_targets,
    is_complete,
    remaining_steps,
)
from kernel.workflow_instance_store import FileWorkflowInstanceStore
from kernel.models import WorkflowModeConfig


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
    """Reset routing vocabulary to defaults around each test."""
    reset_routing_vocabulary()
    yield
    reset_routing_vocabulary()


# ── Model + helpers ─────────────────────────────────────────────────────────


def test_workflow_instance_model_round_trip() -> None:
    """An instance serializes and re-validates without loss."""
    instance = WorkflowInstance(
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
        current_mode_id="search",
        spawned_from="adhoc",
    )
    data = instance.model_dump(mode="json")
    restored = WorkflowInstance.model_validate(data)
    assert restored == instance
    assert restored.instance_id == instance.instance_id


def test_workflow_instance_is_frozen() -> None:
    """The instance is immutable; model_copy is the only mutation path."""
    instance = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="s_0", mode_id="search"),),
        current_step_id="s_0",
    )
    with pytest.raises(Exception):  # noqa: PT011, B017
        instance.current_step_id = "other"  # type: ignore[misc]


def test_advance_step_marks_completed_and_advances_pointer() -> None:
    """Advancing a step adds it to completed_steps and moves the pointer."""
    instance = WorkflowInstance(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding", is_terminal=True),
        ),
        current_step_id="search_0",
        current_mode_id="search",
    )
    advanced = advance_step(instance, "search_0", result_preview="found 3 frameworks")
    assert "search_0" in advanced.completed_steps
    assert advanced.current_step_id == "coding_1"
    assert advanced.current_mode_id == "coding"
    assert advanced.step_results["search_0"] == "found 3 frameworks"
    # Immutability: the original is unchanged.
    assert instance.completed_steps == ()
    assert instance.current_step_id == "search_0"


def test_advance_step_is_idempotent() -> None:
    """Re-advancing an already-completed step is a no-op for the pointer."""
    instance = WorkflowInstance(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding", is_terminal=True),
        ),
        current_step_id="coding_1",
        current_mode_id="coding",
        completed_steps=("search_0",),
    )
    advanced = advance_step(instance, "search_0")
    # search_0 is already completed; pointer stays on coding_1.
    assert advanced.completed_steps == ("search_0",)
    assert advanced.current_step_id == "coding_1"


def test_advance_step_last_step_clears_pointer() -> None:
    """Advancing the last remaining step clears the current pointer."""
    instance = WorkflowInstance(
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search"),
            WorkflowStep(step_id="coding_1", mode_id="coding", is_terminal=True),
        ),
        current_step_id="coding_1",
        current_mode_id="coding",
        completed_steps=("search_0",),
    )
    advanced = advance_step(instance, "coding_1")
    assert is_complete(advanced)
    assert advanced.current_step_id == ""
    assert advanced.current_mode_id == ""


def test_advance_step_unknown_step_is_noop() -> None:
    """Advancing a step id not in declared_steps returns the instance unchanged."""
    instance = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="search_0", mode_id="search"),),
        current_step_id="search_0",
    )
    advanced = advance_step(instance, "nonexistent")
    assert advanced == instance


def test_remaining_steps_excludes_completed() -> None:
    """remaining_steps returns only steps not in completed_steps."""
    instance = WorkflowInstance(
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
    instance = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="a_0", mode_id="a"),),
        completed_steps=("a_0",),
    )
    assert is_complete(instance)


def test_is_complete_false_when_steps_remain() -> None:
    instance = WorkflowInstance(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b"),
        ),
        completed_steps=("a_0",),
    )
    assert not is_complete(instance)


def test_is_complete_true_for_empty_declared_steps() -> None:
    """An instance with no declared steps is trivially complete."""
    assert is_complete(WorkflowInstance())


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


# ── FileWorkflowInstanceStore ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_save_load_round_trip(tmp_path: Path) -> None:
    """An instance saved with an identity loads back for that identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="search_0", mode_id="search"),),
        current_step_id="search_0",
        current_mode_id="search",
    )
    identity = _identity()
    await store.save(instance, identity)
    loaded = await store.load(instance.instance_id, identity)
    assert loaded is not None
    assert loaded.instance_id == instance.instance_id
    assert loaded.current_mode_id == "search"


@pytest.mark.asyncio
async def test_store_load_returns_none_for_wrong_identity(tmp_path: Path) -> None:
    """An instance does not load for a different identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(current_mode_id="search")
    await store.save(instance, _identity())
    other = RecoveryIdentity(
        workspace_id="other",
        user_id="other",
        user_slug="other",
        surface_type="personal_dm",
        surface_id="other",
        memory_scope="personal",
        session_plan_digest="digest",
    )
    assert await store.load(instance.instance_id, other) is None


@pytest.mark.asyncio
async def test_store_load_active_returns_non_complete_instance(tmp_path: Path) -> None:
    """load_active returns the single non-complete instance for an identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    active = WorkflowInstance(
        declared_steps=(
            WorkflowStep(step_id="a_0", mode_id="a"),
            WorkflowStep(step_id="b_1", mode_id="b", is_terminal=True),
        ),
        current_step_id="b_1",
        completed_steps=("a_0",),
    )
    complete = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="x_0", mode_id="x"),),
        completed_steps=("x_0",),
    )
    identity = _identity()
    await store.save(active, identity)
    await store.save(complete, identity)
    loaded = await store.load_active(identity)
    assert loaded is not None
    assert loaded.instance_id == active.instance_id


@pytest.mark.asyncio
async def test_store_load_active_returns_none_when_all_complete(tmp_path: Path) -> None:
    """load_active returns None when every instance for the identity is complete."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    complete = WorkflowInstance(
        declared_steps=(WorkflowStep(step_id="x_0", mode_id="x"),),
        completed_steps=("x_0",),
    )
    identity = _identity()
    await store.save(complete, identity)
    assert await store.load_active(identity) is None


@pytest.mark.asyncio
async def test_store_clear_removes_instance(tmp_path: Path) -> None:
    """clear removes an instance for its identity and returns True."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(current_mode_id="search")
    identity = _identity()
    await store.save(instance, identity)
    assert await store.clear(instance.instance_id, identity) is True
    assert await store.load(instance.instance_id, identity) is None
    # Clearing again is False (already gone).
    assert await store.clear(instance.instance_id, identity) is False


@pytest.mark.asyncio
async def test_store_clear_rejects_wrong_identity(tmp_path: Path) -> None:
    """clear returns False for a non-matching identity."""
    store = FileWorkflowInstanceStore(tmp_path / "workflow-instances.json")
    instance = WorkflowInstance(current_mode_id="search")
    await store.save(instance, _identity())
    other = RecoveryIdentity(
        workspace_id="other",
        user_id="other",
        user_slug="other",
        surface_type="personal_dm",
        surface_id="other",
        memory_scope="personal",
        session_plan_digest="digest",
    )
    assert await store.clear(instance.instance_id, other) is False


@pytest.mark.asyncio
async def test_store_survives_restart(tmp_path: Path) -> None:
    """A new store instance at the same path loads persisted instances."""
    path = tmp_path / "workflow-instances.json"
    store1 = FileWorkflowInstanceStore(path)
    instance = WorkflowInstance(current_mode_id="search")
    identity = _identity()
    await store1.save(instance, identity)
    # Simulate restart: new store object, same file.
    store2 = FileWorkflowInstanceStore(path)
    loaded = await store2.load(instance.instance_id, identity)
    assert loaded is not None
    assert loaded.current_mode_id == "search"