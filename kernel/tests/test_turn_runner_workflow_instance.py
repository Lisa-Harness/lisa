"""Phase 2 tests: wiring ``WorkflowInstance`` into the turn loop.

Covers the hydration of ``chain_satisfied`` from a durable instance at
turn start and the advancement of the instance on a successful
``transition_stage`` handoff — the fix for the glm-5.2 self-assessment
stall where ``coding`` was never reached after resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kernel.models import WorkflowModeConfig
from kernel.turn_runner import TurnHooks, run_turn
from kernel.workflow_instance import WorkflowInstance, WorkflowStep


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests (mirrors test_turn_runner)."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _build_minimal_hooks(complete_fn: Any) -> TurnHooks:
    return TurnHooks(complete=complete_fn)


def _mode_config(
    *,
    mode_id: str,
    satisfies_intents: list[str] | None = None,
    routing_intent_tags: list[str] | None = None,
    routing_intent_keywords: dict[str, list[str]] | None = None,
    routable: bool = True,
) -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=routable,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def _catalog_with_chain() -> dict[str, WorkflowModeConfig]:
    """A catalog for the search → coding → markdown chain."""
    return {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search"]},
        ),
        "coding": _mode_config(
            mode_id="coding",
            satisfies_intents=["coding"],
            routing_intent_tags=["coding"],
            routing_intent_keywords={"coding": ["code", "read"]},
        ),
        "markdown": _mode_config(
            mode_id="markdown",
            satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["write", "markdown"]},
        ),
    }


def _instance_with_chain(current_step_id: str = "search_0") -> WorkflowInstance:
    """An instance mid-flight: search done, coding is the current step."""
    return WorkflowInstance(
        plan_digest="abc",
        declared_steps=(
            WorkflowStep(step_id="search_0", mode_id="search", satisfies_intents=("research",)),
            WorkflowStep(
                step_id="coding_1",
                mode_id="coding",
                satisfies_intents=("coding",),
                depends_on=("search_0",),
            ),
            WorkflowStep(
                step_id="markdown_2",
                mode_id="markdown",
                satisfies_intents=("write",),
                depends_on=("coding_1",),
                is_terminal=True,
            ),
        ),
        completed_steps=("search_0",),
        current_step_id=current_step_id,
        current_mode_id="coding",
        spawned_from="adhoc",
    )


@pytest.mark.asyncio
async def test_chain_satisfied_hydrates_from_instance_completed_steps() -> None:
    """A resumed turn hydrates ``chain_satisfied`` from the instance.

    After default → search completes (search_0 done), the instance carries
    ``completed_steps=("search_0",)``. The next turn in coding mode must
    hydrate ``chain_satisfied`` with ``research`` (search's intent) so the
    routing hint names the remaining gap (write → markdown), not search
    again. This is the core fix for the glm-5.2 stall.
    """
    instance = _instance_with_chain(current_step_id="coding_1")
    system_prompts: list[str] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        # Capture the system prompt (messages[0]) to inspect the routing hint.
        msgs = kwargs.get("messages", [])
        if msgs and isinstance(msgs[0], dict):
            system_prompts.append(str(msgs[0].get("content", "")))
        return _StubCompletion(content="Done coding.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_instance = lambda: instance
    # No updater wired — we only test hydration here.

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="coding",
            query="search the web then read your code then write a markdown",
            hooks=hooks,
        )

    assert result.response_text == "Done coding."
    # The routing hint in the system prompt must NOT suggest `search` — the
    # `research` intent is already satisfied (hydrated from the instance's
    # completed search_0 step). It should name the remaining consumer gap.
    joined = "\n".join(system_prompts)
    # "search" as a handoff target appears in "Hand off to `search`"; the
    # word "research" is a different token and is allowed.
    assert "Hand off to `search`" not in joined, (
        "routing hint re-suggested search after the instance marked it complete"
    )


@pytest.mark.asyncio
async def test_instance_advanced_on_successful_handoff() -> None:
    """A ``transition_stage`` to a new mode advances the instance.

    Starting in coding (current_step_id=coding_1), handing off to markdown
    advances the instance: coding_1 is marked completed, current_step_id
    moves to markdown_2. The updater receives the advanced instance.
    """
    instance = _instance_with_chain(current_step_id="coding_1")
    updated_instances: list[WorkflowInstance] = []

    call_count = 0

    async def handoff_then_finish(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "markdown", "rationale": "code done"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done writing.")

    hooks = _build_minimal_hooks(handoff_then_finish)
    hooks.workflow_instance = lambda: instance

    def updater(updated: WorkflowInstance) -> None:
        updated_instances.append(updated)
        # Mutate in place so workflow_instance() sees the advanced state.
        object.__setattr__(instance, "__dict__", updated.__dict__)

    hooks.workflow_instance_updater = updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="coding",
            query="search the web then read your code then write a markdown",
            hooks=hooks,
        )

    assert result.handoff_to == "markdown"
    # The instance was advanced: coding_1 is completed, current_step_id is markdown_2.
    assert len(updated_instances) >= 1
    advanced = updated_instances[-1]
    assert "coding_1" in advanced.completed_steps
    assert advanced.current_step_id == "markdown_2"
    assert advanced.current_mode_id == "markdown"


@pytest.mark.asyncio
async def test_no_instance_wired_preserves_ephemeral_behavior() -> None:
    """When no workflow_instance hook is wired, the turn runs as before.

    This is the no-compat-path guarantee: the ephemeral ``chain_satisfied``
    behavior is the "no instance wired" case, not a separate code path.
    """
    hooks = _build_minimal_hooks(_make_stub("Hello."))
    # No workflow_instance / workflow_instance_updater wired.
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)
    assert result.response_text == "Hello."


def _make_stub(text: str) -> Any:
    async def _complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=text)

    return _complete


@pytest.mark.asyncio
async def test_adhoc_multistep_chain_creates_durable_instance() -> None:
    """A turn-1 multi-step ad-hoc chain persists a ``WorkflowInstance``.

    The fix for the 2026-08-01 yield-then-continue skip: when the routing
    hint detects a multi-step chain at turn 1 (search → coding → markdown)
    and no instance is already active, the turn runner compiles the chain
    into an ad-hoc ``WorkflowInstance`` (``spawned_from="adhoc"``) and
    persists it via ``workflow_instance_updater``. Without this, the
    ephemeral ``chain_satisfied`` set resets on the next turn and a
    yield → "continue" resume skips the middle step.
    """
    active_instance: list[WorkflowInstance | None] = [None]
    saved_instances: list[WorkflowInstance] = []

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    # Mirror the host wiring: ``workflow_instance`` returns the live active
    # ref (None at turn 1); the updater persists the advanced instance and
    # swaps the active ref in place.
    hooks.workflow_instance = lambda: active_instance[0]

    def _wi_updater(instance: WorkflowInstance) -> None:
        saved_instances.append(instance)
        active_instance[0] = instance

    hooks.workflow_instance_updater = _wi_updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        await run_turn(
            mode_id="default",
            query="search the web then read your code then write a markdown",
            hooks=hooks,
        )

    assert saved_instances, "an ad-hoc WorkflowInstance must be created for a multi-step chain"
    instance = saved_instances[0]
    assert instance.spawned_from == "adhoc"
    mode_ids = [s.mode_id for s in instance.declared_steps]
    assert mode_ids == ["search", "coding", "markdown"]
    assert instance.current_step_id == instance.declared_steps[0].step_id
    assert instance.current_mode_id == "search"


@pytest.mark.asyncio
async def test_adhoc_no_chain_does_not_create_instance() -> None:
    """A single-step request (no multi-step chain) creates no instance.

    Guards against over-eager instance creation for trivial queries that
    the current mode handles in-place. No routing gap → no instance.
    """
    active_instance: list[WorkflowInstance | None] = [None]
    saved_instances: list[WorkflowInstance] = []

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Hello.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_instance = lambda: active_instance[0]

    def _wi_updater(instance: WorkflowInstance) -> None:
        saved_instances.append(instance)
        active_instance[0] = instance

    hooks.workflow_instance_updater = _wi_updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert not saved_instances, "no WorkflowInstance should be created for a trivial query"


def _catalog_with_plan() -> dict[str, WorkflowModeConfig]:
    """A catalog with the search → coding → markdown chain plus plan mode.

    Used by the plan-escalation guard test: the 3-intent sequential query
    triggers plan escalation (``plan`` is a routable target), so the ad-hoc
    block must skip — ``plan`` owns the instance.
    """
    catalog = _catalog_with_chain()
    catalog["plan"] = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    return catalog


@pytest.mark.asyncio
async def test_adhoc_skipped_when_plan_escalation_fires() -> None:
    """When the routing hint is a plan-escalation, no ad-hoc instance is created.

    The 2026-08-01 overlap bug: a 3-intent sequential query triggers both
    plan escalation (the hint says "hand off to `plan`") AND the ad-hoc
    instance creation block. The ad-hoc instance is orphaned — the plan
    instance overwrites it via the updater moments later. The fix guards
    the ad-hoc block with ``is_plan_escalation_hint`` so it skips when
    plan owns the instance.
    """
    from kernel.routing_vocabulary import reset_plan_reliability

    active_instance: list[WorkflowInstance | None] = [None]
    saved_instances: list[WorkflowInstance] = []

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_instance = lambda: active_instance[0]

    def _wi_updater(instance: WorkflowInstance) -> None:
        saved_instances.append(instance)
        active_instance[0] = instance

    hooks.workflow_instance_updater = _wi_updater

    catalog = _catalog_with_plan()
    reset_plan_reliability()
    try:
        with (
            patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
            patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
            patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        ):
            await run_turn(
                mode_id="default",
                query="search the web then read your code then write a markdown",
                hooks=hooks,
            )
    finally:
        reset_plan_reliability()

    # No ad-hoc instance: plan escalation owns the instance. The ad-hoc
    # block must have skipped (the hint was a plan-escalation hint).
    adhoc_instances = [i for i in saved_instances if i.spawned_from == "adhoc"]
    assert not adhoc_instances, (
        "ad-hoc instance must not be created when plan escalation fires "
        f"(got {len(adhoc_instances)} ad-hoc instance(s))"
    )
