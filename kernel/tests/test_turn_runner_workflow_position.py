"""Phase 2 tests: wiring ``WorkflowPosition`` into the turn loop.

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
from kernel.turn_hooks import (
    ModelHooks,
)
from kernel.turn_runner import TurnHooks, run_turn
from kernel.workflow_position import WorkflowPosition, WorkflowStep


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests (mirrors test_turn_runner)."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _build_minimal_hooks(complete_fn: Any) -> TurnHooks:
    return TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
        ),
    )


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


def _instance_with_chain(current_step_id: str = "search_0") -> WorkflowPosition:
    """An instance mid-flight: search done, coding is the current step."""
    return WorkflowPosition(
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
        mode_id="coding",
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
    hooks.workflow.workflow_position = lambda: instance
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

    assert result.response_text.startswith("Done coding.")
    # Origin-blind progress rendering now appends a workflow-progress block
    # for ad-hoc instances too (Phase 2 unification). The test's purpose is
    # the chain_satisfied hydration (verified via the routing hint below),
    # not the exact response text, so we only assert the prefix.
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
    updated_instances: list[WorkflowPosition] = []

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
                            "arguments": (
                                '{"target_stage_id": "markdown", "rationale": "code done"}'
                            ),
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done writing.")

    hooks = _build_minimal_hooks(handoff_then_finish)
    hooks.workflow.workflow_position = lambda: instance

    def updater(updated: WorkflowPosition) -> None:
        updated_instances.append(updated)
        # Mutate in place so workflow_position() sees the advanced state.
        object.__setattr__(instance, "__dict__", updated.__dict__)

    hooks.workflow.workflow_position_updater = updater

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
    assert advanced.mode_id == "markdown"


@pytest.mark.asyncio
async def test_no_instance_wired_preserves_ephemeral_behavior() -> None:
    """When no workflow_position hook is wired, the turn runs as before.

    This is the no-compat-path guarantee: the ephemeral ``chain_satisfied``
    behavior is the "no instance wired" case, not a separate code path.
    """
    hooks = _build_minimal_hooks(_make_stub("Hello."))
    # No workflow_position / workflow_position_updater wired.
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)
    assert result.response_text == "Hello."


def _make_stub(text: str) -> Any:
    async def _complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=text)

    return _complete


@pytest.mark.asyncio
async def test_adhoc_multistep_chain_creates_durable_instance() -> None:
    """A turn-1 multi-step ad-hoc chain persists a ``WorkflowPosition``.

    The fix for the 2026-08-01 yield-then-continue skip: when the routing
    hint detects a multi-step chain at turn 1 (search → coding → markdown)
    and no instance is already active, the turn runner compiles the chain
    into an ad-hoc ``WorkflowPosition`` (``spawned_from="adhoc"``) and
    persists it via ``workflow_position_updater``. Without this, the
    ephemeral ``chain_satisfied`` set resets on the next turn and a
    yield → "continue" resume skips the middle step.

    2026-08-06 zombie-instance fix: persistence is deferred to the first
    successful ``transition_stage`` handoff. The stub model issues one
    ``transition_stage(search)`` call so the deferred persist fires; a
    stub that returns no tool calls would leave no instance (the
    transactional contract — a turn that fails before any handoff must
    not leave a zombie).
    """
    active_instance: list[WorkflowPosition | None] = [None]
    saved_instances: list[WorkflowPosition] = []
    _handoff_issued = {"flag": False}

    async def complete(**_kwargs: Any) -> _StubCompletion:
        # First iteration: issue the transition_stage call the routing
        # hint asked for so the deferred ad-hoc persist fires. After that,
        # return a plain completion to end the turn.
        if not _handoff_issued["flag"]:
            _handoff_issued["flag"] = True
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_handoff",
                        "type": "function",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "search"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    # Mirror the host wiring: ``workflow_position`` returns the live active
    # ref (None at turn 1); the updater persists the advanced instance and
    # swaps the active ref in place.
    hooks.workflow.workflow_position = lambda: active_instance[0]

    def _wi_updater(instance: WorkflowPosition) -> None:
        saved_instances.append(instance)
        active_instance[0] = instance

    hooks.workflow.workflow_position_updater = _wi_updater

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

    assert saved_instances, "an ad-hoc WorkflowPosition must be created on the first handoff"
    # Phase 6c: a bare turn-1 position is created at turn start (no steps);
    # the compiled chain attaches to it at the first successful handoff.
    # The chain assertions target the attach commit — the FIRST saved
    # position carrying declared steps (later saves are the forced-action
    # enforcer driving the remaining handoffs: search → coding → markdown).
    chained = [p for p in saved_instances if p.declared_steps]
    assert chained, "the deferred attach must commit the compiled steps at the first handoff"
    instance = chained[0]
    assert instance.spawned_from == "adhoc"
    mode_ids = [s.mode_id for s in instance.declared_steps]
    assert mode_ids == ["search", "coding", "markdown"]
    # The first step (search) is complete after the deferred attach; the
    # pointer has advanced to the next step (coding).
    assert "search_0" in instance.completed_steps or any(
        s.mode_id == "search" and s.step_id in instance.completed_steps
        for s in instance.declared_steps
    )
    # enter_mode parked the stage pointer in the handoff target.
    assert instance.mode_id == "search"


@pytest.mark.asyncio
async def test_adhoc_no_chain_attaches_no_steps() -> None:
    """A single-step request (no multi-step chain) commits no chain steps.

    Guards against over-eager chain compilation for trivial queries that
    the current mode handles in-place. Phase 6c: a bare turn-1 position
    is created (the "you are here" pointer) — the invariant is that it
    carries no declared steps. No routing gap → no chain.
    """
    active_instance: list[WorkflowPosition | None] = [None]
    saved_instances: list[WorkflowPosition] = []

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Hello.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance[0]

    def _wi_updater(instance: WorkflowPosition) -> None:
        saved_instances.append(instance)
        active_instance[0] = instance

    hooks.workflow.workflow_position_updater = _wi_updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        await run_turn(mode_id="default", query="hello", hooks=hooks)

    chained = [p for p in saved_instances if p.declared_steps]
    assert not chained, (
        "no chain steps should be committed for a trivial query "
        f"(got {len(chained)} position(s) with declared steps)"
    )


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

    active_instance: list[WorkflowPosition | None] = [None]
    saved_instances: list[WorkflowPosition] = []

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance[0]

    def _wi_updater(instance: WorkflowPosition) -> None:
        saved_instances.append(instance)
        active_instance[0] = instance

    hooks.workflow.workflow_position_updater = _wi_updater

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

    # No ad-hoc chain: plan escalation owns the chain. The ad-hoc
    # block must have skipped (the hint was a plan-escalation hint), so no
    # saved position carries ad-hoc declared steps.
    adhoc_chains = [p for p in saved_instances if p.spawned_from == "adhoc" and p.declared_steps]
    assert not adhoc_chains, (
        "ad-hoc chain must not be committed when plan escalation fires "
        f"(got {len(adhoc_chains)} ad-hoc chain(s))"
    )
