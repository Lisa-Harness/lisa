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
