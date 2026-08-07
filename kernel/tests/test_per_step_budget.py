"""Phase 3 tests: per-step iteration sub-budget.

Covers the per-step iteration/tool-call caps that prevent a chained
workflow from starving its final step on the shared ``max_tool_iterations``
budget (the 2026-07-28 glm-5.2 self-assessment stall). When a step
exhausts its per-step budget but the global cap has not fired, the kernel
synthesizes a handoff to the next remaining step instead of the generic
empty-content final completion.

Self-contained: does not import from ``test_turn_runner.py`` (avoids the
conftest circular-import trap documented in repo memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from kernel.models import WorkflowModeConfig
from kernel.turn_runner import TurnHooks, run_turn
from kernel.workflow_position import WorkflowPosition, WorkflowStep


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
    """A catalog for the search → coding chain."""
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
    }


def _instance_two_step(current_step_id: str = "search_0") -> WorkflowPosition:
    """A two-step instance: search (current) → coding (remaining)."""
    return WorkflowPosition(
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
        completed_steps=(),
        current_step_id=current_step_id,
        mode_id="search",
        spawned_from="adhoc",
    )


@pytest.mark.asyncio
async def test_per_step_cap_forces_handoff_before_global_cap() -> None:
    """A step that burns its per-step budget synthesizes a handoff.

    With ``per_step_max_iterations=3`` and ``max_tool_iterations=15``, a
    search step that calls a tool 3 times without transitioning must
    trigger the kernel's per-step-cap handoff to coding (the next
    remaining step) rather than looping until the global cap.
    """
    instance = _instance_two_step(current_step_id="search_0")
    instances: list[WorkflowPosition] = []

    call_count = 0

    async def loop_then_handoff(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # First 3 iterations: call an unregistered no-op tool that keeps
        # the model in search mode (no transition_stage, so the per-step
        # counter does NOT reset). The per-step cap (3) fires before the
        # global cap.
        if call_count <= 3:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_count}",
                        "function": {
                            "name": "noop_tool_not_registered",
                            "arguments": "{}",
                        },
                    }
                ],
            )
        # After the kernel synthesizes the handoff to coding, produce text.
        return _StubCompletion(content="Done coding.")

    hooks = _build_minimal_hooks(loop_then_handoff)
    hooks.workflow_position= lambda: instance

    def updater(updated: WorkflowPosition) -> None:
        instances.append(updated)
        object.__setattr__(instance, "__dict__", updated.__dict__)

    hooks.workflow_position_updater = updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="search",
            query="search the web then read your code",
            hooks=hooks,
            max_tool_iterations=15,
            per_step_max_iterations=3,
            per_step_max_tool_calls=12,
        )

    # The kernel synthesized a handoff to coding before the global cap.
    assert result.handoff_to == "coding", (
        f"per-step cap did not force handoff to coding; got {result.handoff_to!r}"
    )
    assert result.iteration_count < 15, (
        f"global cap fired ({result.iteration_count}); per-step cap should have fired first"
    )


@pytest.mark.asyncio
async def test_global_cap_still_runs_final_completion() -> None:
    """With no instance wired, the per-step cap degrades to the global cap.

    No ``workflow_position`` hook → the per-step-cap intercept falls
    through and the existing empty-content final completion runs. The
    model produces a final answer (or the generic fallback), not a
    synthesized handoff.
    """
    call_count = 0

    async def loop_tools_then_final(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Always call an unregistered tool — exhausts both the per-step
        # and global caps without triggering a handoff/reset.
        if call_count <= 4:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_count}",
                        "function": {
                            "name": "noop_tool_not_registered",
                            "arguments": "{}",
                        },
                    }
                ],
            )
        # The forced final completion (tools=None) produces this.
        return _StubCompletion(content="Final answer.")

    hooks = _build_minimal_hooks(loop_tools_then_final)
    # No workflow_position wired.

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="default",
            query="hello",
            hooks=hooks,
            max_tool_iterations=4,
            per_step_max_iterations=3,
            per_step_max_tool_calls=12,
        )

    # No synthesized handoff (no instance). The final completion or the
    # generic fallback produced the response text.
    assert result.handoff_to is None, (
        f"handoff synthesized without an instance: {result.handoff_to!r}"
    )
    # The global cap (4) fired; we should not have looped forever.
    assert result.iteration_count <= 4


@pytest.mark.asyncio
async def test_step_counter_resets_on_handoff() -> None:
    """A successful handoff resets the per-step counter.

    Starting in search, the model transitions to coding at iteration 2
    (within the per-step cap). The per-step counter resets to 0, so
    coding gets its own full 3-iteration budget. The turn completes in
    coding without the per-step cap firing prematurely.
    """
    instance = _instance_two_step(current_step_id="search_0")
    instances: list[WorkflowPosition] = []

    call_count = 0

    async def handoff_then_finish(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Transition to coding immediately (within the per-step cap).
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "coding", "rationale": "found it"}',
                        },
                    }
                ],
            )
        # In coding now — produce text on the first coding iteration.
        return _StubCompletion(content="Done coding.")

    hooks = _build_minimal_hooks(handoff_then_finish)
    hooks.workflow_position= lambda: instance

    def updater(updated: WorkflowPosition) -> None:
        instances.append(updated)
        object.__setattr__(instance, "__dict__", updated.__dict__)

    hooks.workflow_position_updater = updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="search",
            query="search the web then read your code",
            hooks=hooks,
            max_tool_iterations=15,
            per_step_max_iterations=3,
            per_step_max_tool_calls=12,
        )

    # The handoff to coding succeeded.
    assert result.handoff_to == "coding"
    # The instance was advanced.
    assert len(instances) >= 1
    advanced = instances[-1]
    assert "search_0" in advanced.completed_steps
    assert advanced.current_step_id == "coding_1"
    # The turn completed (did not hit the per-step cap, which would have
    # synthesized a second handoff). The model produced text in coding.
    # The instance is now active (persisted on the first handoff via the
    # deferred-persist path), so the post-turn progress renderer appends
    # a ``## Workflow progress`` block — assert the prefix, not equality.
    assert result.response_text.startswith("Done coding."), (
        f"expected 'Done coding.' prefix, got: {result.response_text!r}"
    )


@pytest.mark.asyncio
async def test_per_step_tool_call_cap_forces_handoff() -> None:
    """The per-step tool-call cap forces a handoff when reached first.

    With ``per_step_max_tool_calls=2`` and a higher iteration cap, a step
    that dispatches 2 tools without transitioning must synthesize the
    handoff to the next remaining step.
    """
    instance = _instance_two_step(current_step_id="search_0")
    instances: list[WorkflowPosition] = []

    call_count = 0

    async def two_tools_then_text(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Keep calling the no-op tool. After 2 tool calls the per-step
        # tool-call cap fires and the kernel synthesizes the handoff to
        # coding. The forced final completion (tools=None) in coding then
        # produces the text.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{call_count}",
                    "function": {
                        "name": "noop_tool_not_registered",
                        "arguments": "{}",
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(two_tools_then_text)
    hooks.workflow_position= lambda: instance

    def updater(updated: WorkflowPosition) -> None:
        instances.append(updated)
        object.__setattr__(instance, "__dict__", updated.__dict__)

    hooks.workflow_position_updater = updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="search",
            query="search the web then read your code",
            hooks=hooks,
            max_tool_iterations=15,
            per_step_max_iterations=8,
            per_step_max_tool_calls=2,
        )

    # The tool-call cap (2) fired before the iteration cap (8), forcing
    # the handoff to coding.
    assert result.handoff_to == "coding", (
        f"per-step tool-call cap did not force handoff; got {result.handoff_to!r}"
    )


@pytest.mark.asyncio
async def test_no_instance_per_step_cap_degrades_to_global() -> None:
    """No instance + per-step cap hit → existing fallback, no crash.

    The per-step-cap intercept checks ``hooks.workflow_position is not
    None``; when no instance is wired, it falls through to the existing
    empty-content final completion. This is the no-compat-path guarantee.
    """
    call_count = 0

    async def always_tool(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count <= 5:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{call_count}",
                        "function": {
                            "name": "noop_tool_not_registered",
                            "arguments": "{}",
                        },
                    }
                ],
            )
        return _StubCompletion(content="Final.")

    hooks = _build_minimal_hooks(always_tool)
    # No workflow_position wired.

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        result = await run_turn(
            mode_id="default",
            query="hello",
            hooks=hooks,
            max_tool_iterations=5,
            per_step_max_iterations=2,
            per_step_max_tool_calls=12,
        )

    # No handoff synthesized (no instance). No crash. The turn produced
    # *some* response (the final completion or the generic fallback).
    assert result.handoff_to is None
    assert result.response_text  # non-empty
