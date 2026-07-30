"""Tests for kernel/turn_runner.py — the shared model+tool loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.execution_policy import CapabilityExecutionPolicy, ExecutionPolicy
from kernel.models import (
    CapabilityManifest,
    OperationExecutionContext,
    OperationResult,
    WorkflowModeConfig,
)
from kernel.seams.protocols import CoordinatorRouteDecision
from kernel.turn_runner import TurnHooks, TurnResult, run_turn

# ── Module-level setup ────────────────────────────────────────────
_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()

# The shared conftest fixture (_mcp_catalog_setup) handles all catalog wiring.


# ── Test helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_stub_complete(response: str, tool_calls: list[dict[str, Any]] | None = None):
    """Return an async complete stub that returns the given response."""

    async def stub(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=response, tool_calls=tool_calls)

    return stub


async def _stub_execute_operation(
    operation_id: str, _ctx: OperationExecutionContext
) -> OperationResult:
    return OperationResult(output_text=f"Executed {operation_id}")


def _build_minimal_hooks(complete_fn) -> TurnHooks:
    return TurnHooks(complete=complete_fn)


# ── Core flow tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_requires_complete_hook() -> None:
    """run_turn raises NotImplementedError when hooks.complete is None."""
    hooks = TurnHooks()  # no complete

    with pytest.raises(NotImplementedError, match="complete"):
        await run_turn(mode_id="default", query="hello", hooks=hooks)


@pytest.mark.asyncio
async def test_run_turn_simple_chat_no_tools() -> None:
    """Basic chat: model returns text immediately, no tool calls."""
    hooks = _build_minimal_hooks(_make_stub_complete("Hello, world!"))

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.response_text == "Hello, world!"
    assert result.tool_call_count == 0
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_run_turn_simple_chat_default_mode() -> None:
    """Chat through the default mode (no pre-turn ops) produces a TurnResult."""
    hooks = _build_minimal_hooks(_make_stub_complete("Response from default mode."))

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.response_text == "Response from default mode."
    assert result.tool_call_count == 0
    assert result.iteration_count == 1


@pytest.mark.asyncio
async def test_run_turn_uses_persisted_route_for_continuation_queries() -> None:
    """A persisted route target is injected as a routing hint for continuation queries."""

    class _StubCoordinator:
        def __init__(self) -> None:
            self.saved_decision: CoordinatorRouteDecision | None = None

        def load(self, _req: Any) -> CoordinatorRouteDecision | None:
            return self.saved_decision

        def save(self, decision: CoordinatorRouteDecision, _req: Any) -> None:
            self.saved_decision = decision

    captured_system_prompts: list[str] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_system_prompts.append(str(kwargs["messages"][0].get("content", "")))
        return _StubCompletion(content="Done.")

    coordinator = _StubCoordinator()
    coordinator.saved_decision = CoordinatorRouteDecision(next_target="markdown")

    hooks = _build_minimal_hooks(complete)
    hooks.route_coordinator = coordinator

    result = await run_turn(mode_id="default", query="proceed", hooks=hooks)

    assert result.response_text == "Done."
    assert captured_system_prompts
    assert "transition_stage" in captured_system_prompts[0]
    assert "markdown" in captured_system_prompts[0]


@pytest.mark.asyncio
async def test_run_turn_planner_action_mode_drives_handoff_without_model_transition() -> None:
    """When a planner action mode's entry operation returns a
    ``target_stage_id`` in ``output_json``, the kernel drives the handoff
    itself by recursing into ``run_turn`` against the target mode — it does
    NOT rely on the model calling ``transition_stage``.

    Regression for the 2026-07-28 incident: weak models stopped after
    ``plan_workflow`` produced a target, emitting a summary instead of
    calling ``transition_stage``. No coordinator decision was saved, so a
    follow-up ``proceed`` had nothing to load and the chain stalled. The
    cutover makes the kernel the handoff driver: the coordinator is saved
    from the plan result, and the target mode runs deterministically.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {"plan": plan_cfg, "search": search_cfg}

    captured_modes: list[str] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        # The recursive turn runs in `search` mode; capture the system
        # prompt's mode marker to prove the handoff happened.
        sys_content = str(kwargs["messages"][0].get("content", ""))
        if "search" in sys_content.lower() or "research" in sys_content.lower():
            captured_modes.append("search")
        else:
            captured_modes.append("other")
        return _StubCompletion(content="Search results.")

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        # Simulate plan_workflow returning a routing target.
        return OperationResult(
            output_text="research intent detected",
            output_json={"target_stage_id": "search", "rationale": "research intent"},
        )

    class _StubCoordinator:
        def __init__(self) -> None:
            self.saved_decision: CoordinatorRouteDecision | None = None

        def load(self, _req: Any) -> CoordinatorRouteDecision | None:
            return self.saved_decision

        def save(self, decision: CoordinatorRouteDecision, _req: Any) -> None:
            self.saved_decision = decision

    coordinator = _StubCoordinator()
    hooks = _build_minimal_hooks(complete)
    hooks.route_coordinator = coordinator

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(mode_id="plan", query="search for rust news", hooks=hooks)

    # The kernel recursed into search mode and produced its response, not
    # the planner's rationale text.
    assert result.response_text == "Search results."
    assert "search" in captured_modes
    # The coordinator was saved from the plan result — so a later
    # `proceed` can resume the chain even if the search turn stalls.
    assert coordinator.saved_decision is not None
    assert coordinator.saved_decision.next_target == "search"


@pytest.mark.asyncio
async def test_run_turn_planner_action_mode_empty_target_falls_back_to_text() -> None:
    """When a planner action mode's entry op returns an empty
    ``target_stage_id`` (no routable mode matched), the kernel terminates
    the turn with a structured recovery prompt (not the raw entry-op
    text) and marks the model unreliable for planning so the hint
    generator stops escalating to ``plan`` for it. The recovery prompt
    names the failure reason and offers the user concrete next steps
    (rephrase, name a target mode directly, or discuss first). This is
    the 2026-07-30 hard-cutover from the old soft-fallthrough that
    returned the raw error text and let the post-handoff hint
    re-hallucinate a target on the next turn."""
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.routing_vocabulary import (
        is_plan_reliable_for_model,
        reset_plan_reliability,
    )

    reset_plan_reliability()
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    patched_catalog = {"plan": plan_cfg}

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        return OperationResult(
            output_text="No installed mode provides a write capability.",
            output_json={"target_stage_id": "", "rationale": "no match"},
        )

    async def complete(**_kwargs: Any) -> _StubCompletion:
        # Should never be called when the target is empty.
        return _StubCompletion(content="should not reach model")

    hooks = _build_minimal_hooks(complete)

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan", query="write a novel", hooks=hooks,
            preferred_model_id="test-planner-model",
        )

    # The recovery prompt incorporates the rationale and offers next steps.
    assert "plan" in result.response_text
    assert "no match" in result.response_text
    assert "Rephrase" in result.response_text or "rephrase" in result.response_text
    assert result.tool_call_count == 0
    # The model is now marked unreliable for planning so the hint generator
    # stops escalating to ``plan`` for it.
    assert not is_plan_reliable_for_model("test-planner-model")
    reset_plan_reliability()


@pytest.mark.asyncio
async def test_run_turn_planner_with_steps_renders_visible_plan_and_persists_instance() -> None:
    """When plan_workflow returns a ``steps`` array (Option B visible plan),
    the turn runner:
    1. Prepends the rendered plan markdown block to the response_text so
       the user sees the plan before the first step's output.
    2. Creates a ``WorkflowInstance`` with ``spawned_from="plan_workflow_op"``
       and persists it via ``workflow_instance_updater``.
    3. Normalizes ``target_stage_id`` to ``steps[0].mode_id`` and recurses
       into the first step.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_instance import WorkflowInstance

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    markdown_cfg = _mode_config(mode_id="markdown", satisfies_intents=["write"])
    patched_catalog = {
        "plan": plan_cfg,
        "search": search_cfg,
        "markdown": markdown_cfg,
    }

    plan_steps = [
        {
            "mode_id": "search",
            "description": "Search for the latest rust async runtime news",
            "rationale": "Provides web_search",
        },
        {
            "mode_id": "markdown",
            "description": "Draft a markdown summary",
            "rationale": "Provides document drafting",
        },
    ]
    plan_block = "## Plan\n\n1. **search** — Search for the latest rust async runtime news\n   _Provides web_search_\n2. **markdown** — Draft a markdown summary\n   _Provides document drafting_\n"

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        return OperationResult(
            output_text=plan_block,
            output_json={
                "target_stage_id": "search",
                "rationale": "Research then write",
                "steps": plan_steps,
            },
        )

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Search results for rust async.")

    saved_instances: list[WorkflowInstance] = []

    def _wi_updater(instance: WorkflowInstance) -> None:
        saved_instances.append(instance)

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_instance_updater = _wi_updater

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan",
            query="research the latest rust async runtime then draft a summary",
            hooks=hooks,
        )

    # The plan block is prepended to the response, followed by the first
    # step's output.
    assert "## Plan" in result.response_text
    assert "search" in result.response_text
    assert "markdown" in result.response_text
    assert "Search results for rust async." in result.response_text
    # The plan block comes BEFORE the step output.
    assert result.response_text.index("## Plan") < result.response_text.index("Search results")

    # A WorkflowInstance was persisted with the correct steps.
    assert len(saved_instances) == 1, "exactly one WorkflowInstance persisted"
    instance = saved_instances[0]
    assert instance.spawned_from == "plan_workflow_op"
    assert len(instance.declared_steps) == 2
    assert instance.declared_steps[0].mode_id == "search"
    assert instance.declared_steps[1].mode_id == "markdown"
    assert instance.declared_steps[1].is_terminal
    assert instance.current_step_id == instance.declared_steps[0].step_id
    assert instance.current_mode_id == "search"


@pytest.mark.asyncio
async def test_run_turn_planner_without_steps_no_plan_block_no_instance() -> None:
    """When plan_workflow returns no ``steps`` (weak-model fallback), the
    turn runner does NOT create a WorkflowInstance and does NOT prepend a
    plan block — same as the pre-Option-B behavior. No regression."""
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_instance import WorkflowInstance

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {"plan": plan_cfg, "search": search_cfg}

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        # No steps — the legacy single-target shape.
        return OperationResult(
            output_text="research intent detected",
            output_json={
                "target_stage_id": "search",
                "rationale": "research intent detected",
                "steps": [],
            },
        )

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Search results.")

    saved_instances: list[WorkflowInstance] = []

    def _wi_updater(instance: WorkflowInstance) -> None:
        saved_instances.append(instance)

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_instance_updater = _wi_updater

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan", query="search for rust news", hooks=hooks,
        )

    # No plan block — the response is just the step output.
    assert "## Plan" not in result.response_text
    assert result.response_text == "Search results."
    # No WorkflowInstance created (no steps to persist).
    assert len(saved_instances) == 0


@pytest.mark.asyncio
async def test_run_turn_mid_turn_handoff_to_planner_drives_next_handoff() -> None:
    """When a mid-turn handoff lands in a planner action mode (e.g.
    ``plan``) and the planner's entry op returns a ``target_stage_id``,
    the kernel synthesizes the next ``transition_stage`` call itself
    instead of injecting the plan result as text and waiting for the
    model to call ``transition_stage``.

    Regression for the 2026-07-28 stall: the escalation path is
    ``default`` → (routing hint) → model calls ``transition_stage`` to
    ``plan`` (mid-turn handoff) → ``plan_workflow`` returns
    ``target_stage_id="search"`` → the model in ``plan`` mode reads the
    plan result and STOPS. The fix sets ``pending_planned_handoff`` from
    the entry-op result; the next loop iteration synthesizes a
    ``transition_stage`` tool call for the planned target and dispatches
    it through the existing handler — no model call, no begging.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult

    default_cfg = _mode_config(mode_id="default")
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {
        "default": default_cfg, "plan": plan_cfg, "search": search_cfg,
    }

    call_log: list[str] = []
    plan_entry_calls = 0
    _default_calls = 0

    async def complete(**kwargs: Any) -> _StubCompletion:
        nonlocal _default_calls
        # Inspect the messages to determine which mode the model is in.
        # The routing hint for the plan escalation names "plan" in the
        # system prompt while still in default mode, so we detect the
        # default mode by the absence of plan/search mode markers AND
        # the presence of the user query (the first call).
        sys_content = str(kwargs["messages"][0].get("content", ""))
        # search mode: has search-specific tool schemas or the search
        # mode intro. The handoff to search rebuilds the system prompt
        # with the search mode's intro, which is empty in this stub
        # catalog — so detect by the absence of the plan routing hint
        # AND the absence of the transition_stage handoff-context block.
        is_plan_mode = "## Handoff Context" in sys_content and "`plan`" in sys_content
        is_search_mode = (
            "## Handoff Context" in sys_content and "`search`" in sys_content
        ) or ("Handed off to mode: search" in str(
            [m.get("content", "") for m in kwargs["messages"]]
        ))
        if is_search_mode:
            call_log.append("model_call:search")
            return _StubCompletion(content="Search results from web_search.")
        if is_plan_mode:
            call_log.append("model_call:plan")
            # The kernel's mid-turn entry-op detection sets
            # pending_planned_handoff, so the NEXT iteration synthesizes
            # the transition_stage call. Return a harmless text.
            return _StubCompletion(content="Planning...")
        # default mode: the model calls transition_stage to hand off to plan
        _default_calls += 1
        call_log.append("model_call:default")
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_default_to_plan",
                    "function": {
                        "name": "transition_stage",
                        "arguments": json.dumps(
                            {"target_stage_id": "plan", "rationale": "escalate to plan"}
                        ),
                    },
                }
            ],
        )

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        nonlocal plan_entry_calls
        plan_entry_calls += 1
        return OperationResult(
            output_text="research intent detected",
            output_json={"target_stage_id": "search", "rationale": "research intent"},
        )

    class _StubCoordinator:
        def __init__(self) -> None:
            self.saved: list[CoordinatorRouteDecision] = []

        def load(self, _req: Any) -> CoordinatorRouteDecision | None:
            return None

        def save(self, decision: CoordinatorRouteDecision, _req: Any) -> None:
            self.saved.append(decision)

    coordinator = _StubCoordinator()
    hooks = _build_minimal_hooks(complete)
    hooks.route_coordinator = coordinator

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default", query="search for rust news then write a summary", hooks=hooks
        )

    # The kernel drove the chain: default → plan → search, and the final
    # response came from the search mode, not the planner's rationale.
    assert result.response_text == "Search results from web_search."
    # The planner entry op ran exactly once (for the mid-turn handoff).
    assert plan_entry_calls == 1
    # The coordinator recorded the planned next target (search) from the
    # plan result, so a later "proceed" can resume.
    assert any(d.next_target == "search" for d in coordinator.saved)


@pytest.mark.asyncio
async def test_run_turn_missing_complete_raises_not_implemented() -> None:
    """A search (action) mode with entry ops needs complete_fn, and the
    operation fails with an OperationExecutionError wrapping the
    NotImplementedError when hooks.complete is missing."""
    from kernel.operations.registry import OperationExecutionError

    hooks = TurnHooks()  # no complete

    with pytest.raises(OperationExecutionError, match="web_search"):
        await run_turn(mode_id="search", query="python asyncio", hooks=hooks)


@pytest.mark.asyncio
async def test_run_turn_tool_loop_single_tool_call() -> None:
    """Model returns one tool call, then text — one tool loop iteration.

    Uses the 'default' mode which has no pre-turn operations, so the
    stub complete function's two-step logic works predictably.
    """
    call_counts = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_counts.append(1)
        if len(call_counts) == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {"name": "web_search", "arguments": '{"q": "test"}'},
                    }
                ],
            )
        return _StubCompletion(content="Here is your search result.")

    hooks = _build_minimal_hooks(two_step)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Here is your search result."
    assert result.tool_call_count == 1
    assert result.iteration_count == 2


@pytest.mark.asyncio
async def test_run_turn_max_tool_iterations_capped() -> None:
    """Tool loop capped at max_tool_iterations."""
    call_counts = []

    async def infinite_tools(**kwargs: Any) -> _StubCompletion:
        call_counts.append(1)
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{len(call_counts)}",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"q": "test"}',
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(infinite_tools)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=3,
    )

    assert result.iteration_count == 3
    assert result.tool_call_count == 3
    assert "did not produce a final response" in result.response_text


@pytest.mark.asyncio
async def test_run_turn_max_tool_iterations_defaults_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """When max_tool_iterations is None, the cap comes from kernel.config.Settings."""
    from kernel import config

    monkeypatch.setattr(config.settings, "max_tool_iterations", 2)

    call_counts: list[int] = []

    async def infinite_tools(**kwargs: Any) -> _StubCompletion:
        call_counts.append(1)
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{len(call_counts)}",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"q": "test"}',
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(infinite_tools)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        # max_tool_iterations intentionally omitted — must read from config
    )

    assert result.iteration_count == 2
    assert result.tool_call_count == 2
    assert "did not produce a final response" in result.response_text


@pytest.mark.asyncio
async def test_run_turn_execution_policy_tightens_model_call_iterations() -> None:
    """execution_policy.max_model_calls tightens max_tool_iterations."""
    call_counts = []

    async def infinite_tools(**kwargs: Any) -> _StubCompletion:
        call_counts.append(1)
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{len(call_counts)}",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"q": "test"}',
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(infinite_tools)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=10,
        execution_policy=ExecutionPolicy(max_model_calls=2),
    )

    assert result.iteration_count == 2
    assert result.tool_call_count == 2
    assert "did not produce a final response" in result.response_text


@pytest.mark.asyncio
async def test_run_turn_execution_policy_seeds_turn_budget_without_explicit_budget() -> None:
    """execution_policy max_capability_calls is adapted to a legacy TurnBudget."""

    async def budget_aware_complete(**kwargs: Any) -> _StubCompletion:
        tools = kwargs.get("tools") or []
        if tools:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Final response.")

    hooks = _build_minimal_hooks(budget_aware_complete)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        execution_policy=ExecutionPolicy(max_capability_calls=1),
    )

    assert result.tool_call_count == 1
    assert result.response_text == "Final response."


@pytest.mark.asyncio
async def test_run_turn_execution_policy_caps_specific_capability_calls() -> None:
    """Capability policy limits apply even when catalog limits are looser."""
    completion_count = 0

    async def repeat_then_finalize(**kwargs: Any) -> _StubCompletion:
        nonlocal completion_count
        completion_count += 1
        if completion_count <= 2:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": f"call_{completion_count}",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(repeat_then_finalize)
    hooks.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=4,
        execution_policy=ExecutionPolicy(
            capability_limits={
                "web_search": CapabilityExecutionPolicy(max_invocations_per_turn=1),
            }
        ),
    )

    assert result.tool_call_count == 1
    assert result.response_text == "Done."


@pytest.mark.asyncio
async def test_run_turn_calls_on_turn_complete() -> None:
    """on_turn_complete is called after a successful turn."""
    side_effects: list[dict[str, Any]] = []

    async def on_complete(
        mode_id: str, query: str, response_text: str, messages: list[dict[str, Any]]
    ) -> None:
        side_effects.append(
            {
                "mode_id": mode_id,
                "query": query,
                "response_text": response_text,
                "message_count": len(messages),
            }
        )

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.on_turn_complete = on_complete

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(side_effects) == 1
    assert side_effects[0]["mode_id"] == "default"
    assert side_effects[0]["query"] == "hello"
    assert side_effects[0]["response_text"] == "Done."
    assert side_effects[0]["message_count"] >= 2  # system + user + assistant


@pytest.mark.asyncio
async def test_run_turn_omits_empty_structured_pre_turn_briefing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured pre-turn ops with empty output_text should not inject JSON fallback."""

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    async def pre_turn_result(
        mode_id: str,
        ctx: OperationExecutionContext,
        ledger: Any = None,
    ) -> OperationResult:
        _ = (mode_id, ctx, ledger)
        return OperationResult(
            output_text='{"briefing_points": [], "output_text": "", "source_kind": "", "source_urls": []}',
            output_json={
                "briefing_points": [],
                "output_text": "",
                "source_kind": "",
                "source_urls": [],
            },
        )

    monkeypatch.setattr("kernel.turn_runner.run_mode_pre_turn_operations", pre_turn_result)

    hooks = _build_minimal_hooks(complete)
    result = await run_turn(mode_id="discuss", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_messages) >= 2
    user_message = str(captured_messages[1]["content"])
    assert "## Pre-turn Briefing" not in user_message
    assert "briefing_points" not in user_message


@pytest.mark.asyncio
async def test_run_turn_includes_non_empty_structured_pre_turn_briefing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured pre-turn ops still inject authored output_text when present."""

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    async def pre_turn_result(
        mode_id: str,
        ctx: OperationExecutionContext,
        ledger: Any = None,
    ) -> OperationResult:
        _ = (mode_id, ctx, ledger)
        return OperationResult(
            output_text='{"output_text": "fallback json dump"}',
            output_json={
                "output_text": "Fresh web context for this discussion:\n- grounded point",
                "briefing_points": ["grounded point"],
            },
        )

    monkeypatch.setattr("kernel.turn_runner.run_mode_pre_turn_operations", pre_turn_result)

    hooks = _build_minimal_hooks(complete)
    result = await run_turn(mode_id="discuss", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_messages) >= 2
    user_message = str(captured_messages[1]["content"])
    assert "## Pre-turn Briefing" in user_message
    assert "Fresh web context for this discussion:" in user_message
    assert "fallback json dump" not in user_message


@pytest.mark.asyncio
async def test_run_turn_prefers_explicit_model_override() -> None:
    """A host-supplied preferred model override beats the mode default."""

    resolved_model_ids: list[str] = []

    def resolve_model(model_id: str) -> Any:
        resolved_model_ids.append(model_id)
        return SimpleNamespace(model_id=model_id, reasoning_model=False)

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.resolve_model = resolve_model

    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=hooks,
        preferred_model_id="haiku",
    )

    assert result.response_text == "Done."
    assert resolved_model_ids == ["haiku"]


@pytest.mark.asyncio
async def test_run_turn_injects_workflow_stage_instruction() -> None:
    """Workflow modes inject the current stage's instruction_body into the
    system prompt via the assemble_system_prompt hook's stage_instruction arg.

    Without this, loaded stage instructions (inquire.md, reason.md, etc.)
    are never injected and the FSM is dead at runtime.
    """
    from kernel.ipsar_fsm import get_workflow_catalog

    captured_stage_instructions: list[str | None] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages = kwargs["messages"]
        return _StubCompletion(content="Done.")

    def assemble_system(mode_instruction: str, stage_instruction: str | None) -> str:
        captured_stage_instructions.append(stage_instruction)
        parts = [mode_instruction]
        if stage_instruction:
            parts.append(stage_instruction)
        return "\n\n".join(parts)

    def workflow_session() -> Any:
        from kernel.models import WorkflowSessionState

        mode = get_workflow_catalog().get("ponder")
        if mode is None:
            return None
        return WorkflowSessionState(
            user_id="test-user",
            user_slug="test",
            mode_id="ponder",
            stage_id="inquire",
            entered_at="2026-01-01T00:00:00Z",
        )

    hooks = _build_minimal_hooks(complete)
    hooks.assemble_system_prompt = assemble_system
    hooks.workflow_session = workflow_session

    result = await run_turn(mode_id="ponder", query="analyze this", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_stage_instructions) >= 1
    stage_instruction = captured_stage_instructions[0]
    assert stage_instruction is not None
    assert "Inquire" in stage_instruction or "inquire" in stage_instruction.lower()


@pytest.mark.asyncio
async def test_run_turn_workflow_transition_dispatches_apply_transition() -> None:
    """When the model calls transition_stage, the loop intercepts it,
    calls apply_transition, updates the session via workflow_session_updater,
    and reports transitioned_to on the TurnResult.
    """
    from kernel.models import WorkflowSessionState

    updated_sessions: list[WorkflowSessionState] = []

    # Mutable session so workflow_session() always returns the latest state.
    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="ponder",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
    )

    def workflow_session() -> Any:
        return session

    def workflow_session_updater(updated: WorkflowSessionState) -> None:
        updated_sessions.append(updated)
        # Mutate in place so workflow_session() sees the new stage.
        session.stage_id = updated.stage_id
        session.turn_count = updated.turn_count
        session.stage_data = updated.stage_data

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
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
                            "arguments": '{"target_stage_id": "reason", "rationale": "enough evidence"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done reasoning.")

    hooks = _build_minimal_hooks(two_step)
    hooks.workflow_session = workflow_session
    hooks.workflow_session_updater = workflow_session_updater

    result = await run_turn(mode_id="ponder", query="analyze this", hooks=hooks)

    assert result.response_text == "Done reasoning."
    assert result.transitioned_to == "reason"
    assert len(updated_sessions) == 1
    assert updated_sessions[0].stage_id == "reason"
    assert updated_sessions[0].turn_count == 1


@pytest.mark.asyncio
async def test_run_turn_workflow_stuck_session_nudge_injects_message() -> None:
    """After stage_nudge_threshold turns in the same stage, the loop
    injects a nudge user-message prompting the model to transition.
    """
    from kernel.models import WorkflowSessionState

    session = WorkflowSessionState(
        user_id="test-user",
        user_slug="test",
        mode_id="ponder",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
        turn_count=4,  # at the nudge threshold
    )

    def workflow_session() -> Any:
        return session

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.clear()
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow_session = workflow_session

    result = await run_turn(mode_id="ponder", query="analyze this", hooks=hooks)

    assert result.response_text == "Done."
    # The nudge message should be present in the messages sent to the model.
    nudge_found = any(
        "been in the" in str(msg.get("content", ""))
        for msg in captured_messages
        if msg.get("role") == "user"
    )
    assert nudge_found


@pytest.mark.asyncio
async def test_run_turn_mcp_tool_not_registered() -> None:
    """When a tool name is not on any MCP server, the pool returns an error
    and the turn continues."""
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "custom_tool_not_registered",
                            "arguments": "{}",
                        },
                    }
                ],
            )
        return _StubCompletion(content="Used custom tool.")

    hooks = _build_minimal_hooks(two_step)
    # No mcp_tool_servers configured — pool has no tools registered.

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Used custom tool."
    assert result.tool_call_count == 1


@pytest.mark.asyncio
async def test_default_hooks_are_noops() -> None:
    """TurnHooks defaults do not crash when invoked directly."""
    hooks = TurnHooks()

    # Default resolve_model returns a stub
    model = hooks.resolve_model("test") if hooks.resolve_model else None
    if hooks.resolve_model:
        assert model is not None

    # Default assemble_system_prompt
    if hooks.assemble_system_prompt:
        result = hooks.assemble_system_prompt("instruction", None)
        assert result == "instruction"

    # Default assemble_user_context returns empty (accepts optional manifest)
    if hooks.assemble_user_context:
        result = hooks.assemble_user_context("m", "q", "b", None)
        assert result == ""

    # Default mcp_tool_servers is an empty list
    assert hooks.mcp_tool_servers == []


@pytest.mark.asyncio
async def test_turn_result_struct() -> None:
    """TurnResult is a plain dataclass with expected fields."""
    result = TurnResult(response_text="ok", tool_call_count=3, iteration_count=2)

    assert result.response_text == "ok"
    assert result.tool_call_count == 3
    assert result.iteration_count == 2


@pytest.mark.asyncio
async def test_run_turn_passes_capability_manifest_to_assemble_user_context() -> None:
    """The turn runner compiles a CapabilityManifest from the live tool schemas
    and passes it to assemble_user_context. This closes the Tooling Paradox:
    the capability context now reflects exactly what is injected into the
    current turn's function schema.
    """
    from kernel.models import CapabilityManifest

    captured_manifests: list[CapabilityManifest | None] = []

    def assemble_user(
        _mode_id: str,
        _query: str,
        _pre_turn_briefing: str,
        capability_manifest: CapabilityManifest | None = None,
    ) -> str:
        captured_manifests.append(capability_manifest)
        return ""

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.assemble_user_context = assemble_user

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(captured_manifests) == 1
    manifest = captured_manifests[0]
    assert manifest is not None
    assert isinstance(manifest, CapabilityManifest)
    # The manifest should have at least one entry (operations from the catalog)
    assert len(manifest.entries) >= 1
    # Every entry should be marked available with confidence 1.0
    for entry in manifest.entries:
        assert entry.status == "available"
        assert entry.confidence == 1.0


# ── Unified FSM routing tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_mode_handoff_via_transition_stage() -> None:
    """When the model calls transition_stage with a mode_id target (not a
    workflow stage), the loop performs a mid-turn mode switch: swaps the
    system prompt, rebuilds the tool surface, and sets handoff_to on the
    TurnResult.
    """
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
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
                            "arguments": '{"target_stage_id": "markdown", "rationale": "user wants to write a doc"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done in markdown mode.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a markdown doc", hooks=hooks)

    assert result.response_text == "Done in markdown mode."
    assert result.handoff_to == "markdown"
    assert result.transitioned_to is None  # no workflow stage transition


@pytest.mark.asyncio
async def test_run_turn_fork_transition_basic() -> None:
    """When the model calls transition_stage with a fork: target, the loop
    fans out child turns in parallel and collects results into fork_results.
    """
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
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
                            "arguments": '{"target_stage_id": "fork:default,default"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Synthesized from fork results.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="research and write", hooks=hooks)

    assert result.response_text == "Synthesized from fork results."
    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    for child in result.fork_results:
        assert child.status in ("ok", "error", "timeout", "cancelled", "no_tools")


@pytest.mark.asyncio
async def test_run_turn_fork_marks_no_tools_child_when_action_mode_skipped_tools() -> None:
    """A fork child whose mode declares action operations/tools but
    completes with zero tool calls is marked ``status="no_tools"`` rather
    than ``"ok"``.

    Regression for the 2026-07-18 incident: the markdown fork child
    produced a hallucinated "Done — wrote it to disk" text response
    without ever calling ``markdown_forge_document`` / ``write_file``,
    and the parent reported "2/2 children succeeded" while nothing landed
    on disk. ``learn`` and ``markdown`` declare non-empty
    ``allowed_operation_ids``/``allowed_tool_ids``; a child running such
    a mode that made no tool calls did not actually do its job.
    """

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
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
                            "arguments": '{"target_stage_id": "fork:learn,markdown"}',
                        },
                    }
                ],
            )
        # Each fork child returns text only — no tool calls.
        return _StubCompletion(content="Done — wrote it to disk.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(
        mode_id="default", query="remember this and write a markdown", hooks=hooks
    )

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    statuses = {child.target.mode_id: child.status for child in result.fork_results}
    # Both learn and markdown declare action tools/operations, so children
    # that made zero tool calls must be flagged no_tools, not ok.
    assert statuses["learn"] == "no_tools"
    assert statuses["markdown"] == "no_tools"
    for child in result.fork_results:
        assert child.status != "ok"


def test_fork_mode_requires_tools_true_for_action_modes() -> None:
    """``_fork_mode_requires_tools`` returns True for modes that declare
    non-empty ``allowed_operation_ids`` or ``allowed_tool_ids`` (learn,
    markdown), and False for chat-only modes (default, ponder)."""
    from kernel.turn_runner import _fork_mode_requires_tools

    assert _fork_mode_requires_tools("learn") is True
    assert _fork_mode_requires_tools("markdown") is True
    assert _fork_mode_requires_tools("default") is False
    assert _fork_mode_requires_tools("nonexistent_mode") is False


def test_fork_mode_action_tool_ids_returns_authored_action_tools() -> None:
    """``_fork_mode_action_tool_ids`` returns the authored *side-effecting*
    operation/tool ids for action modes and an empty set for modes whose
    authored tools are all read-effect (ponder) or absent (default)."""
    from kernel.turn_runner import _fork_mode_action_tool_ids

    markdown_actions = _fork_mode_action_tool_ids("markdown")
    # markdown declares markdown_forge_document / markdown_generate as
    # authored operations/tools — at least one must be present.
    assert "markdown_forge_document" in markdown_actions or "markdown_generate" in markdown_actions
    # ponder's only authored operation is web_search (read-effect) — it
    # must be excluded because ponder's output is its response text, not a
    # side-effect. A child that gathers evidence via fetch_page and
    # produces an opinion has done its job.
    assert _fork_mode_action_tool_ids("ponder") == set()
    # Chat-only modes have no authored action tools.
    assert _fork_mode_action_tool_ids("default") == set()
    assert _fork_mode_action_tool_ids("nonexistent_mode") == set()


@pytest.mark.asyncio
async def test_run_turn_fork_marks_no_tools_child_when_action_tool_skipped() -> None:
    """A fork child that makes tool calls but none of its mode's authored
    *action* tools (only read-only fetches) is marked ``status="no_tools"``.

    Regression for the 2026-07-27 follow-on: the ``markdown`` child did 4
    ``fetch_page`` calls then exited empty without ever calling
    ``markdown_forge_document``, so nothing landed on disk — yet the
    existing ``tool_call_count == 0`` guard passed it as ``ok`` because
    it made *some* tool calls. The validator now checks that at least one
    of the mode's authored action tools was invoked.
    """
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
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
                            "arguments": '{"target_stage_id": "fork:markdown"}',
                        },
                    }
                ],
            )
        # The markdown child calls fetch_page (read-only) instead of its
        # action tool (markdown_forge_document). It made tool calls, but
        # none are its mode's authored action tools.
        return _StubCompletion(
            content="Done — wrote it to disk.",
            tool_calls=[
                {
                    "id": "call_2",
                    "function": {
                        "name": "fetch_page",
                        "arguments": '{"url": "https://example.com"}',
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(
        mode_id="default", query="write a markdown guide", hooks=hooks
    )

    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    # The child made a tool call (fetch_page) but skipped its action tools,
    # so it must be flagged no_tools — not ok.
    assert child.status == "no_tools"
    assert child.target.mode_id == "markdown"


@pytest.mark.asyncio
async def test_run_turn_emits_model_call_stage_events() -> None:
    """The on_stage hook fires model_call and model_call_complete around
    each model completion, carrying iteration and token metadata.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    labels = [label for label, _ in events]
    assert "model_call" in labels
    assert "model_call_complete" in labels
    # model_call_complete should be the last event and carry token counts.
    last_label, last_data = events[-1]
    assert last_label == "model_call_complete"
    assert "iteration" in last_data
    assert "tool_calls" in last_data


@pytest.mark.asyncio
async def test_run_turn_emits_tool_dispatch_stage_events() -> None:
    """The on_stage hook fires tool_dispatch and tool_dispatch_complete
    around each operation/tool dispatch, with a tool_kind label.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)
    hooks.execute_operation = _stub_execute_operation
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert result.tool_call_count == 1
    dispatch_events = [e for e in events if e[0] == "tool_dispatch"]
    complete_events = [e for e in events if e[0] == "tool_dispatch_complete"]
    assert len(dispatch_events) == 1
    assert len(complete_events) == 1
    assert dispatch_events[0][1]["tool"] == "web_search"
    assert dispatch_events[0][1]["tool_kind"] == "operation"
    assert "duration_ms" in complete_events[0][1]


@pytest.mark.asyncio
async def test_run_turn_emits_fork_child_stage_events_per_child() -> None:
    """The on_stage hook fires fork_start, one fork_child_start and one
    fork_child_complete per child, then fork_complete — so a transport
    can render a line per active call.
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
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
                            "arguments": '{"target_stage_id": "fork:default,default"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Synthesized.")

    hooks = _build_minimal_hooks(two_step)
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="fan out", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    labels = [label for label, _ in events]
    assert labels.count("fork_start") == 1
    assert labels.count("fork_child_start") == 2
    assert labels.count("fork_child_complete") == 2
    assert "fork_complete" in labels
    # fork_start should carry the target list.
    start_event = next(e for e in events if e[0] == "fork_start")
    assert start_event[1]["children"] == 2
    assert len(start_event[1]["targets"]) == 2
    # fork_child_start events carry a child_index.
    child_starts = [e for e in events if e[0] == "fork_child_start"]
    child_indices = sorted(e[1]["child_index"] for e in child_starts)
    assert child_indices == [0, 1]
    # fork_complete carries ok/failed counts.
    complete_event = next(e for e in events if e[0] == "fork_complete")
    assert complete_event[1]["children"] == 2
    assert complete_event[1]["ok"] + complete_event[1]["failed"] == 2


@pytest.mark.asyncio
async def test_run_turn_on_stage_hook_failure_is_swallowed() -> None:
    """A failing on_stage hook is logged and swallowed; the turn continues."""
    call_count = 0

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        raise RuntimeError("transport exploded")

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.on_stage = on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    # The turn still completes despite the on_stage hook raising every call.
    assert result.response_text == "Done."


@pytest.mark.asyncio
async def test_routing_fsm_synthesis() -> None:
    """The routing FSM synthesis produces a full routable mesh.

    Every routable mode (including default) can hand off to every other
    routable mode. Regression for the 2026-07-17 coding-mode fork gap: a
    remember+write request issued from coding must be able to fork to
    learn + markdown, which requires coding → learn and coding → markdown
    edges to exist. The conservative default-only edge set suppressed the
    fork hint and made the model fall back to coding_create_worktree.
    """
    from kernel.ipsar_fsm import (
        get_routable_mode_ids,
        get_routing_targets,
        synthesize_routing_transitions,
    )

    transitions = synthesize_routing_transitions()
    routable_ids = get_routable_mode_ids()

    # Every routable mode should be able to hand off to every other routable
    # mode (full mesh, excluding itself).
    for rid in routable_ids:
        targets = get_routing_targets(rid)
        for other in routable_ids:
            if other == rid:
                assert other not in targets, f"{rid} should not hand off to itself"
            else:
                assert other in targets, f"{rid} should be able to hand off to {other}"

    # default specifically must still reach every other routable mode.
    default_targets = transitions.get("default", [])
    for rid in routable_ids:
        if rid != "default":
            assert rid in default_targets, f"default should be able to hand off to {rid}"


def test_parse_fork_target_string() -> None:
    """The fork target parser handles mode_id and mode_id/stage_id syntax."""
    from kernel.ipsar_fsm import parse_fork_target_string

    # Simple mode_ids
    assert parse_fork_target_string("fork:coding,markdown") == [
        ("coding", ""),
        ("markdown", ""),
    ]
    # mode_id/stage_id pairs
    assert parse_fork_target_string("fork:coding/implement,ponder/review") == [
        ("coding", "implement"),
        ("ponder", "review"),
    ]
    # Mixed
    assert parse_fork_target_string("fork:search,coding/implement") == [
        ("search", ""),
        ("coding", "implement"),
    ]
    # Empty
    assert parse_fork_target_string("") == []
    assert parse_fork_target_string("fork:") == []


def test_is_fork_target() -> None:
    """The fork target detector recognizes fork: prefix."""
    from kernel.ipsar_fsm import is_fork_target

    assert is_fork_target("fork:coding,markdown") is True
    assert is_fork_target("FORK:coding") is True
    assert is_fork_target("coding") is False
    assert is_fork_target("reason") is False


@pytest.mark.asyncio
async def test_run_turn_mode_handoff_preserves_chat_history() -> None:
    """When the model hands off to another mode, the chat history (messages
    list) is preserved — only messages[0] (system prompt) is swapped.
    """
    call_count = 0
    captured_messages: list[list[dict[str, Any]]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Capture the messages list at each iteration
        msgs = kwargs.get("messages", [])
        captured_messages.append([dict(m) for m in msgs])
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "markdown"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a doc", hooks=hooks)

    assert result.handoff_to == "markdown"
    # After handoff, the model should have been called at least twice
    assert call_count >= 2
    # The first message (system prompt) should have changed after handoff
    # but the user message should be preserved
    if len(captured_messages) >= 2:
        # messages[0] is system, messages[1] is user
        # The user content should be the same across iterations
        first_user = captured_messages[0][1].get("content", "")
        second_user = captured_messages[1][1].get("content", "")
        # The user message content should contain the original query
        assert "write a doc" in first_user


@pytest.mark.xfail(
    reason=(
        "2026-07-25 fix: the search mode's ``web_search`` entry operation now "
        "correctly aborts the turn when it fails (returns non-JSON), instead of "
        "being swallowed and letting the turn continue. This test's stub returns "
        "free-form text for the entry-op completion, which now trips the "
        "entry-op-failure guard (``test_handoff_entry_op_failure_blocks_turn``) "
        "before the post-handoff hint is recomputed. The hint-recomputation "
        "behavior is still covered by "
        "``test_run_turn_handoff_clears_hint_when_new_mode_satisfies_all`` and "
        "the fork-intent-attribution tests."
    ),
    strict=True,
)
@pytest.mark.asyncio
async def test_run_turn_handoff_recomputes_routing_hint_for_next_step() -> None:
    """After a mid-turn mode handoff, the routing hint is recomputed
    against the new mode's ``satisfies_intents`` and appended to the
    swapped system prompt so the model can chain to the next step.

    Regression for the chained-mode-transition gap (2026-07-23): a request
    like "search for the latest coding practices then write a markdown
    report" expresses two intents (research + write). A single handoff to
    ``search`` resolves "research"; the remaining "write" intent is still a
    gap. Without recomputing the hint after the handoff, the model lands in
    search, satisfies research, and stops — it never gets the signal to
    hand off to markdown. With the fix, the recomputed hint names markdown
    as the next target and is appended to the search-mode system prompt.
    """
    call_count = 0
    captured_system_prompts: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        msgs = kwargs.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            captured_system_prompts.append(str(msgs[0].get("content", "")))
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "search"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done after search and markdown.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(
        mode_id="default",
        query="search for the latest coding practices then write a markdown report",
        hooks=hooks,
    )

    assert result.handoff_to == "search"
    # The model must have been called at least twice (handoff + post-handoff).
    assert call_count >= 2
    # The post-handoff system prompt (captured at call_count >= 2) must
    # contain a fresh routing hint naming markdown as the next target.
    assert len(captured_system_prompts) >= 2
    post_handoff_system = captured_system_prompts[-1]
    assert "Routing Hint" in post_handoff_system
    assert "transition_stage" in post_handoff_system
    assert "markdown" in post_handoff_system


@pytest.mark.asyncio
async def test_run_turn_handoff_clears_hint_when_new_mode_satisfies_all() -> None:
    """When the handoff target satisfies all remaining intents, the stale
    Phase-1 hint is cleared so the guards do not fire on the old one.

    Symmetric counterpart to the chained-handoff test: a single-intent
    request handed off to the mode that satisfies that intent must not
    carry the original hint forward into the new mode's system prompt.
    """
    call_count = 0
    captured_system_prompts: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        msgs = kwargs.get("messages", [])
        if msgs and msgs[0].get("role") == "system":
            captured_system_prompts.append(str(msgs[0].get("content", "")))
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "markdown"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Wrote the markdown doc.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(
        mode_id="default",
        query="write a markdown report of the findings",
        hooks=hooks,
    )

    assert result.handoff_to == "markdown"
    assert call_count >= 2
    assert len(captured_system_prompts) >= 2
    post_handoff_system = captured_system_prompts[-1]
    # Markdown satisfies the "write" intent, so no routing hint should be
    # injected post-handoff (the gap is empty).
    assert "Routing Hint" not in post_handoff_system


def test_routing_hint_after_handoff_names_next_target() -> None:
    """Pure unit test (no live catalog): after a handoff to a mode that
    satisfies one of two detected intents, the recomputed hint names the
    mode covering the remaining intent.

    Regression for the chained-mode-transition gap (2026-07-23): the
    handoff branch in ``_run_turn_inner`` must recompute the hint against
    the new mode's ``satisfies_intents``. This test pins the function's
    contract directly — a request with research+write intents, handed off
    to a mode that satisfies only research, must yield a hint naming the
    mode that covers write. No live catalog MCP required.
    """
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "draft"]},
    )
    patched_catalog = {"search": search_cfg, "markdown": markdown_cfg}

    with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
        # After handing off to search (satisfies "research"), the remaining
        # "write" intent is still a gap — the hint must name markdown.
        hint = _compute_capability_routing_hint(
            mode_id="search",
            query="search for the latest coding practices then write a markdown report",
            capability_manifest=_empty_manifest(),
            routing_targets=["markdown"],
            mode_config=search_cfg,
        )

    assert hint != ""
    assert "Routing Hint" in hint
    assert "transition_stage" in hint
    assert "markdown" in hint
    # No fork — only one intent remains after search satisfied research.
    assert "fork:" not in hint


def test_routing_hint_after_handoff_empty_when_new_mode_satisfies_all() -> None:
    """Pure unit test: after a handoff to a mode that satisfies all
    detected intents, the recomputed hint is empty (gap is zero)."""
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    markdown_cfg = _mode_config(mode_id="markdown", satisfies_intents=["write"])
    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    patched_catalog = {"search": search_cfg, "markdown": markdown_cfg}

    with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
        hint = _compute_capability_routing_hint(
            mode_id="markdown",
            query="write a markdown report of the findings",
            capability_manifest=_empty_manifest(),
            routing_targets=["search"],
            mode_config=markdown_cfg,
        )

    assert hint == ""


def test_routing_hint_sequential_query_does_not_fork() -> None:
    """A sequential multi-intent query ("search ... then code ... then
    write") must produce a single handoff to the first step's target, NOT
    a fork. A fork runs children in parallel with the whole query, so none
    can use another's output — it is the wrong mechanism for a chain.

    Regression for the 2026-07-24 fork-failure incident: the model was told
    to `fork:ponder,markdown,search`, each child got the whole query and a
    re-routing hint, the markdown child hit Azure 500 "model produced invalid
    content", and ponder made 0 tool calls. The fix detects sequential
    connectors and prefers a single-handoff so the chain proceeds step by
    step via post-handoff hint recomputation.
    """
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    search_cfg = _mode_config(
        mode_id="search", satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    coding_cfg = _mode_config(
        mode_id="coding", satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["code", "coding", "codebase", "gap analysis"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown", satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "search": search_cfg, "coding": coding_cfg, "markdown": markdown_cfg,
        "default": default_cfg,
    }

    q = ("do a web search for python coding best practices then go into "
         "coding mode and use it as input for a code-level gap analysis "
         "then write a markdown of it.")
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["search", "coding", "markdown"],
            mode_config=default_cfg,
        )

    assert hint != ""
    assert "transition_stage" in hint
    # Sequential: single handoff to the first step (search), NOT a fork.
    assert "fork:" not in hint
    assert "search" in hint


def test_routing_hint_concurrent_query_still_forks() -> None:
    """A concurrent multi-intent query ("remember this and write a markdown")
    has no sequential connectors, so it must still fork — the two intents
    are independent and can run in parallel.
    """
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    learn_cfg = _mode_config(
        mode_id="learn", satisfies_intents=["learn"],
        routing_intent_tags=["learn"],
        routing_intent_keywords={"learn": ["remember", "note", "capture"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown", satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {"learn": learn_cfg, "markdown": markdown_cfg, "default": default_cfg}

    with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="remember this and write a markdown of it",
            capability_manifest=_empty_manifest(),
            routing_targets=["learn", "markdown"],
            mode_config=default_cfg,
        )

    assert hint != ""
    assert "fork:" in hint
    assert "learn" in hint and "markdown" in hint


def test_routing_hint_refuses_fork_for_producer_consumer_mix() -> None:
    """A query whose gap mixes a producer intent (research/reasoning) with
    a consumer intent (write/coding) must NOT fork — the consumer needs
    the producer's output as input, but a parallel fork gives it none.

    Regression for the 2026-07-27 incident: "form an opinion... write it
    to a markdown doc" was routed as ``fork:ponder,markdown,coding,search``.
    The ``markdown`` child had no evidence to write from, re-did the
    research with 7 ``fetch_page`` calls across all 5 iterations, and
    never reached ``write_file``. The kernel now refuses the fork and
    routes sequential (single handoff to the first producer), letting the
    post-handoff hint chain the consumer after.
    """
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    # ponder = producer (satisfies research/reasoning), markdown = consumer
    # (satisfies write). A fork mixing them is sequential-wrong.
    ponder_cfg = _mode_config(
        mode_id="ponder", satisfies_intents=["research", "reasoning"],
        routing_intent_tags=["research", "reasoning"],
        routing_intent_keywords={
            "research": ["search", "latest", "framework"],
            "reasoning": ["opinion", "analyze", "compare"],
        },
    )
    markdown_cfg = _mode_config(
        mode_id="markdown", satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "ponder": ponder_cfg,
        "markdown": markdown_cfg,
        "default": default_cfg,
    }

    with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="analyze the frameworks and write a markdown doc",
            capability_manifest=_empty_manifest(),
            routing_targets=["ponder", "markdown"],
            mode_config=default_cfg,
        )

    assert hint != ""
    # Must NOT fork — the producer/consumer mix is sequential.
    assert "fork:" not in hint
    # Must route to the first producer (ponder), so the post-handoff hint
    # can chain markdown after ponder produces its reasoning.
    assert "ponder" in hint


def test_routing_hint_does_not_re_escalate_to_plan_after_plan_ran() -> None:
    """Once ``plan`` has run this turn, the plan-escalation branch must NOT
    re-fire on the post-handoff hint recompute.

    Regression for the 2026-07-27 glm-5.2 incident: a complex sequential
    query ("research X then write a markdown doc") escalated ``default →
    plan``. After ``plan`` ran, the host unioned ``plan``'s
    ``satisfies_intents: [plan]`` into ``chain_satisfied`` and passed it
    back as ``extra_satisfied``. Without the ``"plan" not in
    current_satisfied`` guard, section 3b re-fired on every recompute
    ("hand off to `plan` again") because ``plan`` is not a detected query
    intent and so never shrinks the real ``{research, write}`` gap. The
    duplicate-hint dedup eventually suppressed it, leaving the model in
    ``coding`` with no path to ``markdown`` — it reached for
    ``coding_create_worktree`` instead. With the guard, the recompute
    falls through to the single-handoff branch and names ``markdown`` for
    the residual ``write`` gap.
    """
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    plan_cfg = _mode_config(
        mode_id="plan",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
        routing_intent_keywords={"plan": ["plan", "workflow", "multi-step"]},
    )
    # coding is the post-plan landing mode: it satisfies "coding" but NOT
    # "write", so the write intent remains a gap after the handoff.
    coding_cfg = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding", "review"],
        routing_intent_tags=["coding", "review"],
        routing_intent_keywords={
            "coding": ["code", "implement", "refactor", "fix"],
            "review": ["review", "audit", "check"],
        },
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "draft"]},
    )
    patched_catalog = {"plan": plan_cfg, "coding": coding_cfg, "markdown": markdown_cfg}

    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog):
        # Simulate the post-plan → coding handoff: chain_satisfied has
        # "plan" (plan ran) and "coding" is the current mode. The residual
        # gap is {write} (markdown covers it).
        hint = _compute_capability_routing_hint(
            mode_id="coding",
            query="research the latest frameworks then write a markdown report",
            capability_manifest=_empty_manifest(),
            routing_targets=["plan", "markdown"],
            mode_config=coding_cfg,
            extra_satisfied={"plan"},  # plan already ran this turn
        )

    assert hint != ""
    # Must NOT re-escalate to plan — plan already ran.
    assert "plan" not in hint
    # Must name markdown for the residual write gap.
    assert "markdown" in hint
    assert "transition_stage" in hint


def test_single_handoff_target_for_gap_names_markdown_for_write_gap() -> None:
    """``_single_handoff_target_for_gap`` resolves the concrete handoff
    target for a residual intent gap, used by the write-effect correction
    nudge.

    Regression for the 2026-07-27 glm-5.2 incident: after landing in
    ``coding`` with a residual ``write`` gap, the model called
    ``coding_create_worktree``. The correction nudge said "call
    transition_stage with the suggested target_stage_id" — but the only
    target in scope was the stale ``plan`` hint, so the model had no path
    to ``markdown``. With this helper, the nudge names ``markdown``
    explicitly so the model has a concrete escape hatch.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _single_handoff_target_for_gap

    coding_cfg = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding", "review"],
        routing_intent_tags=["coding", "review"],
        routing_intent_keywords={
            "coding": ["code", "implement", "refactor", "fix"],
            "review": ["review", "audit", "check"],
        },
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "draft"]},
    )
    patched_catalog = {"coding": coding_cfg, "markdown": markdown_cfg}

    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog):
        target = _single_handoff_target_for_gap(
            query="research the latest frameworks then write a markdown report",
            routing_targets=["markdown"],
            mode_config=coding_cfg,
            extra_satisfied={"plan"},  # plan already ran
        )

    # markdown covers the whole residual {write} gap → single handoff.
    assert target == "markdown"


def test_single_handoff_target_for_gap_returns_none_when_no_gap() -> None:
    """When the current mode already satisfies all detected intents (no
    residual gap), ``_single_handoff_target_for_gap`` returns ``None`` so
    the nudge falls back to its generic message.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _single_handoff_target_for_gap

    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "draft"]},
    )
    patched_catalog = {"markdown": markdown_cfg}

    with patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog):
        target = _single_handoff_target_for_gap(
            query="write a markdown document",
            routing_targets=["markdown"],
            mode_config=markdown_cfg,  # satisfies write → no gap
        )

    assert target is None


def test_query_is_sequential_detects_connectors() -> None:
    """The sequential connector detector recognizes the built-in fallback
    connectors ('then', 'and then', 'after that', 'next', 'followed by').
    """
    from kernel.turn_runner import _query_is_sequential

    assert _query_is_sequential("search for X then write a markdown")
    assert _query_is_sequential("search for X and then write a markdown")
    assert _query_is_sequential("search for X after that write a markdown")
    assert _query_is_sequential("search for X next write a markdown")
    assert _query_is_sequential("search for X followed by write a markdown")
    # No connector -> concurrent.
    assert not _query_is_sequential("remember this and write a markdown")
    assert not _query_is_sequential("hello world")


def test_query_is_sequential_uses_session_vocabulary() -> None:
    """A session-injected connector ('based on') is detected as sequential
    even though the fallback list does not include it. Regression for the
    2026-07-26 'markdown based on the results' false-fork incident: the
    query expressed a data dependency the fallback vocabulary missed, so
    the routing hint forked search+markdown in parallel and the markdown
    child had no access to the search child's output.
    """
    from kernel.routing_vocabulary import register_routing_vocabulary
    from kernel.turn_runner import _query_is_sequential

    # Without the session vocabulary, 'based on' is NOT sequential — the
    # fallback list does not include it.
    assert not _query_is_sequential("search for X based on the results write a markdown")

    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by", "based on"]
    )
    try:
        assert _query_is_sequential("search for X based on the results write a markdown")
        # Common-word false-positive guard: 'after' alone is not in the
        # session list, so a query using it as a noun is not sequential.
        assert not _query_is_sequential("search for after-school programs and write a markdown")
    finally:
        register_routing_vocabulary(sequential_connectors=None)


def test_routing_hint_sequential_via_session_vocabulary_does_not_fork() -> None:
    """A 'based on the results' query with the session vocabulary wired
    produces a single handoff to the first step, NOT a fork — the
    end-to-end regression for the 2026-07-26 false-fork incident.
    """
    from unittest.mock import patch

    from kernel.routing_vocabulary import register_routing_vocabulary
    from kernel.turn_runner import _compute_capability_routing_hint

    search_cfg = _mode_config(
        mode_id="search", satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown", satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "search": search_cfg, "markdown": markdown_cfg, "default": default_cfg,
    }

    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by", "based on"]
    )
    try:
        q = "search for python news and write a markdown based on the results"
        with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
            hint = _compute_capability_routing_hint(
                mode_id="default", query=q, capability_manifest=_empty_manifest(),
                routing_targets=["search", "markdown"],
                mode_config=default_cfg,
            )
        assert hint != ""
        assert "transition_stage" in hint
        assert "fork:" not in hint
        assert "search" in hint
    finally:
        register_routing_vocabulary(sequential_connectors=None)


def test_routing_hint_chain_memory_skips_completed_steps() -> None:
    """After a sequential chain has handed off through search and coding,
    the post-handoff hint must name markdown (the remaining step), NOT
    re-suggest search — even though "search" still appears in the original
    query and coding's ``satisfies_intents`` is only ["coding"].

    Regression for the 2026-07-24 chain-stall: the post-search hint correctly
    named coding, but the post-coding hint re-detected "research" from the
    query text and re-suggested search, creating an infinite loop. The fix
    passes ``extra_satisfied`` (the cumulative set of intents from all modes
    entered during the turn) so already-completed steps are not re-suggested.
    """
    from unittest.mock import patch

    from kernel.turn_runner import _compute_capability_routing_hint

    search_cfg = _mode_config(
        mode_id="search", satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    coding_cfg = _mode_config(
        mode_id="coding", satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["coding", "code", "codebase", "gap analysis"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown", satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
    )
    patched_catalog = {
        "search": search_cfg, "coding": coding_cfg, "markdown": markdown_cfg,
    }
    q = ("do a web search for python coding best practices then go into "
         "coding mode and use it as input for a code-level gap analysis "
         "then write a markdown of it.")

    # After search + coding: chain_satisfied = {research, coding}.
    # Gap = {write}. Hint must name markdown, NOT search.
    with patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog):
        hint = _compute_capability_routing_hint(
            mode_id="coding", query=q, capability_manifest=_empty_manifest(),
            routing_targets=["search", "markdown"],
            mode_config=coding_cfg,
            extra_satisfied={"research", "coding"},
        )

    assert hint != ""
    assert "markdown" in hint
    assert "search" not in hint
    assert "fork:" not in hint


# ── Proactive capability routing hint tests ─────────────────────────


def _empty_manifest() -> CapabilityManifest:
    """A manifest with no entries — simulates a mode with no tools."""
    return CapabilityManifest(generated_at="2026-01-01T00:00:00Z", entries=[])


def _manifest_with_search() -> CapabilityManifest:
    """A manifest advertising a web_search capability."""
    from kernel.models import CapabilityEntry

    return CapabilityManifest(
        generated_at="2026-01-01T00:00:00Z",
        entries=[
            CapabilityEntry(
                capability_id="web_search",
                kind="tool",
                status="available",
                source="mcp_pool",
            )
        ],
    )


def _mode_config(
    *,
    mode_id: str = "default",
    satisfies_intents: list[str] | None = None,
    required_capabilities: list[str] | None = None,
    routing_intent_tags: list[str] | None = None,
    routing_intent_keywords: dict[str, list[str]] | None = None,
    routable: bool = True,
) -> WorkflowModeConfig:
    """A minimal WorkflowModeConfig for routing-hint tests.

    Only the fields the hint function reads are populated. The default
    ``satisfies_intents=None`` → empty list (every detected intent is a
    gap), matching the behavior of modes that have not opted in (e.g.
    ``default``) and preserving the pre-field semantics for existing tests.
    ``required_capabilities=None`` → empty list (no capability gate).
    ``routing_intent_tags``/``routing_intent_keywords`` default to empty —
    the hint detector attributes intents from a target's patterns/keywords,
    so tests that need a target to "cover" an intent must populate these.
    ``routable`` defaults to True so ``get_routing_targets`` returns the
    target set; pass ``routable=False`` to simulate a non-routable mode.
    """
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=routable,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        required_capabilities=list(required_capabilities) if required_capabilities else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def test_routing_hint_returns_empty_when_no_routing_targets() -> None:
    """No routing targets means no hint is injected (zero tokens)."""
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="understand the latest frameworks and do a SWOT",
        capability_manifest=_empty_manifest(),
        routing_targets=[],
    )
    assert hint == ""


def test_routing_hint_returns_empty_when_mode_has_search_and_no_reasoning() -> None:
    """A mode that already satisfies the research intent gets no hint.

    The gap check is declarative: the current mode authors
    ``satisfies_intents`` in its _mode.md front matter, and the kernel
    believes it. The live capability manifest is no longer consulted for
    the gap check — that coarse ``kind == "tool"`` heuristic treated any
    tool-bearing mode as satisfying every intent, which suppressed
    handoffs from tool-rich modes that lacked the specific intent's tools.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="search",
        query="what is the latest news on rust",
        capability_manifest=_manifest_with_search(),
        routing_targets=["default", "ponder"],
        mode_config=_mode_config(satisfies_intents=["research"]),
    )
    assert hint == ""


def test_routing_hint_prefers_ponder_for_research_then_reason() -> None:
    """A query pairing a research cue with a reasoning cue (understand +
    SWOT + latest) should hand off to ponder, not the generic target list.

    This is the regression case for the incident where 'latest frameworks
    and do a SWOT' was routed to the one-shot search action mode. The
    kernel is a pure mapper: the decision (ponder, not search) is driven
    entirely by the pack-authored ``routing_intent_tags`` on ponder
    ([research, reasoning]) vs search ([research]) — ponder covers both
    detected intents, search covers only one. The kernel emits no
    semantic steering prose ("do not use the one-shot search action
    mode"); it names only the mechanics (transition_stage + ponder). Any
    such prose would be pack-authored routing *input*, not kernel-emitted
    routing *output*.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="get a better understanding of the latest frameworks and agent building tools on the market today and do a SWOT based on it",
        capability_manifest=_empty_manifest(),
        routing_targets=["search", "ponder", "markdown", "coding"],
    )
    assert hint != ""
    assert "ponder" in hint
    assert "transition_stage" in hint
    # The kernel must not narrate the mode's purpose or steer away from
    # other modes in its own prose — that would be pack coupling. The
    # handoff instruction is purely mechanical.
    assert "one-shot" not in hint.lower()
    assert "researches then reasons" not in hint.lower()


def test_routing_hint_detects_analyze_tradeoff_intent() -> None:
    """'analyze the latest tradeoffs' is research-then-reason → ponder."""
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="analyze the latest tradeoffs in serverless frameworks",
        capability_manifest=_empty_manifest(),
        routing_targets=["search", "ponder"],
    )
    assert "ponder" in hint


def test_routing_hint_falls_back_to_generic_list_for_tool_gap() -> None:
    """A pure tool-gap query (write a file) with no reasoning cue lists all
    targets rather than singling out ponder.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="write a file with the results",
        capability_manifest=_empty_manifest(),
        routing_targets=["markdown", "coding", "ponder"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    # No ponder-specific steering copy
    assert "one-shot" not in hint.lower()


def test_routing_hint_search_gap_without_reasoning_lists_targets() -> None:
    """A pure 'search the web' query (no reasoning cue) gets the generic
    target list, not the ponder-preferred copy.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="search the web for cloud costs",
        capability_manifest=_empty_manifest(),
        routing_targets=["search", "ponder"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    assert "one-shot" not in hint.lower()


# ── Two-goal fork routing hint tests ──────────────────────────────────


def test_routing_hint_forks_for_remember_and_write_markdown() -> None:
    """A request pairing remember/learn with write-a-markdown-doc has two
    concurrent goals. The hint must steer the model to a fork fan-out
    (`fork:learn,markdown`), not a single-mode handoff.

    Regression case for docs/lisa-markdown-stage-failure-2026-07-17.md: the
    model called markdown_forge_document / markdown_generate from default
    mode (mode-scoped tools, rejected by tool policy) instead of forking.
    The proactive hint must surface before the model wastes an iteration.
    The kernel is a pure mapper: the decision (fork, not single handoff)
    is driven entirely by set algebra over the pack-authored
    routing_intent_tags (learn covers 'learn', markdown covers 'write',
    neither covers both). The kernel emits no tool-policy narration
    ('mode-scoped tools', 'rejected'); it names only the mechanics
    (transition_stage + the fork target id). Any such narration would be
    kernel-authored routing *output*, which is pack coupling.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="remember that pretty much verbatim and also write a markdown of it",
        capability_manifest=_empty_manifest(),
        routing_targets=["learn", "markdown", "coding", "ponder", "search"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    # Must name the fork target, not a single mode handoff
    assert "fork:learn,markdown" in hint
    # The kernel must not narrate tool-policy semantics in its own prose.
    # The handoff instruction is purely mechanical.
    assert "mode-scoped" not in hint.lower()
    assert "rejected" not in hint.lower()


def test_routing_hint_fork_requires_both_routable_modes() -> None:
    """When one of learn/markdown is not a routing target, do not suggest a
    fork — fall back to the single-goal branches (or no hint).
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    # markdown not routable from here → no fork hint; learn+write cues with
    # no tool/search/research-reason gap means no hint at all.
    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="remember the decision and write a markdown of it",
        capability_manifest=_empty_manifest(),
        routing_targets=["learn", "coding"],  # markdown missing
    )
    assert "fork:" not in hint


def test_routing_hint_fork_preserves_target_case() -> None:
    """The fork target string preserves the original (non-lowercased) mode
    IDs from routing_targets, matching what the FSM catalog expects.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="take a note of this and write a markdown document",
        capability_manifest=_empty_manifest(),
        routing_targets=["learn", "markdown"],
    )
    assert "fork:learn,markdown" in hint


def test_routing_hint_forks_from_coding_mode() -> None:
    """From coding mode, a remember+write-markdown request must still steer
    to fork:learn,markdown once the routing FSM is a full mesh.

    Regression for docs/lisa-coding-mode-fork-failure-2026-07-17.md: with the
    conservative default-only edge set, coding's routing_targets were
    ["default"] and the fork branch's "learn"/"markdown" membership guard
    never passed. The model then fell back to coding_create_worktree
    (approval-gated), producing a worktree approval loop instead of a fork.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="coding",
        query="remember the entire thing and write a markdown document of it",
        capability_manifest=_empty_manifest(),
        # Post-mesh coding routing targets include learn + markdown.
        routing_targets=["default", "learn", "markdown", "ponder", "search"],
    )
    assert hint != ""
    assert "transition_stage" in hint
    assert "fork:learn,markdown" in hint


# ── Single-intent handoff tests (satisfies_intents) ──────────────────
#
# The 2026-07-17 fork fix only unblocked the *two-goal* learn+markdown
# fork from coding. A *single-intent* "write a markdown document" request
# (no "remember" cue) should produce a single coding → markdown handoff,
# not a fork and not an approval-gated coding_create_worktree fallback.
# The gap check must be declarative (mode.satisfies_intents), not the
# coarse capability-manifest "any tool kind" heuristic that treated
# coding's 11 coding_* tools as satisfying the write intent.


def test_routing_hint_single_handoff_coding_to_markdown() -> None:
    """A single-intent write-markdown request from coding mode must still
    steer to markdown, not be suppressed by coding's tool-bearing manifest.

    Regression for the 2026-07-19 coding-mode worktree-approval loop
    (single-intent variant): coding mode advertises 11 tools, all
    ``kind="tool"``, none of which can produce a markdown document. Under
    the old manifest-driven gap check, ``has_tool_capability = True``
    suppressed ``needs_write``, the hint returned "" at the step-2 early
    return, and the model fell back to the approval-gated
    ``coding_create_worktree``. With the declarative ``satisfies_intents``
    check, coding's declared intents (here: just "coding", explicitly NOT
    "write") leave the write intent as a gap, the single-handoff branch
    (step 5) fires, and the hint names markdown.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    coding_manifest = _manifest_with_coding_tools()
    hint = _compute_capability_routing_hint(
        mode_id="coding",
        query="write a markdown document of the report you just produced",
        capability_manifest=coding_manifest,
        routing_targets=["default", "learn", "markdown", "ponder", "search"],
        mode_config=_mode_config(
            mode_id="coding",
            # Coding declares it handles coding intent in-place, NOT write.
            # The pack authors this in coding/_mode.md; the kernel believes it.
            satisfies_intents=["coding"],
        ),
    )
    assert hint != ""
    assert "transition_stage" in hint
    assert "markdown" in hint
    # Single handoff, not a fork — only one intent was detected.
    assert "fork:" not in hint


def test_routing_hint_suppressed_when_mode_satisfies_write_intent() -> None:
    """If the current mode declares ``satisfies_intents: [write]``, a
    write-markdown request produces no hint — the mode handles it in-place.

    This is the symmetric counterpart: just as ``routing_intent_tags``
    declares what a mode covers when routing *to* it, ``satisfies_intents``
    declares what a mode handles when routing *from* it. Markdown mode
    itself would author ``satisfies_intents: [write]``; a write request
    issued from markdown mode should not trigger a handoff hint back to
    itself or anywhere else.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="markdown",
        query="write a markdown document of the report you just produced",
        capability_manifest=_empty_manifest(),
        routing_targets=["default", "learn", "coding", "ponder", "search"],
        mode_config=_mode_config(
            mode_id="markdown",
            satisfies_intents=["write"],
        ),
    )
    assert hint == ""


def test_routing_hint_single_handoff_fires_even_with_rich_manifest() -> None:
    """The gap check ignores the live capability manifest entirely.

    A coding-style manifest with 11 tool entries must NOT suppress the
    write-intent gap. The declarative ``satisfies_intents`` declaration is
    the sole source of truth. This pins the regression: the old
    ``has_tool_capability = "tool" in available_kinds`` check read the
    manifest and suppressed the gap for any tool-bearing mode; the new
    check reads ``mode_config.satisfies_intents`` and is manifest-blind.
    """
    from kernel.turn_runner import _compute_capability_routing_hint

    hint_with_rich_manifest = _compute_capability_routing_hint(
        mode_id="coding",
        query="draft a markdown document of the findings",
        capability_manifest=_manifest_with_coding_tools(),
        routing_targets=["default", "learn", "markdown", "ponder", "search"],
        mode_config=_mode_config(mode_id="coding", satisfies_intents=["coding"]),
    )
    hint_with_empty_manifest = _compute_capability_routing_hint(
        mode_id="coding",
        query="draft a markdown document of the findings",
        capability_manifest=_empty_manifest(),
        routing_targets=["default", "learn", "markdown", "ponder", "search"],
        mode_config=_mode_config(mode_id="coding", satisfies_intents=["coding"]),
    )
    # Both must fire and produce identical steering — the manifest is
    # irrelevant to the gap decision under the declarative check.
    assert hint_with_rich_manifest != ""
    assert "markdown" in hint_with_rich_manifest
    assert hint_with_rich_manifest == hint_with_empty_manifest


def _manifest_with_coding_tools() -> CapabilityManifest:
    """A manifest advertising coding's 11 allowed tools, all kind='tool'.

    Mirrors ``lisa_packs/coding/modes/coding/_mode.md`` allowed_tool_ids.
    Under the old manifest-driven gap check this suppressed ``needs_write``
    because ``"tool" in available_kinds`` was True. Under the declarative
    check it must have no effect on the gap decision.
    """
    from kernel.models import CapabilityEntry

    coding_tool_ids = [
        "coding_list_directory",
        "coding_find_files",
        "coding_read_file",
        "coding_search_code",
        "coding_outline_source",
        "coding_create_worktree",
        "coding_apply_patch",
        "coding_run_check",
        "coding_status",
        "coding_commit",
        "coding_push",
        "coding_cleanup_worktree",
        "github_open_pull_request",
    ]
    return CapabilityManifest(
        generated_at="2026-01-01T00:00:00Z",
        entries=[
            CapabilityEntry(
                capability_id=tid,
                kind="tool",
                status="available",
                source="tool_catalog",
            )
            for tid in coding_tool_ids
        ],
    )


# ── Capability gate tests (required_capabilities) ─────────────────────
#
# The gate validates the resolved model's declared capabilities against the
# mode's required_capabilities before the tool loop starts. When a router is
# wired and the model is missing a required capability, the turn returns a
# clear user-facing block instead of letting a weak model reach for
# write-effect tools it cannot use reliably.


def test_capability_gate_blocks_weak_model_when_router_wired() -> None:
    """A mode requiring ``tools`` with a text-only model + a wired router
    returns a blocked TurnResult naming the missing capability.
    """
    from kernel.turn_runner import _validate_model_capabilities

    mode_config = _mode_config(mode_id="coding", required_capabilities=["tools"])
    model_def = _StubModel(model_id="gemma4:cloud", capabilities=("text",))
    block = _validate_model_capabilities(
        mode_config=mode_config,
        model_def=model_def,
        preferred_model_id="gemma4:cloud",
        router_wired=True,
    )
    assert block is not None
    assert block.missing == ("tools",)
    assert block.result.blocked is True
    assert "gemma4:cloud" in block.result.response_text
    assert "tools" in block.result.response_text
    assert block.result.tool_call_count == 0


def test_capability_gate_passes_when_model_capable() -> None:
    """A model that declares the required capabilities is admitted."""
    from kernel.turn_runner import _validate_model_capabilities

    mode_config = _mode_config(mode_id="coding", required_capabilities=["tools", "code"])
    model_def = _StubModel(model_id="gpt-5.3-codex", capabilities=("text", "tools", "code"))
    block = _validate_model_capabilities(
        mode_config=mode_config,
        model_def=model_def,
        preferred_model_id="gpt-5.3-codex",
        router_wired=True,
    )
    assert block is None


def test_capability_gate_skips_when_no_required_capabilities() -> None:
    """A mode with no required_capabilities never blocks (backward compat)."""
    from kernel.turn_runner import _validate_model_capabilities

    mode_config = _mode_config(mode_id="default")  # no required_capabilities
    model_def = _StubModel(model_id="weak", capabilities=())
    block = _validate_model_capabilities(
        mode_config=mode_config,
        model_def=model_def,
        preferred_model_id="weak",
        router_wired=True,
    )
    assert block is None


def test_capability_gate_advisory_when_no_router_wired() -> None:
    """When no router is wired, the gate is advisory — no block, even if the
    model is missing required capabilities. A host without a router has no
    escalation path, so blocking would dead-end.
    """
    from kernel.turn_runner import _validate_model_capabilities

    mode_config = _mode_config(mode_id="coding", required_capabilities=["tools"])
    model_def = _StubModel(model_id="weak", capabilities=("text",))
    block = _validate_model_capabilities(
        mode_config=mode_config,
        model_def=model_def,
        preferred_model_id="weak",
        router_wired=False,
    )
    assert block is None


def test_capability_gate_skips_when_mode_config_none() -> None:
    """No mode config → no gate (backward compat for hosts without modes)."""
    from kernel.turn_runner import _validate_model_capabilities

    model_def = _StubModel(model_id="weak", capabilities=())
    block = _validate_model_capabilities(
        mode_config=None,
        model_def=model_def,
        preferred_model_id="weak",
        router_wired=True,
    )
    assert block is None


def test_capability_gate_treats_missing_capabilities_field_as_empty() -> None:
    """A model_def without a ``capabilities`` attribute is treated as having
    no declared capabilities — only matters when the mode requires caps.
    """
    from kernel.turn_runner import _validate_model_capabilities

    mode_config = _mode_config(mode_id="coding", required_capabilities=["tools"])
    # A bare object with only model_id — no capabilities attribute.
    model_def = type("Bare", (), {"model_id": "stub"})()
    block = _validate_model_capabilities(
        mode_config=mode_config,
        model_def=model_def,
        preferred_model_id="stub",
        router_wired=True,
    )
    assert block is not None
    assert "tools" in block.missing


@dataclass(frozen=True, slots=True)
class _StubModel:
    """Minimal model stand-in carrying capabilities for gate tests."""

    model_id: str
    capabilities: tuple[str, ...] = ()


# ── Routing-hint write-tool guard tests ──────────────────────────────


def test_tool_effect_reads_write_effect_from_tool_catalog() -> None:
    """``_tool_effect`` returns ``write`` for a tool whose catalog definition
    declares ``effect: write``.
    """
    from kernel.turn_runner import _tool_effect

    tool_def = type("Tool", (), {"effect": "write"})()
    assert (
        _tool_effect(
            "coding_create_worktree",
            operation_catalog={},
            tool_catalog={"coding_create_worktree": tool_def},
        )
        == "write"
    )


def test_tool_effect_reads_write_effect_from_operation_catalog() -> None:
    """``_tool_effect`` prefers the operation catalog on id collision."""
    from kernel.turn_runner import _tool_effect

    op_def = type("Op", (), {"effect": "write"})()
    assert (
        _tool_effect(
            "some_op",
            operation_catalog={"some_op": op_def},
            tool_catalog={},
        )
        == "write"
    )


def test_tool_effect_defaults_read_for_unknown_tool() -> None:
    """Unknown tools (no catalog entry) default to ``read`` so the
    routing-hint guard only fires on declared write-effect tools.
    """
    from kernel.turn_runner import _tool_effect

    assert _tool_effect("unknown", operation_catalog={}, tool_catalog={}) == "read"


def test_tool_effect_defaults_read_when_definition_has_no_effect() -> None:
    """A definition without an ``effect`` attribute defaults to ``read``."""
    from kernel.turn_runner import _tool_effect

    bare_def = type("Bare", (), {})()
    assert (
        _tool_effect(
            "x",
            operation_catalog={"x": bare_def},
            tool_catalog={},
        )
        == "read"
    )


def test_tool_effect_returns_external_write() -> None:
    """``external_write`` is treated as a write-effect for the guard."""
    from kernel.turn_runner import _tool_effect

    tool_def = type("Tool", (), {"effect": "external_write"})()
    assert (
        _tool_effect(
            "github_open_pull_request",
            operation_catalog={},
            tool_catalog={"github_open_pull_request": tool_def},
        )
        == "external_write"
    )


# ── 2026-07-25 incident regression tests ───────────────────────────────
#
# These tests cover the three failure layers that produced the "nonsense
# crew-directory answer": (1) the handoff entry-op failure surfacing +
# blocking instead of swallowing + continuing, (2) the routing-hint
# loop backstops (max-hops + duplicate-hint dedup), (3) the ``no_tools``
# fork child aborting the user-facing turn instead of advisory prose.


@pytest.mark.asyncio
async def test_handoff_entry_op_failure_blocks_turn() -> None:
    """When an action mode's entry operation fails (e.g. ``web_search``
    returned no content), the turn must NOT continue in the data-less
    target mode. It must break with a deterministic failure message and
    ``blocked=True`` so the user sees the provider failure and re-issues,
    instead of the model drifting into an unrelated action mode and
    authoring a wrong artifact (the ``crew/`` incident).

    Setup: the model hands off to ``search``; ``search``'s entry operation
    (``web_search``) is stubbed to raise. Before the fix, the turn continued
    in ``search`` with ``entry_output = "Entry operation failed: ..."`` and
    the model kept going. After the fix, the turn breaks with a failure.
    """
    call_count = {"n": 0}

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "search", "rationale": "research"}',
                        },
                    }
                ],
            )
        # If the turn erroneously continues, the model would produce text
        # here. The fix must prevent reaching this completion.
        return _StubCompletion(content="I will create a crew directory.")

    async def failing_entry_operation(operation_id: str, _ctx: OperationExecutionContext) -> OperationResult:
        # Only the ``web_search`` entry operation (the search mode's entry
        # action) fails. Other operations (e.g. pre-turn memory recall) use
        # the default stub so they don't trip the guard prematurely.
        if operation_id == "web_search":
            raise RuntimeError("web_search returned no content")
        return OperationResult(output_text=f"Executed {operation_id}")

    hooks = _build_minimal_hooks(two_step)
    hooks.execute_operation = failing_entry_operation

    result = await run_turn(
        mode_id="default", query="search for agent frameworks and write a review", hooks=hooks
    )

    # The turn must break with a deterministic failure, not continue.
    assert result.blocked is True
    assert "search" in result.response_text
    assert "failed" in result.response_text.lower()
    # The model must NOT have been allowed to drift into authoring — the
    # response is the framework's deterministic failure, not the model's
    # free text "I will create a crew directory." The error reason may
    # quote the bad model output, but the response must be the framework
    # message, not the model's text standing alone.
    assert "Reconfigure" in result.response_text or "re-issue" in result.response_text
    # The model must NOT have produced its free-text as the sole response.
    assert result.response_text != "I will create a crew directory."


@pytest.mark.asyncio
async def test_no_tools_fork_child_aborts_turn() -> None:
    """A fork child marked ``no_tools`` (hallucinated success) must abort
    the user-facing turn with a deterministic failure, not be handed to
    the model as advisory prose. The model cannot be trusted to surface
    it honestly — it may paraphrase the failure into a false "completed"
    (the 2026-07-25 incident).

    Extends ``test_run_turn_fork_marks_no_tools_child_when_action_mode_skipped_tools``
    (which only asserts the child is *labeled* ``no_tools``) to also
    assert the parent aborts with ``blocked=True`` and a failure
    ``response_text``.
    """
    call_count = {"n": 0}

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "fork:learn,markdown"}',
                        },
                    }
                ],
            )
        # If the turn erroneously continues, the model would emit a false
        # success here. The fix must prevent reaching this completion.
        return _StubCompletion(content="Done — wrote it to disk.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(
        mode_id="default", query="remember this and write a markdown", hooks=hooks
    )

    # The children are labeled no_tools (existing assertion, still holds).
    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    for child in result.fork_results:
        assert child.status == "no_tools"
    # NEW: the parent must abort, not continue as advisory prose.
    assert result.blocked is True
    assert "failed" in result.response_text.lower() or "unverified" in result.response_text.lower()
    # The model must NOT have been allowed to emit its false "Done — wrote it to disk."
    assert result.response_text != "Done — wrote it to disk."


def test_max_handoffs_per_turn_setting_is_bound() -> None:
    """The hard backstop setting exists and is a sane bound. This guards
    against accidental removal — the loop guard depends on it."""
    from kernel.config import settings as kernel_settings

    value = kernel_settings.max_handoffs_per_turn
    assert isinstance(value, int)
    assert 3 <= value <= 20


def test_duplicate_hint_limit_setting_is_bound() -> None:
    """The duplicate-hint dedup threshold exists and is small (1-3)."""
    from kernel.config import settings as kernel_settings

    value = kernel_settings.duplicate_hint_limit
    assert isinstance(value, int)
    assert 1 <= value <= 3


def test_hint_cycle_min_len_setting_is_bound() -> None:
    """The hint-cycle detection threshold exists and is 3 (A, B, A)."""
    from kernel.config import settings as kernel_settings

    value = kernel_settings.hint_cycle_min_len
    assert isinstance(value, int)
    assert value == 3


def test_extract_hint_target_parses_single_handoff() -> None:
    """``_extract_hint_target`` pulls the target id from a single-handoff hint."""
    from kernel.turn_runner import _extract_hint_target

    hint = (
        "## Routing Hint\n\n"
        "Your current mode may lack the tools needed for this request. "
        "Hand off to `markdown` by calling `transition_stage` "
        "with target_stage_id `markdown`."
    )
    assert _extract_hint_target(hint) == "markdown"


def test_extract_hint_target_parses_fork_target() -> None:
    """``_extract_hint_target`` pulls the fork target id from a fork hint."""
    from kernel.turn_runner import _extract_hint_target

    hint = (
        "## Routing Hint\n\n"
        "Hand off by calling `transition_stage` with "
        "target_stage_id `fork:learn,markdown`."
    )
    assert _extract_hint_target(hint) == "fork:learn,markdown"


def test_extract_hint_target_returns_empty_for_no_marker() -> None:
    """``_extract_hint_target`` returns ``""`` when no target marker is present."""
    from kernel.turn_runner import _extract_hint_target

    assert _extract_hint_target("") == ""
    assert _extract_hint_target("no marker here") == ""


def test_turn_state_has_loop_guard_fields() -> None:
    """The ``_TurnState`` dataclass exposes the new loop-guard / failure
    fields the turn runner reads. Guards against the fields being
    silently dropped during a future refactor of _turn_state.py."""
    from kernel.turn_phases._turn_state import _TurnState

    state = _TurnState()
    assert state.entry_failed is False
    assert state.entry_failed_mode == ""
    assert state.entry_failed_reason == ""
    assert state.handoff_count == 0
    assert state.last_hint_text == ""
    assert state.duplicate_hint_count == 0
    assert state.hint_target_history == []
    assert state.forced_plan_target == ""
    assert state.non_plan_tool_dispatches == 0


@pytest.mark.asyncio
async def test_forced_plan_enforcer_when_model_refuses_to_hand_off() -> None:
    """When a plan-escalation routing hint is active and the model calls
    non-``transition_stage`` tools instead of handing off to ``plan``,
    the kernel forces the plan handoff after
    ``forced_plan_non_transition_tool_cap`` non-transition dispatches.

    Regression for the 2026-07-30 incident: the model burned 15
    iterations of tool-cycling in ``default`` mode (≈400k tokens) before
    finally handing off to ``plan``, which then failed to parse. The
    forced-plan enforcer bounds the pre-plan token burn to roughly
    ``cap × per-iteration-tokens`` by synthesizing the
    ``transition_stage(plan)`` call itself.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult

    default_cfg = _mode_config(mode_id="default")
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {
        "default": default_cfg, "plan": plan_cfg, "search": search_cfg,
    }

    call_count = 0
    plan_entry_calls = 0
    forced_plan_fired = False

    async def complete(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count, forced_plan_fired
        call_count += 1
        # Inspect messages to detect which mode we're in.
        sys_content = str(kwargs["messages"][0].get("content", ""))
        # In default mode with a plan-escalation hint: the model does
        # read-only fetches instead of calling transition_stage.
        is_plan_mode = "## Handoff Context" in sys_content and "`plan`" in sys_content
        is_search_mode = (
            "## Handoff Context" in sys_content and "`search`" in sys_content
        ) or ("Handed off to mode: search" in str(
            [m.get("content", "") for m in kwargs["messages"]]
        ))
        if is_search_mode:
            return _StubCompletion(content="Search results from web_search.")
        if is_plan_mode:
            # The kernel's mid-turn entry-op detection sets
            # pending_planned_handoff; the next iteration synthesizes the
            # transition_stage call. Return harmless text.
            return _StubCompletion(content="Planning...")
        # default mode: the model does read-only fetches instead of
        # handing off to plan. After forced_plan_non_transition_tool_cap
        # (default 2) non-transition dispatches, the kernel forces the
        # plan handoff.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{call_count}",
                    "function": {
                        "name": "fetch_page",
                        "arguments": '{"url": "https://example.com"}',
                    },
                }
            ],
        )

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        nonlocal plan_entry_calls
        plan_entry_calls += 1
        return OperationResult(
            output_text="research intent detected",
            output_json={"target_stage_id": "search", "rationale": "research intent"},
        )

    hooks = _build_minimal_hooks(complete)
    hooks.memory_recall_enabled = False

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for rust news then write a summary then code it",
            hooks=hooks,
        )

    # The kernel forced the plan handoff — the final response came from
    # the search mode (the plan_workflow target), not from the default
    # mode's read-only fetches.
    assert result.response_text == "Search results from web_search."
    # The planner entry op ran exactly once (for the forced plan handoff).
    assert plan_entry_calls == 1
    # The model did NOT burn the full 15-iteration budget in default mode
    # before the kernel forced the handoff. With cap=2, the kernel forces
    # the plan handoff on the 3rd non-transition dispatch (after 2
    # fetch_page calls), so the total model calls should be well under
    # the 15-iteration cap.
    assert call_count < 10, (
        f"expected the forced-plan enforcer to fire early; got {call_count} "
        "model calls (the model burned the full iteration budget)"
    )


@pytest.mark.asyncio
async def test_budget_briefing_injected_into_system_prompt() -> None:
    """The per-iteration budget briefing is injected into the system
    prompt (``messages[0]``) so the model can self-scope its work within
    the remaining iteration envelope.

    The briefing is super-brief (a single ``## Turn Budget`` block with
    dot-separated counts) and is replaced — not accumulated — each
    iteration. Iteration 1 shows the full contract; later iterations
    show only the delta (partially-consumed per-target limits).
    """
    captured_system_prompts: list[str] = []
    call_count = 0

    async def capture_and_finish(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Capture the system prompt each iteration.
        captured_system_prompts.append(str(kwargs["messages"][0].get("content", "")))
        # First call: return a tool call so we get a second iteration.
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            )
        # Second call: finish with text.
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(capture_and_finish)
    hooks.execute_operation = _stub_execute_operation
    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=hooks,
        max_tool_iterations=5,
    )

    assert result.response_text == "Done."
    # Two model calls → two captured system prompts.
    assert len(captured_system_prompts) == 2
    # Iteration 1: the briefing must be present.
    assert "## Turn Budget" in captured_system_prompts[0]
    # The briefing reports iterations remaining (4/5 after iteration 1
    # incremented to 1).
    assert "iters: 4/5" in captured_system_prompts[0]
    # Iteration 2: the briefing must still be present (replaced, not
    # accumulated — only one ``## Turn Budget`` block).
    assert "## Turn Budget" in captured_system_prompts[1]
    assert captured_system_prompts[1].count("## Turn Budget") == 1
    # Iteration 2: iterations remaining is now 3/5.
    assert "iters: 3/5" in captured_system_prompts[1]
    # The base system prompt content (mode intro) is preserved — the
    # briefing is appended, not replacing the base.
    assert captured_system_prompts[0].split("## Turn Budget")[0].strip() != ""


@pytest.mark.asyncio
async def test_budget_briefing_disabled_when_config_off() -> None:
    """When ``budget_briefing_enabled`` is False, no budget briefing is
    injected into the system prompt."""
    from kernel.config import settings as kernel_settings

    captured_system_prompts: list[str] = []
    call_count = 0

    async def capture_and_finish(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        captured_system_prompts.append(str(kwargs["messages"][0].get("content", "")))
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(capture_and_finish)
    hooks.execute_operation = _stub_execute_operation
    # Patch the kernel settings singleton to disable the briefing.
    original = kernel_settings.budget_briefing_enabled
    kernel_settings.budget_briefing_enabled = False
    try:
        result = await run_turn(
            mode_id="default",
            query="hello",
            hooks=hooks,
            max_tool_iterations=5,
        )
    finally:
        kernel_settings.budget_briefing_enabled = original

    assert result.response_text == "Done."
    assert len(captured_system_prompts) == 1
    assert "## Turn Budget" not in captured_system_prompts[0]
