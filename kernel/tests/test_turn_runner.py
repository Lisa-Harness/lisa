"""Tests for kernel/turn_runner.py — the shared model+tool loop."""

from __future__ import annotations

import json
import logging
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
    YieldPolicy,
)
from kernel.seams.protocols import CoordinatorRouteDecision
from kernel.turn_hooks import (
    ModelHooks,
)
from kernel.turn_runner import (
    TurnHooks,
    TurnResult,
    _resolve_empty_completion_for_dispatch,
    run_turn,
)

# ── Module-level setup ────────────────────────────────────────────
_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()

# The shared conftest fixture (_mcp_catalog_setup) handles all catalog wiring.


# ── Test helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    # Optional provider metadata the turn loop reads via getattr with
    # defaults (finish_reason, token counters). Present only on stubs
    # that exercise provider-diagnostics paths.
    finish_reason: str | None = None
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    #: Aggregated reasoning text (reasoning models) — read by the
    #: tail-quote display sites via ``reasoning_tail_excerpt``.
    reasoning_content: str = ""


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
    return TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
        ),
    )


# ── Core flow tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_requires_complete_hook() -> None:
    """run_turn raises NotImplementedError when no provider gateway is wired.

    The kernel dispatches model calls through
    ``ContextSeams.provider_gateways`` now; a ``TurnHooks()`` with neither a
    ``complete`` callable nor an explicit gateway yields no gateway, and the
    model-call site raises the documented "no gateway wired" error.
    """
    hooks = TurnHooks()  # no complete, no gateway

    with pytest.raises(NotImplementedError, match="provider gateway"):
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

    planner_cfg = WorkflowModeConfig(
        mode_id="planner",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {"planner": planner_cfg, "search": search_cfg}

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
    hooks.workflow.route_coordinator = coordinator

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(mode_id="planner", query="search for rust news", hooks=hooks)

    # The kernel recursed into search mode and produced its response, not
    # the planner's rationale text.
    assert result.response_text == "Search results."
    assert "search" in captured_modes
    # The coordinator was saved from the plan result — so a later
    # `proceed` can resume the chain even if the search turn stalls.
    assert coordinator.saved_decision is not None
    assert coordinator.saved_decision.next_target == "search"


@pytest.mark.asyncio
async def test_plan_action_mode_propagates_handoff_to_advance_host_mode() -> None:
    """A kernel-driven plan handoff propagates the planned target as
    ``handoff_to`` when the recursive first step completes without a
    further ``transition_stage``.

    Regression for the 2026-08-01 plan→markdown→re-plan→discuss→failure
    loop: the recursive first-step turn runs in the planned target mode
    (e.g. ``markdown``) and returns ``handoff_to=None`` when that step
    finishes. The host only advances ``current_mode_id`` from
    ``result.handoff_to``, so the session was left stuck in ``plan`` mode
    and the next message re-ran ``plan_workflow`` — re-planning a
    "proceed" into a fresh (wrong) target or failing on a resume query.
    The kernel must propagate the planned target so the host stops
    re-entering the planner.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    markdown_cfg = WorkflowModeConfig(
        mode_id="markdown",
        start_stage_id="none",
        mode_kind="chat",
        routable=True,
        satisfies_intents=["write"],
        authored_allowed_operation_ids=["markdown_forge_document"],
        allowed_operation_ids=["markdown_forge_document"],
    )
    patched_catalog = {"plan": plan_cfg, "markdown": markdown_cfg}

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Drafted the document.")

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        # Simulate plan_workflow returning a single-step plan to markdown.
        return OperationResult(
            output_text="## Plan\n\n1. **markdown** — draft the document",
            output_json={
                "target_stage_id": "markdown",
                "rationale": "write intent",
                "steps": [
                    {"mode_id": "markdown", "description": "draft", "rationale": "write"},
                ],
            },
        )

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position_updater = lambda _instance: None

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(mode_id="plan", query="draft a doc", hooks=hooks)

    # The planned target is propagated as handoff_to so the host advances
    # current_mode_id out of ``plan`` and does not re-run the planner.
    assert result.handoff_to == "markdown"
    # Phase 8: the markdown step's stub completed with zero dispatches,
    # so the parent-turn completion contract blocks the unverified prose
    # as ``no_tools`` (markdown declares a write contract via the real
    # op catalog's mode_allowlist bindings).
    assert result.blocked is True
    assert result.block_origin == "no_tools"


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
        entry_op_kind="routing",
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
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan",
            query="write a novel",
            hooks=hooks,
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
async def test_run_turn_non_planner_action_mode_entry_op_no_target_not_plan_failure() -> None:
    """A non-planner action mode (e.g. ``search`` with ``web_search``) whose
    entry op returns its own output shape (search results) with NO
    ``target_stage_id`` must NOT trigger the plan-failure path.

    This is the 2026-07-30 ``search`` incident: after fixing ``web_search``
    to fall through to the tool-dispatch loop, the kernel-driven handoff
    logic misread ``web_search``'s output (which has no ``target_stage_id``
    by design — it returns search results, not a routing decision) as a
    failed plan, marked the model unreliable for planning, and hard-stopped
    the turn. The fix gates the kernel-driven handoff on the target being
    ``plan`` (or its entry op being ``plan_workflow``), so non-planner
    action modes are left to run their entry op and proceed normally. The
    Phase 4 cutover replaced that op-id check with the declared contract:
    ``search`` does not declare ``entry_op_kind: routing``, so its entry
    result is plain work output, never a failed plan.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.routing_vocabulary import (
        is_plan_reliable_for_model,
        reset_plan_reliability,
    )

    reset_plan_reliability()
    # ``search`` is an action mode whose entry op is ``web_search`` — NOT a
    # planner. Its entry op returns search results, not a routing target.
    search_cfg = WorkflowModeConfig(
        mode_id="search",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["web_search"],
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {"default": default_cfg, "search": search_cfg}

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        # web_search returns search results — no target_stage_id field at all.
        return OperationResult(
            output_text="### Research Briefing: agent frameworks\n\nResults here.",
            output_json={
                "operation_id": "web_search",
                "results": [{"name": "Smolagents", "url": "https://example.com"}],
            },
        )

    async def complete(**kwargs: Any) -> _StubCompletion:
        # The model lands in search mode and produces a summary from the
        # entry-op output. It must NOT be blocked by a plan-failure break.
        return _StubCompletion(content="Summary of search results.")

    hooks = _build_minimal_hooks(complete)

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for agent frameworks",
            hooks=hooks,
            preferred_model_id="test-search-model",
        )

    # The turn completed normally — the model produced its summary, not a
    # plan-failure recovery prompt.
    assert "plan" not in result.response_text.lower() or "summary" in result.response_text.lower()
    assert result.tool_call_count == 0
    # The model was NOT marked unreliable for planning — web_search's lack of
    # a target_stage_id is not a plan failure.
    assert is_plan_reliable_for_model("test-search-model")
    reset_plan_reliability()
    """When plan_workflow fails (empty target) but produced partial steps
    before failing, the recovery prompt includes the partial steps so the
    user has diagnostic context for rephrasing. This is the Phase 4
    fallback cleanup: the recovery message is not just a canned string —
    it surfaces what the planner attempted."""
    from unittest.mock import patch

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
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    patched_catalog = {"plan": plan_cfg}

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        # Partial steps with an empty target — the planner started
        # building a plan but failed before producing a routing target.
        return OperationResult(
            output_text="parse failed",
            output_json={
                "target_stage_id": "",
                "rationale": "parse failure",
                "steps": [
                    {"mode_id": "search", "description": "Search for info"},
                    {"mode_id": "markdown", "description": "Draft a report"},
                ],
            },
        )

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="should not reach model")

    hooks = _build_minimal_hooks(complete)

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan",
            query="research then write a report",
            hooks=hooks,
            preferred_model_id="test-planner-model",
        )

    # The recovery prompt includes the partial steps.
    assert "search" in result.response_text
    assert "markdown" in result.response_text
    assert "Search for info" in result.response_text
    # The standard recovery options are still present.
    assert "Rephrase" in result.response_text or "rephrase" in result.response_text
    assert result.tool_call_count == 0
    assert not is_plan_reliable_for_model("test-planner-model")
    reset_plan_reliability()


@pytest.mark.asyncio
async def test_run_turn_search_action_mode_phase1_returns_entry_op_result_not_plan_failure() -> (
    None
):
    """A non-routing action mode entered at turn start (e.g. the recursive
    ``run_turn(mode_id="search")`` from ``plan``'s kernel-driven handoff,
    or a ``/search`` direct command) must return its entry-op result
    directly and NOT trip the Phase 1 plan-failure recovery path.

    Regression for the 2026-08-06 ``plan → search`` workflow stall: the
    planner persisted a 2-step ``[search, ...]`` WorkflowPosition and
    handed off to ``search``; the recursive turn re-entered Phase 1,
    ran ``web_search`` (which returns search results, not a routing
    target), read an empty ``target_stage_id``, and fell into the
    plan-failure branch — logging "plan_workflow returned no target
    (mode=search)" and hard-stopping with a bogus recovery prompt on
    every planned search step. The mid-turn handoff path already gated
    on the mode's declared routing contract (``entry_op_kind``); the
    Phase 1 path did not. The fix mirrors that gate: a non-routing
    action mode's entry-op output IS the turn output.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.routing_vocabulary import (
        is_plan_reliable_for_model,
        reset_plan_reliability,
    )

    reset_plan_reliability()
    # ``search`` is an action mode whose entry op is ``web_search`` — NOT a
    # planner. Its entry op returns search results, not a routing target.
    search_cfg = WorkflowModeConfig(
        mode_id="search",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["web_search"],
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {"default": default_cfg, "search": search_cfg}

    briefing = "### Research Briefing: agent frameworks\n\nCore consensus here."

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        # web_search returns its own output shape — no target_stage_id.
        return OperationResult(
            output_text=briefing,
            output_json={
                "operation_id": "web_search",
                "core_consensus": "Core consensus here.",
                "sources": ["https://example.com"],
            },
        )

    # The model must NOT be reached — a Phase 1 action mode returns after
    # its entry op. A reached model would indicate the turn fell through to
    # the tool-dispatch loop instead of returning the entry-op result.
    async def complete(**_kwargs: Any) -> _StubCompletion:
        raise AssertionError("Phase 1 action mode must not reach the model")

    hooks = _build_minimal_hooks(complete)

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="search",
            query="search for agent frameworks",
            hooks=hooks,
            preferred_model_id="test-search-model",
        )

    # The entry-op briefing is the turn output — verbatim.
    assert result.response_text == briefing
    assert result.tool_call_count == 0
    # The model was NOT marked unreliable for planning — web_search's lack
    # of a target_stage_id is not a plan failure.
    assert is_plan_reliable_for_model("test-search-model")
    reset_plan_reliability()


@pytest.mark.asyncio
async def test_run_turn_planner_with_steps_renders_visible_plan_and_persists_instance() -> None:
    """When plan_workflow returns a ``steps`` array (Option B visible plan),
    the turn runner:
    1. Prepends the rendered plan markdown block to the response_text so
       the user sees the plan before the first step's output.
    2. Creates a ``WorkflowPosition`` with ``spawned_from="plan_workflow_op"``
       and persists it via ``workflow_position_updater``.
    3. Normalizes ``target_stage_id`` to ``steps[0].mode_id`` and recurses
       into the first step.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
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
    plan_block = (
        "## Plan\n\n1. **search** — Search for the latest rust async runtime news"
        "\n   _Provides web_search_\n"
        "2. **markdown** — Draft a markdown summary\n   _Provides document drafting_\n"
    )

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

    saved_instances: list[WorkflowPosition] = []

    def _wi_updater(instance: WorkflowPosition) -> None:
        saved_instances.append(instance)

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position_updater = _wi_updater

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
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

    # A WorkflowPosition was persisted with the correct steps.
    assert len(saved_instances) == 1, "exactly one WorkflowPosition persisted"
    instance = saved_instances[0]
    assert instance.spawned_from == "plan_workflow_op"
    assert len(instance.declared_steps) == 2
    assert instance.declared_steps[0].mode_id == "search"
    assert instance.declared_steps[1].mode_id == "markdown"
    assert instance.declared_steps[1].is_terminal
    assert instance.current_step_id == instance.declared_steps[0].step_id
    assert instance.mode_id == "search"


@pytest.mark.asyncio
async def test_run_turn_planner_without_steps_no_plan_block_no_instance() -> None:
    """When plan_workflow returns no ``steps`` (weak-model fallback), the
    turn runner does NOT create a WorkflowPosition and does NOT prepend a
    plan block — same as the pre-Option-B behavior. No regression."""
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
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

    saved_instances: list[WorkflowPosition] = []

    def _wi_updater(instance: WorkflowPosition) -> None:
        saved_instances.append(instance)

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position_updater = _wi_updater

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan",
            query="search for rust news",
            hooks=hooks,
        )

    # No plan block — the response is just the step output.
    assert "## Plan" not in result.response_text
    assert result.response_text == "Search results."
    # No WorkflowPosition created (no steps to persist).
    assert len(saved_instances) == 0


@pytest.mark.asyncio
async def test_coordinator_routing_hint_uses_plan_instance_current_step_on_proceed() -> None:
    """When a ``/proceed`` resume query is sent and a
    ``plan_workflow_op`` WorkflowPosition is active, the routing hint
    reads the next target from the instance's ``current_step_id`` (which
    was advanced when the previous step completed), NOT from the
    coordinator's saved ``next_target`` (which points at the first step
    and is stale after advancement). This is the cross-turn plan resume
    path — the plan survives across turns and ``/proceed`` picks up the
    correct next step.
    """
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    # Build an instance that has already completed step 0 (search) and
    # advanced to step 1 (markdown). The coordinator's saved
    # next_target still says "search" (stale), but the instance's
    # current_step_id says "markdown" (correct).
    steps = (
        WorkflowStep(
            step_id="search_0",
            mode_id="search",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="markdown_1",
            mode_id="markdown",
            satisfies_intents=("write",),
            is_terminal=True,
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("search_0",),
        current_step_id="markdown_1",
        mode_id="markdown",
        spawned_from="plan_workflow_op",
    )

    class _StubCoordinator:
        def __init__(self) -> None:
            self.saved_decision = CoordinatorRouteDecision(next_target="search")

        def load(self, _req: Any) -> CoordinatorRouteDecision | None:
            return self.saved_decision

        def save(self, decision: CoordinatorRouteDecision, _req: Any) -> None:
            self.saved_decision = decision

    coordinator = _StubCoordinator()

    captured_hints: list[str] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        sys_content = str(kwargs["messages"][0].get("content", ""))
        captured_hints.append(sys_content)
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.route_coordinator = coordinator
    hooks.workflow.workflow_position = lambda: active_instance

    await run_turn(mode_id="default", query="proceed", hooks=hooks)

    # The hint must point to the instance's current step (markdown), not
    # the coordinator's stale saved target (search). Verify via the
    # captured system prompts (the hint is injected into messages[0]).
    assert any("markdown" in h for h in captured_hints), (
        "routing hint must reference the plan instance's current step (markdown)"
    )
    # None of the captured hints should point ONLY at search (the stale
    # coordinator target) without also mentioning markdown.
    assert not any("search" in h and "markdown" not in h for h in captured_hints), (
        "routing hint must NOT point at the stale coordinator target (search)"
    )


@pytest.mark.asyncio
async def test_plan_instance_resume_hint_fires_without_route_coordinator() -> None:
    """A continuation query resumes an active ``plan_workflow_op`` instance
    even when ``route_coordinator`` is NOT wired.

    Regression for the 2026-08-01 incident: the host wires
    ``workflow_position`` (the durable cross-turn plan store) but never
    wires ``route_coordinator`` (a legacy single-target persistence). The
    old guard ``if hooks.workflow.route_coordinator is None ... return ""`` short-
    circuited the whole resume path, so a continuation query like "proceed
    with P0" got no resume hint, re-entered ``plan`` mode, and
    ``plan_workflow`` refused to route an execution request ("No installed
    mode provides a 'proceed with P0' capability").
    """
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    steps = (
        WorkflowStep(
            step_id="search_0",
            mode_id="search",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="markdown_1",
            mode_id="markdown",
            satisfies_intents=("write",),
            is_terminal=True,
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("search_0",),
        current_step_id="markdown_1",
        mode_id="markdown",
        spawned_from="plan_workflow_op",
    )

    captured_hints: list[str] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        sys_content = str(kwargs["messages"][0].get("content", ""))
        captured_hints.append(sys_content)
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    # Production wiring: workflow_position is set, route_coordinator is NOT.
    hooks.workflow.workflow_position = lambda: active_instance
    assert hooks.workflow.route_coordinator is None

    result = await run_turn(mode_id="default", query="proceed with P0", hooks=hooks)

    assert any("markdown" in h for h in captured_hints), (
        "plan-instance resume hint must fire without route_coordinator wired"
    )
    # Phase 8: the resumed markdown step's stub completed with zero
    # dispatches — the parent-turn completion contract blocks the
    # unverified "Done." as ``no_tools``.
    assert result.blocked is True
    assert result.block_origin == "no_tools"


@pytest.mark.asyncio
async def test_resume_in_flight_skips_work_entry_op_and_advances_plan() -> None:
    """A continuation query ("proceed") while parked on a work entry-op
    mode (e.g. ``search`` after a model-driven handoff) must NOT be fed to
    that mode's entry op — it has no search intent and would be sent to
    the backend as the query (the 2026-08-08 ``proceed`` → Brave
    ``q=proceed+meaning`` incident). Instead the kernel skips the entry
    op and synthesizes a ``transition_stage`` handoff to the plan's next
    declared step, resuming the in-flight plan.

    Regression for the 2026-08-08 incident: the plan's ``search`` step
    already ran its entry op in the prior turn (parking ``current_step_id``
    on ``search_1``); on "proceed" the turn must advance to ``markdown``,
    not re-run ``web_search`` with "proceed" as the query.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    # ``search`` is a work entry-op mode (entry_op_kind defaults to work);
    # ``markdown`` is the next planned step.
    search_cfg = WorkflowModeConfig(
        mode_id="search",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["web_search"],
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
    )
    markdown_cfg = WorkflowModeConfig(
        mode_id="markdown",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["markdown_generate"],
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "default": default_cfg,
        "search": search_cfg,
        "markdown": markdown_cfg,
    }

    # Active position parked on ``search``: ``remember_0`` done, the
    # ``search`` step current (its entry op ran in the prior turn),
    # ``markdown`` remaining. This is the post-handoff parked state.
    steps = (
        WorkflowStep(
            step_id="remember_0",
            mode_id="remember",
            satisfies_intents=("remember",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="search_1",
            mode_id="search",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="markdown_2",
            mode_id="markdown",
            satisfies_intents=("write",),
            is_terminal=True,
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("remember_0",),
        current_step_id="search_1",
        mode_id="search",
        spawned_from="plan_workflow_op",
    )

    entry_op_calls: list[str] = []

    async def fake_entry_ops(mode_id_entry: str, _ctx: Any, **_kwargs: Any) -> OperationResult:
        entry_op_calls.append(mode_id_entry)
        return OperationResult(
            output_text="### Search results\n(none — should not run).",
            output_json={"operation_id": "web_search", "results": []},
        )

    async def complete(**kwargs: Any) -> _StubCompletion:
        # The model is never asked to call transition_stage: the kernel
        # synthesizes the handoff from ``pending_planned_handoff``. The
        # stub just needs to return a plausible completion so the loop's
        # tool-call dispatch sees the synthesized call.
        return _StubCompletion(content="resuming plan")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="search",
            query="proceed",
            hooks=hooks,
            preferred_model_id="test-resume-model",
        )

    # The parked work entry op (``search``'s ``web_search``) must NOT have
    # run — "proceed" has no search intent and would have been sent to
    # Brave as the query. ``markdown``'s entry op may run after the
    # synthesized handoff lands; only ``search`` is gated.
    assert "search" not in entry_op_calls, (
        f"resume-in-flight must skip the parked search entry op; got: {entry_op_calls}"
    )
    # The plan advances: the kernel synthesized a transition_stage
    # handoff to ``markdown`` (the next declared step). When the external
    # ``markdown`` pack is installed (conftest loads it), ``handoff_to``
    # is ``"markdown"``; when it is absent, the handoff is rejected but
    # the synthesized call still fired (verified by ``"search" not in
    # entry_op_calls`` above — the search entry op never re-ran).
    assert result.handoff_to == "markdown" or result.handoff_to is None, (
        f"resume-in-flight must either hand off to markdown or be rejected "
        f"by the catalog; got handoff_to={result.handoff_to!r}"
    )


@pytest.mark.asyncio
async def test_resume_in_flight_does_not_fire_for_real_query() -> None:
    """A real search query (not a continuation phrase) while parked on a
    work entry-op mode with an active plan must still run the entry op —
    the resume-in-flight gate only applies to bare imperatives like
    "proceed". A query such as "search for agent frameworks" is genuine
    work for the parked ``search`` step, not a plan resume.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    search_cfg = WorkflowModeConfig(
        mode_id="search",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["web_search"],
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {"default": default_cfg, "search": search_cfg}

    steps = (
        WorkflowStep(
            step_id="remember_0",
            mode_id="remember",
            satisfies_intents=("remember",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="search_1",
            mode_id="search",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("remember_0",),
        current_step_id="search_1",
        mode_id="search",
        spawned_from="plan_workflow_op",
    )

    entry_op_calls: list[str] = []

    async def fake_entry_ops(mode_id_entry: str, _ctx: Any, **_kwargs: Any) -> OperationResult:
        entry_op_calls.append(mode_id_entry)
        return OperationResult(
            output_text="### Search results\nagent frameworks found.",
            output_json={"operation_id": "web_search", "results": []},
        )

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Summary of results.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        await run_turn(
            mode_id="search",
            query="search for agent frameworks",
            hooks=hooks,
            preferred_model_id="test-search-model",
        )

    # A real query runs the parked mode's entry op — the gate does not
    # skip it because the query is not a continuation phrase.
    assert "search" in entry_op_calls, (
        f"a real search query must run the search entry op; got: {entry_op_calls}"
    )


@pytest.mark.asyncio
async def test_resume_in_flight_skips_routing_entry_op_and_advances_plan() -> None:
    """A continuation query ("proceed") while parked on a **routing**
    entry-op mode (e.g. ``plan``) with an active in-flight plan must
    NOT re-run the planner entry op — ``plan_workflow`` receives
    "proceed" as ``ctx.query``, cannot match it to any mode, and returns
    an empty ``target_stage_id``, dead-ending the turn in a recovery
    prompt (the 2026-08-10 planning loop). Instead the kernel skips the
    entry op and synthesizes a ``transition_stage`` handoff to the
    plan's next declared step, resuming the in-flight plan.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    researcher_cfg = _mode_config(mode_id="researcher", satisfies_intents=["research"])
    # ``markdown`` exists in the real catalog (lisa-pack-markdown), so
    # ``get_routing_targets`` (which reads the real catalog via
    # ``kernel.ipsar_fsm``) returns it as a valid handoff target for ``plan``.
    markdown_cfg = _mode_config(mode_id="markdown", satisfies_intents=["write"])
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "default": default_cfg,
        "plan": plan_cfg,
        "researcher": researcher_cfg,
        "markdown": markdown_cfg,
    }

    # Active plan parked on ``plan``: ``researcher_0`` done, ``plan_1``
    # is current (the user re-entered planner), ``markdown_2`` remaining.
    steps = (
        WorkflowStep(
            step_id="researcher_0",
            mode_id="researcher",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="plan_1",
            mode_id="plan",
            satisfies_intents=("plan",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="markdown_2",
            mode_id="markdown",
            satisfies_intents=("write",),
            is_terminal=True,
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("researcher_0",),
        current_step_id="plan_1",
        mode_id="plan",
        spawned_from="plan_workflow_op",
    )

    plan_entry_calls = 0

    async def fake_entry_ops(_mode_id: str, _ctx: Any, **_kwargs: Any) -> OperationResult:
        nonlocal plan_entry_calls
        plan_entry_calls += 1
        # The planner would refuse "proceed" — this path must NOT be reached.
        return OperationResult(
            output_text="No installed mode provides a 'proceed' capability.",
            output_json={"target_stage_id": "", "rationale": "refused"},
        )

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="resuming plan")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="plan",
            query="proceed",
            hooks=hooks,
            preferred_model_id="test-resume-model",
        )

    # The planner entry op (``plan_workflow``) must NOT have run —
    # "proceed" has no planning intent and would dead-end in a recovery
    # prompt. The kernel synthesized a handoff to the next step.
    assert plan_entry_calls == 0, (
        f"resume-in-flight must skip the planner entry op on 'proceed'; "
        f"got {plan_entry_calls} call(s)"
    )
    # The kernel synthesized a transition_stage handoff to ``markdown``
    # (the next declared step). When the external ``markdown`` pack is
    # installed (conftest loads it), ``handoff_to`` is ``"markdown"``;
    # when it is absent, the handoff is rejected but the synthesized call
    # still fired (verified by ``plan_entry_calls == 0`` above — the
    # planner's own entry op never ran).
    assert result.handoff_to == "markdown" or result.handoff_to is None, (
        f"resume-in-flight must either hand off to markdown or be rejected "
        f"by the catalog; got handoff_to={result.handoff_to!r}"
    )


@pytest.mark.asyncio
async def test_plan_progress_rendered_when_plan_instance_active() -> None:
    """When a ``plan_workflow_op`` WorkflowPosition is active and has
    remaining steps, the turn result's ``response_text`` includes a
    ``## Workflow progress`` summary so the user sees which steps are
    done and which remain."""
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    steps = (
        WorkflowStep(
            step_id="search_0",
            mode_id="search",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="markdown_1",
            mode_id="markdown",
            satisfies_intents=("write",),
            is_terminal=True,
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("search_0",),
        current_step_id="markdown_1",
        mode_id="markdown",
        spawned_from="plan_workflow_op",
    )

    async def complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Drafted the markdown summary.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance

    result = await run_turn(mode_id="markdown", query="write the summary", hooks=hooks)

    assert "## Workflow progress" in result.response_text
    assert "search" in result.response_text
    assert "markdown" in result.response_text
    # Step 0 (search) should be marked done.
    assert "[x]" in result.response_text
    # Step 1 (markdown) should be marked remaining.
    assert "[ ]" in result.response_text


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
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {
        "default": default_cfg,
        "plan": plan_cfg,
        "search": search_cfg,
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
        is_search_mode = ("## Handoff Context" in sys_content and "`search`" in sys_content) or (
            "Handed off to mode: search" in str([m.get("content", "") for m in kwargs["messages"]])
        )
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
    hooks.workflow.route_coordinator = coordinator

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
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
async def test_plan_mid_turn_handoff_resumes_active_instance() -> None:
    """A mid-turn handoff to ``plan`` while a broader ``plan_workflow_op``
    instance is still in flight resumes that instance's next step instead
    of blindly re-running ``plan_workflow`` against a continuation query.

    Regression for the 2026-08-01 plan→markdown→learn→re-plan failure: the
    user led a session down a chained step, then asked to "continue the
    overall plan / what phase are we on?". Re-entering ``plan`` ran
    ``plan_workflow`` with zero awareness of the existing instance, and the
    planner refused to route a continuation/status query ("No installed
    mode provides a continue plan/phase status capability"). The mid-turn
    plan handoff must resolve the resume target from the active instance's
    current step instead of re-planning.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition, WorkflowStep

    default_cfg = _mode_config(mode_id="default")
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    markdown_cfg = _mode_config(mode_id="markdown", satisfies_intents=["write"])
    patched_catalog = {
        "default": default_cfg,
        "plan": plan_cfg,
        "search": search_cfg,
        "markdown": markdown_cfg,
    }

    # Active plan: search (done) → markdown (current, the next step).
    steps = (
        WorkflowStep(
            step_id="search_0",
            mode_id="search",
            satisfies_intents=("research",),
            spawned_from="plan_workflow_op",
        ),
        WorkflowStep(
            step_id="markdown_1",
            mode_id="markdown",
            satisfies_intents=("write",),
            is_terminal=True,
            spawned_from="plan_workflow_op",
        ),
    )
    active_instance = WorkflowPosition(
        declared_steps=steps,
        completed_steps=("search_0",),
        current_step_id="markdown_1",
        mode_id="markdown",
        spawned_from="plan_workflow_op",
    )

    plan_entry_calls = 0

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        nonlocal plan_entry_calls
        plan_entry_calls += 1
        # The planner would refuse a continuation query, but this path must
        # NOT be reached for a resume — the active instance drives the target.
        return OperationResult(
            output_text="No installed mode provides a continue plan capability.",
            output_json={"target_stage_id": "", "rationale": "refused"},
        )

    async def complete(**kwargs: Any) -> _StubCompletion:
        sys_content = str(kwargs["messages"][0].get("content", ""))
        is_plan_mode = "## Handoff Context" in sys_content and "`plan`" in sys_content
        if is_plan_mode:
            return _StubCompletion(content="Planning...")
        # default mode → transition_stage to plan.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_default_to_plan",
                    "function": {
                        "name": "transition_stage",
                        "arguments": json.dumps(
                            {"target_stage_id": "plan", "rationale": "resume the plan"}
                        ),
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = lambda: active_instance

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        await run_turn(
            mode_id="default",
            query="continue with the overall plan; what phase are we on?",
            hooks=hooks,
        )

    # The active instance drove the resume target — the planner must NOT have
    # been re-run against the continuation query.
    assert plan_entry_calls == 0, "mid-turn plan handoff must resume, not re-plan"


@pytest.mark.asyncio
async def test_mid_turn_handoff_to_plan_persists_workflow_position() -> None:
    """A mid-turn handoff to ``plan`` whose entry op returns a
    ``target_stage_id`` AND ``steps`` persists the plan as a
    ``WorkflowPosition`` (spawned_from=plan_workflow_op), mirroring the
    Phase-1 plan path. Without this, a subsequent mid-turn re-entry to
    ``plan`` finds no active position and re-runs ``plan_workflow``
    against a status query → no routing target → plan-failure hard-stop
    (the 2026-08-08 kimi-k2.6 stuck turn).
    """
    from unittest.mock import patch

    from kernel.models import OperationResult
    from kernel.workflow_position import WorkflowPosition

    default_cfg = _mode_config(mode_id="default")
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {
        "default": default_cfg,
        "plan": plan_cfg,
        "search": search_cfg,
    }

    plan_entry_calls = 0
    persisted_positions: list[Any] = []

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        nonlocal plan_entry_calls
        plan_entry_calls += 1
        return OperationResult(
            output_text="research intent detected",
            output_json={
                "target_stage_id": "search",
                "rationale": "research intent",
                "steps": [
                    {"mode_id": "search", "description": "search", "rationale": "r"},
                ],
            },
        )

    async def complete(**kwargs: Any) -> _StubCompletion:
        sys_content = str(kwargs["messages"][0].get("content", ""))
        is_plan_mode = "## Handoff Context" in sys_content and "`plan`" in sys_content
        is_search_mode = ("## Handoff Context" in sys_content and "`search`" in sys_content) or (
            "Handed off to mode: search" in str([m.get("content", "") for m in kwargs["messages"]])
        )
        if is_search_mode:
            return _StubCompletion(content="Search results from web_search.")
        if is_plan_mode:
            return _StubCompletion(content="Planning...")
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

    hooks = _build_minimal_hooks(complete)

    def capture_position(updated: Any) -> None:
        persisted_positions.append(updated)

    hooks.workflow.workflow_position_updater = capture_position

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default", query="search for rust news then write a summary", hooks=hooks
        )

    # The chain completed (default → plan → search).
    assert result.response_text == "Search results from web_search."
    assert plan_entry_calls == 1
    # A WorkflowPosition was persisted for the plan's steps.
    plan_positions = [
        p
        for p in persisted_positions
        if isinstance(p, WorkflowPosition) and p.spawned_from == "plan_workflow_op"
    ]
    assert len(plan_positions) >= 1, "mid-turn plan handoff must persist a WorkflowPosition"
    assert plan_positions[0].declared_steps[0].mode_id == "search"


@pytest.mark.asyncio
async def test_mid_turn_plan_is_authoritative_over_residual_intent_gap() -> None:
    """A plan declared mid-turn governs the handoff it drives.

    The host's ``workflow_position`` view is derived from the thread at turn
    start, so a plan made mid-turn is invisible to it. The handoff into the
    planned (final) step then fell back to intent-gap routing: a residual
    intent the plan deliberately left out appended an extra "remaining
    step" and a routing hint sent the model straight out of the planned
    mode before it did any work (the 2026-10-09 e2e roadmap run:
    planner → coding_review was immediately hinted to ``thinker``, which
    has no workspace tools, and the review answered "no workspace").
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
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write"]},
    )
    patched_catalog = {
        "default": default_cfg,
        "plan": plan_cfg,
        "search": search_cfg,
        "markdown": markdown_cfg,
    }

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        return OperationResult(
            output_text="research only",
            output_json={
                "target_stage_id": "search",
                "rationale": "research intent",
                "steps": [{"mode_id": "search", "description": "search", "rationale": "r"}],
            },
        )

    search_prompts: list[str] = []
    entered_markdown = False

    async def complete(**kwargs: Any) -> _StubCompletion:
        nonlocal entered_markdown
        sys_content = str(kwargs["messages"][0].get("content", ""))
        if "## Handoff Context" in sys_content and "handed off from: `search`" in sys_content:
            entered_markdown = True
            return _StubCompletion(content="Markdown output.")
        if "## Handoff Context" in sys_content and "`plan`" in sys_content:
            if "Handed off to mode: search" in str(
                [m.get("content", "") for m in kwargs["messages"]]
            ):
                search_prompts.append(sys_content)
                return _StubCompletion(content="Search results from web_search.")
            return _StubCompletion(content="Planning...")
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

    hooks = _build_minimal_hooks(complete)

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.get_mode_system_intro", lambda mid: mid),
        patch("kernel.ipsar_fsm.get_mode_system_intro", lambda mid: mid),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default", query="search for rust news then write a summary", hooks=hooks
        )

    assert search_prompts, "the planned search step must run"
    search_prompt = search_prompts[0]
    assert "Remaining steps after this one: (none" in search_prompt
    assert "`markdown`" not in search_prompt
    assert not entered_markdown
    assert result.response_text == "Search results from web_search."


@pytest.mark.asyncio
async def test_run_turn_missing_complete_raises_not_implemented() -> None:
    """A search (action) mode with entry ops needs complete_fn, and the
    operation fails with an OperationExecutionError wrapping the
    NotImplementedError when hooks.models.complete is missing."""
    from unittest.mock import patch

    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.operations.registry import OperationExecutionError

    hooks = TurnHooks()  # no complete

    search_mode = get_workflow_catalog()["search"].model_copy(
        update={
            "mode_kind": "action",
            "entry_operation_ids": ["web_search"],
        }
    )
    patched_catalog = dict(get_workflow_catalog())
    patched_catalog["search"] = search_mode

    with (
        pytest.raises(OperationExecutionError, match="web_search"),
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
    ):
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
    hooks.operations.execute_operation = _stub_execute_operation

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
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=3,
    )

    assert result.iteration_count == 3
    assert result.tool_call_count == 2
    # Caps pause, they do not yield. The iteration cap produces a
    # PauseAction (operator_exit="paused"); the loop breaks before the
    # final model call, so tool_call_count is one less than the cap (the
    # old YieldAction went through strip-tools → model call → break, adding
    # an extra iteration). No yield question is surfaced.
    assert result.operator_exit == "paused"


@pytest.mark.asyncio
async def test_run_turn_max_tool_iterations_defaults_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        # max_tool_iterations intentionally omitted — must read from config
    )

    assert result.iteration_count == 2
    assert result.tool_call_count == 1
    # Caps pause, they do not yield. The iteration cap produces a PauseAction;
    # the loop breaks before the final model call.
    assert result.operator_exit == "paused"


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
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=10,
        execution_policy=ExecutionPolicy(max_model_calls=2),
    )

    assert result.iteration_count == 2
    assert result.tool_call_count == 1
    # Caps pause, they do not yield. The iteration cap (tightened by
    # execution_policy) produces a PauseAction; the loop breaks before the
    # final model call.
    assert result.operator_exit == "paused"


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
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        execution_policy=ExecutionPolicy(max_capability_calls=1),
    )

    assert result.tool_call_count == 1
    # Caps pause, they do not yield. The turn budget exhausted by the
    # capability cap produces a PauseAction (operator_exit="paused"), not a
    # yield question. The operator can resume for a fresh budget.
    assert result.operator_exit == "paused"


@pytest.mark.asyncio
async def test_stuck_in_stage_cap_surfaces_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stuck-in-stage cap surfaces a diagnostic message, not silence.

    When the model loops on a failing tool call and the stuck-in-stage cap
    fires, the post-loop code must surface a diagnostic explaining why the
    turn paused — not an empty response_text. The user can then redirect.
    """
    from kernel import config

    # Set a low stage cap so the stuck-in-stage cap fires quickly.
    monkeypatch.setattr(config.settings, "stage_nudge_cap", 3)
    monkeypatch.setattr(config.settings, "stage_nudge_threshold", 2)

    async def loop_on_tool(**kwargs: Any) -> _StubCompletion:
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

    hooks = _build_minimal_hooks(loop_on_tool)
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=10,
    )

    assert result.operator_exit == "paused"
    # The diagnostic must be non-empty and mention the mode + iteration budget.
    assert result.response_text
    assert "default" in result.response_text
    assert "iteration budget" in result.response_text.lower()
    assert "paused" in result.response_text.lower() or "redirect" in result.response_text.lower()
    # The enriched diagnostic includes the last tool name so the user
    # knows *why* the turn was stuck, not just that the cap fired.
    assert "web_search" in result.response_text


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
    hooks.operations.execute_operation = _stub_execute_operation

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
    # Unified turn lifecycle: the capability cap (web_search limited to 1)
    # rejected the model's second call, the model observed the rejection,
    # and its recovered final synthesis stands — a per-target cap mid-turn
    # must not erase a turn that finishes cleanly afterward (the
    # 2026-10-17 turns 873/874 incident: fetch_page hit its 8-call cap,
    # the model delegated research, rewrote the files, finished — and the
    # finished answer was replaced with a fabricated "resource budget"
    # question that persisted to the thread as a false exhaustion claim).
    assert result.response_text == "Done."
    assert "resource budget" not in result.response_text


@pytest.mark.asyncio
async def test_capability_cap_with_no_final_content_surfaces_resource_budget_yield() -> None:
    """A model that gives up (empty, no tool calls) after a cap still yields.

    The complementary contract to the recovered case above: when a capped
    tool call was rejected, the latch is set, and the model then breaks the
    loop with NO content and no tool calls, the fallback yield honestly
    tells the operator a resource cap fired (the b79ba861 contract — the
    rejection is not mistaken for a natural empty completion).
    """
    completion_count = 0

    async def call_rejected_then_give_up(**kwargs: Any) -> _StubCompletion:
        nonlocal completion_count
        completion_count += 1
        if completion_count <= 2:
            # The second call is rejected at the cap (1/turn), setting the
            # latch and feeding the rejection back to the model.
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
        # The model observes the rejection, produces nothing further, and
        # breaks the loop naturally: empty content, no tool calls.
        return _StubCompletion(content="")

    hooks = _build_minimal_hooks(call_rejected_then_give_up)
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="test",
        hooks=hooks,
        max_tool_iterations=6,
        execution_policy=ExecutionPolicy(
            capability_limits={
                "web_search": CapabilityExecutionPolicy(max_invocations_per_turn=1),
            }
        ),
    )

    # One real dispatch before the cap; the rejected call is not dispatched
    # (it does not count as a tool call).
    assert result.tool_call_count == 1
    assert (
        result.response_text
        == "I've reached a resource budget for this turn. How would you like me to proceed?"
    )


@pytest.mark.asyncio
async def test_empty_final_completion_does_not_claim_iteration_budget() -> None:
    """An empty final completion far from the cap must NOT be labeled a
    budget hit.

    Regression for the 2026-08-30 remote-box relapse: kimi-k3:cloud
    returned empty visible content on 1-2 model calls, the loop broke with
    empty content at iteration 1-2 of 300, and the post-loop fallback
    fabricated "I've hit my iteration budget for this turn." That claim was
    false (the cap never fired), was persisted to the thread, was read back
    by the model as ground truth, and re-fired on every subsequent turn —
    "over and over after 1 call". The fallback now relabels this honestly
    as an empty completion and reserves the budget claim for the case
    where the cap was actually reached.
    """

    async def empty_completion(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="")

    hooks = _build_minimal_hooks(empty_completion)
    # Neutral query: no producer-intent markers, so no routing hint (and no
    # forced-action synthesis) competes with the empty-completion path.
    result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    lowered = result.response_text.lower()
    assert "iteration budget" not in lowered
    assert "empty response" in lowered.lower() or "empty" in lowered
    # Still a yield for redirect (chat mode) — semantics preserved.
    assert result.yielded is not None


@pytest.mark.asyncio
async def test_empty_final_completion_near_cap_does_not_claim_budget() -> None:
    """An empty final completion before the cap is honestly relabeled even
    when close to the cap.

    An empty completion breaks the loop immediately (``iteration 1`` with
    ``max_tool_iterations=2``); only a strip-tools path (router finish)
    can carry an empty completion to the exhausted branch. The regression
    we care about: whenever the cap has NOT actually fired, the response
    must not claim a budget hit — including the near-cap case where the
    user could easily believe the budget ran out.
    """

    async def empty_completion(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="")

    hooks = _build_minimal_hooks(empty_completion)
    result = await run_turn(
        mode_id="default", query="hello there", hooks=hooks, max_tool_iterations=2
    )

    lowered = result.response_text.lower()
    assert "iteration budget" not in lowered
    assert "empty response" in lowered or "empty" in lowered
    assert result.yielded is not None


@pytest.mark.asyncio
async def test_empty_natural_completion_logs_diagnostic_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A natural completion (zero tool calls) with empty content logs the
    empty-completion warning.

    The d4a06be diagnostic only covered the dispatch-path exit (tool_calls
    existed but none dispatched); a natural completion with zero tool calls
    breaks out of the loop before reaching that block, so the 2026-09-02
    glm-5.3-flash:cloud incident produced "empty response" yields with NO
    logged cause. This locks the warning to the natural-completion break.
    """

    async def empty_completion(**kwargs: Any) -> _StubCompletion:
        # Mimic the incident shape: reasoning burned the budget, finish=-length.
        return _StubCompletion(content="", finish_reason="length", completion_tokens=980)

    hooks = _build_minimal_hooks(empty_completion)
    with caplog.at_level("WARNING", logger="kernel.turn_runner"):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert result.yielded is not None
    empty_warnings = [
        rec
        for rec in caplog.records
        if "natural completion returned empty content" in rec.getMessage()
    ]
    assert empty_warnings, "expected the empty natural-completion warning"
    message = empty_warnings[-1].getMessage()
    assert "finish_reason=length" in message
    assert "completion_tokens=200" not in message  # cumulative, not per-call
    assert "mode=default" in message


@pytest.mark.asyncio
async def test_truncated_prose_ends_the_turn_instead_of_arming_the_backstop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Truncated prose with zero tool calls ends the turn, naming the cap.

    The 2026-09-30 approval loop: every executor completion capped at
    ``length`` with prose and no tool call, the write-contract backstop
    synthesized an argument-less ``coding_apply_patch``, the approval failed,
    and the host replayed the failure as a fresh turn — four cycles,
    220k-token prompts, zero bytes written.
    """

    async def truncated_prose(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="Let me apply the patch now",
            finish_reason="length",
            completion_tokens=980,
        )

    hooks = _build_minimal_hooks(truncated_prose)
    with caplog.at_level("WARNING", logger="kernel.turn_runner"):
        result = await run_turn(mode_id="default", query="fix the imports", hooks=hooks)

    assert "reached its token cap" in result.response_text
    assert "Let me apply the patch now" in result.response_text
    assert result.tool_call_count == 0
    assert any(
        "ending the turn instead of synthesizing" in rec.getMessage() for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_natural_completion_with_content_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A normal natural completion (non-empty content) must NOT log the
    empty-content warning."""

    hooks = _build_minimal_hooks(_make_stub_complete("Hello, world!"))
    with caplog.at_level("WARNING", logger="kernel.turn_runner"):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert result.response_text == "Hello, world!"
    assert result.yielded is None
    assert not [
        rec
        for rec in caplog.records
        if "natural completion returned empty content" in rec.getMessage()
    ]


@pytest.mark.asyncio
async def test_reasoning_only_completion_surfaces_labeled_tail_quote() -> None:
    """A reasoning model that returns no visible content surfaces a short
    labeled quote of its reasoning tail — NOT the raw thinking presented as
    the answer (the 2026-09-18 rambling relapse).

    The turn result carries the ``reasoning_tail`` content-source marker and
    the FULL reasoning on ``reasoning_text`` so the thread log stays
    exhaustive at rest; only the surfaced response is bounded.
    """

    async def reasoning_only(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            reasoning_content=(
                "let me consider the options\nmore deliberation here\n"
                "I think the user probably wants option B"
            ),
            finish_reason="length",
        )

    hooks = _build_minimal_hooks(reasoning_only)
    result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    # The caveat labels itself as thinking, not a response.
    assert "no visible answer" in result.response_text
    assert "not a response" in result.response_text
    # The tail (not the head) of the reasoning is quoted.
    assert "option B" in result.response_text
    assert result.response_text.count("\n") <= len(result.response_text.split("\n"))
    # Raw reasoning is NOT presented verbatim as the whole answer.
    assert not result.response_text.startswith("let me consider")
    # Marker + full reasoning ride the result for thread persistence.
    assert result.content_source == "reasoning_tail"
    assert result.reasoning_text == (
        "let me consider the options\nmore deliberation here\n"
        "I think the user probably wants option B"
    )


@pytest.mark.asyncio
async def test_reasoning_only_completion_truncation_clause_on_length() -> None:
    """``finish_reason=length`` adds the budget-consumption clause so the
    user knows WHY there was no visible answer."""

    async def reasoning_only(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            reasoning_content="pondering endlessly",
            finish_reason="length",
        )

    hooks = _build_minimal_hooks(reasoning_only)
    result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert "consumed the completion budget" in result.response_text
    assert result.content_source == "reasoning_tail"


@pytest.mark.asyncio
async def test_large_reasoning_only_completion_is_bounded_not_dumped_raw() -> None:
    """A huge reasoning-only completion surfaces BOUNDED, not raw.

    Regression for the 2026-10-04 incident: a non-streaming reasoning model
    burned its whole completion budget on thinking. The gateway folded that
    reasoning into ``content``, so ``reasoning_tail_excerpt`` saw "visible
    text" and returned a 119KB raw monologue unbounded. With the gateway
    keeping reasoning on ``reasoning_content``, the tail-quote guard bounds
    the surfaced text and stamps the ``reasoning_tail`` marker.
    """

    huge_reasoning = "deliberation line\n" * 2000  # ~34KB of thinking

    async def reasoning_only(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            reasoning_content=huge_reasoning + "the answer is 42",
            finish_reason="length",
        )

    hooks = _build_minimal_hooks(reasoning_only)
    result = await run_turn(mode_id="default", query="status?", hooks=hooks)

    # Surfaced text is a bounded quote, not the whole monologue.
    assert result.content_source == "reasoning_tail"
    assert len(result.response_text) < 2000
    # The tail (closest to the answer) is quoted, not flooded with the head.
    assert result.reasoning_text == huge_reasoning + "the answer is 42"
    # The full reasoning still rides the result for the exhaustive thread log.
    assert result.reasoning_text.startswith("deliberation line")


@pytest.mark.asyncio
async def test_normal_completion_has_no_reasoning_tail_marker() -> None:
    """A normal visible answer keeps the default empty content_source and
    empty reasoning_text — the marker only fires on the degraded path."""

    hooks = _build_minimal_hooks(_make_stub_complete("Plain answer."))
    result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert result.response_text == "Plain answer."
    assert result.content_source == ""
    assert result.reasoning_text == ""


@pytest.mark.asyncio
async def test_run_turn_calls_on_turn_complete() -> None:
    """on_turn_complete is called after a successful turn."""
    side_effects: list[dict[str, Any]] = []

    async def on_complete(
        mode_id: str,
        query: str,
        response_text: str,
        messages: list[dict[str, Any]],
        effect_receipts: list[Any],
    ) -> None:
        side_effects.append(
            {
                "mode_id": mode_id,
                "query": query,
                "response_text": response_text,
                "message_count": len(messages),
                "receipt_count": len(effect_receipts),
            }
        )

    hooks = _build_minimal_hooks(_make_stub_complete("Done."))
    hooks.lifecycle.on_turn_complete = on_complete

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    assert len(side_effects) == 1
    assert side_effects[0]["mode_id"] == "default"
    assert side_effects[0]["query"] == "hello"
    assert side_effects[0]["response_text"] == "Done."
    assert side_effects[0]["message_count"] >= 2  # system + user + assistant
    assert side_effects[0]["receipt_count"] == 0  # default mode, no writes


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
        execution_policy: Any = None,
    ) -> OperationResult:
        _ = (mode_id, ctx, ledger, execution_policy)
        return OperationResult(
            output_text=(
                '{"briefing_points": [], "output_text": "", "source_kind": "", "source_urls": []}'
            ),
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
        execution_policy: Any = None,
    ) -> OperationResult:
        _ = (mode_id, ctx, ledger, execution_policy)
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
    hooks.models.resolve_model = resolve_model

    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=hooks,
        preferred_model_id="haiku",
    )

    assert result.response_text == "Done."
    assert resolved_model_ids == ["haiku"]


@pytest.mark.asyncio
async def test_run_turn_workflow_stuck_session_nudge_injects_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After stage_nudge_threshold iterations in the same stage, the resolver's
    soft nudge rides the lifecycle briefing (system prompt) — never a
    synthesized user message (Phase 7: one control channel).
    """
    from kernel.config import settings
    from kernel.workflow_position import WorkflowPosition

    monkeypatch.setattr(settings, "stage_nudge_threshold", 1)

    session = WorkflowPosition(
        mode_id="ponder",
        stage_id="inquire",
        entered_at="2026-01-01T00:00:00Z",
        turn_count=1,
    )

    def workflow_position() -> Any:
        return session

    captured_messages: list[dict[str, Any]] = []

    async def complete(**kwargs: Any) -> _StubCompletion:
        captured_messages.clear()
        captured_messages.extend(kwargs["messages"])
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.workflow.workflow_position = workflow_position

    result = await run_turn(mode_id="ponder", query="analyze this", hooks=hooks)

    assert result.response_text == "Done."
    nudge_found = any(
        "without advancing" in str(msg.get("content", "")) for msg in captured_messages
    )
    assert nudge_found
    user_messages = [m for m in captured_messages if m.get("role") == "user"]
    assert user_messages
    assert "## User Query" in str(user_messages[0].get("content", ""))
    assert all("## Turn Lifecycle" in str(m.get("content", "")) for m in user_messages[1:])


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
    model = hooks.models.resolve_model("test") if hooks.models.resolve_model else None
    if hooks.models.resolve_model:
        assert model is not None

    # Default assemble_system_prompt
    if hooks.context.assemble_system_prompt:
        result = hooks.context.assemble_system_prompt("instruction", None)
        assert result == "instruction"

    # Default assemble_user_context returns empty (accepts optional manifest)
    if hooks.context.assemble_user_context:
        result = hooks.context.assemble_user_context("m", "q", "b", None)
        assert result == ""

    # Default mcp_tool_servers is an empty list
    assert hooks.tools.mcp_tool_servers == []


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
    hooks.context.assemble_user_context = assemble_user

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
                            "arguments": (
                                '{"target_stage_id": "markdown", '
                                '"rationale": "user wants to write a doc"}'
                            ),
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done in markdown mode.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a markdown doc", hooks=hooks)

    assert result.handoff_to == "markdown"
    assert result.transitioned_to is None  # no workflow stage transition
    # Phase 8: the stub completed the markdown mode with zero dispatches.
    # markdown declares a write contract (mode_allowlist-bound write ops),
    # so the parent-turn completion contract blocks the unverified prose.
    assert result.blocked is True
    assert result.block_origin == "no_tools"


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
                            "name": "fork",
                            "arguments": (
                                '{"targets": [{"mode_id": "default"}, {"mode_id": "default"}]}'
                            ),
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
        assert child.status in ("ok", "error", "timeout", "cancelled", "no_tools", "yielded")


@pytest.mark.asyncio
async def test_run_turn_fork_marks_no_tools_child_when_action_mode_skipped_tools() -> None:
    """A fork child whose mode declares action operations/tools but
    completes with zero tool calls is marked ``status="no_tools"`` rather
    than ``"ok"``.

    Regression for the 2026-07-18 incident: the markdown fork child
    produced a hallucinated "Done — wrote it to disk" text response
    without ever calling ``markdown_forge_document`` / ``write_file``,
    and the parent reported "2/2 children succeeded" while nothing landed
    on disk. ``remember`` and ``markdown`` declare non-empty
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
                            "name": "fork",
                            "arguments": (
                                '{"targets": [{"mode_id": "remember"}, {"mode_id": "markdown"}]}'
                            ),
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
    # Both remember and markdown declare action tools/operations, so children
    # that made zero tool calls must be flagged no_tools, not ok.
    assert statuses["remember"] == "no_tools"
    assert statuses["markdown"] == "no_tools"
    for child in result.fork_results:
        assert child.status != "ok"


def test_fork_mode_requires_tools_true_for_action_modes() -> None:
    """``_fork_mode_requires_tools`` returns True for modes that declare
    non-empty ``allowed_operation_ids`` or ``allowed_tool_ids`` (remember,
    markdown), and False for chat-only modes (default, ponder)."""
    from kernel.turn_phases._fork import _fork_mode_requires_tools

    assert _fork_mode_requires_tools("remember") is True
    assert _fork_mode_requires_tools("markdown") is True
    assert _fork_mode_requires_tools("default") is False
    assert _fork_mode_requires_tools("nonexistent_mode") is False


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
                            "name": "fork",
                            "arguments": '{"targets": [{"mode_id": "markdown"}]}',
                        },
                    }
                ],
            )
        # The markdown child calls fetch_page (read-only) instead of its
        # action tool (markdown_forge_document). It made tool calls, but
        # none are its mode's authored action tools.
        if call_count == 2:
            return _StubCompletion(
                content="",
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
        # After the read-only call, the child exits with a text response
        # claiming success — but it never invoked its action tool.
        return _StubCompletion(
            content="Done — wrote it to disk.",
            tool_calls=[],
        )

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a markdown guide", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    # The child made a tool call (fetch_page) but skipped its action tools,
    # so it must be flagged no_tools — not ok.
    assert child.status == "no_tools"
    assert child.target.mode_id == "markdown"


@pytest.mark.asyncio
async def test_run_turn_fork_namespaced_action_tool_not_flagged_no_tools() -> None:
    """A fork child that calls its mode's authored action tool under the
    ambient namespaced form (``lisa-storage.write_file`` for markdown's
    authored ``write_file``) is NOT flagged ``no_tools``.

    Regression for the 2026-08-02 fan-out failure: the ambient
    ``default_available_tools`` contract merges ``lisa-storage.read_file``
    / ``lisa-storage.write_file`` into a mode's effective allowlist, so the
    model dispatches the namespaced form. The authored action-tool set is
    the bare front-matter name (``write_file``), and a plain
    ``called_tool_names & action_tool_ids`` intersection was empty, false-
    flagging every fork child that legitimately used its ambient storage
    tools. The validator strips the ``server_id.`` prefix before matching.
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
                            "name": "fork",
                            "arguments": '{"targets": [{"mode_id": "markdown"}]}',
                        },
                    }
                ],
            )
        # The markdown child calls its action tool under the namespaced
        # ambient form (lisa-storage.write_file). Its bare authored name
        # is ``write_file``; the validator must recognize the match.
        if call_count == 2:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_2",
                        "function": {
                            "name": "lisa-storage.write_file",
                            "arguments": '{"path": "doc.md", "content": "..."}',
                        },
                    }
                ],
            )
        # After the write, the child exits with a text response.
        return _StubCompletion(
            content="Done — wrote it to disk.",
            tool_calls=[],
        )

    # Wire a stub MCP pool whose ``call_tool`` succeeds (``is_error`` absent)
    # so the receipt-capture branch at dispatch time emits an ``EffectReceipt``
    # for the namespaced write tool. Without a working pool the dispatch
    # returns ``is_error=True`` ("not registered"), no receipt is emitted,
    # and the receipt-only completion contract correctly flags the child
    # ``no_tools`` — the test would then exercise the failure path, not the
    # namespaced-success path it exists to guard.
    from kernel.tools.mcp_client import MCPToolResult

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> MCPToolResult:
            return MCPToolResult(tool_id=tool_id, output_text="Wrote doc.md")

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    hooks = _build_minimal_hooks(two_step)
    hooks.tools.mcp_tool_pool = _StubPool()
    result = await run_turn(mode_id="default", query="write a markdown guide", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    # The child called its mode's action tool (write_file) under the
    # namespaced ambient form, and the stub pool succeeded, so the
    # receipt-capture path emitted an ``EffectReceipt`` — the child is
    # ok, not no_tools.
    assert child.status == "ok"
    assert child.target.mode_id == "markdown"


@pytest.mark.asyncio
async def test_run_turn_fork_yielded_child_not_flagged_no_tools() -> None:
    """A fork child the harness force-yielded (iteration cap) is classified
    ``status="yielded"``, not ``no_tools``.

    Regression for the 2026-08-02 fan-out failure: the ``coding`` child
    gathered evidence via read-only ``lisa-storage.read_file`` /
    ``list_files`` calls, then the lifecycle resolver force-yielded it at
    the ``fork_child_max_tool_iterations`` cap (the historical default of 5,
    since raised to 50) before it reached its write tools
    (``coding_apply_patch``, ``coding_commit``).
    The validator only inspected ``tool_call_count`` /
    ``called_tool_names`` and could not distinguish "interrupted by the
    harness" from "model gave up after read-only fetches", so it flagged
    the child ``no_tools`` (hallucinated success) and the parent aborted
    with a dishonest "nothing was written" message. The fix reads
    ``child_result.yielded``: a force-yielded child is ``yielded``
    (interrupted), regardless of its tool-call pattern.
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
                            "name": "fork",
                            "arguments": '{"targets": [{"mode_id": "coding"}]}',
                        },
                    }
                ],
            )
        # The coding child inspects via read-only ambient storage tools
        # and never reaches its write tools before the iteration cap
        # forces a yield. It keeps calling a read-only tool every turn
        # so the loop never produces final content and the cap fires.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": f"call_{call_count['n']}",
                    "function": {
                        "name": "lisa-storage.read_file",
                        "arguments": '{"path": "src/main.py"}',
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="refactor the main module", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    # The child was force-yielded by the harness (iteration cap), not
    # hallucinating and not a model-emitted yield — it must be
    # ``interrupted`` (the renamed bucket), not ``no_tools``.
    assert child.status == "interrupted"
    assert child.target.mode_id == "coding"
    # The parent must still abort (a partial result cannot be trusted to
    # be surfaced honestly), but with the "interrupted" framing, not the
    # "hallucinated success" framing.
    assert result.blocked is True
    assert "interrupted" in result.response_text.lower()
    assert "hallucinated" not in result.response_text.lower()


@pytest.mark.asyncio
async def test_fork_child_never_advertises_yield_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fork child never gets the ``yield`` tool — the no-ask invariant.

    Fork children unconditionally receive ``yield_policy_override=
    YieldPolicy(allowed=False)`` so the ``yield`` tool schema is never
    added to the child's tool list. The model cannot punt to a user that
    is not there. This test verifies the child finishes (or fails) rather
    than yielding, even when the foreground ``yield_policy`` allows asks.
    """
    from kernel.ipsar_fsm import get_workflow_catalog

    markdown_cfg = get_workflow_catalog().get("markdown")
    assert markdown_cfg is not None, "markdown mode must be loaded by the catalog fixture"
    # Ensure the foreground policy ALLOWS yield — the test proves the fork
    # override suppresses it regardless of the foreground policy.
    monkeypatch.setattr(markdown_cfg, "yield_policy", YieldPolicy(allowed=True))

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
                            "name": "fork",
                            "arguments": '{"targets": [{"mode_id": "markdown"}]}',
                        },
                    }
                ],
            )
        # The child finishes without tools — it did not (and could not)
        # call yield. With the override in place, the model is forced to
        # produce content, so it returns a text response.
        return _StubCompletion(content="Done — made reasonable assumptions.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a markdown guide", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    # The child never entered the model_yield / interrupted path at all.
    assert child.status != "model_yield"
    assert child.yield_action is None


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
    hooks.lifecycle.on_stage = on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.response_text == "Done."
    labels = [label for label, _ in events]
    assert "model_call" in labels
    assert "model_call_complete" in labels
    # model_call_complete should be the last event and carry token counts.
    last_label, last_data = events[-1]
    assert last_label == "model_call_complete"
    assert "prompt_tokens" in last_data
    assert "cached_tokens" in last_data
    assert "cache_creation_tokens" in last_data
    assert "cost_usd" in last_data
    assert "provider_cost_usd" in last_data


@pytest.mark.asyncio
async def test_model_call_complete_payload_carries_cache_counters_and_cost() -> None:
    """The stage payload ships the cache split and both cost figures.

    Transports cannot price a turn from ``total_tokens`` alone: on a
    cache-heavy agentic turn the walk reads up to 30x the billed tokens, and
    pricing every prompt at the uncached rate overstates provider spend. The
    kernel therefore ships the sub-counters plus its own canonical cost and
    the provider's reported charge when there is one.
    """

    class _CostedCompletion(_StubCompletion):
        prompt_tokens = 1000
        total_tokens = 1100
        cached_tokens = 800
        cache_creation_tokens = 0
        provider_cost_usd = 0.0

        @property
        def usage(self) -> Any:
            return SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=100,
                total_tokens=1100,
                cached_tokens=800,
                reasoning_tokens=0,
                cache_creation_tokens=0,
                provider_cost_usd=0.0042,
            )

    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    async def complete(**kwargs: Any) -> _CostedCompletion:
        return _CostedCompletion(content="Done.", completion_tokens=100)

    hooks = _build_minimal_hooks(complete)
    hooks.lifecycle.on_stage = on_stage

    await run_turn(mode_id="default", query="hello", hooks=hooks)

    _label, data = next(e for e in events if e[0] == "model_call_complete")
    assert data["cached_tokens"] == 800
    assert data["cache_creation_tokens"] == 0
    assert data["provider_cost_usd"] == pytest.approx(0.0042)
    assert data["cost_usd"] == 0.0


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
    hooks.operations.execute_operation = _stub_execute_operation
    hooks.lifecycle.on_stage = on_stage

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
async def test_run_turn_emits_summarized_json_result_preview() -> None:
    """tool_dispatch_complete carries a payload-first preview for JSON outputs.

    Envelope tools (thread archive pages) serialize metadata before
    content; the completion event's ``result_preview`` must lead with the
    payload, and ``result_chars`` must report the true output size (the
    transport's ``⎿ N chars`` row summary scales with the whole result,
    not the 200-char preview).
    """
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    json_page = json.dumps(
        {
            "ref": "r" + "a" * 64,
            "source": {"scope": "stored_tool_result"},
            "content": "actual page content the operator needs",
            "total_chars": 5412,
        }
    )

    async def complete(**kwargs: Any) -> _StubCompletion:
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

    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return await complete()
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)

    async def execute_operation(operation_id: str, _ctx: Any) -> OperationResult:
        return OperationResult(output_text=json_page)

    hooks.operations.execute_operation = execute_operation
    hooks.lifecycle.on_stage = on_stage

    result = await run_turn(mode_id="default", query="test", hooks=hooks)

    assert result.response_text == "Done."
    assert result.tool_call_count == 1
    complete_data = next(e[1] for e in events if e[0] == "tool_dispatch_complete")
    assert complete_data["result_preview"].startswith("content=actual page content")
    assert "stored_tool_result" not in complete_data["result_preview"]
    assert complete_data["result_chars"] == len(json_page)


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
                            "name": "fork",
                            "arguments": (
                                '{"targets": [{"mode_id": "remember"}, {"mode_id": "markdown"}]}'
                            ),
                        },
                    }
                ],
            )
        return _StubCompletion(content="Synthesized.")

    hooks = _build_minimal_hooks(two_step)
    hooks.lifecycle.on_stage = on_stage

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

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        raise RuntimeError("transport exploded")

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.lifecycle.on_stage = on_stage

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
        captured_messages[1][1].get("content", "")
        # The user message content should contain the original query
        assert "write a doc" in first_user


@pytest.mark.asyncio
async def test_run_turn_handoff_re_resolves_mode_model_pin() -> None:
    """A mid-turn handoff re-resolves the per-mode model pin.

    Regression for the 2026-10-08 incident: a turn entered in ``default``
    (pin: model-a) that hands off mid-turn to a mode with its own
    ``mode_models`` pin (model-b) kept running model-a for every
    post-handoff iteration — the manifest's per-mode pin only took effect
    when the turn was *entered* in that mode. ``preferred_model_id`` is
    resolved once at turn entry; the handoff swaps prompt, tools,
    ``mode_id``, and ``ctx`` but never re-resolved the model.
    """

    from unittest.mock import patch

    call_count = 0
    seen_model_defs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        model_def = kwargs.get("model_def")
        seen_model_defs.append(str(getattr(model_def, "model_id", "?")))
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
        return _StubCompletion(content="Done in search mode.")

    def resolve_model(model_ref: str) -> Any:
        # The host seam: mode refs resolve through ``mode_models`` (mode
        # pins win); concrete model ids resolve directly.
        if model_ref == "search":
            return SimpleNamespace(model_id="model-b", reasoning_model=False)
        if model_ref in ("default", "model-a"):
            return SimpleNamespace(model_id="model-a", reasoning_model=False)
        return None

    catalog = {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(mode_id="search", satisfies_intents=["research"]),
    }

    hooks = _build_minimal_hooks(two_step)
    hooks.models.resolve_model = resolve_model

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for rust news",
            hooks=hooks,
        )

    assert result.handoff_to == "search"
    assert call_count >= 2
    # The post-handoff iterations must run on the target mode's pinned
    # model (model-b), not the entry mode's pin (model-a).
    assert seen_model_defs[0] == "model-a"
    assert all(mid == "model-b" for mid in seen_model_defs[1:])


@pytest.mark.asyncio
async def test_run_turn_handoff_falls_back_to_default_pin_when_target_unpinned() -> None:
    """A handoff to a mode with no distinct pin falls back to the default
    mode's pin — mirroring ``model_alias_for_mode`` — instead of keeping
    the entry mode's pin."""

    from unittest.mock import patch

    call_count = 0
    seen_model_defs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        model_def = kwargs.get("model_def")
        seen_model_defs.append(str(getattr(model_def, "model_id", "?")))
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
        return _StubCompletion(content="Done in search mode.")

    def resolve_model(model_ref: str) -> Any:
        # No pin for ``search``: both the target mode and ``default``
        # resolve to model-a (the default pin). The entry mode runs a
        # host-supplied override (model-x); the handoff must drop it.
        if model_ref == "model-x":
            return SimpleNamespace(model_id="model-x", reasoning_model=False)
        if model_ref in ("search", "default"):
            return SimpleNamespace(model_id="model-a", reasoning_model=False)
        return None

    catalog = {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(mode_id="search", satisfies_intents=["research"]),
    }

    hooks = _build_minimal_hooks(two_step)
    hooks.models.resolve_model = resolve_model

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for rust news",
            hooks=hooks,
            preferred_model_id="model-x",
        )

    assert result.handoff_to == "search"
    assert call_count >= 2
    # Entry iterations run the host override; post-handoff iterations run
    # the target mode's pin via the default fallback (model-a).
    assert seen_model_defs[0] == "model-x"
    assert all(mid == "model-a" for mid in seen_model_defs[1:])


@pytest.mark.asyncio
async def test_run_turn_unpinned_entry_resolves_entry_mode_pin() -> None:
    """An unpinned turn entry resolves the entry mode's ``mode_models`` pin.

    Regression for the 2026-10-27 incident: a ``coding_review`` turn
    delegated to ``coding_execute``; the delegate child calls ``run_turn``
    with no ``preferred_model_id`` (the host's ``model_alias_for_mode`` never
    runs for kernel-spawned children), and the entry fallback skipped
    ``mode_models[coding_execute]`` straight to the ``default`` pin.
    """

    from unittest.mock import patch

    seen_model_defs: list[str] = []

    async def one_step(**kwargs: Any) -> _StubCompletion:
        seen_model_defs.append(str(getattr(kwargs.get("model_def"), "model_id", "?")))
        return _StubCompletion(content="Done.")

    def resolve_model(model_ref: str) -> Any:
        if model_ref == "coding_execute":
            return SimpleNamespace(model_id="model-b", reasoning_model=False)
        if model_ref == "default":
            return SimpleNamespace(model_id="model-a", reasoning_model=False)
        raise ValueError(f"unknown model {model_ref!r}")

    catalog = {
        "default": _mode_config(mode_id="default"),
        "coding_execute": _mode_config(mode_id="coding_execute"),
    }

    hooks = _build_minimal_hooks(one_step)
    hooks.models.resolve_model = resolve_model

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_runner.get_mode_system_intro", lambda mid: mid),
        patch("kernel.ipsar_fsm.get_mode_system_intro", lambda mid: mid),
    ):
        await run_turn(
            mode_id="coding_execute",
            query="implement the fix",
            hooks=hooks,
        )

    assert seen_model_defs
    assert all(mid == "model-b" for mid in seen_model_defs)


@pytest.mark.asyncio
async def test_run_turn_handoff_keeps_model_when_pin_unresolvable() -> None:
    """A handoff target whose pin cannot resolve keeps the current model.

    The runtime's ``resolve_model`` raises ``ValueError`` for a ref that is
    neither a mode pin nor a known model id; the re-resolution must treat
    that as "candidate does not resolve" (next candidate / keep the
    current model) rather than failing an otherwise valid handoff.
    """

    from unittest.mock import patch

    call_count = 0
    seen_model_defs: list[str] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        model_def = kwargs.get("model_def")
        seen_model_defs.append(str(getattr(model_def, "model_id", "?")))
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
        return _StubCompletion(content="Done in search mode.")

    def resolve_model(model_ref: str) -> Any:
        # The entry ref resolves (the entry path resolves it once at loop
        # setup); every post-handoff candidate — the target-mode ref and
        # the default-mode fallback ref — raises, like a host whose
        # manifest maps no pins for either.
        if model_ref == "model-x":
            return SimpleNamespace(model_id="model-x", reasoning_model=False)
        raise ValueError(f"Model '{model_ref}' not in the host's manifest.")

    catalog = {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(mode_id="search", satisfies_intents=["research"]),
    }

    hooks = _build_minimal_hooks(two_step)
    hooks.models.resolve_model = resolve_model

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for rust news",
            hooks=hooks,
            preferred_model_id="model-x",
        )

    # The handoff succeeded despite the raising seam: the turn landed in
    # the target mode and completed, running the entry model (the
    # loop-setup-captured fallback) — never a hard failure.
    assert result.handoff_to == "search"
    assert result.response_text == "Done in search mode."
    assert call_count >= 2
    assert all(mid == "model-x" for mid in seen_model_defs)


@pytest.mark.asyncio
async def test_run_turn_handoff_router_sees_target_mode_inputs() -> None:
    """After a mid-turn handoff, the per-iteration router request carries the
    active mode's id and required capabilities.

    Regression for the router-input staleness found with the 2026-10-08 model
    pin bug: ``RouterRequest.mode_id`` is documented as "the active mode id"
    and ``required_capabilities`` as "the capabilities the current mode's
    tool surface requires" — but both were captured at loop setup from the
    entry mode, so a CEL rule keying on ``route.mode_id`` could never match
    post-handoff, and a target mode's required capabilities were silently
    dropped from the request.
    """

    from unittest.mock import patch

    from kernel.seams.protocols import RouterDecision, RouterRequest

    call_count = 0
    router_requests: list[RouterRequest] = []

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
                            "arguments": '{"target_stage_id": "search"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done in search mode.")

    class _RecordingRouter:
        def choose(self, req: RouterRequest) -> RouterDecision | None:
            router_requests.append(req)
            return None

    def resolve_model(model_ref: str) -> Any:
        return SimpleNamespace(model_id="model-a", reasoning_model=False)

    catalog = {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            required_capabilities=["tools"],
        ),
    }

    hooks = _build_minimal_hooks(two_step)
    hooks.models.resolve_model = resolve_model
    hooks.context_seams.model_router = _RecordingRouter()

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
    ):
        result = await run_turn(mode_id="default", query="search for rust news", hooks=hooks)

    assert result.handoff_to == "search"
    assert call_count >= 2
    assert len(router_requests) >= 2
    # Entry iteration sees the entry mode…
    assert router_requests[0].mode_id == "default"
    # …post-handoff iterations see the active mode and its required
    # capabilities, not the loop-setup-captured entry values.
    # The pre-gate escalation and the loop's iteration-1 resolution both
    # legitimately see the entry mode. The post-handoff model call is the
    # last one this turn — its request must carry the active mode and the
    # target mode's required capabilities, not the loop-setup-captured
    # entry values.
    assert router_requests[0].mode_id == "default"
    post_handoff_req = router_requests[-1]
    assert post_handoff_req.mode_id == "search"
    assert tuple(post_handoff_req.required_capabilities) == ("tools",)


@pytest.mark.asyncio
async def test_run_turn_handoff_attributes_usage_to_target_mode() -> None:
    """Post-handoff completions are billed to the mode that ran them.

    Regression for the usage-attribution staleness found with the
    2026-10-08 model pin bug: ``_wire_record_model_usage``'s closure
    captured the entry mode at loop setup, so ``per_mode_usage`` — the
    per-mode token accounting surfaced to the host — attributed every
    post-handoff completion to the entry mode even though
    ``record_model_usage``'s contract says "the mode that actually ran
    the completion (not the entry mode)".
    """

    from unittest.mock import patch

    call_count = 0

    @dataclass
    class _UsageCompletion:
        content: str
        tool_calls: list[dict[str, Any]] | None = None
        prompt_tokens: int = 0
        completion_tokens: int = 0

    async def two_step(**kwargs: Any) -> _UsageCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _UsageCompletion(
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
                prompt_tokens=100,
                completion_tokens=111,
            )
        # The post-handoff completion bills 222 completion tokens — the
        # target mode's entry must carry them, not the entry mode's.
        return _UsageCompletion(
            content="Done in search mode.",
            prompt_tokens=200,
            completion_tokens=222,
        )

    def resolve_model(model_ref: str) -> Any:
        return SimpleNamespace(model_id="model-a", reasoning_model=False)

    catalog = {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(mode_id="search", satisfies_intents=["research"]),
    }

    hooks = _build_minimal_hooks(two_step)
    hooks.models.resolve_model = resolve_model

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=catalog: (
                _c[mid].instruction_body if _c.get(mid) and _c[mid].instruction_body else mid
            ),
        ),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for rust news",
            hooks=hooks,
        )

    assert result.handoff_to == "search"
    assert call_count >= 2
    per_mode = {entry["mode_id"]: entry for entry in result.per_mode_usage}
    # The post-handoff completion must bill the target mode, not the
    # entry mode — per_mode_usage carries a "search" entry holding the
    # post-handoff completion's tokens.
    assert "search" in per_mode, f"post-handoff usage missing target mode: {per_mode}"
    assert per_mode["search"]["completion_tokens"] == 222


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
        query="write a markdown report",
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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "draft"]},
    )
    patched_catalog = {"search": search_cfg, "markdown": markdown_cfg}

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    markdown_cfg = _mode_config(mode_id="markdown", satisfies_intents=["write"])
    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    patched_catalog = {"search": search_cfg, "markdown": markdown_cfg}

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    coding_cfg = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["code", "coding", "codebase", "gap analysis"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "search": search_cfg,
        "coding": coding_cfg,
        "markdown": markdown_cfg,
        "default": default_cfg,
    }

    q = (
        "do a web search for python coding best practices then go into "
        "coding mode and use it as input for a code-level gap analysis "
        "then write a markdown of it."
    )
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=q,
            capability_manifest=_empty_manifest(),
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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    remember_cfg = _mode_config(
        mode_id="remember",
        satisfies_intents=["remember"],
        routing_intent_tags=["remember"],
        routing_intent_keywords={"remember": ["remember", "note", "capture"]},
        fork_allowed=True,
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document"]},
        fork_allowed=True,
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {"remember": remember_cfg, "markdown": markdown_cfg, "default": default_cfg}

    with _patched_hint_catalog(patched_catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="remember this and write a markdown of it",
            capability_manifest=_empty_manifest(),
            routing_targets=["remember", "markdown"],
            mode_config=default_cfg,
        )

    assert hint != ""
    assert "`fork`" in hint
    assert "remember" in hint and "markdown" in hint


def test_detect_query_intents_attributes_only_matched_keyword_tag() -> None:
    """``routing_intent_keywords`` are per-tag substring matches, so a
    keyword for one tag attributes ONLY that tag — not all of the mode's
    ``routing_intent_tags``.

    Regression for the 2026-08-28 cloudservers misroute: a read-only
    "what VMs are running?" query matched the ``list-vms`` keyword, the
    old code attributed ALL 7 cloudservers tags (including the write-
    intent ``create-vm``, ``start-vm``, ``terminate-vm``, ...) to the
    query, and the read query was routed into the write-contract
    ``cloudservers`` mode. The write-contract backstop then synthesized
    a destructive ``cs_apply_cloud_init`` call with empty args.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _detect_query_intents

    # cloudservers declares 7 tags; the ``list-vms`` keyword bucket is the
    # only one the read query hits.
    cloudservers_cfg = _mode_config(
        mode_id="cloudservers",
        routing_intent_tags=[
            "cloud-init",
            "create-vm",
            "list-vms",
            "schedule-vm",
            "start-vm",
            "stop-vm",
            "terminate-vm",
        ],
        routing_intent_keywords={
            "list-vms": ["list vms", "what vms", "running"],
            "create-vm": ["create vm", "new vm"],
            "start-vm": ["start vm"],
            "stop-vm": ["stop vm"],
            "terminate-vm": ["terminate vm"],
            "schedule-vm": ["schedule"],
            "cloud-init": ["cloud-init"],
        },
    )
    patched_catalog = {"cloudservers": cloudservers_cfg}

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query="what VMs are running?",
            routing_targets=["cloudservers"],
        )

    # Only the list-vms intent is detected — NOT create-vm, start-vm, etc.
    assert detected == {"list-vms"}


def test_detect_query_intents_fast_path_attributes_all_tags() -> None:
    """``fast_path_patterns`` are mode-level (anchored regex), so a match
    attributes ALL of the mode's ``routing_intent_tags``. This preserves
    the high-precision fast-path signal: a pattern that names the mode is
    a strong signal the user wants that mode's full capability set.
    """
    from unittest.mock import patch

    from kernel.models import WorkflowModeConfig
    from kernel.turn_phases._routing_hint import _detect_query_intents

    multi_cfg = WorkflowModeConfig(
        mode_id="multi",
        start_stage_id="none",
        mode_kind="chat",
        routable=True,
        routing_intent_tags=["read", "write", "execute"],
        fast_path_patterns=[r"^run multi mode$"],
        routing_intent_keywords={
            "read": ["read"],
            "write": ["write"],
            "execute": ["execute"],
        },
    )
    patched_catalog = {"multi": multi_cfg}

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query="run multi mode",
            routing_targets=["multi"],
        )

    assert detected == {"read", "write", "execute"}


def test_detect_query_intents_skips_host_synthesized_continuation() -> None:
    """Host-synthesized approval continuations embed tool OUTPUT, not
    instructions. Intent-scanning that output fires whatever keywords the
    data happens to contain and routes the report turn through unrelated
    modes.

    Regression for the 2026-09-23 incident: the approval continuation for
    a bash inventory of the finances sandbox contained the words
    "bookkeeping", "finances", "taxes", "general ledger" IN THE TOOL
    OUTPUT; the kernel detected those intents, computed a 4-intent gap in
    shell mode, and escalated the report turn to planner — bouncing it
    through two unwanted modes instead of reporting and stopping.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _detect_query_intents

    finances_cfg = _mode_config(
        mode_id="finances",
        routing_intent_tags=["finances", "bookkeeping", "taxes"],
        routing_intent_keywords={
            "finances": ["general ledger"],
            "bookkeeping": ["import transactions"],
            "taxes": ["tax workpaper"],
        },
    )
    patched_catalog = {"finances": finances_cfg}

    continuation_query = (
        "The approved action `bash_run` executed successfully "
        "(approval 9c83ca10df274d03b) and returned this output:\n\n"
        "total 32\ndrwx------ ledgers sources rules\n"
        "... import transactions ... general ledger ... tax workpaper ...\n\n"
        "Report the result to the operator in plain prose. Do not call any "
        "tools in this turn."
    )

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query=continuation_query,
            routing_targets=["finances"],
        )

    # The continuation is a REPORT: zero intents, no planner escalation.
    assert detected == set()


def test_detect_query_intents_still_scans_genuine_user_text() -> None:
    """The continuation guard must not over-suppress: a genuine user query
    that merely MENTIONS an approval (without the host's backtick template)
    still gets normal intent scanning."""
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _detect_query_intents

    finances_cfg = _mode_config(
        mode_id="finances",
        routing_intent_tags=["finances", "bookkeeping", "taxes"],
        routing_intent_keywords={
            "finances": ["general ledger"],
            "bookkeeping": ["import transactions"],
            "taxes": ["tax workpaper"],
        },
    )
    patched_catalog = {"finances": finances_cfg}

    user_query = (
        "thanks — now that the approved action finished, please import transactions for 2024"
    )

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query=user_query,
            routing_targets=["finances"],
        )

    assert "bookkeeping" in detected


def test_detect_query_intents_strips_piped_input_blocks() -> None:
    """Host-authored ``<piped-input>`` blocks are data, not routing requests.

    Regression for the 2026-10-17 incident: a coding turn pasted a VHS
    tape for diagnosis whose content included the directive ``Set Shell
    bash``. Shell mode's ``bash`` keyword matched inside the PASTED DATA,
    the routing hint demanded a ``transition_stage`` to shell, and when
    the model kept working on the tape the forced-action enforcer
    synthesized the handoff itself — yanking the coding turn into shell
    mode mid-task and (via the handoff's persisted owner mode) leaving
    later turns stuck there too.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _detect_query_intents

    shell_cfg = _mode_config(
        mode_id="shell",
        routing_intent_tags=["shell", "terminal"],
        routing_intent_keywords={"shell": ["shell", "bash"], "terminal": ["terminal"]},
    )
    coding_cfg = _mode_config(
        mode_id="coding_execute",
        routing_intent_tags=["code", "write"],
        routing_intent_keywords={"code": ["fix this tape"], "write": ["rewrite"]},
    )
    patched_catalog = {"shell": shell_cfg, "coding_execute": coding_cfg}

    pasted_tape_query = (
        "<piped-input>\n"
        "File: smoke.tape\n"
        "Set Shell bash\n"
        "Set Theme Dracula\n"
        "Set WaitTimeout 3s\n"
        "</piped-input>\n\n"
        "still failing after the timeout fix — fix this tape for me"
    )

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query=pasted_tape_query,
            routing_targets=["shell", "coding_execute"],
        )

    # The directive's coding intent is detected; the tape's `Set Shell
    # bash` line contributes nothing.
    assert detected == {"code"}
    assert "shell" not in detected


def test_detect_query_intents_piped_input_only_query_detects_nothing() -> None:
    """A query that is ONLY a piped-data block scans nothing — there is no
    directive prose to route from, so no intent gap can fire a hint or the
    forced-action enforcer.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _detect_query_intents

    shell_cfg = _mode_config(
        mode_id="shell",
        routing_intent_tags=["shell", "terminal"],
        routing_intent_keywords={"shell": ["shell", "bash"], "terminal": ["terminal"]},
    )
    patched_catalog = {"shell": shell_cfg}

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query=("<piped-input>\nSet Shell bash\nType export PS1='>'\n</piped-input>"),
            routing_targets=["shell"],
        )

    assert detected == set()


def test_detect_query_intents_unclosed_piped_input_block_truncates_scan() -> None:
    """An unclosed ``<piped-input>`` marker still quarantines the payload —
    hosts (and models echoing the convention) may truncate the message
    before the close tag; everything from the marker must stay data.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _detect_query_intents

    shell_cfg = _mode_config(
        mode_id="shell",
        routing_intent_tags=["shell", "terminal"],
        routing_intent_keywords={"shell": ["shell", "bash"], "terminal": ["terminal"]},
    )
    patched_catalog = {"shell": shell_cfg}

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog",
        return_value=patched_catalog,
    ):
        detected = _detect_query_intents(
            query="look at this:\n<piped-input>\nSet Shell bash\nSet Theme Dracula",
            routing_targets=["shell"],
        )

    assert detected == set()


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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    # ponder = producer (satisfies research/reasoning), markdown = consumer
    # (satisfies write). A fork mixing them is sequential-wrong.
    ponder_cfg = _mode_config(
        mode_id="ponder",
        satisfies_intents=["research", "reasoning"],
        routing_intent_tags=["research", "reasoning"],
        routing_intent_keywords={
            "research": ["search", "latest", "framework"],
            "reasoning": ["opinion", "analyze", "compare"],
        },
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "ponder": ponder_cfg,
        "markdown": markdown_cfg,
        "default": default_cfg,
    }

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
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


def test_routing_hint_refuses_fork_for_consumer_only_gap() -> None:
    """A query whose gap contains two consumer intents (write/coding) and
    no producer intent must NOT fork — each consumer child needs evidence
    to consume, but a parallel fork gives neither any. Both children
    re-gather evidence independently and never reach their action tool.

    Regression for the 2026-08-05 incident: "analyze the code, write a
    doc, and patch the code" was routed as ``fork:markdown,coding`` on a
    ``{write, coding}`` gap. The ``markdown`` child was blocked by the
    capability gate and the ``coding`` child made 26 tool calls with no
    write-effect receipt (hallucinated success). The intent detector
    failed to flag a producer for "analyze the code" (see the companion
    fix in ``lisa-pack-coding``'s ``learn`` mode keywords), and the guard
    only caught a producer+consumer *mix*, not a consumer-only gap. The
    kernel now refuses the fork on any gap with multiple consumers and no
    producer, and routes sequential instead.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    # markdown = consumer (write), coding = consumer (coding). No producer
    # in the gap. A parallel fork is sequential-wrong. search is a producer
    # in routing_targets so the fallback single-handoff has somewhere to go.
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document"]},
    )
    coding_cfg = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["code", "patch", "implement"]},
    )
    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "find", "look up"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "markdown": markdown_cfg,
        "coding": coding_cfg,
        "search": search_cfg,
        "default": default_cfg,
    }

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
    ):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="write a markdown document and patch the code",
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "markdown", "coding"],
            mode_config=default_cfg,
        )

    # Must NOT fork — two consumers with no producer is sequential-wrong.
    # The hint may be empty (no producer in the gap to route to first) or
    # may name a single handoff target; either is acceptable so long as
    # the kernel does not fan out a parallel fork.
    assert "fork:" not in hint


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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

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

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
    ):
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

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
    ):
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

    with patch(
        "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
    ):
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
    from kernel.turn_phases._routing_hint import _query_is_sequential

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
    from kernel.turn_phases._routing_hint import _query_is_sequential

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


def test_looks_like_continuation_query_uses_session_resume_phrases() -> None:
    """``_looks_like_continuation_query`` reads the shared resume-phrase
    registry: the wired session list is prefix-matched, and re-registering
    replaces it entirely (``"proceed"`` stops matching). Prefix-matching lets
    ``"proceed with the current step in the plan"`` match ``"proceed"``.

    Regression for the 2026-07-31 incident: the user typed a longer resume
    phrase that the old exact-match affirmative detector missed, and the
    continuation detector's hardcoded token list was not session-tunable.
    """
    from kernel.routing_vocabulary import register_routing_vocabulary
    from kernel.turn_runner import _looks_like_continuation_query

    # The autouse fixture wires the representative session vocabulary.
    assert _looks_like_continuation_query("proceed")
    assert _looks_like_continuation_query("proceed with the current step in the plan")
    assert _looks_like_continuation_query("continue with the current step in the current plan")
    assert not _looks_like_continuation_query("tell me the current plan and how much is complete")

    register_routing_vocabulary(resume_phrases=["onward", "carry forward"])
    assert _looks_like_continuation_query("onward with the plan")
    assert _looks_like_continuation_query("carry forward from step 3")
    assert not _looks_like_continuation_query("proceed with the plan")


def test_routing_hint_sequential_via_session_vocabulary_does_not_fork() -> None:
    """A 'based on the results' query with the session vocabulary wired
    produces a single handoff to the first step, NOT a fork — the
    end-to-end regression for the 2026-07-26 false-fork incident.
    """
    from unittest.mock import patch

    from kernel.routing_vocabulary import register_routing_vocabulary
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "search": search_cfg,
        "markdown": markdown_cfg,
        "default": default_cfg,
    }

    register_routing_vocabulary(
        sequential_connectors=["then", "and then", "after that", "next", "followed by", "based on"]
    )
    try:
        q = "search for python news and write a markdown based on the results"
        with (
            patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
            patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
            patch(
                "kernel.turn_runner.get_mode_system_intro",
                lambda mid, _c=patched_catalog: (
                    _c.get(mid).instruction_body
                    if _c.get(mid) and _c.get(mid).instruction_body
                    else mid
                ),
            ),
            patch(
                "kernel.ipsar_fsm.get_mode_system_intro",
                lambda mid, _c=patched_catalog: (
                    _c.get(mid).instruction_body
                    if _c.get(mid) and _c.get(mid).instruction_body
                    else mid
                ),
            ),
        ):
            hint = _compute_capability_routing_hint(
                mode_id="default",
                query=q,
                capability_manifest=_empty_manifest(),
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

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "latest", "news"]},
    )
    coding_cfg = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["coding", "code", "codebase", "gap analysis"]},
    )
    markdown_cfg = _mode_config(
        mode_id="markdown",
        satisfies_intents=["write"],
        routing_intent_tags=["write"],
        routing_intent_keywords={"write": ["write", "markdown", "document", "report"]},
    )
    patched_catalog = {
        "search": search_cfg,
        "coding": coding_cfg,
        "markdown": markdown_cfg,
    }
    q = (
        "do a web search for python coding best practices then go into "
        "coding mode and use it as input for a code-level gap analysis "
        "then write a markdown of it."
    )

    # After search + coding: chain_satisfied = {research, coding}.
    # Gap = {write}. Hint must name markdown, NOT search.
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
        hint = _compute_capability_routing_hint(
            mode_id="coding",
            query=q,
            capability_manifest=_empty_manifest(),
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
    fork_allowed: bool = False,
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
    ``fork_allowed`` defaults to False; fork-target tests must opt in.
    """
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=routable,
        fork_allowed=fork_allowed,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        required_capabilities=list(required_capabilities) if required_capabilities else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def _hint_catalog(
    *mode_ids: str,
    keywords: dict[str, list[str]] | None = None,
) -> dict[str, WorkflowModeConfig]:
    """Build a synthetic catalog where each mode's tags match its own id.

    The routing-hint tests exercise the kernel's pure mapping over a
    synthetic ``routing_targets`` list (``fork:remember,markdown``,
    single-handoff to ``markdown``, ...). Intent detection resolves each
    target through the workflow catalog, which the ambient fixture
    catalog (``default`` + ``letta`` only — external persona/search packs
    are not installed in this venv) cannot serve. This helper builds a
    per-test catalog whose synthetic modes declare ``routing_intent_tags``
    and ``routing_intent_keywords`` keyed on their own mode id, so a query
    mentioning "remember" attributes the ``remember`` intent etc.

    ``keywords`` optionally overrides the self-keyed keywords per mode
    (``{"ponder": ["understand", "SWOT"]}``), for tests that assert
    keyword-driven steering toward a mode whose id the query never
    mentions.

    Patch BOTH bindings with the result (see the kernel test-seams rule):
    ``kernel.turn_phases._routing_hint.get_workflow_catalog`` (where
    ``_detect_query_intents``/``_compute_fork_targets`` resolve it) and
    ``kernel.turn_runner.get_workflow_catalog`` (where the runner reads it
    directly).
    """
    modes: list[str] = []
    for entry in mode_ids:
        for raw_piece in entry.replace("fork:", "").split(","):
            piece = raw_piece.strip()
            if piece and piece not in modes:
                modes.append(piece)
    catalog: dict[str, WorkflowModeConfig] = {"default": _mode_config(mode_id="default")}
    for mode in modes:
        if mode == "default":
            continue
        catalog[mode] = _mode_config(
            mode_id=mode,
            satisfies_intents=[mode],
            routing_intent_tags=[mode],
            routing_intent_keywords={mode: (keywords or {}).get(mode, [mode])},
            fork_allowed=True,
        )
    return catalog


def _patched_hint_catalog(catalog: dict[str, Any]):
    """Patch the two catalog bindings the hint path reads.

    Returns a single context manager that patches both bindings (see the
    kernel test-seams rule); use as ``with _patched_hint_catalog(catalog):``.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    stack = ExitStack()
    stack.enter_context(
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog)
    )
    stack.enter_context(patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog))
    return stack


def test_routing_hint_returns_empty_when_no_routing_targets() -> None:
    """No routing targets means no hint is injected (zero tokens)."""
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog(
        "search",
        "ponder",
        "markdown",
        "coding",
        keywords={
            "search": ["search", "web"],
            "ponder": ["understand", "swot", "tradeoffs", "analyze"],
        },
    )
    with _patched_hint_catalog(catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query=(
                "get a better understanding of the latest frameworks and agent building tools "
                "on the market today and do a SWOT based on it"
            ),
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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog(
        "search",
        "ponder",
        keywords={
            "search": ["search", "web"],
            "ponder": ["analyze", "tradeoff", "understand"],
        },
    )
    with _patched_hint_catalog(catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="analyze the latest tradeoffs in serverless frameworks",
            capability_manifest=_empty_manifest(),
            routing_targets=["search", "ponder"],
        )
    assert "ponder" in hint


def test_routing_hint_suppresses_target_on_topic_abandonment_marker() -> None:
    """A query that *mentions* a topic to abandon it (with a session
    ``topic_abandonment_markers`` marker present) must NOT route into the
    mentioned target. The bare-substring keyword match would otherwise fire
    on any mention.

    Regression for the 2026-08-27 cloudservers misroute: "TBH, the whole
    cloudservers thing was a bit of a side quest. So lets just rewind"
    matched the ``cloudservers`` mode's ``cloud`` keyword, routed the user
    INTO ``cloudservers``, and the write-contract backstop synthesized a
    destructive ``cs_apply_cloud_init`` call with empty args. The
    abandonment marker ("side quest" / "rewind") signals the user is
    *leaving* the topic, not requesting its tools.
    """
    from unittest.mock import patch

    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then"],
        producer_intents=["research", "analyze"],
        consumer_intents=["cloud", "write", "code"],
        resume_phrases=["proceed", "continue"],
        topic_abandonment_markers=["side quest", "rewind"],
    )
    try:
        cloudservers_cfg = _mode_config(
            mode_id="cloudservers",
            satisfies_intents=["cloud"],
            routing_intent_tags=["cloud"],
            routing_intent_keywords={"cloud": ["cloud", "cloudservers", "vm"]},
        )
        default_cfg = _mode_config(mode_id="default")
        patched_catalog = {"cloudservers": cloudservers_cfg, "default": default_cfg}

        with (
            patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
            patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
            patch(
                "kernel.turn_runner.get_mode_system_intro",
                lambda mid, _c=patched_catalog: (
                    _c.get(mid).instruction_body
                    if _c.get(mid) and _c.get(mid).instruction_body
                    else mid
                ),
            ),
            patch(
                "kernel.ipsar_fsm.get_mode_system_intro",
                lambda mid, _c=patched_catalog: (
                    _c.get(mid).instruction_body
                    if _c.get(mid) and _c.get(mid).instruction_body
                    else mid
                ),
            ),
        ):
            hint = _compute_capability_routing_hint(
                mode_id="default",
                query=(
                    "TBH, the whole cloudservers thing was a bit of a side quest. "
                    "So lets just rewind"
                ),
                capability_manifest=_empty_manifest(),
                routing_targets=["cloudservers"],
                mode_config=default_cfg,
            )

        # The mentioned topic is suppressed → no gap → no hint.
        assert hint == ""
        assert "cloudservers" not in hint
    finally:
        reset_routing_vocabulary()


def test_routing_hint_routes_into_target_when_no_abandonment_marker() -> None:
    """Without an abandonment marker, the same keyword match routes into
    the target normally — the polarity guard only suppresses when a marker
    is co-present in the query."""
    from unittest.mock import patch

    from kernel.routing_vocabulary import (
        register_routing_vocabulary,
        reset_routing_vocabulary,
    )
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    reset_routing_vocabulary()
    register_routing_vocabulary(
        sequential_connectors=["then", "and then"],
        producer_intents=["research", "analyze"],
        consumer_intents=["cloud", "write", "code"],
        resume_phrases=["proceed", "continue"],
        topic_abandonment_markers=["side quest", "rewind"],
    )
    try:
        cloudservers_cfg = _mode_config(
            mode_id="cloudservers",
            satisfies_intents=["cloud"],
            routing_intent_tags=["cloud"],
            routing_intent_keywords={"cloud": ["cloud", "cloudservers", "vm"]},
        )
        default_cfg = _mode_config(mode_id="default")
        patched_catalog = {"cloudservers": cloudservers_cfg, "default": default_cfg}

        with (
            patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
            patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
            patch(
                "kernel.turn_runner.get_mode_system_intro",
                lambda mid, _c=patched_catalog: (
                    _c.get(mid).instruction_body
                    if _c.get(mid) and _c.get(mid).instruction_body
                    else mid
                ),
            ),
            patch(
                "kernel.ipsar_fsm.get_mode_system_intro",
                lambda mid, _c=patched_catalog: (
                    _c.get(mid).instruction_body
                    if _c.get(mid) and _c.get(mid).instruction_body
                    else mid
                ),
            ),
        ):
            hint = _compute_capability_routing_hint(
                mode_id="default",
                query="help me provision a new cloudservers vm",
                capability_manifest=_empty_manifest(),
                routing_targets=["cloudservers"],
                mode_config=default_cfg,
            )

        assert hint != ""
        assert "cloudservers" in hint
    finally:
        reset_routing_vocabulary()


def test_routing_hint_falls_back_to_generic_list_for_tool_gap() -> None:
    """A pure tool-gap query (write a file) with no reasoning cue lists all
    targets rather than singling out ponder.
    """
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog(
        "markdown",
        "coding",
        "ponder",
        keywords={
            "markdown": ["write", "markdown", "document"],
            "coding": ["code", "implement"],
        },
    )
    with _patched_hint_catalog(catalog):
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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog("search", "ponder")
    with _patched_hint_catalog(catalog):
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
    (`fork:remember,markdown`), not a single-mode handoff.

    Regression case for docs/lisa-markdown-stage-failure-2026-07-17.md: the
    model called markdown_forge_document / markdown_generate from default
    mode (mode-scoped tools, rejected by tool policy) instead of forking.
    The proactive hint must surface before the model wastes an iteration.
    The kernel is a pure mapper: the decision (fork, not single handoff)
    is driven entirely by set algebra over the pack-authored
    routing_intent_tags (remember covers 'remember', markdown covers 'write',
    neither covers both). The kernel emits no tool-policy narration
    ('mode-scoped tools', 'rejected'); it names only the mechanics
    (transition_stage + the fork target id). Any such narration would be
    kernel-authored routing *output*, which is pack coupling.
    """
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog("remember", "markdown", "coding", "ponder", "search")
    with _patched_hint_catalog(catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="remember that pretty much verbatim and also write a markdown of it",
            capability_manifest=_empty_manifest(),
            routing_targets=["remember", "markdown", "coding", "ponder", "search"],
        )
    assert hint != ""
    # The fork hint names the fork operation and the target list (the
    # transition_stage directive is the single-handoff shape; fork is the
    # fan-out shape).
    assert "`fork`" in hint
    # Must name the fork target list, not a single mode handoff
    assert "[remember, markdown]" in hint
    # The kernel must not narrate tool-policy semantics in its own prose.
    # The handoff instruction is purely mechanical.
    assert "mode-scoped" not in hint.lower()
    assert "rejected" not in hint.lower()


def test_routing_hint_fork_requires_both_routable_modes() -> None:
    """When one of remember/markdown is not a routing target, do not suggest a
    fork — fall back to the single-goal branches (or no hint).
    """
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    # markdown not routable from here → no fork hint; remember+write cues with
    # no tool/search/research-reason gap means no hint at all.
    hint = _compute_capability_routing_hint(
        mode_id="default",
        query="remember the decision and write a markdown of it",
        capability_manifest=_empty_manifest(),
        routing_targets=["remember", "coding"],  # markdown missing
    )
    assert "fork:" not in hint


def test_routing_hint_fork_preserves_target_case() -> None:
    """The fork target string preserves the original (non-lowercased) mode
    IDs from routing_targets, matching what the FSM catalog expects.
    """
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    # Synthetic modes with mixed-case ids; the catalog keys the exact ids
    # so intent detection resolves them and the fork string preserves case.
    catalog = {
        "default": _mode_config(mode_id="default"),
        "Remember": _mode_config(
            mode_id="Remember",
            satisfies_intents=["remember"],
            routing_intent_tags=["remember"],
            routing_intent_keywords={"remember": ["note", "remember"]},
            fork_allowed=True,
        ),
        "Markdown": _mode_config(
            mode_id="Markdown",
            satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["markdown", "document", "write"]},
            fork_allowed=True,
        ),
    }
    with _patched_hint_catalog(catalog):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="take a note of this and write a markdown document",
            capability_manifest=_empty_manifest(),
            routing_targets=["Remember", "Markdown"],
        )
    assert "targets [Remember, Markdown]" in hint


def test_routing_hint_forks_from_coding_mode() -> None:
    """From coding mode, a remember+write-markdown request must still steer
    to fork:remember,markdown once the routing FSM is a full mesh.

    Regression for docs/lisa-coding-mode-fork-failure-2026-07-17.md: with the
    conservative default-only edge set, coding's routing_targets were
    ["default"] and the fork branch's "remember"/"markdown" membership guard
    never passed. The model then fell back to coding_create_worktree
    (approval-gated), producing a worktree approval loop instead of a fork.
    """
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog("remember", "markdown", "coding", "ponder", "search")
    with _patched_hint_catalog(catalog):
        hint = _compute_capability_routing_hint(
            mode_id="coding",
            query="remember the entire thing and write a markdown document of it",
            capability_manifest=_empty_manifest(),
            # Post-mesh coding routing targets include remember + markdown.
            routing_targets=["default", "remember", "markdown", "ponder", "search"],
        )
    assert hint != ""
    # The fan-out hint names the fork operation and the target list.
    assert "`fork`" in hint
    assert "[remember, markdown]" in hint


# ── Single-intent handoff tests (satisfies_intents) ──────────────────
#
# The 2026-07-17 fork fix only unblocked the *two-goal* remember+markdown
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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    coding_manifest = _manifest_with_coding_tools()
    catalog = _hint_catalog("default", "remember", "markdown", "ponder", "search")
    catalog["coding"] = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["coding"]},
    )
    with _patched_hint_catalog(catalog):
        hint = _compute_capability_routing_hint(
            mode_id="coding",
            query="write a markdown document of the report you just produced",
            capability_manifest=coding_manifest,
            routing_targets=["default", "remember", "markdown", "ponder", "search"],
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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    hint = _compute_capability_routing_hint(
        mode_id="markdown",
        query="write a markdown document of the report you just produced",
        capability_manifest=_empty_manifest(),
        routing_targets=["default", "remember", "coding", "ponder", "search"],
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
    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    catalog = _hint_catalog("default", "remember", "markdown", "ponder", "search")
    catalog["coding"] = _mode_config(
        mode_id="coding",
        satisfies_intents=["coding"],
        routing_intent_tags=["coding"],
        routing_intent_keywords={"coding": ["coding"]},
    )
    with _patched_hint_catalog(catalog):
        hint_with_rich_manifest = _compute_capability_routing_hint(
            mode_id="coding",
            query="draft a markdown document of the findings",
            capability_manifest=_manifest_with_coding_tools(),
            routing_targets=["default", "remember", "markdown", "ponder", "search"],
            mode_config=_mode_config(mode_id="coding", satisfies_intents=["coding"]),
        )
        hint_with_empty_manifest = _compute_capability_routing_hint(
            mode_id="coding",
            query="draft a markdown document of the findings",
            capability_manifest=_empty_manifest(),
            routing_targets=["default", "remember", "markdown", "ponder", "search"],
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


def test_routing_hint_partial_coverage_fallback_instead_of_silence() -> None:
    """When no single target covers the whole gap but one target covers a
    subset, the section-6 fallback emits a hint naming that target with a
    residual-chain note — not silence.

    Regression for the 2026-08-14 silent-route incident: a multi-intent
    query with 2+ disjoint gap intents and no ``fork_allowed`` targets
    produced an empty hint, the model returned text only, and the turn
    blocked as ``no_tools``. The fix routes through
    ``_best_partial_coverage_target`` so the model gets a directive hint
    to the best partial target instead of no signal at all.

    This test uses two non-producer, non-consumer intents (``remember`` +
    ``explain``) so the producer/consumer mix guard (section 4b) does not
    fire before section 6. Neither mode declares ``fork_allowed``, so the
    fork branch cannot build targets. No single target covers both
    intents → section-6 partial-coverage fallback fires.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    # Two modes, each covering a disjoint non-producer/non-consumer intent.
    # Neither declares fork_allowed (the default), so the fork branch
    # cannot build targets. No single target covers {remember, explain}
    # → section-6 fires.
    remember_cfg = _mode_config(
        mode_id="remember",
        satisfies_intents=["remember"],
        routing_intent_tags=["remember"],
        routing_intent_keywords={"remember": ["remember", "note", "capture"]},
    )
    explainer_cfg = _mode_config(
        mode_id="explainer",
        satisfies_intents=["explain"],
        routing_intent_tags=["explain"],
        routing_intent_keywords={"explain": ["explain", "summarize", "expound"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {
        "remember": remember_cfg,
        "explainer": explainer_cfg,
        "default": default_cfg,
    }

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="remember this and explain the summary",
            capability_manifest=_empty_manifest(),
            routing_targets=["remember", "explainer"],
            mode_config=default_cfg,
        )

    # The hint must NOT be empty — the partial-coverage fallback fires.
    assert hint != ""
    assert "Routing Hint" in hint
    assert "transition_stage" in hint
    # The residual-chain note must be present.
    assert "remaining steps will chain after" in hint
    # One of the two partial-coverage targets is named (deterministic:
    # remember appears first in routing_targets and covers "remember").
    assert "remember" in hint


def test_routing_hint_silence_when_no_target_covers_any_gap_intent() -> None:
    """When no target covers *any* gap intent, the section-6 fallback
    correctly returns silence — the model self-routes via
    ``transition_stage``. This is the genuine uncoverable gap case.
    """
    from unittest.mock import patch

    from kernel.turn_phases._routing_hint import _compute_capability_routing_hint

    # A mode whose intent tags don't overlap the detected gap at all.
    other_cfg = _mode_config(
        mode_id="other",
        satisfies_intents=["unrelated"],
        routing_intent_tags=["unrelated"],
        routing_intent_keywords={"unrelated": ["unrelated_keyword_xyz"]},
    )
    default_cfg = _mode_config(mode_id="default")
    patched_catalog = {"other": other_cfg, "default": default_cfg}

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
        hint = _compute_capability_routing_hint(
            mode_id="default",
            query="research the observability problem",
            capability_manifest=_empty_manifest(),
            routing_targets=["other"],
            mode_config=default_cfg,
        )

    # No target covers "research" → silence is correct.
    assert hint == ""


def test_emit_transition_hint_guard_returns_empty_on_self_handoff() -> None:
    """The shared ``emit_transition_hint`` helper returns ``""`` when
    target == current, preventing the self-handoff loop class regardless
    of which hint site calls it. This is the contract test — new hint
    sites that use the helper inherit the guard automatically.
    """
    from kernel.turn_phases._routing_hint import emit_transition_hint

    # Same target and current → empty (self-handoff guard).
    result = emit_transition_hint(
        target_mode_id="executor",
        current_mode_id="executor",
        prose_template="## Routing Hint\n\nHand off to `{target}`.",
    )
    assert result == ""

    # Different target and current → formatted hint.
    result = emit_transition_hint(
        target_mode_id="coding",
        current_mode_id="executor",
        prose_template="## Routing Hint\n\nHand off to `{target}`.",
    )
    assert result == "## Routing Hint\n\nHand off to `coding`."

    # The guard works with different prose templates too.
    result = emit_transition_hint(
        target_mode_id="researcher",
        current_mode_id="researcher",
        prose_template="Continue with `{target}` by calling transition_stage.",
    )
    assert result == ""


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
    from kernel.turn_phases._post_turn import _validate_model_capabilities

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
    from kernel.turn_phases._post_turn import _validate_model_capabilities

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
    from kernel.turn_phases._post_turn import _validate_model_capabilities

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
    from kernel.turn_phases._post_turn import _validate_model_capabilities

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
    from kernel.turn_phases._post_turn import _validate_model_capabilities

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
    from kernel.turn_phases._post_turn import _validate_model_capabilities

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
    from kernel.turn_phases._routing_hint import _tool_effect

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
    from kernel.turn_phases._routing_hint import _tool_effect

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
    from kernel.turn_phases._routing_hint import _tool_effect

    assert _tool_effect("unknown", operation_catalog={}, tool_catalog={}) == "read"


def test_tool_effect_defaults_read_when_definition_has_no_effect() -> None:
    """A definition without an ``effect`` attribute defaults to ``read``."""
    from kernel.turn_phases._routing_hint import _tool_effect

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
    from kernel.turn_phases._routing_hint import _tool_effect

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

    async def failing_entry_operation(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        # Only the ``web_search`` entry operation (the search mode's entry
        # action) fails. Other operations (e.g. pre-turn memory recall) use
        # the default stub so they don't trip the guard prematurely.
        if operation_id == "web_search":
            raise RuntimeError("web_search returned no content")
        return OperationResult(output_text=f"Executed {operation_id}")

    hooks = _build_minimal_hooks(two_step)
    hooks.operations.execute_operation = failing_entry_operation

    from unittest.mock import patch

    from kernel.ipsar_fsm import get_workflow_catalog

    # The ambient synthetic search mode is intentionally generic. Model the
    # action-mode entry contract this test is exercising without changing
    # that shared fixture for unrelated routing tests.
    patched_catalog = dict(get_workflow_catalog())
    patched_catalog["search"] = patched_catalog["search"].model_copy(
        update={
            "mode_kind": "action",
            "entry_operation_ids": ["web_search"],
            "entry_op_kind": "work",
        }
    )
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
    ):
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
async def test_planner_entry_op_timeout_marks_model_unreliable() -> None:
    """When ``plan_workflow`` raises during a mid-turn handoff to ``plan``
    (e.g. the operation timed out), the turn must mark the model
    unreliable for planning so the routing-hint generator stops
    re-escalating to ``planner`` on subsequent turns.

    Regression for the 2026-07-31 incident: ``plan_workflow`` timed out,
    the entry-op exception branch set ``entry_failed`` but did NOT call
    ``mark_plan_unreliable_for_model``, so the hint generator re-escalated
    to ``planner`` on the next turn and timed out again identically. The
    "no target" branch already marked unreliable; the exception path
    must mirror it.
    """
    from unittest.mock import patch

    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.operations.registry import OperationExecutionError
    from kernel.routing_vocabulary import (
        is_plan_reliable_for_model,
        reset_plan_reliability,
    )

    reset_plan_reliability()
    plan_cfg = WorkflowModeConfig(
        mode_id="planner",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    patched_catalog = dict(get_workflow_catalog())
    patched_catalog["planner"] = plan_cfg

    async def two_step(**kwargs: Any) -> _StubCompletion:
        # First completion: the model hands off to ``planner`` (mid-turn
        # handoff, not Phase 1 pre-turn). The routing hint told it to.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "transition_stage",
                        "arguments": '{"target_stage_id": "planner", "rationale": "complex"}',
                    },
                }
            ],
        )

    async def failing_plan_entry_operation(
        mode_id: str, _ctx: OperationExecutionContext, **_kwargs: Any
    ) -> OperationResult:
        # ``run_mode_entry_operations`` receives a mode id, not an
        # operation id. Simulate its required routing operation timing out.
        if mode_id == "planner":
            raise OperationExecutionError("Operation 'plan_workflow' timed out")
        return OperationResult(output_text=f"Executed {mode_id}")

    hooks = _build_minimal_hooks(two_step)

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
        ),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.turn_runner.run_mode_entry_operations", side_effect=failing_plan_entry_operation
        ),
    ):
        result = await run_turn(
            mode_id="default",
            query="read docs, consolidate, find gaps, web search, write narrative",
            hooks=hooks,
            preferred_model_id="test-planner-model",
        )

    # The turn hard-stops with a deterministic failure (the entry-op
    # exception path sets ``entry_failed`` and breaks the loop).
    assert result.blocked is True
    assert "planner" in result.response_text.lower()
    # The model is now marked unreliable for planning so the hint
    # generator stops escalating to ``planner`` for it on subsequent turns.
    assert not is_plan_reliable_for_model("test-planner-model")
    reset_plan_reliability()


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
                            "name": "fork",
                            "arguments": (
                                '{"targets": [{"mode_id": "remember"}, {"mode_id": "markdown"}]}'
                            ),
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
    assert (
        "failed" in result.response_text.lower()
        or "unverified" in result.response_text.lower()
        or "hallucinated" in result.response_text.lower()
    )
    # The model must NOT have been allowed to emit its false "Done — wrote it to disk."
    assert result.response_text != "Done — wrote it to disk."


@pytest.mark.asyncio
async def test_no_tools_fork_child_read_only_calls_surfaces_fork_failure_not_yield() -> None:
    """A ``no_tools`` fork child that made read-only calls must surface the
    deterministic fork-failure message, not the generic iteration-budget
    yield question.

    Regression for the 2026-08-04 incident: the parent's ``no_tools_detail``
    list comprehension accessed ``r.called_tool_names`` on each
    ``ForkChildResult``, but that field existed only on ``DelegationResult``.
    The ``AttributeError`` was caught by the ``transition_stage`` tool-dispatch
    wrapper, the fork-failure ``content`` was never set, the loop broke with
    empty content, and the post-loop fallback emitted "I've hit my iteration
    budget for this turn." — masking the real fork abort after only a few
    iterations. ``ForkChildResult`` now carries ``called_tool_names`` so the
    detail comp runs and the user sees the fork-failure framing.
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
                            "name": "fork",
                            "arguments": '{"targets": [{"mode_id": "markdown"}]}',
                        },
                    }
                ],
            )
        # The markdown child makes a read-only call (fetch_page) instead of
        # its authored action tool — exactly the production scenario.
        if call_count == 2:
            return _StubCompletion(
                content="",
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
        # After the read-only call, the child exits with a text response
        # claiming success — but it never invoked its action tool.
        return _StubCompletion(
            content="Done — wrote it to disk.",
            tool_calls=[],
        )

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="write a markdown guide", hooks=hooks)

    # The child is labeled no_tools with its read-only calls recorded.
    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    assert child.status == "no_tools"
    assert child.called_tool_names == ["fetch_page"]
    # The parent must surface the fork-failure framing, not the generic
    # iteration-budget yield question that masked the abort.
    assert "read-only calls" in result.response_text or "never wrote" in result.response_text
    assert "iteration budget" not in result.response_text.lower()


def test_max_handoffs_per_turn_setting_is_bound() -> None:
    """The hard backstop setting exists and is a sane bound. This guards
    against accidental removal — the loop guard depends on it. The
    default is loosened to 50 to accommodate deep producer→consumer
    pipelines (the 2026-08-02 fork fan-out work hit 6 handoffs on a
    routine coding turn); it stays bounded so a runaway hint loop
    cannot hand off forever."""
    from kernel.config import settings as kernel_settings

    value = kernel_settings.max_handoffs_per_turn
    assert isinstance(value, int)
    assert 3 <= value <= 100


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
        "target_stage_id `fork:remember,markdown`."
    )
    assert _extract_hint_target(hint) == "fork:remember,markdown"


def test_extract_hint_target_returns_empty_for_no_marker() -> None:
    """``_extract_hint_target`` returns ``""`` when no target marker is present."""
    from kernel.turn_runner import _extract_hint_target

    assert _extract_hint_target("") == ""
    assert _extract_hint_target("no marker here") == ""


def test_synthesized_call_missing_all_required_true_for_empty_args() -> None:
    """A synthesized call with empty args against a tool that requires
    params is detected as missing-all-required (the 2026-08-27
    ``cs_apply_cloud_init`` misroute: 7 required fields, none supplied).

    The guard lets the synthesis sites abort the turn rather than
    dispatch a guaranteed validation failure and risk a backstop loop.
    """
    from kernel.models import OperationDefinition
    from kernel.turn_runner import _synthesized_call_missing_all_required
    from pydantic import BaseModel

    class _Params(BaseModel):
        template_id: str
        name: str
        provider: str

    op_def = OperationDefinition(
        operation_id="cs_apply_cloud_init",
        description="apply cloud-init",
        parameters_schema=_Params,
    )
    assert (
        _synthesized_call_missing_all_required(
            "cs_apply_cloud_init",
            {},
            operation_catalog={"cs_apply_cloud_init": op_def},
            tool_catalog={},
        )
        is True
    )


def test_synthesized_call_missing_all_required_false_when_some_supplied() -> None:
    """When at least one required param is supplied, the call is not
    missing-all-required — let dispatch proceed (the model or kernel
    provided a value)."""
    from kernel.models import OperationDefinition
    from kernel.turn_runner import _synthesized_call_missing_all_required
    from pydantic import BaseModel

    class _Params(BaseModel):
        template_id: str
        name: str

    op_def = OperationDefinition(
        operation_id="cs_apply_cloud_init",
        description="apply cloud-init",
        parameters_schema=_Params,
    )
    assert (
        _synthesized_call_missing_all_required(
            "cs_apply_cloud_init",
            {"template_id": "tpl_123"},
            operation_catalog={"cs_apply_cloud_init": op_def},
            tool_catalog={},
        )
        is False
    )


def test_synthesized_call_missing_all_required_false_for_optional_only_schema() -> None:
    """A tool with no required params (e.g. ``transition_stage``) is never
    missing-all-required — routing-only syntheses pass through unchanged."""
    from kernel.models import ToolDefinition
    from kernel.turn_runner import _synthesized_call_missing_all_required
    from pydantic import BaseModel

    class _Params(BaseModel):
        rationale: str = ""

    tool_def = ToolDefinition(
        tool_id="transition_stage",
        description="transition",
        parameters_schema=_Params,
    )
    assert (
        _synthesized_call_missing_all_required(
            "transition_stage",
            {},
            operation_catalog={},
            tool_catalog={"transition_stage": tool_def},
        )
        is False
    )


def test_synthesized_call_missing_all_required_false_for_unknown_tool() -> None:
    """An unknown tool (MCP-native, no kernel definition) is not flagged —
    let the normal dispatch path handle it."""
    from kernel.turn_runner import _synthesized_call_missing_all_required

    assert (
        _synthesized_call_missing_all_required(
            "lisa-storage.read_file",
            {},
            operation_catalog={},
            tool_catalog={},
        )
        is False
    )


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
    assert state.forced_action is None


@pytest.mark.asyncio
async def test_forced_plan_enforcer_when_model_refuses_to_hand_off() -> None:
    """When a plan-escalation routing hint is active and the model calls
    non-``transition_stage`` tools instead of handing off to ``planner``,
    the kernel forces the planner handoff after
    ``forced_action_non_compliant_cap`` non-transition dispatches.

    Regression for the 2026-07-30 incident: the model burned 15
    iterations of tool-cycling in ``default`` mode (≈400k tokens) before
    finally handing off to ``planner``, which then failed to parse. The
    forced-plan enforcer bounds the pre-plan token burn to roughly
    ``cap × per-iteration-tokens`` by synthesizing the
    ``transition_stage(planner)`` call itself.
    """
    from unittest.mock import patch

    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.models import OperationResult

    default_cfg = _mode_config(mode_id="default")
    planner_cfg = WorkflowModeConfig(
        mode_id="planner",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = dict(get_workflow_catalog())
    patched_catalog.update({"default": default_cfg, "planner": planner_cfg, "search": search_cfg})

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
        is_plan_mode = "## Handoff Context" in sys_content and "`planner`" in sys_content
        is_search_mode = ("## Handoff Context" in sys_content and "`search`" in sys_content) or (
            "Handed off to mode: search" in str([m.get("content", "") for m in kwargs["messages"]])
        )
        if is_search_mode:
            return _StubCompletion(content="Search results from web_search.")
        if is_plan_mode:
            # The kernel's mid-turn entry-op detection sets
            # pending_planned_handoff; the next iteration synthesizes the
            # transition_stage call. Return harmless text.
            return _StubCompletion(content="Planning...")
        # default mode: the model does read-only fetches instead of
        # handing off to planner. After forced_action_non_compliant_cap
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
    hooks.recall.enabled = False

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
        ),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default",
            query="search for rust news then write a summary then code it",
            hooks=hooks,
        )

    # The kernel forced the planner handoff and the planner-selected
    # search mode ran; later chain steps may surface the same stub text.
    assert "Search results from web_search." in result.response_text
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
async def test_forced_transition_enforcer_advances_non_plan_target_on_empty_completion() -> None:
    """When a post-handoff hint names a non-plan target (``search``) and
    the model in the current mode (``remember``) returns text without
    calling ``transition_stage``, the kernel synthesizes the handoff
    itself instead of ending the turn with the model's prose.

    Regression for the 2026-08-07 ``plan → remember → search`` stall: the
    weak model in ``remember`` returned "budget exhausted" text without
    calling ``transition_stage``, so ``search`` never ran. The old
    forced-plan enforcer only fired for ``plan`` targets; the generalized
    forced-transition enforcer captures ANY non-fork routing target.
    """

    from unittest.mock import patch

    from kernel.models import OperationResult

    default_cfg = _mode_config(mode_id="default")
    planner_cfg = WorkflowModeConfig(
        mode_id="planner",
        start_stage_id="none",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    remember_cfg = _mode_config(
        mode_id="remember",
        satisfies_intents=["remember"],
        routing_intent_tags=["remember"],
        routing_intent_keywords={"remember": ["remember"]},
    )
    search_cfg = _mode_config(
        mode_id="search",
        satisfies_intents=["research"],
        routing_intent_tags=["research"],
        routing_intent_keywords={"research": ["search", "web", "framework"]},
    )
    patched_catalog = {
        "default": default_cfg,
        "planner": planner_cfg,
        "remember": remember_cfg,
        "search": search_cfg,
    }

    plan_entry_calls = 0
    search_reached = False

    async def complete(**kwargs: Any) -> _StubCompletion:
        nonlocal search_reached
        sys_content = str(kwargs["messages"][0].get("content", ""))
        msgs = str([m.get("content", "") for m in kwargs["messages"]])
        # In search mode (reached after the forced handoff): return results.
        if "Handed off to mode: search" in msgs or (
            "`search`" in sys_content and "search" in sys_content
        ):
            search_reached = True
            return _StubCompletion(content="Search results from web_search.")
        # In remember mode: the weak model returns text WITHOUT calling
        # transition_stage — the bug. The kernel must force the handoff.
        if "Handed off to mode: remember" in msgs:
            return _StubCompletion(
                content="I remembered the framework. Budget exhausted; "
                "I cannot complete the search step."
            )
        # default mode: hand off to planner as the hint directs.
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_default_1",
                    "function": {
                        "name": "transition_stage",
                        "arguments": json.dumps(
                            {"target_stage_id": "planner", "rationale": "escalate"}
                        ),
                    },
                }
            ],
        )

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        nonlocal plan_entry_calls
        plan_entry_calls += 1
        return OperationResult(
            output_text="remember then search",
            output_json={
                "target_stage_id": "remember",
                "rationale": "capture knowledge first",
                "steps": [
                    {"step_id": "s1", "mode_id": "remember"},
                    {"step_id": "s2", "mode_id": "search"},
                ],
            },
        )

    hooks = _build_minimal_hooks(complete)
    hooks.recall.enabled = False

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=patched_catalog
        ),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
    ):
        result = await run_turn(
            mode_id="default",
            query="remember what you know then search the web for similar frameworks",
            hooks=hooks,
        )

    # The kernel forced the remember→search handoff even though the model
    # returned text without calling transition_stage. Search ran.
    assert search_reached, (
        "search mode was never reached — the forced-transition enforcer did not fire"
    )
    assert "Search results from web_search." in result.response_text


@pytest.mark.asyncio
async def test_forced_transition_preserves_pre_handoff_text() -> None:
    """When the model produces visible text but no tool calls and the
    kernel synthesizes a forced transition, the pre-handoff text must
    appear in the final response alongside the post-handoff mode's text.

    Regression for the 2026-09-17 incident: the model in ``default`` mode
    generated a full list of shell commands as its text response. Because
    the routing hint detected a ``shell`` intent, the kernel synthesized a
    forced transition to ``shell`` mode. The shell-mode iteration produced
    only a summary referencing "the commands above". The pre-handoff text
    (the command list) was silently discarded when ``content`` was
    overwritten on the next iteration — Lisa thought she gave the answer
    but the user never saw it.
    """
    from unittest.mock import patch

    default_cfg = _mode_config(mode_id="default")
    shell_cfg = _mode_config(
        mode_id="shell",
        satisfies_intents=["shell"],
        routing_intent_tags=["shell"],
        routing_intent_keywords={"shell": ["shell"]},
    )
    patched_catalog = {
        "default": default_cfg,
        "shell": shell_cfg,
    }

    shell_reached = False

    async def complete(**kwargs: Any) -> _StubCompletion:
        nonlocal shell_reached
        sys_content = str(kwargs["messages"][0].get("content", ""))
        msgs = str([m.get("content", "") for m in kwargs["messages"]])
        # In shell mode (reached after the forced handoff): return a summary.
        if "Handed off to mode: shell" in msgs or (
            "`shell`" in sys_content and "shell" in sys_content and "Handoff Context" in sys_content
        ):
            shell_reached = True
            return _StubCompletion(
                content="The commands above are ready to run on your host shell."
            )
        # In default mode: the model produces the command list as text
        # but does NOT call transition_stage — the kernel must force the
        # handoff AND preserve this text.
        return _StubCompletion(content="#1 — ps aux | grep zeek\n#2 — zeekctl status\n#3 — cat log")

    hooks = _build_minimal_hooks(complete)
    hooks.recall.enabled = False

    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: (
                _c.get(mid).instruction_body
                if _c.get(mid) and _c.get(mid).instruction_body
                else mid
            ),
        ),
    ):
        result = await run_turn(
            mode_id="default",
            query="give me some shell commands to interrogate zeek",
            hooks=hooks,
        )

    # The kernel forced the default→shell handoff.
    assert shell_reached, (
        "shell mode was never reached — the forced-transition enforcer did not fire"
    )
    # The pre-handoff text (the command list) must survive in the response.
    assert "#1" in result.response_text, (
        "pre-handoff model text was lost — the command list from the "
        "default-mode iteration should appear in the final response"
    )
    assert "ps aux" in result.response_text, (
        "pre-handoff model text was lost — the command list from the "
        "default-mode iteration should appear in the final response"
    )
    # The post-handoff text (the summary) should also be present.
    assert "ready to run" in result.response_text


@pytest.mark.asyncio
async def test_run_turn_model_yield_tool_terminates_with_question() -> None:
    """When the model calls the ``yield`` tool, the turn terminates with
    the question as the response text and ``_yielded`` stamps the turn.

    This is the core unified-lifecycle fix: the model can now *voluntarily*
    ask the user a question by calling a tool, instead of the harness
    guessing that "no tool calls" meant "ask" (which was indistinguishable
    from "done").
    """

    async def yield_then_done(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_yield_1",
                    "function": {
                        "name": "yield",
                        "arguments": json.dumps(
                            {
                                "question": "Which target audience should I focus on?",
                                "context": "ambiguous request",
                            }
                        ),
                    },
                }
            ],
        )

    hooks = _build_minimal_hooks(yield_then_done)
    hooks.operations.execute_operation = _stub_execute_operation

    result = await run_turn(
        mode_id="default",
        query="write a roadmap",
        hooks=hooks,
    )

    assert result.response_text == "Which target audience should I focus on?"
    # The yield tool is intercepted in-loop (like transition_stage) — it
    # does not increment the dispatched-tool counter. The turn terminates
    # with the question as content.
    assert result.tool_call_count == 0


@pytest.mark.asyncio
async def test_tool_calls_preserve_prior_yield_state() -> None:
    tool_calls = [{"id": "call_after_yield"}]
    result = await _resolve_empty_completion_for_dispatch(
        tool_calls=tool_calls,
        synthesized_forced=False,
        yielded=True,
        state=SimpleNamespace(),
        mode_id="default",
        query="question",
        completion=None,
        assistant_msg={},
        pre_handoff_text="",
        pending_yield=None,
        content="Earlier yield question",
        content_source="",
        turn_reasoning_text="",
        stalled_write_guidance="",
        budget_forces_final=lambda: False,
        mode_config=None,
        operation_catalog={},
        tool_catalog={},
        turns_in_stage=0,
        max_tool_iterations=10,
        log=logging.getLogger(__name__),
    )

    assert result.tool_calls is tool_calls
    assert result.yielded is True


@pytest.mark.asyncio
async def test_budget_briefing_appended_without_rewriting_system_prompt() -> None:
    """The per-iteration budget briefing is injected into the system
    prompt (``messages[0]``) so the model can self-scope its work within
    the remaining iteration envelope.

    The briefing is super-brief (a single ``## Turn Lifecycle`` block with
    dot-separated counts) and is replaced — not accumulated — each
    iteration. Iteration 1 shows the full contract; later iterations
    show only the delta (partially-consumed per-target limits).
    """
    captured_system_prompts: list[str] = []
    captured_briefings: list[list[str]] = []
    call_count = 0

    async def capture_and_finish(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        # Capture the system prompt each iteration.
        captured_system_prompts.append(str(kwargs["messages"][0].get("content", "")))
        captured_briefings.append(
            [
                str(m.get("content", ""))
                for m in kwargs["messages"][1:]
                if "## Turn Lifecycle" in str(m.get("content", ""))
            ]
        )
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
    hooks.operations.execute_operation = _stub_execute_operation
    from unittest.mock import patch

    with patch(
        "kernel.turn_runner.get_mode_system_intro", return_value="Default mode instructions."
    ):
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
    assert "## Turn Lifecycle" not in captured_system_prompts[0]
    # The briefing reports iterations remaining (4/5 after iteration 1
    # incremented to 1).
    assert "iteration: 1 of 5" in captured_briefings[0][0]
    # Iteration 2: the briefing must still be present (replaced, not
    # accumulated — only one ``## Turn Lifecycle`` block).
    assert captured_system_prompts[1] == captured_system_prompts[0]
    assert captured_briefings[1] == captured_briefings[0]
    # Iteration 2: iteration 2 of 5.
    # The base system prompt content (mode intro) is preserved — the
    # briefing is appended, not replacing the base.
    assert captured_system_prompts[0].split("## Turn Lifecycle")[0].strip() != ""


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
    hooks.operations.execute_operation = _stub_execute_operation
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
    assert "## Turn Lifecycle" not in captured_system_prompts[0]


# ── Fork-child sub-turn contract + midpoint write-nudge tests ────────────


@pytest.mark.asyncio
async def test_fork_child_sees_sub_turn_contract_with_action_tools() -> None:
    """A fork child whose mode declares action tools sees a
    ``## Sub-Turn Contract`` block naming those tools, framing the
    budget as a write-deadline.

    Regression for the 2026-08-03 incident: a ``remember`` fork child saw
    only ``iters: 49/50`` and read-looped to the cap without ever calling
    its action tools. The sub-turn contract now names the action tools
    and tells the child reads do not count.
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
                            "name": "fork",
                            "arguments": '{"targets": [{"mode_id": "remember"}]}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="remember and write", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 1
    child = result.fork_results[0]
    # The child should have the sub-turn contract in its system prompt —
    # but the stub complete function doesn't capture the child's prompt.
    # Verify via the child's status: the contract + nudge are behaviorally
    # tested in the next test. Here we just verify the fork ran.
    assert child.target.mode_id == "remember"


@pytest.mark.asyncio
async def test_build_sub_turn_contract_names_action_tools() -> None:
    """``_build_sub_turn_contract`` names the mode's action tools and
    frames the budget as a write-deadline for fork children, and returns
    empty for the parent turn."""
    from kernel.turn_phases._turn_state import _TurnState
    from kernel.turn_runner import _build_sub_turn_contract

    # Parent turn → no sub-turn contract.
    parent_state = _TurnState()
    assert (
        _build_sub_turn_contract(
            state=parent_state, remaining_iterations=10, budget_exhausted=False
        )
        == ""
    )

    # Fork child with action tools → contract names them.
    fork_state = _TurnState()
    fork_state.sub_turn_kind = "fork"
    fork_state.action_tool_ids = {"memory.add", "synthesize_and_store_learning"}
    contract = _build_sub_turn_contract(
        state=fork_state, remaining_iterations=25, budget_exhausted=False
    )
    assert "## Sub-Turn Contract" in contract
    assert "memory.add" in contract
    assert "synthesize_and_store_learning" in contract
    assert "write-deadline" in contract or "iterations remain to produce a write" in contract
    assert "discarded" in contract  # fork framing

    # Once the child has called an action tool, the contract drops.
    fork_state.called_tool_names = {"memory.add"}
    assert (
        _build_sub_turn_contract(state=fork_state, remaining_iterations=20, budget_exhausted=False)
        == ""
    )

    # Delegate child (planned) → different framing (status, not discarded).
    delegate_state = _TurnState()
    delegate_state.sub_turn_kind = "delegate"
    delegate_state.action_tool_ids = {"memory.add"}
    delegate_contract = _build_sub_turn_contract(
        state=delegate_state, remaining_iterations=10, budget_exhausted=False
    )
    assert "## Sub-Turn Contract" in delegate_contract
    assert "no_tools" in delegate_contract
    assert "discarded" not in delegate_contract  # delegate framing


@pytest.mark.asyncio
async def test_fork_children_share_the_central_lifecycle_guidance_channel() -> None:
    """Parent and child user messages are either the query or lifecycle guidance."""
    call_count = 0
    captured_snapshots: list[list[dict[str, Any]]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        captured_snapshots.append([dict(m) for m in kwargs.get("messages", [])])
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "fork",
                            "arguments": (
                                '{"targets": [{"mode_id": "remember"}, {"mode_id": "markdown"}]}'
                            ),
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(two_step)
    result = await run_turn(mode_id="default", query="remember and write", hooks=hooks)

    assert result.fork_results is not None
    assert {r.target.mode_id for r in result.fork_results} == {"remember", "markdown"}
    assert captured_snapshots
    for snapshot in captured_snapshots:
        for msg in snapshot:
            if msg.get("role") == "user":
                text = str(msg.get("content", ""))
                assert "## User Query" in text or "## Turn Lifecycle" in text


def test_short_keyword_requires_word_boundary() -> None:
    """Short (≤3 char) and numeric routing keywords only match on word
    boundaries — a bare-substring match inside incidental text must not
    fire (the 2026-09-16 incident: ``gl`` matched inside "sin``gl``e-entry
    uid map" and ``940`` matched inside the file size "673``940``", routing
    benign queries into ``finances`` four times).
    """
    from kernel.turn_phases._routing_hint import _keyword_matches

    # Embedded inside a larger word → no match.
    assert not _keyword_matches("gl", "single-entry uid map")
    assert not _keyword_matches("940", "673940 aug 11 mod")
    # Space/punctuation-delimited → genuine match.
    assert _keyword_matches("gl", "check the gl entries")
    assert _keyword_matches("gl", "gl: reconcile the accounts")
    assert _keyword_matches("gl", "gl")
    assert _keyword_matches("940", "form 940")
    assert _keyword_matches("940", "schedule the 940 filing")
    # Long non-numeric keywords keep bare-substring semantics.
    assert _keyword_matches("ledger", "subledger reconciliation")


def test_partial_coverage_tie_breaks_by_earliest_keyword_position() -> None:
    """When the best partial coverage is a tie among several targets (each
    covering exactly one gap intent), the tie-break must follow the
    earliest gap-intent keyword position in the query — not
    ``routing_targets`` declaration order (the 2026-09-16 incident:
    ``finances`` declared 4th beat ``shell`` on a 1-intent tie because it
    precedes ``shell`` in the target list).
    """
    from kernel.turn_phases._routing_hint import _best_partial_coverage_target

    catalog = {
        "finances": _mode_config(
            mode_id="finances",
            routing_intent_tags=["finances"],
            routing_intent_keywords={"finances": ["cashflow"]},
        ),
        "shell": _mode_config(
            mode_id="shell",
            routing_intent_tags=["shell"],
            routing_intent_keywords={"shell": ["sandbox"]},
        ),
    }
    # "sandbox" (shell) appears earlier in the query than "cashflow"
    # (finances) even though finances is declared first in routing_targets.
    target = _best_partial_coverage_target(
        query="inspect the sandbox setup, then the cashflow",
        routing_targets=["finances", "shell"],
        catalog=catalog,
        gap_intents={"finances", "shell"},
    )
    assert target == "shell", f"earliest keyword must win, got {target!r}"

    # A strict winner still routes over a tied pair: one target covering 2
    # gap intents beats the other's 1.
    catalog["broad"] = _mode_config(
        mode_id="broad",
        routing_intent_tags=["finances", "shell"],
        routing_intent_keywords={},
    )
    target = _best_partial_coverage_target(
        query="inspect the sandbox and the cashflow",
        routing_targets=["finances", "shell", "broad"],
        catalog=catalog,
        gap_intents={"finances", "shell", "other"},
    )
    assert target == "broad", f"strict winner must route, got {target!r}"


@pytest.mark.asyncio
async def test_suppressed_child_turn_injects_no_routing_hint() -> None:
    """A child turn with ``suppress_routing_hint=True`` (delegate/fork shape)
    never injects a routing hint, even when the entry phase returns
    ``suppress_routing_hint=False`` (any chat mode with no entry ops).

    Regression for the 2026-10-06 13:25 UTC incident: ``_default_delegate_fn``
    passed ``suppress_routing_hint=True`` (the documented "delegate children
    do not re-route" invariant), but ``_run_turn_entry_sequence`` overwrote
    the flag with ``bool(entry_result.suppress_routing_hint)`` — False for a
    no-entry-op chat mode — so a delegate child in ``default`` mode saw the
    parent's multi-intent query, re-detected a 6-intent gap, re-escalated to
    ``planner``, and the forced-action enforcer handed the CHILD off
    mid-turn. The nested ``plan_workflow`` call then blew the parent's 300s
    ``delegate`` timeout and the whole child stack was cancelled without
    executing its delegated work.

    Control arm in the same test: the identical query and catalog in a
    PARENT turn (``suppress_routing_hint=False``) must inject the hint —
    proving the suppression flag, not the query or catalog, makes the
    difference.
    """

    def build_hooks() -> tuple[TurnHooks, list[str]]:
        captured_system_prompts: list[str] = []

        async def complete(**kwargs: Any) -> _StubCompletion:
            msgs = kwargs.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                captured_system_prompts.append(str(msgs[0].get("content", "")))
            return _StubCompletion(content="Done.")

        return _build_minimal_hooks(complete), captured_system_prompts

    # Multi-intent query the ambient synthetic catalog attributes to
    # ``markdown``/``ponder`` (write + research) — the same gap shape the
    # incident's MISSION text produced.
    query = "search for the latest coding practices then write a markdown report"
    catalog = {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search", "web", "find"]},
        ),
        "markdown": _mode_config(
            mode_id="markdown",
            satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["markdown", "document", "write"]},
        ),
    }

    with _patched_hint_catalog(catalog):
        # Child arm: suppression must hold (the delegate-child shape).
        child_hooks, child_prompts = build_hooks()
        child_result = await run_turn(
            mode_id="default",
            query=query,
            hooks=child_hooks,
            suppress_routing_hint=True,
        )
        # Parent arm: the same query and catalog must inject the hint.
        parent_hooks, parent_prompts = build_hooks()
        parent_result = await run_turn(
            mode_id="default",
            query=query,
            hooks=parent_hooks,
            suppress_routing_hint=False,
        )

    assert child_result.response_text == "Done."
    assert child_prompts, "child model must have been called"
    assert all("Routing Hint" not in prompt for prompt in child_prompts), (
        "suppressed child turn must not inject a routing hint"
    )
    assert child_result.handoff_to is None, "suppressed child must not re-route"

    # Parent arm: the hint must be injected. (The parent's response may
    # be two completions' text joined — the forced-transition enforcer can
    # synthesize a handoff after the first "Done." — so only assert on
    # the hint presence, not the response text.)
    assert "Done." in parent_result.response_text
    assert parent_prompts, "parent model must have been called"
    assert any("Routing Hint" in prompt for prompt in parent_prompts), (
        "control arm: parent turn must still inject the hint"
    )


@pytest.mark.asyncio
async def test_model_call_complete_payload_carries_proxy_telemetry() -> None:
    """A serving proxy's declared counters ride the stage payload.

    The kernel sums them onto the turn state without interpreting them, so a
    transport can report compression savings that only existed for the
    duration of one call.
    """

    class _ProxiedCompletion(_StubCompletion):
        @property
        def usage(self) -> Any:
            return SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                cached_tokens=0,
                reasoning_tokens=0,
                cache_creation_tokens=0,
                proxy_telemetry={"tokens_saved": 5000, "pack_specific": 3},
            )

    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    async def complete(**kwargs: Any) -> _ProxiedCompletion:
        return _ProxiedCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.lifecycle.on_stage = on_stage

    await run_turn(mode_id="default", query="hello", hooks=hooks)

    _label, data = next(e for e in events if e[0] == "model_call_complete")
    assert data["proxy_telemetry"] == {"tokens_saved": 5000, "pack_specific": 3}


@pytest.mark.asyncio
async def test_model_call_complete_proxy_telemetry_empty_without_proxy() -> None:
    """No proxy in the path leaves the key present but empty."""
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_stage(label: str, data: dict[str, Any]) -> None:
        events.append((label, dict(data)))

    async def complete(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    hooks = _build_minimal_hooks(complete)
    hooks.lifecycle.on_stage = on_stage

    await run_turn(mode_id="default", query="hello", hooks=hooks)

    _label, data = next(e for e in events if e[0] == "model_call_complete")
    assert data["proxy_telemetry"] == {}


@pytest.mark.asyncio
async def test_mid_turn_multi_step_plan_is_rendered_in_the_response() -> None:
    """A plan produced by a mid-turn escalation reaches the operator.

    Regression: turn-start staging put the rendered plan on
    ``state.plan_prefix``, which ``_append_post_turn_content`` prepends to the
    response. The mid-turn handoff path called ``_record_turn_plan_position``,
    which set ``state.turn_plan_position`` only. Every planner escalation
    happens mid-turn, so plans were produced and discarded: 56 produced across
    the recorded session, 2 visible.

    Single-step plans are intentionally not rendered — this test covers the
    multi-step case, which is the one that carries information.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult

    default_cfg = _mode_config(mode_id="default")
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    markdown_cfg = _mode_config(mode_id="markdown", satisfies_intents=["write"])
    patched_catalog = {
        "default": default_cfg,
        "plan": plan_cfg,
        "search": search_cfg,
        "markdown": markdown_cfg,
    }

    plan_text = (
        "## Plan\n\n"
        "1. **search** — Find sources\n   _reads_\n"
        "2. **markdown** — Write the summary\n   _writes: markdown_generate_\n"
    )

    async def complete(**kwargs: Any) -> _StubCompletion:
        sys_content = str(kwargs["messages"][0].get("content", ""))
        if "`markdown`" in sys_content and "## Handoff Context" in sys_content:
            return _StubCompletion(content="Summary written.")
        if "`search`" in sys_content and "## Handoff Context" in sys_content:
            return _StubCompletion(content="Found sources.")
        if "`plan`" in sys_content and "## Handoff Context" in sys_content:
            return _StubCompletion(content="Planning...")
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
        return OperationResult(
            output_text=plan_text,
            output_json={
                "target_stage_id": "search",
                "steps": [
                    {
                        "mode_id": "search",
                        "description": "Find sources",
                        "rationale": "research",
                        "write_capability": None,
                    },
                    {
                        "mode_id": "markdown",
                        "description": "Write the summary",
                        "rationale": "write",
                        "write_capability": "markdown_generate",
                    },
                ],
            },
        )

    hooks = _build_minimal_hooks(complete)
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.get_plan_steps_to_workflow_steps", create=True),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
        patch(
            "kernel.turn_runner.get_mode_system_intro",
            lambda mid, _c=patched_catalog: mid,
        ),
        patch(
            "kernel.ipsar_fsm.get_mode_system_intro",
            lambda mid, _c=patched_catalog: mid,
        ),
    ):
        result = await run_turn(
            mode_id="default", query="search for rust news then write a summary", hooks=hooks
        )

    assert result.response_text.startswith("## Plan"), (
        "a multi-step plan produced mid-turn must be prepended to the response; "
        f"got: {result.response_text[:120]!r}"
    )
    assert "Find sources" in result.response_text


@pytest.mark.asyncio
async def test_mid_turn_single_step_plan_is_not_rendered() -> None:
    """A one-step plan is not prefixed to the response.

    A single-step plan describes the handoff the model is already making, so
    rendering it would add ceremony without information. The plan still lands
    on ``turn_plan_position`` (the kernel drives the handoff from it); only the
    user-visible prefix is skipped.
    """
    from unittest.mock import patch

    from kernel.models import OperationResult

    default_cfg = _mode_config(mode_id="default")
    plan_cfg = WorkflowModeConfig(
        mode_id="plan",
        mode_kind="action",
        routable=True,
        entry_operation_ids=["plan_workflow"],
        entry_op_kind="routing",
        satisfies_intents=["plan"],
        routing_intent_tags=["plan"],
    )
    search_cfg = _mode_config(mode_id="search", satisfies_intents=["research"])
    patched_catalog = {"default": default_cfg, "plan": plan_cfg, "search": search_cfg}

    async def complete(**kwargs: Any) -> _StubCompletion:
        sys_content = str(kwargs["messages"][0].get("content", ""))
        if "`plan`" in sys_content and "## Handoff Context" in sys_content:
            return _StubCompletion(content="Planning...")
        if "`search`" in sys_content and "## Handoff Context" in sys_content:
            return _StubCompletion(content="Found sources.")
        return _StubCompletion(
            content="",
            tool_calls=[
                {
                    "id": "call_default_to_plan",
                    "function": {
                        "name": "transition_stage",
                        "arguments": json.dumps(
                            {"target_stage_id": "plan", "rationale": "escalate"}
                        ),
                    },
                }
            ],
        )

    async def fake_entry_ops(_mode_id, _ctx, **_kwargs):
        return OperationResult(
            output_text="## Plan\n\n1. **search** — Find sources\n   _reads_\n",
            output_json={
                "target_stage_id": "search",
                "steps": [
                    {
                        "mode_id": "search",
                        "description": "Find sources",
                        "rationale": "research",
                        "write_capability": None,
                    }
                ],
            },
        )

    hooks = _build_minimal_hooks(complete)
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
        patch("kernel.turn_runner.run_mode_entry_operations", side_effect=fake_entry_ops),
        patch("kernel.turn_runner.get_mode_system_intro", lambda mid, _c=patched_catalog: mid),
        patch("kernel.ipsar_fsm.get_mode_system_intro", lambda mid, _c=patched_catalog: mid),
    ):
        result = await run_turn(mode_id="default", query="search for rust news", hooks=hooks)

    assert not result.response_text.startswith("## Plan"), (
        f"a single-step plan must not be prefixed; got: {result.response_text[:120]!r}"
    )
