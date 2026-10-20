"""Tests for the kernel's default ``_default_delegate_fn`` implementation.

Exercises the delegate hook's default implementation (the one wired into
the ``OperationHost`` Protocol's ``delegate`` method via ``ctx.host`` at
turn start) without spawning the catalog MCP subprocess. The
action-tool-floor predicate (spec §6.1) and the ``sub_turn_kind="delegate"``
seam (spec §6.2) are verified directly.
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.config import settings as _kernel_settings
from kernel.delegation import DelegationResult, DelegationSpec
from kernel.models import EffectReceipt, YieldAction
from kernel.turn_runner import TurnHooks, TurnResult, _default_delegate_fn

_ACTION_TOOL = "synthesize_and_store_learning"
_ACTION_MODES = frozenset({"remember"})


@pytest.fixture(autouse=True)
def _stub_resolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the catalog and action-tool resolver at their owning bindings."""

    def _effects(mode_id: str, **_kwargs: Any) -> set[str]:
        return {_ACTION_TOOL} if mode_id in _ACTION_MODES else set()

    monkeypatch.setattr("kernel.ipsar_fsm.get_workflow_catalog", lambda: {})
    monkeypatch.setattr("kernel.turn_runner._defaults.mode_completion_effects", _effects)


def _stub_child(monkeypatch: pytest.MonkeyPatch, outcome: TurnResult | Exception) -> dict[str, Any]:
    """Replace ``run_turn`` with a child that returns or raises ``outcome``."""
    captured: dict[str, Any] = {}

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.update(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    return captured


async def _delegate(spec: DelegationSpec, *, depth: int = 0) -> DelegationResult:
    return await _default_delegate_fn(
        spec, hooks=TurnHooks(), turn_budget=None, execution_policy=None, depth=depth
    )


@pytest.mark.asyncio
async def test_default_delegate_fn_max_depth_guard() -> None:
    """A depth beyond ``delegate_max_depth`` returns ``max_depth`` status."""
    result = await _delegate(
        DelegationSpec(target_mode_id="search", query="test"),
        depth=_kernel_settings.delegate_max_depth + 1,
    )
    assert isinstance(result, DelegationResult)
    assert result.status == "max_depth"
    assert "maximum recursion depth" in result.error_message


@pytest.mark.asyncio
async def test_default_delegate_fn_forwards_sub_turn_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child runs as ``sub_turn_kind="delegate"`` at the parent's depth (spec §6.2)."""
    captured = _stub_child(monkeypatch, TurnResult(response_text="ok"))

    await _delegate(DelegationSpec(target_mode_id="ponder", query="test", max_tool_iterations=25))

    assert captured["sub_turn_kind"] == "delegate"
    assert captured["suppress_routing_hint"] is True
    assert captured["delegate_depth"] == 0  # its own children run at depth+1
    assert captured["max_tool_iterations"] == 25


@pytest.mark.asyncio
async def test_default_delegate_fn_ok_status_with_action_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that invoked an action tool is ``ok`` (spec §6.1 predicate)."""
    _stub_child(
        monkeypatch,
        TurnResult(
            response_text="learned rust",
            tool_call_count=1,
            called_tool_names={_ACTION_TOOL},
            effect_receipts=[EffectReceipt(effect="write", capability_id=_ACTION_TOOL)],
        ),
    )

    result = await _delegate(DelegationSpec(target_mode_id="remember", query="learn rust"))

    assert result.status == "ok"
    assert result.output_text == "learned rust"
    assert result.tool_call_count == 1
    assert _ACTION_TOOL in result.called_tool_names
    assert result.depth == 1


@pytest.mark.parametrize(
    ("mode_id", "outcome", "expected_status", "message_fragments"),
    [
        pytest.param(
            "ponder",
            TurnResult(response_text="rust is fine", tool_call_count=0),
            "ok",
            (),
            id="read-only-mode-ok",
        ),
        pytest.param(
            "remember",
            TurnResult(
                response_text="I learned it (unverified)",
                tool_call_count=3,
                called_tool_names={"recall_memory", "read_file", "list_files"},
            ),
            "no_tools",
            ("action tools",),
            id="read-calls-without-action-tool",
        ),
        pytest.param(
            "remember",
            TurnResult(
                response_text="partial work",
                tool_call_count=2,
                called_tool_names={"read_file"},
                yielded=YieldAction(question="", context="iteration cap reached (50/50)"),
            ),
            "interrupted",
            ("interrupted",),
            id="harness-yield",
        ),
        pytest.param(
            "ponder",
            RuntimeError("child crashed"),
            "error",
            ("runtimeerror", "child crashed"),
            id="child-exception",
        ),
    ],
)
@pytest.mark.asyncio
async def test_default_delegate_fn_classifies_child_outcome(
    monkeypatch: pytest.MonkeyPatch,
    mode_id: str,
    outcome: TurnResult | Exception,
    expected_status: str,
    message_fragments: tuple[str, ...],
) -> None:
    """Read-only calls are not success; harness yields and crashes are distinct statuses."""
    _stub_child(monkeypatch, outcome)

    result = await _delegate(DelegationSpec(target_mode_id=mode_id, query="learn rust"))

    assert result.status == expected_status
    for fragment in message_fragments:
        assert fragment in result.error_message.lower()
    if isinstance(outcome, TurnResult) and expected_status == "ok":
        assert result.output_text == outcome.response_text


@pytest.mark.asyncio
async def test_default_delegate_fn_model_yield_propagates_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model-emitted yield propagates the question to the parent."""
    _stub_child(
        monkeypatch,
        TurnResult(
            response_text="I need to ask the user",
            tool_call_count=0,
            yielded=YieldAction(question="which version of rust?", context="model-emitted yield"),
        ),
    )

    result = await _delegate(DelegationSpec(target_mode_id="remember", query="learn rust"))

    assert result.status == "model_yield"
    assert result.yield_action is not None
    assert result.yield_action.question == "which version of rust?"


# ── Child wall-budget arming (2026-10-19 delegate timeout fix) ────────


@pytest.mark.asyncio
async def test_default_delegate_fn_arms_child_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec carrying the operation timeout arms the child's wall budget.

    The child must run with ``max_wall_ms = timeout - grace`` so
    ``budget_forces_final`` completes it gracefully before the registry's
    hard ``asyncio.wait_for`` cancel discards all child work.
    """
    captured = _stub_child(monkeypatch, TurnResult(response_text="ok"))

    await _delegate(
        DelegationSpec(
            target_mode_id="ponder",
            query="test",
            timeout_seconds=300,
        )
    )

    child_budget = captured["turn_budget"]
    assert child_budget is not None
    expected_ms = (300 - _kernel_settings.delegation_grace_seconds) * 1000
    assert child_budget.max_wall_ms == expected_ms


@pytest.mark.asyncio
async def test_default_delegate_fn_no_timeout_leaves_budget_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec without timeout_seconds (legacy callers) passes the parent budget through."""
    captured = _stub_child(monkeypatch, TurnResult(response_text="ok"))

    await _delegate(DelegationSpec(target_mode_id="ponder", query="test"))

    assert captured["turn_budget"] is None  # _delegate() passes turn_budget=None


@pytest.mark.asyncio
async def test_default_delegate_fn_armed_deadline_tightens_existing_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing parent wall ceiling is min()'d against the armed deadline — never loosened."""
    from kernel.models import TurnBudget

    captured = _stub_child(monkeypatch, TurnResult(response_text="ok"))
    parent_budget = TurnBudget(max_tool_calls=10, max_wall_ms=3_600_000)

    await _default_delegate_fn(
        DelegationSpec(target_mode_id="ponder", query="test", timeout_seconds=300),
        hooks=TurnHooks(),
        turn_budget=parent_budget,
        execution_policy=None,
        depth=0,
    )

    child_budget = captured["turn_budget"]
    assert child_budget.max_wall_ms == (300 - _kernel_settings.delegation_grace_seconds) * 1000
    assert child_budget.max_tool_calls == 10  # non-wall fields pass through
    # The parent's budget object is not mutated.
    assert parent_budget.max_wall_ms == 3_600_000
