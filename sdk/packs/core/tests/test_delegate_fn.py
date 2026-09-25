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


def _stub_catalog(monkeypatch: pytest.MonkeyPatch, modes: dict[str, Any]) -> None:
    """Patch ``get_workflow_catalog`` (import site in ``_fork``) to a stub."""

    from kernel.turn_phases import _fork as fork_mod

    monkeypatch.setattr(fork_mod, "get_workflow_catalog", lambda: modes)


@pytest.mark.asyncio
async def test_default_delegate_fn_max_depth_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A depth beyond ``delegate_max_depth`` returns ``max_depth`` status."""
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="search", query="test")
    hooks = TurnHooks()
    result = await _default_delegate_fn(
        spec,
        hooks=hooks,
        turn_budget=None,
        execution_policy=None,
        depth=_kernel_settings.delegate_max_depth + 1,
    )
    assert isinstance(result, DelegationResult)
    assert result.status == "max_depth"
    assert "maximum recursion depth" in result.error_message


@pytest.mark.asyncio
async def test_default_delegate_fn_ok_status_with_action_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that invoked an action tool is ``ok`` (spec §6.1 predicate)."""
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="remember", query="learn rust")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        # The child passed sub_turn_kind="delegate" (spec §6.2).
        assert kwargs.get("sub_turn_kind") == "delegate"
        assert kwargs.get("suppress_routing_hint") is True
        # The child runs at the parent's depth; its own delegate children
        # are depth+1. Here depth=0 (the parent's depth).
        assert kwargs.get("delegate_depth") == 0
        return TurnResult(
            response_text="learned rust",
            tool_call_count=1,
            called_tool_names={"synthesize_and_store_learning"},
            effect_receipts=[
                EffectReceipt(effect="write", capability_id="synthesize_and_store_learning")
            ],
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    # Patch the canonical action-tool resolver so the child's mode
    # "requires tools" and the receipt/called-name matches the action set.
    # ``mode_completion_effects`` is the single source now; the old
    # ``_fork_mode_action_tool_ids`` no longer exists.
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: {"synthesize_and_store_learning"} if mode_id == "remember" else set(),
    )

    result = await _default_delegate_fn(
        spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0
    )
    assert result.status == "ok"
    assert result.output_text == "learned rust"
    assert result.tool_call_count == 1
    assert "synthesize_and_store_learning" in result.called_tool_names
    assert result.depth == 1


@pytest.mark.asyncio
async def test_default_delegate_fn_no_tools_when_only_read_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that made read-only calls but no action tools is ``no_tools``.

    This is the spec §6.1 correction: 17 read-only calls do NOT count as
    ``ok``. The predicate reuses ``_fork_mode_action_tool_ids``.
    """
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="remember", query="learn rust")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        return TurnResult(
            response_text="I learned it (unverified)",
            tool_call_count=3,
            called_tool_names={"recall_memory", "read_file", "list_files"},
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: {"synthesize_and_store_learning"} if mode_id == "remember" else set(),
    )

    result = await _default_delegate_fn(
        spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0
    )
    assert result.status == "no_tools"
    assert "action tools" in result.error_message


@pytest.mark.asyncio
async def test_default_delegate_fn_ok_for_read_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only mode (no action tools) is ``ok`` with zero tool calls."""
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="ponder", query="opinion on rust")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        return TurnResult(response_text="rust is fine", tool_call_count=0)

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: set(),
    )

    result = await _default_delegate_fn(
        spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0
    )
    assert result.status == "ok"
    assert result.output_text == "rust is fine"


@pytest.mark.asyncio
async def test_default_delegate_fn_interrupted_on_harness_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A harness force-yield (cap/budget) is ``interrupted``, not ``no_tools``."""
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="remember", query="learn rust")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        return TurnResult(
            response_text="partial work",
            tool_call_count=2,
            called_tool_names={"read_file"},
            yielded=YieldAction(
                question="",
                context="iteration cap reached (50/50)",
            ),
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: {"synthesize_and_store_learning"} if mode_id == "remember" else set(),
    )

    result = await _default_delegate_fn(
        spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0
    )
    assert result.status == "interrupted"
    assert "interrupted" in result.error_message.lower()


@pytest.mark.asyncio
async def test_default_delegate_fn_model_yield_propagates_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model-emitted yield propagates the question to the parent."""
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="remember", query="learn rust")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        return TurnResult(
            response_text="I need to ask the user",
            tool_call_count=0,
            yielded=YieldAction(
                question="which version of rust?",
                context="model-emitted yield",
            ),
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: set(),
    )

    result = await _default_delegate_fn(
        spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0
    )
    assert result.status == "model_yield"
    assert result.yield_action is not None
    assert result.yield_action.question == "which version of rust?"


@pytest.mark.asyncio
async def test_default_delegate_fn_passes_sub_turn_kind_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child turn receives ``sub_turn_kind="delegate"`` (spec §6.2)."""
    _stub_catalog(monkeypatch, {})

    captured: dict[str, Any] = {}

    spec = DelegationSpec(target_mode_id="ponder", query="test")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: set(),
    )

    await _default_delegate_fn(spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0)
    assert captured["sub_turn_kind"] == "delegate"
    assert captured["suppress_routing_hint"] is True
    assert captured["delegate_depth"] == 0  # child runs at parent's depth; its children are depth+1


@pytest.mark.asyncio
async def test_default_delegate_fn_uses_custom_max_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec with max_tool_iterations overrides the kernel default."""
    _stub_catalog(monkeypatch, {})

    captured: dict[str, Any] = {}

    spec = DelegationSpec(target_mode_id="ponder", query="test", max_tool_iterations=25)
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.update(kwargs)
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: set(),
    )

    await _default_delegate_fn(spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0)
    assert captured["max_tool_iterations"] == 25


@pytest.mark.asyncio
async def test_default_delegate_fn_error_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception in the child turn is caught and returned as ``error``."""
    _stub_catalog(monkeypatch, {})

    spec = DelegationSpec(target_mode_id="ponder", query="test")
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        raise RuntimeError("child crashed")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.turn_runner.mode_completion_effects",
        lambda mode_id: set(),
    )

    result = await _default_delegate_fn(
        spec, hooks=hooks, turn_budget=None, execution_policy=None, depth=0
    )
    assert result.status == "error"
    assert "RuntimeError" in result.error_message
    assert "child crashed" in result.error_message
