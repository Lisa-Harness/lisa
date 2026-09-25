"""Tests for the kernel's default ``_default_fork_fn`` implementation.

Exercises the fork hook's default implementation (the one wired into
the ``OperationHost`` Protocol's ``fork`` method via ``ctx.host`` at
turn start) without spawning the catalog MCP subprocess. The
``sub_turn_kind="fork"`` seam, budget division, and ``ForkChildResult``
status mapping are verified directly by stubbing ``run_turn``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import (
    EffectReceipt,
    ForkResult,
    ForkSpec,
    ForkTarget,
    YieldAction,
)
from kernel.turn_runner import TurnHooks, TurnResult, _default_fork_fn


def _stub_catalog(monkeypatch: pytest.MonkeyPatch, modes: dict[str, Any]) -> None:
    """Patch ``get_workflow_catalog`` (import site in ``_fork``) to a stub."""

    from kernel.turn_phases import _fork as fork_mod

    monkeypatch.setattr(fork_mod, "get_workflow_catalog", lambda: modes)


def _noop_pool() -> Any:
    """Return a minimal stand-in for ``MCPToolPool`` (unused by stubbed run_turn)."""
    return SimpleNamespace()


@pytest.mark.asyncio
async def test_default_fork_fn_empty_targets_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty targets list returns an empty ForkResult without calling run_turn."""
    _stub_catalog(monkeypatch, {})

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        raise AssertionError("run_turn should not be called for empty targets")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)

    spec = ForkSpec(targets=[])
    result = await _default_fork_fn(
        spec,
        hooks=TurnHooks(),
        pool=_noop_pool(),
        resolve_model=lambda mid: None,
        complete=_noop_complete,
        exec_op=_noop_exec_op,
        turn_budget=None,
        execution_policy=None,
        turn_start_mono=0.0,
    )
    assert isinstance(result, ForkResult)
    assert result.children == []
    assert result.succeeded == 0


@pytest.mark.asyncio
async def test_default_fork_fn_ok_status_with_action_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Children that invoked action tools are ``ok``."""
    _stub_catalog(monkeypatch, {})

    spec = ForkSpec(
        targets=[
            ForkTarget(mode_id="remember", query="save this"),
            ForkTarget(mode_id="markdown", query="write a doc"),
        ]
    )
    hooks = TurnHooks()

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        # The child passed sub_turn_kind="fork".
        assert kwargs.get("sub_turn_kind") == "fork"
        assert kwargs.get("suppress_routing_hint") is True
        mid = kwargs.get("mode_id", "")
        if mid == "remember":
            return TurnResult(
                response_text="remembered",
                tool_call_count=1,
                called_tool_names={"synthesize_and_store_learning"},
                effect_receipts=[
                    EffectReceipt(effect="write", capability_id="synthesize_and_store_learning")
                ],
            )
        # markdown: read-only mode (no action tools), zero tool calls → ok
        return TurnResult(
            response_text="wrote doc",
            tool_call_count=0,
            called_tool_names=set(),
            effect_receipts=[],
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr(
        "kernel.operations.registry.mode_completion_effects",
        lambda mode_id: {"synthesize_and_store_learning"} if mode_id == "remember" else set(),
    )

    result = await _default_fork_fn(
        spec,
        hooks=hooks,
        pool=_noop_pool(),
        resolve_model=lambda mid: None,
        complete=_noop_complete,
        exec_op=_noop_exec_op,
        turn_budget=None,
        execution_policy=None,
        turn_start_mono=0.0,
    )
    assert isinstance(result, ForkResult)
    # remember has action tools and invoked one → ok. markdown has no
    # action tools (read-only mode) and made zero calls → ok (read-only
    # exemption). Both succeed.
    assert result.succeeded == 2
    assert len(result.children) == 2
    statuses = {c.target.mode_id: c.status for c in result.children}
    assert statuses["remember"] == "ok"
    assert statuses["markdown"] == "ok"


@pytest.mark.asyncio
async def test_default_fork_fn_passes_sub_turn_kind_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The child turn is dispatched with ``sub_turn_kind="fork"``."""
    _stub_catalog(monkeypatch, {})

    captured: list[str] = []

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.append(kwargs.get("sub_turn_kind", ""))
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr("kernel.operations.registry.mode_completion_effects", lambda mode_id: set())

    spec = ForkSpec(targets=[ForkTarget(mode_id="default", query="test")])
    result = await _default_fork_fn(
        spec,
        hooks=TurnHooks(),
        pool=_noop_pool(),
        resolve_model=lambda mid: None,
        complete=_noop_complete,
        exec_op=_noop_exec_op,
        turn_budget=None,
        execution_policy=None,
        turn_start_mono=0.0,
    )
    assert captured == ["fork"]
    assert result.succeeded == 1  # read-only mode, zero calls → ok


@pytest.mark.asyncio
async def test_default_fork_fn_uses_custom_max_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``spec.max_tool_iterations`` overrides the kernel default."""
    _stub_catalog(monkeypatch, {})

    captured: list[int] = []

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.append(kwargs.get("max_tool_iterations", 0))
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr("kernel.operations.registry.mode_completion_effects", lambda mode_id: set())

    spec = ForkSpec(
        targets=[ForkTarget(mode_id="default", query="test")],
        max_tool_iterations=25,
    )
    await _default_fork_fn(
        spec,
        hooks=TurnHooks(),
        pool=_noop_pool(),
        resolve_model=lambda mid: None,
        complete=_noop_complete,
        exec_op=_noop_exec_op,
        turn_budget=None,
        execution_policy=None,
        turn_start_mono=0.0,
    )
    assert captured == [25]


@pytest.mark.asyncio
async def test_default_fork_fn_interrupted_on_harness_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A force-yielded child (iteration cap) is classified ``interrupted``."""
    _stub_catalog(monkeypatch, {})

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        return TurnResult(
            response_text="partial",
            yielded=YieldAction(
                question="Iteration cap reached before completing.",
                context="iteration cap reached (50/50)",
            ),
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)
    monkeypatch.setattr("kernel.operations.registry.mode_completion_effects", lambda mode_id: set())

    spec = ForkSpec(targets=[ForkTarget(mode_id="coding", query="build it")])
    result = await _default_fork_fn(
        spec,
        hooks=TurnHooks(),
        pool=_noop_pool(),
        resolve_model=lambda mid: None,
        complete=_noop_complete,
        exec_op=_noop_exec_op,
        turn_budget=None,
        execution_policy=None,
        turn_start_mono=0.0,
    )
    assert len(result.children) == 1
    assert result.children[0].status == "interrupted"
    assert result.succeeded == 0


async def _noop_complete(**kwargs: Any) -> Any:
    """Stand-in for the ``complete`` callable (unused by stubbed run_turn)."""
    return SimpleNamespace(content="", tool_calls=[], finish_reason="stop", truncated=False)


async def _noop_exec_op(operation_id: str, ctx: Any) -> Any:
    """Stand-in for the ``exec_op`` callable (unused by stubbed run_turn)."""
    return SimpleNamespace(output_text="", output_json={})
