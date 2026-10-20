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

_ACTION_TOOL = "synthesize_and_store_learning"


@pytest.fixture(autouse=True)
def _stub_resolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the catalog and action-tool resolver at their owning bindings."""

    def _effects(mode_id: str, **_kwargs: Any) -> set[str]:
        return {_ACTION_TOOL} if mode_id == "remember" else set()

    monkeypatch.setattr("kernel.ipsar_fsm.get_workflow_catalog", lambda: {})
    monkeypatch.setattr("kernel.operations.registry.mode_completion_effects", _effects)


def _noop_pool() -> Any:
    """Return a minimal stand-in for ``MCPToolPool`` (unused by stubbed run_turn)."""
    return SimpleNamespace()


@pytest.mark.asyncio
async def test_default_fork_fn_empty_targets_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty targets list returns an empty ForkResult without calling run_turn."""

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
async def test_default_fork_fn_forwards_sub_turn_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Children run as ``sub_turn_kind="fork"`` with the spec's iteration override."""
    captured: list[dict[str, Any]] = []

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.append(kwargs)
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)

    spec = ForkSpec(
        targets=[ForkTarget(mode_id="default", query="test")],
        max_tool_iterations=25,
    )
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
    assert [kw.get("sub_turn_kind") for kw in captured] == ["fork"]
    assert [kw.get("max_tool_iterations") for kw in captured] == [25]
    assert result.succeeded == 1  # read-only mode, zero calls → ok


@pytest.mark.asyncio
async def test_default_fork_fn_interrupted_on_harness_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A force-yielded child (iteration cap) is classified ``interrupted``."""

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        return TurnResult(
            response_text="partial",
            yielded=YieldAction(
                question="Iteration cap reached before completing.",
                context="iteration cap reached (50/50)",
            ),
        )

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)

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


# ── Fan-out wall-budget arming (2026-10-19 fork timeout fix) ────────────


@pytest.mark.asyncio
async def test_default_fork_fn_arms_child_wall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec carrying the operation timeout arms each child's wall budget.

    Children run in parallel, so the armed deadline (timeout - grace)
    applies to EACH child (tighten-only against the divided parent
    share), not divided across them — each child must force-complete
    before the registry's hard ``asyncio.wait_for`` cancel discards the
    fan-out.
    """
    from kernel.config import settings as _kernel_settings

    captured: list[dict[str, Any]] = []

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.append(kwargs)
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)

    spec = ForkSpec(
        targets=[
            ForkTarget(mode_id="default", query="test one"),
            ForkTarget(mode_id="explainer", query="test two"),
        ],
        timeout_seconds=120,
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

    assert len(captured) == 2
    armed_ms = (120 - _kernel_settings.delegation_grace_seconds) * 1000
    for kw in captured:
        child_budget = kw["turn_budget"]
        assert child_budget is not None
        # Per-child deadline (parallel children), NOT divided by N.
        assert child_budget.max_wall_ms == armed_ms


@pytest.mark.asyncio
async def test_default_fork_fn_armed_deadline_tightens_divided_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent wall share smaller than the deadline passes through unchanged."""
    from kernel.models import TurnBudget

    captured: list[dict[str, Any]] = []

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.append(kwargs)
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)

    # Parent wall = 60s across 2 children → each share 30s, below the
    # armed 90s deadline: the tighter divided share wins.
    spec = ForkSpec(
        targets=[
            ForkTarget(mode_id="default", query="test one"),
            ForkTarget(mode_id="explainer", query="test two"),
        ],
        timeout_seconds=120,
    )
    await _default_fork_fn(
        spec,
        hooks=TurnHooks(),
        pool=_noop_pool(),
        resolve_model=lambda mid: None,
        complete=_noop_complete,
        exec_op=_noop_exec_op,
        turn_budget=TurnBudget(max_tool_calls=10, max_wall_ms=60_000),
        execution_policy=None,
        turn_start_mono=0.0,
    )

    assert len(captured) == 2
    for kw in captured:
        child_budget = kw["turn_budget"]
        assert child_budget is not None
        assert child_budget.max_wall_ms == 30_000  # min(60s/2, 120s-grace)
        assert child_budget.max_tool_calls == 5  # divided field untouched by arming


@pytest.mark.asyncio
async def test_default_fork_fn_no_timeout_leaves_budget_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec without timeout_seconds (legacy callers) passes the parent budget through."""
    captured: list[dict[str, Any]] = []

    async def fake_run_turn(**kwargs: Any) -> TurnResult:
        captured.append(kwargs)
        return TurnResult(response_text="ok")

    monkeypatch.setattr("kernel.turn_runner.run_turn", fake_run_turn)

    spec = ForkSpec(targets=[ForkTarget(mode_id="default", query="test")])
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

    assert captured[0]["turn_budget"] is None


async def _noop_complete(**kwargs: Any) -> Any:
    """Stand-in for the ``complete`` callable (unused by stubbed run_turn)."""
    return SimpleNamespace(content="", tool_calls=[], finish_reason="stop", truncated=False)


async def _noop_exec_op(operation_id: str, ctx: Any) -> Any:
    """Stand-in for the ``exec_op`` callable (unused by stubbed run_turn)."""
    return SimpleNamespace(output_text="", output_json={})
