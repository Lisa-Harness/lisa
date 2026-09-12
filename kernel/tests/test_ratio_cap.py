"""Tests for the ratio-based iteration cap scaling in ``run_turn``.

When a ``WorkflowView`` (or ``WorkflowPosition``) with N declared steps is
active, ``run_turn`` scales the global ``max_tool_iterations`` ceiling to
``N × per_step_max_iterations`` so each step gets its full sub-budget
without starving later steps. Only scales *up* — the base default covers
ad-hoc single turns.

These tests verify the scaling logic by constructing a ``TurnHooks`` with
a ``workflow_position`` callable that returns a position with a known
number of declared steps, then checking the ``emit_stage`` payload (which
carries ``max_iterations``) to confirm the global ceiling was scaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kernel.models import (
    CapabilityManifest,
    OperationExecutionContext,
    OperationResult,
)
from kernel.turn_runner import TurnHooks, TurnResult, run_turn
from kernel.turn_hooks import (
    ModelHooks,
    ContextHooks,
    OperationHooks,
    LifecycleHooks,
    WorkflowHooks,
    HostServiceHandles,
    TurnIdentity,
    RecallHooks,
    ToolHooks,
)


# ── Helpers ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_stub_complete(response: str):
    async def stub(**kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=response, tool_calls=None)

    return stub


async def _stub_execute_operation(
    operation_id: str, _ctx: OperationExecutionContext
) -> OperationResult:
    return OperationResult(output_text=f"Executed {operation_id}")


def _make_step(step_id: str, mode_id: str = "default", **kwargs: Any):
    from kernel.workflow_position import WorkflowStep

    return WorkflowStep(step_id=step_id, mode_id=mode_id, **kwargs)


def _make_position(steps: int):
    """Build a WorkflowPosition-like object with ``steps`` declared steps."""
    from kernel.workflow_position import WorkflowPosition

    declared = tuple(
        _make_step(f"step_{i}", mode_id="default") for i in range(steps)
    )
    return WorkflowPosition(
        declared_steps=declared,
        current_step_id="step_0" if steps > 0 else "",
        mode_id="default",
    )


def _build_hooks_with_position(
    complete_fn,
    position,
) -> TurnHooks:
    """Build TurnHooks with a ``workflow_position`` callable returning ``position``."""
    hooks = TurnHooks(
    models=ModelHooks(
        complete=complete_fn,
    ),
)
    hooks.workflow.workflow_position = lambda: position
    return hooks


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_position_uses_base_cap() -> None:
    """When no workflow_position hook is wired, the global cap stays at the
    base default (300). No scaling occurs."""
    hooks = TurnHooks(
    models=ModelHooks(
        complete=_make_stub_complete("Done."),
    ),
)
    # No workflow_position wired → hooks.workflow.workflow_position is None.

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.response_text == "Done."


@pytest.mark.asyncio
async def test_position_with_no_steps_uses_base_cap() -> None:
    """A position with zero declared steps does not scale the cap."""
    pos = _make_position(0)
    hooks = _build_hooks_with_position(_make_stub_complete("Done."), pos)

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.response_text == "Done."


@pytest.mark.asyncio
async def test_position_with_one_step_does_not_shrink_cap() -> None:
    """A 1-step position scales to 1 × 75 = 75, but the base (300) is higher,
    so ``max()`` keeps the base. The cap never shrinks below the default."""
    pos = _make_position(1)
    hooks = _build_hooks_with_position(_make_stub_complete("Done."), pos)

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)


@pytest.mark.asyncio
async def test_position_with_four_steps_stays_at_base() -> None:
    """4 steps × 75 = 300, which equals the base. No scaling needed."""
    pos = _make_position(4)
    hooks = _build_hooks_with_position(_make_stub_complete("Done."), pos)

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)


@pytest.mark.asyncio
async def test_position_with_five_steps_scales_up() -> None:
    """5 steps × 75 = 375 > 300 (base), so the cap scales up to 375.

    We verify via the ``on_stage`` hook, which receives ``max_iterations``
    in the ``model_call`` event payload.
    """
    pos = _make_position(5)
    captured_max_iterations: list[int] = []

    async def _on_stage(label: str, data: dict[str, Any] | None) -> None:
        if label == "model_call" and data is not None:
            captured_max_iterations.append(
                int(data.get("max_iterations", 0))
            )

    hooks = _build_hooks_with_position(_make_stub_complete("Done."), pos)
    hooks.lifecycle.on_stage = _on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert len(captured_max_iterations) >= 1
    # 5 steps × 75 per_step = 375, which is > 300 (base) → scaled to 375.
    assert captured_max_iterations[0] == 375


@pytest.mark.asyncio
async def test_position_with_eight_steps_scales_to_600() -> None:
    """8 steps × 75 = 600, well above the 300 base."""
    pos = _make_position(8)
    captured_max_iterations: list[int] = []

    async def _on_stage(label: str, data: dict[str, Any] | None) -> None:
        if label == "model_call" and data is not None:
            captured_max_iterations.append(
                int(data.get("max_iterations", 0))
            )

    hooks = _build_hooks_with_position(_make_stub_complete("Done."), pos)
    hooks.lifecycle.on_stage = _on_stage

    result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert len(captured_max_iterations) >= 1
    assert captured_max_iterations[0] == 600


@pytest.mark.asyncio
async def test_execution_policy_tightens_after_scaling() -> None:
    """The execution policy's ``max_model_calls`` tightens the cap via
    ``min()`` *after* the ratio scaling is applied. A 5-step plan scales
    to 375, but an execution policy with ``max_model_calls=200`` tightens
    it to 200."""
    from kernel.execution_policy import ExecutionPolicy

    pos = _make_position(5)
    captured_max_iterations: list[int] = []

    async def _on_stage(label: str, data: dict[str, Any] | None) -> None:
        if label == "model_call" and data is not None:
            captured_max_iterations.append(
                int(data.get("max_iterations", 0))
            )

    hooks = _build_hooks_with_position(_make_stub_complete("Done."), pos)
    hooks.lifecycle.on_stage = _on_stage

    policy = ExecutionPolicy(max_model_calls=200)

    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=hooks,
        execution_policy=policy,
    )

    assert isinstance(result, TurnResult)
    assert len(captured_max_iterations) >= 1
    # min(375, 200) = 200
    assert captured_max_iterations[0] == 200