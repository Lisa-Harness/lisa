"""Declarative eval cases executed under the pytest ``eval`` marker.

A :class:`Case` bundles a pack, a scripted :class:`packstest.runner.Scenario`,
fixture setup, and assertions on the resulting trajectory and pack state.
Cases are deterministic by construction: the ``eval`` marker's contract
(no model calls, no I/O beyond throwaway tmp dirs) is enforced by design —
the model is replaced by the explicit tool-call plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from packstest.context import ToolContextFactory
from packstest.report import Trajectory
from packstest.runner import Scenario, run_scenario

__all__ = ["CASE_MARKER", "Case", "run_case"]

CASE_MARKER = "eval"


@dataclass(slots=True)
class Case:
    """One declarative pack eval.

    Attributes:
        name: Stable case id (also the trajectory file stem).
        pack_path: Pack checkout under test.
        steps: Tool-call plan (see ``packstest.runner.Scenario``).
        pack_config: Policy dict handed to the context factory.
        expect_tool_ids: Exact expected invocation sequence, optional.
        expect_no_errors: Assert no step returns a result-level error.
        assertions: Optional extra callables receiving the trajectory.
    """

    name: str
    pack_path: Path
    steps: list[dict[str, Any]]
    pack_config: dict[str, Any]
    expect_tool_ids: list[str] | None = None
    expect_no_errors: bool = True
    assertions: list[Any] = field(default_factory=list)


async def run_case(
    case: Case,
    factory: ToolContextFactory,
    *,
    out_dir: Path | None = None,
) -> Trajectory:
    """Run one case and evaluate its declared expectations as pytest asserts."""

    scenario = Scenario(name=case.name, pack_path=case.pack_path, steps=case.steps)
    trajectory = await run_scenario(scenario, factory)
    if case.expect_tool_ids is not None:
        assert trajectory.tool_ids == case.expect_tool_ids, (
            f"tool sequence mismatch: {trajectory.tool_ids} != {case.expect_tool_ids}"
        )
    if case.expect_no_errors:
        assert trajectory.all_errors_absent(), f"unexpected tool error: {trajectory.first_error()}"
    for extra in case.assertions:
        extra(trajectory)
    if out_dir is not None:
        from packstest.report import write_trajectory  # noqa: PLC0415  # lazy: report sink

        write_trajectory(trajectory, out_dir)
    return trajectory


def eval_marker_applied(fn: Any) -> Any:
    """Decorator that applies the framework's ``eval`` marker explicitly."""

    return pytest.mark.eval(fn)
