"""Deterministic tool and scenario drivers.

Two dispatch layers:

- :func:`run_tool` — call one tool's ``execute()`` in-process through the
  real ``ToolExecutionContext`` contract and return its ``ToolResult``.
  This is the deterministic unit/integration path: no model, no host,
  subprocesses only where the tool itself needs them (git, ripgrep).
- :func:`run_scenario` — drive a scripted multi-tool flow from a
  :class:`packstest.case.Scenario` plan, capturing every step's trajectory
  for the report layer. Model behavior is simulated by the *plan* itself
  (an explicit tool-call sequence), never by a live provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernel.models import ToolResult

from packstest.catalog import activate_tool, build_tool_catalog
from packstest.context import ToolContextFactory
from packstest.report import ToolRun, Trajectory

__all__ = ["Scenario", "ToolRun", "run_scenario", "run_tool"]


async def run_tool(
    pack_path_or_module: Any,
    tool_id: str,
    factory: ToolContextFactory,
    parameters: dict[str, Any] | None = None,
) -> ToolRun:
    """Dispatch one tool call in-process.

    ``pack_path_or_module`` is either a pack checkout path (the catalog is
    scanned to resolve the tool) or a prebuilt tool dict with ``module_path``.
    """

    if isinstance(pack_path_or_module, dict):
        # Accept either one prebuilt tool dict (has ``module_path``) or a
        # full catalog mapping keyed by tool id.
        if "module_path" in pack_path_or_module:
            tool_dict = pack_path_or_module
        elif tool_id in pack_path_or_module:
            tool_dict = pack_path_or_module[tool_id]
        else:
            return ToolRun(
                tool_id=tool_id,
                result=ToolResult(tool_id=tool_id, error=f"tool {tool_id!r} not in pack catalog"),
                error="not-in-catalog",
            )
    else:
        catalog = build_tool_catalog(pack_path_or_module)
        if tool_id not in catalog:
            return ToolRun(
                tool_id=tool_id,
                result=ToolResult(tool_id=tool_id, error=f"tool {tool_id!r} not in pack catalog"),
                error="not-in-catalog",
            )
        tool_dict = catalog[tool_id]
    tool_id = str(tool_dict.get("tool_id", tool_id))
    execute = activate_tool(tool_dict)
    ctx = factory.context(parameters or {}, tool_id=tool_id)
    # Mirror the production dispatch contract: an unexpected exception from a
    # tool body surfaces as ToolExecutionError upstream, not as a raw
    # traceback. The harness converts it to a ToolRun error so a crashing
    # tool is visible in the trajectory instead of aborting the scenario.
    try:
        result = await execute(ctx)
    except Exception as exc:  # pylint: disable=broad-except
        result = ToolResult(
            tool_id=tool_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        return ToolRun(tool_id=tool_id, result=result, error="raised")
    return ToolRun(tool_id=tool_id, result=result)


@dataclass(slots=True)
class Scenario:
    """A scripted tool-call plan: the deterministic stand-in for the model."""

    name: str
    pack_path: Any
    steps: list[dict[str, Any]] = field(default_factory=list)
    shared: dict[str, Any] = field(default_factory=dict)


async def run_scenario(scenario: Scenario, factory: ToolContextFactory) -> Trajectory:
    """Execute a scenario's tool sequence and capture a trajectory.

    Each step is ``{"tool_id": ..., "parameters": {...}}``. Parameters may
    reference ``shared`` keys via ``"$shared.<key>"`` so a later step can use
    an artifact id emitted by an earlier step's ``output_json``.
    """

    trajectory = Trajectory(name=scenario.name)
    for step in scenario.steps:
        tool_id = str(step["tool_id"])
        parameters = _substitute_shared(dict(step.get("parameters", {})), scenario.shared)
        run = await run_tool(scenario.pack_path, tool_id, factory, parameters)
        trajectory.record(run)
        if run.result.output_json:
            _absorb_shared(scenario.shared, run.result.output_json)
    return trajectory


def _substitute_shared(parameters: dict[str, Any], shared: dict[str, Any]) -> dict[str, Any]:
    """Replace ``"$shared.key"`` placeholders with values from ``shared``."""

    for key, value in list(parameters.items()):
        if isinstance(value, str) and value.startswith("$shared."):
            ref = value.removeprefix("$shared.")
            parameters[key] = shared.get(ref, value)
    return parameters


def _absorb_shared(shared: dict[str, Any], output_json: dict[str, Any]) -> None:
    """Copy whitelisted result keys into ``shared`` for later steps."""

    for key in ("artifact_id", "branch", "commit_sha", "pushed_sha", "worktree_path"):
        if output_json.get(key):
            shared[key] = output_json[key]
