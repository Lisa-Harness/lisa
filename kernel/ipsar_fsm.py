"""Workflow finite-state-machine loader and runtime for IPSAR mode progression.

This module is a pure loader/router: it has zero hardcoded knowledge of specific
modes or stages.  The workflow catalog is loaded from the ``lisa-content`` MCP
server at startup; the kernel never touches the filesystem directly.

    {mode_name}/
      _mode.md         → YAML front-matter declares mode_id, start_stage_id,
                         mode_kind, description, command_pattern, and operation
                         bindings for the user-facing mode entry point
      {stage_id}.md    → YAML front-matter declares stage config; body text is
                         the instruction prose injected into AI prompts

Adding a new workflow mode requires only a new folder with the appropriate files;
no code changes are needed. The ``lisa-content`` server discovers and parses
mode packs from content roots.

Runtime functions (create_workflow_session, apply_transition, etc.) delegate to
the catalog loaded by ``_ensure_catalog_loaded()`` on first access.

Called by: host apps (session management, stage transitions, context injection)
           kernel.ipsar (parse_route for FSM mode detection)
           packaging tools (manifest summary generation)
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml

from kernel.config import settings
from kernel.intake.connector_discovery import discover_installed_signal_collectors
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    UnifiedState,
    WorkflowModeConfig,
    WorkflowSessionState,
    WorkflowStageConfig,
    WorkflowTransitionRequest,
    WorkshopBudgetConfig,
)
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)

# The lisa-content MCP server config, injected by the host at startup.
# When None, the kernel cannot load the mode catalog.
_content_mcp_config: MCPClientConfig | None = None

_EXIT_STAGE_ID = "exit"

# Front-matter delimiter used in instruction markdown files.
_FRONT_MATTER_DELIMITER = "---"

# Callbacks injected by the host app layer so the kernel stays decoupled
# from the concrete operation registry module. Set at startup via
# register_operation_callbacks().
_get_operation_catalog_fn: Callable[[], dict[str, Any]] | None = None
_execute_operation_fn: Callable[[str, OperationExecutionContext], Any] | None = None


class WorkflowValidationError(ValueError):
    """Raised when a workflow mode configuration violates FSM invariants.

    Thrown at startup during catalog validation so misconfigured modes are caught
    before any user traffic arrives.
    """


@dataclass(slots=True)
class WorkflowToolCall:
    """Normalized tool call payload extracted from model completion metadata.

    Produced by the host agent loop when the model returns a 'transition_stage'
    tool call and passed to apply_transition() to advance the session.
    """

    name: str
    arguments: dict[str, object]


def register_operation_callbacks(
    get_operation_catalog_fn: Callable[[], dict[str, Any]],
    execute_operation_fn: Callable[[str, OperationExecutionContext], Any],
) -> None:
    """Inject operation-registry callbacks from the host app layer.

    The kernel cannot import the app-owned wiring layer directly (layer
    boundary), so the host calls this once at startup to wire the two halves
    together.
    """
    global _get_operation_catalog_fn, _execute_operation_fn  # noqa: PLW0603
    _get_operation_catalog_fn = get_operation_catalog_fn
    _execute_operation_fn = execute_operation_fn


def register_content_mcp(config: MCPClientConfig) -> None:
    """Inject the ``lisa-content`` MCP server config for catalog loading.

    Must be called before the workflow catalog is first accessed (before any
    call to ``get_workflow_catalog()``). The kernel calls ``list_modes`` on the
    server to discover and load all workflow modes. The server does all
    filesystem access; the kernel never touches the filesystem directly.
    """
    global _content_mcp_config  # noqa: PLW0603
    _content_mcp_config = config


def _known_connector_ids() -> frozenset[str]:
    """Return installed signal connector IDs discovered from extension entry points."""

    return frozenset(discover_installed_signal_collectors())


def _format_connector_snapshot_value(value: Any) -> str:
    """Render one connector snapshot scalar without domain-specific assumptions."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _render_connector_snapshot_summary(snapshot: Any) -> str:
    """Render one connector snapshot generically from its serialized mapping."""

    if not isinstance(snapshot, Mapping):
        return "present"

    parts: list[str] = []
    status = snapshot.get("status")
    if isinstance(status, str) and status:
        parts.append(f"status={status.upper()}")

    for key, value in snapshot.items():
        if key in {"status", "collected_at", "generated_at", "updated_at"}:
            continue
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={_format_connector_snapshot_value(value)}")
        elif isinstance(value, Mapping):
            parts.append(f"{key}={len(value)} fields")
        elif isinstance(value, list):
            parts.append(f"{key}={len(value)} items")
        if len(parts) >= 5:
            break

    return " | ".join(parts) if parts else "present"


def _get_operation_catalog() -> dict[str, Any]:
    """Return the operation catalog via the injected callback."""
    if _get_operation_catalog_fn is None:
        raise RuntimeError(
            "Operation callbacks not registered. "
            "Call kernel.ipsar_fsm.register_operation_callbacks() at startup."
        )
    return _get_operation_catalog_fn()


async def _execute_operation(
    operation_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute an operation via the injected callback."""
    if _execute_operation_fn is None:
        raise RuntimeError(
            "Operation callbacks not registered. "
            "Call kernel.ipsar_fsm.register_operation_callbacks() at startup."
        )
    result = await _execute_operation_fn(operation_id, ctx)
    if isinstance(result, OperationResult):
        return result
    if isinstance(result, str):
        return OperationResult(output_text=result)
    raise WorkflowValidationError(
        f"Operation '{operation_id}' returned unsupported result type {type(result).__name__}."
    )


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a YAML front-matter block from the top of a markdown string.

    Expects the file to start with a `---` delimiter line, followed by YAML
    content, followed by another `---` delimiter line.  The body is everything
    after the closing delimiter, stripped of leading and trailing whitespace.

    Always uses yaml.safe_load() — never yaml.load() — to avoid arbitrary
    code execution vulnerabilities in untrusted files.

    Args:
        text: Full file contents including the front-matter block.

    Returns:
        A tuple of (front_matter_dict, body_text).  If no valid front-matter
        block is found, returns ({}, text) so callers degrade gracefully.

    Raises:
        WorkflowValidationError: If the YAML in the front-matter is invalid.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        # No front-matter block: treat entire content as body.
        return {}, text

    # Find the closing --- line (must be after at least one content line).
    try:
        close_idx = next(
            i
            for i, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration:
        # Opening --- found but no closing ---: treat entire content as body.
        return {}, text

    yaml_text = "\n".join(lines[1:close_idx])
    body_text = "\n".join(lines[close_idx + 1 :]).strip()

    try:
        front_matter = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise WorkflowValidationError(f"Invalid YAML front-matter: {exc}") from exc

    if not isinstance(front_matter, dict):
        raise WorkflowValidationError(
            f"Front-matter must be a YAML mapping, got {type(front_matter).__name__}."
        )

    return front_matter, body_text


# ---------------------------------------------------------------------------
# Front-matter parsing (pure string function, no filesystem access)
# ---------------------------------------------------------------------------


def _derived_workshop_panel_limit(budget: WorkshopBudgetConfig) -> int:
    """Return the host-bounded maximum panel count allowed for one workshop."""

    remaining_context = settings.workshop_max_context_tokens - budget.shared_context_tokens
    if remaining_context <= 0:
        return 0
    context_limited = remaining_context // budget.per_panel_context_tokens
    return max(0, min(settings.workshop_max_panels_hard_cap, context_limited))


def _validate_workshop_budget(mode: WorkflowModeConfig) -> None:
    """Validate one mode's authored workshop budget against host ceilings."""

    budget = mode.workshop_budget
    if budget.shared_context_tokens > settings.workshop_max_context_tokens:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' workshop shared_context_tokens exceeds host ceiling "
            f"{settings.workshop_max_context_tokens}."
        )
    if budget.synthesis_max_tokens > settings.workshop_max_synthesis_tokens:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' workshop synthesis_max_tokens exceeds host ceiling "
            f"{settings.workshop_max_synthesis_tokens}."
        )
    if budget.max_parallel_panels > settings.workshop_max_parallel_panels_hard_cap:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' workshop max_parallel_panels exceeds host ceiling "
            f"{settings.workshop_max_parallel_panels_hard_cap}."
        )

    derived_panel_limit = _derived_workshop_panel_limit(budget)
    if mode.workshop_panels and derived_panel_limit <= 0:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' workshop budget leaves no room for panel "
            "contexts under the host ceiling."
        )
    if len(mode.workshop_panels) > derived_panel_limit:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' declares {len(mode.workshop_panels)} workshop panels, "
            f"but the authored budget allows at most {derived_panel_limit}."
        )


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------
def validate_workflow_mode(mode: WorkflowModeConfig) -> None:
    """Validate one workflow mode for deterministic exits and legal transitions.

    Checks performed (in order):
    1. start_stage_id exists in stages dict.
    2. At least one terminal stage exists.
    3. Terminal stages have no outgoing transitions.
    4. Non-terminal stages with no transitions must allow user questions.
    5. All transition targets exist in the stages dict.
    6. Self-loop transitions require allow_user_questions=true.
    7. All connector IDs in required_connectors are in the installed connector registry.
    8. DFS reachability: every reachable stage has a path to a terminal stage.

    Args:
        mode: The WorkflowModeConfig to validate.

    Raises:
        WorkflowValidationError: On any invariant violation.
    """
    if mode.mode_kind not in {"workflow", "chat", "action"}:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' has invalid mode_kind '{mode.mode_kind}'."
        )

    _validate_mode_operation_bindings(mode)

    if mode.mode_kind in {"chat", "action"}:
        if mode.stages:
            raise WorkflowValidationError(
                f"Non-workflow mode '{mode.mode_id}' cannot define FSM stages."
            )
        return

    if mode.start_stage_id not in mode.stages:
        if not mode.stages:
            raise WorkflowValidationError(
                f"Workflow mode '{mode.mode_id}' must define at least one stage."
            )
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' has unknown start stage '{mode.start_stage_id}'."
        )

    terminal_stages = [stage for stage in mode.stages.values() if stage.terminal]
    if not terminal_stages:
        raise WorkflowValidationError(
            f"Mode '{mode.mode_id}' must define at least one terminal stage."
        )

    connector_ids = _known_connector_ids()

    for stage_id, stage in mode.stages.items():
        if stage.terminal and stage.allowed_next_stages:
            raise WorkflowValidationError(
                f"Mode '{mode.mode_id}' stage '{stage_id}' is terminal but has transitions."
            )

        if not stage.terminal and not stage.allowed_next_stages and not stage.allow_user_questions:
            raise WorkflowValidationError(
                f"Mode '{mode.mode_id}' stage '{stage_id}' cannot progress "
                "and cannot ask questions."
            )

        for next_stage in stage.allowed_next_stages:
            if next_stage not in mode.stages:
                raise WorkflowValidationError(
                    f"Mode '{mode.mode_id}' stage '{stage_id}' targets "
                    f"unknown stage '{next_stage}'."
                )
            if next_stage == stage_id and not stage.allow_user_questions:
                raise WorkflowValidationError(
                    f"Mode '{mode.mode_id}' stage '{stage_id}' self-loop "
                    "requires allow_user_questions=true."
                )

        for connector_id in stage.required_connectors:
            if connector_id not in connector_ids:
                raise WorkflowValidationError(
                    f"Mode '{mode.mode_id}' stage '{stage_id}' references "
                    f"unknown connector '{connector_id}'. "
                    f"Known connectors: {sorted(connector_ids)}."
                )

    _assert_all_paths_reach_terminal(mode)


def _validate_mode_operation_bindings(mode: WorkflowModeConfig) -> None:
    """Validate that all operation IDs referenced by one mode exist in the operation catalog."""
    from kernel.operations.registry import is_known_tool

    catalog = _get_operation_catalog()
    referenced_operation_ids = (
        mode.entry_operation_ids
        + mode.pre_turn_operation_ids
        + mode.post_turn_operation_ids
        + mode.allowed_operation_ids
    )
    for operation_id in referenced_operation_ids:
        if operation_id not in catalog and not is_known_tool(operation_id):
            raise WorkflowValidationError(
                f"Mode '{mode.mode_id}' references unknown operation '{operation_id}'."
            )


def validate_workflow_catalog(catalog: dict[str, WorkflowModeConfig]) -> None:
    """Validate all workflow modes in the catalog.

    Called at startup after catalog loading to catch misconfigured
    modes before any user traffic arrives.

    Args:
        catalog: Mapping of mode_id -> WorkflowModeConfig to validate.

    Raises:
        WorkflowValidationError: If any mode fails validation.
    """
    for mode in catalog.values():
        validate_workflow_mode(mode)

    for mode in catalog.values():
        _validate_workshop_budget(mode)
        for panel in mode.workshop_panels:
            panel_mode = catalog.get(panel.mode_id)
            if panel_mode is None:
                raise WorkflowValidationError(
                    f"Mode '{mode.mode_id}' workshop panel references unknown "
                    f"mode '{panel.mode_id}'."
                )
            if panel_mode.mode_kind != "workflow":
                raise WorkflowValidationError(
                    f"Mode '{mode.mode_id}' workshop panel mode '{panel.mode_id}' "
                    "must be workflow mode."
                )


def _assert_all_paths_reach_terminal(mode: WorkflowModeConfig) -> None:
    """Assert every reachable branch from start has a route to a terminal stage."""
    adjacency = {
        stage_id: list(stage.allowed_next_stages) for stage_id, stage in mode.stages.items()
    }
    can_reach_terminal: dict[str, bool] = {}

    def compute_reachability(stage_id: str, visiting: set[str]) -> bool:
        if stage_id in can_reach_terminal:
            return can_reach_terminal[stage_id]
        if stage_id in visiting:
            can_reach_terminal[stage_id] = False
            return False

        visiting.add(stage_id)
        stage = mode.stages[stage_id]
        if stage.terminal:
            visiting.remove(stage_id)
            can_reach_terminal[stage_id] = True
            return True

        reachable = False
        for next_stage in adjacency[stage_id]:
            if compute_reachability(next_stage, visiting):
                reachable = True

        visiting.remove(stage_id)
        if stage.allow_user_questions and reachable:
            can_reach_terminal[stage_id] = True
        else:
            can_reach_terminal[stage_id] = reachable
        return can_reach_terminal[stage_id]

    reachable_from_start = _collect_reachable_stages(mode)
    for stage_id in reachable_from_start:
        if not compute_reachability(stage_id, set()):
            raise WorkflowValidationError(
                f"Mode '{mode.mode_id}' has no terminal path from stage '{stage_id}'."
            )


def _collect_reachable_stages(mode: WorkflowModeConfig) -> set[str]:
    """Collect all stages reachable from the start stage."""
    visited: set[str] = set()
    queue: deque[str] = deque([mode.start_stage_id])
    while queue:
        stage_id = queue.popleft()
        if stage_id in visited:
            continue
        visited.add(stage_id)
        for next_stage in mode.stages[stage_id].allowed_next_stages:
            queue.append(next_stage)
    return visited


def load_stage_instruction_markdown(stage: WorkflowStageConfig) -> str:
    """Return the instruction prose for a stage.

    The body text is loaded by the ``lisa-content`` MCP server into
    ``stage.instruction_body`` at catalog load time, so this function performs
    no I/O. If ``instruction_body`` is empty (e.g. test-constructed configs
    with no body), returns an empty string — no disk fallback.

    Args:
        stage: The WorkflowStageConfig for the current session stage.

    Returns:
        Instruction markdown string to inject into the AI prompt.
    """
    return stage.instruction_body


def create_workflow_session(user_id: str, user_slug: str, mode_id: str) -> WorkflowSessionState:
    """Create a fresh session pinned to the configured start stage for a mode."""
    catalog = get_workflow_catalog()
    mode = catalog[mode_id]
    return WorkflowSessionState(
        user_id=user_id,
        user_slug=user_slug,
        mode_id=mode.mode_id,
        stage_id=mode.start_stage_id,
        entered_at=datetime.now(UTC).isoformat(),
    )


def apply_transition(
    session: WorkflowSessionState,
    transition: WorkflowTransitionRequest,
) -> WorkflowSessionState:
    """Validate and apply a stage transition request for an active workflow session."""
    mode = get_workflow_catalog()[session.mode_id]
    current_stage = mode.stages[session.stage_id]

    if transition.target_stage_id not in current_stage.allowed_next_stages:
        raise WorkflowValidationError(
            f"Illegal transition in mode '{session.mode_id}': '{session.stage_id}' -> "
            f"'{transition.target_stage_id}'."
        )

    session.stage_id = transition.target_stage_id
    session.turn_count += 1
    session.entered_at = datetime.now(UTC).isoformat()
    if transition.required_missing_inputs:
        session.stage_data["required_missing_inputs"] = transition.required_missing_inputs
    if transition.rationale:
        session.stage_data["last_transition_rationale"] = transition.rationale
    return session


def is_fsm_mode(mode_id: str) -> bool:
    """Return True when the given mode_id corresponds to a known FSM workflow mode.

    Chat/action modes (those with no stages, e.g. 'default') are excluded even
    though they are present in the catalog — they carry description/hooks but
    have no FSM stage graph and must not enter the workflow turn path.

    Args:
        mode_id: The route mode string returned by ipsar.parse_route().

    Returns:
        True if mode_id is present in the loaded workflow catalog AND has stages.
    """
    mode = get_workflow_catalog().get(mode_id)
    return mode is not None and bool(mode.stages)


def is_terminal_stage(session: WorkflowSessionState) -> bool:
    """Return True when the session currently points at a terminal stage."""
    mode = get_workflow_catalog()[session.mode_id]
    return mode.stages[session.stage_id].terminal


def get_stage_config(session: WorkflowSessionState) -> WorkflowStageConfig:
    """Return stage config for an active workflow session."""
    mode = get_workflow_catalog()[session.mode_id]
    return mode.stages[session.stage_id]


def get_required_connectors_for_stage(session: WorkflowSessionState) -> list[str]:
    """Return connector IDs needed for the active stage's context payload."""
    return list(get_stage_config(session).required_connectors)


def build_filtered_live_context(
    state: UnifiedState,
    connectors: list[str],
) -> str:
    """Render stage-scoped telemetry markdown from the full unified state payload."""
    lines = [f"- Snapshot compiled at: {state.compiled_at}"]
    connector_metadata = discover_installed_signal_collectors()

    for connector_id in connectors:
        display_name = connector_metadata.get(
            connector_id,
            connector_id.replace("_", " ").title(),
        )
        if not isinstance(display_name, str):
            display_name = display_name.display_name
        snapshot = state.signal_snapshots.get(connector_id)
        if snapshot is None:
            lines.append(f"- {display_name}: unavailable")
            continue
        lines.append(f"- {display_name}: {_render_connector_snapshot_summary(snapshot)}")

    if state.active_alerts:
        lines.append(f"- Active alerts: {len(state.active_alerts)}")
    else:
        lines.append("- Active alerts: none")

    return "\n".join(lines)


def get_transition_tool_schema() -> list[dict[str, object]]:
    """Return the structured tool schema used for stage transitions."""
    return [
        build_strict_openai_function_tool(
            name="transition_stage",
            description=(
                "Move to the next workflow stage only after enough evidence exists. "
                "If user input is still missing, ask a question instead of calling this tool."
            ),
            parameters_schema=WorkflowTransitionRequest,
        )
    ]


_workflow_catalog: dict[str, WorkflowModeConfig] | None = None


def _ensure_catalog_loaded() -> None:
    """Lazy-load the workflow catalog on first access via the lisa-content MCP server.

    Deferred from module-import time so the host app has a chance to register
    operation callbacks and the content MCP config before validation runs.
    """
    global _workflow_catalog  # noqa: PLW0603
    if _workflow_catalog is not None:
        return
    if _content_mcp_config is None:
        raise RuntimeError(
            "Content MCP config not registered. "
            "Call register_content_mcp() at startup before using the workflow catalog."
        )
    from kernel.context.mcp_catalog import MCPModeCatalog

    catalog_loader = MCPModeCatalog(_content_mcp_config)
    _workflow_catalog = catalog_loader.get_workflow_catalog()
    validate_workflow_catalog(_workflow_catalog)


def get_workflow_catalog() -> dict[str, WorkflowModeConfig]:
    """Return the validated workflow mode catalog.

    The catalog is populated lazily on first access via the ``lisa-content`` MCP
    server. The host must call ``register_content_mcp()`` at startup before any
    code accesses the catalog. A restart is required to pick up new or modified
    instruction folders.
    """
    _ensure_catalog_loaded()
    return _workflow_catalog  # type: ignore[return-value]


def get_mode_system_intro(mode_id: str) -> str:
    """Return the mode-level system prompt intro string for a given mode.

    Reads the instruction_body from the WorkflowModeConfig loaded at startup
    from the mode's _mode.md body. Falls back to description if body is empty.
    The string is injected ahead of the stage-level instructions.

    Returns an empty string for unknown modes so callers can safely include it
    without a mode membership check.

    Args:
        mode_id: The route mode string (e.g. 'coach', 'chase').

    Returns:
        The instruction body from the mode's _mode.md, falling back to description,
        or '' if the mode is not in the catalog.
    """
    mode = get_workflow_catalog().get(mode_id)
    if not mode:
        return ""
    return mode.instruction_body or mode.description


async def _execute_mode_operation_ids(
    operation_ids: list[str],
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute one ordered list of operation IDs and merge their structured results.

    Args:
        operation_ids: Ordered operation identifiers declared by one mode.
        ctx: Shared execution context for the current mode turn.

    Returns:
        Combined OperationResult. Text outputs are joined with blank lines, extra
        metadata is shallow-merged, and a single-operation JSON payload is passed
        through directly for convenience.
    """
    if not operation_ids:
        return OperationResult()

    output_parts: list[str] = []
    extra: dict[str, Any] = dict(ctx.extra)
    json_results: list[tuple[str, dict[str, Any]]] = []

    for operation_id in operation_ids:
        operation_ctx = ctx.model_copy(update={"operation_id": operation_id, "extra": extra})
        result = await _execute_operation(operation_id, operation_ctx)
        if result.output_text:
            output_parts.append(result.output_text)
        if result.output_json:
            json_results.append((operation_id, result.output_json))
        if result.extra:
            extra.update(result.extra)

    if len(json_results) == 1:
        output_json = json_results[0][1]
    else:
        output_json = {operation_id: payload for operation_id, payload in json_results}

    return OperationResult(
        output_text="\n\n".join(output_parts),
        output_json=output_json,
        extra=extra,
    )


async def run_mode_entry_operations(
    mode_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute the entry operations configured for one action mode."""
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return OperationResult()
    return await _execute_mode_operation_ids(mode.entry_operation_ids, ctx)


async def run_mode_pre_turn_operations(
    mode_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute pre-turn operations for one workflow or chat mode."""
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return OperationResult()
    if mode.pre_turn_operation_ids:
        try:
            return await _execute_mode_operation_ids(mode.pre_turn_operation_ids, ctx)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("pre_turn operations failed for mode '%s': %s", mode_id, exc)
    return OperationResult()


async def run_mode_post_turn_operations(
    mode_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute post-turn operations for one workflow or chat mode."""
    mode = get_workflow_catalog().get(mode_id)
    if mode is None:
        return OperationResult()
    if mode.post_turn_operation_ids:
        try:
            return await _execute_mode_operation_ids(mode.post_turn_operation_ids, ctx)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("post_turn operations failed for mode '%s': %s", mode_id, exc)
    return OperationResult()
