"""Workflow finite-state-machine loader and runtime for IPSAR mode progression.

This module is a pure loader/router: it has zero hardcoded knowledge of specific
modes or stages.  The workflow catalog is discovered from the filesystem at startup:

    sdk_packs/core/modes/
    {mode_name}/
    _mode.md         → YAML front-matter declares mode_id, start_stage_id,
                 mode_kind, description, command_pattern, and operation
                 bindings for the user-facing mode entry point
      {stage_id}.md    → YAML front-matter declares stage config; body text is
                         the instruction prose injected into AI prompts

Adding a new workflow mode requires only a new folder with the appropriate files;
no code changes are needed.

Runtime functions (create_workflow_session, apply_transition, etc.) delegate to
the catalog loaded by load_catalog_from_disk() at module import time.

Called by: host apps (session management, stage transitions, context injection)
           kernel.ipsar (parse_route for FSM mode detection)
           packaging tools (manifest summary generation)
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from kernel.config import settings
from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
    resolve_content_subdir_roots,
)
from kernel.intake.connector_discovery import discover_installed_signal_connectors
from kernel.models import (
    DefaultIntentBridge,
    ModeKind,
    OperationExecutionContext,
    OperationResult,
    UnifiedState,
    WorkflowModeConfig,
    WorkflowSessionState,
    WorkflowStageConfig,
    WorkflowTransitionRequest,
    WorkshopBudgetConfig,
    WorkshopPanelModeConfig,
    WorkshopRenderConfig,
)
from kernel.openai_function_tools import build_strict_openai_function_tool

logger = logging.getLogger(__name__)

_INSTRUCTIONS_ROOT = Path(__file__).parent.parent / "sdk_packs" / "core" / "modes"
# Host-injected discovery roots.  Set by register_content_roots() at startup so
# the kernel package does not hard-code paths relative to its install location.
_injected_core_root: Path | None = None
_injected_sdk_roots: list[Path] | None = None
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


def register_content_roots(
    core_root: Path,
    sdk_roots: list[Path] | None = None,
) -> None:
    """Inject mode-discovery roots from the host app layer.

    Must be called before the workflow catalog is first accessed (before any
    call to get_workflow_catalog()).  Calling after the catalog is already
    loaded has no effect; a process restart is required to pick up new roots.

    The kernel package itself has no built-in content roots; the host app
    supplies the baseline pack mode path (and optionally extra SDK roots) so that the
    kernel can be installed and run independently of the host directory tree.
    """
    global _injected_core_root, _injected_sdk_roots  # noqa: PLW0603
    _injected_core_root = core_root
    if sdk_roots is not None:
        _injected_sdk_roots = list(sdk_roots)


def _known_connector_ids() -> frozenset[str]:
    """Return installed signal connector IDs discovered from extension entry points."""

    return frozenset(discover_installed_signal_connectors())


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
# Disk-scanning catalog loader
# ---------------------------------------------------------------------------


def load_catalog_from_disk(
    instructions_dir: Path | None = None,
) -> dict[str, WorkflowModeConfig]:
    """Discover and load all workflow modes from the instructions directory.

    Scans `instructions_dir` (default: injected baseline mode root) for subdirectories.
    Each subdirectory becomes one workflow mode.  The discovery contract is:

    {mode_name}/_mode.md       — required; front-matter must include:
                        mode_id, start_stage_id.  It may also
                        declare mode_kind, command_pattern,
                        require_query, and operation bindings.
      {mode_name}/{stage_id}.md  — one file per stage; front-matter must include:
                                    stage_id, terminal, allowed_next_stages,
                                    allow_user_questions, required_connectors.

    Stage files whose name starts with `_` are skipped (reserved for mode-level
    config files like `_mode.md`).

    The body text below each stage's front-matter delimiter is stored in
    WorkflowStageConfig.instruction_body so runtime prompt injection never
    needs a disk read after startup.

    Args:
        instructions_dir: Override the default scan root.  Useful in tests.

    Returns:
        Dict mapping mode_id -> WorkflowModeConfig, not yet validated.
        Call validate_workflow_catalog() on the result before use.

    Raises:
        WorkflowValidationError: If a required front-matter field is missing
            or if the YAML is malformed.
    """
    root = instructions_dir or _INSTRUCTIONS_ROOT
    catalog: dict[str, WorkflowModeConfig] = {}

    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue

        mode_config_path = mode_dir / "_mode.md"
        if not mode_config_path.exists():
            logger.warning("Skipping instructions folder '%s': missing _mode.md", mode_dir.name)
            continue

        mode_front_matter, mode_body = parse_front_matter(
            mode_config_path.read_text(encoding="utf-8")
        )

        mode_id = _require_str_field(mode_front_matter, "mode_id", mode_config_path)
        start_stage_id = _require_str_field(mode_front_matter, "start_stage_id", mode_config_path)
        mode_kind = cast(
            ModeKind,
            _require_str_field(mode_front_matter, "mode_kind", mode_config_path),
        )
        mode_description = _require_str_field(mode_front_matter, "description", mode_config_path)
        # command_pattern can be empty for fallback chat modes (e.g. default),
        # but the key must exist in front-matter for explicitness.
        raw_pattern = _require_field(mode_front_matter, "command_pattern", mode_config_path)
        command_pattern = str(raw_pattern)
        require_query = bool(mode_front_matter.get("require_query", False))
        mode_default_model = str(mode_front_matter.get("default_model", "") or "") or None
        workshop_planner_prompt_file = _resolve_mode_relative_file(
            mode_dir=mode_dir,
            authored_path=str(mode_front_matter.get("workshop_planner_prompt_file", "") or ""),
            mode_config_path=mode_config_path,
            location="workshop_planner_prompt_file",
        )
        workshop_synthesis_prompt_file = _resolve_mode_relative_file(
            mode_dir=mode_dir,
            authored_path=str(mode_front_matter.get("workshop_synthesis_prompt_file", "") or ""),
            mode_config_path=mode_config_path,
            location="workshop_synthesis_prompt_file",
        )
        workshop_rendering = _parse_workshop_rendering(mode_front_matter, mode_config_path)

        stages: dict[str, WorkflowStageConfig] = {}

        for stage_path in sorted(mode_dir.glob("*.md")):
            # Skip reserved files (e.g. _mode.md).
            if stage_path.name.startswith("_"):
                continue

            stage_front_matter, stage_body = parse_front_matter(
                stage_path.read_text(encoding="utf-8")
            )

            # Non-workflow modes may carry additional authored markdown assets
            # (for example workshop planner/synthesis prompt blocks) that are
            # not workflow stage definitions.
            if "stage_id" not in stage_front_matter:
                if mode_kind in {"chat", "action"}:
                    continue
                raise WorkflowValidationError(
                    f"Front-matter in '{stage_path}' is missing required field 'stage_id'."
                )

            stage_id = _require_str_field(stage_front_matter, "stage_id", stage_path)
            terminal = _require_bool_field(stage_front_matter, "terminal", stage_path)
            allowed_next_stages = _require_list_field(
                stage_front_matter,
                "allowed_next_stages",
                stage_path,
            )
            allow_user_questions = _require_bool_field(
                stage_front_matter,
                "allow_user_questions",
                stage_path,
            )
            required_connectors = _require_list_field(
                stage_front_matter,
                "required_connectors",
                stage_path,
            )
            stage_description = _require_str_field(stage_front_matter, "description", stage_path)
            stage_default_model = str(stage_front_matter.get("default_model", "") or "") or None

            stages[stage_id] = WorkflowStageConfig(
                stage_id=stage_id,
                instruction_markdown_path=str(stage_path),
                instruction_body=stage_body,
                description=stage_description,
                default_model=stage_default_model,
                required_connectors=required_connectors,
                allowed_next_stages=allowed_next_stages,
                allow_user_questions=allow_user_questions,
                terminal=terminal,
            )

        entry_operation_ids = [
            str(operation_id)
            for operation_id in (mode_front_matter.get("entry_operation_ids") or [])
        ]
        pre_turn_operation_ids = [
            str(operation_id)
            for operation_id in (mode_front_matter.get("pre_turn_operation_ids") or [])
        ]
        post_turn_operation_ids = [
            str(operation_id)
            for operation_id in (mode_front_matter.get("post_turn_operation_ids") or [])
        ]
        allowed_operation_ids = [
            str(operation_id)
            for operation_id in (mode_front_matter.get("allowed_operation_ids") or [])
        ]
        frontdoor_routable = mode_front_matter.get("frontdoor_routable", False)
        if not isinstance(frontdoor_routable, bool):
            raise WorkflowValidationError(
                f"frontdoor_routable must be a boolean in {mode_config_path}"
            )
        frontdoor_priority = mode_front_matter.get("frontdoor_priority", 0)
        if isinstance(frontdoor_priority, bool) or not isinstance(frontdoor_priority, int):
            raise WorkflowValidationError(
                f"frontdoor_priority must be an integer in {mode_config_path}"
            )
        frontdoor_fast_path_patterns_raw = mode_front_matter.get("frontdoor_fast_path_patterns")
        if frontdoor_fast_path_patterns_raw is None:
            frontdoor_fast_path_patterns_raw = []
        if not isinstance(frontdoor_fast_path_patterns_raw, list):
            raise WorkflowValidationError(
                f"frontdoor_fast_path_patterns must be a list in {mode_config_path}"
            )
        frontdoor_fast_path_patterns: list[str] = []
        for idx, raw_pattern in enumerate(frontdoor_fast_path_patterns_raw):
            if not isinstance(raw_pattern, str):
                raise WorkflowValidationError(
                    f"frontdoor_fast_path_patterns[{idx}] must be a string in {mode_config_path}"
                )
            pattern = raw_pattern.strip()
            if not pattern:
                raise WorkflowValidationError(
                    f"frontdoor_fast_path_patterns[{idx}] cannot be empty in {mode_config_path}"
                )
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise WorkflowValidationError(
                    "frontdoor_fast_path_patterns"
                    f"[{idx}] is not a valid regex in {mode_config_path}: {exc}"
                ) from exc
            frontdoor_fast_path_patterns.append(pattern)

        frontdoor_route_hints_raw = mode_front_matter.get("frontdoor_route_hints")
        if frontdoor_route_hints_raw is None:
            frontdoor_route_hints_raw = []
        if not isinstance(frontdoor_route_hints_raw, list):
            raise WorkflowValidationError(
                f"frontdoor_route_hints must be a list in {mode_config_path}"
            )
        frontdoor_route_hints: list[str] = []
        for idx, raw_hint in enumerate(frontdoor_route_hints_raw):
            if not isinstance(raw_hint, str):
                raise WorkflowValidationError(
                    f"frontdoor_route_hints[{idx}] must be a string in {mode_config_path}"
                )
            hint = raw_hint.strip()
            if not hint:
                raise WorkflowValidationError(
                    f"frontdoor_route_hints[{idx}] cannot be empty in {mode_config_path}"
                )
            frontdoor_route_hints.append(hint)

        frontdoor_examples_raw = mode_front_matter.get("frontdoor_examples")
        if frontdoor_examples_raw is None:
            frontdoor_examples_raw = []
        if not isinstance(frontdoor_examples_raw, list):
            raise WorkflowValidationError(
                f"frontdoor_examples must be a list in {mode_config_path}"
            )
        frontdoor_examples: list[str] = []
        for idx, raw_example in enumerate(frontdoor_examples_raw):
            if not isinstance(raw_example, str):
                raise WorkflowValidationError(
                    f"frontdoor_examples[{idx}] must be a string in {mode_config_path}"
                )
            example = raw_example.strip()
            if not example:
                raise WorkflowValidationError(
                    f"frontdoor_examples[{idx}] cannot be empty in {mode_config_path}"
                )
            frontdoor_examples.append(example)

        workshop_budget = _parse_workshop_budget(mode_front_matter, mode_config_path)
        workshop_panels: list[WorkshopPanelModeConfig] = []
        workshop_panels_raw = mode_front_matter.get("workshop_panels") or []
        if not isinstance(workshop_panels_raw, list):
            raise WorkflowValidationError(
                "workshop_panels must be a list, got "
                f"{type(workshop_panels_raw).__name__} in {mode_config_path}"
            )
        for idx, panel_config in enumerate(workshop_panels_raw):
            if not isinstance(panel_config, dict):
                raise WorkflowValidationError(
                    f"workshop_panels[{idx}] must be a dict, got "
                    f"{type(panel_config).__name__} in {mode_config_path}"
                )
            mode_id_value = _require_str_field(
                panel_config,
                "mode_id",
                mode_config_path,
                location=f"workshop_panels[{idx}]",
            )
            panel_id_value = str(panel_config.get("panel_id", "") or "").strip()
            title_value = str(panel_config.get("title", "") or "").strip()
            model_alias_value = str(panel_config.get("model_alias", "") or "").strip()
            query_prefix_value = str(panel_config.get("query_prefix", "") or "").strip()
            query_prompt_file_value = _resolve_mode_relative_file(
                mode_dir=mode_dir,
                authored_path=str(panel_config.get("query_prompt_file", "") or ""),
                mode_config_path=mode_config_path,
                location=f"workshop_panels[{idx}].query_prompt_file",
            )
            workshop_panels.append(
                WorkshopPanelModeConfig(
                    panel_id=panel_id_value,
                    title=title_value,
                    mode_id=mode_id_value,
                    model_alias=model_alias_value,
                    query_prefix=query_prefix_value,
                    query_prompt_file=query_prompt_file_value,
                )
            )

        # Parse default_intent_bridges: list of bridge configs from mode front-matter.
        default_intent_bridges = []
        bridges_raw = mode_front_matter.get("default_intent_bridges") or []
        if not isinstance(bridges_raw, list):
            raise WorkflowValidationError(
                "default_intent_bridges must be a list, got "
                f"{type(bridges_raw).__name__} in {mode_config_path}"
            )
        for idx, bridge_config in enumerate(bridges_raw):
            if not isinstance(bridge_config, dict):
                raise WorkflowValidationError(
                    f"default_intent_bridges[{idx}] must be a dict, got "
                    f"{type(bridge_config).__name__} in {mode_config_path}"
                )
            target_mode_id = _require_str_field(
                bridge_config,
                "target_mode_id",
                mode_config_path,
                location=f"default_intent_bridges[{idx}]",
            )
            matcher_patterns = _require_list_field(
                bridge_config,
                "matcher_patterns",
                mode_config_path,
                location=f"default_intent_bridges[{idx}]",
            )
            matcher_patterns = [str(p) for p in matcher_patterns]
            query_extraction_strategy = str(
                bridge_config.get("query_extraction_strategy", "passthrough")
            )
            prerequisites = [str(p) for p in (bridge_config.get("prerequisites") or [])]
            recent_handoff_source_modes = [
                str(source_mode)
                for source_mode in (bridge_config.get("recent_handoff_source_modes") or [])
            ]
            recent_handoff_required_artifact_keys = [
                str(artifact_key)
                for artifact_key in (
                    bridge_config.get("recent_handoff_required_artifact_keys") or []
                )
            ]
            recent_handoff_context_key = str(
                bridge_config.get("recent_handoff_context_key", "") or ""
            )
            response_appendix = str(bridge_config.get("response_appendix", ""))

            default_intent_bridges.append(
                DefaultIntentBridge(
                    target_mode_id=target_mode_id,
                    matcher_patterns=matcher_patterns,
                    query_extraction_strategy=query_extraction_strategy,
                    prerequisites=prerequisites,
                    recent_handoff_source_modes=recent_handoff_source_modes,
                    recent_handoff_required_artifact_keys=recent_handoff_required_artifact_keys,
                    recent_handoff_context_key=recent_handoff_context_key,
                    response_appendix=response_appendix,
                )
            )

        catalog[mode_id] = WorkflowModeConfig(
            mode_id=mode_id,
            start_stage_id=start_stage_id,
            stages=stages,
            mode_kind=mode_kind,
            artifact_type=str(mode_front_matter.get("artifact_type", "workflow_summary")),
            description=mode_description,
            instruction_body=mode_body.strip(),
            mode_markdown_path=str(mode_config_path),
            default_model=mode_default_model,
            frontdoor_routable=frontdoor_routable,
            frontdoor_priority=frontdoor_priority,
            frontdoor_fast_path_patterns=frontdoor_fast_path_patterns,
            frontdoor_route_hints=frontdoor_route_hints,
            frontdoor_examples=frontdoor_examples,
            command_pattern=command_pattern,
            require_query=require_query,
            entry_operation_ids=entry_operation_ids,
            pre_turn_operation_ids=pre_turn_operation_ids,
            post_turn_operation_ids=post_turn_operation_ids,
            allowed_operation_ids=allowed_operation_ids,
            workshop_panels=workshop_panels,
            workshop_budget=workshop_budget,
            workshop_planner_prompt_file=workshop_planner_prompt_file,
            workshop_synthesis_prompt_file=workshop_synthesis_prompt_file,
            workshop_rendering=workshop_rendering,
            default_intent_bridges=default_intent_bridges,
        )

    return catalog


def _require_str_field(data: dict[str, Any], field: str, source: Path, location: str = "") -> str:
    """Extract a required string field from a front-matter dict.

    Args:
        data: Parsed front-matter dictionary.
        field: Field name that must be present and non-empty.
        source: File path, used only in the error message for debugging.
        location: Optional nested location (e.g. "default_intent_bridges[0]") for context.

    Returns:
        The string value for the given field.

    Raises:
        WorkflowValidationError: If the field is missing or empty.
    """
    value = data.get(field)
    if not value:
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter in '{source}'{loc_str} is missing required field '{field}'."
        )
    return str(value)


def _resolve_mode_relative_file(
    *,
    mode_dir: Path,
    authored_path: str,
    mode_config_path: Path,
    location: str,
) -> str:
    """Validate a mode-authored relative file path and return normalized relative path."""

    raw = authored_path.strip()
    if not raw:
        return ""

    candidate = Path(raw)
    if candidate.is_absolute():
        raise WorkflowValidationError(
            f"Front-matter field '{location}' in '{mode_config_path}' must be a relative path."
        )

    mode_root = mode_dir.resolve()
    resolved = (mode_root / candidate).resolve()
    if not resolved.is_relative_to(mode_root):
        raise WorkflowValidationError(
            f"Front-matter field '{location}' in '{mode_config_path}' "
            "must stay within the mode folder."
        )
    if not resolved.is_file():
        raise WorkflowValidationError(
            f"Front-matter field '{location}' in '{mode_config_path}' "
            f"points to a missing file '{candidate.as_posix()}'."
        )
    return resolved.relative_to(mode_root).as_posix()


def _require_field(data: dict[str, Any], field: str, source: Path) -> Any:
    """Extract one required front-matter field regardless of value type."""
    if field not in data:
        raise WorkflowValidationError(
            f"Front-matter in '{source}' is missing required field '{field}'."
        )
    return data[field]


def _require_bool_field(data: dict[str, Any], field: str, source: Path) -> bool:
    """Extract one required boolean front-matter field."""
    value = _require_field(data, field, source)
    if not isinstance(value, bool):
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}' must be a boolean."
        )
    return value


def _require_list_field(
    data: dict[str, Any], field: str, source: Path, location: str = ""
) -> list[str]:
    """Extract one required list[str] front-matter field."""
    if field not in data:
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter in '{source}'{loc_str} is missing required field '{field}'."
        )
    raw = data[field]
    if not isinstance(raw, list):
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}'{loc_str} must be a list."
        )
    return [str(item) for item in raw]


def _optional_int_field(
    data: dict[str, Any],
    field: str,
    source: Path,
    *,
    location: str = "",
    minimum: int = 0,
) -> int | None:
    """Extract one optional integer field with a minimum bound."""

    if field not in data:
        return None
    value = data[field]
    if not isinstance(value, int):
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}'{loc_str} must be an integer."
        )
    if value < minimum:
        loc_str = f" at {location}" if location else ""
        raise WorkflowValidationError(
            f"Front-matter field '{field}' in '{source}'{loc_str} must be >= {minimum}."
        )
    return value


def _parse_workshop_budget(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> WorkshopBudgetConfig:
    """Parse one authored workshop budget block with host-backed defaults."""

    workshop_budget_raw = mode_front_matter.get("workshop_budget") or {}
    if not isinstance(workshop_budget_raw, dict):
        raise WorkflowValidationError(
            "workshop_budget must be a dict, got "
            f"{type(workshop_budget_raw).__name__} in {mode_config_path}"
        )

    shared_context_tokens = _optional_int_field(
        workshop_budget_raw,
        "shared_context_tokens",
        mode_config_path,
        location="workshop_budget",
        minimum=0,
    )
    per_panel_context_tokens = _optional_int_field(
        workshop_budget_raw,
        "per_panel_context_tokens",
        mode_config_path,
        location="workshop_budget",
        minimum=1,
    )
    synthesis_max_tokens = _optional_int_field(
        workshop_budget_raw,
        "synthesis_max_tokens",
        mode_config_path,
        location="workshop_budget",
        minimum=1,
    )
    max_parallel_panels = _optional_int_field(
        workshop_budget_raw,
        "max_parallel_panels",
        mode_config_path,
        location="workshop_budget",
        minimum=1,
    )

    return WorkshopBudgetConfig(
        shared_context_tokens=(
            shared_context_tokens
            if shared_context_tokens is not None
            else settings.workshop_default_shared_context_tokens
        ),
        per_panel_context_tokens=(
            per_panel_context_tokens
            if per_panel_context_tokens is not None
            else settings.workshop_default_per_panel_context_tokens
        ),
        synthesis_max_tokens=(
            synthesis_max_tokens
            if synthesis_max_tokens is not None
            else settings.workshop_default_synthesis_max_tokens
        ),
        max_parallel_panels=(
            max_parallel_panels
            if max_parallel_panels is not None
            else settings.workshop_default_max_parallel_panels
        ),
    )


def _parse_workshop_rendering(
    mode_front_matter: dict[str, Any],
    mode_config_path: Path,
) -> WorkshopRenderConfig:
    """Parse optional workshop rendering preferences from mode front-matter."""

    workshop_rendering_raw = mode_front_matter.get("workshop_rendering") or {}
    if not isinstance(workshop_rendering_raw, dict):
        raise WorkflowValidationError(
            "workshop_rendering must be a dict, got "
            f"{type(workshop_rendering_raw).__name__} in {mode_config_path}"
        )

    try:
        return WorkshopRenderConfig.model_validate(workshop_rendering_raw)
    except Exception as exc:  # pylint: disable=broad-except
        raise WorkflowValidationError(
            f"Invalid workshop_rendering in '{mode_config_path}': {exc}"
        ) from exc


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
            return
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

    Called at startup after load_catalog_from_disk() to catch misconfigured
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

    The body text is pre-loaded from disk into stage.instruction_body at startup
    by load_catalog_from_disk(), so this function performs no I/O at runtime.
    Falls back to reading the file from disk if instruction_body is empty (e.g.
    for WorkflowStageConfig instances constructed directly in tests).

    Args:
        stage: The WorkflowStageConfig for the current session stage.

    Returns:
        Instruction markdown string to inject into the AI prompt.
    """
    if stage.instruction_body:
        return stage.instruction_body
    # Fallback for test-constructed configs that have a path but no body.
    if stage.instruction_markdown_path:
        return Path(stage.instruction_markdown_path).read_text(encoding="utf-8")
    return ""


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
    connector_metadata = discover_installed_signal_connectors()

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


def load_catalog_merged(
    core_root: Path | None = None,
    sdk_roots: list[Path] | None = None,
) -> dict[str, WorkflowModeConfig]:
    """Discover workflow modes from one core root and zero or more SDK roots.

    The core root is always loaded first so that built-in modes are always
    available.  Each SDK root is then merged in order; if two roots define the
    same mode_id, the *last* one wins (SDK can override core).

    Args:
        core_root: Core modes directory.  Defaults to
            the loaded mode root relative to the workspace root.
        sdk_roots: Additional roots contributed by lisa_sdk or third-party
            operation packages.  Loaded in list order after the core root.

    Returns:
        Merged dict mapping mode_id -> WorkflowModeConfig, not yet validated.
        Call validate_workflow_catalog() on the result if needed.
    """
    merged: dict[str, WorkflowModeConfig] = {}
    merged.update(load_catalog_from_disk(core_root))
    for extra_root in sdk_roots or []:
        merged.update(load_catalog_from_disk(extra_root))
    return merged


def _load_sdk_instructions_roots() -> list[Path]:
    """Collect all SDK instruction roots via entry-points, in-repo lisa_sdk, and env config.

    Discovery order (ascending precedence — later roots override earlier ones):
    1. ``lisa.extensions.content_roots`` entry-points (installed packages).
    2. In-repo ``lisa_sdk/modes`` (convenience for local dev/test).
    3. ``SDK_CONTENT_ROOTS`` env var (explicit overrides).
    """
    roots = resolve_content_subdir_roots(discover_installed_content_roots(), "modes")
    # 2. Auto-include the in-repo lisa_sdk modes directory when it exists.
    repo_sdk = Path(__file__).parent.parent / "lisa_sdk" / "modes"
    if repo_sdk.is_dir() and repo_sdk not in roots:
        roots.append(repo_sdk)
    # 2b. Auto-include in-repo SDK pack modes for local dev/test when entry
    # points are not active in the current environment.
    repo_sdk_packs_root = Path(__file__).parent.parent / "sdk_packs"
    if repo_sdk_packs_root.is_dir():
        for pack_dir in sorted(repo_sdk_packs_root.iterdir()):
            candidate = pack_dir / "modes"
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    # 3. Env-configured overrides.
    for candidate in resolve_content_subdir_roots(discover_configured_content_roots(), "modes"):
        if candidate not in roots:
            roots.append(candidate)
    return roots


_workflow_catalog: dict[str, WorkflowModeConfig] | None = None


def _ensure_catalog_loaded() -> None:
    """Lazy-load the workflow catalog on first access.

    Deferred from module-import time so the host app has a chance to register
    operation callbacks and content roots before validation runs.
    """
    global _workflow_catalog  # noqa: PLW0603
    if _workflow_catalog is not None:
        return
    # Prefer injected roots (set by register_content_roots at startup) so the
    # kernel package is not forced to resolve siblings by relative path.
    # None → load_catalog_from_disk falls back to _INSTRUCTIONS_ROOT
    core_root = _injected_core_root
    sdk_roots = (
        _injected_sdk_roots if _injected_sdk_roots is not None else _load_sdk_instructions_roots()
    )
    _workflow_catalog = load_catalog_merged(core_root=core_root, sdk_roots=sdk_roots)
    validate_workflow_catalog(_workflow_catalog)


def get_workflow_catalog() -> dict[str, WorkflowModeConfig]:
    """Return the validated workflow mode catalog loaded at startup.

    The catalog is populated lazily on first access so the host app can
    register operation callbacks before validation runs. A restart is required
    to pick up new or modified instruction folders.
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
