"""Tool-schema assembly and pruning for the turn runner.

Extracted from ``kernel.turn_runner`` to reduce file complexity. Owns the
final model-facing tool-surface construction: MCP tool translation, in-process
(pack-contributed) tool schema building, operation-catalog filtering, mode
allowlist enforcement, duplicate/invalid schema pruning, and the post-handoff
tool-surface rebuild. Also owns the ``_TRANSITION_TOOL_NAME`` constant used
by both the prune pass and the turn loop's transition interception.

DESIGN — Re-exported by ``kernel.turn_runner``:
    Existing tests import ``_prune_model_tool_schemas``,
    ``_build_in_process_tool_schemas``, and ``_rebuild_tool_schemas_for_mode``
    directly from ``kernel.turn_runner``. That module re-exports every symbol
    below so no test file needs editing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kernel.ipsar_fsm import get_routing_targets, get_transition_tool_schema
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.operations.registry import build_operation_tool_schemas, get_operation_catalog
from kernel.tools.mcp_client import MCPToolPool
from kernel.tools.registry import get_tool_catalog
from kernel.tools.schema_translator import mcp_tool_to_openai_function

if TYPE_CHECKING:
    from kernel.turn_runner import TurnHooks

__all__ = [
    "_TRANSITION_TOOL_NAME",
    "_YIELD_TOOL_NAME",
    "_build_in_process_tool_schemas",
    "_mode_bound_operation_ids",
    "_operation_allowed_for_mode",
    "_prune_model_tool_schemas",
    "_rebuild_tool_schemas_for_mode",
    "_tool_schema_name",
]

# The tool name the model calls to request a workflow stage transition.
# Intercepted by the turn loop before the operation/tool/MCP dispatch branches.
_TRANSITION_TOOL_NAME = "transition_stage"

# The tool name the model calls to yield (ask the user a question).
# Intercepted by the turn loop alongside transition_stage — it is a terminal
# action that breaks the loop and stamps CheckpointBundle.lifecycle_state.
_YIELD_TOOL_NAME = "yield"


def _build_in_process_tool_schemas(
    mode_config: Any,
) -> list[dict[str, Any]]:
    """Build OpenAI tool schemas for in-process (pack-contributed) tools.

    Advertises tools from the internal tool catalog (discovered from packs)
    that are allowed for the current mode. Mirrors
    ``build_operation_tool_schemas`` but for the tool catalog.
    """
    catalog = get_tool_catalog()
    if not catalog:
        return []

    allowed_tool_ids: set[str] | None = None
    if mode_config and (mode_config.allowed_tool_ids or mode_config.tools_authoritative):
        allowed_tool_ids = set(mode_config.allowed_tool_ids)

    schemas: list[dict[str, Any]] = []
    for tool_id, definition in catalog.items():
        if allowed_tool_ids is not None and tool_id not in allowed_tool_ids:
            continue
        if definition.mode_allowlist is not None and mode_config:
            if mode_config.mode_id not in definition.mode_allowlist:
                continue
        schemas.append(
            build_strict_openai_function_tool(
                name=tool_id,
                description=definition.description or f"Execute {tool_id}.",
                parameters_schema=definition.parameters_schema,
            )
        )
    return schemas


def _tool_schema_name(schema: dict[str, Any]) -> str:
    """Return the function name from an OpenAI tool schema."""
    function = schema.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return str(name).strip() if isinstance(name, str) else ""


def _mode_bound_operation_ids(mode_config: Any) -> set[str]:
    """Return operation ids explicitly bound by one mode definition."""
    if mode_config is None:
        return set()
    return set(
        list(getattr(mode_config, "entry_operation_ids", []) or [])
        + list(getattr(mode_config, "pre_turn_operation_ids", []) or [])
        + list(getattr(mode_config, "post_turn_operation_ids", []) or [])
        + list(getattr(mode_config, "allowed_operation_ids", []) or [])
    )


def _operation_allowed_for_mode(
    *,
    operation_id: str,
    definition: Any,
    mode_config: Any,
    tools_authoritative: bool,
    mode_bound_ids: set[str],
) -> bool:
    """Return whether one operation should be advertised for this mode."""
    if mode_config is None:
        return True

    mode_id = str(getattr(mode_config, "mode_id", "") or "")
    mode_allowlist = getattr(definition, "mode_allowlist", None)
    if mode_allowlist and mode_id in mode_allowlist:
        return True
    if operation_id in mode_bound_ids:
        return True

    if tools_authoritative:
        return False
    if mode_allowlist:
        return False
    return True


def _prune_model_tool_schemas(
    *,
    tool_schemas: list[dict[str, Any]],
    mode_config: Any,
    is_workflow: bool,
    has_routing_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply one final model-facing prune pass over assembled tool schemas.

    Assembly-time filtering can evolve across MCP/operation/tool catalog paths.
    This final pass is the payload contract: only the allowed tool set reaches
    the model completion call.
    """

    operation_catalog = get_operation_catalog()
    tool_catalog = get_tool_catalog()
    tools_authoritative = bool(mode_config and getattr(mode_config, "tools_authoritative", True))
    allowed_tool_ids: set[str] | None = None
    if mode_config and (getattr(mode_config, "allowed_tool_ids", []) or tools_authoritative):
        allowed_tool_ids = set(getattr(mode_config, "allowed_tool_ids", []) or [])

    mode_id = str(getattr(mode_config, "mode_id", "") or "")
    mode_bound_ids = _mode_bound_operation_ids(mode_config)
    seen_names: set[str] = set()
    pruned: list[dict[str, Any]] = []
    dropped_duplicate = 0
    dropped_disallowed = 0
    dropped_invalid = 0

    for schema in tool_schemas:
        name = _tool_schema_name(schema)
        if not name:
            dropped_invalid += 1
            continue
        if name in seen_names:
            dropped_duplicate += 1
            continue
        seen_names.add(name)

        if name == _TRANSITION_TOOL_NAME:
            if is_workflow or has_routing_targets:
                pruned.append(schema)
            else:
                dropped_disallowed += 1
            continue

        # The yield tool is advertised by the turn runner based on the mode's
        # yield_policy; it is not in the operation/tool catalog. Let it
        # through so the model can call it.
        if name == _YIELD_TOOL_NAME:
            pruned.append(schema)
            continue

        op_def = operation_catalog.get(name)
        if op_def is not None:
            if _operation_allowed_for_mode(
                operation_id=name,
                definition=op_def,
                mode_config=mode_config,
                tools_authoritative=tools_authoritative,
                mode_bound_ids=mode_bound_ids,
            ):
                pruned.append(schema)
            else:
                dropped_disallowed += 1
            continue

        tool_def = tool_catalog.get(name)
        if tool_def is not None:
            mode_allowlist = getattr(tool_def, "mode_allowlist", None)
            if allowed_tool_ids is not None and name not in allowed_tool_ids:
                dropped_disallowed += 1
                continue
            if mode_allowlist and mode_id and mode_id not in mode_allowlist:
                dropped_disallowed += 1
                continue
            pruned.append(schema)
            continue

        # Unknown tools are MCP tools. Restrict them by mode allowlist when
        # one is active (including authoritative-empty => no unknown tools).
        if allowed_tool_ids is not None and name not in allowed_tool_ids:
            dropped_disallowed += 1
            continue
        pruned.append(schema)

    stats = {
        "input_count": len(tool_schemas),
        "output_count": len(pruned),
        "dropped_duplicate": dropped_duplicate,
        "dropped_disallowed": dropped_disallowed,
        "dropped_invalid": dropped_invalid,
    }
    return pruned, stats


async def _rebuild_tool_schemas_for_mode(
    target_mode_id: str,
    target_mode_config: Any,
    *,
    pool: MCPToolPool,
    hooks: TurnHooks,
    is_workflow_for_target: bool,
) -> list[dict[str, Any]]:
    """Rebuild the tool schema list for a new mode after a mid-turn handoff.

    Mirrors the tool assembly logic in ``_run_turn_inner`` Phase 1 but
    without re-running pre-turn operations or the input guard. The new
    mode's ``allowed_tool_ids`` and ``tools_authoritative`` determine the
    tool surface.
    """
    allowed_server_ids: set[str] | None = None
    if target_mode_config and target_mode_config.allowed_tool_ids:
        allowed_set = set(target_mode_config.allowed_tool_ids)
        allowed_server_ids = {
            c.server_id
            for c in getattr(pool, "configs", [])
            if any(t.startswith(c.server_id + ".") for t in allowed_set)
        }
    elif (
        target_mode_config
        and target_mode_config.tools_authoritative
        and not target_mode_config.allowed_tool_ids
    ):
        allowed_server_ids = set()

    mcp_tools = await pool.list_tools(allowed_server_ids=allowed_server_ids)
    if target_mode_config and (
        target_mode_config.allowed_tool_ids or target_mode_config.tools_authoritative
    ):
        allowed = set(target_mode_config.allowed_tool_ids)
        mcp_tools = [t for t in mcp_tools if t.name in allowed]
    new_tool_schemas: list[dict[str, Any]] = [
        mcp_tool_to_openai_function(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        for spec in mcp_tools
    ]
    new_tool_schemas.extend(
        build_operation_tool_schemas(
            target_mode_id,
            tools_authoritative=bool(target_mode_config and target_mode_config.tools_authoritative),
        )
    )
    new_tool_schemas.extend(_build_in_process_tool_schemas(target_mode_config))
    if is_workflow_for_target:
        new_tool_schemas.extend(get_transition_tool_schema())
    target_has_routing = bool(get_routing_targets(target_mode_id)) if target_mode_config else False
    new_tool_schemas, _ = _prune_model_tool_schemas(
        tool_schemas=new_tool_schemas,
        mode_config=target_mode_config,
        is_workflow=is_workflow_for_target,
        has_routing_targets=target_has_routing,
    )
    return new_tool_schemas
