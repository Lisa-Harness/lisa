"""Registry for internal operation metadata.

Modes are the only user-facing routing surface. Operations are reusable internal
behaviors discovered from content packs via the ``lisa-content`` MCP server.

The kernel never touches the filesystem directly. The ``lisa-content`` server
scans ``operations/`` subdirectories in all discovered content roots, parses
YAML/metadata, and returns the catalog as JSON. The kernel imports the
implementation module locally using the ``module_path`` field.

Called by: host agent loops (direct action-mode execution)
           kernel.ipsar_fsm (mode binding validation)
"""

from __future__ import annotations

# pylint: disable=global-statement
import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolExecutionTelemetry,
)
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.telemetry import get_tracer, mark_span_failed, set_span_attributes
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)

__all__ = [
    "build_operation_tool_schemas",
    "execute_operation",
    "get_operation_catalog",
    "is_known_tool",
    "register_operation_mcp",
    "register_tool_catalog_callback",
]

_OPERATION_TRACER = get_tracer("kernel.operations")

_FRONT_MATTER_DELIMITER = "---"

# The lisa-content MCP server config, injected by the host at startup.
# When None, the operation catalog cannot be loaded.
_operation_mcp_config: MCPClientConfig | None = None
_get_tool_catalog_fn: Callable[[], dict[str, Any]] | None = None


class OperationValidationError(ValueError):
    """Raised when an operation folder or definition violates registry invariants."""


class OperationExecutionError(RuntimeError):
    """Raised when an operation cannot be resolved or executed."""


def is_known_tool(tool_id: str) -> bool:
    """Return True if the given ID corresponds to a registered internal tool."""
    from kernel.tools.registry import get_tool_catalog

    return tool_id in get_tool_catalog()


def register_tool_catalog_callback(get_tool_catalog_fn: Callable[[], dict[str, Any]]) -> None:
    """Inject the tool catalog callback used for operation validation and tool schemas.

    .. deprecated::
        Tools are now discovered from content packs via ``register_tool_mcp``.
        This callback is retained as a thin delegator for hosts that still
        register a hand-built tool catalog. The live tool catalog lives in
        ``kernel.tools.registry``; this function stashes the callback and
        re-validates the operation catalog against it.
    """
    global _get_tool_catalog_fn  # noqa: PLW0603
    _get_tool_catalog_fn = get_tool_catalog_fn
    validate_operation_catalog(_OPERATION_CATALOG)


def _resolve_tool_catalog() -> dict[str, Any]:
    """Return the effective tool catalog, preferring the MCP-loaded registry.

    Falls back to the legacy host callback if the tool registry is empty
    (e.g. a host that hasn't called ``register_tool_mcp`` yet).
    """
    from kernel.tools.registry import get_tool_catalog

    mcp_catalog = get_tool_catalog()
    if mcp_catalog:
        return mcp_catalog
    if _get_tool_catalog_fn is not None:
        return _get_tool_catalog_fn()
    return {}


def register_operation_mcp(config: MCPClientConfig) -> None:
    """Inject the ``lisa-catalog`` MCP server config for operation catalog loading.

    Rebuilds and re-validates ``_OPERATION_CATALOG`` immediately by calling
    ``list_operations`` on the server. The server does all filesystem access;
    the kernel never touches the filesystem directly.
    """
    global _operation_mcp_config, _OPERATION_CATALOG  # noqa: PLW0603
    _operation_mcp_config = config
    from kernel.context.mcp_catalog import MCPOperationCatalog

    catalog_loader = MCPOperationCatalog(config)
    _OPERATION_CATALOG = catalog_loader.get_operation_catalog()
    validate_operation_catalog(_OPERATION_CATALOG)


def _build_tool_schemas(tool_ids: list[str]) -> list[dict[str, Any]]:
    """Return filtered tool schemas for one operation from the effective tool catalog."""
    if not tool_ids:
        return []

    catalog = _resolve_tool_catalog()
    schemas: list[dict[str, Any]] = []
    for tool_id in tool_ids:
        definition = catalog.get(tool_id)
        if definition is None:
            continue
        schemas.append(
            build_strict_openai_function_tool(
                name=tool_id,
                description=definition.description,
                parameters_schema=definition.parameters_schema,
            )
        )
    return schemas


def _wrap_declarative_tool_executor(
    definition: OperationDefinition,
    execute_tool_fn: Any,
) -> Any:
    """Return a tool executor that only allows declared tools for prompt operations."""
    if definition.execution_kind != "prompt" or execute_tool_fn is None:
        return execute_tool_fn

    allowed_tool_ids = set(definition.tool_ids)

    async def _execute_allowed_tool(tool_id: str, ctx: Any) -> Any:
        if tool_id not in allowed_tool_ids:
            raise OperationExecutionError(
                f"Declarative operation '{definition.operation_id}' cannot call undeclared tool '{tool_id}'."
            )
        return await execute_tool_fn(tool_id, ctx)

    return _execute_allowed_tool


def validate_operation_catalog(catalog: dict[str, OperationDefinition]) -> None:
    """Validate the operation catalog for basic invariants."""
    tool_catalog = _resolve_tool_catalog() if _get_tool_catalog_fn is not None else None
    # Also validate against the MCP-loaded tool registry even when no legacy
    # callback is registered, so pack-contributed tools satisfy tool_id refs.
    if tool_catalog is None:
        from kernel.tools.registry import get_tool_catalog

        mcp_tools = get_tool_catalog()
        tool_catalog = mcp_tools if mcp_tools else None
    for operation_id, definition in catalog.items():
        if operation_id != definition.operation_id:
            raise OperationValidationError(
                "Operation registry key "
                f"'{operation_id}' does not match operation_id '{definition.operation_id}'."
            )
        if not definition.description.strip():
            raise OperationValidationError(
                f"Operation '{operation_id}' must have a non-empty description."
            )
        if definition.implementation_module is None:
            raise OperationValidationError(
                f"Operation '{operation_id}' has no implementation module."
            )
        if getattr(definition.implementation_module, "execute", None) is None:
            raise OperationValidationError(
                f"Operation '{operation_id}' implementation is missing execute(ctx)."
            )
        if definition.max_calls_per_turn < 0:
            raise OperationValidationError(
                "Operation "
                f"'{operation_id}' has invalid max_calls_per_turn={definition.max_calls_per_turn}."
            )
        if definition.timeout_seconds < 0:
            raise OperationValidationError(
                f"Operation '{operation_id}' has invalid timeout_seconds={definition.timeout_seconds}."
            )
        if definition.execution_kind == "prompt":
            if not definition.instruction_markdown_path.strip():
                raise OperationValidationError(
                    f"Declarative operation '{operation_id}' is missing instruction_markdown_path."
                )
            if not definition.instruction_body.strip():
                raise OperationValidationError(
                    f"Declarative operation '{operation_id}' must have a non-empty markdown body."
                )
            if definition.completion_max_tokens < 1:
                raise OperationValidationError(
                    f"Declarative operation '{operation_id}' has invalid completion_max_tokens={definition.completion_max_tokens}."
                )
        if tool_catalog is not None:
            for tool_id in definition.tool_ids:
                if tool_id not in tool_catalog:
                    raise OperationValidationError(
                        f"Operation '{operation_id}' references unknown tool '{tool_id}'."
                    )


_OPERATION_CATALOG: dict[str, OperationDefinition] = {}


def get_operation_catalog() -> dict[str, OperationDefinition]:
    """Return the internal operation catalog.

    The catalog is populated by ``register_operation_mcp()`` at startup. If
    it has not been registered, returns an empty dict — callers that need
    operations should ensure the host called ``register_operation_mcp()``.
    """
    return _OPERATION_CATALOG


def _is_operation_allowed_for_mode(
    *,
    operation_id: str,
    definition: OperationDefinition,
    mode_id: str,
) -> bool:
    """Return whether one operation can execute for the given mode.

    Permissions are sourced from two places:
    1) Per-operation ``mode_allowlist`` in colocated ``*/operations/**/*.yml`` files.
    2) Mode-owned operation bindings declared in loaded ``*/modes/*/_mode.md`` files.

    Mode-owned bindings are treated as authoritative for mode execution so
    stale operation-level allowlists do not break chat/action/workflow routing.
    """
    if not mode_id:
        return True

    if definition.mode_allowlist and mode_id in definition.mode_allowlist:
        return True

    # pylint: disable=import-outside-toplevel
    from kernel.ipsar_fsm import get_workflow_catalog

    mode = get_workflow_catalog().get(mode_id)
    if mode:
        mode_bound_operation_ids = set(
            mode.entry_operation_ids
            + mode.pre_turn_operation_ids
            + mode.post_turn_operation_ids
            + mode.allowed_operation_ids
        )
        if operation_id in mode_bound_operation_ids:
            return True

    if definition.mode_allowlist:
        return False
    return True


def _is_operation_bound_to_mode(
    *,
    operation_id: str,
    definition: OperationDefinition,
    mode_id: str,
) -> bool:
    """Return whether an operation is explicitly bound to the given mode.

    Stricter than :func:`_is_operation_allowed_for_mode`: only returns True
    when the operation is in the mode's bound operation IDs or the
    operation's own ``mode_allowlist``. The default fallback (operations
    with no ``mode_allowlist`` are allowed everywhere) is suppressed.

    Used when the mode's ``tools_authoritative`` flag is True.
    """
    if not mode_id:
        return True

    if definition.mode_allowlist and mode_id in definition.mode_allowlist:
        return True

    # pylint: disable=import-outside-toplevel
    from kernel.ipsar_fsm import get_workflow_catalog

    mode = get_workflow_catalog().get(mode_id)
    if mode:
        mode_bound_operation_ids = set(
            mode.entry_operation_ids
            + mode.pre_turn_operation_ids
            + mode.post_turn_operation_ids
            + mode.allowed_operation_ids
        )
        if operation_id in mode_bound_operation_ids:
            return True

    return False


def build_operation_tool_schemas(
    mode_id: str,
    *,
    tools_authoritative: bool = True,
) -> list[dict[str, Any]]:
    """Build OpenAI tool schemas for every operation allowed in the given mode.

    Operations are advertised to the LLM directly from the in-process catalog
    (not through an MCP subprocess). Each schema is built with
    :func:`build_strict_openai_function_tool` so the model sees the
    operation's real ``Parameters`` Pydantic model — not an
    ``arguments_json`` wrapper.

    Returns an empty list if the catalog is empty or no operations are
    allowed for the mode.

    When ``tools_authoritative`` is True, only operations explicitly bound
    to the mode (via ``allowed_operation_ids``, ``entry_operation_ids``,
    ``pre_turn_operation_ids``, or ``post_turn_operation_ids``) or listed
    in the operation's own ``mode_allowlist`` are advertised. The default
    fallback (operations with no ``mode_allowlist`` are allowed everywhere)
    is suppressed.
    """
    catalog = get_operation_catalog()
    schemas: list[dict[str, Any]] = []
    for operation_id, definition in catalog.items():
        if tools_authoritative:
            if not _is_operation_bound_to_mode(
                operation_id=operation_id,
                definition=definition,
                mode_id=mode_id,
            ):
                continue
        elif not _is_operation_allowed_for_mode(
            operation_id=operation_id,
            definition=definition,
            mode_id=mode_id,
        ):
            continue
        if definition.parameters_schema is None:
            continue
        schemas.append(
            build_strict_openai_function_tool(
                name=operation_id,
                description=definition.description or f"Execute {operation_id}.",
                parameters_schema=definition.parameters_schema,
            )
        )
    return schemas


async def execute_operation(
    operation_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    """Execute one internal operation by ID and return its structured result."""
    definition = _OPERATION_CATALOG.get(operation_id)
    if definition is None:
        raise OperationExecutionError(f"Unknown operation '{operation_id}'.")
    if not _is_operation_allowed_for_mode(
        operation_id=operation_id,
        definition=definition,
        mode_id=ctx.mode_id,
    ):
        raise OperationExecutionError(
            f"Operation '{operation_id}' is not allowed for mode '{ctx.mode_id}'."
        )

    # Inject preferred model from definition
    ctx.preferred_model_id = definition.preferred_model_id
    ctx.tool_schemas = _build_tool_schemas(definition.tool_ids)
    ctx.execute_tool_fn = _wrap_declarative_tool_executor(definition, ctx.execute_tool_fn)
    module = definition.implementation_module
    if module is None:
        raise OperationExecutionError(f"Operation '{operation_id}' has no implementation module.")

    execute_fn = getattr(module, "execute", None)
    if execute_fn is None:
        raise OperationExecutionError(f"Operation '{operation_id}' is missing async execute(ctx).")

    with _OPERATION_TRACER.start_as_current_span("operation.execute") as span:
        set_span_attributes(
            span,
            {
                "lisa.target_type": "operation",
                "lisa.target_id": operation_id,
                "lisa.timeout_seconds": definition.timeout_seconds,
                "lisa.mode_id": ctx.mode_id,
            },
        )
        t0 = time.perf_counter()
        status: Literal["success", "error", "timeout"] = "success"
        err_msg = None
        try:
            if definition.timeout_seconds > 0:
                result = await asyncio.wait_for(execute_fn(ctx), timeout=definition.timeout_seconds)
            else:
                result = await execute_fn(ctx)
        except TimeoutError as exc:
            logger.error(
                "operation '%s' timed out after %ss",
                operation_id,
                definition.timeout_seconds,
            )
            status = "timeout"
            err_msg = f"Timed out after {definition.timeout_seconds}s"
            mark_span_failed(span, exc)
            raise OperationExecutionError(f"Operation '{operation_id}' timed out") from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("operation '%s' failed: %s", operation_id, exc)
            status = "error"
            err_msg = f"{type(exc).__name__}: {exc}"
            mark_span_failed(span, exc)
            raise OperationExecutionError(f"Operation '{operation_id}' failed: {err_msg}") from exc
        finally:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            set_span_attributes(
                span,
                {
                    "lisa.status": status,
                    "lisa.duration_ms": duration_ms,
                    "lisa.error_message": err_msg,
                },
            )
            if ctx.event_bus:
                event = ToolExecutionTelemetry(
                    target_id=operation_id,
                    target_type="operation",
                    status=status,
                    duration_ms=duration_ms,
                    error_message=err_msg,
                )
                asyncio.create_task(ctx.event_bus.publish(event))

    if isinstance(result, OperationResult):
        return result
    if isinstance(result, str):
        return OperationResult(output_text=result)
    raise OperationExecutionError(
        f"Operation '{operation_id}' returned unsupported result type {type(result).__name__}."
    )
