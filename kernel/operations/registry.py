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
from typing import Any, Literal

from kernel.approval import (
    ApprovalIdentity,
    ApprovalRequired,
    build_approval_invocation,
)
from kernel.models import (
    OperationDefinition,
    OperationExecutionContext,
    OperationResult,
    ToolExecutionTelemetry,
)
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.pack_configuration import EMPTY_PACK_CONFIGURATION
from kernel.telemetry import get_tracer, mark_span_failed, set_span_attributes
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)

__all__ = [
    "build_operation_tool_schemas",
    "execute_operation",
    "get_operation_catalog",
    "is_known_tool",
    "register_operation_mcp",
]

_OPERATION_TRACER = get_tracer("kernel.operations")

_FRONT_MATTER_DELIMITER = "---"

# The lisa-content MCP server config, injected by the host at startup.
# When None, the operation catalog cannot be loaded.
_operation_mcp_config: MCPClientConfig | None = None


class OperationValidationError(ValueError):
    """Raised when an operation folder or definition violates registry invariants."""


class OperationExecutionError(RuntimeError):
    """Raised when an operation cannot be resolved or executed."""


def is_known_tool(tool_id: str) -> bool:
    """Return True if the given ID corresponds to a registered internal tool."""
    from kernel.tools.registry import get_tool_catalog

    return tool_id in get_tool_catalog()


def _resolve_tool_catalog() -> dict[str, Any]:
    """Return the effective tool catalog from the MCP-loaded registry."""
    from kernel.tools.registry import get_tool_catalog

    return get_tool_catalog()


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
    """Validate the operation catalog for basic invariants.

    Runtime operations whose implementation module cannot be imported (e.g.
    the owning pack's Python package is not installed) are dropped from the
    catalog with a warning rather than raising. This lets the rest of the
    catalog validate and function normally when an external pack's content
    root is present but its code isn't importable. Prompt operations that
    fail to build their ``DeclarativeOperationModule`` still raise — that
    indicates a genuine configuration error, not a missing install.
    """
    tool_catalog = _resolve_tool_catalog() or None
    # Collect unimportable runtime operations to drop after iteration.
    unimportable: list[str] = []
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
            if definition.execution_kind == "prompt":
                raise OperationValidationError(
                    f"Operation '{operation_id}' has no implementation module."
                )
            logger.warning(
                "Operation '%s' implementation module could not be imported; "
                "dropping from catalog (install the owning pack to enable it).",
                operation_id,
            )
            unimportable.append(operation_id)
            continue
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
    # Drop unimportable runtime operations after iteration (dict mutation
    # during iteration is not safe).
    for operation_id in unimportable:
        del catalog[operation_id]


_PROVIDER_PREFIX = "lisa-pack-"


def _provider_slug(raw: str) -> str:
    """Normalize a provider id for same-pack comparison.

    Strips the ``lisa-pack-``/``lisa_pack-`` distribution prefix and lower-cases
    so that a capability's inferred provider id (e.g. ``coding``, from the
    pack directory name) matches the mode's ``mode_provider_id``
    (e.g. ``lisa-pack-coding``, the full distribution name). Without this,
    every same-pack reference would be misflagged as cross-pack because the
    SDK infers provider ids from directory names while ``mode_provider_id``
    carries the full package name.
    """
    value = (raw or "").strip().lower().replace("_", "-")
    if value.startswith(_PROVIDER_PREFIX):
        value = value[len(_PROVIDER_PREFIX) :]
    return value


def validate_export_boundaries(
    operation_catalog: dict[str, OperationDefinition] | None = None,
) -> None:
    """Reject cross-pack references to non-exported capabilities.

    A capability (operation or tool) owned by one pack may be referenced by a
    mode in another pack only when the capability's ``exportable`` flag is
    True. Otherwise the capability is private to its owning pack. This is a
    generic rule over ``provider_id`` and ``exportable`` — the kernel never
    names capability ids.

    Iterates every mode in the workflow catalog and checks each id in its
    ``allowed_operation_ids``, ``allowed_tool_ids``, and lifecycle bindings
    (``entry_operation_ids`` / ``pre_turn_operation_ids`` /
    ``post_turn_operation_ids``) against the owning capability's
    ``exportable`` flag. A reference is allowed when the capability's
    provider matches the mode's provider, or the capability is exported, or
    the capability is not in the catalog (deferred to the existing
    "unknown tool" check).

    Args:
        operation_catalog: Optional pre-resolved operation catalog. When
            omitted, ``get_operation_catalog()`` is used. The tool catalog is
            always resolved via the tool registry.

    Raises:
        OperationValidationError: If a mode references a non-exported
            capability owned by a different pack.
    """
    # pylint: disable=import-outside-toplevel
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.tools.registry import get_tool_catalog

    ops = operation_catalog if operation_catalog is not None else get_operation_catalog()
    tools = get_tool_catalog() or {}
    try:
        modes = get_workflow_catalog()
    except RuntimeError:
        # Catalog not loaded yet (e.g. during early unit tests). Nothing to
        # validate; the check runs again once the workflow catalog is live.
        return

    for mode in modes.values():
        mode_provider = _provider_slug(mode.mode_provider_id)
        referenced_op_ids = set(
            mode.allowed_operation_ids
            + mode.entry_operation_ids
            + mode.pre_turn_operation_ids
            + mode.post_turn_operation_ids
        )
        # Loud failure: a mode with an unresolved ``mode_provider_id`` (author
        # omitted it AND SDK content-root inference returned "") would otherwise
        # silently flag *every* referenced capability as cross-pack, because
        # any non-empty ``op_provider != mode_provider`` is True. Fail fast with
        # a clear startup error instead of producing wrong export-boundary results.
        has_any_reference = bool(referenced_op_ids or mode.allowed_tool_ids)
        if has_any_reference and not mode_provider:
            raise OperationValidationError(
                f"Mode '{mode.mode_id}' has an empty 'mode_provider_id' but "
                "references capabilities. Set `mode_provider_id` in the mode's "
                "front matter, or ensure the SDK content-root path allows "
                "provider-id inference (pack directory name)."
            )
        for op_id in referenced_op_ids:
            op = ops.get(op_id)
            if op is None:
                continue  # unknown-id check is owned by validate_operation_catalog
            op_provider = _provider_slug(op.effective_provider_id())
            if op_provider and op_provider != mode_provider and not op.exportable:
                raise OperationValidationError(
                    f"Mode '{mode.mode_id}' (provider '{mode_provider}') "
                    f"references operation '{op_id}' (provider '{op_provider}') "
                    "which is not exported. Set `exportable: true` on the "
                    "operation or remove the cross-pack reference."
                )
        for tool_id in mode.allowed_tool_ids:
            tool = tools.get(tool_id)
            if tool is None:
                continue
            tool_provider = _provider_slug(tool.effective_provider_id())
            if (
                tool_provider
                and tool_provider != mode_provider
                and not getattr(tool, "exportable", False)
            ):
                raise OperationValidationError(
                    f"Mode '{mode.mode_id}' (provider '{mode_provider}') "
                    f"references tool '{tool_id}' (provider '{tool_provider}') "
                    "which is not exported. Set `exportable: true` on the "
                    "tool or remove the cross-pack reference."
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

    ctx.operation_id = operation_id
    _resolve_execution_configuration(definition, ctx)
    _validate_parameters(definition, ctx)
    await _require_approval(definition, ctx)

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
        except ApprovalRequired:
            raise
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


def _resolve_execution_configuration(
    definition: OperationDefinition,
    ctx: OperationExecutionContext,
) -> None:
    """Attach only the target pack's immutable session configuration."""

    resolver = ctx.pack_configuration_resolver
    pack_key = definition.effective_provider_id()
    if not pack_key or resolver is None:
        ctx.pack_config = EMPTY_PACK_CONFIGURATION
        return
    ctx.pack_config = resolver.for_pack(pack_key)


def _validate_parameters(
    definition: OperationDefinition,
    ctx: OperationExecutionContext,
) -> None:
    """Canonicalize structured inputs before execution and policy checks."""

    if definition.parameters_schema is None:
        return
    try:
        parsed = definition.parameters_schema.model_validate(ctx.params)
    except Exception as exc:  # pylint: disable=broad-except
        raise OperationExecutionError(
            f"Operation '{definition.operation_id}' received invalid parameters: {exc}"
        ) from exc
    ctx.params = parsed.model_dump(mode="json")


async def _require_approval(
    definition: OperationDefinition,
    ctx: OperationExecutionContext,
) -> None:
    """Raise a resumable interruption when an effect requires approval."""

    if not definition.approval_required:
        return
    service = ctx.approval_service
    if service is None:
        raise OperationExecutionError(
            f"Operation '{definition.operation_id}' requires an approval service."
        )
    identity = ApprovalIdentity(
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        surface_type=ctx.surface_type,
        surface_id=ctx.surface_id,
        memory_scope=ctx.memory_scope,
        mode_id=ctx.mode_id,
        session_plan_digest=ctx.session_plan_digest,
    )
    invocation = build_approval_invocation(
        target_kind="operation",
        target_id=definition.operation_id,
        pack_id=definition.pack_id,
        effect=definition.effect,
        parameters=ctx.params,
        identity=identity,
        pack_config=ctx.pack_config,
        safe_preview=definition.approval_preview,
    )
    if ctx.approval_request_id:
        receipt = await service.consume(ctx.approval_request_id, invocation)
        if receipt is None:
            raise OperationExecutionError(
                f"Operation '{definition.operation_id}' has no matching usable approval receipt."
            )
        return
    pending = await service.request(invocation)
    raise ApprovalRequired(pending)
