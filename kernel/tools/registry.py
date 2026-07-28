"""Registry for internal tool metadata discovered from content packs.

Tools are the capability layer (distinct from operations, which are the
behavior layer). Like operations, tools are discovered from ``tools/``
subdirectories in content packs via the ``lisa-content`` MCP server and
dispatched in-process through the imported ``implementation_module``.

The kernel never touches the filesystem directly. The ``lisa-content`` server
scans ``tools/`` subdirectories in all discovered content roots, parses
``.tool.yml`` metadata, and returns the catalog as JSON. The kernel imports
the implementation module locally using the ``module_path`` field.

Tool-id keying convention (two distinct populations, by design):

    * **Internal pack tools** (this registry) are keyed **bare** by the
      ``tool_id`` authored in YAML (e.g. ``web_search``). They have an
      in-process ``implementation_module`` and never spawn a subprocess.
    * **MCP-native tools** (``kernel.tools.mcp_client``) are keyed
      **namespaced** as ``server_id.tool`` (e.g.
      ``lisa-storage.read_model_manifest``). They run in subprocesses and
      have no kernel ``ToolDefinition``.

These two populations do not overlap. The MCP client pool owns the
namespace bridging for the MCP-native population (a bare-name index with a
precomputed fallback — see ``kernel.tools.mcp_client``). This registry and
``kernel.capability`` resolve against the bare-keyed internal population
only, so resolution never spawns a subprocess.

Called by: kernel.turn_runner (tool dispatch in the model+tool loop)
           kernel.operations.registry (tool-id validation for operations)
           kernel.capability (shared resolution layer)
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
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionTelemetry,
    ToolResult,
)
from kernel.pack_configuration import EMPTY_PACK_CONFIGURATION
from kernel.telemetry import get_tracer, mark_span_failed, set_span_attributes
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)

__all__ = [
    "execute_tool",
    "get_tool_catalog",
    "register_tool_mcp",
    "validate_tool_catalog",
]

_TOOL_TRACER = get_tracer("kernel.tools")

# The tool catalog, populated by ``register_tool_mcp()`` at startup.
# When empty, the operation registry's tool-id validation treats all
# tool_ids as unknown.
_TOOL_CATALOG: dict[str, ToolDefinition] = {}


class ToolValidationError(ValueError):
    """Raised when a tool definition violates registry invariants."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot be resolved or executed."""


def get_tool_catalog() -> dict[str, ToolDefinition]:
    """Return the internal tool catalog.

    The catalog is populated by ``register_tool_mcp()`` at startup. If it has
    not been registered, returns an empty dict — callers that need tools
    should ensure the host called ``register_tool_mcp()``.
    """
    return _TOOL_CATALOG


def register_tool_mcp(config: MCPClientConfig) -> None:
    """Inject the ``lisa-catalog`` MCP server config for tool catalog loading.

    Rebuilds and re-validates ``_TOOL_CATALOG`` immediately by calling
    ``list_tools`` on the server. The server does all filesystem access; the
    kernel never touches the filesystem directly.
    """
    global _TOOL_CATALOG  # noqa: PLW0603
    from kernel.context.mcp_catalog import MCPToolCatalog

    catalog_loader = MCPToolCatalog(config)
    _TOOL_CATALOG = catalog_loader.get_tool_catalog()
    validate_tool_catalog(_TOOL_CATALOG)


def validate_tool_catalog(catalog: dict[str, ToolDefinition]) -> None:
    """Validate the tool catalog for basic invariants.

    Tools whose implementation module cannot be imported (e.g. the owning
    pack's Python package is not installed) are dropped from the catalog
    with a warning rather than raising. This lets the rest of the catalog
    validate and function normally when an external pack's content root is
    present but its code isn't importable.
    """
    unimportable: list[str] = []
    for tool_id, definition in catalog.items():
        if tool_id != definition.tool_id:
            raise ToolValidationError(
                f"Tool registry key '{tool_id}' does not match tool_id '{definition.tool_id}'."
            )
        if not definition.description.strip():
            raise ToolValidationError(f"Tool '{tool_id}' must have a non-empty description.")
        if definition.implementation_module is None:
            logger.warning(
                "Tool '%s' implementation module could not be imported; "
                "dropping from catalog (install the owning pack to enable it).",
                tool_id,
            )
            unimportable.append(tool_id)
            continue
        if getattr(definition.implementation_module, "execute", None) is None:
            raise ToolValidationError(f"Tool '{tool_id}' implementation is missing execute(ctx).")
        if definition.max_calls_per_turn < 0:
            raise ToolValidationError(
                f"Tool '{tool_id}' has invalid max_calls_per_turn={definition.max_calls_per_turn}."
            )
        if definition.timeout_seconds < 0:
            raise ToolValidationError(
                f"Tool '{tool_id}' has invalid timeout_seconds={definition.timeout_seconds}."
            )
    for tool_id in unimportable:
        del catalog[tool_id]


async def execute_tool(
    tool_id: str,
    ctx: ToolExecutionContext,
) -> ToolResult:
    """Execute one internal tool by ID and return its structured result.

    Dispatches to the tool's ``implementation_module.execute(ctx)`` under a
    timeout, mirroring ``registry.execute_operation``. Tools are
    runtime-only; there is no declarative tool path.
    """
    definition = _TOOL_CATALOG.get(tool_id)
    if definition is None:
        raise ToolExecutionError(f"Unknown tool '{tool_id}'.")
    if definition.mode_allowlist is not None and ctx.mode_id not in definition.mode_allowlist:
        raise ToolExecutionError(f"Tool '{tool_id}' is not allowed for mode '{ctx.mode_id}'.")
    ctx.tool_id = tool_id
    _resolve_execution_configuration(definition, ctx)
    _validate_parameters(definition, ctx)
    await _require_approval(definition, ctx)
    module = definition.implementation_module
    if module is None:
        raise ToolExecutionError(f"Tool '{tool_id}' has no implementation module.")

    execute_fn = getattr(module, "execute", None)
    if execute_fn is None:
        raise ToolExecutionError(f"Tool '{tool_id}' is missing async execute(ctx).")

    with _TOOL_TRACER.start_as_current_span("tool.execute") as span:
        set_span_attributes(
            span,
            {
                "lisa.target_type": "tool",
                "lisa.target_id": tool_id,
                "lisa.timeout_seconds": definition.timeout_seconds,
                "lisa.mode_id": ctx.mode_id,
            },
        )
        t0 = time.perf_counter()
        status: Literal["success", "error", "timeout"] = "success"
        err_msg: str | None = None
        result: Any
        try:
            if definition.timeout_seconds > 0:
                result = await asyncio.wait_for(execute_fn(ctx), timeout=definition.timeout_seconds)
            else:
                result = await execute_fn(ctx)
        except TimeoutError as exc:
            logger.error(
                "tool '%s' timed out after %ss",
                tool_id,
                definition.timeout_seconds,
            )
            status = "timeout"
            err_msg = f"Timed out after {definition.timeout_seconds}s"
            mark_span_failed(span, exc)
            raise ToolExecutionError(f"Tool '{tool_id}' timed out") from exc
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("tool '%s' failed: %s", tool_id, exc)
            status = "error"
            err_msg = f"{type(exc).__name__}: {exc}"
            mark_span_failed(span, exc)
            raise ToolExecutionError(f"Tool '{tool_id}' failed: {err_msg}") from exc
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
                    target_id=tool_id,
                    target_type="tool",
                    status=status,
                    duration_ms=duration_ms,
                    error_message=err_msg,
                )
                asyncio.create_task(ctx.event_bus.publish(event))

    if isinstance(result, ToolResult):
        return result
    if isinstance(result, str):
        return ToolResult(tool_id=tool_id, output_text=result)
    raise ToolExecutionError(
        f"Tool '{tool_id}' returned unsupported result type {type(result).__name__}."
    )


def _resolve_execution_configuration(
    definition: ToolDefinition,
    ctx: ToolExecutionContext,
) -> None:
    """Attach only the target pack's immutable session configuration."""

    resolver = ctx.pack_configuration_resolver
    pack_key = definition.effective_provider_id()
    if not pack_key or resolver is None:
        ctx.pack_config = EMPTY_PACK_CONFIGURATION
        return
    ctx.pack_config = resolver.for_pack(pack_key)


def _validate_parameters(definition: ToolDefinition, ctx: ToolExecutionContext) -> None:
    """Canonicalize structured inputs before execution and policy checks."""

    if definition.parameters_schema is None:
        return
    try:
        parsed = definition.parameters_schema.model_validate(ctx.parameters)
    except Exception as exc:  # pylint: disable=broad-except
        raise ToolExecutionError(
            f"Tool '{definition.tool_id}' received invalid parameters: {exc}"
        ) from exc
    ctx.parameters = parsed.model_dump(mode="json")


async def _require_approval(
    definition: ToolDefinition,
    ctx: ToolExecutionContext,
) -> None:
    """Raise a resumable interruption when an effect requires approval."""

    if not definition.approval_required:
        return
    service = ctx.approval_service
    if service is None:
        raise ToolExecutionError(f"Tool '{definition.tool_id}' requires an approval service.")
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
        target_kind="tool",
        target_id=definition.tool_id,
        pack_id=definition.pack_id,
        effect=definition.effect,
        parameters=ctx.parameters,
        identity=identity,
        pack_config=ctx.pack_config,
        safe_preview=definition.approval_preview,
    )
    if ctx.approval_request_id:
        receipt = await service.consume(ctx.approval_request_id, invocation)
        if receipt is None:
            raise ToolExecutionError(
                f"Tool '{definition.tool_id}' has no matching usable approval receipt."
            )
        return
    pending = await service.request(invocation)
    raise ApprovalRequired(pending)
