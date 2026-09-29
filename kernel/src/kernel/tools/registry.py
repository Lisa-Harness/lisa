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
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from kernel.approval import (
    ApprovalIdentity,
    ApprovalRequired,
    build_approval_invocation,
)
from kernel.capability_binding import CapabilityRef, ExecutionFloorView
from kernel.models import (
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionTelemetry,
    ToolResult,
)
from kernel.telemetry import get_tracer, mark_span_failed, set_span_attributes
from kernel.tools.mcp_client import MCPClientConfig

logger = logging.getLogger(__name__)
EMPTY_PACK_CONFIGURATION = MappingProxyType({})

#: Strong references to in-flight telemetry publish tasks (RUF006). Tasks are
#: fire-and-forget telemetry; the set keeps them garbage-collectable only
#: after completion and is pruned on each publish.
_PENDING_TELEMETRY_TASKS: set[asyncio.Task[None]] = set()

__all__ = [
    "execute_tool",
    "get_tool_catalog",
    "project_execution_floor",
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


async def _execute_under_tracer(
    tool_id: str,
    definition: ToolDefinition,
    execute_fn: Any,
    ctx: ToolExecutionContext,
) -> Any:
    """Run ``execute_fn(ctx)`` under the tool span with timeout + telemetry.

    The span records the dispatch outcome, and a fire-and-forget telemetry
    event is published on the EventBus in the ``finally`` block. Telemetry
    is strictly best-effort: a publish failure must never replace the tool's
    own result/exception, so the coroutine is wrapped before scheduling —
    a synchronous raise here would otherwise surface from this ``finally``
    and mask the real dispatch outcome (2026-09-08 fetch_page incident).
    """
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
                _publish_tool_telemetry(ctx, tool_id, status, duration_ms, err_msg)
    return result


def _publish_tool_telemetry(
    ctx: ToolExecutionContext,
    tool_id: str,
    status: Literal["success", "error", "timeout"],
    duration_ms: int,
    err_msg: str | None,
) -> None:
    """Schedule the fire-and-forget tool telemetry publish on the EventBus."""
    event = ToolExecutionTelemetry(
        target_id=tool_id,
        target_type="tool",
        status=status,
        duration_ms=duration_ms,
        error_message=err_msg,
    )
    event_bus = ctx.event_bus
    if event_bus is None:
        return
    try:
        task = asyncio.create_task(
            event_bus.publish("tool.telemetry", event.model_dump(mode="json"))
        )
    except Exception:  # pylint: disable=broad-except
        logger.debug("tool telemetry publish failed for %s", tool_id, exc_info=True)
        return
    _PENDING_TELEMETRY_TASKS.add(task)
    task.add_done_callback(_PENDING_TELEMETRY_TASKS.discard)


def _coerce_tool_result(tool_id: str, result: Any) -> ToolResult:
    """Wrap a raw tool outcome into a :class:`ToolResult`."""
    if isinstance(result, ToolResult):
        return result
    if isinstance(result, str):
        return ToolResult(tool_id=tool_id, output_text=result)
    raise ToolExecutionError(
        f"Tool '{tool_id}' returned unsupported result type {type(result).__name__}."
    )


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
    provider = ctx.capability_binding_provider
    _resolve_execution_configuration(definition, ctx)
    ctx.capability_binding_provider = None
    if provider is not None:
        ctx.execution_floor_for = lambda child_id: project_execution_floor(provider, child_id)

        async def dispatch_child(child_id: str, parameters: Mapping[str, Any]) -> ToolResult:
            child_ctx = ctx.model_copy(
                update={
                    "tool_id": child_id,
                    "parameters": dict(parameters),
                    "pack_config": EMPTY_PACK_CONFIGURATION,
                    "capability_binding_provider": provider,
                    "approval_request_id": "",
                }
            )
            return await execute_tool(child_id, child_ctx)

        ctx.delegated_tool_dispatch = dispatch_child
    _validate_parameters(definition, ctx)
    await _require_approval(definition, ctx)
    module = definition.implementation_module
    if module is None:
        raise ToolExecutionError(f"Tool '{tool_id}' has no implementation module.")

    execute_fn = getattr(module, "execute", None)
    if execute_fn is None:
        raise ToolExecutionError(f"Tool '{tool_id}' is missing async execute(ctx).")

    result = await _execute_under_tracer(tool_id, definition, execute_fn, ctx)
    return _coerce_tool_result(tool_id, result)


def project_execution_floor(provider: Any, tool_id: str) -> ExecutionFloorView | None:
    """Expose only a child's execution posture, not its pack configuration."""
    definition = _TOOL_CATALOG.get(tool_id)
    if definition is None or not definition.owner_id:
        return None
    binding = provider.binding_for(CapabilityRef("tool", tool_id, definition.owner_id))
    if binding is None or binding.owner_id != definition.owner_id:
        return None
    sandbox = binding.configuration.get("sandbox")
    floor = sandbox.get("floor") if isinstance(sandbox, Mapping) else None
    if not isinstance(floor, Mapping):
        return None
    paths = floor.get("paths")
    return ExecutionFloorView(
        execution=floor.get("execution") if isinstance(floor.get("execution"), str) else None,
        persistence=floor.get("persistence") if isinstance(floor.get("persistence"), str) else None,
        paths=MappingProxyType(
            {str(path): str(access) for path, access in paths.items()}
        )
        if isinstance(paths, Mapping)
        else MappingProxyType({}),
    )


def _resolve_execution_configuration(
    definition: ToolDefinition,
    ctx: ToolExecutionContext,
) -> None:
    """Attach only the target pack's immutable session configuration."""

    provider = ctx.capability_binding_provider
    owner_id = definition.owner_id
    if not owner_id or provider is None:
        ctx.pack_config = EMPTY_PACK_CONFIGURATION
    else:
        binding = provider.binding_for(CapabilityRef("tool", definition.tool_id, owner_id))
        if binding is None:
            raise ToolExecutionError(
                f"Tool '{definition.tool_id}' has no active capability binding."
            )
        if binding.owner_id != owner_id:
            raise ToolExecutionError(
                f"Tool '{definition.tool_id}' binding owner does not match its catalog owner."
            )
        ctx.pack_config = binding.configuration
    # Propagate the catalog-declared streaming posture onto the call context
    # so tools don't have to re-resolve the catalog to know whether they may
    # publish incremental output. The host subscribes only for tools that
    # declare ``stream_output: true``, so the flag here mirrors the
    # declaration and hosts that never wired a subscriber simply receive
    # nothing (the publish path is best-effort on the tool side).
    ctx.stream_output = definition.stream_output


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
        pack_id=definition.owner_id,
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
