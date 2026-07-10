"""Registry for internal tool metadata discovered from content packs.

Tools are the capability layer (distinct from operations, which are the
behavior layer). Like operations, tools are discovered from ``tools/``
subdirectories in content packs via the ``lisa-content`` MCP server and
dispatched in-process through the imported ``implementation_module``.

The kernel never touches the filesystem directly. The ``lisa-content`` server
scans ``tools/`` subdirectories in all discovered content roots, parses
``.tool.yml`` metadata, and returns the catalog as JSON. The kernel imports
the implementation module locally using the ``module_path`` field.

Called by: kernel.turn_runner (tool dispatch in the model+tool loop)
           kernel.operations.registry (tool-id validation for operations)
"""

from __future__ import annotations

# pylint: disable=global-statement
import asyncio
import logging
import time
from typing import Any, Literal

from kernel.models import (
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
    ToolExecutionTelemetry,
)
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
    """Inject the ``lisa-content`` MCP server config for tool catalog loading.

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
    """Validate the tool catalog for basic invariants."""
    for tool_id, definition in catalog.items():
        if tool_id != definition.tool_id:
            raise ToolValidationError(
                f"Tool registry key '{tool_id}' does not match tool_id '{definition.tool_id}'."
            )
        if not definition.description.strip():
            raise ToolValidationError(
                f"Tool '{tool_id}' must have a non-empty description."
            )
        if definition.implementation_module is None:
            raise ToolValidationError(
                f"Tool '{tool_id}' has no implementation module."
            )
        if getattr(definition.implementation_module, "execute", None) is None:
            raise ToolValidationError(
                f"Tool '{tool_id}' implementation is missing execute(ctx)."
            )
        if definition.max_calls_per_turn < 0:
            raise ToolValidationError(
                f"Tool '{tool_id}' has invalid max_calls_per_turn={definition.max_calls_per_turn}."
            )
        if definition.timeout_seconds < 0:
            raise ToolValidationError(
                f"Tool '{tool_id}' has invalid timeout_seconds={definition.timeout_seconds}."
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