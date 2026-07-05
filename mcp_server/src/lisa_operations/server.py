"""Lisa Operations MCP server — dynamic operation registry exposure.

Dynamically exposes every operation in the kernel's operation registry as an
MCP tool. On startup, calls ``get_operation_catalog()`` to discover all
operations, then registers one MCP tool per operation with:

- ``name``: ``operation_id``
- ``description``: from the operation's metadata
- ``inputSchema``: from the operation's Pydantic ``Parameters`` model (if any)

On ``call_tool``, executes the operation via ``execute_operation()`` and returns
the result text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel.models import OperationExecutionContext, OperationResult
from kernel.operations.registry import execute_operation, get_operation_catalog

logger = logging.getLogger(__name__)

__all__ = ["mcp", "main", "build_tool_catalog", "call_operation"]


mcp = FastMCP("Lisa Operations Server")


def _schema_to_json(schema: type[Any] | None) -> dict[str, Any]:
    """Convert a Pydantic model class to a JSON schema dict, or return empty."""
    if schema is None:
        return {"type": "object", "properties": {}}
    try:
        return schema.model_json_schema()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to serialize schema for operation: %s", exc)
        return {"type": "object", "properties": {}}


def build_tool_catalog() -> dict[str, dict[str, Any]]:
    """Return ``{operation_id: {description, input_schema}}`` for every operation."""
    catalog = get_operation_catalog()
    out: dict[str, dict[str, Any]] = {}
    for op_id, op_def in catalog.items():
        out[op_id] = {
            "description": op_def.description or f"Execute {op_id}",
            "input_schema": _schema_to_json(op_def.parameters_schema),
        }
    return out


async def call_operation(operation_id: str, arguments: dict[str, Any]) -> OperationResult:
    """Execute one operation by ID with a minimal execution context."""
    ctx = OperationExecutionContext(
        operation_id=operation_id,
        params=dict(arguments),
    )
    return await execute_operation(operation_id, ctx)


@mcp.tool()
async def list_operations() -> str:
    """List all available kernel operations and their descriptions."""
    catalog = build_tool_catalog()
    lines = [f"- {op_id}: {info['description']}" for op_id, info in sorted(catalog.items())]
    return "\n".join(lines) if lines else "No operations registered."


@mcp.tool()
async def run_operation(operation_id: str, arguments: str = "{}") -> str:
    """Execute a kernel operation by ID.

    ``arguments`` is a JSON string of keyword arguments passed to the operation.
    Returns the operation's ``output_text`` (and ``output_json`` if non-empty).
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments — {exc}"
    try:
        result = await call_operation(operation_id, args)
    except Exception as exc:  # noqa: BLE001
        return f"Error executing '{operation_id}': {exc}"
    parts = [result.output_text]
    if result.output_json:
        parts.append(json.dumps(result.output_json, default=str))
    return "\n".join(part for part in parts if part)


def main() -> None:
    """Entry point for the lisa-operations MCP server (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()