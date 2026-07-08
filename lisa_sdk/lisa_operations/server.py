"""Lisa Operations MCP server - dynamic operation registry exposure.

This server exposes one MCP tool per kernel operation. Tools are registered
at import time from ``get_operation_catalog()`` so clients can discover and
invoke operations directly by ``operation_id``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from kernel.models import OperationExecutionContext, OperationResult
from kernel.operations.registry import execute_operation, get_operation_catalog

logger = logging.getLogger(__name__)

__all__ = ["build_tool_catalog", "call_operation", "main", "mcp"]


mcp = FastMCP("Lisa Operations Server")


def _schema_to_json(schema: type[Any] | None) -> dict[str, Any]:
    """Convert a Pydantic model class to JSON schema, or return empty schema."""
    if schema is None:
        return {"type": "object", "properties": {}}
    try:
        return schema.model_json_schema()
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
    memory_scope = str(arguments.get("memory_scope", "personal"))
    if memory_scope not in {"personal", "shared", "channel"}:
        memory_scope = "personal"
    ctx = OperationExecutionContext(
        operation_id=operation_id,
        params=dict(arguments),
        query=str(arguments.get("query", "")),
        workspace_id=str(arguments.get("workspace_id", "default")),
        user_slug=str(arguments.get("user_slug", "default")),
        memory_scope=cast(Any, memory_scope),
        surface_id=str(arguments.get("surface_id", "")),
    )
    return await execute_operation(operation_id, ctx)


def _register_operation_tools() -> None:
    """Register one MCP tool per operation from the kernel catalog."""
    catalog = build_tool_catalog()
    for operation_id, info in catalog.items():
        description = info["description"]

        def _build_operation_tool(op_id: str) -> Any:
            async def _operation_tool(arguments_json: str = "{}") -> str:
                try:
                    kwargs = json.loads(arguments_json) if arguments_json else {}
                except json.JSONDecodeError as exc:
                    return f"Error executing '{op_id}': invalid JSON arguments - {exc}"
                if not isinstance(kwargs, dict):
                    return f"Error executing '{op_id}': arguments_json must decode to an object"
                try:
                    result = await call_operation(op_id, kwargs)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    return f"Error executing '{op_id}': {exc}"
                if result.output_json:
                    payload = json.dumps(result.output_json, default=str)
                    return "\n".join(part for part in [result.output_text, payload] if part)
                return result.output_text

            _operation_tool.__name__ = f"op_{op_id.replace('-', '_').replace('.', '_')}"
            _operation_tool.__doc__ = f"Execute operation '{op_id}'."
            return _operation_tool

        mcp.add_tool(
            _build_operation_tool(operation_id),
            name=operation_id,
            description=description,
        )


_register_operation_tools()


def main() -> None:
    """Entry point for the lisa-operations MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper. No
    path config is needed today, but ``--data-dir`` is required for
    consistency across all first-party servers.
    """
    from lisa_sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-operations")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-operations")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-operations starting (data_dir=%s)", data_dir)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
