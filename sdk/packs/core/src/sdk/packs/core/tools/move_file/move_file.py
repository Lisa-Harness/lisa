"""Approval-gated wrapper for the optional ``storage.move_file`` capability."""

from __future__ import annotations

import json

from kernel.models import ToolExecutionContext, ToolResult
from pydantic import BaseModel, Field, ValidationError

__all__ = ["MOVE_TOOL_ID", "Parameters", "execute", "preflight"]

MOVE_TOOL_ID = "storage.move_file"


class Parameters(BaseModel):
    """Structured move request."""

    source: str = Field(..., min_length=1, description="Existing sandbox-relative file path.")
    destination: str = Field(
        ..., min_length=1, description="New sandbox-relative path; must not already exist."
    )
    expected_sha256: str = Field(
        default="",
        pattern=r"^(?:[0-9a-fA-F]{64})?$",
        description="Optional content hash reported by stat_file.",
    )
    create_parents: bool = Field(
        default=False,
        description="Create missing destination parent directories before moving.",
    )


def _tool_result_error(message: str, ctx: ToolExecutionContext) -> ToolResult:
    return ToolResult(tool_id=ctx.tool_id or "move_file", error=message)


def _validated(ctx: ToolExecutionContext) -> Parameters | ToolResult:
    try:
        return Parameters.model_validate(ctx.parameters or {})
    except ValidationError as exc:
        return _tool_result_error(f"Invalid move_file parameters: {exc}", ctx)


async def preflight(ctx: ToolExecutionContext) -> ToolResult | None:
    """Reject requests with no storage MCP seam before pausing for approval."""
    validated = _validated(ctx)
    if isinstance(validated, ToolResult):
        return validated
    if ctx.mcp_tool_call is None:
        return _tool_result_error(
            f"move_file requires the storage capability ({MOVE_TOOL_ID}), but no MCP "
            "tool pool is attached to this tool context.",
            ctx,
        )
    return None


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Move the requested file through the session's storage capability."""
    parsed = _validated(ctx)
    if isinstance(parsed, ToolResult):
        return parsed
    if ctx.mcp_tool_call is None:
        return _tool_result_error(
            f"move_file requires the storage capability ({MOVE_TOOL_ID}), but no MCP "
            "tool pool is attached to this tool context.",
            ctx,
        )

    result = await ctx.mcp_tool_call(
        MOVE_TOOL_ID,
        parsed.model_dump(mode="json"),
    )
    if result.error:
        return _tool_result_error(result.error, ctx)
    try:
        payload = json.loads(result.output_text)
    except (json.JSONDecodeError, TypeError):
        return _tool_result_error(
            f"{MOVE_TOOL_ID} returned an unexpected payload: {result.output_text[:200]!r}",
            ctx,
        )
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return _tool_result_error(
            f"{MOVE_TOOL_ID} returned an unexpected payload: {result.output_text[:200]!r}",
            ctx,
        )
    return ToolResult(
        tool_id=ctx.tool_id or "move_file",
        output_text=result.output_text,
        output_json=payload,
    )
