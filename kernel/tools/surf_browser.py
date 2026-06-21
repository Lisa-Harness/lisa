"""Surf browser tool.

Runs one browser navigation through the shared surf runtime, persists trace
artifacts, and returns raw trace data plus the primary screenshot bytes.

Called by:
    - app.operations.surf_browser operation
"""

from __future__ import annotations

from kernel.config import settings
from kernel.models import ToolExecutionContext, ToolResult

TOOL_ID = "surf_browser"
TOOL_DESCRIPTION = "Navigate to one public URL through the shared surf browser runtime."


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Run one surf navigation and return raw trace data.

    Args:
        ctx: Tool execution request. `query` must be the target URL text.
            The app side must inject `ctx.browser_automation` (a callable
            matching `surf_url`) and `ctx.blob_io` (with `write_surf_trace`).

    Returns:
        ToolResult with the serialized SurfTrace in `output_json.trace`, the
        structured page snapshot in `output_json.page`, and the page screenshot
        bytes in `extra.screenshot_bytes` when available.
    """
    if not settings.surf_enabled:
        return ToolResult(
            tool_id=TOOL_ID,
            error="Surf mode is disabled right now. Set SURF_ENABLED=true to enable it.",
        )

    if ctx.browser_automation is None:
        return ToolResult(
            tool_id=TOOL_ID,
            error="Surf browser is unavailable: browser_automation not injected.",
        )

    try:
        result = await ctx.browser_automation.surf_url(ctx.query)
    except ValueError as exc:
        return ToolResult(tool_id=TOOL_ID, error=f"Surf blocked: {exc}")

    screenshot_bytes = result.screenshots.get("page.png") or None
    if ctx.blob_io is not None:
        try:
            ctx.blob_io.write_surf_trace(
                result.trace.model_dump_json(indent=2),
                result.screenshots,
                result.artifact_prefix,
            )
        except Exception:  # pylint: disable=broad-except
            result.trace.screenshot_blob_path = ""

    return ToolResult(
        tool_id=TOOL_ID,
        output_json={
            "trace": result.trace.model_dump(mode="json"),
            "page": result.page_capture.model_dump(mode="json") if result.page_capture else {},
            "artifact_prefix": result.artifact_prefix,
        },
        extra={"screenshot_bytes": screenshot_bytes},
    )
