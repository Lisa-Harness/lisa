"""Surf session flush tool.

Deletes the persisted shared surf browser session so subsequent navigations
start from a clean browser context.

Called by:
    - SDK-owned surf browse operations that need to reset the shared session
"""

from __future__ import annotations

from kernel.models import ToolExecutionContext, ToolResult

TOOL_ID = "surf_session_flush"
TOOL_DESCRIPTION = "Delete the persisted shared surf browser session state."


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Flush the persisted surf session state.

    Args:
        ctx: Tool execution request. The host side must inject
            `ctx.browser_automation` with a `flush()` method.

    Returns:
        ToolResult with `output_json.deleted` indicating whether session state
        existed and was removed.
    """
    if ctx.browser_automation is None:
        return ToolResult(
            tool_id=TOOL_ID,
            error="Surf session flush is unavailable: browser_automation not injected.",
        )

    deleted = ctx.browser_automation.flush()
    return ToolResult(
        tool_id=TOOL_ID,
        output_json={"deleted": deleted},
    )
