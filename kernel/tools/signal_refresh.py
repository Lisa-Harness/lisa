"""Signal refresh tool.

Refreshes one or more external signal connectors through the shared heartbeat
execution path and returns the normalized UnifiedState payload.
"""

from __future__ import annotations

from kernel.intake.connector_discovery import discover_installed_signal_connectors
from kernel.models import ToolExecutionContext, ToolResult

TOOL_ID = "signal_refresh"
TOOL_DESCRIPTION = "Refresh external signal connectors and return the hot UnifiedState snapshot."


def _default_connector_ids() -> list[str]:
    """Return the installed signal connector IDs used when callers omit a target list."""

    return list(discover_installed_signal_connectors())


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Refresh connectors and return the resulting unified state payload.

    Args:
        ctx: Tool execution request. `parameters.connector_ids` may specify a
            list of connector IDs; defaults to all currently supported connectors.
            The app side must inject `ctx.state_provider` (a callable matching
            `refresh_current_state_from_connectors`).

    Returns:
        ToolResult carrying the refreshed UnifiedState as JSON plus the final
        connector target list that was used.
    """
    connector_ids = ctx.parameters.get("connector_ids") or _default_connector_ids()
    normalized_connector_ids = [str(connector_id) for connector_id in connector_ids]
    run_mode = str(ctx.parameters.get("run_mode", "on_demand"))

    if ctx.state_provider is None:
        return ToolResult(
            tool_id=TOOL_ID,
            error="Signal refresh is unavailable: state_provider not injected.",
        )

    refreshed_state = await ctx.state_provider(
        connector_ids=normalized_connector_ids,
        run_mode=run_mode,
    )
    return ToolResult(
        tool_id=TOOL_ID,
        output_json={
            "connector_ids": normalized_connector_ids,
            "state": refreshed_state.model_dump(mode="json"),
        },
    )
