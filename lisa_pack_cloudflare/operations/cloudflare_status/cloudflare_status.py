"""Cloudflare status entry operation.

Reads the cached Cloudflare signal snapshot from the unified state and renders
a human-readable infrastructure health summary.
"""

from __future__ import annotations

from kernel.models import OperationExecutionContext, OperationResult, UnifiedState
from lisa_pack_cloudflare.models import CloudflareMetrics

_D1_UTILIZATION_WARNING_PCT = 90.0


def _render_status(metrics: CloudflareMetrics) -> str:
    """Render a CloudflareMetrics snapshot as a concise status summary."""

    lines: list[str] = [f"## Cloudflare Status — {metrics.status.upper()}"]
    lines.append("")

    if metrics.status == "unconfigured":
        lines.append(
            "Cloudflare credentials are not configured. "
            "Set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ZONE_ID` to enable monitoring."
        )
        return "\n".join(lines)

    lines.append(f"- Zone status: {metrics.zone_status or 'unknown'}"
                 + (" (PAUSED)" if metrics.zone_paused else ""))
    lines.append(f"- Edge requests (1h): {metrics.worker_requests_1h:,}")
    lines.append(f"- Edge 5xx rate: {metrics.edge_5xx_rate:.2f}%"
                 f" ({metrics.edge_5xx_1h} errors)")
    lines.append(f"- Edge 4xx rate: {metrics.edge_4xx_1h:,} responses")
    lines.append(f"- Worker routes: {metrics.worker_routes_count}")
    lines.append(f"- Worker scripts: {metrics.worker_scripts_count}")
    lines.append(f"- WAF events (1h): {metrics.waf_total_events_1h}")

    if metrics.pages_any_failed:
        lines.append("- ⚠️ One or more Pages projects have a failing latest deployment.")

    if metrics.d1_max_utilization_pct >= _D1_UTILIZATION_WARNING_PCT:
        lines.append(f"- ⚠️ D1 max utilization: {metrics.d1_max_utilization_pct:.1f}%")

    if metrics.triage_notes:
        lines.append("")
        lines.append("### Triage Notes")
        for note in metrics.triage_notes[:8]:
            lines.append(f"- {note}")

    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the Cloudflare status operation and return the summary text."""

    state: UnifiedState | None = None
    if ctx.state_provider is not None:
        state = ctx.state_provider()

    if state is None:
        return OperationResult(
            output_text=(
                "⚠️ No cached runtime state available. "
                "The Cloudflare signal snapshot has not been collected yet."
            )
        )

    snapshot = state.signal_snapshots.get("cloudflare")
    if snapshot is None:
        return OperationResult(
            output_text=(
                "⚠️ No Cloudflare snapshot in cached state. "
                "Ensure the Cloudflare signal connector is configured and has run at least once."
            )
        )

    try:
        metrics = CloudflareMetrics.model_validate(snapshot)
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"⚠️ Failed to parse Cloudflare snapshot: {type(exc).__name__}: {exc}"
        )

    return OperationResult(output_text=_render_status(metrics))