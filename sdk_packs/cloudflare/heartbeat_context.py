"""Extension-owned heartbeat prompt-context fragments for Cloudflare."""

from __future__ import annotations

from kernel.extension_runtime import (
    ExtensionHeartbeatContextFragment,
    ExtensionRuntimeContext,
)
from kernel.models import UnifiedState
from sdk_packs.cloudflare.models import CloudflareMetrics


def build_heartbeat_context_fragments(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionHeartbeatContextFragment]:
    """Expose Cloudflare context without app-owned rendering branches."""

    del runtime_context
    lines = ["## Cloudflare"]

    snapshot = state.signal_snapshots.get("cloudflare")
    if snapshot is not None:
        cloudflare = CloudflareMetrics.model_validate(snapshot)
        lines.append(
            f"- Status: {cloudflare.status.upper()} | Errors 1h: {cloudflare.worker_errors_1h} "
            f"({cloudflare.worker_error_rate:.1f}%) | 5xx 1h: {cloudflare.edge_5xx_1h} "
            f"({cloudflare.edge_5xx_rate:.2f}%) | Requests: {cloudflare.worker_requests_1h} | "
            f"Zone: {(cloudflare.zone_status or 'unknown').upper()}"
            + (" (paused)" if cloudflare.zone_paused else "")
        )
    else:
        lines.append("- No Cloudflare snapshot yet.")

    return [
        ExtensionHeartbeatContextFragment(
            order=11,
            prompt_context_blocks=("\n".join(lines),),
        )
    ]


__all__ = ["build_heartbeat_context_fragments"]
