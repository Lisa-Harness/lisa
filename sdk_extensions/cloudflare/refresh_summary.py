"""Extension-owned refresh-summary card details for Cloudflare snapshots."""

from __future__ import annotations

from kernel.models import UnifiedState
from lisa_sdk.rendering import RefreshSummarySection
from sdk_extensions.cloudflare.models import CloudflareMetrics


def build_refresh_summary_sections(state: UnifiedState) -> list[RefreshSummarySection]:
    """Render Cloudflare-specific detail lines for the host refresh summary card."""

    snapshot = state.signal_snapshots.get("cloudflare")
    if snapshot is None:
        return []

    cloudflare = CloudflareMetrics.model_validate(snapshot)
    return [
        RefreshSummarySection(
            connector_id="cloudflare",
            lines=(
                f"Requests {cloudflare.worker_requests_1h:,} | "
                f"errors {cloudflare.worker_errors_1h:,} "
                f"({cloudflare.worker_error_rate:.1f}%)",
                f"Edge 5xx {cloudflare.edge_5xx_1h:,} ({cloudflare.edge_5xx_rate:.2f}%)",
            ),
            order=10,
        )
    ]


__all__ = ["build_refresh_summary_sections"]
