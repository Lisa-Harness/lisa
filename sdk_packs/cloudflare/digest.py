"""Extension-owned digest sections for Cloudflare."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionDigestSection, ExtensionRuntimeContext
from kernel.models import UnifiedState
from sdk_packs.cloudflare.models import CloudflareMetrics


def build_digest_sections(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionDigestSection]:
    """Expose Cloudflare digest lines without host-owned rendering branches."""

    del runtime_context
    snapshot = state.signal_snapshots.get("cloudflare")
    if snapshot is None:
        return []

    cloudflare = CloudflareMetrics.model_validate(snapshot)
    return [
        ExtensionDigestSection(
            lines=(
                "- Cloudflare: "
                f"{cloudflare.status.upper()} | errors 1h: {cloudflare.worker_errors_1h} "
                f"({cloudflare.worker_error_rate:.1f}%) | 5xx 1h: {cloudflare.edge_5xx_1h} "
                f"({cloudflare.edge_5xx_rate:.2f}%) | requests: {cloudflare.worker_requests_1h} | "
                f"zone: {(cloudflare.zone_status or 'unknown').upper()}"
                + (" (paused)" if cloudflare.zone_paused else ""),
            ),
            order=11,
        )
    ]


__all__ = ["build_digest_sections"]
