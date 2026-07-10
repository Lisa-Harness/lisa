"""Extension-owned anomaly detectors for cached Cloudflare telemetry."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionAnomalyDetector, ExtensionRuntimeContext
from kernel.models import AnomalyAlert, UnifiedState
from sdk_packs.cloudflare.models import CloudflareMetrics


async def detect_cloudflare_anomalies(state: UnifiedState) -> list[AnomalyAlert]:
    """Return low-cost anomaly alerts derived from cached Cloudflare snapshots."""

    snapshot = state.signal_snapshots.get("cloudflare")
    if snapshot is None:
        return []

    cloudflare = CloudflareMetrics.model_validate(snapshot)
    if cloudflare.edge_5xx_rate <= 5.0:
        return []

    return [
        AnomalyAlert(
            severity="critical",
            source="cloudflare",
            message=(
                f"Edge 5xx error rate is {cloudflare.edge_5xx_rate:.2f}% "
                f"(threshold: 5%) — {cloudflare.edge_5xx_1h} "
                "server errors in the last hour"
            ),
            detected_at="",
        )
    ]


def build_anomaly_detectors(
    runtime_context: ExtensionRuntimeContext,
) -> list[ExtensionAnomalyDetector]:
    """Expose the Cloudflare-owned anomaly detectors to heartbeat."""

    del runtime_context
    return [detect_cloudflare_anomalies]


__all__ = ["build_anomaly_detectors", "detect_cloudflare_anomalies"]
