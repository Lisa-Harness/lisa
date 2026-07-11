"""Cloudflare processing loop — anomaly detection from cached snapshots.

Folds the former ``anomalies.py`` anomaly detector into a ``ProcessingLoop``
that runs after signal collection during the bottom-half tick.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.intake.contracts import ProcessContext, ProcessedState
from kernel.models import AnomalyAlert
from lisa_pack_cloudflare.models import CloudflareMetrics


class CloudflareProcessingLoop:
    """Anomaly detection for cached Cloudflare telemetry."""

    loop_id = "cloudflare"

    async def process(self, ctx: ProcessContext) -> ProcessedState:
        """Examine the Cloudflare snapshot and emit alerts when thresholds are breached."""
        if ctx.state is None:
            return ProcessedState(loop_id=self.loop_id)

        snapshot = ctx.state.signal_snapshots.get("cloudflare")
        if snapshot is None:
            return ProcessedState(loop_id=self.loop_id)

        cloudflare = CloudflareMetrics.model_validate(snapshot)
        if cloudflare.edge_5xx_rate <= 5.0:
            return ProcessedState(loop_id=self.loop_id)

        return ProcessedState(
            loop_id=self.loop_id,
            active_alerts=[
                AnomalyAlert(
                    severity="critical",
                    source="cloudflare",
                    message=(
                        f"Edge 5xx error rate is {cloudflare.edge_5xx_rate:.2f}% "
                        f"(threshold: 5%) — {cloudflare.edge_5xx_1h} "
                        "server errors in the last hour"
                    ),
                    detected_at=datetime.now(UTC).isoformat(),
                )
            ],
        )


__all__ = ["CloudflareProcessingLoop"]