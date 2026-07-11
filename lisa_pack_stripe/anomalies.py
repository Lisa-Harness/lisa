"""Extension-owned anomaly detectors for cached Stripe telemetry."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionAnomalyDetector, ExtensionRuntimeContext
from kernel.models import AnomalyAlert, UnifiedState
from lisa_pack_stripe.models import StripeMetrics


async def detect_stripe_anomalies(state: UnifiedState) -> list[AnomalyAlert]:
    """Return low-cost anomaly alerts derived from cached Stripe snapshots."""

    snapshot = state.signal_snapshots.get("stripe")
    if snapshot is None:
        return []

    stripe = StripeMetrics.model_validate(snapshot)
    alerts: list[AnomalyAlert] = []

    if stripe.subscriptions_active == 0 and stripe.trials_active == 0:
        alerts.append(
            AnomalyAlert(
                severity="warn",
                source="stripe",
                message=(
                    "No active subscriptions or trials detected — "
                    "billing funnel may be stalled or credentials may be misconfigured."
                ),
                detected_at="",
            )
        )

    return alerts


def build_anomaly_detectors(
    runtime_context: ExtensionRuntimeContext,
) -> list[ExtensionAnomalyDetector]:
    """Expose the Stripe-owned anomaly detectors to heartbeat."""

    del runtime_context
    return [detect_stripe_anomalies]


__all__ = ["build_anomaly_detectors", "detect_stripe_anomalies"]