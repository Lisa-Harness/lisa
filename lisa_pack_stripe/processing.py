"""Stripe processing loop — anomaly detection from cached snapshots.

Folds the former ``anomalies.py`` anomaly detector into a ``ProcessingLoop``
that runs after signal collection during the bottom-half tick.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.intake.contracts import ProcessContext, ProcessedState
from kernel.models import AnomalyAlert
from lisa_pack_stripe.models import StripeMetrics


class StripeProcessingLoop:
    """Anomaly detection for cached Stripe telemetry."""

    loop_id = "stripe"

    async def process(self, ctx: ProcessContext) -> ProcessedState:
        """Examine the Stripe snapshot and emit alerts when the billing funnel is stalled."""
        if ctx.state is None:
            return ProcessedState(loop_id=self.loop_id)

        snapshot = ctx.state.signal_snapshots.get("stripe")
        if snapshot is None:
            return ProcessedState(loop_id=self.loop_id)

        stripe = StripeMetrics.model_validate(snapshot)
        if not (stripe.subscriptions_active == 0 and stripe.trials_active == 0):
            return ProcessedState(loop_id=self.loop_id)

        return ProcessedState(
            loop_id=self.loop_id,
            active_alerts=[
                AnomalyAlert(
                    severity="warn",
                    source="stripe",
                    message=(
                        "No active subscriptions or trials detected — "
                        "billing funnel may be stalled or credentials may be misconfigured."
                    ),
                    detected_at=datetime.now(UTC).isoformat(),
                )
            ],
        )


__all__ = ["StripeProcessingLoop"]