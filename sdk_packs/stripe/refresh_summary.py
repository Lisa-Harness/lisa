"""Extension-owned refresh-summary card details for Stripe snapshots."""

from __future__ import annotations

from kernel.models import UnifiedState
from sdk.rendering import RefreshSummarySection
from sdk_packs.stripe.models import StripeMetrics


def build_refresh_summary_sections(state: UnifiedState) -> list[RefreshSummarySection]:
    """Render Stripe-specific detail lines for the host refresh summary card."""

    snapshot = state.signal_snapshots.get("stripe")
    if snapshot is None:
        return []

    stripe = StripeMetrics.model_validate(snapshot)
    return [
        RefreshSummarySection(
            connector_id="stripe",
            lines=(
                f"Active subs {stripe.subscriptions_active}",
                f"Active trials {stripe.trials_active}",
            ),
            order=10,
        )
    ]


__all__ = ["build_refresh_summary_sections"]


