"""Extension-owned digest sections for Stripe."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionDigestSection, ExtensionRuntimeContext
from kernel.models import UnifiedState
from sdk_extensions.stripe.models import StripeMetrics


def build_digest_sections(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionDigestSection]:
    """Expose Stripe digest lines without host-owned rendering branches."""

    del runtime_context
    snapshot = state.signal_snapshots.get("stripe")
    if snapshot is None:
        return []

    stripe = StripeMetrics.model_validate(snapshot)
    return [
        ExtensionDigestSection(
            lines=(
                "- Stripe: "
                f"{stripe.status.upper()} | {stripe.subscriptions_active} active subs | "
                f"{stripe.trials_active} active trials",
            ),
            order=13,
        )
    ]


__all__ = ["build_digest_sections"]
