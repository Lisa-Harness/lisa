"""Extension-owned heartbeat prompt-context fragments for Stripe."""

from __future__ import annotations

from kernel.extension_runtime import (
    ExtensionHeartbeatContextFragment,
    ExtensionRuntimeContext,
)
from kernel.models import UnifiedState
from sdk_packs.stripe.models import StripeMetrics


def build_heartbeat_context_fragments(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionHeartbeatContextFragment]:
    """Expose Stripe context without app-owned rendering branches."""

    del runtime_context
    lines = ["## Stripe"]

    snapshot = state.signal_snapshots.get("stripe")
    if snapshot is not None:
        stripe = StripeMetrics.model_validate(snapshot)
        lines.append(
            f"- Status: {stripe.status.upper()} | {stripe.subscriptions_active} active subs | "
            f"{stripe.trials_active} active trials"
        )
    else:
        lines.append("- No Stripe snapshot yet.")

    return [
        ExtensionHeartbeatContextFragment(
            order=13,
            prompt_context_blocks=("\n".join(lines),),
        )
    ]


__all__ = ["build_heartbeat_context_fragments"]

