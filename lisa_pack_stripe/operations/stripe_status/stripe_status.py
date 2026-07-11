"""Stripe status entry operation.

Reads the cached Stripe signal snapshot from the unified state and renders
a human-readable billing and subscription health summary.
"""

from __future__ import annotations

from kernel.models import OperationExecutionContext, OperationResult, UnifiedState
from lisa_pack_stripe.models import StripeMetrics


def _render_status(metrics: StripeMetrics) -> str:
    """Render a StripeMetrics snapshot as a concise billing summary."""

    lines: list[str] = [f"## Stripe Status — {metrics.status.upper()}"]
    lines.append("")

    if metrics.status == "unconfigured":
        lines.append(
            "Stripe credentials are not configured. "
            "Set `STRIPE_API_KEY` to enable monitoring."
        )
        return "\n".join(lines)

    mrr_dollars = metrics.mrr_cents / 100.0
    lines.append(f"- Active subscriptions: {metrics.subscriptions_active}")
    lines.append(f"- Active trials: {metrics.trials_active}")
    lines.append(f"- Trials started (24h): {metrics.trials_started_24h}")
    lines.append(f"- MRR: ${mrr_dollars:,.2f}")

    if metrics.subscriptions_active == 0 and metrics.trials_active == 0:
        lines.append("- ⚠️ No active subscriptions or trials detected.")

    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the Stripe status operation and return the summary text."""

    state: UnifiedState | None = None
    if ctx.state_provider is not None:
        state = ctx.state_provider()

    if state is None:
        return OperationResult(
            output_text=(
                "⚠️ No cached runtime state available. "
                "The Stripe signal snapshot has not been collected yet."
            )
        )

    snapshot = state.signal_snapshots.get("stripe")
    if snapshot is None:
        return OperationResult(
            output_text=(
                "⚠️ No Stripe snapshot in cached state. "
                "Ensure the Stripe signal connector is configured and has run at least once."
            )
        )

    try:
        metrics = StripeMetrics.model_validate(snapshot)
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"⚠️ Failed to parse Stripe snapshot: {type(exc).__name__}: {exc}"
        )

    return OperationResult(output_text=_render_status(metrics))