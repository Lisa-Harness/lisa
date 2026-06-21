"""Extension-owned context evidence for Stripe."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.models import ContextEvidence, UnifiedState
from lisa_sdk.config import settings
from sdk_extensions.stripe.models import StripeMetrics


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _default_collected_at(state: UnifiedState) -> str:
    return state.compiled_at or datetime.now(UTC).isoformat()


def _freshness_expires_at(collected_at: str) -> str:
    collected_at_dt = _parse_iso(collected_at) or datetime.now(UTC)
    ttl = timedelta(seconds=max(settings.heartbeat_interval_seconds * 2, 60))
    return (collected_at_dt + ttl).isoformat()


def build_context_evidence(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ContextEvidence]:
    """Expose Stripe context evidence without host-owned summary branches."""

    del runtime_context
    raw_snapshot = state.signal_snapshots.get("stripe")
    if raw_snapshot is None:
        return []

    stripe = StripeMetrics.model_validate(raw_snapshot)
    collected_at = _default_collected_at(state)

    mrr_usd = stripe.mrr_cents / 100.0
    summary_parts = [
        f"Status {stripe.status.upper()}",
        f"active_subs={stripe.subscriptions_active}",
        f"active_trials={stripe.trials_active}",
    ]
    if stripe.trials_started_24h > 0:
        summary_parts.append(f"trials_started_24h={stripe.trials_started_24h}")
    if mrr_usd > 0:
        summary_parts.append(f"mrr=${mrr_usd:.2f}")

    evidence_collected_at = stripe.collected_at or collected_at
    return [
        ContextEvidence(
            evidence_id="connector:stripe",
            source="stripe",
            connector_id="stripe",
            evidence_type="connector_status",
            title="Stripe subscription snapshot",
            body=" | ".join(summary_parts),
            collected_at=evidence_collected_at,
            freshness_expires_at=_freshness_expires_at(evidence_collected_at),
            confidence=0.8,
            salience_score=0.45,
            tags=["stripe", "billing", stripe.status],
        )
    ]


__all__ = ["build_context_evidence"]
