"""Extension-owned context evidence for Clarity collection state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.models import ContextEvidence, UnifiedState
from sdk_extensions.clarity.models import ClarityLiveInsightsResponse


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
    return (collected_at_dt + timedelta(hours=24)).isoformat()


def build_context_evidence(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ContextEvidence]:
    """Expose Clarity-owned collection evidence."""

    del runtime_context
    raw_collection = state.intel_collections.get("clarity")
    if raw_collection is None:
        return []

    payload = (
        raw_collection.model_dump() if hasattr(raw_collection, "model_dump") else raw_collection
    )
    clarity = ClarityLiveInsightsResponse.model_validate(payload)
    bullets = clarity.summary_bullets[:4]
    if not bullets and not clarity.collection_metadata:
        return []

    body = (
        "\n".join(f"- {bullet}" for bullet in bullets)
        if bullets
        else str(clarity.collection_metadata)
    )
    collected_at = clarity.generated_at or _default_collected_at(state)
    return [
        ContextEvidence(
            evidence_id="collection:clarity",
            source="clarity",
            evidence_type="intel_collection",
            title="Clarity insights",
            body=body,
            collected_at=collected_at,
            freshness_expires_at=_freshness_expires_at(collected_at),
            confidence=0.8,
            salience_score=0.55,
            tags=["clarity", "intel_collection"],
        )
    ]


__all__ = ["build_context_evidence"]
