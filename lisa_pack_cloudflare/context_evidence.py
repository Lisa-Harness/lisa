"""Extension-owned context evidence for Cloudflare.

Atomizes the cached Cloudflare snapshot into stable, accurately-labeled
:class:`ContextEvidence` candidates via the pure :func:`assess` projector.
Each finding becomes one evidence record with accurate ``evidence_type``,
``tags``, ``salience_score`` (0.0 when healthy so the mapper drops it
outside ``cloudflare`` mode), and ``provenance``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.models import ContextEvidence, UnifiedState
from lisa_pack_cloudflare.assessment import assess
from lisa_pack_cloudflare.models import CloudflareMetrics
from sdk.config import settings


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
    """Expose Cloudflare context evidence without host-owned summary branches.

    Each finding from :func:`assess` becomes one :class:`ContextEvidence`
    record. Healthy findings have ``salience_score=0.0`` so the session-owned
    context mapper drops them outside ``cloudflare`` mode (lexical overlap
    alone cannot elevate them). Alerts (severity=critical) have high
    salience so they reach the ``incidents`` lane globally.
    """

    del runtime_context
    raw_snapshot = state.signal_snapshots.get("cloudflare")
    if raw_snapshot is None:
        return []

    cloudflare = CloudflareMetrics.model_validate(raw_snapshot)
    assessment = assess(cloudflare)
    collected_at = _default_collected_at(state)
    evidence_collected_at = cloudflare.collected_at or collected_at

    evidence: list[ContextEvidence] = []
    for finding in assessment.findings:
        severity_to_salience = {"critical": 1.0, "warn": 0.85, "info": finding.salience}
        salience = severity_to_salience.get(finding.severity, finding.salience)

        evidence.append(
            ContextEvidence(
                evidence_id=f"cloudflare:{finding.id}",
                source="cloudflare",
                collector_id="cloudflare",
                evidence_type=finding.evidence_type or "connector_status",
                title=finding.title,
                body=finding.body,
                collected_at=evidence_collected_at,
                freshness_expires_at=_freshness_expires_at(evidence_collected_at),
                confidence=0.8,
                salience_score=salience,
                tags=list(finding.tags),
                metadata={"provenance": "system_observed", "severity": finding.severity},
            )
        )

    return evidence


__all__ = ["build_context_evidence"]

