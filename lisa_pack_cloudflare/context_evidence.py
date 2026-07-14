"""Extension-owned context evidence for Cloudflare."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.models import ContextEvidence, UnifiedState
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
    """Expose Cloudflare context evidence without host-owned summary branches."""

    del runtime_context
    raw_snapshot = state.signal_snapshots.get("cloudflare")
    if raw_snapshot is None:
        return []

    cloudflare = CloudflareMetrics.model_validate(raw_snapshot)
    summary_parts = [
        f"Status {cloudflare.status.upper()}",
        f"requests_1h={cloudflare.worker_requests_1h}",
        f"errors_1h={cloudflare.worker_errors_1h} ({cloudflare.worker_error_rate:.1f}%)",
        f"edge_5xx_1h={cloudflare.edge_5xx_1h} ({cloudflare.edge_5xx_rate:.2f}%)",
    ]
    if cloudflare.worker_cpu_ms_p99 > 0:
        summary_parts.append(f"cpu_p99={cloudflare.worker_cpu_ms_p99:.0f}ms")
    if cloudflare.pages_projects:
        summary_parts.append(f"pages_failed={cloudflare.pages_any_failed}")
    if cloudflare.waf_total_events_1h > 0:
        summary_parts.append(f"waf_events_1h={cloudflare.waf_total_events_1h}")
    if cloudflare.d1_max_utilization_pct > 0:
        summary_parts.append(f"d1_max_util_pct={cloudflare.d1_max_utilization_pct:.1f}%")
    if cloudflare.do_requests_1h > 0:
        summary_parts.append(f"do_req_1h={cloudflare.do_requests_1h}")
    if cloudflare.zone_status:
        summary_parts.append(f"zone={cloudflare.zone_status}")

    salience_score = min(
        1.0,
        max(
            cloudflare.worker_error_rate / 10.0,
            cloudflare.edge_5xx_rate / 5.0,
            cloudflare.d1_max_utilization_pct / 100.0,
            0.75 if cloudflare.pages_any_failed else 0.0,
            0.5 if cloudflare.status != "ok" else 0.0,
        ),
    )
    collected_at = _default_collected_at(state)

    evidence_collected_at = cloudflare.collected_at or collected_at
    return [
        ContextEvidence(
            evidence_id="collector:cloudflare",
            source="cloudflare",
            collector_id="cloudflare",
            evidence_type="connector_status",
            title="Cloudflare health",
            body=" | ".join(summary_parts),
            collected_at=evidence_collected_at,
            freshness_expires_at=_freshness_expires_at(evidence_collected_at),
            confidence=0.8,
            salience_score=salience_score,
            tags=["cloudflare", "infra", cloudflare.status],
        )
    ]


__all__ = ["build_context_evidence"]

