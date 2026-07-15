"""Pure assessment projector — interprets a cached Cloudflare snapshot.

This module contains NO API collection logic. It is a deterministic projection
from :class:`CloudflareMetrics` into a :class:`CloudflareAssessment` containing
typed findings, hypotheses, and investigation anchors. The connector calls
:func:`assess` after collecting; the context-evidence builder, processing loop,
and pre-turn report operation all consume the result.
"""

from __future__ import annotations

from typing import Any

from lisa_pack_cloudflare.config import get_cloudflare_connector_settings
from lisa_pack_cloudflare.models import CloudflareAssessment, CloudflareMetrics, Finding

_EDGE_5XX_DEGRADED_RATE = 5.0
_D1_UTILIZATION_WARNING_PCT = 90.0
_WORKER_CPU_NOTE_THRESHOLD_MS = 40.0


def _health_finding(metrics: CloudflareMetrics) -> Finding:
    """Build the top-level health finding."""
    parts: list[str] = [
        f"Status {metrics.status.upper()}",
        f"requests_1h={metrics.worker_requests_1h}",
        f"errors_1h={metrics.worker_errors_1h} ({metrics.worker_error_rate:.1f}%)",
        f"edge_5xx_1h={metrics.edge_5xx_1h} ({metrics.edge_5xx_rate:.2f}%)",
    ]
    if metrics.worker_cpu_ms_p99 > 0:
        parts.append(f"cpu_p99={metrics.worker_cpu_ms_p99:.0f}ms")
    if metrics.zone_status:
        parts.append(f"zone={metrics.zone_status}")

    salience = 0.0
    if metrics.status != "ok":
        salience = max(
            metrics.worker_error_rate / 10.0,
            metrics.edge_5xx_rate / 5.0,
            0.5 if metrics.status != "ok" else 0.0,
        )

    return Finding(
        id="health",
        severity="critical" if metrics.status == "error" else ("warn" if metrics.status != "ok" else "info"),
        title="Cloudflare health",
        body=" | ".join(parts),
        evidence_type="connector_status",
        tags=["cloudflare", "infra", metrics.status],
        salience=salience,
    )


def _traffic_finding(metrics: CloudflareMetrics) -> Finding:
    """Build the edge traffic finding."""
    parts = [
        f"requests_1h={metrics.worker_requests_1h}",
        f"edge_4xx_1h={metrics.edge_4xx_1h}",
        f"edge_5xx_1h={metrics.edge_5xx_1h}",
        f"edge_5xx_rate={metrics.edge_5xx_rate:.2f}%",
    ]
    if metrics.edge_5xx_by_status:
        breakdown = ", ".join(
            f"{code}:{count}" for code, count in sorted(metrics.edge_5xx_by_status.items())
        )
        parts.append(f"5xx_breakdown={breakdown}")
    if metrics.sampled_paths_1h:
        parts.append(f"top_paths={','.join(metrics.sampled_paths_1h[:3])}")

    salience = 0.0
    if metrics.edge_5xx_rate >= _EDGE_5XX_DEGRADED_RATE:
        salience = min(1.0, metrics.edge_5xx_rate / 5.0)

    return Finding(
        id="traffic",
        severity="warn" if salience > 0 else "info",
        title="Edge traffic",
        body=" | ".join(parts),
        evidence_type="traffic",
        tags=["cloudflare", "traffic", "edge"],
        salience=salience,
    )


def _workers_finding(metrics: CloudflareMetrics) -> Finding:
    """Build the Workers inventory + performance finding."""
    parts = [
        f"routes={metrics.worker_routes_count}",
        f"scripts={metrics.worker_scripts_count}",
        f"subrequests_1h={metrics.worker_subrequests_1h}",
        f"cpu_avg={metrics.worker_cpu_avg:.1f}ms",
        f"duration_avg={metrics.worker_duration_avg:.1f}ms",
    ]
    error_scripts = [
        s for s in metrics.worker_scripts_detail if int(s.get("errors", 0)) > 0
    ]
    if error_scripts:
        names = ", ".join(
            f"{s.get('script_name', '?')}({s.get('errors', 0)})"
            for s in error_scripts[:5]
        )
        parts.append(f"failing_scripts={names}")

    salience = 0.0
    if metrics.worker_cpu_avg >= _WORKER_CPU_NOTE_THRESHOLD_MS:
        salience = 0.6
    if error_scripts:
        salience = max(salience, 0.7)

    return Finding(
        id="workers",
        severity="warn" if salience > 0 else "info",
        title="Workers inventory",
        body=" | ".join(parts),
        evidence_type="workers",
        tags=["cloudflare", "workers"],
        salience=salience,
    )


def _deployments_finding(metrics: CloudflareMetrics) -> Finding:
    """Build the Pages deployments finding."""
    if not metrics.pages_projects:
        return Finding(
            id="deployments",
            title="Pages deployments",
            body="No Pages projects configured.",
            evidence_type="deployments",
            tags=["cloudflare", "pages"],
            salience=0.0,
        )

    failed = [
        p for p in metrics.pages_projects
        if str(p.get("latest_status", "")).lower() in {"failure", "failed"}
    ]
    parts = [f"projects={len(metrics.pages_projects)}"]
    if failed:
        names = ", ".join(str(p.get("name", "?")) for p in failed[:5])
        parts.append(f"failed={names}")

    salience = 0.75 if failed else 0.0

    return Finding(
        id="deployments",
        severity="warn" if failed else "info",
        title="Pages deployments",
        body=" | ".join(parts),
        evidence_type="deployments",
        tags=["cloudflare", "pages"],
        salience=salience,
    )


def _security_finding(metrics: CloudflareMetrics) -> Finding:
    """Build the WAF security finding."""
    parts = [f"waf_events_1h={metrics.waf_total_events_1h}"]
    if metrics.waf_events_1h:
        breakdown = ", ".join(
            f"{action}:{count}" for action, count in sorted(metrics.waf_events_1h.items())
        )
        parts.append(f"actions={breakdown}")

    salience = 0.4 if metrics.waf_total_events_1h > 0 else 0.0

    return Finding(
        id="security",
        title="WAF events",
        body=" | ".join(parts),
        evidence_type="security",
        tags=["cloudflare", "waf", "security"],
        salience=salience,
    )


def _data_finding(metrics: CloudflareMetrics) -> Finding:
    """Build the D1/KV/DO data finding."""
    parts = [
        f"kv_ops_1h={metrics.kv_ops_1h}",
        f"do_requests_1h={metrics.do_requests_1h}",
        f"d1_queries_1h={metrics.d1_queries_1h}",
        f"d1_max_util_pct={metrics.d1_max_utilization_pct:.1f}%",
    ]
    if metrics.do_errors_1h > 0:
        parts.append(f"do_errors_1h={metrics.do_errors_1h}")

    salience = 0.0
    if metrics.d1_max_utilization_pct >= _D1_UTILIZATION_WARNING_PCT:
        salience = 0.7

    return Finding(
        id="data",
        severity="warn" if salience > 0 else "info",
        title="D1 / KV / Durable Objects",
        body=" | ".join(parts),
        evidence_type="data",
        tags=["cloudflare", "d1", "kv", "do"],
        salience=salience,
    )


def _build_investigation_anchors(metrics: CloudflareMetrics) -> dict[str, Any]:
    """Build stable coordinates the top-half model can pass to MCP tools."""
    connector_settings = get_cloudflare_connector_settings()
    anchors: dict[str, Any] = {
        "zone_id": connector_settings.zone_id,
        "account_id": connector_settings.account_id,
    }
    if metrics.sampled_paths_1h:
        anchors["top_paths"] = metrics.sampled_paths_1h[:5]
    error_scripts = [
        str(s.get("script_name", ""))
        for s in metrics.worker_scripts_detail
        if int(s.get("errors", 0)) > 0
    ]
    if error_scripts:
        anchors["failing_scripts"] = error_scripts[:5]
    if metrics.edge_5xx_by_status:
        anchors["edge_5xx_by_status"] = metrics.edge_5xx_by_status
    return anchors


def _investigation_handoff_finding(metrics: CloudflareMetrics) -> Finding:
    """Build a compact handoff finding with stable anchors for MCP investigation."""
    anchors = _build_investigation_anchors(metrics)
    body_parts: list[str] = []
    if anchors.get("zone_id"):
        body_parts.append(f"zone_id={anchors['zone_id']}")
    if anchors.get("account_id"):
        body_parts.append(f"account_id={anchors['account_id']}")
    if anchors.get("failing_scripts"):
        body_parts.append(f"scripts={','.join(anchors['failing_scripts'])}")

    return Finding(
        id="investigation-handoff",
        title="Investigation handoff",
        body=" | ".join(body_parts) if body_parts else "No anchors available.",
        evidence_type="investigation_handoff",
        tags=["cloudflare", "handoff"],
        salience=0.0,
    )


def _build_hypotheses(metrics: CloudflareMetrics) -> list[str]:
    """Build human-readable investigation hypotheses from the snapshot."""
    hypotheses: list[str] = []
    if metrics.edge_5xx_rate >= _EDGE_5XX_DEGRADED_RATE:
        hypotheses.append(
            f"Edge 5xx rate {metrics.edge_5xx_rate:.2f}% exceeds 5% threshold — "
            "investigate origin health and Worker error logs."
        )
    if metrics.zone_paused:
        hypotheses.append("Zone is PAUSED — traffic bypasses the Cloudflare edge.")
    if metrics.zone_status and metrics.zone_status != "active":
        hypotheses.append(f"Zone status is '{metrics.zone_status}' (expected 'active').")
    if metrics.pages_any_failed:
        hypotheses.append(
            "One or more Pages projects have a failing latest deployment — "
            "check build logs and deployment status."
        )
    if metrics.d1_max_utilization_pct >= _D1_UTILIZATION_WARNING_PCT:
        hypotheses.append(
            f"D1 max utilization {metrics.d1_max_utilization_pct:.1f}% — "
            "consider sharding or archiving."
        )
    return hypotheses


def assess(metrics: CloudflareMetrics) -> CloudflareAssessment:
    """Project a :class:`CloudflareMetrics` snapshot into a :class:`CloudflareAssessment`.

    This is a pure function: no I/O, no API calls, no side effects. It
    produces typed findings, investigation hypotheses, and stable anchors
    the top-half model can use when interrogating the Cloudflare API via MCP.
    """
    if metrics.status == "unconfigured":
        return CloudflareAssessment(
            status="unconfigured",
            findings=[
                Finding(
                    id="health",
                    title="Cloudflare unconfigured",
                    body="Cloudflare credentials are not configured. Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ZONE_ID.",
                    evidence_type="connector_status",
                    tags=["cloudflare", "unconfigured"],
                    salience=0.3,
                ),
            ],
        )

    findings: list[Finding] = [
        _health_finding(metrics),
        _traffic_finding(metrics),
        _workers_finding(metrics),
        _deployments_finding(metrics),
        _security_finding(metrics),
        _data_finding(metrics),
        _investigation_handoff_finding(metrics),
    ]

    return CloudflareAssessment(
        status=metrics.status,
        findings=findings,
        hypotheses=_build_hypotheses(metrics),
        investigation_anchors=_build_investigation_anchors(metrics),
    )


__all__ = ["assess"]