"""Tests for the pure assessment projector."""

from __future__ import annotations

from lisa_pack_cloudflare.assessment import assess
from lisa_pack_cloudflare.models import CloudflareMetrics


def _ok_metrics() -> CloudflareMetrics:
    return CloudflareMetrics(
        collected_at="2026-07-15T12:00:00Z",
        worker_requests_1h=10000,
        worker_errors_1h=10,
        edge_4xx_1h=200,
        edge_5xx_1h=5,
        edge_5xx_rate=0.05,
        worker_error_rate=0.1,
        zone_status="active",
        zone_paused=False,
        status="ok",
    )


def _degraded_metrics() -> CloudflareMetrics:
    return CloudflareMetrics(
        collected_at="2026-07-15T12:00:00Z",
        worker_requests_1h=10000,
        worker_errors_1h=500,
        edge_4xx_1h=300,
        edge_5xx_1h=600,
        edge_5xx_rate=6.0,
        edge_5xx_by_status={"500": 400, "502": 200},
        worker_error_rate=5.0,
        zone_status="active",
        zone_paused=False,
        status="degraded",
        worker_scripts_detail=[
            {"script_name": "api-worker", "requests": 5000, "errors": 50, "cpu_avg_ms": 10.0, "duration_avg_ms": 20.0},
        ],
        pages_projects=[
            {"name": "site", "latest_status": "failure"},
        ],
        pages_any_failed=True,
        d1_max_utilization_pct=95.0,
        waf_events_1h={"block": 150},
        waf_total_events_1h=150,
        sampled_paths_1h=["/api/health", "/api/data"],
    )


def _unconfigured_metrics() -> CloudflareMetrics:
    return CloudflareMetrics(
        collected_at="2026-07-15T12:00:00Z",
        status="unconfigured",
    )


def test_assess_ok_snapshot_produces_zero_salience_findings() -> None:
    """Healthy findings should have salience=0.0 so the mapper drops them outside cloudflare mode."""
    assessment = assess(_ok_metrics())
    assert assessment.status == "ok"
    # At least 7 findings
    assert len(assessment.findings) >= 7
    # All findings should have salience 0.0 when healthy
    for finding in assessment.findings:
        assert finding.salience == 0.0, f"Finding {finding.id} has nonzero salience when healthy"
    # No hypotheses when healthy
    assert assessment.hypotheses == []


def test_assess_degraded_snapshot_produces_high_salience_findings() -> None:
    """Degraded findings should have nonzero salience and produce hypotheses."""
    assessment = assess(_degraded_metrics())
    assert assessment.status == "degraded"
    # At least one finding with nonzero salience
    assert any(f.salience > 0 for f in assessment.findings)
    # Should have hypotheses
    assert len(assessment.hypotheses) >= 2
    # Traffic finding should have high salience due to 5xx rate
    traffic = next(f for f in assessment.findings if f.id == "traffic")
    assert traffic.salience > 0.5
    # Deployments finding should have salience due to failed Pages
    deployments = next(f for f in assessment.findings if f.id == "deployments")
    assert deployments.salience > 0.5
    # Data finding should have salience due to D1 utilization
    data = next(f for f in assessment.findings if f.id == "data")
    assert data.salience > 0.5


def test_assess_unconfigured_snapshot_produces_unconfigured_finding() -> None:
    """Unconfigured snapshot should produce a single unconfigured finding."""
    assessment = assess(_unconfigured_metrics())
    assert assessment.status == "unconfigured"
    assert len(assessment.findings) == 1
    assert assessment.findings[0].id == "health"
    assert assessment.findings[0].tags == ["cloudflare", "unconfigured"]


def test_assessment_includes_investigation_handoff_finding() -> None:
    """The assessment should include an investigation-handoff finding."""
    assessment = assess(_degraded_metrics())
    handoff = next((f for f in assessment.findings if f.id == "investigation-handoff"), None)
    assert handoff is not None
    assert handoff.evidence_type == "investigation_handoff"


def test_assessment_findings_have_stable_ids() -> None:
    """Finding IDs should be stable across calls with the same metrics shape."""
    ids1 = {f.id for f in assess(_ok_metrics()).findings}
    ids2 = {f.id for f in assess(_degraded_metrics()).findings}
    # All 7 stable IDs should be present in both (unconfigured only has 1)
    expected_ids = {
        "health",
        "traffic",
        "workers",
        "deployments",
        "security",
        "data",
        "investigation-handoff",
    }
    assert ids1 == expected_ids
    assert ids2 == expected_ids