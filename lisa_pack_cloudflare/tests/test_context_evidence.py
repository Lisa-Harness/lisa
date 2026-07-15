"""Tests for atomized context evidence."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.models import UnifiedState
from lisa_pack_cloudflare.context_evidence import build_context_evidence


def _state_with_snapshot(snapshot: dict) -> UnifiedState:
    return UnifiedState(
        compiled_at="2026-07-15T12:00:00Z",
        signal_snapshots={"cloudflare": snapshot},
    )


def _ok_snapshot() -> dict:
    return {
        "collected_at": "2026-07-15T12:00:00Z",
        "worker_requests_1h": 10000,
        "worker_errors_1h": 10,
        "edge_4xx_1h": 200,
        "edge_5xx_1h": 5,
        "edge_5xx_rate": 0.05,
        "worker_error_rate": 0.1,
        "zone_status": "active",
        "status": "ok",
    }


def _degraded_snapshot() -> dict:
    return {
        "collected_at": "2026-07-15T12:00:00Z",
        "worker_requests_1h": 10000,
        "worker_errors_1h": 500,
        "edge_4xx_1h": 300,
        "edge_5xx_1h": 600,
        "edge_5xx_rate": 6.0,
        "edge_5xx_by_status": {"500": 400, "502": 200},
        "worker_error_rate": 5.0,
        "zone_status": "active",
        "status": "degraded",
        "worker_scripts_detail": [
            {"script_name": "api", "requests": 5000, "errors": 50, "cpu_avg_ms": 10.0, "duration_avg_ms": 20.0},
        ],
        "pages_projects": [{"name": "site", "latest_status": "failure"}],
        "pages_any_failed": True,
        "d1_max_utilization_pct": 95.0,
        "waf_events_1h": {"block": 150},
        "waf_total_events_1h": 150,
        "sampled_paths_1h": ["/api/health"],
    }


def test_build_context_evidence_returns_empty_when_no_snapshot() -> None:
    """No snapshot in state → empty list."""
    state = UnifiedState(compiled_at="2026-07-15T12:00:00Z")
    evidence = build_context_evidence(ExtensionRuntimeContext(state_provider=None), state)
    assert evidence == []


def test_build_context_evidence_atomizes_into_stable_candidates() -> None:
    """Each finding becomes one evidence record with a stable cloudflare: prefixed id."""
    evidence = build_context_evidence(
        ExtensionRuntimeContext(state_provider=None),
        _state_with_snapshot(_ok_snapshot()),
    )
    ids = {e.evidence_id for e in evidence}
    expected_ids = {
        "cloudflare:health",
        "cloudflare:traffic",
        "cloudflare:workers",
        "cloudflare:deployments",
        "cloudflare:security",
        "cloudflare:data",
        "cloudflare:investigation-handoff",
    }
    assert ids == expected_ids


def test_healthy_evidence_has_zero_salience() -> None:
    """Healthy findings should have salience 0.0 so the mapper drops them outside cloudflare mode."""
    evidence = build_context_evidence(
        ExtensionRuntimeContext(state_provider=None),
        _state_with_snapshot(_ok_snapshot()),
    )
    for e in evidence:
        assert e.salience_score == 0.0, f"Evidence {e.evidence_id} has nonzero salience when healthy"


def test_degraded_evidence_has_nonzero_salience() -> None:
    """Degraded findings should have nonzero salience so they can be admitted."""
    evidence = build_context_evidence(
        ExtensionRuntimeContext(state_provider=None),
        _state_with_snapshot(_degraded_snapshot()),
    )
    assert any(e.salience_score > 0 for e in evidence)
    # The traffic finding should have high salience
    traffic = next(e for e in evidence if e.evidence_id == "cloudflare:traffic")
    assert traffic.salience_score > 0.5


def test_evidence_has_correct_source_and_collector() -> None:
    """All evidence should be sourced from cloudflare."""
    evidence = build_context_evidence(
        ExtensionRuntimeContext(state_provider=None),
        _state_with_snapshot(_ok_snapshot()),
    )
    for e in evidence:
        assert e.source == "cloudflare"
        assert e.collector_id == "cloudflare"
        assert "cloudflare" in e.tags


def test_evidence_has_system_observed_provenance() -> None:
    """All evidence should carry system_observed provenance in metadata."""
    evidence = build_context_evidence(
        ExtensionRuntimeContext(state_provider=None),
        _state_with_snapshot(_ok_snapshot()),
    )
    for e in evidence:
        assert e.metadata.get("provenance") == "system_observed"