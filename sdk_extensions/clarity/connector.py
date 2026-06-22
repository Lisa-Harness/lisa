"""Microsoft Clarity live-insights signal connector.

Monitors the Clarity Data Export API for aggregated UX metrics.
Polled by the clarity poll job (once per day by default per budget).

What it monitors:
- Aggregated metric rows from /project-live-insights (1-day window)
- Rage clicks, dead clicks, quick backs, JS errors, script errors
- Traffic overview: sessions, users, engagement time, scroll depth

What it emits:
- connector status: 'ok', 'error', or 'unconfigured'
- intel_collection_key: 'clarity' → written into UnifiedState.intel_collections

Configuration:
- CLARITY_API_TOKEN: Bearer token for the Clarity Data Export API
- CLARITY_PROJECT_ID: Clarity project ID to query
- CLARITY_BASE_URL: base URL override (default: https://www.clarity.ms/export-data/api/v1)
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import aiohttp
import httpx

from kernel.intake.connectors import SignalConnector
from sdk_extensions.clarity.config import get_clarity_connector_settings
from sdk_extensions.clarity.models import ClarityLiveInsightsResponse, ClarityMetricRow

logger = logging.getLogger(__name__)


def _extract_metrics_payload(data: object) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        metrics = data.get("metrics")
        if isinstance(metrics, list):
            return [row for row in metrics if isinstance(row, dict)]
    return []


def _parse_metric_row(metric: dict[str, Any]) -> ClarityMetricRow:
    raw_information = metric.get("information", [])
    information = [entry for entry in raw_information if isinstance(entry, dict)]
    metric_name = str(metric.get("metric_name") or metric.get("metricName") or "").strip()
    return ClarityMetricRow(
        metric_name=metric_name or "unknown",
        information=information,
    )


def _build_summary_bullets(metrics: list[ClarityMetricRow]) -> list[str]:
    if not metrics:
        return ["Clarity snapshot returned no aggregated metric rows."]
    ranked = sorted(metrics, key=lambda row: len(row.information), reverse=True)
    top_metrics = ", ".join(
        f"{row.metric_name} ({len(row.information)})" for row in ranked[:3]
    )
    return [
        f"Clarity snapshot contains {len(metrics)} aggregated metrics.",
        f"Top Clarity metrics: {top_metrics}",
    ]


def hydrate_live_insights_snapshot(
    snapshot: ClarityLiveInsightsResponse,
    *,
    generated_at: str | None = None,
) -> ClarityLiveInsightsResponse:
    """Populate the IntelCollection fields that survive UnifiedState storage."""
    if not snapshot.metrics:
        raw_source: object = snapshot.raw
        if isinstance(snapshot.raw, dict) and "payload" in snapshot.raw:
            raw_source = snapshot.raw.get("payload")
        snapshot.metrics = [
            _parse_metric_row(metric) for metric in _extract_metrics_payload(raw_source)
        ]

    snapshot.generated_at = generated_at or snapshot.generated_at or datetime.now(UTC).isoformat()
    snapshot.window_hours = snapshot.window_hours or 24
    snapshot.item_count = len(snapshot.metrics)
    snapshot.source_counts = {"clarity": len(snapshot.metrics)} if snapshot.metrics else {}
    snapshot.summary_bullets = _build_summary_bullets(snapshot.metrics)
    snapshot.collection_metadata = {
        "metrics": [metric.model_dump() for metric in snapshot.metrics],
        "raw": snapshot.raw,
    }
    return snapshot


async def fetch_live_insights(
    num_of_days: int = 1,
    dimensions: Iterable[str] | None = None,
) -> ClarityLiveInsightsResponse:
    """Fetch aggregated live insights from the Clarity Data Export API."""
    connector_settings = get_clarity_connector_settings()
    token = connector_settings.api_token

    if not token:
        raise RuntimeError("CLARITY_API_TOKEN is not configured in settings")

    url = f"{connector_settings.base_url.rstrip('/')}/project-live-insights"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params: dict[str, str] = {
        "numOfDays": str(max(1, min(3, int(num_of_days)))),
        "projectId": connector_settings.project_id,
    }
    if dimensions:
        for i, d in enumerate(list(dimensions)[:3], start=1):
            params[f"dimension{i}"] = d

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()

    raw_payload = data if isinstance(data, dict) else {"payload": data}
    metrics = [_parse_metric_row(metric) for metric in _extract_metrics_payload(data)]
    return hydrate_live_insights_snapshot(
        ClarityLiveInsightsResponse(metrics=metrics, raw=raw_payload)
    )


class ClaritySignalConnector(SignalConnector[ClarityLiveInsightsResponse]):
    """Signal connector for Microsoft Clarity live-insights data."""

    connector_id = "clarity"
    display_name = "Microsoft Clarity"
    intel_collection_key = "clarity"

    def missing_configuration(self) -> list[str]:
        connector_settings = get_clarity_connector_settings()
        missing: list[str] = []
        if not connector_settings.api_token:
            missing.append("CLARITY_API_TOKEN")
        if not connector_settings.project_id:
            missing.append("CLARITY_PROJECT_ID")
        return missing

    async def collect(
        self,
        client: httpx.AsyncClient,
        collected_at: str,
    ) -> ClarityLiveInsightsResponse:
        del client, collected_at
        return await fetch_live_insights(num_of_days=1)

    def build_unconfigured_snapshot(self, collected_at: str) -> ClarityLiveInsightsResponse:
        return ClarityLiveInsightsResponse(
            summary_bullets=["Clarity API token or project ID is not configured."],
            generated_at=collected_at,
        )

    def build_error_snapshot(self, collected_at: str) -> ClarityLiveInsightsResponse:
        return ClarityLiveInsightsResponse(
            summary_bullets=["Clarity collection failed."],
            generated_at=collected_at,
        )
