"""Cloudflare zone and worker status signal connector.

Monitors Cloudflare Analytics GraphQL API for zone health and worker traffic metrics.
Polled by a host scheduler.

What it monitors:
- Zone status and pause state (via REST API)
- Edge 5xx and 4xx error rates over 1-hour sliding window (via GraphQL Analytics API)
- Worker route and script inventory (via REST API)
- Top traffic paths and user agents (via GraphQL adaptive groups)

What signals it emits:
- connector status: 'ok', 'degraded', or 'error'
- edge_5xx_rate: percentage of responses with 5xx status codes
- worker_requests_1h: total requests in the last complete hour
- worker_error_rate: percentage of responses with 5xx status codes (internal server errors)
- triage_notes: human-readable hints for debugging

Classification logic:
- Status is 'degraded' if:
  - Edge 5xx rate >= 5% (empirically correlates with customer-visible latency)
  - Zone status != 'active' (indicates zone misconfiguration)
  - Zone is paused (traffic bypasses Cloudflare edge)
- Status is 'error' if collection fails
- Status is 'ok' otherwise

Threshold tuning notes (based on customer incident history):
- 5% edge 5xx: strong signal of service degradation. Below 5%, events are typically noise.
- 20% edge 4xx with low/no 5xx: indicates scanner/invalid-request pressure, not service issue.
- 40% health-check share of top-path traffic: indicates health-check probes dominate real traffic.

API timeouts: all Cloudflare API calls use external_signal_timeout_seconds (typically 15s).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from kernel.signal_connectors.base import SignalConnector
from sdk_extensions.cloudflare.config import get_cloudflare_connector_settings
from sdk_extensions.cloudflare.models import CloudflareMetrics

logger = logging.getLogger(__name__)

# Hourly bucket dataset available from Cloudflare GraphQL Analytics API.
# Replaces the deprecated Zone Analytics Dashboard REST endpoint.
_GRAPHQL_QUERY = """
query ZoneHttpRequests($zoneTag: String!, $start: Time!, $end: Time!) {
  viewer {
    zones(filter: {zoneTag: $zoneTag}) {
      httpRequests1hGroups(
        limit: 1
        orderBy: [datetime_DESC]
        filter: {datetime_geq: $start, datetime_lt: $end}
      ) {
        sum {
          requests
          responseStatusMap { edgeResponseStatus requests }
        }
        dimensions { datetime }
      }
    }
  }
}
"""

_GRAPHQL_REQUEST_SHAPE_QUERY = """
query ZoneHttpRequestShape($zoneTag: String!, $start: Time!, $end: Time!, $limit: Int!) {
    viewer {
        zones(filter: {zoneTag: $zoneTag}) {
            httpRequestsAdaptiveGroups(
                limit: $limit
                orderBy: [count_DESC]
                filter: {datetime_geq: $start, datetime_lt: $end}
            ) {
                dimensions {
                    clientRequestPath
                    userAgent
                }
                count
            }
        }
    }
}
"""

_ACCOUNT_GRAPHQL_QUERY = """
query AccountMetrics($accountTag: String!, $start: Time!, $end: Time!) {
  viewer {
    accounts(filter: {accountTag: $accountTag}) {
      workersInvocationsAdaptive: workersInvocationsAdaptiveGroups(
        limit: 100
        filter: {datetime_geq: $start, datetime_lt: $end}
      ) {
        sum { requests, errors, subrequests }
        avg { cpuTime, duration }
                dimensions { scriptName }
      }
      kvOperations: kvOperationsAdaptiveGroups(
        limit: 100
        filter: {datetime_geq: $start, datetime_lt: $end}
      ) {
        sum { requests }
        dimensions { actionType }
      }
      durableObjects: durableObjectsInvocationsAdaptiveGroups(
        limit: 100
        filter: {datetime_geq: $start, datetime_lt: $end}
      ) {
        sum { requests, errors, activeTime, storageReadRows, storageWriteRows }
      }
      d1Analytics: d1AnalyticsAdaptiveGroups(
        limit: 100
        filter: {datetime_geq: $start, datetime_lt: $end}
      ) {
        sum { readQueries, writeQueries, rowsRead, rowsWritten }
                dimensions { databaseId }
      }
    }
  }
}
"""

_GRAPHQL_WAF_QUERY = """
query ZoneFirewallEvents($zoneTag: String!, $start: Time!, $end: Time!, $limit: Int!) {
    viewer {
        zones(filter: {zoneTag: $zoneTag}) {
            firewallEventsAdaptiveGroups(
                limit: $limit
                orderBy: [count_DESC]
                filter: {datetime_geq: $start, datetime_lt: $end}
            ) {
                dimensions { action }
                count
            }
        }
    }
}
"""

_MAX_REQUEST_SHAPE_ROWS = 10
_MAX_WORKER_ITEMS = 10
_EDGE_5XX_DEGRADED_RATE_THRESHOLD = 5.0
_EDGE_4XX_NOTABLE_RATE_THRESHOLD = 20.0
_WORKER_CPU_NOTE_THRESHOLD_MS = 40.0
_D1_UTILIZATION_DEGRADED_PCT = 90.0
_D1_SIZE_LIMIT_FALLBACK_BYTES = 500_000_000
_WAF_BLOCK_NOTABLE_COUNT = 100


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class CloudflareSignalConnector(SignalConnector[CloudflareMetrics]):
    """Cloudflare zone and worker health monitor.

    Singleton connector that polls Cloudflare APIs for zone health and worker status.
    Injected as a dependency into SignalConnectorExecutor, initialized once at app startup.

    Attributes:
        connector_id: 'cloudflare' (used to identify this connector in logs and results)
        display_name: 'Cloudflare' (human-readable name for UI/reports)

    Dependencies:
        - CLOUDFLARE_API_TOKEN (SecretStr env var): API token with read:zone and read:worker scopes
        - CLOUDFLARE_ZONE_ID (str env var): zone ID to monitor
        - CLOUDFLARE_ACCOUNT_ID (str env var, optional): account ID for worker script inventory
    """
    connector_id = "cloudflare"
    display_name = "Cloudflare"

    def missing_configuration(self) -> list[str]:
        """Check if required Cloudflare API credentials are configured.

        Returns:
            List of missing env vars: ['CLOUDFLARE_API_TOKEN'] and/or ['CLOUDFLARE_ZONE_ID'].
            Empty list if both are set.
        """
        connector_settings = get_cloudflare_connector_settings()
        missing: list[str] = []
        if not connector_settings.api_token:
            missing.append("CLOUDFLARE_API_TOKEN")
        if not connector_settings.zone_id:
            missing.append("CLOUDFLARE_ZONE_ID")
        return missing

    def build_unconfigured_snapshot(self, collected_at: str) -> CloudflareMetrics:
        """Build a snapshot when Cloudflare API credentials are missing.

        Args:
            collected_at: ISO 8601 timestamp (UTC).

        Returns:
            CloudflareMetrics with status='unconfigured' and minimal default data.
        """
        return CloudflareMetrics(collected_at=collected_at, status="unconfigured")

    def build_error_snapshot(self, collected_at: str) -> CloudflareMetrics:
        """Build a snapshot when collection fails after retries.

        Args:
            collected_at: ISO 8601 timestamp (UTC).

        Returns:
            CloudflareMetrics with status='error' and minimal default data.
        """
        return CloudflareMetrics(collected_at=collected_at, status="error")

    async def collect(self, client: httpx.AsyncClient, collected_at: str) -> CloudflareMetrics:
        """Collect Cloudflare zone and worker health metrics.

        Fetches metrics from four Cloudflare API endpoints:
        1. GraphQL Analytics API: hourly edge response counts and 5xx/4xx breakdown
        2. REST zone endpoint: zone status, pause state, name
        3. REST worker routes endpoint: list of worker routes for the zone
        4. REST worker scripts endpoint: list of worker scripts in the account
        5. GraphQL adaptive groups: top traffic paths and user agents

        The analytics data is queried for the most recently *completed* hourly bucket
        (one full hour ago from the top of the current hour). This avoids false "0 requests"
        readings due to Cloudflare's ~5-minute ingestion lag.

        Status classification:
        - 'degraded' if edge_5xx_rate >= 5% OR zone_status != 'active' OR zone_paused
        - 'ok' otherwise
        - 'error' is handled by the executor on exception

        Args:
            client: httpx.AsyncClient for making API requests.
            collected_at: ISO 8601 UTC timestamp when collection started.

        Returns:
            CloudflareMetrics with status, traffic counts, error rates, and triage notes.

        Raises:
            httpx.HTTPStatusError: If any API call returns non-2xx status.
            httpx.TimeoutException: If any API call exceeds external_signal_timeout_seconds.
            httpx.NetworkError: If network connectivity fails.
        """
        connector_settings = get_cloudflare_connector_settings()
        headers = {
            "Authorization": f"Bearer {connector_settings.api_token}",
            "Content-Type": "application/json",
        }
        # Use the most recently *completed* hourly bucket — two hours back from the
        # top of the current hour — so we always query a fully-populated window.
        # (Cloudflare hourly analytics can have up to ~5 min ingestion lag; using
        # the previous complete hour avoids false "0 requests" readings.)
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        end = now - timedelta(hours=1)       # top of last completed hour
        one_hour_ago = end - timedelta(hours=1)  # start of that same hour

        response = await client.post(
            "https://api.cloudflare.com/client/v4/graphql",
            headers=headers,
            json={
                "query": _GRAPHQL_QUERY,
                "variables": {
                    "zoneTag": connector_settings.zone_id,
                    "start": one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            },
            timeout=connector_settings.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("errors"):
            logger.warning(
                "cloudflare graphql httpRequests1hGroups returned errors: %s",
                data["errors"],
            )

        zone_details = await self._fetch_zone_details(client, headers)
        worker_routes = await self._fetch_zone_worker_routes(client, headers)
        worker_scripts = await self._fetch_account_worker_scripts(client, headers)
        request_shape = await self._fetch_request_shape(
            client,
            headers,
            one_hour_ago=one_hour_ago,
            end=end,
        )
        waf_events_1h = await self._fetch_waf_events(
            client,
            headers,
            one_hour_ago=one_hour_ago,
            end=end,
        )
        pages_projects = await self._fetch_pages_projects(client, headers)
        d1_databases = await self._fetch_d1_databases(client, headers)
        account_metrics = await self._fetch_account_metrics(
            client,
            headers,
            one_hour_ago=one_hour_ago,
            end=end,
        )

        zones = data.get("data", {}).get("viewer", {}).get("zones", [])
        groups = zones[0].get("httpRequests1hGroups", []) if zones else []
        sums = groups[0].get("sum", {}) if groups else {}

        requests = sums.get("requests", 0)
        status_map = sums.get("responseStatusMap", [])
        errors_4xx = sum(
            entry.get("requests", 0)
            for entry in status_map
            if 400 <= entry.get("edgeResponseStatus", 0) < 500
        )
        errors_5xx = sum(
            entry.get("requests", 0)
            for entry in status_map
            if 500 <= entry.get("edgeResponseStatus", 0) < 600
        )
        edge_5xx_by_status = {
            str(entry.get("edgeResponseStatus", 0)): int(entry.get("requests", 0))
            for entry in status_map
            if 500 <= entry.get("edgeResponseStatus", 0) < 600
        }

        # 4xx are client errors and rarely indicate infrastructure degradation unless extreme.
        # We only count 5xx (server errors) towards the worker/infrastructure error rate.
        total_errors = errors_5xx
        error_rate = (total_errors / max(requests, 1)) * 100
        edge_4xx_rate = (errors_4xx / max(requests, 1)) * 100
        edge_5xx_rate = (errors_5xx / max(requests, 1)) * 100

        zone_status = str(zone_details.get("status") or "")
        zone_paused = bool(zone_details.get("paused") or False)

        # Deep analytics aggregation
        worker_subrequests = 0
        worker_cpu_avg = 0.0
        worker_duration_avg = 0.0
        worker_cpu_weighted_total = 0.0
        worker_duration_weighted_total = 0.0
        worker_weight_total = 0
        kv_ops = 0
        do_reqs = 0
        do_errs = 0
        do_read_rows = 0
        do_write_rows = 0
        d1_read_q = 0
        d1_write_q = 0
        d1_r_rows = 0
        d1_w_rows = 0
        worker_script_rows: list[dict[str, Any]] = []
        d1_queries_by_database: dict[str, int] = {}
        d1_reads_by_database: dict[str, int] = {}

        if account_metrics:
            # Workers
            for w in account_metrics.get("workers", []):
                if not isinstance(w, dict):
                    continue
                sum_data = w.get("sum", {}) if isinstance(w.get("sum", {}), dict) else {}
                avg_data = w.get("avg", {}) if isinstance(w.get("avg", {}), dict) else {}
                dimensions = (
                    w.get("dimensions", {}) if isinstance(w.get("dimensions", {}), dict) else {}
                )

                requests_for_worker = _to_int(sum_data.get("requests", 0))
                errors_for_worker = _to_int(sum_data.get("errors", 0))
                subrequests_for_worker = _to_int(sum_data.get("subrequests", 0))
                cpu_avg_for_worker = _to_float(avg_data.get("cpuTime", 0.0))
                duration_avg_for_worker = _to_float(avg_data.get("duration", 0.0))

                worker_subrequests += subrequests_for_worker
                if requests_for_worker > 0:
                    worker_cpu_weighted_total += cpu_avg_for_worker * requests_for_worker
                    worker_duration_weighted_total += duration_avg_for_worker * requests_for_worker
                    worker_weight_total += requests_for_worker

                script_name = str(dimensions.get("scriptName") or "unknown")
                worker_script_rows.append(
                    {
                        "script_name": script_name,
                        "requests": requests_for_worker,
                        "errors": errors_for_worker,
                        "cpu_avg_ms": round(cpu_avg_for_worker, 3),
                        "duration_avg_ms": round(duration_avg_for_worker, 3),
                    }
                )

            if worker_weight_total > 0:
                worker_cpu_avg = worker_cpu_weighted_total / worker_weight_total
                worker_duration_avg = worker_duration_weighted_total / worker_weight_total

            # KV
            for k in account_metrics.get("kv", []):
                if not isinstance(k, dict):
                    continue
                kv_ops += _to_int(k.get("sum", {}).get("requests", 0))

            # DO
            for d in account_metrics.get("do", []):
                if not isinstance(d, dict):
                    continue
                do_reqs += _to_int(d.get("sum", {}).get("requests", 0))
                do_errs += _to_int(d.get("sum", {}).get("errors", 0))
                do_read_rows += _to_int(d.get("sum", {}).get("storageReadRows", 0))
                do_write_rows += _to_int(d.get("sum", {}).get("storageWriteRows", 0))

            # D1
            for d1 in account_metrics.get("d1", []):
                if not isinstance(d1, dict):
                    continue
                sum_data = d1.get("sum", {}) if isinstance(d1.get("sum", {}), dict) else {}
                dimensions = (
                    d1.get("dimensions", {}) if isinstance(d1.get("dimensions", {}), dict) else {}
                )
                db_id = str(dimensions.get("databaseId") or "")
                read_queries = _to_int(sum_data.get("readQueries", 0))
                write_queries = _to_int(sum_data.get("writeQueries", 0))
                rows_read = _to_int(sum_data.get("rowsRead", 0))
                rows_written = _to_int(sum_data.get("rowsWritten", 0))
                total_queries = read_queries + write_queries

                d1_read_q += read_queries
                d1_write_q += write_queries
                d1_r_rows += rows_read
                d1_w_rows += rows_written
                if db_id:
                    d1_queries_by_database[db_id] = (
                        d1_queries_by_database.get(db_id, 0) + total_queries
                    )
                    d1_reads_by_database[db_id] = d1_reads_by_database.get(db_id, 0) + rows_read

        worker_scripts_detail = sorted(
            worker_script_rows,
            key=lambda row: (
                -_to_int(row.get("errors", 0)),
                -_to_int(row.get("requests", 0)),
                str(row.get("script_name", "")),
            ),
        )[:_MAX_WORKER_ITEMS]

        d1_databases_with_runtime = []
        d1_max_utilization_pct = 0.0
        for database in d1_databases:
            if not isinstance(database, dict):
                continue
            db_id = str(database.get("id") or "")
            enriched = dict(database)
            if db_id:
                enriched["queries_1h"] = d1_queries_by_database.get(db_id, 0)
                enriched["rows_read_1h"] = d1_reads_by_database.get(db_id, 0)
            util_pct = _to_float(enriched.get("utilization_pct", 0.0))
            d1_max_utilization_pct = max(d1_max_utilization_pct, util_pct)
            d1_databases_with_runtime.append(enriched)

        pages_any_failed = any(
            str(project.get("latest_status") or "").lower() in {"failure", "failed"}
            for project in pages_projects
            if isinstance(project, dict)
        )
        waf_total_events_1h = sum(waf_events_1h.values())

        top_paths = [
            row
            for row in request_shape.get("top_paths", [])
            if isinstance(row, dict)
        ]
        top_agents = [
            row
            for row in request_shape.get("top_user_agents", [])
            if isinstance(row, dict)
        ]
        sampled_paths = [
            str(row.get("path") or "")
            for row in top_paths
            if str(row.get("path") or "")
        ]
        sampled_agents = [
            str(row.get("user_agent") or "")
            for row in top_agents
            if str(row.get("user_agent") or "")
        ]

        connector_status = "ok"
        if edge_5xx_rate >= _EDGE_5XX_DEGRADED_RATE_THRESHOLD or zone_paused:
            connector_status = "degraded"
        if zone_status and zone_status != "active":
            connector_status = "degraded"
        if pages_any_failed:
            connector_status = "degraded"
        if d1_max_utilization_pct >= _D1_UTILIZATION_DEGRADED_PCT:
            connector_status = "degraded"

        triage_notes = self._build_triage_notes(
            requests=requests,
            edge_4xx=errors_4xx,
            edge_4xx_rate=edge_4xx_rate,
            edge_5xx=errors_5xx,
            edge_5xx_by_status=edge_5xx_by_status,
            top_paths=top_paths,
            zone_status=zone_status,
            zone_paused=zone_paused,
            worker_routes=worker_routes,
            worker_scripts=worker_scripts,
            worker_scripts_detail=worker_scripts_detail,
            do_errors=do_errs,
            worker_duration=worker_duration_avg,
            pages_projects=pages_projects,
            waf_events_1h=waf_events_1h,
            d1_databases=d1_databases_with_runtime,
        )

        handoff_packet = {
            "source": "cloudflare",
            "generated_at": collected_at,
            "zone_id": connector_settings.zone_id,
            "account_id": connector_settings.account_id,
            "zone": {
                "status": zone_status,
                "paused": zone_paused,
                "name": zone_details.get("name", ""),
            },
            "traffic_1h": {
                "requests": requests,
                "errors_4xx": errors_4xx,
                "errors_5xx": errors_5xx,
                "error_rate_percent": round(error_rate, 4),
                "edge_5xx_rate_percent": round(edge_5xx_rate, 4),
                "edge_5xx_by_status": edge_5xx_by_status,
            },
            "request_shape_1h": {
                "top_paths": top_paths,
                "top_user_agents": top_agents,
            },
            "workers": {
                "routes_count": len(worker_routes),
                "scripts_count": len(worker_scripts),
                "routes": worker_routes,
                "scripts": worker_scripts,
                "scripts_detail": worker_scripts_detail,
                "subrequests": worker_subrequests,
                "cpu_avg": worker_cpu_avg,
                "duration_avg": worker_duration_avg,
            },
            "pages": {
                "projects": pages_projects,
                "any_failed": pages_any_failed,
            },
            "security": {
                "waf_events_1h": waf_events_1h,
                "waf_total_events_1h": waf_total_events_1h,
            },
            "kv": {"ops": kv_ops},
            "do": {
                "requests": do_reqs,
                "errors": do_errs,
                "storage_read_rows": do_read_rows,
                "storage_write_rows": do_write_rows,
            },
            "d1": {
                "read_queries": d1_read_q,
                "write_queries": d1_write_q,
                "rows_read": d1_r_rows,
                "rows_written": d1_w_rows,
                "total_queries": d1_read_q + d1_write_q,
                "max_utilization_pct": round(d1_max_utilization_pct, 3),
                "databases": d1_databases_with_runtime,
            },
            "triage_notes": triage_notes,
        }

        return CloudflareMetrics(
            collected_at=collected_at,
            worker_requests_1h=requests,
            worker_errors_1h=total_errors,
            edge_4xx_1h=errors_4xx,
            edge_5xx_1h=errors_5xx,
            edge_5xx_by_status=edge_5xx_by_status,
            worker_error_rate=error_rate,
            edge_5xx_rate=edge_5xx_rate,
            worker_cpu_avg=worker_cpu_avg,
            worker_subrequests_1h=worker_subrequests,
            worker_duration_avg=worker_duration_avg,
            kv_ops_1h=kv_ops,
            do_requests_1h=do_reqs,
            do_errors_1h=do_errs,
            do_storage_read_rows_1h=do_read_rows,
            do_storage_write_rows_1h=do_write_rows,
            d1_read_queries_1h=d1_read_q,
            d1_write_queries_1h=d1_write_q,
            d1_rows_read_1h=d1_r_rows,
            d1_rows_written_1h=d1_w_rows,
            d1_queries_1h=d1_read_q + d1_write_q,
            d1_max_utilization_pct=d1_max_utilization_pct,
            d1_databases=d1_databases_with_runtime,
            waf_events_1h=waf_events_1h,
            waf_total_events_1h=waf_total_events_1h,
            sampled_paths_1h=sampled_paths,
            sampled_user_agents_1h=sampled_agents,
            worker_routes_count=len(worker_routes),
            worker_scripts_count=len(worker_scripts),
            worker_routes=worker_routes,
            worker_scripts=worker_scripts,
            worker_scripts_detail=worker_scripts_detail,
            pages_projects=pages_projects,
            pages_any_failed=pages_any_failed,
            triage_notes=triage_notes,
            handoff_packet=handoff_packet,
            zone_status=zone_status,
            zone_paused=zone_paused,
            status=connector_status,
            raw={
                "http_requests_1h_sum": sums,
                "zone": zone_details,
                "worker_routes": worker_routes,
                "worker_scripts": worker_scripts,
                "worker_scripts_detail": worker_scripts_detail,
                "request_shape_1h": request_shape,
                "waf_events_1h": waf_events_1h,
                "pages_projects": pages_projects,
                "d1_databases": d1_databases_with_runtime,
                "account_metrics": account_metrics,
                "handoff_packet": handoff_packet,
            },
        )

    async def _fetch_waf_events(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        one_hour_ago: datetime,
        end: datetime,
    ) -> dict[str, int]:
        """Fetch one-hour WAF event action counts for the zone.

        Best-effort: returns empty dict when permission or dataset is unavailable.
        """
        connector_settings = get_cloudflare_connector_settings()
        try:
            response = await client.post(
                "https://api.cloudflare.com/client/v4/graphql",
                headers=headers,
                json={
                    "query": _GRAPHQL_WAF_QUERY,
                    "variables": {
                        "zoneTag": connector_settings.zone_id,
                        "start": one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "limit": _MAX_REQUEST_SHAPE_ROWS,
                    },
                },
                timeout=connector_settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                logger.debug("cloudflare waf graphql returned errors: %s", payload["errors"])
                return {}

            zones = payload.get("data", {}).get("viewer", {}).get("zones", [])
            groups = zones[0].get("firewallEventsAdaptiveGroups", []) if zones else []
            action_counts: dict[str, int] = {}
            for group in groups:
                if not isinstance(group, dict):
                    continue
                dimensions = group.get("dimensions", {})
                if not isinstance(dimensions, dict):
                    continue
                action = str(dimensions.get("action") or "unknown").strip().lower()
                if not action:
                    action = "unknown"
                action_counts[action] = (
                    action_counts.get(action, 0) + _to_int(group.get("count", 0))
                )
            return action_counts
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare waf events fetch failed: %s", exc)
            return {}

    async def _fetch_pages_projects(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Fetch Cloudflare Pages projects and latest deployment status.

        Best-effort: returns empty list if account id is missing or call fails.
        """
        account_id = get_cloudflare_connector_settings().account_id
        if not account_id:
            return []
        try:
            response = await client.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects",
                headers=headers,
                timeout=get_cloudflare_connector_settings().timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result")
            if not isinstance(result, list):
                return []

            projects: list[dict[str, Any]] = []
            for project in result[:_MAX_WORKER_ITEMS]:
                if not isinstance(project, dict):
                    continue
                latest_deployment = (
                    project.get("latest_deployment", {})
                    if isinstance(project.get("latest_deployment", {}), dict)
                    else {}
                )
                status = str(latest_deployment.get("status") or "unknown").strip().lower()
                project_name = str(project.get("name") or "").strip()
                project_url = str(
                    latest_deployment.get("url") or project.get("subdomain") or ""
                ).strip()
                if project_url and not project_url.startswith("http"):
                    project_url = f"https://{project_url}"
                projects.append(
                    {
                        "name": project_name,
                        "latest_status": status,
                        "url": project_url,
                    }
                )
            return projects
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare pages projects fetch failed: %s", exc)
            return []

    async def _fetch_d1_databases(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Fetch D1 database inventory and utilization hints.

        Best-effort: returns empty list if account id is missing or call fails.
        """
        account_id = get_cloudflare_connector_settings().account_id
        if not account_id:
            return []
        try:
            response = await client.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database",
                headers=headers,
                timeout=get_cloudflare_connector_settings().timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result")
            if not isinstance(result, list):
                return []

            databases: list[dict[str, Any]] = []
            for database in result:
                if not isinstance(database, dict):
                    continue
                file_size = _to_int(database.get("file_size", 0))
                size_limit_bytes = _to_int(
                    database.get("size")
                    or database.get("size_limit")
                    or _D1_SIZE_LIMIT_FALLBACK_BYTES,
                    _D1_SIZE_LIMIT_FALLBACK_BYTES,
                )
                utilization_pct = (file_size / max(size_limit_bytes, 1)) * 100.0
                databases.append(
                    {
                        "id": str(database.get("uuid") or database.get("id") or ""),
                        "name": str(database.get("name") or ""),
                        "file_size": file_size,
                        "size_limit_bytes": size_limit_bytes,
                        "utilization_pct": round(utilization_pct, 3),
                    }
                )
            return sorted(
                databases,
                key=lambda entry: _to_float(entry.get("utilization_pct", 0.0)),
                reverse=True,
            )[:_MAX_WORKER_ITEMS]
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare d1 databases fetch failed: %s", exc)
            return []

    async def _fetch_zone_details(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Fetch zone metadata (status, pause state, name) from the REST API.

        Calls: GET /zones/{zone_id}

        This call is best-effort. Missing scopes or transient failures are logged
        but do not fail the connector, since primary traffic/error signals come from
        the GraphQL analytics dataset.

        Args:
            client: httpx.AsyncClient for the request.
            headers: Authorization and content-type headers.

        Returns:
            Dict with zone metadata (status, paused, name, etc.), or empty dict on error.
        """
        connector_settings = get_cloudflare_connector_settings()
        try:
            response = await client.get(
                f"https://api.cloudflare.com/client/v4/zones/{connector_settings.zone_id}",
                headers=headers,
                timeout=connector_settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result")
            return result if isinstance(result, dict) else {}
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare zone details fetch failed: %s", exc)
            return {}

    async def _fetch_zone_worker_routes(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> list[str]:
        """Fetch list of worker routes deployed for the zone.

        Calls: GET /zones/{zone_id}/workers/routes

        Each route maps a URL pattern to a worker script name. Useful for inventory
        and debugging (e.g., "health-check worker should be receiving traffic").

        Best-effort: errors are logged but do not fail the connector.

        Args:
            client: httpx.AsyncClient for the request.
            headers: Authorization and content-type headers.

        Returns:
            List of route strings (e.g., ['script_name -> /api/*']), limited to _MAX_WORKER_ITEMS.
            Empty list on error.
        """
        connector_settings = get_cloudflare_connector_settings()
        try:
            response = await client.get(
                f"https://api.cloudflare.com/client/v4/zones/{connector_settings.zone_id}/workers/routes",
                headers=headers,
                timeout=connector_settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result")
            if not isinstance(result, list):
                return []
            routes: list[str] = []
            for route in result[:_MAX_WORKER_ITEMS]:
                if not isinstance(route, dict):
                    continue
                pattern = str(route.get("pattern") or "")
                script = str(route.get("script") or route.get("script_name") or "")
                routes.append(f"{script} -> {pattern}" if script else pattern)
            return [route for route in routes if route]
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare worker routes fetch failed: %s", exc)
            return []

    async def _fetch_account_worker_scripts(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> list[str]:
        """Fetch list of worker scripts deployed in the account.

        Calls: GET /accounts/{account_id}/workers/scripts

        Requires CLOUDFLARE_ACCOUNT_ID to be set. Useful for understanding deployed
        worker assets and cross-referencing with zone routes.

        Best-effort: skips if account_id is not set, or errors are logged but do not fail.

        Args:
            client: httpx.AsyncClient for the request.
            headers: Authorization and content-type headers.

        Returns:
            List of worker script names (e.g., ['health-check', 'api-gateway']),
            limited to _MAX_WORKER_ITEMS. Empty list if account_id not configured or on error.
        """
        connector_settings = get_cloudflare_connector_settings()
        account_id = connector_settings.account_id
        if not account_id:
            return []
        try:
            response = await client.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/workers/scripts",
                headers=headers,
                timeout=connector_settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("result")
            if not isinstance(result, list):
                return []
            scripts: list[str] = []
            for script in result[:_MAX_WORKER_ITEMS]:
                if not isinstance(script, dict):
                    continue
                script_name = str(script.get("id") or script.get("name") or "")
                if script_name:
                    scripts.append(script_name)
            return scripts
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare worker scripts fetch failed: %s", exc)
            return []

    async def _fetch_request_shape(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        one_hour_ago: datetime,
        end: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch the top traffic paths and user agents over the query window.

        Calls: POST /graphql with httpRequestsAdaptiveGroups query

        Groups traffic by (clientRequestPath, userAgent) dimensions and returns
        the top _MAX_REQUEST_SHAPE_ROWS by request count. Useful for:
        - Understanding what routes carry the most traffic
        - Detecting scanner/bot user agents
        - Identifying health-check routes that may skew error analysis

        Best-effort: errors are logged but do not fail the connector.

        Args:
            client: httpx.AsyncClient for the request.
            headers: Authorization and content-type headers.
            one_hour_ago: Start of the query window (datetime UTC).
            end: End of the query window (datetime UTC).

        Returns:
            Dict with keys 'top_paths' and 'top_user_agents', each containing
            list of dicts like [{'path': '/api/v1', 'requests': 1000}, ...].
            Empty lists on error.
        """
        connector_settings = get_cloudflare_connector_settings()
        try:
            response = await client.post(
                "https://api.cloudflare.com/client/v4/graphql",
                headers=headers,
                json={
                    "query": _GRAPHQL_REQUEST_SHAPE_QUERY,
                    "variables": {
                        "zoneTag": connector_settings.zone_id,
                        "start": one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "limit": _MAX_REQUEST_SHAPE_ROWS,
                    },
                },
                timeout=connector_settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                logger.debug(
                    "cloudflare request shape graphql returned errors: %s",
                    payload["errors"],
                )
                return {"top_paths": [], "top_user_agents": []}

            zones = payload.get("data", {}).get("viewer", {}).get("zones", [])
            groups = zones[0].get("httpRequestsAdaptiveGroups", []) if zones else []
            path_rollup: dict[str, int] = {}
            agent_rollup: dict[str, int] = {}

            for group in groups:
                if not isinstance(group, dict):
                    continue
                dimensions = group.get("dimensions", {})
                requests = int(group.get("count", 0) or 0)
                if not isinstance(dimensions, dict):
                    continue
                path = str(dimensions.get("clientRequestPath") or "").strip()
                user_agent = str(dimensions.get("userAgent") or "").strip()
                if path:
                    path_rollup[path] = path_rollup.get(path, 0) + requests
                if user_agent:
                    agent_rollup[user_agent] = agent_rollup.get(user_agent, 0) + requests

            top_paths = [
                {"path": path, "requests": count}
                for path, count in sorted(
                    path_rollup.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:_MAX_WORKER_ITEMS]
            ]
            top_agents = [
                {"user_agent": user_agent, "requests": count}
                for user_agent, count in sorted(
                    agent_rollup.items(), key=lambda item: item[1], reverse=True
                )[:_MAX_WORKER_ITEMS]
            ]
            return {"top_paths": top_paths, "top_user_agents": top_agents}
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare request shape fetch failed: %s", exc)
            return {"top_paths": [], "top_user_agents": []}

    async def _fetch_account_metrics(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        one_hour_ago: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """Fetch account-level deep analytics (Workers, KV, DO, D1).

        Best-effort: returns empty metrics if account_id missing or API fails.
        """
        connector_settings = get_cloudflare_connector_settings()
        account_id = connector_settings.account_id
        if not account_id:
            return {}

        try:
            response = await client.post(
                "https://api.cloudflare.com/client/v4/graphql",
                headers=headers,
                json={
                    "query": _ACCOUNT_GRAPHQL_QUERY,
                    "variables": {
                        "accountTag": account_id,
                        "start": one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                },
                timeout=connector_settings.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                logger.debug("cloudflare account metrics graphql error: %s", payload["errors"])
                return {}

            accounts = payload.get("data", {}).get("viewer", {}).get("accounts", [])
            if not accounts:
                return {}

            acc = accounts[0]
            metrics = {
                "workers": acc.get("workersInvocationsAdaptive", []),
                "kv": acc.get("kvOperations", []),
                "do": acc.get("durableObjects", []),
                "d1": acc.get("d1Analytics", []),
            }
            return metrics
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("cloudflare account metrics fetch failed: %s", exc)
            return {}

    def _build_triage_notes(
        self,
        requests: int,
        edge_4xx: int,
        edge_4xx_rate: float,
        edge_5xx: int,
        edge_5xx_by_status: dict[str, int],
        top_paths: list[dict[str, Any]],
        zone_status: str,
        zone_paused: bool,
        worker_routes: list[str],
        worker_scripts: list[str],
        worker_scripts_detail: list[dict[str, Any]],
        pages_projects: list[dict[str, Any]],
        waf_events_1h: dict[str, int],
        d1_databases: list[dict[str, Any]],
        do_errors: int = 0,
        worker_duration: float = 0.0,
    ) -> list[str]:
        """Build human-readable triage notes explaining the zone health snapshot.

        These notes are sent to the operator and should explain:
        - What signals indicate problems
        - Why the zone might be degraded
        - What actions the operator can take

        Examples of notes:
        - "Edge 5xx breakdown (status:count): 502:45, 503:12" (helps debug errors)
        - "Elevated edge 4xx rate (25%) with low/no 5xx can indicate scanner pressure" (context)
        - "Health-check traffic appears dominant (45%) — worker logs may look sparse" (caveat)
        - "Zone is paused; traffic may bypass Cloudflare edge" (config issue)

        Args:
            requests: Total edge requests in the 1h window.
            edge_4xx: Count of 4xx errors.
            edge_4xx_rate: Percentage of requests that are 4xx.
            edge_5xx: Count of 5xx errors.
            edge_5xx_by_status: Dict mapping status code (str) to count (int).
            top_paths: List of top traffic paths with request counts.
            zone_status: Zone status from REST API (e.g., 'active', 'pending').
            zone_paused: Whether zone is paused.
            worker_routes: List of deployed routes (for inventory cross-check).
            worker_scripts: List of deployed scripts (for inventory cross-check).
            do_errors: Durable Object internal errors.
            worker_duration: Avg worker execution duration (ms).

        Returns:
            List of note strings. Always includes at least one note (default if empty).
        """
        notes: list[str] = []
        if requests == 0:
            notes.append(
                "No traffic observed in the last complete hour from Cloudflare analytics."
            )

        if get_cloudflare_connector_settings().account_id == "":
            notes.append(
                "CLOUDFLARE_ACCOUNT_ID is not configured, "
                "so deep analytics (Workers CPU, KV, D1, DO) are unavailable."
            )

        if edge_5xx > 0 and edge_5xx_by_status:
            breakdown = ", ".join(
                f"{status}:{count}" for status, count in sorted(edge_5xx_by_status.items())
            )
            notes.append(f"Edge 5xx breakdown (status:count): {breakdown}.")

        if do_errors > 0:
            notes.append(
                f"Durable Object internal errors detected: {do_errors} errors in the last hour."
            )

        if worker_duration >= _WORKER_CPU_NOTE_THRESHOLD_MS:
            notes.append(
                f"Worker duration is high (avg {worker_duration:.2f}ms), "
                "approaching the 50ms standard limit."
            )

        if worker_scripts_detail:
            for row in worker_scripts_detail[:_MAX_WORKER_ITEMS]:
                script_errors = _to_int(row.get("errors", 0))
                script_cpu_avg = _to_float(row.get("cpu_avg_ms", 0.0))
                if script_errors > 0 and script_cpu_avg >= _WORKER_CPU_NOTE_THRESHOLD_MS:
                    notes.append(
                        "Worker "
                        f"'{row.get('script_name', 'unknown')}' is averaging "
                        f"{script_cpu_avg:.2f}ms CPU and throwing {script_errors} errors."
                    )

        if edge_4xx > 0 and edge_4xx_rate >= _EDGE_4XX_NOTABLE_RATE_THRESHOLD:
            notes.append(
                f"Elevated edge 4xx rate ({edge_4xx_rate:.2f}%) with low/no 5xx can indicate "
                "scanner/invalid-request pressure rather than service degradation."
            )

        if waf_events_1h:
            waf_blocks = _to_int(waf_events_1h.get("block", 0))
            if waf_blocks >= _WAF_BLOCK_NOTABLE_COUNT:
                notes.append(
                    f"WAF blocked {waf_blocks} requests in the last hour; "
                    "confirm whether 4xx pressure is scanner traffic."
                )

        for project in pages_projects:
            if not isinstance(project, dict):
                continue
            latest_status = str(project.get("latest_status") or "").lower()
            if latest_status in {"failure", "failed"}:
                project_name = str(project.get("name") or "unknown")
                notes.append(
                    f"Pages project '{project_name}' has a failing latest deployment."
                )

        for database in d1_databases:
            if not isinstance(database, dict):
                continue
            utilization_pct = _to_float(database.get("utilization_pct", 0.0))
            if utilization_pct >= _D1_UTILIZATION_DEGRADED_PCT:
                db_name = str(database.get("name") or database.get("id") or "unknown")
                notes.append(
                    f"D1 database '{db_name}' is at {utilization_pct:.1f}% of its size limit."
                )

        if top_paths and requests > 0:
            top_path = top_paths[0]
            top_path_value = str(top_path.get("path") or "")
            top_path_requests = int(top_path.get("requests", 0) or 0)
            if top_path_value and top_path_requests > 0:
                share = (top_path_requests / requests) * 100
                if "health" in top_path_value.lower() and share >= 40:
                    notes.append(
                        "Health-check traffic appears dominant in the last hour; "
                        "worker logs can look sparse "
                        "if only health-check probes hit the route."
                    )

        if not worker_routes:
            notes.append(
                "No worker routes were returned for this zone. "
                "Verify route bindings and script deployment."
            )

        if not worker_scripts and get_cloudflare_connector_settings().account_id:
            notes.append(
                "No worker scripts were returned for this account. "
                "Verify API token scopes and account context."
            )

        if zone_paused:
            notes.append("Zone is paused; traffic may bypass Cloudflare edge execution paths.")

        if zone_status and zone_status != "active":
            notes.append(f"Zone status is {zone_status}, not active.")

        if not notes:
            notes.append("No obvious Cloudflare-side anomaly was detected from the available APIs.")
        return notes
