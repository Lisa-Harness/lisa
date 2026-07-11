"""Cloudflare signal connector snapshot model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CloudflareMetrics(BaseModel):
    collected_at: str
    worker_requests_1h: int = 0
    worker_errors_1h: int = 0
    edge_4xx_1h: int = 0
    edge_5xx_1h: int = 0
    edge_5xx_by_status: dict[str, int] = Field(default_factory=dict)
    worker_error_rate: float = 0.0
    edge_5xx_rate: float = 0.0
    worker_cpu_ms_p99: float = 0.0
    worker_cpu_avg: float = 0.0
    worker_subrequests_1h: int = 0
    worker_duration_avg: float = 0.0
    kv_ops_1h: int = 0
    do_requests_1h: int = 0
    do_errors_1h: int = 0
    do_storage_read_rows_1h: int = 0
    do_storage_write_rows_1h: int = 0
    d1_read_queries_1h: int = 0
    d1_write_queries_1h: int = 0
    d1_rows_read_1h: int = 0
    d1_rows_written_1h: int = 0
    d1_queries_1h: int = 0
    d1_max_utilization_pct: float = 0.0
    d1_databases: list[dict[str, Any]] = Field(default_factory=list)
    waf_events_1h: dict[str, int] = Field(default_factory=dict)
    waf_total_events_1h: int = 0
    sampled_paths_1h: list[str] = Field(default_factory=list)
    sampled_user_agents_1h: list[str] = Field(default_factory=list)
    worker_routes_count: int = 0
    worker_scripts_count: int = 0
    worker_routes: list[str] = Field(default_factory=list)
    worker_scripts: list[str] = Field(default_factory=list)
    worker_scripts_detail: list[dict[str, Any]] = Field(default_factory=list)
    pages_projects: list[dict[str, Any]] = Field(default_factory=list)
    pages_any_failed: bool = False
    triage_notes: list[str] = Field(default_factory=list)
    handoff_packet: dict[str, Any] = Field(default_factory=dict)
    zone_status: str = ""
    zone_paused: bool = False
    status: str = "ok"  # "ok" | "degraded" | "down"
    raw: dict[str, Any] = Field(default_factory=dict)
