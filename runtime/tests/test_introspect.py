"""Tests for the live introspection module (host/introspect.py)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from kernel.models import IntrospectSnapshot
from kernel.tools.mcp_client import MCPToolSpec
from runtime.introspect import gather_introspection


@pytest.mark.asyncio
async def test_gather_introspection_returns_valid_snapshot() -> None:
    """gather_introspection returns a populated IntrospectSnapshot even with no resources."""
    snapshot = await gather_introspection()
    assert isinstance(snapshot, IntrospectSnapshot)
    assert snapshot.captured_at
    # Without resources, telemetry fields are at defaults
    assert snapshot.asyncio_tasks >= 0
    assert snapshot.python_threads >= 1
    assert snapshot.python_modules > 0
    assert snapshot.operational_health in {"healthy", "degraded", "critical", "unknown"}


@pytest.mark.asyncio
async def test_gather_introspection_counts_asyncio_tasks() -> None:
    """The asyncio task count reflects the running loop's tasks."""

    async def _dummy_task() -> None:
        await asyncio.sleep(0.5)

    task = asyncio.create_task(_dummy_task())
    try:
        snapshot = await gather_introspection()
        assert snapshot.asyncio_tasks >= 1
    finally:
        task.cancel()
        with contextlib_suppress():
            await task


@pytest.mark.asyncio
async def test_gather_introspection_reads_token_usage_file() -> None:
    """Token usage summary is populated from the JSONL tail."""
    with TemporaryDirectory() as td:
        token_path = Path(td) / "token-usage.jsonl"
        # Write a few token usage records
        records = [
            {
                "model_id": "gpt-4.1-mini",
                "total_tokens": 500,
                "cached_tokens": 200,
                "cost_usd": 0.01,
                "emitted_at": "2026-07-09T10:00:00Z",
            },
            {
                "model_id": "gpt-4.1-mini",
                "total_tokens": 300,
                "cached_tokens": 100,
                "cost_usd": 0.006,
                "emitted_at": "2026-07-09T11:00:00Z",
            },
            {
                "model_id": "deepseek-chat",
                "total_tokens": 1000,
                "cost_usd": 0.003,
                "emitted_at": "2026-07-09T12:00:00Z",
            },
        ]
        with token_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        snapshot = await gather_introspection(token_usage_path=token_path)
        assert snapshot.token_usage_24h is not None
        assert snapshot.token_usage_24h.event_count == 3
        assert snapshot.token_usage_24h.total_tokens == 1800
        # billed = total - cached: (500-200) + (300-100) + (1000-0) = 1500
        assert snapshot.token_usage_24h.billed_tokens == 1500
        assert snapshot.token_usage_24h.total_cost_estimate == pytest.approx(0.019, rel=0.01)
        # gpt-4.1-mini has 800 tokens (500+300), deepseek has 1000 — deepseek is top
        assert snapshot.token_usage_24h.top_model == "deepseek-chat"
        assert snapshot.token_usage_24h.top_model_tokens == 1000


@pytest.mark.asyncio
async def test_gather_introspection_token_usage_none_when_file_missing() -> None:
    """Token usage summary is None when the file doesn't exist."""
    snapshot = await gather_introspection(
        token_usage_path=Path("/nonexistent/path/token-usage.jsonl"),
    )
    assert snapshot.token_usage_24h is None


@pytest.mark.asyncio
async def test_gather_introspection_reads_log_errors() -> None:
    """Log errors are extracted from the turn log tail."""
    with TemporaryDirectory() as td:
        log_path = Path(td) / "turns.jsonl"
        records = [
            {"level": "info", "event": "turn.completed", "message": "ok"},
            {
                "level": "warning",
                "event": "mcp.connect_failed",
                "message": "mem0 unreachable",
                "server": "mem0",
            },
            {"level": "error", "event": "operation.timeout", "message": "self_model timed out"},
            {"level": "info", "event": "turn.completed", "message": "ok"},
        ]
        with log_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        snapshot = await gather_introspection(log_path=log_path)
        assert len(snapshot.recent_log_errors) == 2
        assert snapshot.recent_log_errors[0].level == "warning"
        assert snapshot.recent_log_errors[0].event == "mcp.connect_failed"
        assert snapshot.recent_log_errors[1].level == "error"
        assert snapshot.logfile_tail_lines == 4


@pytest.mark.asyncio
async def test_gather_introspection_mcp_pool_health() -> None:
    """MCP pool health is derived from warm_server_ids and warm_failed_ids."""
    mock_pool = MagicMock()
    mock_pool.configs = [
        MagicMock(
            server_id="lisa_memory", warm=True, url=None, command="python", runtime_kind="venv_bin"
        ),
        MagicMock(
            server_id="lisa-guard", warm=True, url=None, command="python", runtime_kind="venv_bin"
        ),
        MagicMock(
            server_id="mem0", warm=False, url="https://mem0.ai", command="", runtime_kind="http"
        ),
    ]
    mock_pool.warm_server_ids.return_value = ["lisa_memory"]
    mock_pool.warm_failed_ids.return_value = ["lisa-guard"]
    mock_pool.cold_failed_ids.return_value = ["mem0"]
    mock_pool.permanently_failed_ids.return_value = ["lisa-guard"]
    mock_pool.restart_counts.return_value = {"lisa-guard": 3}
    mock_pool.last_errors.return_value = {
        "lisa-guard": "connection refused",
        "mem0": "401 unauthorized",
    }
    mock_pool.tools_by_server.return_value = {
        "lisa_memory": [
            MCPToolSpec("lisa_memory", "add_memory", "Add a memory.", {}),
            MCPToolSpec("lisa_memory", "search_memory", "Search memories.", {}),
        ],
    }

    snapshot = await gather_introspection(mcp_pool=mock_pool)
    assert len(snapshot.mcp_pool_health) == 3
    assert snapshot.mcp_pool_health["lisa_memory"].status == "warm"
    assert snapshot.mcp_pool_health["lisa_memory"].transport == "venv_bin"
    assert snapshot.mcp_pool_health["lisa_memory"].tool_count == 2
    assert snapshot.mcp_pool_health["lisa-guard"].status == "failed"
    assert snapshot.mcp_pool_health["lisa-guard"].permanently_failed is True
    assert snapshot.mcp_pool_health["lisa-guard"].restart_count == 3
    assert snapshot.mcp_pool_health["lisa-guard"].last_error == "connection refused"
    # A cold server that tried and failed reports `failed`, not `cold`.
    assert snapshot.mcp_pool_health["mem0"].status == "failed"
    assert snapshot.mcp_pool_health["mem0"].transport == "http"
    assert snapshot.mcp_pool_health["mem0"].url == "https://mem0.ai"
    assert snapshot.mcp_pool_health["mem0"].last_error == "401 unauthorized"
    assert snapshot.failed_mcp_servers == 2
    assert snapshot.operational_health in {"degraded", "critical"}


@pytest.mark.asyncio
async def test_gather_introspection_operational_health_critical_from_errors() -> None:
    """Many recent errors should push operational health into critical."""
    with TemporaryDirectory() as td:
        log_path = Path(td) / "turns.jsonl"
        with log_path.open("w", encoding="utf-8") as f:
            for _ in range(9):
                f.write(
                    json.dumps(
                        {
                            "level": "error",
                            "event": "test.error",
                            "message": "boom",
                        }
                    )
                    + "\n"
                )

        snapshot = await gather_introspection(log_path=log_path)
        assert snapshot.recent_error_count == 9
        assert snapshot.operational_health == "critical"
        assert any("recent errors" in reason for reason in snapshot.operational_health_reasons)


@pytest.mark.asyncio
async def test_gather_introspection_uptime_from_process_start() -> None:
    """Uptime is computed from the process_start_time parameter."""
    start = datetime.now(UTC)
    # Wait a tiny bit so uptime is > 0
    await asyncio.sleep(0.05)
    snapshot = await gather_introspection(process_start_time=start)
    assert snapshot.uptime_seconds > 0.0


@pytest.mark.asyncio
async def test_gather_introspection_psutil_optional() -> None:
    """When psutil is not installed, process fields return 0.0 gracefully."""
    with patch("builtins.__import__", side_effect=ImportError("no psutil")):
        snapshot = await gather_introspection()
    # rss_mb and cpu_percent should be 0.0 when psutil is unavailable
    assert snapshot.process_rss_mb == 0.0
    assert snapshot.process_cpu_percent == 0.0


@pytest.mark.asyncio
async def test_gather_introspection_proxy_pool_health_empty_without_pool() -> None:
    """No proxy pool configured -> the snapshot's proxy health list is empty."""
    snapshot = await gather_introspection()
    assert snapshot.proxy_pool_health == []


@pytest.mark.asyncio
async def test_gather_introspection_collects_proxy_pool_health() -> None:
    """Proxy health views (with declared stats) land in the snapshot."""
    from kernel.proxy_launch.pool import ProxyHealth as PoolProxyHealth

    class _StubProxyPool:
        def health(self):
            return [
                PoolProxyHealth(
                    proxy_id="headroom.headroom",
                    status="warm",
                    protocol="openai",
                    providers=("ollama",),
                    base_url="http://127.0.0.1:18787/v1",
                    warm=True,
                    required=False,
                    status_url_path="/stats",
                    status_fields=("tokens_saved",),
                    stats=None,
                ),
                PoolProxyHealth(
                    proxy_id="down.down",
                    status="failed",
                    protocol="openai",
                    providers=("openai",),
                    base_url="http://127.0.0.1:18788/v1",
                    warm=True,
                    required=False,
                    status_url_path=None,
                    status_fields=(),
                    stats=None,
                ),
            ]

        async def fetch_stats(self, proxy_id: str):
            assert proxy_id == "headroom.headroom"
            return {"tokens_saved": 2645547, "nested": {"dropped": True}}

    snapshot = await gather_introspection(proxy_pool=_StubProxyPool())

    assert len(snapshot.proxy_pool_health) == 2
    warm = snapshot.proxy_pool_health[0]
    assert warm.proxy_id == "headroom.headroom"
    assert warm.status == "warm"
    assert warm.providers == ["ollama"]
    assert warm.declared_stats_path == "/stats"
    assert warm.stats == {"tokens_saved": 2645547}
    failed = snapshot.proxy_pool_health[1]
    assert failed.status == "failed"
    assert failed.stats == {}
    # A failed proxy does not attempt a stats fetch (fetch_stats asserts).
    assert failed.proxy_id == "down.down"


def contextlib_suppress():
    """Local import of contextlib.suppress to avoid top-level import."""
    import contextlib

    return contextlib.suppress(asyncio.CancelledError)
