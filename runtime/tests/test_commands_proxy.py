"""Tests for the ``/proxy`` command render paths in ``runtime.commands._proxy``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import ProxyHealthStatus
from runtime.commands import _proxy


def _fake_runtime(
    pool: Any,
    routes: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        resources=SimpleNamespace(
            mcp_tool_pool=None,
            proxy_pool=pool,
            proxy_routes=routes or {},
        ),
        session=SimpleNamespace(
            current_mode_id="",
            current_stage_id="",
            next_model_id="",
        ),
    )


@pytest.fixture
def _stub_command_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runtime.commands._simple._set_turn_context", lambda *a, **k: None)
    monkeypatch.setattr("runtime.commands._simple._append_turn_log", lambda *a, **k: None)
    monkeypatch.setattr("runtime.commands._simple._append_thread_entry", lambda *a, **k: None)


def _proxy_health(proxy_id: str, status: str, **overrides: Any) -> ProxyHealthStatus:
    return ProxyHealthStatus(
        proxy_id=proxy_id,
        status=status,
        **overrides,
    )


async def _stub_gather(pool: Any) -> list[ProxyHealthStatus]:
    return getattr(pool, "preloaded_health", [])


# ── render_proxy_summary ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_render_proxy_summary_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_proxy, "gather_proxy_pool_health", _stub_gather)
    pool = SimpleNamespace(preloaded_health=[])

    text = await _proxy.render_proxy_summary(_fake_runtime(pool))

    assert text == "No provider proxies are configured."


@pytest.mark.asyncio
async def test_render_proxy_summary_renders_health_and_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_proxy, "gather_proxy_pool_health", _stub_gather)
    pool = SimpleNamespace(
        preloaded_health=[
            _proxy_health(
                "headroom.headroom",
                "warm",
                protocol="openai",
                providers=["ollama"],
                base_url="http://127.0.0.1:18787/v1",
                declared_stats_path="/stats",
                stats={"tokens_saved_1h": 113678, "savings_pct_1h": 8.7},
            ),
            _proxy_health(
                "down.down",
                "failed",
                protocol="openai",
                providers=["openai"],
                base_url="http://127.0.0.1:18788/v1",
            ),
        ]
    )
    route = SimpleNamespace(proxy_id="headroom.headroom", base_url="http://127.0.0.1:18787/v1")
    runtime = _fake_runtime(pool, routes={"ollama": route})

    text = await _proxy.render_proxy_summary(runtime)

    lines = text.splitlines()
    assert lines[0] == "## Provider Proxies"
    warm_row = next(line for line in lines if line.startswith("| headroom.headroom |"))
    assert "| warm | openai | ollama |" in warm_row
    assert "tokens_saved_1h=113678" in warm_row
    assert "savings_pct_1h=8.70" in warm_row
    failed_row = next(line for line in lines if line.startswith("| down.down |"))
    assert "| failed |" in failed_row
    assert failed_row.rstrip().endswith("| — |")
    assert "## Provider Routing" in text
    route_row = next(line for line in lines if line.startswith("| ollama |"))
    assert "headroom.headroom" in route_row
    assert "http://127.0.0.1:18787/v1" in route_row


@pytest.mark.asyncio
async def test_render_proxy_summary_health_without_routes_notes_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_proxy, "gather_proxy_pool_health", _stub_gather)
    pool = SimpleNamespace(
        preloaded_health=[
            _proxy_health("headroom.headroom", "warm", providers=["ollama"], stats={})
        ]
    )

    text = await _proxy.render_proxy_summary(_fake_runtime(pool, routes={}))

    assert "## Provider Routing" in text
    assert "every provider client uses its direct endpoint" in text


@pytest.mark.asyncio
async def test_render_proxy_summary_failed_proxy_stats_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_proxy, "gather_proxy_pool_health", _stub_gather)
    pool = SimpleNamespace(
        preloaded_health=[
            _proxy_health(
                "down.down",
                "failed",
                providers=["ollama"],
                stats={"tokens_saved_1h": 5},
            )
        ]
    )

    text = await _proxy.render_proxy_summary(_fake_runtime(pool))

    warm_row = next(line for line in text.splitlines() if line.startswith("| down.down |"))
    # Even when a failed proxy somehow carries stats, the Stats column stays —.
    assert "tokens_saved_1h=5" not in warm_row
    assert warm_row.rstrip().endswith("| — |")


@pytest.mark.asyncio
async def test_handle_proxy_command_returns_wrapped_response(
    monkeypatch: pytest.MonkeyPatch,
    _stub_command_log: None,
) -> None:
    monkeypatch.setattr(_proxy, "gather_proxy_pool_health", _stub_gather)
    monkeypatch.setattr(_proxy, "handle_simple_command", lambda *a, **k: "wrapped")
    pool = SimpleNamespace(preloaded_health=[])

    result = await _proxy.handle_proxy_command(_fake_runtime(pool), "/proxy", "/proxy")

    assert result == "wrapped"