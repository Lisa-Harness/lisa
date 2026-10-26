"""Tests for host-side construction of credential-bound provider gateways."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.proxy_launch.pool import ProxyRoute
from runtime.config import HostConfig
from runtime.gateway_provider import HostGatewayProvider, _ollama_client_instance
from runtime.provider_endpoints import proxy_route_view
from runtime.runtime import HostRuntime
from runtime.turn_hooks_builder import _transport_forwarder


class _CapturingBedrockGateway:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _GatewayClasses:
    def gateway_for(self, provider: str) -> type | None:
        return _CapturingBedrockGateway if provider == "aws" else None


def test_aws_gateway_uses_lisa_credentials_and_caches_instance(monkeypatch: Any) -> None:
    monkeypatch.setenv("LISA_AWS_REGION", "eu-west-1")
    monkeypatch.setenv("LISA_AWS_ACCESS_KEY_ID", "lisa-access")
    monkeypatch.setenv("LISA_AWS_SECRET_ACCESS_KEY", "lisa-secret")
    monkeypatch.setenv("LISA_AWS_SESSION_TOKEN", "lisa-session")

    provider = HostGatewayProvider(runtime=object(), discovered_provider=_GatewayClasses())
    gateway = provider.gateway_for("aws")

    assert isinstance(gateway, _CapturingBedrockGateway)
    assert gateway.kwargs == {
        "region_name": "eu-west-1",
        "aws_access_key_id": "lisa-access",
        "aws_secret_access_key": "lisa-secret",
        "aws_session_token": "lisa-session",
    }
    assert provider.gateway_for("aws") is gateway


def test_aws_gateway_uses_standard_aws_fallbacks(monkeypatch: Any) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

    provider = HostGatewayProvider(runtime=object(), discovered_provider=_GatewayClasses())
    gateway = provider.gateway_for("aws")

    assert isinstance(gateway, _CapturingBedrockGateway)
    assert gateway.kwargs == {
        "region_name": "ap-southeast-2",
        "aws_access_key_id": "aws-access",
        "aws_secret_access_key": "aws-secret",
        "aws_session_token": None,
    }


@pytest.mark.asyncio
async def test_ollama_client_uses_ready_openai_proxy_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("LISA_OLLAMA_HOST", "http://direct-ollama:11434")
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
    runtime.resources._proxy_routes["ollama"] = ProxyRoute(
        proxy_id="headroom",
        protocol="openai",
        base_url="http://127.0.0.1:18787/v1",
    )

    client = _ollama_client_instance(runtime)
    try:
        assert str(client.base_url) == "http://127.0.0.1:18787/v1/"
    finally:
        await client.close()


def test_proxy_route_view_reports_proxy_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
    runtime.resources._proxy_routes["ollama"] = ProxyRoute(
        proxy_id="headroom.headroom",
        protocol="openai",
        base_url="http://127.0.0.1:18787/v1",
    )

    view = proxy_route_view(runtime, "ollama", "openai")

    assert view == {
        "via": "proxy",
        "proxy_id": "headroom.headroom",
        "base_url": "http://127.0.0.1:18787/v1",
    }


def test_proxy_route_view_reports_direct_when_no_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))

    view = proxy_route_view(runtime, "ollama", "openai")

    assert view == {"via": "direct"}


def test_proxy_route_view_requires_protocol_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
    runtime.resources._proxy_routes["anthropic"] = ProxyRoute(
        proxy_id="headroom",
        protocol="anthropic",
        base_url="http://127.0.0.1:18790",
    )

    # Wrong protocol: the anthropic-wire route does not serve openai clients.
    assert proxy_route_view(runtime, "anthropic", "openai") == {"via": "direct"}
    assert proxy_route_view(runtime, "anthropic", "anthropic")["via"] == "proxy"
    assert proxy_route_view(runtime, "anthropic")["via"] == "proxy"


@pytest.mark.asyncio
async def test_model_stage_forwarder_stamps_route_on_each_iteration(tmp_path: Any) -> None:
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
    runtime.resources._proxy_routes["ollama"] = ProxyRoute(
        proxy_id="compression", protocol="openai", base_url="http://127.0.0.1:18787/v1"
    )
    runtime.resources._proxy_routes["anthropic"] = ProxyRoute(
        proxy_id="anthropic-compression",
        protocol="anthropic",
        base_url="http://127.0.0.1:18790",
    )
    received: list[dict[str, Any]] = []

    class Transport:
        async def on_stage(self, label: str, data: dict[str, Any]) -> None:
            received.append(data)

    on_stage, _ = _transport_forwarder(Transport(), runtime)
    original = {"provider": "ollama", "iteration": 1}
    await on_stage("model_call", original)
    await on_stage("model_call", {"provider": "ollama", "iteration": 2})
    await on_stage("model_call", {"provider": "anthropic", "iteration": 3})
    assert "proxy_route" not in original
    assert [event["proxy_route"]["proxy_id"] for event in received] == [
        "compression",
        "compression",
        "anthropic-compression",
    ]


def test_proxy_route_view_reports_compiled_but_bypassed_proxy(tmp_path: Any) -> None:
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
    runtime.resources._proxy_pool = SimpleNamespace(
        health=lambda: [
            SimpleNamespace(proxy_id="compression", protocol="openai", providers=("ollama",))
        ]
    )
    assert proxy_route_view(runtime, "ollama") == {
        "via": "direct",
        "bypassed_proxy_id": "compression",
    }


@pytest.mark.asyncio
async def test_dead_proxy_route_falls_back_direct_and_respawns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A proxy that died mid-session degrades the route to direct.

    Reproduction of the 2026-10-26 headroom death: the proxy process
    exited while the host was running; the cached Ollama client kept
    targeting the dead loopback port because nothing re-checked route
    liveness, and the next model call hung on the dead socket until the
    stream idle watchdog fired. ``proxy_route_for`` now verifies the
    owning process is alive, drops the stale route when it is not, and
    the client factory resets its cache so the model call proceeds
    against the direct endpoint. ``ensure_proxy_route`` respawns the
    proxy once (bounded) and the route comes back warm.
    """
    from kernel.proxy_launch.pool import ProxyPool
    from runtime.gateway_provider import _proxy_base_url

    monkeypatch.setenv("LISA_OLLAMA_HOST", "http://direct-ollama:11434")
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))

    pool = ProxyPool([])
    runtime.resources._proxy_pool = pool
    route = ProxyRoute(
        proxy_id="headroom",
        protocol="openai",
        base_url="http://127.0.0.1:18787/v1",
    )
    # Mirror the real shape: resources._proxy_routes is synced FROM
    # pool.routes (set at build time), so both hold the route. An adopted
    # route has no process handle — the pool presumes it alive.
    pool._routes["ollama"] = route
    runtime.resources._proxy_routes = pool.routes

    # A route with no owning process entry (adopted-shaped) reads alive.
    assert (
        _proxy_base_url(runtime, "ollama", "http://direct-ollama:11434/v1/")
        == "http://127.0.0.1:18787/v1/"
    )

    # The pool now knows the route maps to an exited process: the
    # host-owned death shape.
    class _DeadProcess:
        returncode = 137  # killed, the headroom death shape

    pool._processes["headroom"] = _DeadProcess()  # type: ignore[assignment]
    # Sync resolution: the stale route is dropped, direct is returned,
    # and the cached ollama client (if any) is invalidated so the next
    # factory call re-resolves its base URL.
    runtime.providers.ollama_client = object()
    assert (
        _proxy_base_url(runtime, "ollama", "http://direct-ollama:11434/v1/")
        == "http://direct-ollama:11434/v1/"
    )
    assert runtime.providers.ollama_client is None
    assert runtime.resources.proxy_route_for("ollama", "openai") is None

    # Async path: respawn finds no surviving config (pool compiled empty
    # — the pack was removed), so the route stays gone.
    assert await runtime.resources.ensure_proxy_route("ollama", "openai") is None
