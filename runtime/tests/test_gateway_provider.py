"""Tests for host-side construction of credential-bound provider gateways."""

from __future__ import annotations

from typing import Any

import pytest
from kernel.proxy_launch.pool import ProxyRoute
from runtime.config import HostConfig
from runtime.gateway_provider import HostGatewayProvider, _ollama_client_instance
from runtime.runtime import HostRuntime


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
