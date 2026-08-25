"""Tests for host-side construction of credential-bound provider gateways."""

from __future__ import annotations

from typing import Any

from host.gateway_provider import HostGatewayProvider


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