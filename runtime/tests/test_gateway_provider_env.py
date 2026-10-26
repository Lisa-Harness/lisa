"""Env-var branch coverage for the host provider gateway factories.

These tests never construct real SDK clients: they assert the
error/caching paths and constructor kwargs captured by fakes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from openai import AsyncOpenAI
from runtime.gateway_provider import (
    HostGatewayProvider,
    _anthropic_http_client_instance,
    _azure_client_instance,
    _env,
    _openai_client_instance,
    _openrouter_client_factory,
    _proxy_base_url,
)


class _FakeRuntime:
    """Minimal runtime stand-in exposing providers/config/resources slots."""

    def __init__(self, *, config: Any = None, **provider_slots: Any) -> None:
        self.providers = SimpleNamespace(
            azure_client=None,
            openai_client=None,
            anthropic_http_client=None,
            openrouter_client=None,
            ollama_client=None,
        )
        vars(self.providers).update(provider_slots)
        self.config = config or SimpleNamespace(prompt_cache_enabled=True)
        self.resources = SimpleNamespace(proxy_route_for=lambda p, proto: None, proxy_routes={})


def _runtime(*, config: Any = None, **provider_slots: Any) -> _FakeRuntime:
    return _FakeRuntime(config=config, **provider_slots)


class _CapturingGateway:
    """Records its constructor kwargs; stands in for any gateway class."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _GatewayClasses:
    """Gateway-class provider returning ``_CapturingGateway`` for the given names."""

    def __init__(self, names: set[str]) -> None:
        self._names = names

    def gateway_for(self, provider: str) -> type | None:
        return _CapturingGateway if provider in self._names else None


# ── _env ─────────────────────────────────────────────────────────────


def test_env_returns_first_non_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISA_TEST_A", "  ")
    monkeypatch.setenv("LISA_TEST_B", "b-value")
    monkeypatch.setenv("LISA_TEST_C", "c-value")

    assert _env("LISA_TEST_A") == ""
    assert _env("LISA_TEST_A", "LISA_TEST_B", "LISA_TEST_C") == "b-value"
    assert _env("LISA_TEST_MISSING", "LISA_TEST_C") == "c-value"
    assert _env("LISA_TEST_MISSING_1", "LISA_TEST_MISSING_2") == ""


# ── HostGatewayProvider.gateway_for basics ───────────────────────────


def test_gateway_for_blank_provider_returns_none() -> None:
    provider = HostGatewayProvider(runtime=_runtime(), discovered_provider=_GatewayClasses(set()))

    assert provider.gateway_for("") is None
    assert provider.gateway_for("   ") is None


def test_gateway_for_unknown_provider_returns_none() -> None:
    provider = HostGatewayProvider(runtime=_runtime(), discovered_provider=_GatewayClasses(set()))

    assert provider.gateway_for("anthropic") is None


def test_gateway_for_normalizes_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISA_GOOGLE_API_KEY", "gk")
    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=_GatewayClasses({"google"}),
    )

    gateway = provider.gateway_for("  GOOGLE ")
    assert isinstance(gateway, _CapturingGateway)
    assert gateway.kwargs["api_key"] == "gk"
    # Cached instance is returned on repeat lookups.
    assert provider.gateway_for("google") is gateway


def test_gateway_for_exploding_construction_returns_none() -> None:
    class _ExplodingGateway:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("bad kwargs")

    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=SimpleNamespace(gateway_for=lambda p: _ExplodingGateway),
    )

    assert provider.gateway_for("azure_openai") is None


def test_gateway_for_unknown_provider_falls_through_to_none() -> None:
    """A provider name no branch matches still returns ``None`` (PLR0911 tail)."""

    class _AnyGateway:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=SimpleNamespace(gateway_for=lambda p: _AnyGateway),
    )

    assert provider.gateway_for("mystery_provider") is None


# ── Azure branches ───────────────────────────────────────────────────


def test_azure_client_instance_missing_config_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("LISA_AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Azure OpenAI is not configured"):
        _azure_client_instance(_runtime())


@pytest.mark.asyncio
async def test_azure_client_instance_cached_hit() -> None:
    real_client = AsyncOpenAI(
        base_url="https://cached.openai.azure.com/openai/v1/",
        api_key="cached-key",
    )
    runtime = _runtime(azure_client=real_client)
    try:
        assert _azure_client_instance(runtime) is real_client
    finally:
        await real_client.close()


@pytest.mark.asyncio
async def test_azure_client_instance_constructs_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://ep.openai.azure.com")
    monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "ak")

    runtime = _runtime()
    client = _azure_client_instance(runtime)
    try:
        assert str(client.base_url) == "https://ep.openai.azure.com/openai/v1/"
        assert "api-version" not in client.default_query
        assert runtime.providers.azure_client is client
        # Second call is the cache-hit branch.
        assert _azure_client_instance(runtime) is client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_azure_v1_endpoint_is_not_appended_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://ep.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "ak-std")

    runtime = _runtime()
    client = _azure_client_instance(runtime)
    try:
        assert str(client.base_url) == "https://ep.openai.azure.com/openai/v1/"
        assert runtime.providers.azure_client is client
    finally:
        await client.close()


# ── OpenAI-compatible branch ─────────────────────────────────────────


def test_openai_client_instance_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenAI is not configured"):
        _openai_client_instance(_runtime())


@pytest.mark.asyncio
async def test_openai_client_instance_constructs_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_OPENAI_API_KEY", "sk-test")

    runtime = _runtime()
    client = _openai_client_instance(runtime)
    try:
        assert isinstance(client, AsyncOpenAI)
        assert runtime.providers.openai_client is client
        assert _openai_client_instance(runtime) is client
    finally:
        await client.close()


# ── Anthropic HTTP client ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_http_client_direct_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_ANTHROPIC_BASE_URL", raising=False)
    runtime = _runtime()

    client = _anthropic_http_client_instance(runtime)
    try:
        assert str(client.base_url) == "https://api.anthropic.com"
        assert runtime.providers.anthropic_http_client is client
        # cache-hit branch
        assert _anthropic_http_client_instance(runtime) is client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_http_client_honors_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISA_ANTHROPIC_BASE_URL", "https://anthropic-proxy.test/")
    runtime = _runtime()

    client = _anthropic_http_client_instance(runtime)
    try:
        assert str(client.base_url) == "https://anthropic-proxy.test"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_http_client_uses_proxy_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_ANTHROPIC_BASE_URL", raising=False)
    route = SimpleNamespace(base_url="http://127.0.0.1:19999/anthropic")
    runtime = _runtime()
    runtime.resources = SimpleNamespace(proxy_route_for=lambda p, proto: route, proxy_routes={})

    client = _anthropic_http_client_instance(runtime)
    try:
        assert str(client.base_url) == "http://127.0.0.1:19999/anthropic/"
    finally:
        await client.aclose()


# ── _proxy_base_url ──────────────────────────────────────────────────


def test_proxy_base_url_no_route_returns_direct() -> None:
    runtime = _runtime()

    assert (
        _proxy_base_url(runtime, "openai", "https://direct.example/") == "https://direct.example/"
    )


def test_proxy_base_url_route_appends_slash() -> None:
    route = SimpleNamespace(base_url="http://127.0.0.1:18000/v1/")
    runtime = _runtime()
    runtime.resources = SimpleNamespace(proxy_route_for=lambda p, proto: route, proxy_routes={})

    assert (
        _proxy_base_url(runtime, "openai", "https://direct.example/")
        == "http://127.0.0.1:18000/v1/"
    )


# ── OpenRouter factory ───────────────────────────────────────────────


def test_openrouter_factory_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    build = _openrouter_client_factory(_runtime())

    with pytest.raises(ValueError, match="OpenRouter is not configured"):
        build()


@pytest.mark.asyncio
async def test_openrouter_factory_builds_with_headers_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("LISA_OPENROUTER_BASE_URL", "https://openrouter.example/api/v1")
    monkeypatch.setenv("LISA_OPENROUTER_REFERER", "https://lisa.example")
    monkeypatch.setenv("LISA_OPENROUTER_TITLE", "Lisa Test")

    runtime = _runtime()
    build = _openrouter_client_factory(runtime)
    client = build()
    try:
        assert str(client.base_url) == "https://openrouter.example/api/v1/"
        headers = client.default_headers
        assert headers["HTTP-Referer"] == "https://lisa.example"
        assert headers["X-OpenRouter-Title"] == "Lisa Test"
        assert runtime.providers.openrouter_client is client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_openrouter_factory_reuses_cached_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the factory read a dead ``_openrouter_client`` slot, so every
    # model call built a fresh AsyncOpenAI client. Abandoned clients closed
    # their sockets from ``__del__`` after GC had already released the fd,
    # unregistering whatever reused that fd number (e.g. a subprocess pidfd)
    # and hanging coding tools until their deadline.
    monkeypatch.setenv("LISA_OPENROUTER_API_KEY", "or-key")

    runtime = _runtime()
    build = _openrouter_client_factory(runtime)
    client = build()
    try:
        assert build() is client
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_openrouter_factory_env_variants_no_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key-std")
    monkeypatch.delenv("LISA_OPENROUTER_REFERER", raising=False)
    monkeypatch.delenv("OPENROUTER_REFERER", raising=False)
    monkeypatch.delenv("LISA_OPENROUTER_TITLE", raising=False)
    monkeypatch.delenv("OPENROUTER_TITLE", raising=False)
    monkeypatch.delenv("LISA_OPENROUTER_BASE_URL", raising=False)

    runtime = _runtime()
    client = _openrouter_client_factory(runtime)()
    try:
        assert "HTTP-Referer" not in client.default_headers
        assert "X-OpenRouter-Title" not in client.default_headers
    finally:
        await client.close()


# ── Gateway-construction branches through the provider ──────────────


@pytest.mark.asyncio
async def test_anthropic_gateway_receives_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISA_ANTHROPIC_API_KEY", "anth-key")
    monkeypatch.setenv("LISA_ANTHROPIC_VERSION", "2024-01-01")

    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=_GatewayClasses({"anthropic"}),
    )
    gateway = provider.gateway_for("anthropic")

    assert isinstance(gateway, _CapturingGateway)
    assert gateway.kwargs["api_key"] == "anth-key"
    assert gateway.kwargs["api_version"] == "2024-01-01"
    assert gateway.kwargs["prompt_cache_enabled"] is True
    client = gateway.kwargs["client_factory"]()
    try:
        assert str(client.base_url) == "https://api.anthropic.com"
    finally:
        await client.aclose()


def test_anthropic_gateway_defaults_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LISA_ANTHROPIC_VERSION", raising=False)

    provider = HostGatewayProvider(
        runtime=_runtime(config=SimpleNamespace(prompt_cache_enabled=False)),
        discovered_provider=_GatewayClasses({"anthropic"}),
    )
    gateway = provider.gateway_for("anthropic")

    assert isinstance(gateway, _CapturingGateway)
    assert gateway.kwargs["api_key"] == ""  # missing key still constructs
    assert gateway.kwargs["api_version"] == "2023-06-01"
    assert gateway.kwargs["prompt_cache_enabled"] is False


def test_google_gateway_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LISA_GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LISA_GOOGLE_API_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_API_BASE_URL", raising=False)

    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=_GatewayClasses({"google"}),
    )
    gateway = provider.gateway_for("google")

    assert isinstance(gateway, _CapturingGateway)
    assert gateway.kwargs["api_key"] == ""
    assert gateway.kwargs["base_url"] == "https://generativelanguage.googleapis.com"


@pytest.mark.asyncio
async def test_openai_gateway_client_factory_missing_key_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LISA_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=_GatewayClasses({"openai"}),
    )
    gateway = provider.gateway_for("openai")

    assert isinstance(gateway, _CapturingGateway)
    with pytest.raises(ValueError, match="OpenAI is not configured"):
        gateway.kwargs["client_factory"]()


@pytest.mark.asyncio
async def test_azure_gateway_constructed_with_dual_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_AZURE_OPENAI_ENDPOINT", "https://ep.openai.azure.com")
    monkeypatch.setenv("LISA_AZURE_OPENAI_API_KEY", "ak")

    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=_GatewayClasses({"azure_openai"}),
    )
    gateway = provider.gateway_for("azure_openai")

    assert isinstance(gateway, _CapturingGateway)
    client = gateway.kwargs["complete_client_factory"]()
    try:
        assert str(client.base_url).startswith("https://ep.openai.azure.com")
    finally:
        await client.close()


def test_ollama_gateway_factory_selected() -> None:
    provider = HostGatewayProvider(
        runtime=_runtime(),
        discovered_provider=_GatewayClasses({"ollama"}),
    )
    gateway = provider.gateway_for("ollama")

    assert isinstance(gateway, _CapturingGateway)
    assert callable(gateway.kwargs["client_factory"])
