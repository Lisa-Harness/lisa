"""Runtime tests for ``providers: [all]`` wildcard expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.mcp_launch.policy import RuntimeKind
from kernel.model_manifest import ManifestModelEntry, ModelManifest
from kernel.proxy_launch.manifest import load_proxy_server_manifest
from kernel.proxy_launch.policy import (
    ALL_PROVIDERS,
    ProviderValueSource,
    ProxyProtocol,
    ProxyServerRegistry,
    ProxyServerSpec,
)
from runtime.config import HostConfig
from runtime.provider_endpoints import direct_provider_base_urls
from runtime.runtime import HostRuntime

from runtime import mcp_pool

_YAML = """
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: {providers}
    port: {port}
    provider_overrides:
      all:
        env:
          OPENAI_TARGET_API_URL:
            value_from: direct_base_url
"""


def _runtime_with_models(tmp_path: Path, providers: list[str]) -> HostRuntime:
    runtime = HostRuntime(HostConfig.from_env(data_dir=str(tmp_path)))
    runtime.session.manifest = ModelManifest(
        models=tuple(
            ManifestModelEntry(key=p, model_id=f"{p}/m", provider=p, display_name=p)
            for p in providers
        )
    )
    return runtime


def _wildcard_spec() -> ProxyServerSpec:
    return ProxyServerSpec(
        proxy_id="headroom",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=(ALL_PROVIDERS,),
        port=18787,
        provider_options={
            ALL_PROVIDERS: __import__(
                "kernel.proxy_launch.policy", fromlist=["ProviderProxyOptions"]
            ).ProviderProxyOptions(
                env={"OPENAI_TARGET_API_URL": ProviderValueSource.DIRECT_BASE_URL}
            )
        },
    )


def test_wildcard_expands_to_manifest_providers(tmp_path: Path) -> None:
    runtime = _runtime_with_models(tmp_path, ["ollama", "openrouter"])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    specs = {s.proxy_id: s for s in registry.all_specs()}
    assert sorted(specs) == ["headroom.ollama", "headroom.openrouter"]
    assert specs["headroom.ollama"].providers == ("ollama",)
    assert specs["headroom.openrouter"].providers == ("openrouter",)
    # Sequential ports with distinct values so sibling processes don't collide.
    ports = {specs[pid].port for pid in specs}
    assert sorted(ports) == [18787, 18788]
    # The wildcard overlay propagates into each provider's own options map.
    overlay = specs["headroom.ollama"].provider_options["ollama"]
    assert overlay.env["OPENAI_TARGET_API_URL"] is ProviderValueSource.DIRECT_BASE_URL


def test_wildcard_with_no_manifest_providers_expands_to_nothing(tmp_path: Path) -> None:
    runtime = _runtime_with_models(tmp_path, [])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert registry.all_specs() == []


def test_named_providers_are_untouched(tmp_path: Path) -> None:
    runtime = _runtime_with_models(tmp_path, ["ollama", "openrouter"])
    registry = ProxyServerRegistry()
    registry.register(
        ProxyServerSpec(
            proxy_id="headroom",
            runtime_kind=RuntimeKind.BINARY,
            command_path="/opt/headroom",
            protocol=ProxyProtocol.OPENAI,
            providers=("ollama",),
            port=18787,
        )
    )

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert [s.providers for s in registry.all_specs()] == [("ollama",)]


def test_all_sentinel_parses_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(_YAML.format(providers="[all]", port=18787), encoding="utf-8")

    specs = load_proxy_server_manifest(manifest)

    assert specs[0].providers == (ALL_PROVIDERS,)
    assert specs[0].provider_options[ALL_PROVIDERS].env["OPENAI_TARGET_API_URL"] is (
        ProviderValueSource.DIRECT_BASE_URL
    )


def test_all_mixed_with_named_providers_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(_YAML.format(providers="[all, ollama]", port=18787), encoding="utf-8")

    from kernel.proxy_launch.manifest import ProxyServerManifestError

    with pytest.raises(ProxyServerManifestError, match=r"must be the only"):
        load_proxy_server_manifest(manifest)


def test_wildcard_skips_providers_no_proxy_can_serve(tmp_path: Path) -> None:
    """A wildcard must never abort startup on a credential-only provider.

    Regression for the startup abort: expansion used to map ``all`` onto every
    manifest provider and then resolve a direct base URL for each, so any
    manifest containing a ``google``/``aws``/``azure_openai`` model raised
    ``ValueError`` out of ``build_runtime``. Whether a proxy can serve a
    provider is gateway knowledge, so the answer comes from the
    ``ProviderGatewayProvider`` seam (``proxy_protocol_for``), which reports
    ``None`` for the credential-only gateways.
    """
    runtime = _runtime_with_models(tmp_path, ["aws", "google", "ollama", "openrouter"])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert sorted(s.proxy_id for s in registry.all_specs()) == [
        "headroom.ollama",
        "headroom.openrouter",
    ]
    # Ports are allocated over the surviving set only — a skipped provider
    # must not burn a port number.
    assert sorted(s.port for s in registry.all_specs()) == [18787, 18788]
    # And the compile path that follows expansion no longer raises.
    providers = sorted({p for s in registry.all_specs() for p in s.providers})
    assert set(direct_provider_base_urls(runtime, providers)) == {"ollama", "openrouter"}


def test_wildcard_skips_providers_of_another_wire_protocol(tmp_path: Path) -> None:
    """An ``openai`` proxy has nothing to serve for an ``anthropic`` model."""
    runtime = _runtime_with_models(tmp_path, ["anthropic", "ollama"])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert [s.proxy_id for s in registry.all_specs()] == ["headroom.ollama"]


def test_wildcard_uses_the_gateway_seam_not_a_host_provider_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expansion asks the seam, so a new gateway pack needs no host edit.

    The host keeps no provider list of its own: a gateway pack that reports a
    proxy protocol for a provider the SDK does not ship is routable
    immediately, and one that reports nothing is skipped even if the SDK would
    have answered otherwise.
    """
    runtime = _runtime_with_models(tmp_path, ["customwire"])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    class _Seam:
        def gateway_for(self, provider: str) -> None:
            return None

        def proxy_protocol_for(self, provider: str) -> str | None:
            return "openai" if provider == "customwire" else None

    monkeypatch.setattr(mcp_pool, "_active_provider_gateway_provider", lambda _rt: _Seam())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert [s.proxy_id for s in registry.all_specs()] == ["headroom.customwire"]


def test_provider_gateway_seam_without_proxy_support_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway provider predating the seam is never treated as routable.

    An older third-party provider has no ``proxy_protocol_for``. Reading that
    as "everything is routable" would force a proxy onto a gateway that builds
    its client from credentials; reading it as "nothing is routable" only
    leaves the proxy unexpanded — the fail-closed posture.
    """
    runtime = _runtime_with_models(tmp_path, ["ollama"])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    class _LegacySeam:
        def gateway_for(self, provider: str) -> None:
            return None

    monkeypatch.setattr(mcp_pool, "_active_provider_gateway_provider", lambda _rt: _LegacySeam())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert registry.all_specs() == []


def test_provider_gateway_seam_with_no_active_provider_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No gateway pack installed means no proxy can serve any provider."""
    runtime = _runtime_with_models(tmp_path, ["ollama"])
    registry = ProxyServerRegistry()
    registry.register(_wildcard_spec())

    monkeypatch.setattr(mcp_pool, "_active_provider_gateway_provider", lambda _rt: None)

    mcp_pool._expand_wildcard_providers(registry, runtime)

    assert registry.all_specs() == []


def test_wildcard_does_not_clobber_an_explicit_sibling(tmp_path: Path) -> None:
    """An author's explicit ``<proxy_id>.<provider>`` spec wins over expansion.

    Expansion used to ``update()`` siblings by id, silently replacing an
    explicitly declared proxy (and its port, args, and overlays) with the
    generated one.
    """
    runtime = _runtime_with_models(tmp_path, ["ollama", "openrouter"])
    registry = ProxyServerRegistry()
    explicit = ProxyServerSpec(
        proxy_id="headroom.ollama",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom-explicit",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama",),
        port=19100,
    )
    registry.register(explicit)
    registry.register(_wildcard_spec())

    mcp_pool._expand_wildcard_providers(registry, runtime)

    specs = {s.proxy_id: s for s in registry.all_specs()}
    assert specs["headroom.ollama"] is explicit
    assert specs["headroom.ollama"].port == 19100
    assert specs["headroom.ollama"].command_path == "/opt/headroom-explicit"
    assert "headroom.openrouter" in specs
