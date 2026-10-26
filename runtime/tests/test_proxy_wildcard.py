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
