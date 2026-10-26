"""Tests for host-owned direct provider endpoint resolution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.proxy_launch.pool import ProxyRoute
from runtime.config import HostConfig
from runtime.model_manifest import load_model_manifest
from runtime.provider_endpoints import (
    direct_provider_base_url,
    direct_provider_base_urls,
    normalize_ollama_host_to_base_url,
)
from runtime.runtime import HostRuntime
from runtime.testing import build_test_runtime


def _make_runtime(tmp_path: Path, **config_kwargs: Any) -> HostRuntime:
    return build_test_runtime(tmp_path, models_yaml="", **config_kwargs)


def test_normalize_ollama_host_appends_v1_once() -> None:
    assert normalize_ollama_host_to_base_url("") == ""
    assert normalize_ollama_host_to_base_url("127.0.0.1:11434") == "http://127.0.0.1:11434/v1/"
    assert normalize_ollama_host_to_base_url("http://127.0.0.1:11434/v1/") == (
        "http://127.0.0.1:11434/v1/"
    )


def test_default_direct_endpoints_without_configuration(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    assert direct_provider_base_url(runtime, "ollama") == "http://localhost:11434/v1/"
    assert direct_provider_base_url(runtime, "openai") == "https://api.openai.com/v1/"
    assert direct_provider_base_url(runtime, "openrouter") == "https://openrouter.ai/api/v1"
    assert direct_provider_base_url(runtime, "anthropic") == "https://api.anthropic.com"


def test_environment_overrides_direct_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_OLLAMA_HOST", "127.0.0.1:11500")
    monkeypatch.setenv("LISA_OPENROUTER_BASE_URL", "https://custom.example.com/api/v1")
    monkeypatch.setenv("LISA_ANTHROPIC_BASE_URL", "https://anthropic.example.com")
    runtime = _make_runtime(tmp_path)

    assert direct_provider_base_url(runtime, "ollama") == "http://127.0.0.1:11500/v1/"
    assert direct_provider_base_url(runtime, "openrouter") == "https://custom.example.com/api/v1"
    assert direct_provider_base_url(runtime, "anthropic") == "https://anthropic.example.com"


def test_direct_endpoints_bypass_proxy_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_OLLAMA_HOST", "127.0.0.1:11434")
    runtime = _make_runtime(tmp_path)
    runtime.resources._proxy_routes["ollama"] = ProxyRoute(
        proxy_id="headroom",
        protocol="openai",
        base_url="http://127.0.0.1:18787/v1",
    )

    assert direct_provider_base_url(runtime, "ollama") == "http://127.0.0.1:11434/v1/"


def _write_manifest(tmp_path: Path, models_yaml: str) -> None:
    manifest_dir = tmp_path / "config"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "models.yaml").write_text(models_yaml, encoding="utf-8")


def test_manifest_single_base_url_wins_over_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LISA_OLLAMA_HOST", "127.0.0.1:11434")
    runtime = _make_runtime(tmp_path)
    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: grok}",
                "models:",
                "  - model_id: grok",
                "    provider: openai",
                "    base_url: https://api.x.ai/v1",
            ]
        )
        + "\n",
    )
    runtime.session.manifest = load_model_manifest(tmp_path / "config" / "models.yaml")

    assert direct_provider_base_url(runtime, "openai") == "https://api.x.ai/v1"


def test_manifest_base_url_env_is_resolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_runtime(tmp_path)
    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: grok}",
                "models:",
                "  - model_id: grok",
                "    provider: openai",
                "    base_url_env: XAI_BASE_URL",
            ]
        )
        + "\n",
    )
    runtime.session.manifest = load_model_manifest(tmp_path / "config" / "models.yaml")

    monkeypatch.setenv("XAI_BASE_URL", "https://env.x.ai/v1")
    assert direct_provider_base_url(runtime, "openai") == "https://env.x.ai/v1"


def test_manifest_mixed_base_urls_raise(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: grok}",
                "models:",
                "  - model_id: grok",
                "    provider: openai",
                "    base_url: https://api.x.ai/v1",
                "  - model_id: grok-eu",
                "    provider: openai",
                "    base_url: https://eu.x.ai/v1",
            ]
        )
        + "\n",
    )
    runtime.session.manifest = load_model_manifest(tmp_path / "config" / "models.yaml")

    with pytest.raises(ValueError, match="mixed base URLs"):
        direct_provider_base_url(runtime, "openai")


def test_unknown_provider_direct_endpoint_raises(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    with pytest.raises(ValueError, match="No direct base URL"):
        direct_provider_base_url(runtime, "not-a-provider")


def test_direct_provider_base_urls_resolves_sets(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    resolved = direct_provider_base_urls(runtime, ["ollama", "openai"])

    assert resolved == {
        "ollama": "http://localhost:11434/v1/",
        "openai": "https://api.openai.com/v1/",
    }


def test_runtime_without_manifest_uses_defaults(tmp_path: Path) -> None:
    config = HostConfig.from_env(data_dir=str(tmp_path))
    runtime = HostRuntime(config)

    assert direct_provider_base_url(runtime, "ollama") == "http://localhost:11434/v1/"


def test_defaults_come_from_the_gateway_seam_not_a_host_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host keeps no provider→endpoint table.

    A gateway pack that declares an endpoint for a provider the SDK never
    shipped is resolved immediately, and one that declares none is reported
    as unknown — the host does not fall back to a built-in of its own.
    """
    import kernel.provider_gateway_discovery as discovery

    runtime = _make_runtime(tmp_path)

    class _SeamfulEntry:
        pack_name = "third-party-gateways"

        def gateway_for(self, provider: str) -> None:
            return None

        def direct_base_url_for(self, provider: str) -> str | None:
            return "https://custom.internal/v1/" if provider == "customwire" else None

    fake_entry = SimpleNamespace(provider=_SeamfulEntry(), pack_name="third-party-gateways")
    monkeypatch.setattr(discovery, "discover_provider_gateway_factories", lambda: [fake_entry])

    assert direct_provider_base_url(runtime, "customwire") == "https://custom.internal/v1/"
    with pytest.raises(ValueError, match="No direct base URL"):
        direct_provider_base_url(runtime, "openai")


def test_host_config_override_wins_over_the_gateway_default(tmp_path: Path) -> None:
    """A configured override is a host setting and outranks the seam default."""
    config = HostConfig.from_env(
        data_dir=str(tmp_path), provider_base_urls={"openai": "https://proxy.internal/v1"}
    )
    runtime = HostRuntime(config)

    assert direct_provider_base_url(runtime, "openai") == "https://proxy.internal/v1"


def test_manifest_endpoint_outranks_a_host_override(tmp_path: Path) -> None:
    """The manifest is per-session truth, so it wins over host configuration."""
    _write_manifest(
        tmp_path,
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: grok}",
                "models:",
                "  - model_id: grok",
                "    provider: openai",
                "    base_url: https://api.x.ai/v1",
            ]
        )
        + "\n",
    )
    config = HostConfig.from_env(
        data_dir=str(tmp_path), provider_base_urls={"openai": "https://proxy.internal/v1"}
    )
    runtime = HostRuntime(config)
    runtime.session.manifest = load_model_manifest(tmp_path / "config" / "models.yaml")

    assert direct_provider_base_url(runtime, "openai") == "https://api.x.ai/v1"
