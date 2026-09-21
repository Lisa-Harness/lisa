"""Regression tests for pack-contributed provider proxy declarations."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
from kernel.mcp_launch.policy import RuntimeKind
from kernel.proxy_launch.compiler import ProxyClientConfig, compile_proxy_config
from kernel.proxy_launch.manifest import (
    ProxyServerManifestError,
    load_proxy_server_manifest,
)
from kernel.proxy_launch.policy import (
    ProviderProxyOptions,
    ProviderValueSource,
    ProxyProtocol,
    ProxyServerSpec,
)
from kernel.proxy_launch.pool import ProxyConnectionError, ProxyPool

_READY_SERVER = """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import time

if len(sys.argv) > 2:
    time.sleep(float(sys.argv[2]))

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    self.send_response(200)
    self.end_headers()

  def log_message(self, format, *args):
    return

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_load_proxy_manifest_parses_openai_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_BIN", "/opt/headroom")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: ${HEADROOM_BIN}
    cli_args: [proxy]
    protocol: openai
    providers: [ollama, openrouter]
    port: 18787
    listen:
      port_env: HEADROOM_PORT
      base_path: /v1
    readiness:
      url_path: /readyz
      timeout_seconds: 3
""",
        encoding="utf-8",
    )

    specs = load_proxy_server_manifest(manifest)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.command_path == "/opt/headroom"
    assert spec.protocol is ProxyProtocol.OPENAI
    assert spec.providers == ("ollama", "openrouter")
    assert spec.port_env == "HEADROOM_PORT"
    assert spec.base_path == "/v1"


def test_load_proxy_manifest_rejects_invalid_protocol(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: invalid
    runtime_kind: binary
    command_path: /bin/true
    protocol: bespoke
    providers: [ollama]
    port: 18787
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match="invalid protocol"):
        load_proxy_server_manifest(manifest)


def test_load_proxy_manifest_parses_provider_overrides(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: layered
    runtime_kind: binary
    command_path: /opt/headroom
    cli_args: [proxy]
    protocol: openai
    providers: [ollama, openai]
    port: 18787
    provider_overrides:
      ollama:
        cli_args: [--ollama-switch]
        env:
          OPENAI_TARGET_API_URL:
            value_from: direct_base_url
      openai:
        env:
          OPENAI_API_KEY: sk-literal
""",
        encoding="utf-8",
    )

    specs = load_proxy_server_manifest(manifest)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.provider_options["ollama"].cli_args == ("--ollama-switch",)
    assert (
        spec.provider_options["ollama"].env["OPENAI_TARGET_API_URL"]
        is ProviderValueSource.DIRECT_BASE_URL
    )
    assert spec.provider_options["openai"].env["OPENAI_API_KEY"] == "sk-literal"


def test_load_proxy_manifest_rejects_override_for_undeclared_provider(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: stray
    runtime_kind: binary
    command_path: /bin/true
    protocol: openai
    providers: [ollama]
    port: 18787
    provider_overrides:
      openai:
        cli_args: [--switch]
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match="not among declared providers"):
        load_proxy_server_manifest(manifest)


def test_load_proxy_manifest_rejects_unknown_value_from(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: badsource
    runtime_kind: binary
    command_path: /bin/true
    protocol: openai
    providers: [ollama]
    port: 18787
    provider_overrides:
      ollama:
        env:
          TARGET:
            value_from: magic
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match="unknown value_from"):
        load_proxy_server_manifest(manifest)


def test_load_proxy_manifest_rejects_empty_provider_override(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: empty
    runtime_kind: binary
    command_path: /bin/true
    protocol: openai
    providers: [ollama]
    port: 18787
    provider_overrides:
      ollama: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match="is empty"):
        load_proxy_server_manifest(manifest)


def test_compile_proxy_config_injects_listen_port_env() -> None:
    spec = ProxyServerSpec(
        proxy_id="headroom",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama",),
        port=18787,
        port_env="HEADROOM_PORT",
    )

    config = compile_proxy_config(spec)

    assert config.command == "/opt/headroom"
    assert config.env["HEADROOM_PORT"] == "18787"
    assert config.base_url == "http://127.0.0.1:18787/v1"
    assert config.readiness_path == "/readyz"


def test_compile_proxy_config_layers_provider_overlays_in_provider_order() -> None:
    spec = ProxyServerSpec(
        proxy_id="layered",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama", "openai"),
        port=18787,
        cli_args=["proxy"],
        provider_options={
            "ollama": ProviderProxyOptions(
                cli_args=("--ollama-switch",),
                env={"OPENAI_TARGET_API_URL": ProviderValueSource.DIRECT_BASE_URL},
            ),
            "openai": ProviderProxyOptions(
                cli_args=("--openai-switch",),
                env={"OPENAI_API_KEY": "sk-literal"},
            ),
        },
    )

    config = compile_proxy_config(
        spec,
        provider_base_urls={"ollama": "http://localhost:11434/v1/"},
    )

    assert config.args == ["proxy", "--ollama-switch", "--openai-switch"]
    assert config.env["OPENAI_TARGET_API_URL"] == "http://localhost:11434/v1/"
    assert config.env["OPENAI_API_KEY"] == "sk-literal"


def test_compile_proxy_config_allows_identical_conflicting_env_writes() -> None:
    spec = ProxyServerSpec(
        proxy_id="agreeing",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama", "openai"),
        port=18787,
        provider_options={
            "ollama": ProviderProxyOptions(env={"TARGET": ProviderValueSource.DIRECT_BASE_URL}),
            "openai": ProviderProxyOptions(env={"TARGET": ProviderValueSource.DIRECT_BASE_URL}),
        },
    )

    config = compile_proxy_config(
        spec,
        provider_base_urls={
            "ollama": "http://localhost:11434/v1/",
            "openai": "http://localhost:11434/v1/",
        },
    )

    assert config.env["TARGET"] == "http://localhost:11434/v1/"


def test_compile_proxy_config_rejects_conflicting_env_writes() -> None:
    spec = ProxyServerSpec(
        proxy_id="conflicting",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama", "openai"),
        port=18787,
        provider_options={
            "ollama": ProviderProxyOptions(env={"TARGET": ProviderValueSource.DIRECT_BASE_URL}),
            "openai": ProviderProxyOptions(env={"TARGET": ProviderValueSource.DIRECT_BASE_URL}),
        },
    )

    with pytest.raises(ValueError, match="conflicting values"):
        compile_proxy_config(
            spec,
            provider_base_urls={
                "ollama": "http://localhost:11434/v1/",
                "openai": "https://api.openai.com/v1/",
            },
        )


def test_compile_proxy_config_requires_provider_base_url_for_source() -> None:
    spec = ProxyServerSpec(
        proxy_id="unresolved",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama",),
        port=18787,
        provider_options={
            "ollama": ProviderProxyOptions(env={"TARGET": ProviderValueSource.DIRECT_BASE_URL}),
        },
    )

    with pytest.raises(ValueError, match="no direct base URL available"):
        compile_proxy_config(spec)


@pytest.mark.asyncio
async def test_proxy_pool_registers_ready_route_and_closes_owned_process() -> None:
    port = _unused_port()
    config = ProxyClientConfig(
        proxy_id="test-proxy",
        command=sys.executable,
        args=["-c", _READY_SERVER, str(port)],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=3,
        adopt_existing=False,
        warm=True,
        required=True,
    )
    pool = ProxyPool([config])

    await pool.start()

    route = pool.route_for("ollama", "openai")
    assert route is not None
    assert route.proxy_id == "test-proxy"
    process = pool._processes["test-proxy"]
    assert process.returncode is None

    await pool.aclose()

    assert process.returncode is not None
    assert pool.routes == {}


@pytest.mark.asyncio
async def test_proxy_pool_resolves_conflicts_in_manifest_order() -> None:
    first_port = _unused_port()
    second_port = _unused_port()
    first = ProxyClientConfig(
        proxy_id="first",
        command=sys.executable,
        args=["-c", _READY_SERVER, str(first_port), "0.3"],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=first_port,
        base_url=f"http://127.0.0.1:{first_port}/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=3,
        adopt_existing=False,
        warm=True,
        required=True,
    )
    second = ProxyClientConfig(
        proxy_id="second",
        command=sys.executable,
        args=["-c", _READY_SERVER, str(second_port)],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=second_port,
        base_url=f"http://127.0.0.1:{second_port}/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=3,
        adopt_existing=False,
        warm=True,
        required=True,
    )
    pool = ProxyPool([first, second])
    try:
        await pool.start()

        route = pool.route_for("ollama", "openai")
        assert route is not None
        assert route.proxy_id == "second"
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_proxy_pool_raises_for_required_proxy_that_exits() -> None:
    config = ProxyClientConfig(
        proxy_id="failing-proxy",
        command=sys.executable,
        args=["-c", "raise SystemExit(1)"],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=_unused_port(),
        base_url="http://127.0.0.1:18787/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=1,
        adopt_existing=False,
        warm=True,
        required=True,
    )

    with pytest.raises(
        ProxyConnectionError,
        match="Required proxy 'failing-proxy' failed to start",
    ):
        await ProxyPool([config]).start()
