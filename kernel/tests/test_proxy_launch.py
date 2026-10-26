"""Regression tests for pack-contributed provider proxy declarations."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import httpx
import pytest
from kernel.mcp_launch.policy import RuntimeKind
from kernel.proxy_launch.compiler import ProxyClientConfig, compile_proxy_config
from kernel.proxy_launch.manifest import (
    ProxyServerManifestError,
    load_proxy_server_manifest,
)
from kernel.proxy_launch.policy import (
    ALL_PROVIDERS,
    ProviderProxyOptions,
    ProviderValueSource,
    ProxyProtocol,
    ProxyServerSpec,
    ProxyStatusSpec,
)
from kernel.proxy_launch.pool import ProxyConnectionError, ProxyPool, _select_stats_fields

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

_STATS_SERVER = """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys

class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == '/readyz':
      self.send_response(200)
      self.end_headers()
      return
    body = json.dumps({
      'savings': {'total_tokens': 1234},
      'cost': {'total_saved_usd': 7.5, 'nested': [1]},
      'requests': {'total': 42, 'failed_by_provider': {'x': 1}},
      'tokens': 999999,
      'list_field': [1, 2, 3],
    }).encode()
    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Content-Length', str(len(body)))
    self.end_headers()
    self.wfile.write(body)

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


def test_compile_proxy_config_missing_binary_returns_none() -> None:
    """An empty BINARY command_path means the optional binary is not installed.

    Pack manifests point proxies at opt-in third-party binaries via
    ``command_path: ${VAR}``; an unset var expands to "" and the pack
    contract says the proxy degrades to absent, not "startup failed"
    (the 2026-09-21 headroom incident: ``lisa mcp`` could not start at all).
    """
    spec = ProxyServerSpec(
        proxy_id="headroom",
        runtime_kind=RuntimeKind.BINARY,
        command_path="",  # e.g. ${HEADROOM_MCP_BIN} unset
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama",),
        port=18787,
    )

    assert compile_proxy_config(spec) is None


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


def test_load_proxy_manifest_parses_status_block(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: [ollama]
    port: 18787
    status:
      url_path: /stats
      fields: [tokens_saved_1h, savings_pct_1h, lifetime]
""",
        encoding="utf-8",
    )

    specs = load_proxy_server_manifest(manifest)

    assert len(specs) == 1
    status = specs[0].status_spec
    assert status is not None
    assert status.url_path == "/stats"
    assert status.fields == ("tokens_saved_1h", "savings_pct_1h", "lifetime")


def test_load_proxy_manifest_status_block_defaults_url_path(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: [ollama]
    port: 18787
    status: {}
""",
        encoding="utf-8",
    )

    specs = load_proxy_server_manifest(manifest)

    status = specs[0].status_spec
    assert status is not None
    assert status.url_path == "/stats"
    assert status.fields == ()


def test_load_proxy_manifest_accepts_all_providers_sentinel(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: [all]
    port: 18787
    provider_overrides:
      all:
        env:
          OPENAI_TARGET_API_URL:
            value_from: direct_base_url
""",
        encoding="utf-8",
    )

    specs = load_proxy_server_manifest(manifest)

    assert specs[0].providers == ("all",)
    assert specs[0].provider_options[ALL_PROVIDERS].env["OPENAI_TARGET_API_URL"] is (
        ProviderValueSource.DIRECT_BASE_URL
    )


def test_load_proxy_manifest_rejects_all_mixed_with_named_providers(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: [all, ollama]
    port: 18787
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match=r"must be the only"):
        load_proxy_server_manifest(manifest)


@pytest.mark.parametrize("url_path", ["stats", "//outside.example/stats"])
def test_load_proxy_manifest_rejects_relative_status_url_path(
    tmp_path: Path, url_path: str
) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        f"""
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: [ollama]
    port: 18787
    status:
      url_path: {url_path}
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match=r"status\.url_path must start with '/'"):
        load_proxy_server_manifest(manifest)


@pytest.mark.parametrize(
    "fields", ["savings_pct_1h", "null", "[1]", "['']", "[savings..tokens]", "[.tokens]"]
)
def test_load_proxy_manifest_rejects_non_list_status_fields(tmp_path: Path, fields: str) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        f"""
proxies:
  - proxy_id: headroom
    runtime_kind: binary
    command_path: /opt/headroom
    protocol: openai
    providers: [ollama]
    port: 18787
    status:
      url_path: /stats
      fields: {fields}
""",
        encoding="utf-8",
    )

    with pytest.raises(ProxyServerManifestError, match=r"status\.fields must be a list"):
        load_proxy_server_manifest(manifest)


def test_proxy_stats_exposes_nothing_without_an_allowlist() -> None:
    assert _select_stats_fields({"recent_request": {"prompt": "private"}}, ()) == {}


def test_compile_proxy_config_carries_status_spec() -> None:
    spec = ProxyServerSpec(
        proxy_id="headroom",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama",),
        port=18787,
        status_spec=ProxyStatusSpec(
            url_path="/stats",
            fields=("tokens_saved_1h", "savings_pct_1h"),
        ),
    )

    config = compile_proxy_config(spec)

    assert config.status_url_path == "/stats"
    assert config.status_fields == ("tokens_saved_1h", "savings_pct_1h")


def test_compile_proxy_config_without_status_spec_defaults_none() -> None:
    spec = ProxyServerSpec(
        proxy_id="plain",
        runtime_kind=RuntimeKind.BINARY,
        command_path="/opt/headroom",
        protocol=ProxyProtocol.OPENAI,
        providers=("ollama",),
        port=18787,
    )

    config = compile_proxy_config(spec)

    assert config.status_url_path is None
    assert config.status_fields == ()


@pytest.mark.asyncio
async def test_proxy_pool_health_view_reports_warm_failed_and_cold() -> None:
    """health() maps ready/failed/unattempted proxies to a sync liveness view."""
    config = ProxyClientConfig(
        proxy_id="never-started",
        command=sys.executable,
        args=["-c", "pass"],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=_unused_port(),
        base_url="http://127.0.0.1:1/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=0.1,
        status_url_path="/stats",
        status_fields=("tokens_saved_1h",),
        adopt_existing=False,
        warm=True,
        required=False,
    )
    pool = ProxyPool([config])
    try:
        await pool.start()  # not required: start failure recorded, not raised

        assert pool.failed_proxy_ids == {"never-started"}
        views = pool.health()
        assert len(views) == 1
        view = views[0]
        assert view.proxy_id == "never-started"
        assert view.status == "failed"
        assert view.status_url_path == "/stats"
        assert view.status_fields == ("tokens_saved_1h",)
        assert view.stats is None

        # A proxy that never got a start attempt (pool fresh, config warm)
        # reads as skipped before start() runs.
        fresh = ProxyPool(
            [
                ProxyClientConfig(
                    proxy_id="fresh",
                    command=sys.executable,
                    args=["-c", "pass"],
                    env={},
                    cwd=None,
                    protocol="openai",
                    providers=("ollama",),
                    port=_unused_port(),
                    base_url="http://127.0.0.1:2/v1",
                    readiness_path="/readyz",
                    readiness_timeout_seconds=1,
                    adopt_existing=False,
                    warm=True,
                    required=False,
                )
            ]
        )
        fresh_views = {v.proxy_id: v for v in fresh.health()}
        assert fresh_views["fresh"].status == "skipped"
        await fresh.aclose()
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_proxy_pool_fetch_stats_returns_declared_fields() -> None:
    """fetch_stats() GETs the declared path and selects declared fields.

    Fields are dotted paths: a scalar path contributes one entry, an object
    path contributes its scalar members as ``<path>.<member>``.
    """
    port = _unused_port()
    config = ProxyClientConfig(
        proxy_id="stats-proxy",
        command=sys.executable,
        args=["-c", _STATS_SERVER, str(port)],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=3,
        status_url_path="/stats",
        status_fields=("savings.total_tokens", "cost.total_saved_usd", "requests"),
        adopt_existing=False,
        warm=True,
        required=True,
    )
    pool = ProxyPool([config])
    try:
        await pool.start()

        stats = await pool.fetch_stats("stats-proxy")
        assert stats is not None
        assert stats["savings.total_tokens"] == 1234
        assert stats["cost.total_saved_usd"] == 7.5
        # An object path flattens to its scalar members only.
        assert stats["requests.total"] == 42
        assert "requests.failed_by_provider" not in stats
        # Undeclared keys are dropped.
        assert "tokens" not in stats
        assert "list_field" not in stats
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_proxy_pool_fetch_stats_skips_missing_paths() -> None:
    """A declared path that misses or names a non-object is skipped, not fatal."""
    port = _unused_port()
    config = ProxyClientConfig(
        proxy_id="sparse-proxy",
        command=sys.executable,
        args=["-c", _STATS_SERVER, str(port)],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=3,
        status_url_path="/stats",
        status_fields=("missing.path", "savings.total_tokens", "list_field"),
        adopt_existing=False,
        warm=True,
        required=True,
    )
    pool = ProxyPool([config])
    try:
        await pool.start()

        stats = await pool.fetch_stats("sparse-proxy")
        assert stats is not None
        assert stats["savings.total_tokens"] == 1234
        assert "missing.path" not in stats
        assert "list_field" not in stats
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_proxy_pool_fetch_stats_none_without_declaration() -> None:
    port = _unused_port()
    config = ProxyClientConfig(
        proxy_id="bare",
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
    try:
        await pool.start()
        assert pool.health()[0].status == "warm"
        assert await pool.fetch_stats("bare") is None
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_proxy_pool_stats_errors_are_explicit_and_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _unused_port()
    config = ProxyClientConfig(
        proxy_id="stats-error",
        command=sys.executable,
        args=["-c", _STATS_SERVER, str(port)],
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
        status_url_path="/stats",
        status_fields=("savings.total_tokens",),
    )
    pool = ProxyPool([config])
    original = ProxyPool._get_json

    async def fail_stats(port: int, path: str) -> dict[str, object]:
        raise httpx.ConnectError("telemetry connection lost")

    try:
        await pool.start()
        monkeypatch.setattr(ProxyPool, "_get_json", staticmethod(fail_stats))
        assert await pool.fetch_stats(config.proxy_id) is None
        assert "telemetry connection lost" in pool.stats_error(config.proxy_id)
        assert pool.health()[0].status == "warm"
        monkeypatch.setattr(ProxyPool, "_get_json", staticmethod(original))
        assert await pool.fetch_stats(config.proxy_id) == {"savings.total_tokens": 1234}
        assert pool.stats_error(config.proxy_id) == ""
    finally:
        await pool.aclose()


def test_compile_proxy_registry_skips_missing_binary() -> None:
    """compile_proxy_registry filters proxies whose binary is not installed.

    A missing optional binary must not abort the whole compile: the healthy
    proxies still land in the result and the missing one is warned about.
    """
    from kernel.proxy_launch.compiler import compile_proxy_registry
    from kernel.proxy_launch.policy import ProxyServerRegistry

    registry = ProxyServerRegistry()
    registry.register(
        ProxyServerSpec(
            proxy_id="absent",
            runtime_kind=RuntimeKind.BINARY,
            command_path="",  # e.g. ${HEADROOM_MCP_BIN} unset
            protocol=ProxyProtocol.OPENAI,
            providers=("ollama",),
            port=18787,
        )
    )
    registry.register(
        ProxyServerSpec(
            proxy_id="present",
            runtime_kind=RuntimeKind.BINARY,
            command_path="/opt/headroom",
            protocol=ProxyProtocol.OPENAI,
            providers=("ollama",),
            port=18788,
        )
    )

    configs = compile_proxy_registry(registry)

    assert [config.proxy_id for config in configs] == ["present"]


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


@pytest.mark.asyncio
async def test_proxy_pool_detects_dead_process_and_respawns() -> None:
    """route_alive/respawn cover the mid-session proxy death.

    Reproduction of the 2026-10-26 headroom death: the proxy exited
    mid-session; nothing ever re-checked route liveness, so the host's
    cached provider client kept targeting the dead loopback port and
    every model call hung until the stream idle watchdog fired. The pool
    now exposes a sync liveness check, drops the dead route on request,
    and can respawn the proxy once with the same readiness bounds as
    startup.
    """
    port = _unused_port()
    config = ProxyClientConfig(
        proxy_id="dying",
        command=sys.executable,
        args=["-c", _READY_SERVER, str(port)],
        env={},
        cwd=None,
        protocol="openai",
        providers=("ollama",),
        port=port,
        base_url=f"http://127.0.0.1:{port}/v1",
        readiness_path="/readyz",
        readiness_timeout_seconds=5,
        adopt_existing=False,
        warm=True,
        required=False,
    )
    pool = ProxyPool([config])
    try:
        await pool.start()
        assert pool.route_alive("ollama") is True

        # Simulate the mid-session death: kill and reap the child.
        process = pool._processes["dying"]
        process.kill()
        await process.wait()

        assert pool.route_alive("ollama") is False

        pool.drop_route("ollama")
        assert pool.route_alive("ollama") is False
        assert pool.routes.get("ollama") is None

        # Respawn comes back warm on the same port.
        assert await pool.respawn("ollama") is True
        assert pool.route_alive("ollama") is True
        assert pool.routes["ollama"].proxy_id == "dying"

        # One bounded attempt: a provider with no config never respawns.
        assert await pool.respawn("unknown-provider") is False
    finally:
        await pool.aclose()
