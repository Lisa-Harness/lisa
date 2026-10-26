"""Tests for the mcp_servers.yaml manifest loader.

Covers:
- load_mcp_server_manifest: missing file returns empty.
- load_mcp_server_manifest: parses server declarations.
- ${VAR} env expansion at load time.
- disabled entries are skipped.
- data_dir_arg null vs custom flag vs default.
- HTTP server entries (runtime_kind: http) parse url + headers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.mcp_launch.policy import AuthKind, AuthSpec, RuntimeKind
from kernel.mcp_launch.server_manifest import (
    MCPServerManifestError,
    load_mcp_server_manifest,
)

# ── Manifest loader ─────────────────────────────────────────────────────────


def test_load_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_mcp_server_manifest(tmp_path / "nope.yaml") == []


def test_load_manifest_expands_against_explicit_env(tmp_path: Path) -> None:
    """``env=`` expands ${VAR} without touching the ambient environment.

    The host passes its resolved MCP environment explicitly; a variable
    present only in that mapping expands, and a variable present only in
    the ambient environment does not.
    """
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: explicit-env
    runtime_kind: http
    url: "https://${EXPLICIT_HOST}/mcp"
    headers:
      Authorization: "Bearer ${EXPLICIT_TOKEN}"
""",
        encoding="utf-8",
    )

    specs = load_mcp_server_manifest(
        manifest,
        env={"EXPLICIT_HOST": "example.internal", "EXPLICIT_TOKEN": "tok"},
    )

    assert specs[0].url == "https://example.internal/mcp"
    assert specs[0].headers["Authorization"] == "Bearer tok"


def test_load_manifest_parses_protocol_version_pin(tmp_path: Path) -> None:
    """``protocol_version:`` pins the HTTP server's protocol version.

    A server migrating to the modern stateless protocol (e.g. ``2026-07-28``)
    can declare its version so the pool adopts it directly at connect time
    instead of negotiating (whose ``initialize()`` fallback rejects the
    modern version the server itself advertises).
    """
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: modern-remote
    runtime_kind: http
    url: "https://mcp.example.internal/mcp"
    protocol_version: "2026-07-28"
""",
        encoding="utf-8",
    )

    specs = load_mcp_server_manifest(manifest)

    assert specs[0].protocol_version == "2026-07-28"


def test_load_manifest_rejects_protocol_version_on_non_http(tmp_path: Path) -> None:
    """``protocol_version`` is only meaningful for HTTP servers."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: stdio-server
    runtime_kind: npx
    npx_package: "@example/mcp"
    protocol_version: "2026-07-28"
""",
        encoding="utf-8",
    )

    with pytest.raises(MCPServerManifestError, match=r"protocol_version.*runtime_kind=http"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_rejects_empty_protocol_version(tmp_path: Path) -> None:
    """A set-but-empty ``protocol_version`` is a config error, not a pin."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: modern-remote
    runtime_kind: http
    url: "https://mcp.example.internal/mcp"
    protocol_version: ""
""",
        encoding="utf-8",
    )

    with pytest.raises(MCPServerManifestError, match="non-empty string"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_parses_servers(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: mem0
    runtime_kind: npx
    npx_package: "@mem0/mcp"
    data_dir_arg: null
    env:
      MEM0_API_KEY: "${MEM0_API_KEY_TEST}"
    tool_policy:
      denied_tools: [delete_memory]
  - server_id: letta
    runtime_kind: npx
    npx_package: "@letta/mcp"
    disabled: true
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    assert specs[0].server_id == "mem0"
    assert specs[0].runtime_kind is RuntimeKind.NPX
    assert specs[0].npx_package == "@mem0/mcp"
    assert specs[0].data_dir_arg is None
    assert specs[0].tool_policy.denied_tools == ["delete_memory"]


def test_load_manifest_parses_pinned_binary_download(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: lisa-storage-go
    runtime_kind: binary
    binary_download:
      repository: Lisa-Harness/lisa-go
      version: v0.1.0
      binary_name: lisa-storage
""",
        encoding="utf-8",
    )

    specs = load_mcp_server_manifest(manifest)

    assert specs[0].binary_download is not None
    assert specs[0].binary_download.repository == "Lisa-Harness/lisa-go"
    assert specs[0].binary_download.version == "v0.1.0"
    assert specs[0].binary_download.binary_name == "lisa-storage"


def test_load_manifest_allows_binary_download_without_version(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: lisa-storage-go
    runtime_kind: binary
    binary_download:
      repository: Lisa-Harness/lisa-go
      binary_name: lisa-storage
""",
        encoding="utf-8",
    )

    specs = load_mcp_server_manifest(manifest)

    assert specs[0].binary_download is not None
    assert specs[0].binary_download.version is None


@pytest.mark.parametrize(
    ("runtime_kind", "binary_download", "command_path", "message"),
    [
        (
            "python",
            "repository: org/server\n      version: v1.2.3\n      binary_name: tool",
            "",
            "requires runtime_kind=binary",
        ),
        (
            "binary",
            "repository: org/server\n      version: latest\n      binary_name: tool",
            "",
            "invalid binary_download",
        ),
        (
            "binary",
            "repository: org/server\n      version: v1.2.3\n      binary_name: ../tool",
            "",
            "invalid binary_download",
        ),
        (
            "binary",
            "repository: org/server\n      version: v1.2.3\n      binary_name: tool",
            "command_path: ./tool",
            "cannot both be set",
        ),
    ],
)
def test_load_manifest_rejects_invalid_binary_download(
    tmp_path: Path,
    runtime_kind: str,
    binary_download: str,
    command_path: str,
    message: str,
) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        f"""
servers:
  - server_id: downloader
    runtime_kind: {runtime_kind}
    {command_path}
    binary_download:
      {binary_download}
""",
        encoding="utf-8",
    )

    with pytest.raises(MCPServerManifestError, match=message):
        load_mcp_server_manifest(manifest)


def test_load_manifest_rejects_legacy_memory_section(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
memory:
  default_backend: mem0
servers: []
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="top-level 'memory' is no longer supported"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEM0_API_KEY_TEST", "secret123")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: mem0
    runtime_kind: npx
    npx_package: "@mem0/mcp"
    data_dir_arg: null
    env:
      MEM0_API_KEY: "${MEM0_API_KEY_TEST}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].env_policy.forced_overrides["MEM0_API_KEY"] == "secret123"


def test_load_manifest_expands_cli_args_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cli_args`` ${VAR} references are expanded at load time.

    Regression for the 2026-08-25 thread-read-empty incident: ``cli_args``
    were stored verbatim (``str(a)``), so the lisa-thread server received a
    literal ``${LISA_USER_SLUG}`` path and read a nonexistent store.
    """
    monkeypatch.setenv("LISA_THREAD_DIR_TEST", "/tmp/threads")
    monkeypatch.setenv("LISA_USER_SLUG_TEST", "alice")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: lisa-thread
    runtime_kind: python
    module: mcp_server.thread.server
    cli_args:
      - "--thread-dir"
      - "${LISA_THREAD_DIR_TEST}"
      - "--user-slug"
      - "${LISA_USER_SLUG_TEST}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].cli_args == [
        "--thread-dir",
        "/tmp/threads",
        "--user-slug",
        "alice",
    ]


def test_load_manifest_cli_args_unexpanded_var_becomes_empty(tmp_path: Path) -> None:
    """An unset ${VAR} in cli_args expands to empty string (not a literal ${VAR})."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: srv
    runtime_kind: python
    module: srv.server
    cli_args: ["--flag", "${DEFINITELY_UNSET_VAR_XYZ}"]
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].cli_args == ["--flag", ""]


def test_load_manifest_data_dir_arg_custom_flag(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: mem0-sh
    runtime_kind: uv
    package: mem0ai
    module: mem0.server.mcp
    data_dir_arg: "--store-dir"
    data_dir_subdir: "mem0"
    ensure_data_dir: true
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].data_dir_arg == "--store-dir"
    assert specs[0].data_dir_subdir == "mem0"
    assert specs[0].ensure_data_dir is True


def test_load_manifest_default_data_dir_arg(tmp_path: Path) -> None:
    """When data_dir_arg is omitted, it defaults to --data-dir (first-party)."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: custom
    runtime_kind: python
    module: custom.server
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].data_dir_arg == "--data-dir"


def test_load_manifest_workspace_root_arg(tmp_path: Path) -> None:
    """workspace_root_arg: null opts out; default is --workspace-root."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: optout
    runtime_kind: binary
    command_path: /usr/bin/tool
    workspace_root_arg: null
  - server_id: default
    runtime_kind: python
    module: custom.server
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].workspace_root_arg is None
    assert specs[1].workspace_root_arg == "--workspace-root"


def test_load_manifest_invalid_runtime_kind_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: bad
    runtime_kind: docker
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid runtime_kind"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_parses_http_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """runtime_kind: http parses url + headers with env expansion."""
    monkeypatch.setenv("MEM0_API_KEY_TEST", "secret123")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: mem0
    runtime_kind: http
    url: https://mcp.mem0.ai/mcp
    headers:
      Authorization: "Bearer ${MEM0_API_KEY_TEST}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    assert specs[0].server_id == "mem0"
    assert specs[0].runtime_kind is RuntimeKind.HTTP
    assert specs[0].url == "https://mcp.mem0.ai/mcp"
    assert specs[0].headers["Authorization"] == "Bearer secret123"


def test_load_manifest_http_requires_url(tmp_path: Path) -> None:
    """runtime_kind: http without url raises."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: mem0
    runtime_kind: http
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="url"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_http_url_expands_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """runtime_kind: http url supports ${VAR} expansion at load time."""
    monkeypatch.setenv("CLOUDSERVER_MCP_URL", "https://api.cloudservers.app/mcp")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: cloudservers
    runtime_kind: http
    url: "${CLOUDSERVER_MCP_URL}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    assert specs[0].runtime_kind is RuntimeKind.HTTP
    assert specs[0].url == "https://api.cloudservers.app/mcp"


def test_load_manifest_parses_bearer_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """auth: {type: bearer, token} parses with ${VAR} expansion."""
    monkeypatch.setenv("GITHUB_TOKEN_TEST", "ghp_secret")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: github
    runtime_kind: http
    url: https://mcp.github.com/mcp
    auth:
      type: bearer
      token: "${GITHUB_TOKEN_TEST}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    auth = specs[0].auth
    assert isinstance(auth, AuthSpec)
    assert auth.kind is AuthKind.BEARER
    assert auth.token == "ghp_secret"
    assert auth.is_authenticated() is True


def test_load_manifest_parses_oauth_pkce_auth(tmp_path: Path) -> None:
    """auth: {type: oauth2-pkce} parses client_id/scopes/auth_server."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: github
    runtime_kind: http
    url: https://mcp.github.com/mcp
    auth:
      type: oauth2-pkce
      client_id: my-app
      scopes: ["repo", "read:org"]
      auth_server: "https://github.com/login/oauth/authorize"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    auth = specs[0].auth
    assert auth.kind is AuthKind.OAUTH2_PKCE
    assert auth.client_id == "my-app"
    assert auth.scopes == ("repo", "read:org")
    assert auth.auth_server == "https://github.com/login/oauth/authorize"
    assert auth.is_authenticated() is True


def test_load_manifest_parses_header_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """auth: {type: header, header_name, token} parses with ${VAR} expansion."""
    monkeypatch.setenv("CLOUDSERVERS_TOKEN", "tok_abc123")
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: CloudServers
    runtime_kind: http
    url: https://api.cloudservers.app/mcp
    auth:
      type: header
      header_name: X-Scoped-Budget-Token
      token: "${CLOUDSERVERS_TOKEN}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    auth = specs[0].auth
    assert auth.kind is AuthKind.HEADER
    assert auth.header_name == "X-Scoped-Budget-Token"
    assert auth.token == "tok_abc123"
    assert auth.is_authenticated() is True


def test_load_manifest_header_auth_requires_header_name(tmp_path: Path) -> None:
    """auth: {type: header} without header_name raises."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: cs
    runtime_kind: http
    url: https://api.example.com/mcp
    auth:
      type: header
      token: "tok"
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="requires 'header_name'"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_header_auth_requires_token(tmp_path: Path) -> None:
    """auth: {type: header} without token raises."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: cs
    runtime_kind: http
    url: https://api.example.com/mcp
    auth:
      type: header
      header_name: X-Custom-Token
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="requires 'token'"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_bearer_auth_requires_token(tmp_path: Path) -> None:
    """auth: {type: bearer} without token raises."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: github
    runtime_kind: http
    url: https://mcp.github.com/mcp
    auth:
      type: bearer
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="requires 'token'"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_invalid_auth_type_raises(tmp_path: Path) -> None:
    """An unknown auth type raises a manifest error."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: github
    runtime_kind: http
    url: https://mcp.github.com/mcp
    auth:
      type: bogus
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="invalid auth type"):
        load_mcp_server_manifest(manifest)


# ── warm / required / disabled launch policy ────────────────────────────────


def test_load_manifest_parses_warm_and_required(tmp_path: Path) -> None:
    """warm and required fields are parsed onto the MCPServerSpec."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: lisa-context
    runtime_kind: venv_bin
    venv_dir: "/fake/venv"
    warm: true
    required: true
  - server_id: lisa-guard
    runtime_kind: venv_bin
    venv_dir: "/fake/venv"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    context = next(s for s in specs if s.server_id == "lisa-context")
    assert context.warm is True
    assert context.required is True
    search = next(s for s in specs if s.server_id == "lisa-guard")
    assert search.warm is False
    assert search.required is False


def test_load_manifest_disabled_skips_entry(tmp_path: Path) -> None:
    """disabled: true skips the entry entirely — it is not in the specs list."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: active
    runtime_kind: python
    module: active.server
  - server_id: inactive
    runtime_kind: python
    module: inactive.server
    disabled: true
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert len(specs) == 1
    assert specs[0].server_id == "active"


def test_compile_mcp_config_passes_warm_required(tmp_path: Path) -> None:
    """compile_mcp_config passes warm/required from the spec to the config."""
    from kernel.mcp_launch.compiler import compile_mcp_config
    from kernel.mcp_launch.policy import MCPServerSpec, RuntimeKind

    spec = MCPServerSpec(
        server_id="lisa-context",
        runtime_kind=RuntimeKind.VENV_BIN,
        venv_dir="/fake/venv",
        warm=True,
        required=True,
    )
    config = compile_mcp_config(spec)
    assert config.warm is True
    assert config.required is True


# ── default_available_tools ─────────────────────────────────────────────────


def test_default_available_tools_parsed(tmp_path: Path) -> None:
    """default_available_tools parses to a tuple of bare tool names."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    default_available_tools:
      - read_file
      - stat_file
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].default_available_tools == ("read_file", "stat_file")


def test_default_available_tools_defaults_empty(tmp_path: Path) -> None:
    """Omitting default_available_tools yields an empty tuple."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].default_available_tools == ()


def test_default_available_tools_rejects_namespaced_id(tmp_path: Path) -> None:
    """A namespaced id in default_available_tools is rejected (bare names only)."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    default_available_tools:
      - lisa-storage.read_file
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="must be a bare tool name"):
        load_mcp_server_manifest(manifest)


def test_default_available_tools_rejects_duplicate(tmp_path: Path) -> None:
    """Duplicate bare tool names in default_available_tools are rejected."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    default_available_tools:
      - read_file
      - read_file
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="duplicate default_available_tools"):
        load_mcp_server_manifest(manifest)


def test_default_available_tools_rejects_non_list(tmp_path: Path) -> None:
    """default_available_tools must be a list, not a scalar/mapping."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    default_available_tools: read_file
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="'default_available_tools' must be a list"):
        load_mcp_server_manifest(manifest)


def test_capability_parsed(tmp_path: Path) -> None:
    """Capability declarations include their canonical published tool set."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    capability: storage
    capability_tools: [read_file]
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].capability == "storage"
    assert specs[0].capability_tools == ("read_file",)
    assert specs[0].tool_map == {}


def test_capability_defaults_none(tmp_path: Path) -> None:
    """Omitting capability yields None (server_id-prefixed tools)."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].capability is None


def test_capability_rejects_invalid_slug(tmp_path: Path) -> None:
    """capability must be a bare lowercase slug — dots and uppercase are rejected."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: s
    capability: "Storage.V2"
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="invalid capability"):
        load_mcp_server_manifest(manifest)


def test_tool_map_parsed_with_capability(tmp_path: Path) -> None:
    """tool_map parses to a published→native mapping alongside capability."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-memory
    capability: memory
    capability_tools: [add, search]
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    tool_map:
      add: add_memory
      search: search_memory
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].capability == "memory"
    assert specs[0].capability_tools == ("add", "search")
    assert specs[0].tool_map == {"add": "add_memory", "search": "search_memory"}


def test_capability_requires_canonical_tool_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    capability: storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="'capability_tools' is required"):
        load_mcp_server_manifest(manifest)


def test_tool_map_rejects_reused_native_name(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-memory
    capability: memory
    capability_tools: [add, search]
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    tool_map:
      add: native_tool
      search: native_tool
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="multiple published tools"):
        load_mcp_server_manifest(manifest)


def test_optional_capability_tools_are_extensions_not_required_contract(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-memory
    capability: memory
    capability_tools: [add, search, list]
    capability_optional_tools: [get, retire]
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    tool_map:
      add: add_memory
      search: search_memory
      list: list_memories
      get: get_memory
      retire: retire_memory
""",
        encoding="utf-8",
    )
    specs = load_mcp_server_manifest(manifest)
    assert specs[0].capability_tools == ("add", "search", "list")
    assert specs[0].capability_optional_tools == ("get", "retire")


def test_tool_map_without_capability_raises(tmp_path: Path) -> None:
    """tool_map without a capability claim is rejected."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    tool_map:
      read_file: read_file
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="'tool_map' requires 'capability'"):
        load_mcp_server_manifest(manifest)


def test_tool_map_rejects_namespaced_key(tmp_path: Path) -> None:
    """tool_map keys are bare published names — prefixed ids are rejected."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: lisa-storage
    capability: storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    tool_map:
      storage.read_file: read_file
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="must be a bare published tool name"):
        load_mcp_server_manifest(manifest)


def test_tool_map_rejects_non_mapping(tmp_path: Path) -> None:
    """tool_map must be a mapping."""
    from kernel.mcp_launch.server_manifest import MCPServerManifestError

    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
servers:
  - server_id: s
    capability: storage
    runtime_kind: venv_bin
    venv_dir: "${FRAMEWORK_ROOT}"
    tool_map: read_file
""",
        encoding="utf-8",
    )
    with pytest.raises(MCPServerManifestError, match="'tool_map' must be a mapping"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_rejects_unadoptable_protocol_version(tmp_path: Path) -> None:
    """A pin the client cannot adopt is a config error, caught at parse time.

    Otherwise a typo (``2026-7-28``) or a handshake-era version parses cleanly
    and fails much later inside ``session.adopt()``, where the error names a
    mutually-unsupported set instead of the manifest line that is wrong.
    """
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
servers:
  - server_id: remote
    runtime_kind: http
    url: https://example.test/mcp
    protocol_version: "2026-7-28"
""",
        encoding="utf-8",
    )

    with pytest.raises(MCPServerManifestError, match=r"protocol_version.*not a version"):
        load_mcp_server_manifest(manifest)


def test_load_manifest_accepts_every_adoptable_protocol_version(tmp_path: Path) -> None:
    """The accepted set comes from the SDK, so it cannot drift from adopt()."""
    from kernel.mcp_launch.server_manifest import _adoptable_protocol_versions

    versions = _adoptable_protocol_versions()
    assert versions, "the installed SDK must expose its modern versions"

    for version in versions:
        manifest = tmp_path / f"mcp_servers_{version}.yaml"
        manifest.write_text(
            f"""
servers:
  - server_id: remote
    runtime_kind: http
    url: https://example.test/mcp
    protocol_version: "{version}"
""",
            encoding="utf-8",
        )

        specs = load_mcp_server_manifest(manifest)

        assert specs[0].protocol_version == version
