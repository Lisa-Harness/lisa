"""Tests for the mcp_servers.yaml manifest loader.

Covers:
- load_mcp_server_manifest: missing file returns empty.
- load_mcp_server_manifest: parses servers + memory backend config.
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
    load_mcp_server_manifest,
)

# ── Manifest loader ─────────────────────────────────────────────────────────


def test_load_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    specs, mem = load_mcp_server_manifest(tmp_path / "nope.yaml")
    assert specs == []
    assert mem.default_backend is None


def test_load_manifest_parses_servers_and_memory(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: mem0
  backends:
    - id: mem0
      kind: http
      url: https://mcp.mem0.ai/mcp
      headers:
        Authorization: "Bearer ${MEM0_API_KEY_TEST}"
      tool_names:
        add: add_memory
        search: search_memories
    - id: lisa-context
      kind: stdio
      command: "${LISA_MCP_COMMAND}"
      args: ["--data-dir", "/data"]
      tool_names:
        add: add_memory
        search: search_memory
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
    specs, mem = load_mcp_server_manifest(manifest)
    assert len(specs) == 1  # letta is disabled
    assert specs[0].server_id == "mem0"
    assert specs[0].runtime_kind is RuntimeKind.NPX
    assert specs[0].npx_package == "@mem0/mcp"
    # data_dir_arg null → no injection
    assert specs[0].data_dir_arg is None
    # tool policy carried through
    assert specs[0].tool_policy.denied_tools == ["delete_memory"]
    # memory backend config — peer list, two backends
    assert mem.default_backend == "mem0"
    backend_ids = [b.id for b in mem.backends]
    assert backend_ids == ["mem0", "lisa-context"]
    mem0_backend = next(b for b in mem.backends if b.id == "mem0")
    assert mem0_backend.kind == "http"
    assert mem0_backend.url == "https://mcp.mem0.ai/mcp"
    assert mem0_backend.tool_names["search"] == "search_memories"
    lisa_backend = next(b for b in mem.backends if b.id == "lisa-context")
    assert lisa_backend.kind == "stdio"
    assert lisa_backend.command == "${LISA_MCP_COMMAND}"  # expansion is at host time
    assert lisa_backend.tool_names["add"] == "add_memory"


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
    specs, _ = load_mcp_server_manifest(manifest)
    assert specs[0].env_policy.forced_overrides["MEM0_API_KEY"] == "secret123"


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
    specs, _ = load_mcp_server_manifest(manifest)
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
    specs, _ = load_mcp_server_manifest(manifest)
    assert specs[0].data_dir_arg == "--data-dir"


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
    specs, _ = load_mcp_server_manifest(manifest)
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
    specs, _ = load_mcp_server_manifest(manifest)
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
    specs, _ = load_mcp_server_manifest(manifest)
    auth = specs[0].auth
    assert auth.kind is AuthKind.OAUTH2_PKCE
    assert auth.client_id == "my-app"
    assert auth.scopes == ("repo", "read:org")
    assert auth.auth_server == "https://github.com/login/oauth/authorize"
    assert auth.is_authenticated() is True


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


# ── Default-backend / server-id consistency ─────────────────────────────────


def test_default_backend_not_in_backends_raises(tmp_path: Path) -> None:
    """default_backend must reference a declared memory.backends id."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: mem0
  backends:
    - id: lisa_memory
      kind: stdio
      command: lisa-memory
      args: []
      tool_names: {add: add_memory}
servers: []
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="not in memory.backends"):
        load_mcp_server_manifest(manifest)


def test_default_backend_no_matching_server_id_raises(tmp_path: Path) -> None:
    """default_backend must match a servers[] server_id (so scoped dispatch
    can route). A backend id with no matching server is a manifest error."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: mem0
  backends:
    - id: mem0
      kind: http
      url: https://mcp.mem0.ai/mcp
      headers: {Authorization: "Bearer x"}
      tool_names: {add: add_memory}
servers:
  - server_id: letta
    runtime_kind: http
    url: https://letta.example/mcp
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="no matching server_id"):
        load_mcp_server_manifest(manifest)


def test_default_backend_lisa_memory_allowed_without_server_entry(tmp_path: Path) -> None:
    """The first-party lisa_memory backend is registered in code, not in the
    manifest's servers list, so it's allowed as default without a servers entry."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: lisa_memory
  backends:
    - id: lisa_memory
      kind: stdio
      command: lisa-memory
      args: []
      tool_names: {add: add_memory}
servers: []
""",
        encoding="utf-8",
    )
    specs, mem = load_mcp_server_manifest(manifest)
    assert mem.default_backend == "lisa_memory"
    assert specs == []


def test_default_backend_matching_server_id_loads(tmp_path: Path) -> None:
    """default_backend whose id matches a servers[] server_id loads cleanly."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: mem0
  backends:
    - id: mem0
      kind: http
      url: https://mcp.mem0.ai/mcp
      headers: {Authorization: "Bearer x"}
      tool_names: {add: add_memory}
servers:
  - server_id: mem0
    runtime_kind: http
    url: https://mcp.mem0.ai/mcp
""",
        encoding="utf-8",
    )
    specs, mem = load_mcp_server_manifest(manifest)
    assert mem.default_backend == "mem0"
    assert specs[0].server_id == "mem0"


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
    specs, _ = load_mcp_server_manifest(manifest)
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
    specs, _ = load_mcp_server_manifest(manifest)
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


def test_load_manifest_parses_result_maps(tmp_path: Path) -> None:
    """MemoryBackendSpec parses result_maps from YAML (http backend)."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: mem0
  backends:
    - id: mem0
      kind: http
      url: https://mcp.mem0.ai/mcp
      headers:
        Authorization: "Bearer test-key"
      tool_names:
        add: add_memory
        search: search_memories
        list: get_memories
      result_maps:
        add: lisa_packs.mem0.result_maps.mem0_add_result_map
        search: lisa_packs.mem0.result_maps.mem0_search_result_map
        list: lisa_packs.mem0.result_maps.mem0_list_result_map
servers:
  - server_id: mem0
    runtime_kind: http
    url: https://mcp.mem0.ai/mcp
    headers:
      Authorization: "Bearer test-key"
""",
        encoding="utf-8",
    )
    specs, mem = load_mcp_server_manifest(manifest)
    assert mem.default_backend == "mem0"
    assert len(mem.backends) == 1
    backend = mem.backends[0]
    assert backend.result_maps == {
        "add": "lisa_packs.mem0.result_maps.mem0_add_result_map",
        "search": "lisa_packs.mem0.result_maps.mem0_search_result_map",
        "list": "lisa_packs.mem0.result_maps.mem0_list_result_map",
    }


def test_load_manifest_result_maps_default_empty(tmp_path: Path) -> None:
    """Backends without result_maps get an empty dict (pass-through)."""
    manifest = tmp_path / "mcp_servers.yaml"
    manifest.write_text(
        """
version: 1
memory:
  default_backend: lisa_memory
  backends:
    - id: lisa_memory
      kind: stdio
      command: lisa-memory
      tool_names:
        add: add_memory
        search: search_memory
        list: list_memories
servers:
  - server_id: lisa-memory
    runtime_kind: venv_bin
    entry_point: lisa-memory
""",
        encoding="utf-8",
    )
    specs, mem = load_mcp_server_manifest(manifest)
    assert mem.default_backend == "lisa_memory"
    backend = mem.backends[0]
    assert backend.result_maps == {}


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
    specs, _ = load_mcp_server_manifest(manifest)
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
    specs, _ = load_mcp_server_manifest(manifest)
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
