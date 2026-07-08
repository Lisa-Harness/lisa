"""Tests for the mcp_servers.yaml manifest loader and the MemoryStoreRouter.

Covers:
- load_mcp_server_manifest: missing file returns empty.
- load_mcp_server_manifest: parses servers + memory backend config.
- ${VAR} env expansion at load time.
- disabled entries are skipped.
- data_dir_arg null vs custom flag vs default.
- MemoryStoreRouter delegates to default; backend() returns named.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kernel.mcp_launch.policy import RuntimeKind
from kernel.mcp_launch.server_manifest import (
    MemoryBackendConfig,
    load_mcp_server_manifest,
)
from kernel.memory_protocol import MemoryStore
from kernel.memory_router import MemoryStoreRouter


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
  - server_id: mem0-stdio
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
    assert specs[0].server_id == "mem0-stdio"
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


# ── MemoryStoreRouter ──────────────────────────────────────────────────────


def test_router_delegates_to_default() -> None:
    default = AsyncMock(spec=MemoryStore)
    default.record_memory = AsyncMock(return_value="record")
    default.recall = AsyncMock(return_value="recalled")
    default.list_memories = AsyncMock(return_value=[])

    router = MemoryStoreRouter(default=default)  # type: ignore[arg-type]
    assert router.default is default
    assert router.backend("__default__") is default
    assert router.registered_backends() == ["__default__"]


def test_router_backend_named_access() -> None:
    default = AsyncMock(spec=MemoryStore)
    mem0 = AsyncMock(spec=MemoryStore)
    router = MemoryStoreRouter(default=default, backends={"mem0": mem0})  # type: ignore[arg-type]
    assert router.backend("mem0") is mem0
    assert router.backend("unknown") is None
    assert "mem0" in router.registered_backends()


def test_memory_store_protocol_is_runtime_checkable() -> None:
    """MCPMemoryClient satisfies the MemoryStore protocol by duck typing."""
    from kernel.mcp_client import MCPMemoryClient

    client = MCPMemoryClient("python", ["-m", "x"])
    assert isinstance(client, MemoryStore)