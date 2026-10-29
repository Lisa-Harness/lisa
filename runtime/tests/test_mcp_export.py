"""Tests for ``runtime.mcp_export`` — the runtime side of the client-config export.

``sdk.tests.test_mcp_host_export`` covers the pure projection. Here we cover
what only the runtime knows: resolving an in-process config's console script
and reading the compiled configs from ``RuntimeResources`` without starting
the pool.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.tools.mcp_client import MCPClientConfig
from runtime.mcp_export import export_runtime_mcp_servers, resolve_in_process_command


def _config(server_id: str, **overrides: Any) -> MCPClientConfig:
    values: dict[str, Any] = {
        "server_id": server_id,
        "command": f"/venv/bin/{server_id}",
        "args": ["--data-dir", "/home/u/.lisa"],
        "runtime_kind": "venv_bin",
    }
    values.update(overrides)
    return MCPClientConfig(**values)


def _runtime(warm: list[MCPClientConfig], cold: list[MCPClientConfig] | None = None):
    return SimpleNamespace(
        resources=SimpleNamespace(
            mcp_warm_configs=warm,
            mcp_cold_configs=cold or [],
            mcp_tool_pool=None,
        ),
        paths=SimpleNamespace(data_dir=Path("/home/u/.lisa")),
    )


# ── resolve_in_process_command ────────────────────────────────────────


def test_resolve_in_process_command_finds_installed_script(tmp_path: Path) -> None:
    script = tmp_path / "bin" / "lisa-memory"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")

    assert resolve_in_process_command("lisa-memory", venv_root=tmp_path) == str(script)


def test_resolve_in_process_command_returns_none_when_absent(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()

    assert resolve_in_process_command("lisa-memory", venv_root=tmp_path) is None


def test_resolve_in_process_command_returns_none_without_bin_dir(tmp_path: Path) -> None:
    assert resolve_in_process_command("lisa-memory", venv_root=tmp_path) is None


# ── export_runtime_mcp_servers ─────────────────────────────────────────


def test_export_reads_warm_and_cold_configs() -> None:
    runtime = _runtime(
        warm=[_config("lisa-catalog")],
        cold=[_config("lisa-memory"), _config("lisa-storage")],
    )

    export = export_runtime_mcp_servers(runtime)

    assert [server.server_id for server in export.servers] == [
        "lisa-catalog",
        "lisa-memory",
        "lisa-storage",
    ]
    assert export.data_dir == "/home/u/.lisa"


def test_export_empty_when_no_configs_compiled() -> None:
    """No compiled configs is "nothing configured", not "consult the live pool".

    Reading the live pool instead would report only *started* servers and
    silently omit warm-but-unconnected ones.
    """
    export = export_runtime_mcp_servers(_runtime([]))

    assert export.is_empty
    assert export.skipped == ()
    assert export.data_dir == "/home/u/.lisa"


def test_export_resolves_in_process_command_from_installed_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "bin" / "lisa-memory"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        "runtime.paths.venv_root",
        lambda: tmp_path,
    )

    export = export_runtime_mcp_servers(
        _runtime([_config("lisa-memory", command="", runtime_kind="in_process")])
    )

    assert export.servers[0].command == str(script)


def test_export_skips_in_process_when_script_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "bin").mkdir()
    monkeypatch.setattr("runtime.paths.venv_root", lambda: tmp_path)

    export = export_runtime_mcp_servers(
        _runtime([_config("lisa-memory", command="", runtime_kind="in_process")])
    )

    assert export.is_empty
    assert "command_resolver" in export.skipped[0][1]


def test_export_forwards_selection_flags() -> None:
    runtime = _runtime(
        [
            _config("lisa-memory"),
            _config("lisa-content", tool_policy=None),
            _config("remote", command="", runtime_kind="http", url="https://example.test/mcp"),
        ]
    )

    export = export_runtime_mcp_servers(runtime, server_ids=["remote"], include_http=True)

    assert [server.server_id for server in export.servers] == ["remote"]
    assert export.servers[0].transport == "http"


def test_export_does_not_require_a_started_pool() -> None:
    """The pool is ``None`` in this fixture; the export must still work."""
    runtime = _runtime([_config("lisa-storage")])

    assert runtime.resources.mcp_tool_pool is None
    assert export_runtime_mcp_servers(runtime).servers[0].server_id == "lisa-storage"
