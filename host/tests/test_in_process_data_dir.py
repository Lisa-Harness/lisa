"""Regression tests for the 2026-09-04 in-process ``--data-dir`` wipe.

Incident: the CLI host runs first-party MCP servers in-process
(``in_process_mcp=True``). The rewrite in ``compile_mcp_configs`` replaced
``command``/``args`` with empty values, which silently dropped the compiled
``--data-dir`` flag. The pool's arg-reconstruction helpers then fed
``data_dir=""`` to ``lisa-storage._init`` → ``Path("")`` = the host process
cwd → the files sandbox resolved to ``<cwd>/files``. Research documents
appended via ``lisa-storage.append_file`` landed in the host's cwd (repo root
on 2026-09-02/03, ``$HOME`` on 2026-09-04) instead of ``<data_dir>/files``.
``lisa verify`` still passed because command-path spawns (``/models apply``
etc.) always received the correct ``--data-dir``.

Two layers are covered:

1. Host compile layer — the in-process rewrite must preserve ``args`` (and
   ``command``) so flag reconstruction still finds ``--data-dir``.
2. Kernel pool layer — an in-process config with no resolvable data dir must
   fail closed (``MCPConnectionError``), never silently connect with a
   cwd-relative data dir.
"""

from __future__ import annotations

import asyncio
import json as _json
import os as _os
from pathlib import Path

import pytest
from kernel.mcp_launch.compiler import compile_registry
from kernel.mcp_launch.policy import MCPServerRegistry
from kernel.mcp_launch.server_manifest import load_mcp_server_manifest
from host.testing import build_test_runtime
from kernel.tools.mcp_client import (
    MCPClientConfig,
    MCPConnectionError,
    MCPToolPool,
)

from host import mcp_pool
from host.runtime import HostRuntime, _make_in_process_resolver

_REPO_MANIFEST = (
    Path(__file__).resolve().parents[2] / "host" / "src" / "host" / "mcp_servers.yaml"
)


def _make_runtime(tmp_path: Path) -> HostRuntime:
    """Build a runtime wired for in-process MCP (the CLI host's mode)."""
    return build_test_runtime(tmp_path, models_yaml="", user_slug="testing")


def _compile_first_party(tmp_path: Path) -> dict[str, MCPClientConfig]:
    """Compile the repo's host manifest against a temp data dir.

    Returns configs keyed by server_id for the first-party servers. Uses the
    real compiler path (registry → compile_registry) so the assertion covers
    exactly what production compiles.
    """
    specs, _memory = load_mcp_server_manifest(_REPO_MANIFEST)
    registry = MCPServerRegistry()
    for spec in specs:
        registry.register(spec)
    compiled = compile_registry(
        registry,
        data_dir=str(tmp_path / "data"),
        workspace_root=str(tmp_path / "framework"),
    )
    return {cfg.server_id: cfg for cfg in compiled}


def test_in_process_rewrite_preserves_data_dir_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process rewrite must keep compiled args (root-cause regression).

    Reproduces the 2026-09-04 escape: wiping ``args`` dropped ``--data-dir``,
    so the storage server's files sandbox became cwd-relative. After the fix,
    an in-process lisa-storage config still carries the host data dir.
    """
    runtime = _make_runtime(tmp_path)
    runtime._in_process_mcp = True

    monkeypatch.setenv("FRAMEWORK_ROOT", str(tmp_path / "framework"))
    monkeypatch.setenv("LISA_TIMEZONE", "UTC")
    # lisa-storage is cold; keep the compile hermetic by stubbing pack roots.
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    configs = {cfg.server_id: cfg for cfg in [*warm, *cold]}

    cfg = configs["lisa-storage"]
    assert cfg.runtime_kind == "in_process"
    # The compiled args must survive the rewrite so the pool's
    # _in_process_data_dir can read the host data dir out of them.
    assert "--data-dir" in cfg.args
    assert cfg.args[cfg.args.index("--data-dir") + 1] == str(runtime.paths.data_dir)


def test_in_process_storage_connect_uses_host_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an in-process lisa-storage sandbox roots at the data dir.

    Compiles the real manifest, applies the in-process rewrite, connects the
    pool in-process, and writes a file through ``lisa-storage.write_file``.
    The file must land under ``<data_dir>/files`` — never a cwd-relative
    ``files/``.
    """
    runtime = _make_runtime(tmp_path)
    runtime._in_process_mcp = True

    monkeypatch.setenv("FRAMEWORK_ROOT", str(tmp_path / "framework"))
    monkeypatch.setenv("LISA_TIMEZONE", "UTC")
    monkeypatch.setattr(mcp_pool, "load_pack_server_exclusions", lambda _path: set())

    warm, cold = mcp_pool.compile_mcp_configs(runtime)
    configs = [cfg for cfg in [*warm, *cold] if cfg.server_id == "lisa-storage"]
    assert len(configs) == 1
    storage_cfg = configs[0]
    assert storage_cfg.runtime_kind == "in_process"

    cwd_before = _os.getcwd()

    pool = MCPToolPool([storage_cfg], in_process_resolver=_make_in_process_resolver())

    async def _drive() -> str:
        await pool.start()
        try:
            # Server-scoped dispatch lazily connects the cold in-process
            # storage server (mirrors how memory aliases and operations call
            # first-party servers in production). Scoped calls take the
            # BARE tool name — the scoped index is keyed by bare names.
            out = await pool.call_tool(
                "write_file",
                {"path": "regression/sandbox-root.md", "content": "# probe"},
                server_id="lisa-storage",
            )
            return out.output_text
        finally:
            await pool.aclose()

    output = asyncio.run(_drive())
    assert _json.loads(output).get("ok") is True
    landed = runtime.paths.data_dir / "files" / "regression" / "sandbox-root.md"
    assert landed.is_file(), f"write landed outside the sandbox: {output}"
    # And nothing escaped to the process cwd.
    assert not (Path(cwd_before) / "files" / "regression" / "sandbox-root.md").exists()


def test_in_process_data_dir_fail_closed() -> None:
    """An in-process config with no resolvable data dir must fail closed.

    Pre-fix behavior: ``_in_process_data_dir`` returned ``""`` →
    ``Path("")`` = cwd → cwd-relative files sandbox (the escape). Post-fix:
    ``MCPConnectionError`` with an actionable message.
    """
    config = MCPClientConfig(
        server_id="lisa-storage",
        command="",
        args=[],
        warm=False,
        required=False,
        runtime_kind="in_process",
        data_dir_path=None,
        ensure_data_dir=False,
    )
    with pytest.raises(MCPConnectionError, match="no --data-dir"):
        MCPToolPool._in_process_data_dir(config)


def test_in_process_data_dir_falls_back_to_data_dir_path() -> None:
    """The ``data_dir_path`` fallback still works when args are absent."""
    config = MCPClientConfig(
        server_id="lisa-storage",
        command="",
        args=[],
        warm=False,
        required=False,
        runtime_kind="in_process",
        data_dir_path="/tmp/lisa-fallback-data",
        ensure_data_dir=True,
    )
    assert MCPToolPool._in_process_data_dir(config) == "/tmp/lisa-fallback-data"


def test_in_process_data_dir_prefers_args_over_fallback() -> None:
    """An explicit ``--data-dir`` in args wins over ``data_dir_path``."""
    config = MCPClientConfig(
        server_id="lisa-storage",
        command="",
        args=["--data-dir", "/tmp/lisa-from-args"],
        warm=False,
        required=False,
        runtime_kind="in_process",
        data_dir_path="/tmp/lisa-fallback-data",
        ensure_data_dir=True,
    )
    assert MCPToolPool._in_process_data_dir(config) == "/tmp/lisa-from-args"