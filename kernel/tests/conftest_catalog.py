"""Shared test fixture for MCP-mediated catalog loading.

Spawns the ``lisa-catalog`` MCP server with the ``sdk/packs/core`` content root
and registers it with the kernel so ``get_workflow_catalog()`` and
``get_operation_catalog()`` work in tests.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from kernel.ipsar_fsm import (
    get_workflow_catalog,
    register_content_mcp,
    register_operation_callbacks,
)
from kernel.operations.registry import (
    execute_operation,
    get_operation_catalog,
    register_operation_mcp,
)
from kernel.tools.mcp_client import MCPClientConfig

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()
_LEARNING_ROOT: Path = (
    Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "memory"
).resolve()
# search, markdown, and letta migrated to ../lisa-packs/<name>
# (external submodules; note the hyphenated repo name — the inner Python
# package dir is still ``lisa_packs``). Only include a content root when the
# submodule is checked out alongside the framework; tests that need a
# migrated mode skip when absent.
_PACKS_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / ".." / "lisa-packs").resolve()


def _installed_pack_names() -> set[str]:
    """Return the set of pack names already installed via entry points.

    A pack installed into the venv (e.g. via ``uv sync --extra ext``)
    registers a ``lisa.extensions.content_roots`` entry point that
    resolves to ``site-packages/lisa_packs/<name>``. When such an install
    is present, the source-tree root under ``_PACKS_ROOT`` is redundant
    and must NOT be added as an extra content root — adding both makes
    the catalog scan the same ``operations/`` twice and raises
    "Duplicate operation ID" errors.
    """
    try:
        from kernel.content_roots import discover_installed_content_roots
    except Exception:  # pylint: disable=broad-except
        return set()
    names: set[str] = set()
    for root in discover_installed_content_roots():
        # The installed root path ends in lisa_packs/<name>.
        if root.parent.name == "lisa_packs":
            names.add(root.name)
    return names


def _external_pack_roots() -> list[Path]:
    """Return the content roots of migrated packs that are checked out.

    Skips a pack's source-tree root when that pack is already installed
    (e.g. via ``uv sync --extra ext``), because the installed root under
    the venv's ``site-packages/lisa_packs/<name>`` already covers it.
    Adding both produces a duplicate ``operations/`` scan → "Duplicate
    operation ID" catalog errors at fixture setup.
    """
    installed = _installed_pack_names()
    roots: list[Path] = []
    for name in ("search", "markdown", "letta"):
        if name in installed:
            continue
        root = (_PACKS_ROOT / name / "lisa_packs" / name).resolve()
        # Content without importable operation code loads modes whose ops are
        # dropped, shadowing the synthetic test modes in conftest.py.
        if root.exists() and _pack_code_importable(name):
            roots.append(root)
    return roots


def _pack_code_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(f"lisa_packs.{name}") is not None
    except ModuleNotFoundError:
        return False


def _make_catalog_mcp_config(*, log_path: Path) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog with the core pack.

    ``log_path`` routes the child's JSONL log out of the repo root (the
    ``configure_logging`` fallback writes ``<cwd>/lisa-catalog.jsonl`` when
    ``LISA_MCP_LOG_PATH`` is unset, and these configs leave cwd=None so the
    subprocess inherits the pytest cwd). Mirrors the env injection that
    ``host.mcp_pool.build_catalog_mcp_config`` does in production.
    """
    content_roots = [_CORE_ROOT, _LEARNING_ROOT, *_external_pack_roots()]
    return MCPClientConfig(
        server_id="lisa-catalog-test",
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.catalog.server",
            "--data-dir",
            str(_CORE_ROOT),
        ],
        env={
            "SDK_CONTENT_ROOTS": os.pathsep.join(str(r) for r in content_roots),
            "LISA_SESSION_PROVIDER_SELECTIONS": (
                '{"modes":{"default":"lisa-pack-core","read":"lisa-pack-core"},'
                '"tools":{"web_search":"lisa-pack-brave"}}'
            ),
            "LISA_MCP_LOG_PATH": str(log_path),
        },
    )


@pytest.fixture(scope="module", autouse=True)
def _mcp_catalog_setup(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Wire the catalog MCP and operation callbacks so the kernel can load catalogs.

    Order matters: operation MCP first, then operation callbacks, then content MCP.
    ``validate_workflow_catalog`` calls ``_validate_mode_operation_bindings`` which
    needs the operation callbacks registered before it can validate mode bindings.

    The catalog is eagerly loaded here during module setup (sync context) so async
    tests don't need to call ``asyncio.run()`` from inside a running event loop.
    """
    log_dir = tmp_path_factory.mktemp("lisa-catalog")
    config = _make_catalog_mcp_config(log_path=log_dir / "lisa-catalog.jsonl")
    register_operation_mcp(config)
    register_operation_callbacks(get_operation_catalog, execute_operation)
    register_content_mcp(config)
    # Eagerly load the workflow catalog now (sync context, before any async test runs)
    get_workflow_catalog()
