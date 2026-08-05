"""Pure config builders for first-party MCP servers (lisa-content, lisa-catalog).

These functions turn content roots, disabled-pack sets, and provider
selections into :class:`~kernel.tools.mcp_client.MCPClientConfig` entries
without any I/O, runtime state, or host imports. They are the reusable
spawning-config surface that any host (CLI, web, external) can call to
wire the framework's content and catalog MCP servers.

Lifecycle (pool start / deferred-start / close) is NOT owned here — it is
host-owned because anyio cancel scopes bind to the host's event loop.

The builders accept the packs-venv site-packages path as an explicit
argument (``packs_venv_site_packages``) so this module has no dependency
on ``hosts.core.paths``. Hosts resolve that path via their own path
machinery (``hosts.core.paths.packs_venv_site_packages`` in this repo;
external hosts bring their own) and pass it in.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.session_plan import ProviderSelections
from kernel.tools.mcp_client import MCPClientConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────
# Shared with the child ``lisa-content`` / ``lisa-catalog`` MCP servers
# so they receive the same provider-preference and disabled-pack signals.

_LISA_SESSION_PROVIDER_SELECTIONS_ENV = "LISA_SESSION_PROVIDER_SELECTIONS"
_LISA_DISABLED_PACKS_ENV = "LISA_DISABLED_PACKS"
_LISA_PACKS_VENV_SITE_PACKAGES_ENV = "LISA_PACKS_VENV_SITE_PACKAGES"

# ── Pack-root helpers (pure) ────────────────────────────────────────


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    """De-duplicate a path iterable, preserving first-seen order."""
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


# ── Public config builders ──────────────────────────────────────────


def session_provider_selection_payload(
    selections: ProviderSelections,
) -> dict[str, dict[str, str]]:
    """Convert immutable plan ownership choices into child-process payload data."""
    return {
        "modes": dict(selections.modes),
        "operations": dict(selections.operations),
        "tools": dict(selections.tools),
    }


def build_content_mcp_config(
    *,
    data_dir: Path,
    content_roots: list[Path],
    disabled_packs: set[str],
    provider_selections: ProviderSelections,
    packs_venv_site_packages: Path | None = None,
    overlays_dir: Path | None = None,
) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-content for discovered pack roots.

    The kernel calls ``list_modes``, ``list_operations``, and
    ``read_authored_text`` on this server to discover and load all content
    packs. The server does all filesystem access; the kernel never touches
    the filesystem directly.

    ``SDK_CONTENT_ROOTS`` includes both the pack root (for modes/operations
    discovery) and the ``content/`` subdirectory (so ``read_authored_text``
    can resolve authored text assets like ``prompts/system-prompt.md``
    without a ``content/`` prefix). When ``overlays_dir`` is set and exists,
    it is prepended so overlay content takes precedence over pack content.

    Args:
        data_dir: Host data directory (passed to the child as ``--data-dir``).
        content_roots: Active SDK content roots (pack roots + configured roots).
        disabled_packs: Normalized pack names to exclude in the child process.
        provider_selections: Explicit SessionPlan ownership decisions.
        packs_venv_site_packages: Optional path to the packs venv's
            site-packages directory. When provided and existing, it is
            exported via ``LISA_PACKS_VENV_SITE_PACKAGES`` so the child's
            ``mcp_server._packs_venv_site`` module can call
            ``site.addsitedir()`` on it, activating the editable-install
            ``__editable__.*.pth`` files that register meta-path finders
            for ``lisa_packs.<name>`` → source directory mappings. The host
            process calls ``inject_packs_venv_site_packages()`` at startup
            to put the packs venv on ``sys.path``, but child processes are
            fresh interpreters that only inherit env vars — without this,
            operation modules like
            ``lisa_packs.markdown.operations.forge.forge_document`` raise
            ``ModuleNotFoundError: No module named 'lisa_packs'`` in the child.
        overlays_dir: Optional user-authored overlays directory. When set
            and the directory exists, it is prepended to
            ``SDK_CONTENT_ROOTS`` and ``LISA_OVERLAY_ROOT`` is set so the
            catalog server can identify overlay entries for precedence logic.
    """
    roots_for_child = content_roots
    subdir_roots = [root / "content" for root in roots_for_child if (root / "content").is_dir()]
    configured_roots = _unique_paths([*roots_for_child, *subdir_roots])
    env: dict[str, str] = {
        "SDK_CONTENT_ROOTS": os.pathsep.join(str(root) for root in configured_roots)
    }

    # Prepend the overlays directory so overlay content takes precedence.
    # The catalog server uses LISA_OVERLAY_ROOT to identify which entries
    # are overlay-authored and should auto-win in duplicate resolution.
    if overlays_dir is not None and overlays_dir.is_dir():
        env["SDK_CONTENT_ROOTS"] = str(overlays_dir) + os.pathsep + env["SDK_CONTENT_ROOTS"]
        env["LISA_OVERLAY_ROOT"] = str(overlays_dir)
    elif overlays_dir is not None:
        logger.warning(
            "Overlays directory '%s' does not exist — overlays disabled.",
            overlays_dir,
        )

    # Pass disabled pack names to the lisa-content child process so that
    # entry-point discovery in the child also excludes disabled packs.
    if disabled_packs:
        env[_LISA_DISABLED_PACKS_ENV] = os.pathsep.join(sorted(disabled_packs))

    selection_payload = session_provider_selection_payload(provider_selections)
    if any(selection_payload.values()):
        env[_LISA_SESSION_PROVIDER_SELECTIONS_ENV] = json.dumps(
            selection_payload,
            ensure_ascii=True,
            sort_keys=True,
        )

    # Inject the packs venv site-packages into the child's PYTHONPATH so
    # editable-installed packs (lisa_packs.<name>) are importable in the
    # lisa-content / lisa-catalog subprocesses. The host calls
    # inject_packs_venv_site_packages() at CLI startup, but the child
    # processes are fresh Python interpreters that don't inherit sys.path
    # modifications — only env vars.
    if packs_venv_site_packages is not None and packs_venv_site_packages.is_dir():
        env[_LISA_PACKS_VENV_SITE_PACKAGES_ENV] = str(packs_venv_site_packages)

    # Route this server's logs to <data_dir>/lisa-content.jsonl. The
    # content/catalog servers are built directly here rather than via
    # kernel.mcp_launch.compiler (which sets this for spec-driven servers),
    # so without this the child's configure_logging falls back to
    # ``<cwd>/lisa-content.jsonl`` — i.e. the project root — because these
    # configs leave cwd=None and the child inherits the host's cwd.
    env.setdefault("LISA_MCP_LOG_PATH", str(Path(data_dir) / "lisa-content.jsonl"))
    return MCPClientConfig(
        server_id="lisa-content",
        command=sys.executable,
        args=[
            "-m",
            "mcp_server.content.server",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        runtime_kind="python",
    )


def build_catalog_mcp_config(
    *,
    content_config: MCPClientConfig,
    data_dir: Path,
) -> MCPClientConfig:
    """Build an MCPClientConfig that spawns lisa-catalog for catalog tools.

    The kernel calls ``list_modes``, ``list_operations``, and ``list_tools``
    on this server to discover and load all content-pack catalogs. The
    server does all filesystem access; the kernel never touches the
    filesystem directly.

    Shares the same env/roots setup as the content server — the only
    difference is the module (``mcp_server.catalog.server`` vs
    ``mcp_server.content.server``).
    """
    # Reuse the content server's env (which already sets SDK_CONTENT_ROOTS,
    # LISA_DISABLED_PACKS, the packs-venv PYTHONPATH, and LISA_MCP_LOG_PATH),
    # but override the log path so this server writes to its own file. Like
    # build_content_mcp_config, this is built directly rather than via
    # kernel.mcp_launch.compiler, so without this the child's
    # configure_logging falls back to ``<cwd>/lisa-catalog.jsonl`` (project
    # root) since cwd is None and inherited from the host.
    env = dict(content_config.env) if content_config.env else {}
    env["LISA_MCP_LOG_PATH"] = str(Path(data_dir) / "lisa-catalog.jsonl")
    return MCPClientConfig(
        server_id="lisa-catalog",
        command=content_config.command,
        args=[
            "-m",
            "mcp_server.catalog.server",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        runtime_kind="python",
    )


__all__ = [
    "build_catalog_mcp_config",
    "build_content_mcp_config",
    "session_provider_selection_payload",
]
