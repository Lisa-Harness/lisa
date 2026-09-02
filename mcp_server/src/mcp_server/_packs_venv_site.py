"""Activate the packs venv site-packages in MCP server child processes.

The host spawns ``lisa-content`` and ``lisa-catalog`` as child Python
processes. The host calls ``inject_packs_venv_site_packages()`` at CLI
startup to put the packs venv on ``sys.path``, but child processes are
fresh interpreters that only inherit env vars — ``sys.path`` modifications
do not cross process boundaries.

Editable installs (``uv pip install -e``) write ``__editable__.*.pth`` files
into the packs venv site-packages. These ``.pth`` files contain ``import``
statements that register meta-path finders mapping ``lisa_packs.<name>``
to the source directories. Python only executes ``import`` lines in
``.pth`` files when a directory is processed by ``site.addsitedir()`` —
``PYTHONPATH`` adds a directory to ``sys.path`` but does NOT process
``.pth`` files.

This module reads ``LISA_PACKS_VENV_SITE_PACKAGES`` (set by the host in
``build_content_mcp_config``) and calls ``site.addsitedir()`` so the
editable-install ``.pth`` files are activated. It is imported
side-effect-first at the top of ``mcp_server.catalog.server`` and
``mcp_server.content.server``.

When the env var is unset or the directory does not exist, this is a
no-op — bundled packs and non-editable installs don't need it.
"""

from __future__ import annotations

import importlib
import os
import site
import sys
from pathlib import Path

__all__ = ["activate_packs_venv_site"]

_PACKS_VENV_SITE_PACKAGES_ENV = "LISA_PACKS_VENV_SITE_PACKAGES"


def activate_packs_venv_site() -> Path | None:
    """Activate the packs venv site-packages via ``site.addsitedir``.

    Returns the activated path, or ``None`` when the env var is unset or
    the directory does not exist. Idempotent: if the directory is already
    on ``sys.path``, ``addsitedir`` is still safe to call (it re-processes
    ``.pth`` files but deduplicates ``sys.path`` entries).
    """
    raw = os.getenv(_PACKS_VENV_SITE_PACKAGES_ENV, "").strip()
    if not raw:
        return None
    site_packages = Path(raw)
    if not site_packages.is_dir():
        return None
    site_str = str(site_packages)
    # site.addsitedir processes .pth files (including the import lines
    # that register editable-install meta-path finders) and appends the
    # directory to sys.path. This is the mechanism that makes
    # `import lisa_packs.markdown.operations...` resolve to the editable
    # source directory in the child process.
    site.addsitedir(site_str)
    if site_str not in sys.path:
        sys.path.insert(0, site_str)
    # importlib.metadata caches the distribution list at first access;
    # after addsitedir exposes new dist-info dirs (editable installs in
    # the packs venv), invalidate the import cache so the next
    # entry_points() call re-scans and discovers the packs-venv
    # distributions. Without this, the catalog subprocess could miss the
    # packs-venv entry points (or pick them up inconsistently), which
    # contributes to Duplicate operation ID collisions when the same pack
    # is visible from more than one site-packages.
    importlib.invalidate_caches()
    return site_packages


# Activate on import so child servers get the packs venv as soon as they
# start, before any `from sdk.content import ...` triggers catalog loading.
activate_packs_venv_site()
