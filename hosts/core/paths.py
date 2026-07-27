"""Central path resolution for Lisa framework hosts.

Resolves the framework root, venv root, and packs venv in a way that works
both in a repo checkout (dev) and a ``uv tool install`` environment (no
repo checkout present).

Resolution modes
----------------
- **Repo mode**: the module file lives inside a git checkout. The framework
  root is ``Path(__file__).resolve().parents[2]`` (the repo root), and the
  venv root is ``<repo>/.venv``.
- **Tool mode**: the module file lives inside an isolated ``uv tool`` venv
  (no ``.git`` dir and no workspace ``pyproject.toml`` above it). The
  framework root is ``sys.prefix`` (the tool venv root), and the venv root
  is also ``sys.prefix`` — the ``Scripts/`` (Windows) or ``bin/`` (POSIX)
  directory that holds the ``lisa-*`` console scripts.

The packs venv (``~/.lisa/packs-venv``) is a separate, dedicated venv where
runtime ``/pack install`` commands land new packs. Its site-packages
directory is injected into ``sys.path`` at CLI startup so that
``importlib.metadata.entry_points`` discovers packs installed there.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────

LISA_PACKS_VENV_ENV = "LISA_PACKS_VENV"

# ── Defaults ─────────────────────────────────────────────────────────

_PACKS_VENV_DIR_NAME = "packs-venv"


def _repo_root_from_module() -> Path:
    """Return the directory three parents up from this module (repo root)."""
    return Path(__file__).resolve().parents[2]


def _is_repo_checkout(path: Path) -> bool:
    """Return True if ``path`` looks like a Lisa framework repo checkout.

    A checkout is identified by the presence of a ``.git`` directory or a
    ``pyproject.toml`` with a ``[tool.uv.workspace]`` section at the top
    level. In a ``uv tool`` install, neither exists.
    """
    if (path / ".git").exists():
        return True
    pyproject = path / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib

            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            return isinstance(payload.get("tool", {}).get("uv", {}).get("workspace"), dict)
        except Exception:  # pylint: disable=broad-except
            return False
    return False


def is_tool_install() -> bool:
    """Return True when running inside a ``uv tool install`` environment.

    This is the case when the framework root does not look like a repo
    checkout (no ``.git``, no workspace ``pyproject.toml``). The venv root
    is then ``sys.prefix``.
    """
    return not _is_repo_checkout(_repo_root_from_module())


def framework_root() -> Path:
    """Return the framework root directory.

    In repo mode this is the repository root. In tool mode this is
    ``sys.prefix`` (the tool venv root). Used for locating non-venv
    assets that ship with the repo (e.g. ``hosts/little_lisa/.env`` for
    dev back-compat).
    """
    repo = _repo_root_from_module()
    if _is_repo_checkout(repo):
        return repo
    return Path(sys.prefix)


def venv_root() -> Path:
    """Return the venv root directory containing console-script entry points.

    In repo mode this is ``<repo>/.venv``. In tool mode this is
    ``sys.prefix``. The MCP server manifest's ``${FRAMEWORK_ROOT}`` env var
    is set to this value so ``venv_dir: "${FRAMEWORK_ROOT}"`` resolves to
    the directory whose ``Scripts/`` (Windows) or ``bin/`` (POSIX)
    subdirectory holds the ``lisa-context``, ``lisa-catalog``, etc.
    executables.
    """
    repo = _repo_root_from_module()
    if _is_repo_checkout(repo):
        return repo / ".venv"
    return Path(sys.prefix)


def packs_venv_for(data_dir: Path) -> Path:
    """Return the packs venv path for a resolved host ``data_dir``.

    Resolution order:
      1. ``LISA_PACKS_VENV`` — explicit override (absolute venv path).
         Takes precedence over everything, including ``data_dir``.
      2. ``<data_dir>/packs-venv`` — the packs venv always lives inside
         the host's resolved data directory, so a ``--data-dir`` override
         is honored consistently across pack install/uninstall and the
         rest of the host.

    Use :func:`default_packs_venv` for the startup-time resolution that
    also consults ``LISA_DATA_DIR`` and the platform default data dir.
    """
    override = os.getenv(LISA_PACKS_VENV_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return data_dir / _PACKS_VENV_DIR_NAME


def default_packs_venv() -> Path:
    """Return the default packs venv path (startup-time resolution).

    Resolution order (matches :func:`HostConfig.from_env`):
      1. ``LISA_PACKS_VENV`` — explicit override (absolute venv path).
      2. ``LISA_DATA_DIR`` — the packs venv is ``<data_dir>/packs-venv``.
         A relative value is resolved against the current working
         directory, matching ``HostConfig.from_env``'s ``.resolve()``.
      3. ``~/.lisa/packs-venv`` — the default data dir.

    A trailing path separator on ``LISA_DATA_DIR`` (e.g. ``artifacts/``) is
    handled correctly by :class:`pathlib.Path`.

    Prefer :func:`packs_venv_for` when a resolved ``data_dir`` is already
    available (e.g. from a ``HostRuntime``) so that a ``--data-dir`` CLI
    override is honored consistently instead of re-reading the env var.
    """
    override = os.getenv(LISA_PACKS_VENV_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    data_dir_env = os.getenv("LISA_DATA_DIR", "").strip()
    if data_dir_env:
        return Path(data_dir_env).expanduser().resolve() / _PACKS_VENV_DIR_NAME
    from hosts.core.config import default_data_dir  # noqa: PLC0415

    return default_data_dir() / _PACKS_VENV_DIR_NAME


def packs_venv_site_packages(venv_path: Path) -> Path:
    """Return the site-packages directory inside a venv.

    On Windows this is ``<venv>/Lib/site-packages``; on POSIX it is
    ``<venv>/lib/python<major>.<minor>/site-packages``.
    """
    if os.name == "nt":
        return venv_path / "Lib" / "site-packages"
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return venv_path / "lib" / py_dir / "site-packages"


def inject_packs_venv_site_packages() -> Path | None:
    """Insert the packs venv site-packages into ``sys.path`` if it exists.

    Called at CLI startup **before** any ``importlib.metadata.entry_points``
    call so that packs installed into the packs venv are discoverable via
    entry points. Returns the injected path, or ``None`` if the packs venv
    does not exist yet (first launch — no packs installed).

    Uses ``site.addsitedir()`` (not just ``sys.path.insert``) so that
    ``.pth`` files in the packs venv site-packages are processed. This is
    critical for editable installs (``uv pip install -e``): they write
    ``__editable__.*.pth`` files whose ``import`` lines register meta-path
    finders mapping ``lisa_packs.<name>`` to source directories. Without
    ``addsitedir``, those ``import`` lines never execute and
    ``import lisa_packs.<name>`` fails with ``ModuleNotFoundError``.
    """
    import site as _site

    venv = default_packs_venv()
    site_packages = packs_venv_site_packages(venv)
    if site_packages.is_dir():
        site_str = str(site_packages)
        # site.addsitedir processes .pth files (including the import lines
        # that register editable-install meta-path finders) and appends to
        # sys.path. Safe to call even if already on sys.path.
        _site.addsitedir(site_str)
        if site_str not in sys.path:
            sys.path.insert(0, site_str)
            logger.debug("Injected packs venv site-packages: %s", site_str)
        return site_packages
    return None


__all__ = [
    "LISA_PACKS_VENV_ENV",
    "default_packs_venv",
    "framework_root",
    "inject_packs_venv_site_packages",
    "is_tool_install",
    "packs_venv_for",
    "packs_venv_site_packages",
    "venv_root",
]
