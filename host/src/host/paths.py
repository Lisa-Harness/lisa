"""Central path resolution for Lisa framework hosts.

Resolves the framework root, venv root, and packs venv in a way that works
both in a repo checkout (dev) and a ``uv tool install`` environment (no
repo checkout present).

Resolution modes
----------------
- **Repo mode**: the module file lives inside a git checkout. The framework
  root is found by walking up from ``Path(__file__)`` until a repo marker
  (``.git`` or a ``[tool.uv.workspace]`` ``pyproject.toml``) is found, and
  the venv root is ``<repo>/.venv``.
- **Tool mode**: the module file lives inside an isolated ``uv tool`` venv
  (no ``.git`` dir and no workspace ``pyproject.toml`` above it). The
  framework root is ``sys.prefix`` (the tool venv root), and the venv root
  is also ``sys.prefix`` — the ``Scripts/`` (Windows) or ``bin/`` (POSIX)
  directory that holds the ``lisa-*`` console scripts.

The packs venv (``~/.lisa/venv``) is a separate, dedicated venv where
runtime ``/pack install`` commands land new packs. Its site-packages
directory is injected into ``sys.path`` at CLI startup so that
``importlib.metadata.entry_points`` discovers packs installed there.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tomllib
from importlib import metadata as importlib_metadata
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────

LISA_PACKS_VENV_ENV = "LISA_PACKS_VENV"

# ── Defaults ─────────────────────────────────────────────────────────

_PACKS_VENV_DIR_NAME = "venv"
_PROFILES_SUBDIR_NAME = "profiles"


def _repo_root_from_module() -> Path:
    """Return the repository root by walking up from this module's location.

    The module lives at ``<repo>/host/src/host/paths.py`` (src layout), so a
    fixed parent count is fragile: the pre-src-layout version anchored on
    ``parents[1]``, which silently broke when the tree was restructured. Walk
    up instead, stopping at the filesystem root. Callers combine this with
    :func:`_is_repo_checkout` to distinguish repo mode from tool mode.
    """
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if _is_repo_checkout(candidate):
            return candidate
    return current


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
      2. ``<data_dir>/venv`` — the packs venv always lives inside
         the host's resolved data directory, so a ``--data-dir`` override
         is honored consistently across pack install/uninstall and the
         rest of the host.

    When ``data_dir`` is a **profile** (``<base>/profiles/<name>`` — see
    :func:`host.config.profile_data_dir`), the venv stays shared with the
    profile's base data dir (``<base>/venv``) instead of a per-profile
    duplicate: packs are installed once and discovered from every
    profile. A profile that wants its own venv sets ``LISA_PACKS_VENV``.

    Use :func:`default_packs_venv` for the startup-time resolution that
    also consults ``LISA_DATA_DIR``, ``LISA_PROFILE``, and the platform
    default data dir.
    """
    override = os.getenv(LISA_PACKS_VENV_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    base = profile_base_data_dir(data_dir)
    if base is not None:
        return base / _PACKS_VENV_DIR_NAME
    return data_dir / _PACKS_VENV_DIR_NAME


def profile_base_data_dir(data_dir: Path) -> Path | None:
    """Return the profile's base data dir when ``data_dir`` is a profile.

    A profile data dir has the shape ``<base>/profiles/<name>``. When
    ``data_dir`` matches that shape, return ``<base>``; otherwise return
    ``None``. The detection is structural (parent dir is named
    ``profiles/``), not identity-based, so it works for any base —
    ``~/.lisa/profiles/taxes`` → ``~/.lisa``, a custom
    ``LISA_DATA_DIR=/srv/lisa`` → ``/srv/lisa/profiles/x`` → ``/srv/lisa``.
    ``..`` and other escapes are impossible because
    :func:`host.config.profile_data_dir` validates names to a single
    path segment.
    """
    data_dir = Path(data_dir).resolve()
    if data_dir.parent.name != _PROFILES_SUBDIR_NAME:
        return None
    return data_dir.parent.parent


def default_packs_venv() -> Path:
    """Return the default packs venv path (startup-time resolution).

    Resolution order (matches :func:`HostConfig.from_env`):
      1. ``LISA_PACKS_VENV`` — explicit override (absolute venv path).
      2. ``LISA_DATA_DIR`` — the packs venv is ``<data_dir>/venv``.
         A relative value is resolved against the current working
         directory, matching ``HostConfig.from_env``'s ``.resolve()``.
      3. ``LISA_PROFILE`` — the packs venv is the BASE data dir's venv
         (``<base>/venv``), shared with the profile. See
         :func:`packs_venv_for` for the profile/base rules.
      4. ``~/.lisa/venv`` — the default data dir.

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
    profile_env = os.getenv("LISA_PROFILE", "").strip()
    if profile_env:
        # Invalid names fall through to the platform default: the CLI
        # surfaces the validation error from HostConfig.from_env; this
        # startup-time resolver must not crash before the error panel.
        try:
            from host.config import profile_data_dir  # noqa: PLC0415

            profile = profile_data_dir(profile_env)
        except ValueError:
            pass
        else:
            base = profile_base_data_dir(profile)
            if base is not None:
                return base / _PACKS_VENV_DIR_NAME
            return profile / _PACKS_VENV_DIR_NAME
    from host.config import default_data_dir  # noqa: PLC0415

    return default_data_dir() / _PACKS_VENV_DIR_NAME


def _venv_python_dir(venv_path: Path) -> str:
    """Return the ``python<major>.<minor>`` dir name for a venv.

    The packs venv is created by ``uv venv`` and may target a different
    Python than the host interpreter (e.g. host runs 3.13 but ``uv`` resolved
    3.11 from ``requires-python``). Using ``sys.version_info`` here would
    compute ``python3.13/...`` while the actual layout is
    ``python3.11/...`` and produce ``FileNotFoundError`` on first launch.

    Resolution order:
      1. ``pyvenv.cfg`` ``version_info`` (Python 3.13+) or ``version``
         field — the authoritative venv Python version.
      2. Scan ``<venv>/lib/`` for a ``python<major>.<minor>`` directory.
      3. Fall back to the running interpreter's version.
    """
    cfg = venv_path / "pyvenv.cfg"
    if cfg.is_file():
        try:
            for line in cfg.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key in ("version_info", "version") and value:
                    parts = value.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return f"python{parts[0]}.{parts[1]}"
        except OSError:
            pass
    lib_dir = venv_path / "lib"
    if lib_dir.is_dir():
        for child in sorted(lib_dir.iterdir()):
            if child.is_dir() and child.name.startswith("python"):
                # python3.11, python3.13, etc. — first match wins.
                if child.name[6:].replace(".", "").isdigit():
                    return child.name
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def packs_venv_site_packages(venv_path: Path) -> Path:
    """Return the site-packages directory inside a venv.

    On Windows this is ``<venv>/Lib/site-packages``; on POSIX it is
    ``<venv>/lib/python<major>.<minor>/site-packages``. The ``pythonX.Y``
    segment is read from the venv's own ``pyvenv.cfg`` (or scanned from
    ``lib/``) rather than ``sys.version_info``, because the packs venv
    may target a different Python than the host interpreter.
    """
    if os.name == "nt":
        return venv_path / "Lib" / "site-packages"
    return venv_path / "lib" / _venv_python_dir(venv_path) / "site-packages"


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


def framework_seed_requirements() -> list[str]:
    """Return install requirements for the framework packages that external
    packs depend on.

    Repo mode (development checkout): returns local directory paths for
    ``kernel/``, ``sdk/``, ``mcp_server/``, and bundled ``sdk/packs/*`` —
    the caller installs them with ``uv pip install -e``.

    Tool mode (``uv tool install`` from git): the venv has no framework
    source tree, but each framework distribution records where it was
    installed from in its PEP 610 ``direct_url.json``. For git-sourced
    installs that yields a commit-pinned ``git+<url>@<commit>
    #subdirectory=<sub>`` requirement that can be re-installed into the
    packs venv, giving its resolver a concrete ``lisa-kernel`` candidate
    instead of failing with "not found in the package registry".

    Returns an empty list when no framework package source can be located
    (callers should then skip seeding rather than fail).
    """
    # Repo mode: framework member directories under the framework root.
    if not is_tool_install():
        fw_root = framework_root()
        dirs = [
            str(fw_root / member_dir)
            for dist_name, member_dir in _REPO_SEED_MEMBERS.items()
            if (fw_root / member_dir).is_dir()
        ]
        packs_root = fw_root / "sdk" / "packs"
        if packs_root.is_dir():
            for child in sorted(packs_root.iterdir()):
                if child.is_dir() and (child / "pyproject.toml").is_file():
                    dirs.append(str(child))
        return dirs

    # Tool mode: reconstruct install requirements from PEP 610 direct_url.json
    # metadata recorded for each framework distribution in this venv.
    requirements: list[str] = []
    seen: set[str] = set()
    for dist_name in _TOOL_SEED_DISTRIBUTIONS:
        try:
            dist = importlib_metadata.distribution(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        direct_url_raw = dist.read_text("direct_url.json")
        if not direct_url_raw:
            continue
        try:
            payload = json.loads(direct_url_raw)
        except json.JSONDecodeError:
            continue
        requirement = _requirement_from_direct_url(dist_name, payload)
        if requirement and requirement not in seen:
            seen.add(requirement)
            requirements.append(requirement)
    return requirements


def _requirement_from_direct_url(dist_name: str, payload: object) -> str | None:
    """Convert one PEP 610 ``direct_url.json`` payload to a pip requirement.

    Handles the two shapes the installer produces:

    - VCS installs (git mode): ``git+<url>@<commit>#subdirectory=<sub>``.
      The commit pin makes the requirement reproducible and prevents the
      resolver from re-resolving a moving branch.
    - Local-directory installs (``--source local``): the recorded absolute
      path, reinstalled non-editable into the packs venv.

    Returns ``None`` for shapes that cannot be reinstalled (e.g. registry
    installs), letting the caller skip them.
    """
    if not isinstance(payload, dict):
        return None
    url = payload.get("url")
    vcs_info = payload.get("vcs_info")
    if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git":
        if not isinstance(url, str) or not url:
            return None
        commit = vcs_info.get("commit_id")
        requirement = f"{dist_name} @ git+{url}"
        if isinstance(commit, str) and commit:
            requirement += f"@{commit}"
        subdirectory = payload.get("subdirectory")
        if isinstance(subdirectory, str) and subdirectory:
            requirement += f"#subdirectory={subdirectory}"
        return requirement
    dir_info = payload.get("dir_info")
    if isinstance(url, str) and url.startswith("file://"):
        path = url.removeprefix("file://")
        if isinstance(dir_info, dict) and dir_info.get("editable"):
            return f"-e {path}"
        return path
    return None


# Repo-mode seed members: distribution name → directory under framework root.
_REPO_SEED_MEMBERS: dict[str, str] = {
    "lisa-kernel": "kernel",
    "lisa-sdk": "sdk",
    "lisa-mcp-servers": "mcp_server",
}

# Tool-mode seed distributions: framework packages external packs depend on.
_TOOL_SEED_DISTRIBUTIONS: tuple[str, ...] = (
    "lisa-kernel",
    "lisa-sdk",
    "lisa-mcp-servers",
    "lisa-pack-core",
    "lisa-pack-memory",
    "lisa-pack-introspect",
    "lisa-pack-recovery",
)


__all__ = [
    "LISA_PACKS_VENV_ENV",
    "default_packs_venv",
    "framework_root",
    "framework_seed_requirements",
    "inject_packs_venv_site_packages",
    "is_tool_install",
    "packs_venv_for",
    "packs_venv_site_packages",
    "venv_root",
]
