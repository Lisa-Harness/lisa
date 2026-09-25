"""Runtime pack management: install/uninstall, activation, spec validation, venv seeding.

This is the largest command cluster (~715 lines). It owns:

- Lock-file I/O (``_load_installed_packs_lock``, ``_write_installed_packs_lock``).
- Pack discovery + alias resolution (``installed_pack_roots_by_name``,
  ``pack_alias_map``, ``resolve_pack_name``).
- Capability-provider auto-selection on install (``_auto_select_capability_provider_on_install``,
  ``_clear_selected_provider_for_package``, ``_pack_contributes_memory_capability``).
- Activation toggling (``set_pack_enabled``).
- Trust-policy env helpers (``pack_allowed_prefixes``, ``pack_allowed_git_hosts``,
  ``pack_require_git_ref``, ``pack_git_root_template``).
- Spec parsing/validation (``_is_git_pack_spec``, ``_is_local_pack_spec``,
  ``_extract_package_name_from_spec``, ``_validate_pack_spec_for_install``,
  ``_normalize_pack_install_spec``, ``_is_bare_pack_name``, ``_expand_bare_pack_name``).
- Packs-venv lifecycle (``_seed_packs_venv_with_framework``, ``_ensure_packs_venv``).
- Install/uninstall drivers (``run_pack_install``, ``run_pack_uninstall``).
- In-place update drivers (``run_pack_update``, ``run_pack_update_all``).

The ``/pack`` dispatcher (``handle_pack_command``) lives in ``__init__.py``
because it delegates to ``_simple`` and ``_lists`` in addition to this module.

Dependency direction: ``_packs`` → ``kernel.*`` (``content_roots``,
``pack_activation``, ``mcp_launch.server_manifest``), ``host.content_roots``,
``host.paths``, ``host.runtime``. No imports from sibling
``commands._*`` modules — this cluster is self-contained.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.content_roots import discover_installed_content_roots
from kernel.pack_activation import normalize_pack_name

from host.content_roots import (
    always_on_pack_names,
    infer_pack_package_name,
    pack_aliases,
    save_installed_packs_lock,
)
from host.content_roots import (
    load_installed_packs_lock as _content_roots_load_lock,
)
from host.paths import (
    framework_root as _framework_root,
)
from host.paths import (
    framework_seed_requirements,
)
from host.paths import (
    packs_venv_for as _packs_venv_for,
)
from host.runtime import HostRuntime

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────

LISA_PACK_ALLOWED_PREFIXES_ENV = "LISA_PACK_ALLOWED_PREFIXES"
LISA_PACK_GIT_ALLOW_HOSTS_ENV = "LISA_PACK_GIT_ALLOW_HOSTS"
LISA_PACK_REQUIRE_GIT_REF_ENV = "LISA_PACK_REQUIRE_GIT_REF"
LISA_PACK_GIT_ROOT_ENV = "LISA_PACK_GIT_ROOT"

# Lock source kinds that carry a re-installable ``source_ref`` (eligible for
# in-place update). Mirrors ``host.content_roots._RUNTIME_INSTALL_SOURCE_KINDS``.
_RUNTIME_UPDATE_SOURCE_KINDS = frozenset({"package", "git", "local"})

# Placeholder ``source_ref`` values stamped by lock seeding for packs that
# were installed outside the runtime installer (see
# ``host.content_roots.seed_bundled_packs_lock``). These are not re-installable
# specs; updating from them would ``uv pip install`` the literal string.
_PLACEHOLDER_SOURCE_REFS = frozenset({"package", "git", "local"})

_FRAMEWORK_ROOT = _framework_root()

# Default git root template for expanding bare pack names (e.g.
# ``lisa-pack-brave`` → ``git+https://github.com/Lisa-Harness/lisa-pack-brave@main``).
# The ``{pack}`` placeholder is replaced with the bare package name. Set
# ``LISA_PACK_GIT_ROOT`` to override with a custom template or to empty ("")
# to disable expansion and fall back to PyPI resolution for bare names.
_DEFAULT_PACK_GIT_ROOT = "git+https://github.com/Lisa-Harness/{pack}@main"


# ── Lock-file I/O ────────────────────────────────────────────────────


def _load_installed_packs_lock(runtime: HostRuntime) -> dict[str, Any]:
    """Load installed pack provenance metadata from disk."""
    return _content_roots_load_lock(runtime.config.data_dir)


def _write_installed_packs_lock(runtime: HostRuntime, payload: dict[str, Any]) -> None:
    """Persist installed pack provenance metadata to disk.

    Delegates to :func:`host.content_roots.save_installed_packs_lock`
    so the lock file has one writer with a consistent serialization format.
    """
    save_installed_packs_lock(runtime.config.data_dir, payload)


# ── Pack discovery + alias resolution ────────────────────────────────


def installed_pack_roots_by_name() -> dict[str, Path]:
    """Return a ``{package_name: root}`` map for all discovered installed pack roots."""
    roots: dict[str, Path] = {}
    for root in discover_installed_content_roots():
        package_name = infer_pack_package_name(root)
        if not package_name or package_name in roots:
            continue
        roots[package_name] = root
    return roots


def pack_alias_map(lock: dict[str, Any]) -> dict[str, str]:
    """Build a ``{alias: package_name}`` map from installed roots, always-on, and lock."""
    alias_map: dict[str, str] = {}
    installed_roots = installed_pack_roots_by_name()

    for package_name, root in installed_roots.items():
        for alias in pack_aliases(package_name, root=root):
            alias_map.setdefault(alias, package_name)

    for package_name in always_on_pack_names():
        for alias in pack_aliases(package_name):
            alias_map.setdefault(alias, package_name)

    entries = lock.get("packs") if isinstance(lock, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        package_name = normalize_pack_name(str(entry.get("package_name") or ""))
        if not package_name:
            continue
        for alias in pack_aliases(package_name):
            alias_map.setdefault(alias, package_name)

    return alias_map


def resolve_pack_name(
    raw: str,
    lock: dict[str, Any] | None = None,
) -> str:
    """Resolve a user-supplied pack name or alias to its canonical package name."""
    normalized = normalize_pack_name(raw)
    if not normalized:
        return ""

    lock_payload = lock if isinstance(lock, dict) else {}
    aliases = pack_alias_map(lock_payload)
    if normalized in aliases:
        return aliases[normalized]

    prefixed = normalized if normalized.startswith("lisa-pack-") else f"lisa-pack-{normalized}"
    return aliases.get(prefixed, "")


# ── Capability-provider selection ────────────────────────────────────


def _clear_selected_provider_for_package(
    lock: dict[str, Any],
    package_name: str,
) -> list[str]:
    """Remove selected-capability-provider entries that point to ``package_name``."""
    selected = lock.get("selected_capability_providers")
    if not isinstance(selected, dict):
        lock["selected_capability_providers"] = {}
        return []

    cleared: list[str] = []
    for capability_id, provider_id in list(selected.items()):
        if not isinstance(capability_id, str) or not isinstance(provider_id, str):
            continue
        if normalize_pack_name(provider_id) != package_name:
            continue
        selected.pop(capability_id, None)
        cleared.append(capability_id)
    return cleared


def _pack_contributes_memory_capability(package_name: str) -> bool:
    """Return True if the pack contributes a memory capability."""
    normalized = normalize_pack_name(package_name)
    if not normalized:
        return False

    from kernel.mcp_launch.server_manifest import (  # noqa: PLC0415
        MCPServerManifestError,
        load_mcp_server_manifest,
    )

    for root in discover_installed_content_roots():
        if infer_pack_package_name(root) != normalized:
            continue
        manifest_path = root / "mcp_servers.yaml"
        if not manifest_path.is_file():
            continue
        try:
            _, memory = load_mcp_server_manifest(manifest_path)
        except MCPServerManifestError:
            continue
        if memory.default_backend or memory.backends:
            return True

    return normalized.startswith("lisa-pack-mem")


def _auto_select_capability_provider_on_install(
    lock: dict[str, Any],
    package_name: str,
) -> None:
    """Auto-select a newly-installed pack as the memory capability provider if eligible."""
    selected = lock.get("selected_capability_providers")
    if not isinstance(selected, dict):
        selected = {}
        lock["selected_capability_providers"] = selected

    capability_id = "memory"
    if selected.get(capability_id):
        return
    if package_name in always_on_pack_names():
        return
    if not _pack_contributes_memory_capability(package_name):
        return

    selected[capability_id] = package_name


# ── Activation toggling ──────────────────────────────────────────────


def set_pack_enabled(
    runtime: HostRuntime,
    package_name: str,
    *,
    enabled: bool,
) -> tuple[bool, str]:
    """Persist explicit activation for one pack and request host restart."""
    lock = _load_installed_packs_lock(runtime)
    resolved = resolve_pack_name(package_name, lock)
    if not resolved:
        return False, (
            f"Unknown pack '{package_name}'. Run /pack list to see available packs "
            "and their canonical names."
        )

    always_on = always_on_pack_names(
        runtime.config.mcp_manifest_path,
        always_on_packs=runtime.config.always_on_packs,
    )
    if not enabled and resolved in always_on:
        return False, f"Pack '{resolved}' is always on and cannot be disabled."

    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    index = next(
        (
            i
            for i, entry in enumerate(entries)
            if normalize_pack_name(str(entry.get("package_name") or "")) == resolved
        ),
        -1,
    )
    now_iso = datetime.now(UTC).isoformat()

    if index >= 0:
        entry = dict(entries[index])
        entry["package_name"] = resolved
        entry["enabled"] = enabled
        entry["updated_at"] = now_iso
        entry.setdefault("source_kind", "bundled")
        if entry.get("source_kind") == "bundled":
            entry.setdefault("source_ref", "bundled")
        entries[index] = entry
    else:
        entries.append(
            {
                "package_name": resolved,
                "enabled": enabled,
                "source_kind": "bundled",
                "source_ref": "bundled",
                "updated_at": now_iso,
            }
        )

    lock["packs"] = entries

    cleared: list[str] = []
    if not enabled:
        cleared = _clear_selected_provider_for_package(lock, resolved)

    _write_installed_packs_lock(runtime, lock)
    action = "enable" if enabled else "disable"
    runtime.session.request_restart(reason=f"pack {action}: {resolved}")

    if enabled:
        return (
            True,
            f"Enabled '{resolved}'. Restart required to reload content roots and MCP manifests.",
        )

    if cleared:
        return (
            True,
            f"Disabled '{resolved}' and cleared provider selections for: "
            f"{', '.join(sorted(cleared))}. "
            "Restart required to reload content roots and MCP manifests.",
        )
    return (
        True,
        f"Disabled '{resolved}'. Restart required to reload content roots and MCP manifests.",
    )


# ── Trust-policy env helpers ─────────────────────────────────────────


def pack_allowed_prefixes() -> list[str]:
    """Return allowed package-name prefixes for runtime pack installs."""
    configured = os.getenv(LISA_PACK_ALLOWED_PREFIXES_ENV, "").strip()
    if not configured:
        return ["lisa-pack-"]
    values = [value.strip().lower() for value in configured.split(",") if value.strip()]
    return values or ["lisa-pack-"]


def pack_allowed_git_hosts() -> set[str]:
    """Return allowed hosts for git-based pack installation refs."""
    configured = os.getenv(LISA_PACK_GIT_ALLOW_HOSTS_ENV, "").strip()
    if not configured:
        return {"github.com"}
    values = {value.strip().lower() for value in configured.split(",") if value.strip()}
    return values or {"github.com"}


def pack_require_git_ref() -> bool:
    """Return whether git install refs must pin @tag/@branch/@sha."""
    configured = os.getenv(LISA_PACK_REQUIRE_GIT_REF_ENV, "true").strip().lower()
    return configured not in {"0", "false", "no", "off"}


def pack_git_root_template() -> str:
    """Return the git root template for expanding bare pack names.

    Reads ``LISA_PACK_GIT_ROOT`` (default:
    ``git+https://github.com/Lisa-Harness/{pack}@main``). The ``{pack}``
    placeholder is replaced with the bare package name at expansion time.
    Set the env var to an empty string to disable bare-name expansion.
    """
    return os.getenv(LISA_PACK_GIT_ROOT_ENV, _DEFAULT_PACK_GIT_ROOT).strip()


# ── Spec parsing / validation ────────────────────────────────────────


def _is_git_pack_spec(spec: str) -> bool:
    return spec.strip().lower().startswith("git+")


def _is_local_pack_spec(spec: str) -> bool:
    raw = spec.strip().strip('"')
    if not raw:
        return False
    return Path(raw).expanduser().is_dir()


def _read_local_pack_project_name(path: Path) -> str:
    """Return ``project.name`` from a local pack's pyproject.toml, or empty."""
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return ""
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    project = payload.get("project")
    if not isinstance(project, dict):
        return ""
    name = project.get("name")
    return str(name).strip() if isinstance(name, str) else ""


def _extract_package_name_from_spec(spec: str) -> str:
    """Infer package name from package or git install spec."""
    raw = spec.strip()
    if not raw:
        return ""
    if _is_local_pack_spec(raw):
        path = Path(raw.strip('"')).expanduser()
        project_name = _read_local_pack_project_name(path)
        if project_name:
            return project_name
        return path.name.strip()
    if _is_git_pack_spec(raw):
        egg_marker = "#egg="
        lowered = raw.lower()
        marker_index = lowered.find(egg_marker)
        if marker_index >= 0:
            return raw[marker_index + len(egg_marker) :].strip()
        base = raw.split("@", 1)[0].rstrip("/")
        tail = base.rsplit("/", 1)[-1]
        return tail[:-4] if tail.lower().endswith(".git") else tail

    first = raw.split()[0]
    name = re.split(r"[<>=!~\[]", first, maxsplit=1)[0]
    return name.strip()


def _extract_git_host(spec: str) -> str:
    """Best-effort host extraction for git+https refs."""
    raw = spec.strip()
    if not raw.lower().startswith("git+https://"):
        return ""
    remainder = raw[len("git+https://") :]
    if not remainder:
        return ""
    return remainder.split("/", 1)[0].strip().lower()


def _validate_pack_spec_for_install(spec: str) -> tuple[bool, str]:  # noqa: C901, PLR0911
    """Validate a runtime pack install reference against host trust policy."""
    normalized = spec.strip()
    if not normalized:
        return False, "Usage: /pack install <package-ref-or-git-ref>"

    if _is_local_pack_spec(normalized):
        pack_path = Path(normalized.strip('"')).expanduser()
        project_name = _read_local_pack_project_name(pack_path)
        if not project_name:
            return False, (
                f"Local pack '{pack_path}' is missing a valid pyproject.toml project.name."
            )
        package_name = project_name.lower()
        if not any(package_name.startswith(prefix) for prefix in pack_allowed_prefixes()):
            return False, (
                f"Pack '{package_name}' rejected: allowed package prefixes are "
                f"{', '.join(pack_allowed_prefixes())}."
            )
        return True, ""

    if _is_git_pack_spec(normalized):
        host = _extract_git_host(normalized)
        if not host:
            return False, "Only git+https refs are supported for runtime git installs."
        if host not in pack_allowed_git_hosts():
            return False, f"Git host '{host}' is not in the allowlist."
        if pack_require_git_ref() and "@" not in normalized:
            return False, "Git refs must pin @tag/@branch/@sha (policy LISA_PACK_REQUIRE_GIT_REF)."
        package_name = _extract_package_name_from_spec(normalized).lower()
        if not any(package_name.startswith(prefix) for prefix in pack_allowed_prefixes()):
            return False, (
                f"Pack '{package_name}' rejected: allowed package prefixes are "
                f"{', '.join(pack_allowed_prefixes())}."
            )
        return True, ""

    package_name = _extract_package_name_from_spec(normalized).lower()
    if not package_name:
        return False, "Invalid package ref."
    if not any(package_name.startswith(prefix) for prefix in pack_allowed_prefixes()):
        return False, (
            f"Pack '{package_name}' rejected: allowed package prefixes are "
            f"{', '.join(pack_allowed_prefixes())}."
        )
    return True, ""


# ── Bundled packs + packs venv lifecycle ─────────────────────────────


async def _seed_packs_venv_with_framework(packs_venv: Path) -> tuple[bool, str]:
    """Seed the packs venv with editable installs of framework packages.

    External packs (installed from git or PyPI) declare dependencies on
    ``lisa-harness-kernel``, ``lisa-harness-sdk``, ``lisa-harness-mcp-servers``, and any bundled
    ``lisa-pack-*`` — all workspace packages that are **not** published to
    PyPI. Without seeding, ``uv pip install`` fails:
    *"lisa-harness-kernel was not found in the package registry"*.

    This installs the framework packages as editable ``-e`` installs from
    their source directories so the packs venv can resolve those dependencies.
    Seeding is idempotent: ``uv pip install -e`` is a no-op if the package is
    already present at the same version, so this is safe to call on every
    install (existing venvs are retro-seeded without side effects).
    """
    return await asyncio.to_thread(_seed_packs_venv_with_framework_sync, packs_venv)


def _seed_packs_venv_with_framework_sync(packs_venv: Path) -> tuple[bool, str]:
    """Synchronous variant of :func:`_seed_packs_venv_with_framework`.

    Used by the startup auto-install path (which runs in the synchronous
    ``_initialize_runtime`` of every host) and wrapped in
    :func:`asyncio.to_thread` by the async variant above.

    Seed targets come from :func:`host.paths.framework_seed_requirements`:

    - Repo mode: local member directories (``kernel/``, ``sdk/``,
      ``mcp_server/``, bundled ``sdk/packs/*``), installed editable.
    - Tool mode (``uv tool install`` from git): commit-pinned
      ``git+`` requirements reconstructed from each framework package's
      PEP 610 ``direct_url.json``. The tool venv has no framework source
      tree, so without this the seed list would be empty and external pack
      installs would fail with "lisa-harness-kernel was not found in the package
      registry".
    """
    import subprocess

    seed_targets: list[str] = []
    for target in framework_seed_requirements():
        if target.startswith("-e "):
            # Editable local path (repo mode).
            seed_targets.append("-e")
            seed_targets.append(target.removeprefix("-e "))
        else:
            seed_targets.append(target)
    if not seed_targets:
        return True, ""

    completed = subprocess.run(
        ["uv", "pip", "install", "--python", str(packs_venv), *seed_targets],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_FRAMEWORK_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        output = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, (
            f"Failed to seed packs venv with framework packages at '{packs_venv}'.\n{preview}"
        )
    logger.info("Seeded packs venv at %s with framework packages", packs_venv)
    return True, ""


async def _ensure_packs_venv(packs_venv: Path) -> tuple[bool, str]:
    """Ensure the packs venv exists as a real virtual environment.

    ``uv pip install --python <path>`` requires a real venv (a directory with
    ``pyvenv.cfg`` and an interpreter), not merely an empty directory. If the
    venv is missing or is not a valid venv, run ``uv venv`` to create one. The
    packs venv lives inside the data dir, so the parent directory is created
    first.

    Seeding (editable installs of framework workspace packages **and** bundled
    packs under ``sdk/packs/``) is performed by the caller —
    :func:`run_pack_install` calls :func:`_seed_packs_venv_with_framework` on
    **every** install, not only on venv creation, so that existing venvs that
    predate a bundled pack are retro-seeded (seeding is idempotent). This
    module's ``_ensure_packs_venv`` only owns venv creation, not seeding.

    Returns ``(True, "")`` on success (venv exists or was created) or
    ``(False, message)`` if venv creation failed.
    """
    return await asyncio.to_thread(_ensure_packs_venv_sync, packs_venv)


def _ensure_packs_venv_sync(packs_venv: Path) -> tuple[bool, str]:
    """Synchronous variant of :func:`_ensure_packs_venv`.

    Used by the startup auto-install path and wrapped in
    :func:`asyncio.to_thread` by the async variant above.
    """
    import subprocess

    # A valid venv has pyvenv.cfg at its root. If present, assume usable.
    if (packs_venv / "pyvenv.cfg").is_file():
        return True, ""

    packs_venv.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["uv", "venv", str(packs_venv)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_FRAMEWORK_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        output = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, (f"Failed to create packs venv at '{packs_venv}'.\n{preview}")
    logger.info("Created packs venv at %s", packs_venv)
    return True, ""


def _is_bare_pack_name(spec: str) -> bool:
    """Return True if *spec* is a bare package name with no version pins or extras.

    A bare name is one that is not a local path, not a ``git+`` URL, not a
    plain ``https://``/``http://`` URL, and contains no version operators
    (``>=``, ``==``, …), extras (``[...]``), or whitespace. Only bare names
    are eligible for expansion via :func:`pack_git_root_template`.
    """
    raw = spec.strip().strip('"')
    if not raw:
        return False
    if _is_git_pack_spec(raw) or _is_local_pack_spec(raw):
        return False
    lowered = raw.lower()
    if lowered.startswith("https://") or lowered.startswith("http://"):
        return False
    # Version operators, extras brackets, or spaces disqualify.
    return not re.search(r"[<>=!~\[\s]", raw)


def _expand_bare_pack_name(spec: str) -> str:
    """Expand a bare pack name to a git+ URL using the configured root template.

    Returns *spec* unchanged when the template is empty (feature disabled) or
    when the template does not contain the ``{pack}`` placeholder.
    """
    template = pack_git_root_template()
    if not template or "{pack}" not in template:
        return spec
    return template.replace("{pack}", spec.strip())


def _normalize_pack_install_spec(spec: str) -> str:
    """Normalize a pack install spec before validation.

    Three normalizations are applied:

    1. Bare ``https://``/``http://`` GitHub-style URLs are rewritten to the
       ``git+https://`` form that :func:`_is_git_pack_spec` and uv both
       expect. This lets users paste a plain repo URL (e.g.
       ``https://github.com/Lisa-Harness/lisa-pack-brave``) instead
       of having to remember the ``git+`` prefix.
    2. Bare pack names matching the allowed prefix (e.g.
       ``lisa-pack-brave``) are expanded to a ``git+`` URL using
       :func:`pack_git_root_template` (default:
       ``git+https://github.com/Lisa-Harness/{pack}@main``). This is a
       shorthand so users can type just the pack name for framework packs
       instead of a full git URL. Set ``LISA_PACK_GIT_ROOT`` to override the
       template or to empty to disable expansion.

    Both normalizations route through the git-validation path (host allowlist
    + ref-pinning policy) so error messages stay actionable.

    Local paths and specs already carrying ``git+`` are returned as-is.
    """
    raw = spec.strip().strip('"')
    if not raw:
        return spec
    lowered = raw.lower()
    if lowered.startswith("https://") or lowered.startswith("http://"):
        return "git+" + raw
    if _is_bare_pack_name(raw):
        expanded = _expand_bare_pack_name(raw)
        if expanded != raw:
            return expanded
    return spec


# ── Install / uninstall drivers ──────────────────────────────────────


async def run_pack_install(
    runtime: HostRuntime,
    spec: str,
) -> tuple[bool, str]:
    """Install one pack ref using uv pip and persist provenance metadata."""
    return await asyncio.to_thread(run_pack_install_sync, runtime, spec, request_restart=True)


def run_pack_install_sync(
    runtime: HostRuntime,
    spec: str,
    *,
    request_restart: bool = True,
    editable: bool = False,
) -> tuple[bool, str]:
    """Synchronous pack-install core shared by the REPL command and boot auto-install.

    Validates the spec against the host trust policy, ensures + seeds the
    packs venv, runs ``uv pip install``, and persists the install lock. When
    ``request_restart`` is ``True`` (the REPL/``/pack install`` path), the
    runtime is flagged for restart so the user is prompted to relaunch. When
    ``False`` (the boot auto-install path), the caller manages restart
    semantics itself — it may batch multiple installs and request one restart
    after the loop, or none if entry-point discovery sees the new packs
    in-process.

    When ``editable`` is ``True`` and the resolved install target is a local
    directory, ``-e`` is prepended to the ``uv pip install`` args so the pack
    is installed as an editable install. Non-local targets (git refs, PyPI
    package names) ignore ``editable`` — ``uv`` rejects ``-e`` for non-local
    specs, so it is only added for local directory sources.
    """
    import subprocess

    normalized_spec = _normalize_pack_install_spec(spec)
    is_allowed, reason = _validate_pack_spec_for_install(normalized_spec)
    if not is_allowed:
        return False, reason

    is_local = _is_local_pack_spec(normalized_spec)
    install_target = normalized_spec.strip().strip('"') if is_local else normalized_spec
    packs_venv = _packs_venv_for(runtime.paths.data_dir)
    ensured, ensure_err = _ensure_packs_venv_sync(packs_venv)
    if not ensured:
        return False, f"Install failed for '{normalized_spec}'.\n{ensure_err}"
    seeded, seed_err = _seed_packs_venv_with_framework_sync(packs_venv)
    if not seeded:
        return False, f"Install failed for '{normalized_spec}'.\n{seed_err}"
    install_args = ["uv", "pip", "install", "--python", str(packs_venv)]
    if editable and is_local:
        install_args.append("-e")
    install_args.append(install_target)
    completed = subprocess.run(
        install_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_FRAMEWORK_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        output = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, f"Install failed for '{normalized_spec}'.\n{preview}"

    package_name = normalize_pack_name(_extract_package_name_from_spec(normalized_spec))
    if not package_name:
        return (
            False,
            f"Install succeeded for '{normalized_spec}' but package name could not be inferred.",
        )
    source_kind = (
        "local" if is_local else ("git" if _is_git_pack_spec(normalized_spec) else "package")
    )
    lock = _load_installed_packs_lock(runtime)
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    now_iso = datetime.now(UTC).isoformat()
    filtered = [
        entry
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or "")) != package_name
    ]
    entry_payload: dict[str, Any] = {
        "package_name": package_name,
        "installed_at": now_iso,
        "source_kind": source_kind,
        "source_ref": install_target,
        "enabled": True,
        "updated_at": now_iso,
    }
    if is_local and editable:
        entry_payload["editable"] = True
    filtered.append(entry_payload)
    lock["packs"] = filtered
    _auto_select_capability_provider_on_install(lock, package_name)
    _write_installed_packs_lock(runtime, lock)

    if request_restart:
        runtime.session.request_restart(reason=f"pack install: {package_name}")
    editable_label = " (editable)" if is_local and editable else ""
    return True, (
        f"Installed '{package_name}' from {source_kind} source{editable_label}. "
        "Restart required to reload entry points, pack catalogs, and MCP manifests."
    )


async def run_pack_uninstall(
    runtime: HostRuntime,
    package_name: str,
) -> tuple[bool, str]:
    """Uninstall one pack and update provenance metadata."""
    requested = package_name.strip()
    if not requested:
        return False, "Usage: /pack uninstall <package-name>"

    lock = _load_installed_packs_lock(runtime)
    normalized = resolve_pack_name(requested, lock) or normalize_pack_name(requested)
    if not normalized:
        return False, "Usage: /pack uninstall <package-name>"

    packs_venv = _packs_venv_for(runtime.paths.data_dir)
    ensured, ensure_err = await _ensure_packs_venv(packs_venv)
    if not ensured:
        return False, f"Uninstall failed for '{normalized}'.\n{ensure_err}"
    process = await asyncio.create_subprocess_exec(
        "uv",
        "pip",
        "uninstall",
        "-y",
        "--python",
        str(packs_venv),
        normalized,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_FRAMEWORK_ROOT),
    )
    output_bytes, _ = await process.communicate()
    output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""
    if process.returncode != 0:
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, f"Uninstall failed for '{normalized}'.\n{preview}"

    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    filtered = [
        entry
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or "")) != normalized
    ]
    lock["packs"] = filtered
    cleared = _clear_selected_provider_for_package(lock, normalized)
    _write_installed_packs_lock(runtime, lock)

    runtime.session.request_restart(reason=f"pack uninstall: {normalized}")
    if cleared:
        cleared_text = ", ".join(sorted(cleared))
        return True, (
            f"Uninstalled '{normalized}' and cleared provider selections for: {cleared_text}. "
            "Restart required to unload entry points, pack catalogs, and MCP manifests."
        )
    return True, (
        f"Uninstalled '{normalized}'. "
        "Restart required to unload entry points, pack catalogs, and MCP manifests."
    )


# ── Update drivers ───────────────────────────────────────────────────


async def run_pack_update(
    runtime: HostRuntime,
    package_name: str,
) -> tuple[bool, str]:
    """Upgrade one installed pack in place from its recorded install source.

    Resolves the pack name from the lock (aliases accepted), then re-runs the
    install core with the pack's recorded ``source_ref`` so git pins, local
    paths, and PyPI refs all refresh through the same trust-validated path.
    """
    return await asyncio.to_thread(run_pack_update_sync, runtime, package_name)


def run_pack_update_sync(
    runtime: HostRuntime,
    package_name: str,
    *,
    request_restart: bool = True,
) -> tuple[bool, str]:
    """Synchronous in-place update of one installed pack.

    Shares the install core (``run_pack_install_sync``) so provenance and
    capability-provider selection stay identical to a fresh install, but the
    update driver owns restart semantics so the restart reason reflects
    ``pack update`` rather than ``pack install``.

    Only packs with a real lock entry carrying a re-installable
    ``source_ref`` are updatable: bundled packs (``source_kind == "bundled"``)
    ship with the framework tree and have nothing to re-fetch, and
    placeholder refs stamped by lock seeding (``"package"``) identify packs
    installed outside the runtime installer — re-running those would
    re-install the literal placeholder string.
    """
    requested = package_name.strip()
    if not requested:
        return False, "Usage: /pack update <package-name>"

    lock = _load_installed_packs_lock(runtime)
    resolved = resolve_pack_name(requested, lock)
    if not resolved:
        return False, (
            f"Unknown pack '{requested}'. Run /pack list to see available packs "
            "and their canonical names."
        )

    entry = next(
        (
            item
            for item in lock.get("packs", [])
            if isinstance(item, dict)
            and normalize_pack_name(str(item.get("package_name") or "")) == resolved
        ),
        None,
    )
    if entry is None:
        return False, (
            f"Pack '{resolved}' has no installed-packs lock entry (not installed by "
            "the runtime installer); there is no recorded source to update from."
        )
    source_kind = str(entry.get("source_kind") or "").strip().lower()
    source_ref = str(entry.get("source_ref") or "").strip()
    if source_kind == "bundled" or not source_ref or source_ref in _PLACEHOLDER_SOURCE_REFS:
        return False, (
            f"Pack '{resolved}' is bundled with the framework (no install source to "
            "refresh). Upgrade the framework itself to update it."
        )

    ok, details = run_pack_install_sync(runtime, source_ref, request_restart=False)
    if not ok:
        return False, details
    if request_restart:
        runtime.session.request_restart(reason=f"pack update: {resolved}")
    return True, details.replace(
        "Installed '",
        f"Updated '{resolved}' from '{source_ref}'. Installed '",
        1,
    )


async def run_pack_update_all(runtime: HostRuntime) -> tuple[bool, str]:
    """Upgrade every runtime-installed pack with a recorded install source.

    Iterates the lock's runtime entries (``package``, ``git``, ``local``),
    skipping bundled packs, entries with no usable ``source_ref``, and
    placeholder refs stamped by lock seeding, and re-installs each remaining
    entry from its recorded source. A single restart is requested after the
    batch regardless of how many packs were refreshed; per-pack failures do
    not stop the batch and are listed in the output.
    """
    lock = _load_installed_packs_lock(runtime)
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    updatable: list[tuple[str, str]] = []
    for entry in entries:
        package_name = normalize_pack_name(str(entry.get("package_name") or ""))
        source_kind = str(entry.get("source_kind") or "").strip().lower()
        source_ref = str(entry.get("source_ref") or "").strip()
        if not package_name or source_kind not in _RUNTIME_UPDATE_SOURCE_KINDS:
            continue
        if not source_ref or source_ref in _PLACEHOLDER_SOURCE_REFS:
            continue
        updatable.append((package_name, source_ref))

    if not updatable:
        return True, "No runtime-installed packs with recorded sources to update."

    updated: list[str] = []
    failed: list[tuple[str, str]] = []
    for package_name, source_ref in updatable:
        ok, details = run_pack_install_sync(
            runtime,
            source_ref,
            request_restart=False,
        )
        if ok:
            updated.append(package_name)
        else:
            failed.append((package_name, details))

    runtime.session.request_restart(reason=f"pack update-all: {len(updated)} pack(s)")

    lines: list[str] = []
    if updated:
        lines.append(
            "Updated " + ", ".join(sorted(updated)) + ". "
            "Restart required to reload entry points, pack catalogs, and MCP manifests."
        )
    else:
        lines.append("No packs were updated.")
    for package_name, details in failed:
        preview = details.splitlines()[-1].strip() if details.strip() else ""
        suffix = f" ({preview})" if preview else ""
        lines.append(f"- Update failed for '{package_name}'{suffix}.")
    return bool(updated), "\n".join(lines)
