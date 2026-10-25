"""Transport-neutral content root and pack activation discovery.

This module consolidates the pure, stateless logic for:

- Inferring pack package names from discovered content roots.
- Resolving the "always-on" pack set from a host manifest's
  ``pack_policy.always_on_packs``.
- Reading the installed-pack provenance lock file to determine which packs
  are runtime-installed and enabled.
- Deciding which installed content roots are active for the current host
  configuration.
- Materialising the SDK content / authored-content / mode / operation root
  lists that downstream MCP content servers and kernel discovery consume.

All functions are transport-neutral: they take explicit ``data_dir`` and
``host_manifest_path`` parameters rather than reading global runtime state.
A transport host wires its ``HostConfig``-derived paths into these functions at
startup.
"""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
    resolve_content_subdir_roots,
)
from kernel.extension_names import normalize_pack_name

logger = logging.getLogger(__name__)

_PACK_LOCK_FILE_NAME = "installed-packs.json"
EXCLUDED_PACK_NAMES: frozenset[str] = frozenset({"lisa-harness-sdk"})
_DEFAULT_ALWAYS_ON_PACKAGES: tuple[str, ...] = (
    "lisa-pack-core",
    "lisa-pack-introspect",
)
_RUNTIME_INSTALL_SOURCE_KINDS = {"package", "git", "local"}


def _is_bundled_pack_root(root: Path) -> bool:
    """Return True if ``root`` is a bundled pack under ``sdk/packs/``.

    Bundled packs ship with the framework tree (``<framework_root>/sdk/packs/
    <name>``). External packs discovered via entry points — whether installed
    editable from ``lisa_packs/<name>`` or runtime ``pip install``ed into the
    packs venv — are not bundled and should be stamped ``source_kind=
    "package"`` by the seeder.
    """
    try:
        from runtime.paths import framework_root

        bundled_root = framework_root() / "sdk" / "packs"
    except Exception:  # pylint: disable=broad-except
        return False
    try:
        root.resolve().relative_to(bundled_root.resolve())
        return True
    except ValueError:
        return False


def unique_paths(paths: list[Path]) -> list[Path]:
    """Return ``paths`` with duplicates (by resolved path) removed, order kept."""
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def host_mcp_manifest_path(manifest_path: Path | None = None) -> Path:
    """Return the host MCP manifest path.

    Defaults to ``host/mcp_servers.yaml`` when ``manifest_path`` is
    not supplied. Transport hosts with a different manifest location pass
    it explicitly.
    """
    if manifest_path is not None:
        return manifest_path
    return Path(__file__).resolve().parent / "mcp_servers.yaml"


def infer_pack_package_name(root: Path) -> str:
    """Infer the pack package name for one discovered content root."""
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            payload = {}
        project = payload.get("project") if isinstance(payload, dict) else None
        if isinstance(project, dict):
            name = project.get("name")
            if isinstance(name, str) and name.strip():
                return normalize_pack_name(name)

    root_name = normalize_pack_name(root.name)
    # SDK workspace packs live under ``sdk/packs/<name>`` and external packs
    # are installed under ``site-packages/lisa_packs/<name>``. Both map to
    # the canonical ``lisa-pack-<name>`` distribution name so session-plan
    # capability selections such as
    # ``providers.capabilities.memory: lisa-pack-mem0`` resolve correctly.
    if root.parent.name == "packs" and root.parent.parent.name == "sdk" and root_name:
        return f"lisa-pack-{root_name}"
    if root.parent.name == "lisa_packs" and root_name:
        return f"lisa-pack-{root_name}"
    if root_name.startswith("lisa-pack-"):
        return root_name
    return root_name


def installed_pack_version(root: Path, package_name: str) -> str | None:
    """Return the installed distribution version for the pack at ``root``.

    Non-editable packs keep their ``*.dist-info`` beside ``lisa_packs/`` in
    the packs venv, so ``root``'s ancestors are searched first; editable
    installs fall back to ``sys.path`` (which includes the packs venv).
    """
    import sys
    from importlib import metadata

    search_path = [str(parent) for parent in root.parents]
    search_path.extend(entry for entry in sys.path if entry not in search_path)
    for distribution in metadata.distributions(name=package_name, path=search_path):
        version = distribution.version
        if isinstance(version, str) and version:
            return version
    return None


def pack_aliases(package_name: str, *, root: Path | None = None) -> set[str]:
    """Build the set of alias strings that refer to ``package_name``."""
    aliases: set[str] = set()
    normalized = normalize_pack_name(package_name)
    if normalized:
        aliases.add(normalized)
        if normalized.startswith("lisa-pack-"):
            aliases.add(normalized[len("lisa-pack-") :])
    if root is not None:
        root_alias = normalize_pack_name(root.name)
        if root_alias:
            aliases.add(root_alias)
    return aliases


def load_pack_policy_dict(host_manifest_path: Path, *, warning_subject: str) -> dict[str, Any]:
    """Load and return ``pack_policy`` from the host manifest."""
    if not host_manifest_path.exists():
        return {}

    try:
        raw = yaml.safe_load(host_manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning(
            "Failed to read %s from %s",
            warning_subject,
            host_manifest_path,
            exc_info=True,
        )
        return {}

    if not isinstance(raw, dict):
        return {}
    policy = raw.get("pack_policy") or {}
    if not isinstance(policy, dict):
        return {}
    return policy


def runtime_pack_lock_path(data_dir: Path) -> Path:
    """Return the path to the installed pack provenance lock file."""
    from sdk.layout import STATE_DIR

    return data_dir / STATE_DIR / _PACK_LOCK_FILE_NAME


def load_disabled_pack_names(data_dir: Path) -> set[str]:
    """Return normalized names explicitly disabled in the installed-packs lock."""
    lock_path = runtime_pack_lock_path(data_dir)
    if not lock_path.exists():
        return set()
    try:
        decoded = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read pack lock %r", str(lock_path), exc_info=True)
        return set()
    if not isinstance(decoded, dict) or not isinstance(decoded.get("packs"), list):
        return set()
    disabled: set[str] = set()
    for item in decoded["packs"]:
        if isinstance(item, dict) and not bool(item.get("enabled", True)):
            package_name = normalize_pack_name(str(item.get("package_name") or ""))
            if package_name:
                disabled.add(package_name)
    return disabled


def save_installed_packs_lock(data_dir: Path, payload: dict[str, Any]) -> None:
    """Persist installed pack provenance metadata to disk.

    Single writer for the lock file. Both startup seeding and the pack
    management commands go through this helper so the serialization format
    (indent, trailing newline, ensure_ascii) stays consistent.
    """
    path = runtime_pack_lock_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _normalize_lock_pack_entry(item: Any) -> dict[str, Any] | None:
    """Return a normalized lock entry for ``item``, or None if it is unusable.

    Non-dict items and entries without a resolvable package name are dropped
    so malformed lock files degrade into an empty pack list rather than
    breaking startup.
    """
    if not isinstance(item, dict):
        return None
    package_name = normalize_pack_name(str(item.get("package_name") or ""))
    if not package_name:
        return None
    source_kind = str(item.get("source_kind") or "package").strip().lower() or "package"
    return {
        **item,
        "package_name": package_name,
        "source_kind": source_kind,
        "enabled": bool(item.get("enabled", True)),
    }


def _normalize_selected_providers(selected: Any) -> dict[str, str]:
    """Normalize the ``selected_capability_providers`` mapping from the lock."""
    selected_map: dict[str, str] = {}
    if not isinstance(selected, dict):
        return selected_map
    for capability_raw, provider_raw in selected.items():
        if not isinstance(capability_raw, str) or not isinstance(provider_raw, str):
            continue
        capability_id = capability_raw.strip().lower()
        provider_id = normalize_pack_name(provider_raw)
        if capability_id and provider_id:
            selected_map[capability_id] = provider_id
    return selected_map


def load_installed_packs_lock(data_dir: Path) -> dict[str, Any]:
    """Load installed pack provenance metadata from disk."""
    path = runtime_pack_lock_path(data_dir)
    default_payload: dict[str, Any] = {
        "packs": [],
        "selected_capability_providers": {},
    }
    if not path.exists():
        return default_payload
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_payload
    if not isinstance(decoded, dict):
        return default_payload

    packs = decoded.get("packs")
    if not isinstance(packs, list):
        packs = []

    normalized_packs: list[dict[str, Any]] = []
    for item in packs:
        entry = _normalize_lock_pack_entry(item)
        if entry is not None:
            normalized_packs.append(entry)

    return {
        "packs": normalized_packs,
        "selected_capability_providers": _normalize_selected_providers(
            decoded.get("selected_capability_providers")
        ),
    }


def seed_bundled_packs_lock(data_dir: Path) -> bool:
    """Seed the lock with all discovered bundled packs on startup.

    Closes the installed-packs lifecycle hole: without this, bundled packs
    that have never been explicitly toggled are absent from the lock, so
    ``load_disabled_pack_names`` (fail-open) treats them as enabled while
    ``pack_is_active`` (content roots) treats them as disabled — two
    conflicting defaults for the same question.

    Seeding runs on every startup. It is merge-safe and idempotent:

    - Packs already present in the lock are never touched (their
      ``enabled`` / ``source_kind`` / ``source_ref`` / ``updated_at``
      survive unchanged, including explicit ``enabled: false``).
    - Only packs discovered via ``discover_installed_content_roots()`` that
      have no lock entry are appended, with ``enabled: true``,
      ``source_kind: "bundled"``, ``source_ref: "bundled"``.
    - ``selected_capability_providers`` and any runtime-installed entries
      are preserved verbatim.

    Args:
        data_dir: Host data directory containing ``installed-packs.json``.

    Returns:
        ``True`` if the lock was written (new bundled entries were added),
        ``False`` if no change was needed.
    """
    lock = load_installed_packs_lock(data_dir)
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    known = {
        normalize_pack_name(str(entry.get("package_name") or ""))
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or ""))
    }

    added = False
    now_iso = datetime.now(UTC).isoformat()
    for root in discover_installed_content_roots():
        package_name = infer_pack_package_name(root)
        if not package_name or package_name in known:
            continue
        if package_name in EXCLUDED_PACK_NAMES:
            continue
        is_bundled = _is_bundled_pack_root(root)
        entries.append(
            {
                "package_name": package_name,
                "enabled": True,
                "source_kind": "bundled" if is_bundled else "package",
                "source_ref": "bundled" if is_bundled else "package",
                "updated_at": now_iso,
            }
        )
        known.add(package_name)
        added = True

    if not added:
        return False

    lock["packs"] = entries
    save_installed_packs_lock(data_dir, lock)
    return True


def always_on_pack_names(
    host_manifest_path: Path | None = None,
    *,
    always_on_packs: tuple[str, ...] | None = None,
) -> set[str]:
    """Resolve the set of always-on pack names.

    ``always_on_packs`` overrides the module default; when ``None`` the
    default tuple is used. Additional names are merged from the host
    manifest's ``pack_policy.always_on_packs`` list.
    """
    defaults = always_on_packs if always_on_packs is not None else _DEFAULT_ALWAYS_ON_PACKAGES
    always_on = {normalize_pack_name(name) for name in defaults}
    manifest_path = host_mcp_manifest_path(host_manifest_path)
    policy = load_pack_policy_dict(manifest_path, warning_subject="pack activation policy")

    configured = policy.get("always_on_packs") or []
    if not isinstance(configured, list):
        return always_on

    for item in configured:
        if not isinstance(item, str):
            continue
        normalized = normalize_pack_name(item)
        if normalized:
            always_on.add(normalized)
    return always_on


def runtime_installed_pack_names(lock: dict[str, Any]) -> set[str]:
    """Extract the set of runtime-installed package names from a lock dict."""
    entries = lock.get("packs") if isinstance(lock, dict) else None
    if not isinstance(entries, list):
        return set()

    runtime_installed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_kind = str(entry.get("source_kind") or "").strip().lower()
        if source_kind not in _RUNTIME_INSTALL_SOURCE_KINDS:
            continue
        package_name = normalize_pack_name(str(entry.get("package_name") or ""))
        if package_name:
            runtime_installed.add(package_name)
    return runtime_installed


def runtime_enabled_pack_map(lock: dict[str, Any]) -> dict[str, bool]:
    """Build a ``{package_name: enabled}`` map from a lock dict."""
    entries = lock.get("packs") if isinstance(lock, dict) else None
    if not isinstance(entries, list):
        return {}

    enabled_map: dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        package_name = normalize_pack_name(str(entry.get("package_name") or ""))
        if not package_name:
            continue
        enabled_map[package_name] = bool(entry.get("enabled", True))
    return enabled_map


def pack_is_active(
    package_name: str,
    *,
    always_on_packs: set[str],
    runtime_installed: set[str],
    enabled_map: dict[str, bool],
    session_included_packs: set[str] | None = None,
    session_excluded_packs: set[str] | None = None,
) -> bool:
    """Decide whether a pack is active under host, session, then lock policy."""
    if package_name in always_on_packs:
        return True
    if session_excluded_packs and package_name in session_excluded_packs:
        return False
    if session_included_packs and package_name in session_included_packs:
        return True
    if session_included_packs:
        return False
    if package_name in runtime_installed:
        return enabled_map.get(package_name, True)
    return enabled_map.get(package_name, False)


def active_installed_pack_roots(
    data_dir: Path,
    *,
    host_manifest_path: Path | None = None,
    always_on_packs: tuple[str, ...] | None = None,
    session_pack_include: Iterable[str] = (),
    session_pack_exclude: Iterable[str] = (),
) -> list[Path]:
    """Return installed content roots active under host, session, and lock policy."""
    lock = load_installed_packs_lock(data_dir)
    always_on = always_on_pack_names(host_manifest_path, always_on_packs=always_on_packs)
    runtime_installed = runtime_installed_pack_names(lock)
    enabled_map = runtime_enabled_pack_map(lock)
    session_included = _normalize_pack_names(session_pack_include)
    session_excluded = _normalize_pack_names(session_pack_exclude)

    # Combine entry-point-discovered roots (bundled packs installed in the
    # framework venv) with packs venv-discovered roots (external packs
    # installed in the separate packs venv). Entry-point discovery cannot
    # find packs venv packages because ``importlib.metadata`` does not
    # re-scan after ``sys.path`` is modified at startup.
    from kernel.content_roots import (
        discover_installed_content_roots as _discover_entry_point_roots,
    )
    from kernel.content_roots import (
        discover_packs_venv_content_roots as _discover_packs_venv_roots,
    )

    from runtime.paths import (
        default_packs_venv,
        packs_venv_for,
        packs_venv_site_packages,
    )

    def _venv_resolver(data_dir: Path | None) -> Path:
        if data_dir is not None:
            return packs_venv_for(Path(data_dir))
        return default_packs_venv()

    entry_point_roots = _discover_entry_point_roots()
    packs_venv_roots = _discover_packs_venv_roots(
        data_dir,
        venv_path_resolver=_venv_resolver,
        site_packages_resolver=packs_venv_site_packages,
    )
    entry_point_root_count = len(entry_point_roots)
    all_roots = [*entry_point_roots, *packs_venv_roots]
    # The same pack may be discovered from both surfaces when an external
    # pack is also installed into the main venv (e.g. via
    # ``uv sync --extra ext``): entry-point discovery returns the main-venv
    # ``lisa_packs/<name>`` root and the packs venv scan returns the source
    # ``lisa_packs/<name>`` root. Two distinct paths for one pack produce
    # duplicate catalog entries → ``CatalogCollisionError`` at boot. Dedupe
    # by package name, preferring the packs venv root (the authoritative
    # home for external packs); bundled packs only appear via entry points.
    selected_by_name: dict[str, Path] = {}
    root_from_packs_venv: dict[str, bool] = {}
    for index, root in enumerate(all_roots):
        package_name = infer_pack_package_name(root)
        if not package_name:
            continue
        if not pack_is_active(
            package_name,
            always_on_packs=always_on,
            runtime_installed=runtime_installed,
            enabled_map=enabled_map,
            session_included_packs=session_included,
            session_excluded_packs=session_excluded,
        ):
            continue
        from_packs_venv = index >= entry_point_root_count
        existing = selected_by_name.get(package_name)
        if existing is None:
            selected_by_name[package_name] = root
            root_from_packs_venv[package_name] = from_packs_venv
            continue
        # Prefer the packs venv-discovered root for external packs so the
        # authoritative install wins over an accidental main-venv copy.
        if from_packs_venv and not root_from_packs_venv[package_name]:
            selected_by_name[package_name] = root
            root_from_packs_venv[package_name] = True
    return list(selected_by_name.values())


def sdk_content_roots(
    data_dir: Path,
    *,
    host_manifest_path: Path | None = None,
    always_on_packs: tuple[str, ...] | None = None,
    session_pack_include: Iterable[str] = (),
    session_pack_exclude: Iterable[str] = (),
) -> list[Path]:
    """Return the full SDK content root list (active installed + configured)."""
    always_on = always_on_pack_names(host_manifest_path, always_on_packs=always_on_packs)
    session_included = _normalize_pack_names(session_pack_include)
    session_excluded = _normalize_pack_names(session_pack_exclude)
    configured_roots = [
        root
        for root in discover_configured_content_roots()
        if _configured_root_is_active(
            root,
            always_on_packs=always_on,
            session_included_packs=session_included,
            session_excluded_packs=session_excluded,
        )
    ]
    roots = [
        *active_installed_pack_roots(
            data_dir,
            host_manifest_path=host_manifest_path,
            always_on_packs=always_on_packs,
            session_pack_include=session_pack_include,
            session_pack_exclude=session_pack_exclude,
        ),
        *configured_roots,
    ]
    return unique_paths([root for root in roots if root.is_dir()])


def authored_content_roots(
    data_dir: Path,
    *,
    host_manifest_path: Path | None = None,
    always_on_packs: tuple[str, ...] | None = None,
    session_pack_include: Iterable[str] = (),
    session_pack_exclude: Iterable[str] = (),
) -> list[Path]:
    """Return the ``content/`` subdirectories that exist under SDK content roots."""
    roots = sdk_content_roots(
        data_dir,
        host_manifest_path=host_manifest_path,
        always_on_packs=always_on_packs,
        session_pack_include=session_pack_include,
        session_pack_exclude=session_pack_exclude,
    )
    return unique_paths([root / "content" for root in roots if (root / "content").is_dir()])


def sdk_mode_roots(
    data_dir: Path,
    *,
    host_manifest_path: Path | None = None,
    always_on_packs: tuple[str, ...] | None = None,
    session_pack_include: Iterable[str] = (),
    session_pack_exclude: Iterable[str] = (),
) -> list[Path]:
    """Return the ``modes`` subdirectories across SDK content roots."""
    roots = sdk_content_roots(
        data_dir,
        host_manifest_path=host_manifest_path,
        always_on_packs=always_on_packs,
        session_pack_include=session_pack_include,
        session_pack_exclude=session_pack_exclude,
    )
    return unique_paths(resolve_content_subdir_roots(roots, "modes"))


def sdk_operation_roots(
    data_dir: Path,
    *,
    host_manifest_path: Path | None = None,
    always_on_packs: tuple[str, ...] | None = None,
    session_pack_include: Iterable[str] = (),
    session_pack_exclude: Iterable[str] = (),
) -> list[Path]:
    """Return the ``operations`` subdirectories across SDK content roots."""
    roots = sdk_content_roots(
        data_dir,
        host_manifest_path=host_manifest_path,
        always_on_packs=always_on_packs,
        session_pack_include=session_pack_include,
        session_pack_exclude=session_pack_exclude,
    )
    return unique_paths(resolve_content_subdir_roots(roots, "operations"))


def _normalize_pack_names(pack_names: Iterable[str]) -> set[str]:
    """Normalize a session selector without accepting blank pack identifiers."""

    return {
        normalized for pack_name in pack_names if (normalized := normalize_pack_name(pack_name))
    }


def _configured_root_is_active(
    root: Path,
    *,
    always_on_packs: set[str],
    session_included_packs: set[str],
    session_excluded_packs: set[str],
) -> bool:
    """Keep configured roots active unless the session explicitly excludes them."""

    package_name = infer_pack_package_name(root)
    if package_name in always_on_packs:
        return True
    if package_name in session_excluded_packs:
        return False
    return not session_included_packs or package_name in session_included_packs


def resolve_pack_session_path(
    *,
    pack: str,
    session: str,
    pack_roots: Iterable[Path],
) -> Path:
    """Resolve ``<pack_root>/sessions/<session>.md`` for one installed pack.

    Walks ``pack_roots`` (the output of :func:`active_installed_pack_roots`)
    and returns the first ``sessions/<session>.md`` file inside a pack whose
    inferred package name aliases the requested ``pack`` (so both
    ``lisa-pack-core`` and the bare slug ``core`` match the same install).

    Raises:
        SessionPlanError: if the pack is not present in ``pack_roots`` or the
            pack does not ship a ``sessions/<session>.md`` file.

    The lookup is pure: callers pass the discovered roots so this function
    avoids touching the filesystem for discovery and stays unit-testable
    without an installed pack environment.
    """

    from kernel.session_plan import SessionPlanError

    requested = normalize_pack_name(pack)
    if not requested:
        raise SessionPlanError("from: directive has an empty pack identifier")
    session_name = session.strip()
    if not session_name:
        raise SessionPlanError("from: directive has an empty session identifier")
    session_file_name = (
        session_name if session_name.lower().endswith(".md") else f"{session_name}.md"
    )

    matched_root: Path | None = None
    for root in pack_roots:
        package_name = infer_pack_package_name(root)
        if not package_name:
            continue
        if requested in pack_aliases(package_name, root=root):
            matched_root = root
            break

    if matched_root is None:
        raise SessionPlanError(
            f"from: pack '{pack}' is not installed; cannot resolve session '{session}'"
        )

    session_path = matched_root / "sessions" / session_file_name
    if not session_path.is_file():
        raise SessionPlanError(f"from: pack '{pack}' has no session '{session}' at {session_path}")
    return session_path


# ── Pack-contributed redaction patterns ───────────────────────────────


_PACK_REDACTION_PATTERNS_FILE = "redaction_patterns.yml"


def discover_pack_redaction_patterns(
    data_dir: Path,
    *,
    host_manifest_path: Path | None = None,
    always_on_packs: tuple[str, ...] | None = None,
    session_pack_include: Iterable[str] = (),
    session_pack_exclude: Iterable[str] = (),
) -> list[str]:
    """Scan active pack content roots for contributed redaction patterns.

    Each pack may ship ``content/redaction_patterns.yml`` — a declarative
    YAML file with a ``patterns`` key containing a list of raw regex
    strings. The host merges all discovered patterns from all active
    packs into a single list, deduplicated and order-stable (first
    occurrence wins).

    This is the pack contribution point for thread-read redaction: a
    pack ships its best-practice redaction set alongside its content,
    and the host picks it up automatically without any entry-point
    registration. Disabled packs are excluded via the same
    session/always-on filtering as every other content-root scan.
    """
    roots = authored_content_roots(
        data_dir,
        host_manifest_path=host_manifest_path,
        always_on_packs=always_on_packs,
        session_pack_include=session_pack_include,
        session_pack_exclude=session_pack_exclude,
    )
    patterns: list[str] = []
    seen: set[str] = set()
    for root in roots:
        path = root / _PACK_REDACTION_PATTERNS_FILE
        if not path.is_file():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("redaction: failed to parse %s: %s", path, exc)
            continue
        if not isinstance(raw, dict):
            continue
        pack_patterns = raw.get("patterns")
        if not isinstance(pack_patterns, list):
            continue
        for pattern in pack_patterns:
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            cleaned = pattern.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                patterns.append(cleaned)
    return patterns


__all__ = [
    "active_installed_pack_roots",
    "always_on_pack_names",
    "authored_content_roots",
    "discover_pack_redaction_patterns",
    "host_mcp_manifest_path",
    "infer_pack_package_name",
    "load_disabled_pack_names",
    "load_installed_packs_lock",
    "load_pack_policy_dict",
    "pack_aliases",
    "pack_is_active",
    "resolve_pack_session_path",
    "runtime_enabled_pack_map",
    "runtime_installed_pack_names",
    "runtime_pack_lock_path",
    "save_installed_packs_lock",
    "sdk_content_roots",
    "sdk_mode_roots",
    "sdk_operation_roots",
    "seed_bundled_packs_lock",
    "unique_paths",
]
