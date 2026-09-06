"""Startup auto-install of session-required packs.

When a session plan declares ``resources.packs.required`` and one or more of
those packs are not importable at host boot, this module installs them from
the sources declared in ``resources.packs.auto_install`` (or the configured
git-root template for bare names) before content-root discovery runs.

Design constraints (see ``docs/SESSION_CONTEXT_AUTHORING_GUIDE.md``):

- Auto-install applies **only** to ``required`` packs — never ``include``.
- Git refs **must** pin ``@tag/@branch/@sha``. The host treats pinning as
  mandatory for auto-install regardless of ``LISA_PACK_REQUIRE_GIT_REF``.
- Trust policy env vars (``LISA_PACK_ALLOWED_PREFIXES``,
  ``LISA_PACK_GIT_ALLOW_HOSTS``, ``LISA_PACK_GIT_ROOT``) are honored — a
  session context cannot override them.
- Failure of a required pack is a hard startup error; the runtime does not
  start in a degraded state.

This module is synchronous: it runs inside the host's ``_initialize_runtime``
before the event loop starts. The install core
(:func:`host.commands._packs.run_pack_install_sync`) uses
``subprocess.run`` for the ``uv pip install`` invocation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.content_roots import (
    CONTENT_ROOT_ENTRY_POINT_GROUP,
    discover_installed_content_roots,
    discover_packs_venv_content_roots,
)
from kernel.pack_activation import normalize_pack_name
from kernel.session_plan import PackInstallSpec

if TYPE_CHECKING:
    from host.runtime import HostRuntime

logger = logging.getLogger(__name__)


def _installed_pack_names(runtime: HostRuntime) -> set[str]:
    """Return normalized pack names currently importable by the host.

    Combines two discovery surfaces because external packs live in the
    separate packs venv, whose site-packages are not visible to
    ``importlib.metadata`` in the host process:

    1. Main-venv entry points (``importlib.metadata.entry_points``) —
       bundled packs (core/introspect/memory) and any external pack the
       user also installed into the main venv (e.g. via ``uv sync --extra ext``).
    2. Packs-venv content roots (``discover_packs_venv_content_roots``) —
       external packs installed via ``uv pip install -e`` into the packs
       venv. This is the authoritative home for external packs; without it,
       auto-install re-runs every boot because it never sees them as
       installed.
    """
    names: set[str] = set()
    try:
        discover_installed_content_roots()
        import importlib.metadata

        eps = importlib.metadata.entry_points(group=CONTENT_ROOT_ENTRY_POINT_GROUP)
        for ep in eps:
            dist_name = getattr(ep, "dist", None)
            if dist_name is not None:
                names.add(normalize_pack_name(dist_name.name))
            # Also cover the case where the entry-point name itself is the pack id.
            names.add(normalize_pack_name(ep.name))
    except Exception:  # pragma: no cover - defensive; fail-open discovery
        logger.warning(
            "auto-install: entry-point discovery raised; assuming no packs installed",
            exc_info=True,
        )

    # External packs installed in the packs venv are invisible to
    # importlib.metadata; recover them from the packs venv content-root
    # filesystem scan so auto-install does not re-install packs the user
    # already installed there.
    try:
        from host.content_roots import infer_pack_package_name
        from host.paths import (
            default_packs_venv,
            packs_venv_for,
            packs_venv_site_packages,
        )

        def _venv_resolver(data_dir: Path | None) -> Path:
            if data_dir is not None:
                return packs_venv_for(Path(data_dir))
            return default_packs_venv()

        for root in discover_packs_venv_content_roots(
            runtime.data_dir,
            venv_path_resolver=_venv_resolver,
            site_packages_resolver=packs_venv_site_packages,
        ):
            package_name = infer_pack_package_name(root)
            if package_name:
                names.add(normalize_pack_name(package_name))
    except Exception:  # pragma: no cover - defensive; fail-open discovery
        logger.warning(
            "auto-install: packs venv discovery raised; only main-venv packs counted",
            exc_info=True,
        )
    return names


def _resolve_auto_install_spec(
    pack_name: str,
    auto_install: dict[str, PackInstallSpec],
    *,
    session_source_dir: Path | None = None,
) -> tuple[str, bool, str]:
    """Return ``(spec, editable, source_label)`` for one required pack.

    Prefers an explicit ``auto_install[pack_name].source``. Falls back to the
    bare pack name (expanded by :func:`run_pack_install_sync` via
    ``LISA_PACK_GIT_ROOT``). Raises ``ValueError`` when a git ref is unpinned
    — auto-install treats ref-pinning as mandatory regardless of
    ``LISA_PACK_REQUIRE_GIT_REF``.

    When the source is a relative local path, it is resolved against
    ``session_source_dir`` (the session-context file's directory) so a
    session can reference packs checked out alongside it
    (e.g. ``source: ../lisa-packs/brave``). Absolute paths and
    non-local sources (git+, bare names) are passed through unchanged.
    """
    entry = auto_install.get(pack_name)
    if entry is not None:
        source = getattr(entry, "source", "") or ""
        editable = bool(getattr(entry, "editable", True))
        if source:
            if source.lower().startswith("git+") and "@" not in source:
                raise ValueError(
                    f"auto-install: pack '{pack_name}' has an unpinned git source "
                    f"{source!r}; pin @tag/@branch/@sha is mandatory for auto-install."
                )
            resolved_source = _resolve_local_source(source, session_source_dir)
            return resolved_source, editable, "auto_install.source"
    # Fall back to the bare name; run_pack_install_sync expands it via
    # LISA_PACK_GIT_ROOT (default template pins @main). The default template
    # already includes @main, so bare-name fallback satisfies the pin rule.
    return pack_name, False, "git-root-fallback"


def _resolve_local_source(source: str, session_source_dir: Path | None) -> str:
    """Resolve a relative local path against the session source directory.

    Non-local sources (``git+`` refs, bare package names, absolute paths) are
    returned unchanged. A relative path that looks like a local directory is
    resolved against ``session_source_dir`` and returned as an absolute path
    string. When ``session_source_dir`` is ``None``, the source is returned
    unchanged (the host's CWD applies, matching the REPL ``/pack install``
    behavior).
    """
    raw = source.strip().strip('"')
    if not raw:
        return source
    lowered = raw.lower()
    if (
        lowered.startswith("git+")
        or lowered.startswith("https://")
        or lowered.startswith("http://")
    ):
        return source
    # Bare package names (no path separators, no version operators) are left
    # for _normalize_pack_install_spec to expand via LISA_PACK_GIT_ROOT.
    if not _looks_like_path(raw):
        return source
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if session_source_dir is None:
        return source
    return str((session_source_dir / candidate).resolve())


def _looks_like_path(raw: str) -> bool:
    """Heuristic: does this string look like a filesystem path rather than a bare package name?"""
    if not raw:
        return False
    # Path separators (Windows or POSIX) or a leading ./ or ../ → path.
    if raw.startswith(("./", "../", ".\\", "..\\")):
        return True
    if "/" in raw or "\\" in raw:
        return True
    return False


def auto_install_required_packs(runtime: HostRuntime) -> list[str]:
    """Install any missing ``required`` packs before content-root discovery.

    Returns the list of pack names installed (empty when all required packs
    are already importable). On failure of any required pack, raises
    ``RuntimeError`` with the last failure detail — the host should surface
    this as a startup error and exit rather than starting degraded.

    The caller owns restart semantics: if this returns a non-empty list,
    ``importlib.metadata`` may not see the new entry points in-process, so
    the caller should call ``runtime.request_restart(...)`` and let the host
    relaunch. On the second boot pass the packs are present, this function is
    a no-op, and discovery proceeds normally.
    """
    plan_packs = runtime.session_plan.manifest.resources.packs
    required = tuple(normalize_pack_name(name) for name in plan_packs.required)
    if not required:
        return []

    installed = _installed_pack_names(runtime)
    missing = [name for name in required if name not in installed]
    if not missing:
        return []

    from host.commands._packs import run_pack_install_sync

    session_source_dir: Path | None = None
    if runtime.session_plan.source_dir:
        session_source_dir = Path(runtime.session_plan.source_dir)

    auto_install_map = plan_packs.auto_install
    installed_now: list[str] = []
    last_error = ""
    for pack_name in missing:
        try:
            spec, editable, source_label = _resolve_auto_install_spec(
                pack_name, auto_install_map, session_source_dir=session_source_dir
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        logger.info(
            "auto-install: required pack %r missing; installing from %s (%s, editable=%s)",
            pack_name,
            spec,
            source_label,
            editable,
        )
        ok, detail = run_pack_install_sync(runtime, spec, request_restart=False, editable=editable)
        if ok:
            installed_now.append(pack_name)
        else:
            last_error = detail
            logger.error("auto-install: failed to install %r:\n%s", pack_name, detail)

    if installed_now:
        logger.info(
            "auto-install: installed %d pack(s): %s",
            len(installed_now),
            ", ".join(installed_now),
        )
        return installed_now

    # Nothing installed and at least one missing required pack failed.
    raise RuntimeError(
        f"auto-install: required pack(s) missing and install failed. Last error:\n{last_error}"
    )


__all__ = ["auto_install_required_packs"]
