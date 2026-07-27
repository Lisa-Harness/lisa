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
(:func:`hosts.core.commands._packs.run_pack_install_sync`) uses
``subprocess.run`` for the ``uv pip install`` invocation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kernel.content_roots import CONTENT_ROOT_ENTRY_POINT_GROUP, discover_installed_content_roots
from kernel.pack_activation import normalize_pack_name
from kernel.session_plan import PackInstallSpec

if TYPE_CHECKING:
    from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)


def _installed_pack_names() -> set[str]:
    """Return normalized pack names currently importable via entry points."""
    try:
        discover_installed_content_roots()
    except Exception:  # pragma: no cover - defensive; fail-open discovery
        logger.warning(
            "auto-install: entry-point discovery raised; assuming no packs installed",
            exc_info=True,
        )
        return set()
    # Re-read the entry points to map dist names → normalized pack names.
    import importlib.metadata

    names: set[str] = set()
    try:
        eps = importlib.metadata.entry_points(group=CONTENT_ROOT_ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - defensive
        return set()
    for ep in eps:
        dist_name = getattr(ep, "dist", None)
        if dist_name is not None:
            names.add(normalize_pack_name(dist_name.name))
        # Also cover the case where the entry-point name itself is the pack id.
        names.add(normalize_pack_name(ep.name))
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
    (e.g. ``source: ../lisa-packs/brave_search``). Absolute paths and
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
    if lowered.startswith("git+") or lowered.startswith("https://") or lowered.startswith("http://"):
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

    installed = _installed_pack_names()
    missing = [name for name in required if name not in installed]
    if not missing:
        return []

    from hosts.core.commands._packs import run_pack_install_sync

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
        ok, detail = run_pack_install_sync(
            runtime, spec, request_restart=False, editable=editable
        )
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