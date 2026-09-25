"""Transport-neutral session context resolution and compilation.

This module resolves a session-context reference into a strict v2
:class:`~kernel.session_plan.SessionPlan` via
:func:`compile_selected_session_plan`. The v2 compiler is the single
session-identity path — there is no legacy ``user_contexts/`` resolution,
no old-style ``extract_session_context_controls``, and no dynamic
user-context body loading. All session controls (surface, resources,
providers, routing, context lanes/sources, history) come from the compiled
``SessionPlan.manifest``.

Resolution order (see :func:`resolve_session_context_source`):

1. Explicit local ``.md`` file path (CLI arg).
2. ``LISA_SESSION_CONTEXT_PATH`` env var (must point at an existing file).
3. ``<data_dir>/config/session_context.md``.
4. Bundled core default session.

There is no remote fetch, registry, slug resolution, or disk cache: the
session plan is a local authored artifact, resolved from disk on every
startup.

Dependency direction
--------------------
``session_context.py`` ← ``runtime.py`` / ``config.py``  (callers
import from session_context; session_context never imports transport code)

Public surface
--------------
- :func:`compile_selected_session_plan`
- :func:`resolve_session_context_source`
- :func:`bundled_default_session_path`
- :func:`extract_front_matter_thread_name`
- :class:`SessionContextSource`
- :class:`ReplayControls` (transitional — used by context_assembly)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kernel.ipsar_fsm import parse_front_matter
from kernel.session_plan import SessionPlan, compile_session_plan

from host.session_from_resolver import resolve_session_from_directives

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionContextSource:
    """Resolved strict session source, retaining its authored identity."""

    source_uri: str
    cached_path: Path
    text: str


def bundled_default_session_path() -> Path:
    """Return the mandatory bundled v2 session used when none is selected."""

    # Resolve through the import system so the path works in editable installs
    # (src layout) and installed wheels alike. The core pack's content root is
    # discovered via its entry point, not a hardcoded relative path.
    try:
        import importlib.metadata

        eps = importlib.metadata.entry_points(group="lisa.extensions.content_roots")
        for ep in eps:
            if ep.name == "core":
                root_loader = ep.load()
                candidate = Path(root_loader()) / "sessions" / "default.md"
                if candidate.is_file():
                    return candidate
    except Exception:
        pass  # fall through to the legacy path below
    return (
        Path(__file__).resolve().parents[1] / "sdk" / "packs" / "core" / "sessions" / "default.md"
    )


LISA_SESSION_CONTEXT_PATH_ENV = "LISA_SESSION_CONTEXT_PATH"


def resolve_session_context_source(
    *,
    session_context: str,
    data_dir: Path,
) -> SessionContextSource:
    """Resolve one strict session source — local file only.

    Resolution order:

    1. Explicit ``.md`` file path (CLI arg). Must exist and be a Markdown
       file; a directory or non-.md path raises ``ValueError``.
    2. ``LISA_SESSION_CONTEXT_PATH`` (when set). An explicit env override
       whose file must exist — a missing file raises instead of silently
       falling through, so a configured-but-broken environment is loud.
    3. ``<data_dir>/config/session_context.md`` when authored.
    4. The bundled core default session.
    """
    requested = session_context.strip()
    if requested:
        explicit_path = Path(requested).expanduser()
        if explicit_path.is_dir() or (
            explicit_path.exists() and explicit_path.suffix.lower() != ".md"
        ):
            raise ValueError("Session context path must be a Markdown file.")
        if explicit_path.is_file():
            return _read_session_source(
                source_uri=str(explicit_path.resolve()),
                path=explicit_path,
            )
        raise ValueError(f"Session context file not found: {explicit_path}")

    env_path = os.getenv(LISA_SESSION_CONTEXT_PATH_ENV, "").strip()
    if env_path:
        env_candidate = Path(env_path).expanduser()
        if env_candidate.is_file():
            return _read_session_source(
                source_uri=str(env_candidate.resolve()),
                path=env_candidate,
            )
        raise FileNotFoundError(
            f"{LISA_SESSION_CONTEXT_PATH_ENV}='{env_path}' is not an existing file."
        )

    from sdk.layout import CONFIG_DIR

    candidate = (data_dir / CONFIG_DIR / "session_context.md").resolve()
    if candidate.is_file():
        return _read_session_source(source_uri=str(candidate), path=candidate)

    default_path = bundled_default_session_path()
    if not default_path.is_file():
        raise FileNotFoundError(f"Bundled default session is missing: {default_path}")
    return _read_session_source(source_uri=str(default_path.resolve()), path=default_path)


def compile_selected_session_plan(
    *,
    session_context: str,
    data_dir: Path,
) -> SessionPlan:
    """Resolve and strictly compile the one session plan for a host runtime.

    Host-layer ``from:`` and ``body:`` directives (see
    :mod:`host.session_from_resolver`) are resolved before the strict
    kernel compiler runs. Pack discovery is only paid when a directive is
    actually present in the resolved source — otherwise this is a straight
    passthrough to :func:`compile_session_plan`.
    """

    source = resolve_session_context_source(
        session_context=session_context,
        data_dir=data_dir,
    )
    resolved_text = resolve_session_from_directives(
        text=source.text,
        source_uri=source.source_uri,
        pack_session_lookup=_make_pack_session_lookup(data_dir=data_dir),
    )
    plan = compile_session_plan(text=resolved_text, source_uri=source.source_uri)
    return plan.model_copy(update={"source_dir": str(source.cached_path.parent.resolve())})


def _make_pack_session_lookup(*, data_dir: Path) -> Callable[..., Path]:
    """Build a ``pack_session_lookup`` closure for ``from:`` resolution.

    Pack discovery (:func:`host.content_roots.active_installed_pack_roots`)
    runs lazily — only when a ``from:`` directive actually triggers a lookup
    — so sessions without directives pay zero discovery cost.
    """

    from host.content_roots import (
        active_installed_pack_roots,
        resolve_pack_session_path,
    )

    def _lookup(*, pack: str, session: str) -> Path:
        pack_roots = active_installed_pack_roots(data_dir)
        return resolve_pack_session_path(pack=pack, session=session, pack_roots=pack_roots)

    return _lookup


def _read_session_source(*, source_uri: str, path: Path) -> SessionContextSource:
    """Read a resolved Markdown session source with a useful startup error."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read session context {source_uri}: {exc}") from exc
    return SessionContextSource(
        source_uri=source_uri,
        cached_path=path.resolve(),
        text=text,
    )


# ── Transitional: ReplayControls ────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ReplayControls:
    """Replay-policy controls used by context_assembly.

    Transitional: the v2 ``ContextHistoryPlan`` in
    ``SessionPlan.manifest.context.history`` is the authoritative source
    for history settings. This dataclass remains as the type
    annotation in ``assemble_dynamic_user_context`` until that function
    is refactored to accept ``ContextHistoryPlan`` directly.

    The lossy digest knobs (``include_prior_conversation_digest``,
    ``digest_max_lines``, ``digest_summary_max_chars``) were removed in the
    memory × thread realignment — the thread manifest is the only
    prior-history representation.
    """

    trigger_mode: str | None = None
    include_thread_manifest: bool | None = None
    include_recent_conversation: bool | None = None
    include_diagnostic_reflection: bool | None = None
    history_max_tokens: int | None = None
    reflection_max_tokens: int | None = None
    min_query_chars: int | None = None


# ── Helpers used by the v2 resolution path ──────────────────────────


def extract_front_matter_thread_name(path: Path) -> str | None:
    """Return the ``name`` value from YAML front matter if present.

    Delegates to the canonical kernel :func:`parse_front_matter` so the
    parsing logic stays in one place. Returns ``None`` when the file
    cannot be read or no ``name`` key is present.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    front_matter, _ = parse_front_matter(text)
    name = front_matter.get("name")
    if isinstance(name, str):
        return name.strip() or None
    return None
