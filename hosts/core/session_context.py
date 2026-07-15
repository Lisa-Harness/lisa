"""Transport-neutral session context resolution and compilation.

This module resolves a session-context reference (path, slug, or auto) into
a strict v2 :class:`~kernel.session_plan.SessionPlan` via
:func:`compile_selected_session_plan`. The v2 compiler is the single
session-identity path — there is no legacy ``user_contexts/`` resolution,
no old-style ``extract_session_context_controls``, and no dynamic
user-context body loading. All session controls (surface, resources,
providers, routing, context lanes/sources, history) come from the compiled
``SessionPlan.manifest``.

Resolution order (see :func:`resolve_session_context_source`):

1. Explicit local file, remote URL, or registry slug.
2. Auto-discover ``session_context.md`` in cwd or the data directory.
3. Bundled core default session.

Dependency direction
--------------------
``session_context.py`` ← ``runtime.py`` / ``config.py``  (callers
import from session_context; session_context never imports transport code)

Public surface
--------------
- :func:`compile_selected_session_plan`
- :func:`resolve_session_context_source`
- :func:`bundled_default_session_path`
- :func:`session_context_registry_url`
- :func:`extract_front_matter_thread_name`
- :class:`SessionContextSource`
- :class:`ReplayControls` (transitional — used by context_assembly)
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from kernel.ipsar_fsm import parse_front_matter
from kernel.session_plan import SessionPlan, compile_session_plan

from hosts.core.lisa_config import context_slug_to_thread_name
from hosts.core.session_context_fetch import fetch_session_context_url

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionContextSource:
    """Resolved strict session source, retaining its authored identity."""

    source_uri: str
    cached_path: Path
    text: str


def bundled_default_session_path() -> Path:
    """Return the mandatory bundled v2 session used when none is selected."""

    return (
        Path(__file__).resolve().parents[2]
        / "sdk"
        / "packs"
        / "core"
        / "sessions"
        / "default.md"
    )


def resolve_session_context_source(
    *,
    session_context: str,
    data_dir: Path,
    force_refresh: bool = False,
) -> SessionContextSource:
    """Resolve one strict session source without legacy registry or directory rules.

    Explicit local files, remote URLs, and registry slugs take precedence.
    With no explicit reference, an authored ``session_context.md`` in the
    working directory or data directory wins over the bundled core default.
    """

    requested = session_context.strip()
    if requested:
        return _resolve_explicit_session_context_source(
            requested,
            data_dir=data_dir,
            force_refresh=force_refresh,
        )

    checked_paths: set[Path] = set()
    for root in (Path.cwd(), data_dir):
        candidate = (root / "session_context.md").resolve()
        if candidate in checked_paths:
            continue
        checked_paths.add(candidate)
        if candidate.is_file():
            return _read_session_source(source_uri=str(candidate), path=candidate)

    default_path = bundled_default_session_path()
    if not default_path.is_file():
        raise FileNotFoundError(f"Bundled default session is missing: {default_path}")
    return _read_session_source(source_uri=str(default_path.resolve()), path=default_path)


def _resolve_explicit_session_context_source(
    requested: str,
    *,
    data_dir: Path,
    force_refresh: bool,
) -> SessionContextSource:
    """Resolve an explicitly requested local file, URL, artifact, or registry slug."""

    remote = urllib.parse.urlparse(requested)
    if remote.scheme in {"http", "https"}:
        cached_path = fetch_session_context_url(
            requested,
            data_dir=data_dir,
            force_refresh=force_refresh,
        )
        return _read_session_source(source_uri=requested, path=cached_path)

    explicit_path = Path(requested).expanduser()
    if explicit_path.exists():
        if explicit_path.is_dir() or explicit_path.suffix.lower() != ".md":
            raise ValueError("Session context file path must be a Markdown file.")
        return _read_session_source(
            source_uri=str(explicit_path.resolve()),
            path=explicit_path,
        )

    artifact_name = (
        requested if requested.lower().endswith(".md") else f"{requested}.md"
    )
    artifact_path = data_dir / artifact_name
    if artifact_path.is_file():
        return _read_session_source(
            source_uri=str(artifact_path.resolve()),
            path=artifact_path,
        )

    registry_slug = requested[:-3] if requested.lower().endswith(".md") else requested
    registry_uri = f"{session_context_registry_url()}/{registry_slug}.md"
    try:
        cached_path = fetch_session_context_url(
            registry_uri,
            data_dir=data_dir,
            force_refresh=force_refresh,
        )
    except ValueError as exc:
        raise ValueError(f"Session context '{requested}' was not found.") from exc
    return _read_session_source(source_uri=registry_uri, path=cached_path)


def compile_selected_session_plan(
    *,
    session_context: str,
    data_dir: Path,
    force_refresh: bool = False,
) -> SessionPlan:
    """Resolve and strictly compile the one session plan for a host runtime."""

    source = resolve_session_context_source(
        session_context=session_context,
        data_dir=data_dir,
        force_refresh=force_refresh,
    )
    return compile_session_plan(text=source.text, source_uri=source.source_uri)


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
    for history/digest settings. This dataclass remains as the type
    annotation in ``assemble_dynamic_user_context`` until that function
    is refactored to accept ``ContextHistoryPlan`` directly.
    """

    trigger_mode: str | None = None
    include_prior_conversation_digest: bool | None = None
    include_recent_conversation: bool | None = None
    include_diagnostic_reflection: bool | None = None
    history_max_tokens: int | None = None
    reflection_max_tokens: int | None = None
    min_query_chars: int | None = None
    digest_max_lines: int | None = None
    digest_summary_max_chars: int | None = None


# ── Helpers used by the v2 resolution path ──────────────────────────


LISA_SESSION_CONTEXT_REGISTRY_URL_ENV = "LISA_SESSION_CONTEXT_REGISTRY_URL"


def session_context_registry_url() -> str:
    """Return the registry URL for remote authored session contexts."""
    configured = os.getenv(LISA_SESSION_CONTEXT_REGISTRY_URL_ENV, "").strip()
    if configured:
        return configured.rstrip("/")
    return "https://raw.githubusercontent.com/lisa-framework/sessions/main"


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