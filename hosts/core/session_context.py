"""Transport-neutral session context resolution.

This module resolves a session-context reference (slug, path, or auto) into
a ``(thread_name, source_path, is_active)`` tuple and loads the active
session context text, honouring ``applies_to`` front-matter filtering by
mode.

Resolution order (see :func:`resolve_session_context`):

1. If empty: auto-discover ``session_context.md`` or ``session_context/``
   in the current directory and data directory.
2. Literal path: if it points to a file or directory.
3. Data-directory slug: if ``{data_dir}/{slug}.md`` or
   ``{data_dir}/{slug}/`` exists.
4. ``user_contexts/`` slug: if ``user_contexts/{slug}.md`` exists in the
   authored-content registry (loaded via ``content_loader``).

All functions are transport-neutral: they take explicit ``data_dir``,
``user_slug``, ``session_context_ref``, ``current_mode_id``,
``session_context_active``, and ``content_loader`` parameters rather than
reading global runtime state. A transport host wires its own
``HostConfig`` / ``HostRuntime`` values into these functions at startup.

Dependency direction
--------------------
``session_context.py`` ← ``main.py`` / ``handle_message.py``  (callers
import from session_context; session_context never imports main or any
transport code)

Public surface
--------------
- :func:`resolve_session_context`
- :func:`resolve_session_context_auto`
- :func:`resolve_session_context_literal_path`
- :func:`resolve_session_context_artifacts`
- :func:`resolve_session_context_artifacts_session_context`
- :func:`resolve_session_context_user_contexts`
- :func:`resolve_session_context_identity`
- :func:`extract_session_context_controls`
- :func:`extract_front_matter_thread_name`
- :class:`SessionContextControls`
- :class:`SurfaceControls`
- :class:`PackPreferences`
- :class:`PolicyControls`
- :func:`load_selected_session_context_text`
- :func:`load_active_session_context_text`
- :func:`load_session_context_directory`
- :func:`applies_to_mode`
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# ── Session-context front-matter controls ───────────────────────────


@dataclass(frozen=True, slots=True)
class SurfaceControls:
    """Privacy surface identity from session-context front matter.

    All fields are optional — a missing field means "no override" so the
    host's CLI/env/default value is used.
    """

    surface_type: str | None = None
    surface_id: str | None = None
    memory_scope: str | None = None


@dataclass(frozen=True, slots=True)
class PackPreferences:
    """Per-pack preferences from the ``packs`` front-matter map.

    Each key is a qualified pack name; each value is a dict of
    preferences. Currently recognised preference keys:
    - ``enabled`` (bool): intent for lock-state; startup-only.
    """

    preferences: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyControls:
    """Global pack-policy controls from the ``policy`` front-matter block.

    Currently recognised keys:
    - ``always_on_packs`` (list[str]): packs that cannot be disabled.
    - ``exclude_servers`` (list[str]): pack MCP server IDs to suppress.
    - ``preferred_capability_providers`` (dict[str, str]): capability → pack.
    """

    always_on_packs: tuple[str, ...] = ()
    exclude_servers: tuple[str, ...] = ()
    preferred_capability_providers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReplayControls:
    """Replay-policy controls from the ``replay`` front-matter block.

    All fields are optional and default to "no override" so host/runtime
    defaults remain authoritative when a key is absent.
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


from kernel.models import ContextRoutingConfig

@dataclass(frozen=True, slots=True)
class SessionContextControls:
    """All runtime controls extracted from session-context front matter.

    Every field defaults to "no override" so the host's existing
    CLI/env/default resolution is used when the front matter doesn't
    declare a key. This dataclass is transport-neutral; the host wires
    it into ``HostConfig`` / ``HostRuntime`` at startup.
    """

    name: str | None = None
    surface: SurfaceControls = field(default_factory=SurfaceControls)
    start_mode: str | None = None
    packs: PackPreferences = field(default_factory=PackPreferences)
    policy: PolicyControls = field(default_factory=PolicyControls)
    replay: ReplayControls = field(default_factory=ReplayControls)
    context_routing: ContextRoutingConfig | None = None


LISA_SESSION_CONTEXT_REGISTRY_URL_ENV = "LISA_SESSION_CONTEXT_REGISTRY_URL"

def session_context_registry_url() -> str:
    """Return the registry URL for remote authored session contexts."""
    configured = os.getenv(LISA_SESSION_CONTEXT_REGISTRY_URL_ENV, "").strip()
    if configured:
        return configured.rstrip("/")
    return "https://raw.githubusercontent.com/lisa-framework/sessions/main"


def resolve_session_context_remote_url(
    s: str,
    *,
    data_dir: Path,
    force_refresh: bool = False,
) -> tuple[str, str, bool] | None:
    """Resolve a remote URL session context.

    Returns ``(thread_name, cached_path, True)`` or raises ValueError if fetch fails.
    Returns ``None`` if ``s`` is not a URL.
    """
    parsed = urllib.parse.urlparse(s)
    if parsed.scheme not in {"http", "https"}:
        return None

    cached_path = fetch_session_context_url(s, data_dir=data_dir, force_refresh=force_refresh)
    # Default name to stem of URL if no front-matter name
    default_name = Path(parsed.path).stem
    if not default_name or default_name == ".":
        default_name = "session_context"
    tn = extract_front_matter_thread_name(cached_path) or default_name
    return tn, str(cached_path.absolute()), True


def resolve_session_context_remote_registry(
    s: str,
    *,
    data_dir: Path,
    force_refresh: bool = False,
) -> tuple[str, str, bool] | None:
    """Resolve a slug from the remote session registry.

    Returns ``(thread_name, cached_path, True)`` or None if missing/error.
    """
    context_name = s[:-3] if s.lower().endswith(".md") else s
    url = f"{session_context_registry_url()}/{context_name}.md"
    try:
        cached_path = fetch_session_context_url(url, data_dir=data_dir, force_refresh=force_refresh)
    except ValueError as e:
        logger.debug("session-context: remote registry fetch failed for %s: %s", url, e)
        return None

    tn = extract_front_matter_thread_name(cached_path) or context_name
    return tn, str(cached_path.absolute()), True


def resolve_session_context(
    *,
    session_context: str,
    data_dir: Path,
    content_loader: Callable[[str], str],
    force_refresh: bool = False,
) -> tuple[str, str, bool]:
    """Resolve the session context source.

    Returns ``(thread_name, source_path, is_active)``. ``is_active`` is
    ``True`` when the resolved source is a filesystem path (auto-discovered
    or literal) and ``False`` when it is a registry-relative path
    (``user_contexts/…``).

    Resolution order:
    1. If empty: auto-discover ``session_context.md`` or
       ``session_context/`` in cwd and the data directory.
    2. Literal path: if it is a file or directory.
    3. Data-directory slug: if ``{data_dir}/{slug}.md`` or
       ``{data_dir}/{slug}/`` exists.
    4. ``user_contexts/`` slug: if ``user_contexts/{slug}.md`` exists.
    """
    s = session_context.strip()

    if not s:
        return resolve_session_context_auto(data_dir=data_dir)
    if (result := resolve_session_context_remote_url(s, data_dir=data_dir, force_refresh=force_refresh)) is not None:
        return result
    if (result := resolve_session_context_literal_path(s)) is not None:
        return result
    if (result := resolve_session_context_artifacts(s, data_dir=data_dir)) is not None:
        return result
    if (
        result := resolve_session_context_user_contexts(s, content_loader=content_loader)
    ) is not None:
        return result
    if (result := resolve_session_context_remote_registry(s, data_dir=data_dir, force_refresh=force_refresh)) is not None:
        return result
    raise ValueError(f"Session context '{s}' was not found.")


def resolve_session_context_auto(
    *,
    data_dir: Path,
) -> tuple[str, str, bool]:
    """Auto-discover ``session_context.md`` or ``session_context/`` in cwd and data dir."""
    for base in (Path("."), data_dir):
        md_path = base / "session_context.md"
        if md_path.is_file():
            tn = extract_front_matter_thread_name(md_path) or "session_context"
            return tn, str(md_path.absolute()), True
        dir_path = base / "session_context"
        if dir_path.is_dir():
            return "session_context", str(dir_path.absolute()), True
    return "", "", False


def resolve_session_context_literal_path(
    s: str,
) -> tuple[str, str, bool] | None:
    """Resolve a literal file or directory path.

    Returns ``None`` when the path does not exist. Raises ``ValueError`` if
    the path is a file that does not end in ``.md``.
    """
    p = Path(s).expanduser()
    if not p.exists():
        return None
    if p.is_dir():
        return "session_context", str(p.absolute()), True
    if p.suffix.lower() != ".md":
        raise ValueError("Session context file path must end with '.md'.")
    tn = extract_front_matter_thread_name(p) or context_slug_to_thread_name(s)
    return tn, str(p.absolute()), True


def resolve_session_context_artifacts(
    s: str,
    *,
    data_dir: Path,
) -> tuple[str, str, bool] | None:
    """Resolve a session context relative to the data directory.

    Checks ``{data_dir}/{s}.md`` and ``{data_dir}/{s}/``. When the slug is
    ``session_context``, delegates to
    :func:`resolve_session_context_artifacts_session_context`.
    """
    md_path = data_dir / f"{s}.md"
    if md_path.is_file():
        tn = extract_front_matter_thread_name(md_path) or context_slug_to_thread_name(s)
        return tn, str(md_path.absolute()), True
    dir_path = data_dir / s
    if dir_path.is_dir():
        return "session_context", str(dir_path.absolute()), True
    if s == "session_context":
        return resolve_session_context_artifacts_session_context(data_dir)
    return None


def resolve_session_context_artifacts_session_context(
    data_dir: Path,
) -> tuple[str, str, bool] | None:
    """Special-case resolution for the ``session_context`` slug in the data dir."""
    md_path = data_dir / "session_context.md"
    if md_path.is_file():
        tn = extract_front_matter_thread_name(md_path) or "session_context"
        return tn, str(md_path.absolute()), True
    dir_path = data_dir / "session_context"
    if dir_path.is_dir():
        return "session_context", str(dir_path.absolute()), True
    return None


def resolve_session_context_user_contexts(
    s: str,
    *,
    content_loader: Callable[[str], str],
) -> tuple[str, str, bool] | None:
    """Resolve a ``user_contexts/`` slug (legacy registry-relative path).

    Returns ``None`` when the authored content is missing or empty. The
    returned ``is_active`` is ``False`` because this is a registry path,
    not a filesystem path.
    """
    context_name = s[:-3] if s.lower().endswith(".md") else s
    relative_path = f"user_contexts/{context_name}.md"
    try:
        content = content_loader(relative_path)
    except FileNotFoundError:
        return None
    if not str(content).strip():
        return None
    return context_name, relative_path, False


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


def resolve_session_context_identity(
    *,
    session_context: str,
    data_dir: Path,
    force_refresh: bool = False,
) -> tuple[str, str]:
    """Resolve a session-context reference into ``(ref, name)`` for identity.

    Unlike :func:`resolve_session_context`, this never raises and does not
    require a ``content_loader``. Used by :meth:`HostConfig.from_env` to
    set ``user_slug`` (thread name + status display) from the session
    context's ``name`` front matter.

    Resolution:
    1. Empty: auto-discover ``session_context.md`` or ``session_context/``
       in cwd and data dir. Returns ``("", "")`` if nothing found.
    2. Remote URL (http/https): ``name`` from front matter, or URL stem.
    3. Literal ``.md`` file: ``name`` from front matter, or stem.
    4. Literal directory: name is ``"session_context"``.
    5. Data-dir slug (``{data_dir}/{slug}.md``): ``name`` from front matter,
       or stem.
    6. Data-dir directory (``{data_dir}/{slug}/``): name is
       ``"session_context"``.
    7. Remote registry slug: ``name`` from front matter, or slug stem.
    8. Otherwise (registry slug or unknown): name is the slug stem.
    """
    s = session_context.strip()

    if not s:
        name, ref, _ = resolve_session_context_auto(data_dir=data_dir)
        return ref, name

    parsed = urllib.parse.urlparse(s)
    if parsed.scheme in {"http", "https"}:
        try:
            cached_path = fetch_session_context_url(s, data_dir=data_dir, force_refresh=force_refresh)
            default_name = Path(parsed.path).stem
            if not default_name or default_name == ".":
                default_name = "session_context"
            name = extract_front_matter_thread_name(cached_path) or default_name
            return str(cached_path.absolute()), name
        except ValueError:
            return s, context_slug_to_thread_name(s)

    p = Path(s).expanduser()
    if p.is_file():
        if p.suffix.lower() == ".md":
            name = extract_front_matter_thread_name(p) or context_slug_to_thread_name(s)
            return str(p.absolute()), name
        return s, context_slug_to_thread_name(s)
    if p.is_dir():
        return str(p.absolute()), "session_context"

    md_path = data_dir / f"{s}.md"
    if md_path.is_file():
        name = extract_front_matter_thread_name(md_path) or context_slug_to_thread_name(s)
        return str(md_path.absolute()), name
    dir_path = data_dir / s
    if dir_path.is_dir():
        return str(dir_path.absolute()), "session_context"

    try:
        context_name = s[:-3] if s.lower().endswith(".md") else s
        url = f"{session_context_registry_url()}/{context_name}.md"
        cached_path = fetch_session_context_url(url, data_dir=data_dir, force_refresh=force_refresh)
        name = extract_front_matter_thread_name(cached_path) or context_name
        return str(cached_path.absolute()), name
    except ValueError:
        pass

    # Registry slug or unknown — use stem-based name.
    return s, context_slug_to_thread_name(s)


def extract_session_context_controls(path: Path) -> SessionContextControls:
    """Extract all runtime controls from a session-context file's front matter.

    Returns an empty :class:`SessionContextControls` (all fields "no
    override") when the file cannot be read or has no front matter.
    Unknown keys are ignored with debug logging. Invalid values
    degrade gracefully — the offending key is skipped rather than
    raising.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # pylint: disable=broad-except
        return SessionContextControls()
    front_matter, _ = parse_front_matter(text)
    if not isinstance(front_matter, dict) or not front_matter:
        return SessionContextControls()

    name = front_matter.get("name")
    name_str = name.strip() if isinstance(name, str) else None

    # ── surface block ──
    surface = SurfaceControls()
    surface_raw = front_matter.get("surface")
    if isinstance(surface_raw, dict):
        surface = SurfaceControls(
            surface_type=_str_or_none(surface_raw.get("surface_type"))
            or _str_or_none(surface_raw.get("type")),
            surface_id=_str_or_none(surface_raw.get("surface_id"))
            or _str_or_none(surface_raw.get("id")),
            memory_scope=_str_or_none(surface_raw.get("memory_scope")),
        )
    else:
        # Allow flat top-level keys as a convenience.
        st = _str_or_none(front_matter.get("surface_type"))
        sid = _str_or_none(front_matter.get("surface_id"))
        ms = _str_or_none(front_matter.get("memory_scope"))
        if st or sid or ms:
            surface = SurfaceControls(
                surface_type=st, surface_id=sid, memory_scope=ms
            )

    # ── start_mode ──
    start_mode = _str_or_none(front_matter.get("start_mode"))

    # ── packs map ──
    packs = PackPreferences()
    packs_raw = front_matter.get("packs")
    if isinstance(packs_raw, dict):
        normalized: dict[str, dict[str, Any]] = {}
        for pack_name, prefs in packs_raw.items():
            if not isinstance(pack_name, str):
                continue
            if not isinstance(prefs, dict):
                logger.debug(
                    "session-context: packs.%s ignored (not a dict)",
                    pack_name,
                )
                continue
            normalized[pack_name] = prefs
        if normalized:
            packs = PackPreferences(preferences=normalized)

    # ── policy block ──
    policy = PolicyControls()
    policy_raw = front_matter.get("policy")
    if isinstance(policy_raw, dict):
        always_on = _str_list(policy_raw.get("always_on_packs"))
        exclude = _str_list(policy_raw.get("exclude_servers"))
        cap_prefs_raw = policy_raw.get("preferred_capability_providers")
        cap_prefs: dict[str, str] = {}
        if isinstance(cap_prefs_raw, dict):
            for cap, pack in cap_prefs_raw.items():
                cap_s = cap.strip() if isinstance(cap, str) else ""
                pack_s = pack.strip() if isinstance(pack, str) else ""
                if cap_s and pack_s:
                    cap_prefs[cap_s] = pack_s
        policy = PolicyControls(
            always_on_packs=tuple(always_on),
            exclude_servers=tuple(exclude),
            preferred_capability_providers=cap_prefs,
        )

    # ── replay block ──
    replay = ReplayControls()
    replay_raw = front_matter.get("replay")
    if isinstance(replay_raw, dict):
        replay = ReplayControls(
            trigger_mode=_normalize_replay_trigger_mode(replay_raw.get("trigger_mode"))
            or _normalize_replay_trigger_mode(replay_raw.get("trigger")),
            include_prior_conversation_digest=_bool_or_none(
                replay_raw.get("include_prior_conversation_digest")
            ),
            include_recent_conversation=_bool_or_none(
                replay_raw.get("include_recent_conversation")
            ),
            include_diagnostic_reflection=_bool_or_none(
                replay_raw.get("include_diagnostic_reflection")
            ),
            history_max_tokens=_int_or_none(replay_raw.get("history_max_tokens")),
            reflection_max_tokens=_int_or_none(replay_raw.get("reflection_max_tokens")),
            min_query_chars=_int_or_none(replay_raw.get("min_query_chars")),
            digest_max_lines=_int_or_none(replay_raw.get("digest_max_lines")),
            digest_summary_max_chars=_int_or_none(replay_raw.get("digest_summary_max_chars")),
        )

    # ── context_routing block ──
    context_routing: ContextRoutingConfig | None = None
    routing_raw = front_matter.get("context_routing")
    if isinstance(routing_raw, dict):
        try:
            context_routing = ContextRoutingConfig.model_validate(routing_raw)
        except Exception as e:
            logger.warning("Failed to validate context_routing block: %s", e)

    return SessionContextControls(
        name=name_str,
        surface=surface,
        start_mode=start_mode,
        packs=packs,
        policy=policy,
        replay=replay,
        context_routing=context_routing,
    )


def _str_or_none(value: Any) -> str | None:
    """Return a stripped string or ``None`` for non-string / empty values."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


def _str_list(value: Any) -> list[str]:
    """Coerce a YAML value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    return []


def _bool_or_none(value: Any) -> bool | None:
    """Coerce a YAML bool-ish value to bool or None."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    return None


def _int_or_none(value: Any) -> int | None:
    """Coerce a YAML numeric/string value to int or None."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _normalize_replay_trigger_mode(value: Any) -> str | None:
    """Normalize replay trigger mode to one of: always/conditional/manual/hybrid."""

    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"always", "conditional", "manual", "hybrid"}:
        return normalized
    return None


def load_selected_session_context_text(
    *,
    session_context_ref: str,
    session_context_active: bool,
    current_mode_id: str,
    content_loader: Callable[[str], str],
) -> str:
    """Load the active session context text, filtering by ``applies_to`` front matter.

    When ``session_context_active`` is ``True``, delegates to
    :func:`load_active_session_context_text` (filesystem path or
    directory). Otherwise, loads the authored text via ``content_loader``
    without mode filtering.
    """
    if not session_context_ref:
        return ""
    if session_context_active:
        return load_active_session_context_text(
            session_context_ref,
            current_mode_id=current_mode_id,
        )
    return content_loader(session_context_ref)


def load_active_session_context_text(
    context_ref: str,
    *,
    current_mode_id: str,
) -> str:
    """Load session context from a file path or directory, filtering by ``applies_to``.

    When ``context_ref`` is a directory, delegates to
    :func:`load_session_context_directory`. Files are read directly and
    filtered by ``applies_to`` front matter.
    """
    path = Path(context_ref)
    if path.is_dir():
        return load_session_context_directory(path, current_mode=current_mode_id)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    front_matter, body = parse_front_matter(text)
    if not applies_to_mode(front_matter, current_mode_id):
        return ""
    return body


def load_session_context_directory(path: Path, current_mode: str) -> str:
    """Load and concatenate session context files from a directory.

    Files are sorted by name. Each file's body is included only if its
    ``applies_to`` front matter matches ``current_mode`` (or is absent).
    Bodies are joined with blank lines.
    """
    parts: list[str] = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        try:
            text = child.read_text(encoding="utf-8")
        except OSError:
            continue
        front_matter, body = parse_front_matter(text)
        if not applies_to_mode(front_matter, current_mode):
            continue
        parts.append(body)
    return "\n\n".join(parts)


def applies_to_mode(front_matter: dict[str, Any], current_mode: str) -> bool:
    """Return ``True`` if the front matter ``applies_to`` includes the current mode.

    When ``applies_to`` is absent the file applies to all modes. A non-list
    value is coerced to a single-element list.
    """
    applies_to = front_matter.get("applies_to")
    if applies_to is None:
        return True
    if not isinstance(applies_to, list):
        applies_to = [applies_to]
    return current_mode in applies_to