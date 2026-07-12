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
- :func:`extract_front_matter_thread_name`
- :func:`load_selected_session_context_text`
- :func:`load_active_session_context_text`
- :func:`load_session_context_directory`
- :func:`applies_to_mode`
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kernel.ipsar_fsm import parse_front_matter

from hosts.core.lisa_config import context_slug_to_thread_name

logger = logging.getLogger(__name__)


def resolve_session_context(
    *,
    session_context: str,
    data_dir: Path,
    content_loader: Callable[[str], str],
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
    if (result := resolve_session_context_literal_path(s)) is not None:
        return result
    if (result := resolve_session_context_artifacts(s, data_dir=data_dir)) is not None:
        return result
    if (
        result := resolve_session_context_user_contexts(s, content_loader=content_loader)
    ) is not None:
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
    """Return the display name from YAML front matter if present.

    Checks ``display_name`` first, then falls back to ``thread_name``.
    Delegates to the canonical kernel :func:`parse_front_matter` so the
    parsing logic stays in one place. Returns ``None`` when the file
    cannot be read or no name is present.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    front_matter, _ = parse_front_matter(text)
    display_name = front_matter.get("display_name")
    if isinstance(display_name, str):
        return display_name.strip() or None
    thread_name = front_matter.get("thread_name")
    if isinstance(thread_name, str):
        return thread_name.strip() or None
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