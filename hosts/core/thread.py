"""Transport-neutral thread persistence.

This module consolidates the pure, stateless logic for reading and writing
the per-user conversation thread JSONL file and its digest sidecar:

- Loading recent thread entries (oldest entries truncated to a limit).
- Appending a new entry tagged with the user slug, mode, query, and response.
- Loading the thread digest sidecar (``thread_digest.md``).
- Building a short human-readable excerpt of recent thread entries for
  prompt injection.

All functions are transport-neutral: they take explicit ``thread_path`` and
``user_slug`` parameters rather than reading global runtime state. A
transport host (Little Lisa, web host, VS Code host) wires its own
``HostRuntime.thread_path`` and ``HostConfig.user_slug`` into these
functions at startup.

Dependency direction
--------------------
``thread.py`` ← ``main.py`` / ``handle_message.py``  (callers import from
thread; thread never imports main or any transport code)

``thread.py`` → ``hosts.core.host_commands`` (leaf: command-prefix registry),
``hosts.core.lisa_config`` (constants). Neither imports thread back.

Public surface
--------------
- :func:`load_recent_thread_entries`
- :func:`append_thread_entry`
- :func:`load_thread_digest`
- :func:`build_recent_thread_excerpt`
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hosts.core.host_commands import is_host_command_query as _is_host_command_query
from hosts.core.lisa_config import _THREAD_DIGEST_FILE_NAME

logger = logging.getLogger(__name__)

# ``_is_host_command_query`` is re-exported here for backward compatibility
# only — callers should import directly from ``hosts.core.host_commands``.
# The persistence layer (this module) must not own command-name vocabulary;
# the registry lives in ``host_commands`` (a leaf module with no
# ``hosts.core`` imports) so both the command dispatch layer and the
# persistence layer can share it without a cycle.
__all__ = [
    "append_thread_entry",
    "build_recent_thread_excerpt",
    "load_recent_thread_entries",
    "load_thread_digest",
]


def load_recent_thread_entries(
    *,
    thread_path: Path | None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Load up to ``limit`` recent thread entries from the JSONL file.

    Returns an empty list when ``thread_path`` is ``None`` or the file does
    not exist. Malformed lines are silently skipped.
    """
    if thread_path is None or not thread_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in thread_path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            entries.append(decoded)
    return entries


def append_thread_entry(
    *,
    thread_path: Path | None,
    user_slug: str,
    mode_id: str,
    query: str,
    response_text: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Append a new thread entry to the JSONL file.

    No-ops when ``thread_path`` is ``None`` or ``user_slug`` is empty. Creates
    the parent directory if needed.
    """
    if thread_path is None or not user_slug:
        return
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "user_slug": user_slug,
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
    }
    if provenance:
        entry["provenance"] = provenance
        
    thread_path.parent.mkdir(parents=True, exist_ok=True)
    with thread_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def load_thread_digest(*, thread_path: Path | None) -> str:
    """Load the thread digest sidecar (``thread_digest.md``) if it exists.

    Returns an empty string when ``thread_path`` is ``None`` or the digest
    file is absent or unreadable.
    """
    if thread_path is None:
        return ""
    digest_path = thread_path.parent / _THREAD_DIGEST_FILE_NAME
    if not digest_path.is_file():
        return ""
    try:
        return digest_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def build_recent_thread_excerpt(
    *,
    thread_path: Path | None,
    limit: int = 3,
) -> str:
    """Build a short human-readable excerpt of recent thread entries.

    Returns an empty string when there are no entries. Each line shows the
    query and a preview of the response (or the query itself if the
    response is empty).

    Host-only command entries (``/pack disable github``, ``/help``,
    ``/tokens``, ...) are stripped: they are operator UI gestures, not
    dialogue, and replaying them as "recent conversation" makes the model
    treat a UI gesture as something the user said. The model already
    observes their effect via the live capability manifest / pack lock.

    The loader reads up to ``limit * 4`` raw entries (a bounded over-read)
    so that after stripping host commands the excerpt can still yield up
    to ``limit`` genuine dialogue turns in normal sessions.
    """
    # Bounded over-read so we can filter host commands and still fill the
    # excerpt with real dialogue. ``* 4`` is a heuristic: most sessions are
    # dominated by dialogue, with occasional ``/pack`` or ``/help`` runs.
    # If a user fires many host commands in a row the excerpt may run
    # short, which is the correct outcome (there is no fresh dialogue to
    # surface). Capped to avoid loading a pathological thread in full.
    raw_limit = max(limit, min(limit * 4, 200))
    entries = load_recent_thread_entries(thread_path=thread_path, limit=raw_limit)
    if not entries:
        return ""
    lines = ["Recent thread context:"]
    emitted = 0
    for entry in reversed(entries):
        if emitted >= limit:
            break
        query = str(entry.get("query") or "").strip()
        response = str(entry.get("response_text") or "").strip()
        if not query and not response:
            continue
        # Strip host-only commands (``/pack``, ``/help``, ``/tokens`` ...)
        # from the conversation transcript fed back to the model.
        if _is_host_command_query(query):
            continue
        body = response or query
        lines.append(f"- {query or 'unknown'}: {body}")
        emitted += 1
    if emitted == 0:
        return ""
    # Restore chronological order (oldest first) for readability.
    return "\n".join([lines[0], *reversed(lines[1:])])