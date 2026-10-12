"""Ambient runtime facts for prompt context injection.

Owns the runtime-neutral helpers that survive the context-broker
unification:

- :func:`build_ambient_facts` — compact ambient runtime facts for prompt
  context injection.
- :func:`resolve_host_name` — OS hostname resolution honoring platform env
  vars.

Context *selection* (ranking, gating, budgeting, rendering) is owned by the
session-owned :class:`~kernel.context.context_mapper.ContextMapper`. The
legacy namespace broker (``select_ranked_context`` /
``render_context_packet_markdown``) and the write-only module-global
extension-item builder that survived alongside it are both gone — the
production builder reaches the turn loop explicitly through
``ContextSeams.extension_item_builder``, not a module global.
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "build_ambient_facts",
    "resolve_host_name",
]


def resolve_host_name() -> str:
    """Resolve the OS hostname for ambient context injection.

    Honors the platform-conventional env vars first so a user can override
    what the model sees (``HOSTNAME`` on POSIX, ``COMPUTERNAME`` on Windows),
    then falls back to :func:`socket.gethostname`. Returns an empty string
    only if every source is unavailable.
    """

    for env_var in ("HOSTNAME", "COMPUTERNAME"):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    try:
        return socket.gethostname() or ""
    except OSError:
        return ""


def build_ambient_facts(
    *,
    timezone: str = "UTC",
    host_os: str = "",
    host_name: str = "",
    now: datetime | None = None,
) -> str:
    """Build compact ambient runtime facts for prompt context injection."""

    now_utc = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    configured_timezone = (timezone or "UTC").strip() or "UTC"
    timezone_used = configured_timezone
    target_zone: tzinfo
    try:
        target_zone = ZoneInfo(configured_timezone)
    except ZoneInfoNotFoundError:
        # ZoneInfo may fail when the IANA tz database is unavailable
        # (e.g. Windows without the ``tzdata`` package).  Fall back to
        # the system local timezone rather than silently reporting UTC,
        # so the model sees the correct local time when the configured
        # zone can't be loaded.  This mirrors the status-line badge
        # fallback in ``hosts/cli/repl.py``.
        target_zone = datetime.now().astimezone().tzinfo or UTC
        timezone_used = str(target_zone) if target_zone is not UTC else "UTC"
    local_now = now_utc.astimezone(target_zone)

    lines = [
        f"- local_date: {local_now.strftime('%Y-%m-%d')}",
        f"- local_time: {local_now.strftime('%H:%M %Z')}",
        f"- local_weekday: {local_now.strftime('%A')}",
        f"- timezone: {timezone_used}",
        f"- utc_now: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if host_os.strip():
        lines.append(f"- host_os: {host_os.strip()}")
    if host_name.strip():
        lines.append(f"- host: {host_name.strip()}")
    return "\n".join(lines)
