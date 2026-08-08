"""Ambient runtime facts and extension-evidence builder registration.

This module owns the runtime-neutral helpers that survive the context-broker
unification:

- :func:`build_ambient_facts` — compact ambient runtime facts for prompt
  context injection.
- :func:`resolve_host_name` — OS hostname resolution honoring platform env
  vars.
- The module-level extension-evidence builder (``set_extension_evidence_builder``
  / ``reset_extension_evidence_builder``) used by the SDK's entry-point-aware
  builder.

Context *selection* (ranking, gating, budgeting, rendering) is owned by the
session-owned :class:`~kernel.context.context_mapper.ContextMapper`. The
legacy namespace broker (``select_ranked_context`` /
``render_context_packet_markdown``) was removed in favor of that single
engine.
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kernel.context.extension_evidence import NoOpExtensionEvidenceBuilder
from kernel.seams.protocols import ExtensionEvidenceBuilder


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


# Module-level extension-evidence builder. Hosts override via
# ``set_extension_evidence_builder`` when they need the production builder.
# Callers that need explicit injection (e.g. tests, kernel seams) pass the
# builder explicitly; when omitted, this module global is used.
_extension_evidence_builder: ExtensionEvidenceBuilder = NoOpExtensionEvidenceBuilder()


def set_extension_evidence_builder(builder: ExtensionEvidenceBuilder) -> None:
    """Override the module-level extension-evidence builder."""

    global _extension_evidence_builder
    _extension_evidence_builder = builder


def reset_extension_evidence_builder() -> None:
    """Reset the extension-evidence builder to the NoOp default (for tests)."""

    global _extension_evidence_builder
    _extension_evidence_builder = NoOpExtensionEvidenceBuilder()


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
