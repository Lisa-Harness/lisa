"""Simple shortcut command handlers.

Handles ``/tokens``, ``/settings``, ``/pack`` passthrough, ``/help``,
``/debug``. These handlers set turn context, append thread/log entries, and
return static-rendered output. They are the shared sink that every other
command handler delegates to via :func:`handle_simple_command`.

Dependency direction: ``_simple`` → ``_log``, ``_render``. No imports from
``_packs``/``_lists``/``_models``/``_slice``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from host.commands._log import (
    _append_thread_entry,
    _append_turn_log,
    _set_turn_context,
)
from host.commands._render import (
    format_help,
    render_capability_help,
    render_command_help,
    render_debug_turn,
    render_last_turn,
)
from host.runtime import HostRuntime

logger = logging.getLogger(__name__)


def handle_simple_command(
    runtime: HostRuntime,
    message: str,
    mode_id: str,
    response_text: str,
) -> str:
    """Handle a simple shortcut command that just renders static output.

    Sets turn context, appends to thread/log, and returns the response text.
    """
    _set_turn_context(
        runtime,
        timestamp_utc=datetime.now(UTC).isoformat(),
        raw_message=message,
        mode_id=mode_id,
        handoff=f"{mode_id} shortcut",
    )
    _append_turn_log(
        context={
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_message": message,
            "mode_id": mode_id,
            "handoff": f"{mode_id} shortcut",
        },
        response_text=response_text,
    )
    _append_thread_entry(
        runtime,
        mode_id=mode_id,
        query=message,
        response_text=response_text,
    )
    runtime.session.current_mode_id = mode_id
    runtime.session.current_stage_id = "-"
    runtime.session.next_model_id = "-"
    return response_text


# ── /help ────────────────────────────────────────────────────────────


async def handle_help_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
) -> str:
    """Handle ``/help``, ``/help capabilities``, and ``/help <command>``.

    ``/help`` (or ``/help`` with an empty topic) renders the pack-aware
    overview via :func:`format_help`. Any non-empty topic delegates to
    :func:`render_command_help`, which resolves built-in host commands
    (``pack``, ``models``, ``mcp``, ``modes``, ``operations``, ``tools``,
    etc.) to dedicated help text and pack-registered commands to the
    pack's own ``help_text``.
    """
    help_topic = ""
    if lowered.startswith("/help"):
        help_topic = stripped[5:].strip().lower()
    if not help_topic:
        response_text = format_help(runtime)
    elif help_topic in {"capabilities", "capability"}:
        response_text = render_capability_help(runtime)
    else:
        response_text = render_command_help(runtime, help_topic)
    return handle_simple_command(runtime, message, "help", response_text)


# ── /debug ───────────────────────────────────────────────────────────


def handle_debug_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/debug [N]``: the complete, unedited model input for turn N."""
    arg = stripped[6:].strip()
    response_text = render_debug_turn(runtime, arg)
    return handle_simple_command(runtime, message, "debug", response_text)


def handle_last_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/last [N]``: reprint a previous response, verbatim."""
    arg = stripped[5:].strip()
    response_text = render_last_turn(runtime, arg)
    return handle_simple_command(runtime, message, "last", response_text)
