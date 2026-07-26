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

from hosts.core.commands._log import (
    _append_thread_entry,
    _append_turn_log,
    _set_turn_context,
)
from hosts.core.commands._render import (
    format_help,
    render_capability_debug,
    render_capability_help,
    render_debug_context,
    render_debug_self,
)
from hosts.core.runtime import HostRuntime

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
    runtime.current_mode_id = mode_id
    runtime.current_stage_id = "-"
    runtime.next_model_id = "-"
    return response_text


# ── /help ────────────────────────────────────────────────────────────


async def handle_help_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
) -> str:
    """Handle ``/help`` and ``/help capabilities``."""
    help_topic = ""
    if lowered.startswith("/help"):
        help_topic = stripped[5:].strip().lower()
    if help_topic in {"capabilities", "capability"}:
        response_text = render_capability_help(runtime)
    else:
        response_text = format_help()
    return handle_simple_command(runtime, message, "help", response_text)


# ── /debug ───────────────────────────────────────────────────────────


def handle_debug_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/debug`` subcommands."""
    subcommand = stripped[6:].strip().lower()
    if subcommand in {"", "help"}:
        response_text = "Supported debug commands: /debug context, /debug capabilities, /debug self"
    elif subcommand == "context":
        response_text = render_debug_context(runtime)
    elif subcommand == "capabilities":
        response_text = render_capability_debug(runtime)
    elif subcommand == "self":
        response_text = render_debug_self(runtime)
    else:
        response_text = (
            f"Unknown debug command '{subcommand}'. Supported: context, capabilities, self."
        )
    return handle_simple_command(runtime, message, "debug", response_text)
