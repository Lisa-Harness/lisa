"""Transport-neutral message dispatcher for Lisa framework hosts.

This is the core ``handle_message`` entry point that all transport hosts
(cli, web, vscode) call to execute one user turn. It:
1. Dispatches host shortcut commands (/help, /settings, /consolidate, etc.)
   and returns structured results.
2. Routes natural-language and slash-command turns through the front-door
   router and kernel turn runner.
3. Emits transport callbacks at each pipeline stage.
4. Persists thread/log state on the runtime.

It does NOT import Rich, Typer, aiohttp, or any transport-specific code.
Rendering is the transport host's job.

NOTE: This is the Phase 1 interface stub. The full implementation will be
wired in Phase 3 when the transport-neutral services are extracted from
``hosts/little_lisa/main.py``. For now it provides the function signature
and a minimal path so the package boundary is testable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hosts.core.types import HostCommandResult

if TYPE_CHECKING:
    from hosts.core.callbacks import TransportCallbacks
    from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)


async def handle_message(
    runtime: HostRuntime,
    message: str,
    transport: TransportCallbacks | None = None,
) -> HostCommandResult:
    """Route one user message through the host runtime.

    Args:
        runtime: The typed host runtime with config, MCP pool, and state.
        message: The raw user message (may be a slash command or natural
            language).
        transport: Optional transport callbacks for streaming, debug
            rendering, and error delivery. When ``None``, a
            ``NullTransport`` is used (headless / test mode).

    Returns:
        A ``HostCommandResult`` with the response text and optional
        structured data for transport rendering.
    """
    from hosts.core.callbacks import NullTransport

    tport = transport or NullTransport()
    stripped = message.strip()
    lowered = stripped.lower()

    logger.debug("handle_message: %s", stripped[:80])

    # ── Host command dispatch ─────────────────────────────────────────
    result = await _dispatch_host_command(runtime, stripped, lowered, message, tport)
    if result is not None:
        return result

    # ── Model turn (front-door + run_turn) ────────────────────────────
    # Phase 3 will wire the full front-door routing + run_turn path here.
    # For now, return a placeholder so the interface is testable.
    await tport.on_stage("model_call", {"message": stripped[:200]})
    result_text = f"[hosts.core] Turn execution not yet wired. Message: {stripped[:200]}"
    await tport.on_response(result_text)
    return HostCommandResult(text=result_text)


async def _dispatch_host_command(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    message: str,
    transport: TransportCallbacks,
) -> HostCommandResult | None:
    """Try each host shortcut command; return result or None to continue.

    Phase 3 will wire the full command dispatch (help, settings, tokens,
    consolidate, checkpoint, restore, pack, debug). For now, only a
    minimal /help stub is provided so the interface is testable.
    """
    if lowered.startswith("/help") or lowered == "help":
        text = (
            "Lisa Framework Host\n\n"
            "Commands:\n"
            "- /help — show this help\n"
            "- /settings — show runtime settings\n"
            "- /tokens — show token usage\n"
            "- /debug context|frontdoor|capabilities|self — diagnostics\n"
            "- /consolidate — run explicit slice\n"
            "- /checkpoint [label] — create checkpoint\n"
            "- /restore <id> — restore checkpoint\n"
            "- /checkpoints — list checkpoints\n"
            "- /pack list|enable|disable|install|uninstall — pack management\n"
            "- /deep <request> — route through deep-designated model\n"
            "- /quit — exit (transport-owned)\n\n"
            "Natural-language turns pass through the front-door router."
        )
        await transport.on_response(text)
        return HostCommandResult(text=text)

    return None