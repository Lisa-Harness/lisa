"""Capability snapshot, debug renderers, runtime settings, help text, token stats.

Pure rendering helpers that read runtime state and emit Markdown strings.
No command dispatch lives here — the ``handle_*`` functions in ``_simple``
and ``_slice`` call these renderers.

Dependency direction: ``_render`` → ``_log`` (``_last_capability_manifest``,
``_last_turn_context_dict``), ``_routing`` (``mode_support_gap``),
``kernel.*``, ``host.*``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from typing import Any

from kernel.host_command_discovery import discover_pack_host_commands
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import get_operation_catalog
from kernel.tools.registry import get_tool_catalog

from host.commands._log import _last_capability_manifest, _last_turn_context_dict
from host.commands._routing import mode_support_gap
from host.runtime import HostRuntime
from host.thread import load_recent_thread_entries
from host.token_accounting import format_session_exit_summary_lines

logger = logging.getLogger(__name__)


# ── Capability snapshot ─────────────────────────────────────────────


def build_capability_snapshot(runtime: HostRuntime) -> dict[str, Any]:
    """Build a capability snapshot dict for diagnostics and help rendering."""
    catalog = get_workflow_catalog()
    manifest = runtime.manifest
    operations = get_operation_catalog()
    tools = get_tool_catalog()
    supported_modes = [
        mode_id for mode_id, mode in sorted(catalog.items()) if mode_support_gap(mode) is None
    ]
    unsupported_modes = {
        mode_id: (mode_support_gap(mode) or "additional host wiring")
        for mode_id, mode in sorted(catalog.items())
        if mode_support_gap(mode) is not None
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "user_slug": runtime.config.user_slug,
        "timezone": runtime.config.timezone,
        "timezone_source": runtime.config.timezone_source,
        "supported_modes": supported_modes,
        "unsupported_modes": unsupported_modes,
        "operations": sorted(operations.keys()),
        "tools": sorted(tools.keys()),
        "models": [
            {
                "model_id": model.model_id,
                "provider": model.provider,
                "roles": sorted(
                    role_name
                    for role_name, model_ref in (
                        manifest.roles.items() if manifest is not None else []
                    )
                    if model_ref == model.model_id
                ),
                "reasoning_model": model.reasoning_model,
            }
            for model in (manifest.models if manifest is not None else [])
        ],
    }


def render_capability_help(runtime: HostRuntime) -> str:
    """Render a Markdown capability snapshot for ``/help capabilities``."""
    snapshot = build_capability_snapshot(runtime)
    mode_preview = ", ".join(snapshot["supported_modes"][:8])
    model_preview = ", ".join(model["model_id"] for model in snapshot["models"][:4]) or "none"
    unsupported_count = len(snapshot["unsupported_modes"])
    lines = [
        "## Capability Snapshot",
        f"- User context: {snapshot['user_slug']}",
        f"- Timezone: {snapshot['timezone']} ({snapshot['timezone_source']})",
        f"- Supported modes ({len(snapshot['supported_modes'])}): {mode_preview}",
        f"- Operations: {len(snapshot['operations'])}",
        f"- Tools: {len(snapshot['tools'])}",
        f"- Models: {model_preview}",
        f"- Unsupported installed modes: {unsupported_count} (run /debug capabilities)",
    ]
    live = _last_capability_manifest(runtime)
    if live is not None:
        meta = live.metadata
        active_mode = meta.get("mode_id", "default")
        active_count = len(live.entries)
        active_names = sorted(e.capability_id for e in live.entries)
        lines.append("")
        lines.append(f"### Active Mode: {active_mode}")
        lines.append(f"- callable_this_turn: {active_count} ({', '.join(active_names)})")
        lines.append(f"- tools_authoritative: {'yes' if meta.get('tools_authoritative') else 'no'}")
    return "\n".join(lines)


def render_capability_debug(runtime: HostRuntime) -> str:
    """Render a JSON capability debug payload for ``/debug capabilities``."""
    snapshot = build_capability_snapshot(runtime)
    live = _last_capability_manifest(runtime)
    payload: dict[str, Any] = {
        "total_horizon": {
            "supported_modes": snapshot["supported_modes"],
            "unsupported_modes": snapshot["unsupported_modes"],
            "operations": snapshot["operations"],
            "tools": snapshot["tools"],
            "models": snapshot["models"],
        },
    }
    if live is not None:
        payload["active_mode"] = {
            "mode_id": live.metadata.get("mode_id", "default"),
            "tools_authoritative": live.metadata.get("tools_authoritative", False),
            "callable": sorted(e.capability_id for e in live.entries),
            "kind_counts": live.metadata.get("kind_counts", {}),
        }
    else:
        payload["active_mode"] = None
    return "## Capability Debug\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


def render_debug_context(runtime: HostRuntime) -> str:
    """Render the last turn context for ``/debug context``."""
    context = _last_turn_context_dict(runtime) or {}
    if not context:
        return "No completed turn context is available yet."
    turn_input = context.get("turn_input") if isinstance(context, dict) else None
    if not isinstance(turn_input, dict):
        return "No assembled turn-input summary is available yet."
    payload = turn_input.get("assembled_user_payload")
    if not isinstance(payload, dict):
        return "No assembled user payload is available yet."
    lines = ["## Last Turn Context"]

    # System prompt summary
    system_prompt_info = turn_input.get("system_prompt")
    if isinstance(system_prompt_info, dict):
        lines.append("\n### system_prompt")
        lines.append(f"- chars: {system_prompt_info.get('chars', 0)}")
        lines.append(f"- preview: {system_prompt_info.get('preview', '(empty)')}")
        if system_prompt_info.get("omitted_for_brevity"):
            lines.append("- (truncated for brevity)")
    else:
        lines.append("\n### system_prompt\n(not available)")

    # Session context
    session_ref = runtime.config.session_context
    session_active = bool(session_ref)
    lines.append("\n### session_context")
    lines.append(f"- ref: {session_ref or '(none)'}")
    lines.append(f"- active: {'true' if session_active else 'false'}")

    for key in (
        "pre_turn_briefing",
        "memory_recall",
        "dynamic_user_context",
        "selected_user_context",
        "user_query",
    ):
        value = str(payload.get(key) or "").strip()
        lines.append(f"\n### {key}\n{value or '(empty)'}")
    return "\n".join(lines)


def render_debug_self(runtime: HostRuntime) -> str:
    """Render the host self-model for ``/debug self``."""
    snapshot = build_capability_snapshot(runtime)

    thread_entries = load_recent_thread_entries(
        thread_path=runtime.thread_path,
        limit=10,
    )
    state_snapshot = runtime.state_snapshot
    lines = [
        "## Host Self Model",
        f"- user_slug: {snapshot['user_slug']}",
        f"- timezone: {snapshot['timezone']} ({snapshot['timezone_source']})",
        f"- supported_modes: {len(snapshot['supported_modes'])}",
        f"- operations: {len(snapshot['operations'])}",
        f"- tools: {len(snapshot['tools'])}",
        f"- models: {len(snapshot['models'])}",
        f"- local_thread_entries: {len(thread_entries)}",
        f"- local_state_snapshot_loaded: {'yes' if isinstance(state_snapshot, dict) else 'no'}",
    ]
    live = _last_capability_manifest(runtime)
    if live is not None:
        meta = live.metadata
        active_mode = meta.get("mode_id", "default")
        active_count = len(live.entries)
        active_names = sorted(e.capability_id for e in live.entries)
        lines.append("")
        lines.append(f"### Active Mode: {active_mode}")
        lines.append(f"- callable_this_turn: {active_count} ({', '.join(active_names)})")
        lines.append(f"- tools_authoritative: {'yes' if meta.get('tools_authoritative') else 'no'}")
    return "\n".join(lines)


def render_runtime_settings(runtime: HostRuntime) -> str:
    """Render effective runtime settings for ``/settings``."""
    return "\n".join(
        [
            "## Runtime Settings",
            f"- timezone: {runtime.config.timezone}",
            f"- timezone_source: {runtime.config.timezone_source}",
            f"- session_context: {runtime.config.user_slug or 'none'}",
            f"- debug: {'on' if runtime.config.debug else 'off'}",
            f"- log_level: {runtime.config.effective_log_level}",
        ]
    )


def format_token_stats(runtime: HostRuntime) -> str:
    """Return on-demand detailed token statistics as Markdown."""
    state = dataclasses.asdict(runtime.tokens)
    return "\n".join(
        format_session_exit_summary_lines(
            state,
            session_started_utc=runtime.tokens.session_started_utc,
        )
    )


# ── Help ─────────────────────────────────────────────────────────────


def _pack_command_entries() -> list[Any]:
    """Return pack-registered host command entries for help rendering.

    Mirrors the lazy discovery in :func:`host.handle_message._pack_host_commands`
    without importing the dispatch layer (which would create a cycle, since
    ``handle_message`` imports ``commands``). Discovery lives in
    ``kernel.host_command_discovery`` and only imports kernel modules.
    """
    return discover_pack_host_commands()


def _pack_command_help_lines() -> list[str]:
    """Build ``/help`` lines for every pack-registered slash command.

    Uses each :class:`HostCommandSpec.help_text` if the pack supplied one;
    otherwise falls back to a minimal ``- <prefix> — pack command`` line so
    every installed command is discoverable even when the pack author omitted
    ``help_text``.
    """
    lines: list[str] = []
    for entry in _pack_command_entries():
        for spec in entry.specs:
            summary = (spec.help_text or "").strip()
            if not summary:
                summary = f"{spec.prefix} (pack command from {entry.pack_name})"
            lines.append(f"- {summary}")
    return lines


def format_help(runtime: HostRuntime | None = None) -> str:
    """Compose a capability-accurate, pack-aware help message.

    The overview has four sections:

    1. **Mode commands** — derived from the live workflow catalog: every
       supported mode with a ``command_pattern`` gets a ``/<mode>`` line built
       from its authored ``description``.
    2. **Host commands** — the host-intrinsic shortcuts (``/help``, ``/pack``,
       ``/models``, ``/modes``, ``/operations``, ``/tools``, ``/mcp``,
       ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``, ``/intake``,
       ``/quit``).
    3. **Pack commands** — every slash command registered by an installed
       pack via ``lisa.extensions.host_commands``, using the pack's own
       ``help_text`` so the host never owns pack command vocabulary.
    4. **Routing & inventory** — the default-chat note plus pointers to
       ``/help capabilities`` and ``/help <command>`` for per-command guidance.

    Only the host-intrinsic shortcuts are hardcoded; mode and pack commands
    reflect what is actually installed at startup.
    """
    catalog = get_workflow_catalog()
    lines: list[str] = ["Lisa Framework Host", ""]
    if runtime is not None:
        mode_id = runtime.current_mode_id or "default"
        lines.append(f"User: {runtime.config.user_slug}  |  Mode: {mode_id}")
        lines.append("")
    lines.append("Supported commands:")

    # Mode commands — derived from the catalog, not hardcoded.
    mode_lines: list[str] = []
    default_mode = catalog.get("default")
    for mode_id, mode in sorted(catalog.items()):
        if mode_support_gap(mode) is not None or mode_id == "default":
            continue
        if not mode.command_pattern:
            continue
        summary = (mode.description or "").strip()
        if not summary:
            summary = f"{mode_id} mode"
        mode_lines.append(f"- /{mode_id} <request> — {summary}")
    if mode_lines:
        lines.extend(mode_lines)

    # Host-intrinsic commands — the minimal hardcoded surface.
    lines.extend(
        [
            "- /help [command|capabilities] — this overview, per-command guidance, "
            "or the runtime capability snapshot.",
            "- /pack list|enable <pack>|disable <pack>|install <spec>|uninstall <package> "
            "manages pack activation and runtime installs (restart required after changes).",
            "- /models show|discover|apply|reload|edit — manage the active model manifest "
            "(run /help models for switches).",
            "- /modes — list every discovered mode (supported and unsupported).",
            "- /operations — list internal operation IDs for allowlist authoring.",
            "- /tools — list internal (pack-discovered) tools with limits and timeouts.",
            "- /mcp [server_id] — MCP server health summary, or per-server tool detail.",
            "- /settings — effective profile/timezone/debug/log-level runtime settings.",
            "- /tokens — session token breakdown by model, mode, and stage.",
            "- /debug context|capabilities|self — local diagnostics.",
            "- /consolidate (or /slice) — explicit thread/memory compaction slice.",
            "- /intake — one intake tick (collect signals, process, write state) in-process.",
            "- /quit — exit the interactive console loop.",
        ]
    )

    # Pack-registered commands — derived from discovery, not hardcoded.
    pack_lines = _pack_command_help_lines()
    if pack_lines:
        lines.append("")
        lines.append("Pack commands:")
        lines.extend(pack_lines)

    default_description = (
        (default_mode.description or "general conversation").strip()
        if default_mode is not None
        else "general conversation"
    )
    lines.append(f"- Any other text uses default chat ({default_description}).")

    lines.extend(
        [
            "",
            "Modes and operations added through SDK_CONTENT_ROOTS run when they only "
            "need host callbacks. Pack commands are discovered via the "
            "`lisa.extensions.host_commands` entry-point group.",
            "",
            "Run `/help <command>` (e.g. /help pack, /help models, /help mcp) for "
            "per-command guidance and switches, or `/help capabilities` for the live "
            "mode/operation/tool/model inventory.",
        ]
    )
    return "\n".join(lines)


def render_command_help(runtime: HostRuntime, command: str) -> str:
    """Render per-command guidance for ``/help <command>``.

    Delegates to the relevant help-text provider. Built-in host commands
    have explicit help here or in their own modules (e.g.
    :func:`_models_help_text`); pack-registered commands surface the
    ``help_text`` their pack authored. Unknown commands fall back to a
    pointer back to ``/help``.
    """
    topic = command.strip().lstrip("/").lower()
    if not topic:
        return format_help(runtime)
    if topic in {"capabilities", "capability"}:
        return render_capability_help(runtime)

    builtin = _builtin_command_help(topic, runtime)
    if builtin is not None:
        return builtin

    pack_help = _pack_command_help_for_topic(topic)
    if pack_help is not None:
        return pack_help

    return (
        f"No help available for '/{topic}'. Run /help to list every command, or "
        "/help capabilities for the live inventory."
    )


def _builtin_command_help(topic: str, runtime: HostRuntime) -> str | None:
    """Return dedicated help text for a built-in host command, or ``None``.

    ``None`` means ``topic`` is not a built-in; the caller then checks
    pack-registered commands.
    """
    if topic == "pack":
        return _pack_command_help_text()
    if topic == "models":
        # Lazy import: _models imports _simple which imports _render — a
        # module-level import here would create a cycle.
        from host.commands._models import _models_help_text  # noqa: PLC0415

        return _models_help_text(runtime)
    if topic == "mcp":
        return _mcp_command_help_text()
    if topic in {"modes", "operations", "tools"}:
        return _inventory_command_help_text(topic)
    if topic in {"help", "settings", "tokens", "debug", "consolidate",
                 "slice", "intake", "quit"}:
        return _host_shortcut_help_text(topic)
    return None


def _pack_command_help_for_topic(topic: str) -> str | None:
    """Return help text for a pack-registered command matching ``topic``.

    Surfaces the pack's own ``help_text``. Returns ``None`` when no
    pack command prefix matches.
    """
    for entry in _pack_command_entries():
        for spec in entry.specs:
            if spec.prefix.lstrip("/").lower() == topic:
                summary = (spec.help_text or "").strip()
                if not summary:
                    summary = f"{spec.prefix} (pack command from {entry.pack_name})"
                return (
                    f"## /{topic}\n\n"
                    f"- {summary}\n\n"
                    f"Pack: {entry.pack_name}\n\n"
                    f"Run `/{topic}` (with its arguments, if any) to invoke it."
                )
    return None


def _pack_command_help_text() -> str:
    """Help text for ``/help pack``."""
    return "\n".join(
        [
            "## /pack",
            "Manage pack activation and runtime installs.",
            "",
            "Subcommands:",
            "- /pack list — show discovered packs, activation state, and selected providers.",
            "- /pack enable <pack> — activate an installed (bundled/runtime) pack.",
            "- /pack disable <pack> — deactivate an installed pack "
            "(always-on packs cannot be disabled).",
            "- /pack install <spec> — install a pack from a git URL, local path, or bare name.",
            "    Append ` --resume <message>` to replay a turn after the restart.",
            "- /pack uninstall <package> — remove an installed pack (restart required).",
            "",
            "Activation changes take effect after a restart; install/uninstall reseed "
            "the packs venv and re-discover modes, operations, and commands.",
        ]
    )


def _mcp_command_help_text() -> str:
    """Help text for ``/help mcp``."""
    return "\n".join(
        [
            "## /mcp",
            "MCP server introspection readout from the live tool pool.",
            "",
            "Usage:",
            "- /mcp — per-server summary (status, transport, endpoint, tool count, last error).",
            "- /mcp <server_id> — warm a cold server on demand and list its tools with "
            "required schema args.",
            "",
            "The summary reads the already-warmed pool index; the detail path lazily "
            "connects cold servers. Configure servers via the MCP manifest "
            "(`lisa mcp` on the CLI exposes the same data through `--with-mcp`).",
        ]
    )


def _inventory_command_help_text(topic: str) -> str:
    """Help text for ``/help modes`` / ``/operations`` / ``/tools``."""
    return "\n".join(
        [
            f"## /{topic}",
            f"List the live {topic} catalog discovered from installed packs and "
            "SDK_CONTENT_ROOTS.",
            "",
            f"Run `/{topic}` to render the full table. These catalogs back the "
            "capability snapshot shown by /help capabilities:",
            "- /modes — mode id, kind, command pattern, and description "
            "(unsupported modes show their support gap).",
            "- /operations — operation id, execution kind, description, and bound tools "
            "(the id column feeds `allowed_operation_ids` front matter).",
            "- /tools — internal tool id, description, max calls/turn, and timeout "
            "(MCP tools are surfaced by `lisa tools --with-mcp`, not this command).",
        ]
    )


def _host_shortcut_help_text(topic: str) -> str:
    """Help text for the remaining host-intrinsic shortcuts."""
    blurbs: dict[str, str] = {
        "help": (
            "## /help\n\n"
            "- /help — overall command overview (host, mode, and pack commands).\n"
            "- /help <command> — per-command guidance and switches "
            "(e.g. /help pack, /help models, /help mcp).\n"
            "- /help capabilities — live capability snapshot "
            "(modes, operations, tools, models)."
        ),
        "settings": (
            "## /settings\n\n"
            "Show the effective runtime settings: profile, timezone, debug level, "
            "log level, and resolved paths."
        ),
        "tokens": (
            "## /tokens\n\n"
            "Show the session token breakdown by model, mode, and stage, plus the "
            "running session-exit summary."
        ),
        "debug": (
            "## /debug\n\n"
            "Local diagnostics:\n"
            "- /debug context — assembled turn context (mode, route, capability manifest).\n"
            "- /debug capabilities — JSON capability snapshot.\n"
            "- /debug self — runtime self-inspection (pool, providers, state)."
        ),
        "consolidate": (
            "## /consolidate (or /slice)\n\n"
            "Run the explicit thread/memory compaction slice in-process and persist "
            "the result."
        ),
        "slice": (
            "## /slice (or /consolidate)\n\n"
            "Alias for /consolidate: run the explicit thread/memory compaction slice."
        ),
        "intake": (
            "## /intake\n\n"
            "Run one intake tick in-process: collect signals, process them, and write "
            "the resulting state."
        ),
        "quit": (
            "## /quit\n\n"
            "Exit the interactive console loop. (CLI-only; never appended to the thread.)"
        ),
    }
    return blurbs.get(topic, f"## /{topic}\n\nRun /{topic} for details.")
