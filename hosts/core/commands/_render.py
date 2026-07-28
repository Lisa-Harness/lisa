"""Capability snapshot, debug renderers, runtime settings, help text, token stats.

Pure rendering helpers that read runtime state and emit Markdown strings.
No command dispatch lives here — the ``handle_*`` functions in ``_simple``
and ``_slice`` call these renderers.

Dependency direction: ``_render`` → ``_log`` (``_last_capability_manifest``,
``_last_turn_context_dict``), ``_routing`` (``mode_support_gap``),
``kernel.*``, ``hosts.core.*``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from typing import Any

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import get_operation_catalog
from kernel.tools.registry import get_tool_catalog

from hosts.core.commands._log import _last_capability_manifest, _last_turn_context_dict
from hosts.core.commands._routing import mode_support_gap
from hosts.core.model_runtime import load_manifest
from hosts.core.runtime import HostRuntime
from hosts.core.thread import load_recent_thread_entries
from hosts.core.token_accounting import format_session_exit_summary_lines

logger = logging.getLogger(__name__)


# ── Capability snapshot ─────────────────────────────────────────────


def build_capability_snapshot(runtime: HostRuntime) -> dict[str, Any]:
    """Build a capability snapshot dict for diagnostics and help rendering."""
    catalog = get_workflow_catalog()
    if runtime.manifest is None:
        runtime.manifest = load_manifest(
            {}, data_dir=runtime.data_dir, path=runtime.models_path
        )
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
                    for role_name, model_ref in manifest.roles.items()
                    if model_ref == model.model_id
                ),
                "reasoning_model": model.reasoning_model,
            }
            for model in manifest.models
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


def format_help() -> str:
    """Compose a capability-accurate help message.

    Mode commands are derived from the live workflow catalog: every supported
    mode with a ``command_pattern`` gets a ``/<mode>`` line built from its
    authored ``description``. Only host-only shortcuts (``/help``, ``/settings``,
    ``/tokens``, ``/debug``, ``/consolidate``, ``/quit``) and the
    runtime entry-point summary are hardcoded — everything else reflects the
    packs actually installed at startup.
    """
    catalog = get_workflow_catalog()
    lines: list[str] = ["Lisa Framework Host", "", "Supported commands:"]

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

    # Host-only shortcuts — the minimal hardcoded surface.
    lines.extend(
        [
            "- /pack list|enable|disable|install|uninstall manages pack activation "
            "and runtime installs (restart required after changes).",
            "- Natural-language turns route directly to the current mode; "
            "the model hands off mid-turn via the transition_stage tool.",
            "- /help capabilities shows the runtime capability snapshot.",
            "- /settings shows effective profile/timezone/debug/log-level runtime settings.",
            "- /tokens shows detailed session token breakdown by model, mode, and stage.",
            "- /debug context|capabilities|self shows local diagnostics.",
            "- /consolidate (or /slice) runs the explicit thread/memory compaction slice.",
            "- /intake runs one intake tick (collect signals, process, write state) in-process.",
            "- /pack list|enable|disable|install|uninstall manages packs.",
            "- /quit exits the interactive console loop.",
        ]
    )

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
            "need host callbacks.",
            "",
            "Run `/help capabilities` for the live mode/operation/tool/model inventory.",
        ]
    )
    return "\n".join(lines)
