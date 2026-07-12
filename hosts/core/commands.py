"""Transport-neutral host command dispatch for Lisa framework hosts.

This module consolidates the slash-command handlers that do not require the
model: ``/help``, ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``
(``/slice``), ``/checkpoint``, ``/restore``, ``/checkpoints``, and ``/pack``.
All functions produce plain text output (Markdown strings) — no Rich
panels, no console printing, no transport-specific rendering.

Each command handler accepts a :class:`HostRuntime` (the typed replacement
for the old ``_RUNTIME_STATE`` dict) and optionally a
:class:`TransportCallbacks` for log/thread operations. The runtime owns all
mutable state; the commands read from and write to it through its typed
accessors.

Dependency direction
--------------------
``commands.py`` ← ``handle_message.py`` / transport hosts  (callers import
from commands; commands never import handle_message or any transport code)

``commands.py`` → ``hosts.core.thread``, ``hosts.core.content_roots``,
``hosts.core.token_accounting``, ``hosts.core.model_runtime``,
``hosts.core.state_provider``, ``kernel.*``  (commands imports these;
none of them import commands)

Public surface
---------------
- :func:`handle_simple_command`
- :func:`handle_help_command`
- :func:`handle_debug_command`
- :func:`run_explicit_slice_command`
- :func:`handle_checkpoint_command`
- :func:`handle_restore_command`
- :func:`handle_list_checkpoints_command`
- :func:`handle_pack_command`
- :func:`format_help`
- :func:`render_capability_help`
- :func:`render_capability_debug`
- :func:`render_debug_context`
- :func:`render_debug_self`
- :func:`render_debug_frontdoor`
- :func:`render_runtime_settings`
- :func:`format_token_stats`
- :func:`build_capability_snapshot`
- :func:`predict_model`
- :func:`set_runtime_route_preview`
- :func:`explicit_deep_query`
- :func:`preferred_model_id_for_route`
- :func:`model_alias_for_mode`
- :func:`frontdoor_model_alias`
- :func:`render_pack_list`
- :func:`mode_support_gap`
- :func:`format_unsupported_mode`
- :func:`format_operation_error`
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import time as _time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from kernel.content_roots import discover_installed_content_roots
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    CapabilityManifest,
    ConversationSurface,
    OperationExecutionContext,
    ToolResult,
    WorkflowModeConfig,
    WorkflowSessionState,
)
from kernel.operations.registry import (
    OperationExecutionError,
    execute_operation,
    get_operation_catalog,
)
from kernel.pack_activation import normalize_pack_name
from kernel.slice import SliceMode, SliceOutcome, SlicePolicy
from kernel.tools.registry import get_tool_catalog

from hosts.core.content_roots import (
    always_on_pack_names,
    infer_pack_package_name,
    pack_aliases,
    pack_is_active,
    runtime_enabled_pack_map,
    runtime_installed_pack_names,
    runtime_pack_lock_path,
)
from hosts.core.content_roots import (
    load_installed_packs_lock as _content_roots_load_lock,
)
from hosts.core.lisa_config import _resolve_slice_tuning
from hosts.core.model_runtime import StubModelDefinition, load_manifest, resolve_little_lisa_model
from hosts.core.runtime import HostRuntime, TurnContext
from hosts.core.state_provider import FileBackedStateProvider, build_unified_state
from hosts.core.thread import append_thread_entry, load_recent_thread_entries
from hosts.core.token_accounting import format_session_exit_summary_lines

logger = logging.getLogger(__name__)

# ── Environment-variable name constants ──────────────────────────────

LISA_PACK_ALLOWED_PREFIXES_ENV = "LISA_PACK_ALLOWED_PREFIXES"
LISA_PACK_GIT_ALLOW_HOSTS_ENV = "LISA_PACK_GIT_ALLOW_HOSTS"
LISA_PACK_REQUIRE_GIT_REF_ENV = "LISA_PACK_REQUIRE_GIT_REF"

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]

_MODE_MODEL_ALIAS_MAP = {
    "default": "default",
    "discuss": "default",
    "search": "default",
    "deep": "deep",
}


# ── Private no-op stubs (logging wired later) ────────────────────────


def _append_log_event(
    *,
    level: str,
    event: str,
    payload: dict[str, Any],
    debug_payload: dict[str, Any] | None = None,
) -> None:
    """No-op stub — structured logging will be wired in a later phase."""
    del level, event, payload, debug_payload


def _append_turn_log(*, context: dict[str, Any], response_text: str) -> None:
    """No-op stub — turn logging will be wired in a later phase."""
    del context, response_text


# ── Private helpers ──────────────────────────────────────────────────


def _append_thread_entry(
    runtime: HostRuntime,
    *,
    mode_id: str,
    query: str,
    response_text: str,
) -> None:
    """Delegate to ``hosts.core.thread.append_thread_entry`` with runtime paths."""
    append_thread_entry(
        thread_path=runtime.thread_path,
        user_slug=runtime.config.user_slug,
        mode_id=mode_id,
        query=query,
        response_text=response_text,
    )


def _set_turn_context(runtime: HostRuntime, **kwargs: Any) -> None:
    """Build a ``TurnContext`` and store it on the runtime."""
    ctx = TurnContext(**kwargs)
    runtime.set_last_turn_context(ctx)


def _resolve_int_env(env_var: str, default: int) -> int:
    """Parse a positive integer from an env var with a sensible default."""
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


_MIN_RESTORE_ARGS = 2


def _resolve_model(runtime: HostRuntime, model_id: str) -> StubModelDefinition:
    """Resolve one model identifier against the runtime's manifest cache."""
    if runtime.manifest is None:
        runtime.manifest = load_manifest({})
    return resolve_little_lisa_model(
        model_id,
        runtime_state={"manifest": runtime.manifest},
    )


def _last_capability_manifest(runtime: HostRuntime) -> CapabilityManifest | None:
    """Return the last capability manifest stored on the runtime, if any."""
    return getattr(runtime, "last_capability_manifest", None)


def _last_turn_context_dict(runtime: HostRuntime) -> dict[str, Any] | None:
    """Return the last turn context as a dict (for debug rendering compatibility)."""
    ctx = runtime.last_turn_context
    if ctx is None:
        return None
    return dataclasses.asdict(ctx)


def _last_frontdoor_debug(runtime: HostRuntime) -> dict[str, Any] | None:
    """Return the last front-door debug payload from the runtime."""
    return runtime.last_frontdoor_debug


# ── Mode / model alias helpers ───────────────────────────────────────


def frontdoor_model_alias() -> str:
    """Return the model alias used for front-door routing."""
    return "front-door"


def model_alias_for_mode(mode_id: str) -> str:
    """Return the model alias for a given mode id."""
    return _MODE_MODEL_ALIAS_MAP.get(mode_id, "default")


def explicit_deep_query(message: str) -> str | None:
    """Extract a deep-query from a ``/deep ...`` message; return None if not deep."""
    stripped = message.lstrip()
    match = re.match(r"^/deep(?:[\s,:;]+(?P<query>.+))?$", stripped, flags=re.IGNORECASE)
    if not match:
        return None
    return str(match.group("query") or "").strip()


def preferred_model_id_for_route(*, raw_message: str, mode_id: str, deep_route: bool) -> str:
    """Return the preferred model id for a route decision.

    ``raw_message`` is kept in the signature for callers that may use it for
    message-aware model selection in future hosts; the core host routes by
    mode alias only.
    """
    del raw_message
    if deep_route:
        return "deep"
    return model_alias_for_mode(mode_id)


def predict_model(runtime: HostRuntime, preferred_model_id: str) -> tuple[str, str]:
    """Resolve a preferred model id to (model_id, provider), tolerating errors."""
    try:
        model = _resolve_model(runtime, preferred_model_id)
        return model.model_id, model.provider
    except Exception:  # pylint: disable=broad-except
        return preferred_model_id, "unknown"


def set_runtime_route_preview(
    runtime: HostRuntime,
    *,
    mode_id: str,
    preferred_model_id: str,
) -> None:
    """Set the runtime's current mode/stage/model route preview."""
    catalog = get_workflow_catalog()
    mode = catalog.get(mode_id)
    stage_id = "-"
    if mode is not None and mode.mode_kind == "workflow":
        stage_id = str(mode.start_stage_id).strip() or "-"
    model_id, provider = predict_model(runtime, preferred_model_id)
    runtime.current_mode_id = mode_id
    runtime.current_stage_id = stage_id or "-"
    runtime.next_model_id = model_id
    runtime.next_model_provider = provider
    logger.debug(
        "route preview set: mode=%s stage=%s model=%s provider=%s",
        mode_id,
        stage_id or "-",
        model_id,
        provider,
    )


# ── Mode support ─────────────────────────────────────────────────────


def mode_support_gap(_mode: WorkflowModeConfig) -> str | None:
    """Return the missing host capability for modes the host cannot run.

    The core host delegates to the kernel's ``run_turn`` which handles all
    modes. There are no gaps — always returns ``None``.
    """
    return None


def format_unsupported_mode(mode: WorkflowModeConfig) -> str:
    """Return an explicit capability gap for installed modes the host does not wire."""
    requirement = mode_support_gap(mode) or "additional host wiring"
    return "\n".join(
        [
            f"Mode '{mode.mode_id}' is installed in the public packs, "
            "but this host does not wire it.",
            f"It requires {requirement}.",
            "Run --list-modes to see what this host supports.",
        ]
    )


def format_operation_error(exc: OperationExecutionError) -> str:
    """Format an operation execution error for user-facing display."""
    message = str(exc).strip()
    lowered = message.lower()
    if "timed out" in lowered:
        return (
            f"{message}\n\n"
            "The host kept running, but this operation exceeded its local time budget. "
            "Try a narrower query, a faster model, or a larger operation timeout."
        )
    return message


# ── Capability snapshot ─────────────────────────────────────────────


def build_capability_snapshot(runtime: HostRuntime) -> dict[str, Any]:
    """Build a capability snapshot dict for diagnostics and help rendering."""
    catalog = get_workflow_catalog()
    if runtime.manifest is None:
        runtime.manifest = load_manifest({})
    manifest = runtime.manifest
    operations = get_operation_catalog()
    tools = get_tool_catalog()
    supported_modes = [
        mode_id
        for mode_id, mode in sorted(catalog.items())
        if mode_support_gap(mode) is None
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
                "designations": sorted(model.designations),
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


def render_debug_frontdoor(runtime: HostRuntime) -> str:
    """Render front-door routing diagnostics for ``/debug frontdoor``."""
    payload = _last_frontdoor_debug(runtime)
    if payload is None:
        return "No front-door routing diagnostics are available yet."
    return "## Front-Door Debug\n\n```json\n" + json.dumps(payload, indent=2) + "\n```"


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
    ``/tokens``, ``/debug``, ``/consolidate``, ``/deep``, ``/quit``) and the
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
            "- /deep <request> routes through default mode on the deep-designated model.",
            "- /pack list|enable|disable|install|uninstall manages pack activation "
            "and runtime installs (restart required after changes).",
            "- Natural-language turns pass through the front-door route planner first.",
            "- /help capabilities shows the runtime capability snapshot.",
            "- /settings shows effective profile/timezone/debug/log-level runtime settings.",
            "- /tokens shows detailed session token breakdown by model, mode, and stage.",
            "- /debug context|capabilities|self|frontdoor shows local diagnostics.",
            "- /consolidate (or /slice) runs the explicit thread/memory compaction slice.",
            "- /checkpoint [label] saves a state+workflow checkpoint to lisa-storage.",
            "- /restore <checkpoint_id> restores state from a saved checkpoint.",
            "- /checkpoints lists all saved checkpoints (newest first).",
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


# ── Simple command (shared by /tokens, /settings, /pack, /help, /debug) ──


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
        response_text = (
            "Supported debug commands: "
            "/debug context, /debug capabilities, /debug self, /debug frontdoor"
        )
    elif subcommand == "context":
        response_text = render_debug_context(runtime)
    elif subcommand == "capabilities":
        response_text = render_capability_debug(runtime)
    elif subcommand == "self":
        response_text = render_debug_self(runtime)
    elif subcommand == "frontdoor":
        response_text = render_debug_frontdoor(runtime)
    else:
        response_text = (
            f"Unknown debug command '{subcommand}'. "
            "Supported: context, capabilities, self, frontdoor."
        )
    return handle_simple_command(runtime, message, "debug", response_text)


# ── /consolidate (/slice) ──────────────────────────────────────────


async def run_explicit_slice_command(
    runtime: HostRuntime,
    *,
    message: str,
) -> str:
    """Run the explicit-mode slice as a user-invoked command.

    This is the escape hatch for hosts whose opportunistic pre/post
    slices are disabled (or whose state provider has no work to do).
    Calling ``/consolidate`` (or ``/slice``) invokes the same slice
    machinery the kernel uses, but with a one-shot user-controlled
    budget read from the ``LISA_SLICE_BUDGET_MS`` env var.
    """
    _, explicit_budget_ms = _resolve_slice_tuning()
    provider = FileBackedStateProvider(
        thread_path=runtime.thread_path,
        max_thread_entries=_resolve_int_env(
            "LISA_THREAD_MAX_ENTRIES",
            runtime.config.thread_max_entries,
        ),
    )
    policy = SlicePolicy(
        mode=SliceMode.EXPLICIT,
        budget_ms=explicit_budget_ms,
    )
    started = _time.monotonic()
    outcome: SliceOutcome = await provider.slice(ctx=None, policy=policy)
    elapsed_ms = (_time.monotonic() - started) * 1000.0
    runtime.last_slice_finished_at = _time.monotonic()
    timestamp_utc = datetime.now(UTC).isoformat()
    slice_data = {
        "budget_ms": explicit_budget_ms,
        "provider": type(provider).__name__,
        "work_items_completed": outcome.work_items_completed,
        "last_item": outcome.last_item,
        "notes": list(outcome.notes),
        "elapsed_ms": round(elapsed_ms, 3),
        "policy_budget_ms": policy.budget_ms,
    }
    _set_turn_context(
        runtime,
        timestamp_utc=timestamp_utc,
        raw_message=message,
        mode_id="consolidate",
        handoff="explicit slice command",
        turn_input={"explicit_slice": slice_data},
    )
    _append_turn_log(
        context={
            "timestamp_utc": timestamp_utc,
            "raw_message": message,
            "mode_id": "consolidate",
            "handoff": "explicit slice command",
            "explicit_slice": slice_data,
        },
        response_text="",
    )
    _append_thread_entry(
        runtime,
        mode_id="consolidate",
        query=message,
        response_text="",
    )
    runtime.current_mode_id = "consolidate"
    runtime.current_stage_id = "-"
    runtime.next_model_id = "-"
    if outcome.work_items_completed == 0:
        return (
            f"Slice ran in {elapsed_ms:.1f} ms with no work items. "
            "Thread and memory are within their caps; nothing to compact."
        )
    note_text = "; ".join(outcome.notes) if outcome.notes else "ok"
    return (
        f"Slice drained {outcome.work_items_completed} work item(s) in "
        f"{elapsed_ms:.1f} ms (budget {policy.budget_ms} ms). "
        f"Last item: {outcome.last_item or '<none>'}. Notes: {note_text}."
    )


# ── /checkpoint ──────────────────────────────────────────────────────


async def handle_checkpoint_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/checkpoint [label]`` — create an enriched checkpoint.

    Delegates to the ``checkpoint_create`` operation from lisa-pack-checkpoint,
    which calls the model to summarize the conversation, then persists an
    enriched ``CheckpointBundle`` via ``lisa-storage``.
    """
    parts = stripped.split(None, 1)
    label = parts[1].strip() if len(parts) > 1 else ""

    # Load turn history from the thread JSONL.
    turn_history = load_recent_thread_entries(
        thread_path=runtime.thread_path,
        limit=50,
    )
    normalized_history = [
        {
            "timestamp_utc": entry.get("timestamp_utc", ""),
            "mode_id": entry.get("mode_id", ""),
            "query": entry.get("query", ""),
            "response_text": entry.get("response_text", ""),
        }
        for entry in turn_history
    ]

    # Read current UnifiedState.
    state = build_unified_state(state_snapshot=runtime.state_snapshot)

    pool = runtime.mcp_tool_pool
    if pool is None:
        return "No storage pool available to persist checkpoint."

    async def _checkpoint_tool_executor(tool_id: str, _ctx: Any) -> ToolResult:
        """Route checkpoint tools to lisa-storage via server-scoped dispatch."""
        params = getattr(_ctx, "parameters", {}) or {}
        mcp_result = await pool.call_tool(tool_id, params, server_id="lisa-storage")
        return ToolResult(
            tool_id=tool_id,
            output_text=mcp_result.output_text,
            error=mcp_result.output_text if mcp_result.is_error else "",
        )

    ctx = OperationExecutionContext(
        workspace_id="lite-host",
        user_id=str(runtime.config.user_slug or "lite-user"),
        user_slug=str(runtime.config.user_slug or "default"),
        surface_type=cast(ConversationSurface, runtime.config.surface_type),
        surface_id=runtime.config.surface_id,
        memory_scope="personal",
        mode_id=str(runtime.current_mode_id or "default"),
        query=message,
        complete_fn=getattr(runtime, "complete_fn", None),
        resolve_model_fn=lambda model_id: _resolve_model(runtime, model_id),
        execute_tool_fn=_checkpoint_tool_executor,
        state_provider=lambda: state,
        params={
            "label": label,
            "turn_history": normalized_history,
        },
    )

    try:
        result = await execute_operation("checkpoint_create", ctx)
        return result.output_text or "Checkpoint created."
    except Exception as exc:  # pylint: disable=broad-except
        return f"Checkpoint failed: {exc}"


# ── /restore ─────────────────────────────────────────────────────────


async def handle_restore_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/restore <checkpoint_id>`` — restore state from a checkpoint.

    Delegates to the ``checkpoint_restore`` operation from lisa-pack-checkpoint,
    which loads the bundle via ``lisa-storage`` and returns the full contents.
    The host then restores UnifiedState from the returned ``output_json``.
    """
    parts = stripped.split(None, 1)
    if len(parts) < _MIN_RESTORE_ARGS or not parts[1].strip():
        return "Usage: /restore <checkpoint_id>"

    checkpoint_id = parts[1].strip()
    pool = runtime.mcp_tool_pool
    if pool is None:
        return "No storage pool available to restore from."

    async def _checkpoint_tool_executor(tool_id: str, _ctx: Any) -> ToolResult:
        """Route checkpoint tools to lisa-storage via server-scoped dispatch."""
        params = getattr(_ctx, "parameters", {}) or {}
        mcp_result = await pool.call_tool(tool_id, params, server_id="lisa-storage")
        return ToolResult(
            tool_id=tool_id,
            output_text=mcp_result.output_text,
            error=mcp_result.output_text if mcp_result.is_error else "",
        )

    ctx = OperationExecutionContext(
        workspace_id="lite-host",
        user_id=str(runtime.config.user_slug or "lite-user"),
        user_slug=str(runtime.config.user_slug or "default"),
        surface_type=cast(ConversationSurface, runtime.config.surface_type),
        surface_id=runtime.config.surface_id,
        memory_scope="personal",
        mode_id=str(runtime.current_mode_id or "default"),
        query=message,
        execute_tool_fn=_checkpoint_tool_executor,
        params={"checkpoint_id": checkpoint_id},
    )

    try:
        result = await execute_operation("checkpoint_restore", ctx)

        # Restore state from the returned bundle.
        bundle_data = result.output_json or {}
        if not isinstance(bundle_data, dict):
            bundle_data = {}

        unified_state_dict = bundle_data.get("unified_state") or {}
        if unified_state_dict:
            runtime.state_snapshot = unified_state_dict

        # Workflow session restore requires turn hooks wiring not yet
        # available in the core runtime. When the runtime exposes a
        # ``workflow_session_updater``, this will be wired here.
        wf_session_dict = bundle_data.get("workflow_session_state") or {}
        if wf_session_dict:
            updater = getattr(runtime, "workflow_session_updater", None)
            if updater is not None:
                try:
                    session = WorkflowSessionState.model_validate(wf_session_dict)
                    updater(session)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("restore: workflow session restore failed: %s", exc)
            else:
                logger.debug(
                    "restore: workflow session state present but no updater wired on runtime"
                )

        return result.output_text or f"Restored checkpoint {checkpoint_id}"
    except Exception as exc:  # pylint: disable=broad-except
        return f"Restore failed: {exc}"


# ── /checkpoints ─────────────────────────────────────────────────────


async def handle_list_checkpoints_command(
    runtime: HostRuntime,
    message: str,
) -> str:
    """Handle ``/checkpoints`` — list all saved checkpoints.

    ``message`` is accepted for API symmetry with other command handlers but
    is not used in the list path.
    """
    del message
    pool = runtime.mcp_tool_pool
    if pool is None:
        return "No storage pool available to list checkpoints."

    try:
        result = await pool.call_tool("list_checkpoints", {}, server_id="lisa-storage")
        result_text = getattr(result, "output_text", "") or ""

        data = json.loads(result_text)
        if isinstance(data, dict) and "error" in data:
            return f"List failed: {data['error']}"
        if not isinstance(data, list) or not data:
            return "No checkpoints saved."

        lines = ["Saved checkpoints (newest first):"]
        for entry in data:
            cid = entry.get("checkpoint_id", "")
            label = entry.get("label") or ""
            created = entry.get("created_at", "")[:19]
            stage = entry.get("stage_id") or ""
            turns = entry.get("turn_count") or 0
            lines.append(
                f"  {cid}  {created}  stage={stage}  turns={turns}"
                + (f"  label={label}" if label else "")
            )
        return "\n".join(lines)
    except Exception as exc:  # pylint: disable=broad-except
        return f"List failed: {exc}"


# ── /pack ────────────────────────────────────────────────────────────


def _load_installed_packs_lock(runtime: HostRuntime) -> dict[str, Any]:
    """Load installed pack provenance metadata from disk."""
    return _content_roots_load_lock(runtime.config.data_dir)


def _write_installed_packs_lock(runtime: HostRuntime, payload: dict[str, Any]) -> None:
    """Persist installed pack provenance metadata to disk."""
    path = runtime_pack_lock_path(runtime.config.data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def installed_pack_roots_by_name() -> dict[str, Path]:
    """Return a ``{package_name: root}`` map for all discovered installed pack roots."""
    roots: dict[str, Path] = {}
    for root in discover_installed_content_roots():
        package_name = infer_pack_package_name(root)
        if not package_name or package_name in roots:
            continue
        roots[package_name] = root
    return roots


def pack_alias_map(lock: dict[str, Any]) -> dict[str, str]:
    """Build a ``{alias: package_name}`` map from installed roots, always-on, and lock."""
    alias_map: dict[str, str] = {}
    installed_roots = installed_pack_roots_by_name()

    for package_name, root in installed_roots.items():
        for alias in pack_aliases(package_name, root=root):
            alias_map.setdefault(alias, package_name)

    for package_name in always_on_pack_names():
        for alias in pack_aliases(package_name):
            alias_map.setdefault(alias, package_name)

    entries = lock.get("packs") if isinstance(lock, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        package_name = normalize_pack_name(str(entry.get("package_name") or ""))
        if not package_name:
            continue
        for alias in pack_aliases(package_name):
            alias_map.setdefault(alias, package_name)

    return alias_map


def resolve_pack_name(
    raw: str,
    lock: dict[str, Any] | None = None,
) -> str:
    """Resolve a user-supplied pack name or alias to its canonical package name."""
    normalized = normalize_pack_name(raw)
    if not normalized:
        return ""

    lock_payload = lock if isinstance(lock, dict) else {}
    aliases = pack_alias_map(lock_payload)
    if normalized in aliases:
        return aliases[normalized]

    prefixed = normalized if normalized.startswith("lisa-pack-") else f"lisa-pack-{normalized}"
    return aliases.get(prefixed, "")


def _clear_selected_provider_for_package(
    lock: dict[str, Any],
    package_name: str,
) -> list[str]:
    """Remove selected-capability-provider entries that point to ``package_name``."""
    selected = lock.get("selected_capability_providers")
    if not isinstance(selected, dict):
        lock["selected_capability_providers"] = {}
        return []

    cleared: list[str] = []
    for capability_id, provider_id in list(selected.items()):
        if not isinstance(capability_id, str) or not isinstance(provider_id, str):
            continue
        if normalize_pack_name(provider_id) != package_name:
            continue
        selected.pop(capability_id, None)
        cleared.append(capability_id)
    return cleared


def _pack_contributes_memory_capability(package_name: str) -> bool:
    """Return True if the pack contributes a memory capability."""
    normalized = normalize_pack_name(package_name)
    if not normalized:
        return False

    from kernel.mcp_launch.server_manifest import (  # noqa: PLC0415
        MCPServerManifestError,
        load_mcp_server_manifest,
    )

    for root in discover_installed_content_roots():
        if infer_pack_package_name(root) != normalized:
            continue
        manifest_path = root / "mcp_servers.yaml"
        if not manifest_path.is_file():
            continue
        try:
            _, memory = load_mcp_server_manifest(manifest_path)
        except MCPServerManifestError:
            continue
        if memory.default_backend or memory.backends:
            return True

    return normalized.startswith("lisa-pack-mem")


def _auto_select_capability_provider_on_install(
    lock: dict[str, Any],
    package_name: str,
) -> None:
    """Auto-select a newly-installed pack as the memory capability provider if eligible."""
    selected = lock.get("selected_capability_providers")
    if not isinstance(selected, dict):
        selected = {}
        lock["selected_capability_providers"] = selected

    capability_id = "memory"
    if selected.get(capability_id):
        return
    if package_name in always_on_pack_names():
        return
    if not _pack_contributes_memory_capability(package_name):
        return

    selected[capability_id] = package_name


def set_pack_enabled(
    runtime: HostRuntime,
    package_name: str,
    *,
    enabled: bool,
) -> tuple[bool, str]:
    """Persist explicit activation for one pack and request host restart."""
    lock = _load_installed_packs_lock(runtime)
    resolved = resolve_pack_name(package_name, lock)
    if not resolved:
        return False, (
            f"Unknown pack '{package_name}'. Run /pack list to see available packs "
            "and their canonical names."
        )

    always_on = always_on_pack_names(
        runtime.config.mcp_manifest_path,
        always_on_packs=runtime.config.always_on_packs,
    )
    if not enabled and resolved in always_on:
        return False, f"Pack '{resolved}' is always on and cannot be disabled."

    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    index = next(
        (
            i
            for i, entry in enumerate(entries)
            if normalize_pack_name(str(entry.get("package_name") or "")) == resolved
        ),
        -1,
    )
    now_iso = datetime.now(UTC).isoformat()

    if index >= 0:
        entry = dict(entries[index])
        entry["package_name"] = resolved
        entry["enabled"] = enabled
        entry["updated_at"] = now_iso
        entry.setdefault("source_kind", "bundled")
        if entry.get("source_kind") == "bundled":
            entry.setdefault("source_ref", "bundled")
        entries[index] = entry
    else:
        entries.append(
            {
                "package_name": resolved,
                "enabled": enabled,
                "source_kind": "bundled",
                "source_ref": "bundled",
                "updated_at": now_iso,
            }
        )

    lock["packs"] = entries

    cleared: list[str] = []
    if not enabled:
        cleared = _clear_selected_provider_for_package(lock, resolved)

    _write_installed_packs_lock(runtime, lock)
    action = "enable" if enabled else "disable"
    runtime.request_restart(reason=f"pack {action}: {resolved}")

    if enabled:
        return (
            True,
            f"Enabled '{resolved}'. Restart required to reload content roots and MCP manifests.",
        )

    if cleared:
        return (
            True,
            f"Disabled '{resolved}' and cleared provider selections for: "
            f"{', '.join(sorted(cleared))}. "
            "Restart required to reload content roots and MCP manifests.",
        )
    return (
        True,
        f"Disabled '{resolved}'. Restart required to reload content roots and MCP manifests.",
    )


def pack_allowed_prefixes() -> list[str]:
    """Return allowed package-name prefixes for runtime pack installs."""
    configured = os.getenv(LISA_PACK_ALLOWED_PREFIXES_ENV, "").strip()
    if not configured:
        return ["lisa-pack-"]
    values = [value.strip().lower() for value in configured.split(",") if value.strip()]
    return values or ["lisa-pack-"]


def pack_allowed_git_hosts() -> set[str]:
    """Return allowed hosts for git-based pack installation refs."""
    configured = os.getenv(LISA_PACK_GIT_ALLOW_HOSTS_ENV, "").strip()
    if not configured:
        return {"github.com"}
    values = {value.strip().lower() for value in configured.split(",") if value.strip()}
    return values or {"github.com"}


def pack_require_git_ref() -> bool:
    """Return whether git install refs must pin @tag/@branch/@sha."""
    configured = os.getenv(LISA_PACK_REQUIRE_GIT_REF_ENV, "true").strip().lower()
    return configured not in {"0", "false", "no", "off"}


def _is_git_pack_spec(spec: str) -> bool:
    return spec.strip().lower().startswith("git+")


def _is_local_pack_spec(spec: str) -> bool:
    raw = spec.strip().strip('"')
    if not raw:
        return False
    return Path(raw).expanduser().is_dir()


def _read_local_pack_project_name(path: Path) -> str:
    """Return ``project.name`` from a local pack's pyproject.toml, or empty."""
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return ""
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    project = payload.get("project")
    if not isinstance(project, dict):
        return ""
    name = project.get("name")
    return str(name).strip() if isinstance(name, str) else ""


def _extract_package_name_from_spec(spec: str) -> str:
    """Infer package name from package or git install spec."""
    raw = spec.strip()
    if not raw:
        return ""
    if _is_local_pack_spec(raw):
        path = Path(raw.strip('"')).expanduser()
        project_name = _read_local_pack_project_name(path)
        if project_name:
            return project_name
        return path.name.strip()
    if _is_git_pack_spec(raw):
        egg_marker = "#egg="
        lowered = raw.lower()
        marker_index = lowered.find(egg_marker)
        if marker_index >= 0:
            return raw[marker_index + len(egg_marker):].strip()
        base = raw.split("@", 1)[0].rstrip("/")
        tail = base.rsplit("/", 1)[-1]
        return tail[:-4] if tail.lower().endswith(".git") else tail

    first = raw.split()[0]
    name = re.split(r"[<>=!~\[]", first, maxsplit=1)[0]
    return name.strip()


def _extract_git_host(spec: str) -> str:
    """Best-effort host extraction for git+https refs."""
    raw = spec.strip()
    if not raw.lower().startswith("git+https://"):
        return ""
    remainder = raw[len("git+https://"):]
    if not remainder:
        return ""
    return remainder.split("/", 1)[0].strip().lower()


def _validate_pack_spec_for_install(spec: str) -> tuple[bool, str]:  # noqa: C901, PLR0911
    """Validate a runtime pack install reference against host trust policy."""
    normalized = spec.strip()
    if not normalized:
        return False, "Usage: /pack install <package-ref-or-git-ref>"

    if _is_local_pack_spec(normalized):
        pack_path = Path(normalized.strip('"')).expanduser()
        project_name = _read_local_pack_project_name(pack_path)
        if not project_name:
            return False, (
                f"Local pack '{pack_path}' is missing a valid pyproject.toml project.name."
            )
        package_name = project_name.lower()
        if not any(package_name.startswith(prefix) for prefix in pack_allowed_prefixes()):
            return False, (
                f"Pack '{package_name}' rejected: allowed package prefixes are "
                f"{', '.join(pack_allowed_prefixes())}."
            )
        return True, ""

    if _is_git_pack_spec(normalized):
        host = _extract_git_host(normalized)
        if not host:
            return False, "Only git+https refs are supported for runtime git installs."
        if host not in pack_allowed_git_hosts():
            return False, f"Git host '{host}' is not in the allowlist."
        if pack_require_git_ref() and "@" not in normalized:
            return False, "Git refs must pin @tag/@branch/@sha (policy LISA_PACK_REQUIRE_GIT_REF)."
        package_name = _extract_package_name_from_spec(normalized).lower()
        if not any(package_name.startswith(prefix) for prefix in pack_allowed_prefixes()):
            return False, (
                f"Pack '{package_name}' rejected: allowed package prefixes are "
                f"{', '.join(pack_allowed_prefixes())}."
            )
        return True, ""

    package_name = _extract_package_name_from_spec(normalized).lower()
    if not package_name:
        return False, "Invalid package ref."
    if not any(package_name.startswith(prefix) for prefix in pack_allowed_prefixes()):
        return False, (
            f"Pack '{package_name}' rejected: allowed package prefixes are "
            f"{', '.join(pack_allowed_prefixes())}."
        )
    return True, ""


async def run_pack_install(
    runtime: HostRuntime,
    spec: str,
) -> tuple[bool, str]:
    """Install one pack ref using uv pip and persist provenance metadata."""
    is_allowed, reason = _validate_pack_spec_for_install(spec)
    if not is_allowed:
        return False, reason

    install_target = spec.strip().strip('"') if _is_local_pack_spec(spec) else spec
    process = await asyncio.create_subprocess_exec(
        "uv",
        "pip",
        "install",
        install_target,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_FRAMEWORK_ROOT),
    )
    output_bytes, _ = await process.communicate()
    output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""
    if process.returncode != 0:
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, f"Install failed for '{spec}'.\n{preview}"

    package_name = normalize_pack_name(_extract_package_name_from_spec(spec))
    if not package_name:
        return False, f"Install succeeded for '{spec}' but package name could not be inferred."
    source_kind = (
        "local"
        if _is_local_pack_spec(spec)
        else ("git" if _is_git_pack_spec(spec) else "package")
    )
    lock = _load_installed_packs_lock(runtime)
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    now_iso = datetime.now(UTC).isoformat()
    filtered = [
        entry
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or "")) != package_name
    ]
    filtered.append(
        {
            "package_name": package_name,
            "installed_at": now_iso,
            "source_kind": source_kind,
            "source_ref": install_target,
            "enabled": True,
            "updated_at": now_iso,
        }
    )
    lock["packs"] = filtered
    _auto_select_capability_provider_on_install(lock, package_name)
    _write_installed_packs_lock(runtime, lock)

    runtime.request_restart(reason=f"pack install: {package_name}")
    return True, (
        f"Installed '{package_name}' from {source_kind} source. "
        "Restart required to reload entry points, pack catalogs, and MCP manifests."
    )


async def run_pack_uninstall(
    runtime: HostRuntime,
    package_name: str,
) -> tuple[bool, str]:
    """Uninstall one pack and update provenance metadata."""
    requested = package_name.strip()
    if not requested:
        return False, "Usage: /pack uninstall <package-name>"

    lock = _load_installed_packs_lock(runtime)
    normalized = resolve_pack_name(requested, lock) or normalize_pack_name(requested)
    if not normalized:
        return False, "Usage: /pack uninstall <package-name>"

    process = await asyncio.create_subprocess_exec(
        "uv",
        "pip",
        "uninstall",
        "-y",
        normalized,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(_FRAMEWORK_ROOT),
    )
    output_bytes, _ = await process.communicate()
    output = output_bytes.decode("utf-8", errors="replace") if output_bytes else ""
    if process.returncode != 0:
        preview = "\n".join(output.splitlines()[-12:]).strip()
        return False, f"Uninstall failed for '{normalized}'.\n{preview}"

    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    filtered = [
        entry
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or "")) != normalized
    ]
    lock["packs"] = filtered
    cleared = _clear_selected_provider_for_package(lock, normalized)
    _write_installed_packs_lock(runtime, lock)

    runtime.request_restart(reason=f"pack uninstall: {normalized}")
    if cleared:
        cleared_text = ", ".join(sorted(cleared))
        return True, (
            f"Uninstalled '{normalized}' and cleared provider selections for: {cleared_text}. "
            "Restart required to unload entry points, pack catalogs, and MCP manifests."
        )
    return True, (
        f"Uninstalled '{normalized}'. "
        "Restart required to unload entry points, pack catalogs, and MCP manifests."
    )


def escape_markdown_cell(value: str) -> str:
    """Escape a string for safe use inside a Markdown pipe-table cell."""
    return value.replace("|", "\\|").replace("\n", " ")


def resolve_pack_source_kind(
    name: str,
    entry: dict[str, Any],
    always_on: set[str],
    runtime_installed: set[str],
) -> str:
    """Resolve the source-kind label for a pack entry."""
    source_kind = str(entry.get("source_kind") or "").strip().lower()
    if source_kind:
        return source_kind
    if name in always_on:
        return "always-on"
    if name in runtime_installed:
        return "package"
    return "bundled"


def render_pack_list(runtime: HostRuntime) -> str:
    """Render discovered packs, activation state, and capability-provider selections."""
    lock = _load_installed_packs_lock(runtime)
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    installed_roots = installed_pack_roots_by_name()
    always_on = always_on_pack_names(
        runtime.config.mcp_manifest_path,
        always_on_packs=runtime.config.always_on_packs,
    )
    rt_installed = runtime_installed_pack_names(lock)
    enabled_map = runtime_enabled_pack_map(lock)
    entry_map = {
        normalize_pack_name(str(entry.get("package_name") or "")): entry
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or ""))
    }

    all_packages = set(installed_roots) | set(entry_map) | always_on
    if not all_packages:
        return "No packs discovered yet."

    lines = ["## Pack Activation", ""]
    lines.append("| Pack | State | Source | Root |")
    lines.append("| --- | --- | --- | --- |")
    for package_name in sorted(all_packages):
        entry = entry_map.get(package_name) or {}
        source_kind = resolve_pack_source_kind(
            package_name,
            entry,
            always_on,
            rt_installed,
        )
        active = pack_is_active(
            package_name,
            always_on_packs=always_on,
            runtime_installed=rt_installed,
            enabled_map=enabled_map,
        )
        if package_name in always_on:
            activation = "always-on"
        elif active:
            activation = "enabled"
        else:
            activation = "disabled"
        root = installed_roots.get(package_name)
        root_text = str(root) if root is not None else "(not discovered in current environment)"
        lines.append(
            f"| {escape_markdown_cell(package_name)} | {escape_markdown_cell(activation)} | "
            f"{escape_markdown_cell(source_kind)} | {escape_markdown_cell(root_text)} |"
        )

    selected = lock.get("selected_capability_providers")
    if isinstance(selected, dict) and selected:
        lines.append("")
        lines.append("## Selected Capability Providers")
        lines.append("")
        lines.append("| Capability | Provider |")
        lines.append("| --- | --- |")
        for capability_id, provider_id in sorted(selected.items()):
            cap = escape_markdown_cell(str(capability_id))
            prov = escape_markdown_cell(str(provider_id))
            lines.append(f"| {cap} | {prov} |")

    lines.append("")
    lines.append("Tip: use /pack enable <name> or /pack disable <name> for bundled/runtime packs.")
    return "\n".join(lines)


async def handle_pack_command(  # noqa: C901, PLR0911, PLR0912
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/pack`` commands for runtime pack management."""
    command_text = stripped[5:].strip()
    if not command_text:
        return handle_simple_command(
            runtime,
            message,
            "pack",
            "Usage: /pack list | /pack enable <pack> | /pack disable <pack> "
            "| /pack install <spec> | /pack uninstall <package>",
        )

    verb, _, remainder = command_text.partition(" ")
    action = verb.strip().lower()
    arg = remainder.strip()

    resume_message = ""
    if action in {"install", "uninstall"} and " --resume " in arg:
        arg, _, maybe_resume = arg.partition(" --resume ")
        resume_message = maybe_resume.strip()

    if action == "list":
        return handle_simple_command(runtime, message, "pack", render_pack_list(runtime))

    if action == "enable":
        if not arg:
            return handle_simple_command(runtime, message, "pack", "Usage: /pack enable <pack>")
        ok, details = set_pack_enabled(runtime, arg, enabled=True)
        return handle_simple_command(runtime, message, "pack", details)

    if action == "disable":
        if not arg:
            return handle_simple_command(runtime, message, "pack", "Usage: /pack disable <pack>")
        ok, details = set_pack_enabled(runtime, arg, enabled=False)
        return handle_simple_command(runtime, message, "pack", details)

    if action == "install":
        ok, details = await run_pack_install(runtime, arg)
        if not ok:
            return handle_simple_command(runtime, message, "pack", details)
        if resume_message:
            runtime.request_restart(
                reason=runtime.restart_reason or "pack install",
            )
            os.environ["LISA_RESTART_RESUME_MESSAGE"] = resume_message
        return handle_simple_command(runtime, message, "pack", details)

    if action == "uninstall":
        ok, details = await run_pack_uninstall(runtime, arg)
        if not ok:
            return handle_simple_command(runtime, message, "pack", details)
        if resume_message:
            runtime.request_restart(
                reason=runtime.restart_reason or "pack uninstall",
            )
            os.environ["LISA_RESTART_RESUME_MESSAGE"] = resume_message
        return handle_simple_command(runtime, message, "pack", details)

    return handle_simple_command(
        runtime,
        message,
        "pack",
        "Unknown /pack action. Supported: list, enable, disable, install, uninstall.",
    )