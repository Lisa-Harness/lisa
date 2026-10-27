"""Capability snapshot, debug renderers, runtime settings, help text, token stats.

Pure rendering helpers that read runtime state and emit Markdown strings.
No command dispatch lives here — the ``handle_*`` functions in ``_simple``
and ``_slice`` call these renderers.

Dependency direction: ``_render`` → ``_log`` (``_last_capability_manifest``),
``_routing`` (``mode_support_gap``), ``kernel.*``, ``runtime.*``.
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
from kernel.workflow_view import WorkflowView, render_workflow_progress

from runtime.commands._log import _last_capability_manifest
from runtime.commands._routing import mode_support_gap
from runtime.host_commands import is_host_command_query
from runtime.runtime import HostRuntime
from runtime.thread import load_thread_range, next_turn_index
from runtime.token_accounting import format_session_exit_summary_lines

logger = logging.getLogger(__name__)


# ── Capability snapshot ─────────────────────────────────────────────


def build_capability_snapshot(runtime: HostRuntime) -> dict[str, Any]:
    """Build a capability snapshot dict for diagnostics and help rendering."""
    catalog = get_workflow_catalog()
    manifest = runtime.session.manifest
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
                "modes": sorted(
                    mode_id
                    for mode_id, model_ref in manifest.mode_models.items()
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
        f"- Unsupported installed modes: {unsupported_count}",
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


def render_debug_turn(runtime: HostRuntime, arg: str) -> str:
    """Render the complete, unedited raw model input for turn ``arg``.

    ``arg`` is the stripped argument after ``/debug`` — empty for the
    default (the most recently completed turn), or a non-negative integer
    turn index. Reads the thread log directly (the durable source of
    truth) rather than in-memory runtime state, so this works identically
    right after a restart.
    """
    arg = arg.strip()
    if arg:
        if not arg.isdigit():
            return "Usage: /debug [turn_number]"
        turn_index = int(arg)
    else:
        turn_index = next_turn_index(thread_path=runtime.paths.thread_path) - 1
        if turn_index < 0:
            return "No completed turns yet."

    entries = load_thread_range(
        thread_path=runtime.paths.thread_path,
        start_turn=turn_index,
        end_turn=turn_index,
    )
    if not entries:
        return f"Turn {turn_index} not found in the thread log."
    entry = entries[0]

    header = [
        f"## Turn {turn_index}",
        f"- timestamp_utc: {entry.get('timestamp_utc', '')}",
        f"- mode_id: {entry.get('mode_id', '')}",
        f"- query: {entry.get('query', '')}",
    ]

    turn_input = entry.get("turn_input")
    if not isinstance(turn_input, dict) or not turn_input.get("calls"):
        header.append(
            "\nNo captured model input for this turn (host-command turn, or the "
            "turn predates raw-input capture)."
        )
        return "\n".join(header)

    header.append("\n```json")
    header.append(json.dumps(turn_input["calls"], indent=2, ensure_ascii=False))
    header.append("```")
    return "\n".join(header)


# How many prior entries the default (no-argument) /last walk scans before
# giving up with the friendly message. Host-command turns (``/settings``,
# a previous ``/last``, ...) are skipped, so consecutive shortcut use still
# finds the last model response without unbounded segment reads.
_LAST_LOOKBACK = 8


def resolve_last_turn(runtime: HostRuntime, arg: str) -> tuple[int | None, str]:
    """Resolve a ``/last`` request to ``(turn_index, response_text)``.

    ``arg`` is the stripped argument — empty for the default (the previous
    model turn's response) or a non-negative integer turn index. The turn
    index is the global thread number of the replayed entry — the same
    number the prompt surfaces render — so the operator can address the
    replayed turn without counting. ``None`` means the result is a
    diagnostic (usage error, not-found turn, no-completed-turns, or a
    default walk with no prior model response), not a replayed response.

    Reads the durable thread directly (the same source of truth as
    :func:`render_debug_turn`), so resolution works identically after a
    restart.
    """
    arg = arg.strip()
    if arg:
        if not arg.isdigit():
            return None, "Usage: /last [turn_number]"
        return _resolve_explicit_turn(runtime, int(arg))
    return _resolve_default_turn(runtime)


def _resolve_explicit_turn(runtime: HostRuntime, turn_index: int) -> tuple[int | None, str]:
    """Load one explicit turn's response, verbatim, or a not-found/empty diagnostic."""
    entries = load_thread_range(
        thread_path=runtime.paths.thread_path,
        start_turn=turn_index,
        end_turn=turn_index,
    )
    if not entries:
        return None, f"Turn {turn_index} not found in the thread log."
    response_text = str(entries[0].get("response_text") or "")
    if not response_text:
        return None, f"Turn {turn_index} has no response text."
    return turn_index, response_text


def _resolve_default_turn(runtime: HostRuntime) -> tuple[int | None, str]:
    """Walk the recent window back to the last model response.

    The walk skips host-command turns and empty responses so ``/last``
    after ``/settings`` (or another ``/last``) still replays the last
    *model* response. The replayed entry's own ``turn`` stamp is the
    number; entries without one fall back to their positional index.
    """
    next_turn = next_turn_index(thread_path=runtime.paths.thread_path)
    if next_turn <= 0:
        return None, "No completed turns yet."
    start_turn = max(0, next_turn - _LAST_LOOKBACK)
    entries = load_thread_range(
        thread_path=runtime.paths.thread_path,
        start_turn=start_turn,
        end_turn=next_turn - 1,
    )
    for offset, entry in enumerate(reversed(entries)):
        if is_host_command_query(str(entry.get("query") or "")):
            continue
        response_text = str(entry.get("response_text") or "")
        if response_text:
            raw_turn = entry.get("turn")
            if isinstance(raw_turn, int) and raw_turn >= 0:
                return raw_turn, response_text
            return start_turn + (len(entries) - 1 - offset), response_text
    return None, "No prior model response in the recent window."


def render_last_turn(runtime: HostRuntime, arg: str) -> str:
    """Return a previous turn's response text, verbatim (no chrome).

    ``arg`` is the stripped argument after ``/last`` — empty for the default
    (the previous model turn's response) or a non-negative integer turn
    index. Verbatim companion of :func:`resolve_last_turn` (which also
    reports the replayed turn index): transports that need the number call
    that instead; this stays byte-identical to the response for piped/
    machine-readable consumers.
    """
    return resolve_last_turn(runtime, arg)[1]


def render_workflow_status(view: WorkflowView) -> str:
    """Render the active workflow for ``/workflow status``.

    Reuses the same progress block the model receives in its turn briefing
    (so the operator sees exactly what steers routing) and appends the action
    that retires it. A completed-but-unretired view cannot reach here —
    ``derive_workflow_view`` returns ``None`` once every step is complete.
    """
    progress = render_workflow_progress(view)
    if not progress:
        return "## Workflow\n\nThe workflow has no steps to report."
    completed = len(view.completed_steps)
    total = len(view.declared_steps)
    lines = [
        "## Workflow",
        "",
        progress,
        "",
        f"**{completed} of {total} step(s) complete.**",
        "",
        "Conclude this workflow with `/workflow conclude [reason]` — the plan "
        "stops being injected into turn briefings and stops steering routing. "
        "The conclusion is recorded in the thread.",
    ]
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
    """Return on-demand detailed token statistics as Markdown.

    Combines the in-memory session roll-up (turn/model/mode/stage) with a
    lifetime view read from the durable ``token-usage.jsonl`` telemetry log.
    The lifetime section aggregates every persisted event across all
    sessions for this host's data directory.
    """
    state = dataclasses.asdict(runtime.session.tokens)
    lines = format_session_exit_summary_lines(
        state,
        session_started_utc=runtime.session.tokens.session_started_utc,
    )
    lifetime_lines = _format_lifetime_token_usage(runtime.paths.token_usage_path)
    if lifetime_lines:
        lines.extend(lifetime_lines)
    return "\n".join(lines)


def _aggregate_lifetime_usage(
    lines: list[str],
) -> dict[str, Any] | None:
    """Aggregate the parsed token-usage JSONL lines into lifetime totals.

    Returns:
        A dict of aggregate totals (prompt/completion/total/cached/reasoning/
        cache_creation tokens, cost, event count, first/last timestamps,
        and the per-model breakdown), or ``None`` when no parsable event
        lines were found.
    """
    prompt = 0
    completion = 0
    total = 0
    cached = 0
    reasoning = 0
    cost = 0.0
    event_count = 0
    first_at = ""
    last_at = ""
    model_totals: dict[str, dict[str, int]] = {}
    cache_creation = 0

    for raw_line in lines:
        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        event_count += 1
        rec_prompt = int(record.get("prompt_tokens") or 0)
        rec_completion = int(record.get("completion_tokens") or 0)
        rec_total = int(record.get("total_tokens") or 0)
        rec_cached = int(record.get("cached_tokens") or 0)
        rec_reasoning = int(record.get("reasoning_tokens") or 0)
        rec_cache_creation = int(record.get("cache_creation_tokens") or 0)
        rec_cost = float(record.get("cost_usd") or 0.0)
        prompt += rec_prompt
        completion += rec_completion
        total += rec_total
        cached += rec_cached
        reasoning += rec_reasoning
        cache_creation += rec_cache_creation
        cost += rec_cost
        emitted_at = str(record.get("emitted_at") or "")
        if emitted_at:
            if not first_at or emitted_at < first_at:
                first_at = emitted_at
            if not last_at or emitted_at > last_at:
                last_at = emitted_at
        model_id = str(record.get("model_id") or "")
        if model_id:
            entry = model_totals.setdefault(
                model_id,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "cache_creation_tokens": 0,
                },
            )
            entry["prompt_tokens"] += rec_prompt
            entry["completion_tokens"] += rec_completion
            entry["total_tokens"] += rec_total
            entry["cached_tokens"] += rec_cached
            entry["reasoning_tokens"] += rec_reasoning
            entry["cache_creation_tokens"] += rec_cache_creation

    if event_count == 0:
        return None
    return {
        "prompt": prompt,
        "completion": completion,
        "total": total,
        "cached": cached,
        "reasoning": reasoning,
        "cache_creation": cache_creation,
        "cost": cost,
        "event_count": event_count,
        "first_at": first_at,
        "last_at": last_at,
        "model_totals": model_totals,
    }


def _render_lifetime_usage_lines(agg: dict[str, Any]) -> list[str]:
    """Render the aggregated lifetime totals as the Markdown section lines."""
    prompt = agg["prompt"]
    completion = agg["completion"]
    total = agg["total"]
    cached = agg["cached"]
    reasoning = agg["reasoning"]
    cache_creation = agg["cache_creation"]
    cost = agg["cost"]
    event_count = agg["event_count"]
    first_at = agg["first_at"]
    last_at = agg["last_at"]
    model_totals: dict[str, dict[str, int]] = agg["model_totals"]

    billed = max(0, total - cached)
    # Cache-read share: tokens served from cache as a fraction of everything
    # the provider *input* (prompt + cache reads). Zero-denominator guard so
    # an all-empty record set renders 0.0 rather than raising.
    cache_read_share = cached / (prompt + cached) if (prompt + cached) > 0 else 0.0
    out: list[str] = [
        "",
        "### Lifetime (all sessions)",
        f"- Events: {event_count}",
        f"- Span: {first_at or 'n/a'} → {last_at or 'n/a'}",
        f"- Estimated cost: ${cost:.4f}",
        f"- Prompt cache: {cached:,} read ({cache_read_share:.1%} of input), "
        f"{cache_creation:,} created",
        "| Prompt | Completion | Total | Cached | Reasoning | Billed (total-cached) |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {prompt} | {completion} | {total} | {cached} | {reasoning} | {billed} |",
    ]

    if model_totals:
        out.extend(
            [
                "",
                "#### Lifetime by Model",
                "| Model | Prompt | Completion | Total | Cached | Reasoning | "
                "Billed (total-cached) |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for model_key in sorted(model_totals):
            entry = model_totals[model_key]
            model_total = entry["total_tokens"]
            model_cached = entry["cached_tokens"]
            out.append(
                f"| {model_key} | {entry['prompt_tokens']} | "
                f"{entry['completion_tokens']} | {model_total} | "
                f"{model_cached} | {entry['reasoning_tokens']} | "
                f"{max(0, model_total - model_cached)} |"
            )

    return out


def _format_lifetime_token_usage(token_usage_path: Any) -> list[str]:
    """Read the full ``token-usage.jsonl`` and return a lifetime Markdown section.

    Aggregates every persisted ``TokenUsageEvent`` (across all sessions) into
    prompt / completion / total / cached / reasoning / billed totals, a
    per-model breakdown, total estimated cost, event count, and the
    first/last event timestamps. Returns an empty list when the log is
    missing or empty so ``/tokens`` degrades gracefully to the session view.
    """
    if token_usage_path is None:
        return []
    try:
        path = token_usage_path
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if not lines:
        return []

    agg = _aggregate_lifetime_usage(lines)
    if agg is None:
        return []
    return _render_lifetime_usage_lines(agg)


# ── Help ─────────────────────────────────────────────────────────────


def _pack_command_entries() -> list[Any]:
    """Return pack-registered host command entries for help rendering.

    Mirrors the lazy discovery in :func:`runtime.handle_message._pack_host_commands`
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
                summary = f"pack command from {entry.pack_name}"
            lines.append(f"- {spec.prefix} — {summary}")
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
    lines: list[str] = ["Lisa Harness Host", ""]
    if runtime is not None:
        mode_id = runtime.session.current_mode_id or "default"
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
            "|update <pack>|update --all manages pack activation and runtime installs "
            "(changes are active on the next launch).",
            "- /models show|discover|apply|reload|edit — manage the active model manifest "
            "(run /help models for switches).",
            "- /modes — list every discovered mode (supported and unsupported).",
            "- /operations — list internal operation IDs for allowlist authoring.",
            "- /tools — list internal (pack-discovered) tools with limits and timeouts.",
            "- /mcp [server_id] — MCP server health summary, or per-server tool detail.",
            "- /proxy — provider proxy health and routing summary.",
            "- /settings — effective profile/timezone/debug/log-level runtime settings.",
            "- /policy [mode ask|trust|plan | bare on|off | rule add|remove|list] — "
            "show or change the approval "
            "policy (the decision dial and rules behind approval prompts).",
            "- /tokens — session token breakdown by model, mode, and stage.",
            "- /debug context|capabilities|self — local diagnostics.",
            "- /workflow [status|conclude [reason]] — show the active multi-step "
            "workflow and its step progress, or conclude it durably so it stops "
            "steering routing.",
            "- /last [N] — reprint a previous response verbatim (default: the last "
            "model turn; no model call).",
            "- /consolidate (or /slice) — explicit memory-consolidation slice.",
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
    if topic == "models":
        # Lazy import: _models imports _simple which imports _render — a
        # module-level import here would create a cycle.
        from runtime.commands._models import _models_help_text

        return _models_help_text(runtime)
    if topic in {"pack", "mcp", "proxy"}:
        return _BUILTIN_HELP_RENDERERS[topic]()
    if topic in {"modes", "operations", "tools"}:
        return _inventory_command_help_text(topic)
    if topic in _HOST_SHORTCUT_TOPICS:
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
                    summary = f"pack command from {entry.pack_name}"
                return (
                    f"## /{topic}\n\n"
                    f"- {spec.prefix} — {summary}\n\n"
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
            "    Append ` --resume <message>` to replay a turn after the next launch.",
            "- /pack uninstall <package> — remove an installed pack.",
            "- /pack update <pack> — reinstall an installed pack in place from its "
            "recorded source.",
            "- /pack update --all — reinstall every runtime-installed pack from its "
            "recorded source.",
            "",
            "Activation changes take effect on the next launch; install/uninstall "
            "reseed the packs venv and modes, operations, and commands are "
            "re-discovered on the next boot.",
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


def _proxy_command_help_text() -> str:
    """Help text for ``/help proxy``."""
    return "\n".join(
        [
            "## /proxy",
            "Provider proxy health and routing readout from the live proxy pool.",
            "",
            "Usage:",
            "- /proxy — per-proxy summary (status, protocol, providers, endpoint, "
            "pack-declared live stats) plus the provider routing map (which providers "
            "route through a proxy and which go direct).",
            "",
            "Proxies are declared by packs or the user in the MCP manifest's "
            "`proxies:` block. A proxy that declares a `status:` block exposes live "
            "counters (e.g. tokens saved); liveness-only proxies show `warm`/`failed` "
            "without stats.",
        ]
    )


def _inventory_command_help_text(topic: str) -> str:
    """Help text for ``/help modes`` / ``/operations`` / ``/tools``."""
    return "\n".join(
        [
            f"## /{topic}",
            f"List the live {topic} catalog discovered from installed packs and SDK_CONTENT_ROOTS.",
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
        "policy": (
            "## /policy\n\n"
            "Show or change the approval policy. Bare /policy renders the active "
            "document: the mode dial, sandbox downgrade, bare-mode guard, the "
            "allow/ask/deny rule lists, and any load errors.\n\n"
            "- /policy mode ask|trust|plan — set the fallback dial. trust "
            "auto-allows write/execute actions (external_write still prompts, "
            "and the bare-mode guard still prompts unsandboxed execute); plan "
            "denies every effectful action; ask prompts for anything not settled "
            "by a rule or the sandbox.\n"
            "- /policy bare on|off — toggle bare_requires_approval: when on, an "
            "execute action outside the sandbox always prompts even under trust.\n"
            "- /policy rule add <allow|ask|deny> [--tool T] [--effect E] "
            "[--pack P] [--prefix PX] [--when EXPR] — append a rule. The when "
            "expression is a CEL boolean over action.* (tool, effect, pack, "
            "command, argv, parameters), identity.*, and containment; it is "
            "compile-checked before the write.\n"
            "- /policy rule remove <list>:<n> — drop a rule by its address.\n"
            "- /policy rule list — render the rules with their addresses.\n\n"
            "Changes take effect immediately — the engine re-reads policy.yaml "
            "on every approval request."
        ),
        "tokens": (
            "## /tokens\n\n"
            "Show the session token breakdown by model, mode, and stage, plus the "
            "running session-exit summary."
        ),
        "debug": (
            "## /debug\n\n"
            "Show the complete, unedited raw model input for turn N "
            "(default: the previous turn): `/debug [N]`."
        ),
        "last": (
            "## /last\n\n"
            "Reprint a previous response, verbatim. `/last` → the previous "
            "model turn's response (host-command turns are skipped); "
            "`/last N` → turn N's response. Reads the durable thread — no "
            "model call, no context cost, works after restart."
        ),
        "consolidate": (
            "## /consolidate (or /slice)\n\n"
            "Run the explicit slice in-process: the memory pack's consolidation "
            "slice distills episodic records and mines sealed thread segments "
            "into durable memory. History is never compacted — the thread is lossless."
        ),
        "slice": (
            "## /slice (or /consolidate)\n\n"
            "Alias for /consolidate: run the explicit memory-consolidation slice."
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


# Resolved after the renderer definitions above (module-level tables
# evaluate at import, so function names must already be bound).
_BUILTIN_HELP_RENDERERS = {
    "pack": _pack_command_help_text,
    "mcp": _mcp_command_help_text,
    "proxy": _proxy_command_help_text,
}

_HOST_SHORTCUT_TOPICS = frozenset(
    {
        "help",
        "settings",
        "tokens",
        "debug",
        "last",
        "consolidate",
        "slice",
        "intake",
        "quit",
        "policy",
    }
)
