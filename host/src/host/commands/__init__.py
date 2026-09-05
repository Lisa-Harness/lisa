"""Transport-neutral host command dispatch for Lisa framework hosts.

This package consolidates the slash-command handlers that do not require the
model: ``/help``, ``/settings``, ``/tokens``, ``/debug``, ``/consolidate``
(``/slice``), ``/intake``, ``/pack``, ``/models``, ``/modes``, ``/operations``, ``/tools``.
All functions produce plain text output (Markdown strings) — no Rich
panels, no console printing, no transport-specific rendering.

Each command handler accepts a :class:`HostRuntime` (the typed replacement
for the old ``_RUNTIME_STATE`` dict) and optionally a
:class:`TransportCallbacks` for log/thread operations. The runtime owns all
mutable state; the commands read from and write to it through its typed
accessors.

Dependency direction
--------------------
``commands`` ← ``handle_message.py`` / transport hosts  (callers import
from commands; commands never import handle_message or any transport code)

``commands`` → ``host.thread``, ``host.content_roots``,
``host.token_accounting``, ``host.model_runtime``,
``host.state_provider``, ``kernel.*``  (commands imports these;
none of them import commands)

Public surface
---------------
- :func:`handle_simple_command`
- :func:`handle_help_command`
- :func:`handle_debug_command`
- :func:`run_explicit_slice_command`
- :func:`handle_intake_command`
- :func:`handle_pack_command`
- :func:`handle_models_command`
- :func:`format_help`
- :func:`render_capability_help`
- :func:`render_debug_turn`
- :func:`render_runtime_settings`
- :func:`format_token_stats`
- :func:`build_capability_snapshot`
- :func:`predict_model`
- :func:`set_runtime_route_preview`
- :func:`preferred_model_id_for_route`
- :func:`model_alias_for_mode`
- :func:`render_pack_list`
- :func:`render_modes_list`
- :func:`render_operations_list`
- :func:`render_tools_list`
- :func:`mode_support_gap`
- :func:`format_unsupported_mode`
- :func:`format_operation_error`

The private helpers tests reach for (``_resolve_model``,
``_parse_models_subcommand``, ``_render_discover_notes``, the pack spec
helpers, etc.) are also re-exported here so existing
``from host.commands import X`` and ``from host import commands``
usages keep working without edits.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from host.commands._lists import (
    escape_markdown_cell,
    render_modes_list,
    render_operations_list,
    render_pack_list,
    render_tools_list,
    resolve_pack_source_kind,
)
from host.commands._log import (
    _append_log_event,
    _append_thread_entry,
    _append_turn_log,
    _last_capability_manifest,
    _resolve_int_env,
    _resolve_model,
    _set_turn_context,
)
from host.commands._mcp import (
    handle_mcp_command,
    render_mcp_server_detail,
    render_mcp_summary,
)
from host.commands._models import (
    _handle_models_apply,
    _handle_models_discover,
    _models_help_text,
    _parse_models_subcommand,
    _render_discover_notes,
    _render_models_manifest,
    handle_models_command,
)
from host.commands._packs import (
    _DEFAULT_PACK_GIT_ROOT,
    _FRAMEWORK_ROOT,
    LISA_PACK_ALLOWED_PREFIXES_ENV,
    LISA_PACK_GIT_ALLOW_HOSTS_ENV,
    LISA_PACK_GIT_ROOT_ENV,
    LISA_PACK_REQUIRE_GIT_REF_ENV,
    _auto_select_capability_provider_on_install,
    _clear_selected_provider_for_package,
    _ensure_packs_venv,
    _expand_bare_pack_name,
    _extract_git_host,
    _extract_package_name_from_spec,
    _is_bare_pack_name,
    _is_git_pack_spec,
    _is_local_pack_spec,
    _load_installed_packs_lock,
    _normalize_pack_install_spec,
    _pack_contributes_memory_capability,
    _read_local_pack_project_name,
    _seed_packs_venv_with_framework,
    _validate_pack_spec_for_install,
    _write_installed_packs_lock,
    installed_pack_roots_by_name,
    pack_alias_map,
    pack_allowed_git_hosts,
    pack_allowed_prefixes,
    pack_git_root_template,
    pack_require_git_ref,
    resolve_pack_name,
    run_pack_install,
    run_pack_uninstall,
    run_pack_update,
    run_pack_update_all,
    set_pack_enabled,
)
from host.commands._render import (
    build_capability_snapshot,
    format_help,
    format_token_stats,
    render_capability_help,
    render_command_help,
    render_debug_turn,
    render_runtime_settings,
)
from host.commands._routing import (
    format_operation_error,
    format_unsupported_mode,
    mode_support_gap,
    model_alias_for_mode,
    predict_model,
    preferred_model_id_for_route,
    set_runtime_route_preview,
)
from host.commands._simple import (
    handle_debug_command,
    handle_help_command,
    handle_simple_command,
)
from host.commands._slice import (
    handle_intake_command,
    run_explicit_slice_command,
)
from host.runtime import HostRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "LISA_PACK_ALLOWED_PREFIXES_ENV",
    "LISA_PACK_GIT_ALLOW_HOSTS_ENV",
    "LISA_PACK_GIT_ROOT_ENV",
    "LISA_PACK_REQUIRE_GIT_REF_ENV",
    "_DEFAULT_PACK_GIT_ROOT",
    "_FRAMEWORK_ROOT",
    "_append_log_event",
    "_append_thread_entry",
    "_append_turn_log",
    "_auto_select_capability_provider_on_install",
    "_clear_selected_provider_for_package",
    "_ensure_packs_venv",
    "_expand_bare_pack_name",
    "_extract_git_host",
    "_extract_package_name_from_spec",
    "_handle_models_apply",
    "_handle_models_discover",
    "_is_bare_pack_name",
    "_is_git_pack_spec",
    "_is_local_pack_spec",
    "_last_capability_manifest",
    "_load_installed_packs_lock",
    "_models_help_text",
    "_normalize_pack_install_spec",
    "_pack_contributes_memory_capability",
    "_parse_models_subcommand",
    "_read_local_pack_project_name",
    "_render_discover_notes",
    "_render_models_manifest",
    "_resolve_int_env",
    "_resolve_model",
    "_seed_packs_venv_with_framework",
    "_set_turn_context",
    "_validate_pack_spec_for_install",
    "_write_installed_packs_lock",
    "build_capability_snapshot",
    "discover_models",
    "escape_markdown_cell",
    "format_help",
    "format_operation_error",
    "format_token_stats",
    "format_unsupported_mode",
    "handle_debug_command",
    "handle_help_command",
    "handle_intake_command",
    "handle_mcp_command",
    "handle_models_command",
    "handle_pack_command",
    "handle_simple_command",
    "installed_pack_roots_by_name",
    "mode_support_gap",
    "model_alias_for_mode",
    "pack_alias_map",
    "pack_allowed_git_hosts",
    "pack_allowed_prefixes",
    "pack_git_root_template",
    "pack_require_git_ref",
    "predict_model",
    "preferred_model_id_for_route",
    "render_capability_help",
    "render_command_help",
    "render_debug_turn",
    "render_mcp_server_detail",
    "render_mcp_summary",
    "render_modes_list",
    "render_operations_list",
    "render_pack_list",
    "render_runtime_settings",
    "render_tools_list",
    "resolve_pack_name",
    "resolve_pack_source_kind",
    "run_explicit_slice_command",
    "run_pack_install",
    "run_pack_uninstall",
    "run_pack_update",
    "run_pack_update_all",
    "set_pack_enabled",
    "set_runtime_route_preview",
]


# Lazy attribute table for symbols that must remain monkeypatchable.
#
# ``host/tests/test_core.py:522`` does
# ``monkeypatch.setattr("host.commands.discover_models", ...)``, so
# ``discover_models`` must resolve to the real import when unpatched but
# yield to the monkeypatch when set. Exposing these as module-level imports
# would freeze the reference; exposing them via ``__getattr__`` lets
# ``setattr`` on this module shadow each entry. Each value is a zero-arg
# callable that performs the lazy import on first access.
def _import_discover_models() -> object:
    from host.model_discovery import discover_models  # noqa: PLC0415

    return discover_models


def _import_build_manifest_payload() -> object:
    from host.model_discovery import build_manifest_payload  # noqa: PLC0415

    return build_manifest_payload


def _import_render_manifest_yaml() -> object:
    from host.model_discovery import render_manifest_yaml  # noqa: PLC0415

    return render_manifest_yaml


def _import_write_manifest_yaml() -> object:
    from host.model_discovery import write_manifest_yaml  # noqa: PLC0415

    return write_manifest_yaml


def _import_router_preset_annotations() -> object:
    from host.model_discovery.manifest import (  # noqa: PLC0415
        _router_preset_annotations,
    )

    return _router_preset_annotations


def _import_load_model_manifest() -> object:
    from host.model_manifest import load_model_manifest  # noqa: PLC0415

    return load_model_manifest


def _import_active_manifest_path() -> object:
    from host.model_runtime import active_manifest_path  # noqa: PLC0415

    return active_manifest_path


def _import_proposal_manifest_path() -> object:
    from host.model_runtime import proposal_manifest_path  # noqa: PLC0415

    return proposal_manifest_path


def _import_reload_manifest() -> object:
    from host.model_runtime import reload_manifest  # noqa: PLC0415

    return reload_manifest


def _import_load_manifest() -> object:
    from host.model_runtime import load_manifest  # noqa: PLC0415

    return load_manifest


def _import_resolve_little_lisa_model() -> object:
    from host.model_runtime import resolve_model  # noqa: PLC0415

    return resolve_model


def _import_stub_model_definition() -> object:
    from host.model_runtime import StubModelDefinition  # noqa: PLC0415

    return StubModelDefinition


def _import_discovery_report() -> object:
    from host.model_discovery import DiscoveryReport  # noqa: PLC0415

    return DiscoveryReport


def _import_probe_status() -> object:
    from host.model_discovery import ProbeStatus  # noqa: PLC0415

    return ProbeStatus


def _import_resolve_slice_tuning() -> object:
    from host.lisa_config import _resolve_slice_tuning  # noqa: PLC0415

    return _resolve_slice_tuning


def _import_file_backed_state_provider() -> object:
    from host.state_provider import FileBackedStateProvider  # noqa: PLC0415

    return FileBackedStateProvider


def _import_format_session_exit_summary_lines() -> object:
    from host.token_accounting import (  # noqa: PLC0415
        format_session_exit_summary_lines,
    )

    return format_session_exit_summary_lines


def _import_append_thread_entry() -> object:
    from host.thread import append_thread_entry  # noqa: PLC0415

    return append_thread_entry


def _import_load_recent_thread_entries() -> object:
    from host.thread import load_recent_thread_entries  # noqa: PLC0415

    return load_recent_thread_entries


_LAZY_IMPORTS: dict[str, Callable[[], object]] = {
    "DiscoveryReport": _import_discovery_report,
    "FileBackedStateProvider": _import_file_backed_state_provider,
    "ProbeStatus": _import_probe_status,
    "StubModelDefinition": _import_stub_model_definition,
    "_resolve_slice_tuning": _import_resolve_slice_tuning,
    "_router_preset_annotations": _import_router_preset_annotations,
    "active_manifest_path": _import_active_manifest_path,
    "append_thread_entry": _import_append_thread_entry,
    "build_manifest_payload": _import_build_manifest_payload,
    "discover_models": _import_discover_models,
    "format_session_exit_summary_lines": _import_format_session_exit_summary_lines,
    "load_manifest": _import_load_manifest,
    "load_model_manifest": _import_load_model_manifest,
    "load_recent_thread_entries": _import_load_recent_thread_entries,
    "proposal_manifest_path": _import_proposal_manifest_path,
    "reload_manifest": _import_reload_manifest,
    "render_manifest_yaml": _import_render_manifest_yaml,
    "resolve_little_lisa_model": _import_resolve_little_lisa_model,
    "write_manifest_yaml": _import_write_manifest_yaml,
}


def __getattr__(name: str) -> object:
    """Lazy attribute access for symbols that must remain monkeypatchable.

    See ``_LAZY_IMPORTS`` for the rationale: tests patch
    ``host.commands.discover_models`` etc., so these symbols resolve
    via this hook rather than frozen module-level imports.
    """
    factory = _LAZY_IMPORTS.get(name)
    if factory is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return factory()


# ── /pack dispatcher ─────────────────────────────────────────────────


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
            "| /pack install <spec> | /pack uninstall <package> | /pack update <pack> "
            "| /pack update --all",
        )

    verb, _, remainder = command_text.partition(" ")
    action = verb.strip().lower()
    arg = remainder.strip()

    resume_message = ""
    if action in {"install", "uninstall", "update"} and " --resume " in arg:
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

    if action == "update":
        if not arg or arg.lower() in {"help", "--help", "-h"}:
            return handle_simple_command(
                runtime,
                message,
                "pack",
                "Usage: /pack update <pack> | /pack update --all",
            )
        if arg.lower() in {"--all", "all"}:
            ok, details = await run_pack_update_all(runtime)
            if not ok:
                return handle_simple_command(runtime, message, "pack", details)
            if resume_message:
                runtime.request_restart(
                    reason=runtime.restart_reason or "pack update --all",
                )
                os.environ["LISA_RESTART_RESUME_MESSAGE"] = resume_message
            return handle_simple_command(runtime, message, "pack", details)

        ok, details = await run_pack_update(runtime, arg)
        if not ok:
            return handle_simple_command(runtime, message, "pack", details)
        if resume_message:
            runtime.request_restart(
                reason=runtime.restart_reason or "pack update",
            )
            os.environ["LISA_RESTART_RESUME_MESSAGE"] = resume_message
        return handle_simple_command(runtime, message, "pack", details)

    return handle_simple_command(
        runtime,
        message,
        "pack",
        "Unknown /pack action. Supported: list, enable, disable, install, uninstall, update.",
    )
