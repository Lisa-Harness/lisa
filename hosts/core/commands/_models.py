"""``/models`` command handlers: show/discover/apply/reload/edit/help.

The ``handle_models_command`` dispatcher was a C901=14 function with 10
return statements. It is split into per-subcommand handlers
(``_handle_models_show``, ``_handle_models_reload``, ``_handle_models_discover``,
``_handle_models_apply``, ``_handle_models_edit``) so each is under the
complexity budget, and the dispatcher is a thin ``if/elif`` router.

Dependency direction: ``_models`` → ``_simple`` (``handle_simple_command``),
``hosts.core.model_discovery``, ``hosts.core.model_manifest``,
``hosts.core.model_runtime``. No imports from ``_packs``/``_lists``/``_render``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from hosts.core.commands._simple import handle_simple_command
from hosts.core.model_discovery import (
    DiscoveryReport,
    ProbeStatus,
    build_manifest_payload,
    discover_models,
    render_manifest_yaml,
    write_manifest_yaml,
)
from hosts.core.model_manifest import load_model_manifest
from hosts.core.model_runtime import (
    active_manifest_path,
    proposal_manifest_path,
    reload_manifest,
)
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)


def _render_models_manifest(runtime: HostRuntime) -> str:
    active_path = active_manifest_path(data_dir=runtime.data_dir)
    proposal_path = proposal_manifest_path(data_dir=runtime.data_dir)

    try:
        manifest = runtime.manifest or load_model_manifest(active_path)
    except Exception as exc:  # pylint: disable=broad-except
        return (
            "No active models manifest is available yet.\n\n"
            f"- expected_path: {active_path}\n"
            "- next_step: run /models discover to generate models.proposed.yaml, "
            "then edit and run /models apply\n"
            f"- load_error: {exc}"
        )

    lines = [
        "## Models",
        f"- active_manifest: {active_path}",
        f"- proposal_manifest: {proposal_path}",
        f"- model_count: {len(manifest.models)}",
    ]
    # Invert manifest.roles so each model knows which roles it holds.
    roles_by_model: dict[str, list[str]] = {}
    for role_name, model_ref in sorted(manifest.roles.items()):
        roles_by_model.setdefault(model_ref, []).append(role_name)
    for model in manifest.models:
        roles = ", ".join(roles_by_model.get(model.model_id, [])) or "-"
        reasoning = "yes" if model.reasoning_model else "no"
        lines.append(
            f"- {model.model_id} (provider={model.provider}, reasoning={reasoning}, roles={roles})"
        )
    if manifest.mode_models:
        mode_map = ", ".join(
            f"{mode_id}->{model_id}" for mode_id, model_id in sorted(manifest.mode_models.items())
        )
        lines.append(f"- mode_models: {mode_map}")
    if manifest.roles:
        role_map = ", ".join(
            f"{role_name}->{model_id}" for role_name, model_id in sorted(manifest.roles.items())
        )
        lines.append(f"- roles: {role_map}")
    return "\n".join(lines)


def _models_help_text(runtime: HostRuntime) -> str:
    active_path = active_manifest_path(data_dir=runtime.data_dir)
    proposal_path = proposal_manifest_path(data_dir=runtime.data_dir)
    return "\n".join(
        [
            "## /models",
            "- /models show: show active models from data-dir manifest.",
            "- /models discover: probe configured providers and write a proposal manifest.",
            "- /models discover --force: overwrite an existing proposal.",
            "- /models apply: validate and promote the proposal as active manifest.",
            "- /models reload: reload active manifest from disk.",
            "",
            f"- active_path: {active_path}",
            f"- proposal_path: {proposal_path}",
        ]
    )


def _parse_models_subcommand(stripped: str) -> tuple[str, tuple[str, ...]]:
    """Tokenize ``/models ...`` into ``(subcommand, args)``.

    Tolerates any whitespace between ``/models`` and the subcommand, and
    rejects ``/modelsdiscover`` (no space) by returning ``("", ())`` which
    routes to the help branch.
    """

    parts = stripped.split()
    if not parts or parts[0].lower() != "/models":
        return "", ()
    sub = parts[1].lower() if len(parts) > 1 else ""
    args = tuple(parts[2:])
    return sub, args


def _render_discover_notes(report: DiscoveryReport) -> list[str]:
    """Render UI-safe per-provider notes from a discovery report.

    Only non-OK/non-SKIPPED results produce a note line. The safe
    messages have already been redacted at the probe layer, so no raw
    exception text (and no API key) can reach the UI through this path.
    """

    lines: list[str] = []
    for result in report.results:
        note = result.note_line()
        if note is not None:
            lines.append(f"- {note}")
    return lines


# ── Per-subcommand handlers (split from the old monolithic dispatcher) ──


async def _handle_models_discover(
    runtime: HostRuntime,
    message: str,
    args: tuple[str, ...],
    proposal_path: Path,
) -> str:
    """Handle ``/models discover [--force]``."""
    force = "--force" in args
    if proposal_path.exists() and not force:
        response_text = "\n".join(
            [
                f"Proposal already exists at {proposal_path}.",
                "Use /models discover --force to overwrite (your edits will be lost).",
            ]
        )
        return handle_simple_command(runtime, message, "models", response_text)

    report = await discover_models()
    payload = build_manifest_payload(report.models)
    if not payload.get("models"):
        notes_lines = _render_discover_notes(report) or ["- none"]
        response_text = "\n".join(
            [
                "No models were discovered from configured providers.",
                "Set provider credentials/endpoints first, then rerun /models discover.",
                "",
                "Notes:",
                *notes_lines,
            ]
        )
        return handle_simple_command(runtime, message, "models", response_text)

    write_manifest_yaml(
        proposal_path,
        payload,
        router_block=payload.get("router"),
    )
    provider_summary: dict[str, int] = {}
    for item in report.models:
        provider_summary[item.provider] = provider_summary.get(item.provider, 0) + 1
    provider_text = ", ".join(
        f"{provider}:{count}" for provider, count in sorted(provider_summary.items())
    )
    skipped = [result.provider for result in report.results if result.status is ProbeStatus.SKIPPED]
    notes_lines = _render_discover_notes(report)
    lines = [
        "Model discovery completed.",
        f"- discovered_models: {len(report.models)} ({provider_text})",
        f"- proposal_written: {proposal_path}",
        "- next_step: edit proposal as needed, then run /models apply",
        "- presets: see sdk/router/README.md for copy-paste router presets",
    ]
    if skipped:
        lines.append(f"- skipped (not configured): {', '.join(skipped)}")
    lines.append("- notes:")
    lines.extend(notes_lines if notes_lines else ["- none"])
    response_text = "\n".join(lines)
    return handle_simple_command(runtime, message, "models", response_text)


def _handle_models_apply(
    runtime: HostRuntime,
    message: str,
    proposal_path: Path,
    active_path: Path,
) -> str:
    """Handle ``/models apply``."""
    if not proposal_path.exists():
        response_text = (
            f"Proposal manifest not found at {proposal_path}. Run /models discover first."
        )
        return handle_simple_command(runtime, message, "models", response_text)

    try:
        manifest = load_model_manifest(proposal_path)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-emit via yaml.safe_load + render_manifest_yaml to strip
        # YAML comments (hand-edits) from the active manifest. The
        # proposal is already minimal — no preset annotations are
        # injected during discover.
        raw = proposal_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw)
        active_path.write_text(render_manifest_yaml(parsed), encoding="utf-8")
        runtime.manifest = manifest
        response_text = "\n".join(
            [
                "Applied models proposal.",
                f"- active_manifest: {active_path}",
                f"- model_count: {len(manifest.models)}",
                f"- proposal_retained: {proposal_path}",
            ]
        )
    except Exception as exc:  # pylint: disable=broad-except
        response_text = (
            f"Failed to apply proposal from {proposal_path}: {exc}\n"
            "Fix the proposal file and retry /models apply."
        )
    return handle_simple_command(runtime, message, "models", response_text)


async def handle_models_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/models`` workflows (show/discover/apply/reload/help).

    Thin dispatcher — each subcommand delegates to a dedicated handler so the
    per-branch complexity stays under the ruff C901 budget.
    """

    subcommand, args = _parse_models_subcommand(stripped)
    active_path = active_manifest_path(data_dir=runtime.data_dir)
    proposal_path = proposal_manifest_path(data_dir=runtime.data_dir)

    if subcommand in {"", "help"}:
        return handle_simple_command(runtime, message, "models", _models_help_text(runtime))

    if subcommand == "show":
        return handle_simple_command(runtime, message, "models", _render_models_manifest(runtime))

    if subcommand == "reload":
        try:
            runtime.manifest = reload_manifest(
                {}, data_dir=runtime.data_dir, path=runtime.models_path
            )
            response_text = _render_models_manifest(runtime)
        except Exception as exc:  # pylint: disable=broad-except
            response_text = (
                f"Failed to reload models from {active_path}: {exc}\n"
                "Run /models discover to generate a proposal first."
            )
        return handle_simple_command(runtime, message, "models", response_text)

    if subcommand == "discover":
        return await _handle_models_discover(runtime, message, args, proposal_path)

    if subcommand == "apply":
        return _handle_models_apply(runtime, message, proposal_path, active_path)

    if subcommand == "edit":
        response_text = (
            "CLI editor integration is handled by the CLI host layer. "
            f"Edit either {proposal_path} or {active_path}, then run /models reload."
        )
    else:
        response_text = (
            f"Unknown /models command '{subcommand}'. "
            "Supported: show, discover, apply, reload, edit, help."
        )
    return handle_simple_command(runtime, message, "models", response_text)
