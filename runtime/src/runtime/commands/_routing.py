"""Mode/model alias, route-preview, and mode-support helpers.

Owns the routing-side helpers that translate mode ids → model aliases, set
the runtime's route preview, and describe the host's capability gaps
relative to installed modes. The core host delegates all mode execution to
the kernel, so :func:`mode_support_gap` always returns ``None`` here.

Dependency direction: ``_routing`` → ``_log`` (for ``_resolve_model``),
``kernel.*``. No imports from ``_render``/``_simple``/``_packs``/``_lists``.
"""

from __future__ import annotations

import logging

from kernel.model_manifest import ModelManifest
from kernel.models import WorkflowModeConfig
from kernel.operations.registry import (
    OperationContentError,
    OperationDegradedError,
    OperationExecutionError,
    OperationTimeoutError,
)

from runtime.commands._log import _resolve_model
from runtime.runtime import HostRuntime

logger = logging.getLogger(__name__)


# ── Mode / model alias helpers ───────────────────────────────────────


def model_alias_for_mode(
    mode_id: str,
    *,
    manifest: ModelManifest | None = None,
) -> str:
    """Return the model alias for a given mode id.

    Single precedence: ``mode_models[mode_id]`` wins; if the mode isn't
    pinned explicitly, fall back to the ``default`` mode's pin. There is
    no hardcoded alias map — the manifest is the single source of truth.
    """
    if manifest is None:
        return "default"
    normalized = mode_id.strip().lower()
    model_ref = manifest.mode_models.get(normalized)
    if model_ref:
        return model_ref
    return manifest.mode_models.get("default", "default")


def preferred_model_id_for_route(
    *,
    raw_message: str,
    mode_id: str,
    manifest: ModelManifest | None = None,
) -> str:
    """Return the preferred model id for a route decision.

    ``raw_message`` is kept in the signature for callers that may use it for
    message-aware model selection in future hosts; the core host routes by
    mode alias only. When a manifest with a ``mode_models`` override for the
    mode is provided, that explicit model reference wins.
    """
    del raw_message
    return model_alias_for_mode(mode_id, manifest=manifest)


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
    model_id, provider = predict_model(runtime, preferred_model_id)
    runtime.session.current_mode_id = mode_id
    runtime.session.current_stage_id = "-"
    runtime.session.next_model_id = model_id
    runtime.session.next_model_provider = provider
    logger.debug(
        "route preview set: mode=%s model=%s provider=%s",
        mode_id,
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
    if isinstance(exc, OperationTimeoutError):
        return (
            f"{message}\n\n"
            "The host kept running, but this operation exceeded its local time budget. "
            "Try a narrower query, a faster model, or a larger operation timeout."
        )
    if isinstance(exc, OperationContentError):
        return (
            f"{message}\n\n"
            "The model produced empty or structurally invalid output for this operation. "
            "Try a more specific prompt, a different model, or a smaller target length."
        )
    if isinstance(exc, OperationDegradedError):
        return (
            f"{message}\n\n"
            "The operation completed with warnings — the artifact may be partial. "
            "Review it before relying on it, or retry with a narrower prompt."
        )
    lowered = message.lower()
    if "timed out" in lowered:
        return (
            f"{message}\n\n"
            "The host kept running, but this operation exceeded its local time budget. "
            "Try a narrower query, a faster model, or a larger operation timeout."
        )
    return message
