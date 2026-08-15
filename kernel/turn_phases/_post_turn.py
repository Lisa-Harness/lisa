"""Post-turn helpers and capability-gate utilities for the turn runner.

Extracted from ``kernel.turn_runner`` to reduce file complexity. Owns the
capability-gate check (``_validate_model_capabilities`` +
``_CapabilityGateBlock``), the result-stringification helpers
(``_stringify_result``, ``_serialize_tool_json``,
``_stringify_pack_tool_result``), the tool-context builder
(``_to_tool_context``), and the completion-usage extraction helpers
(``_coerce_non_negative_int``, ``_coerce_non_negative_float``,
``_extract_completion_usage``).

DESIGN — Re-exported by ``kernel.turn_runner``:
    Existing tests import ``_validate_model_capabilities``,
    ``_extract_completion_usage``, and ``_coerce_non_negative_int`` directly
    from ``kernel.turn_runner``. That module re-exports every symbol below so
    no test file needs editing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kernel.models import BlockDiagnostic, OperationResult, ToolExecutionContext, ToolResult

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kernel.turn_runner import TurnResult

__all__ = [
    "_CapabilityGateBlock",
    "_coerce_non_negative_float",
    "_coerce_non_negative_int",
    "_extract_completion_usage",
    "_serialize_tool_json",
    "_stringify_pack_tool_result",
    "_stringify_result",
    "_to_tool_context",
    "_validate_model_capabilities",
]


@dataclass(frozen=True, slots=True)
class _CapabilityGateBlock:
    """Result of a failed capability-gate check."""

    result: TurnResult
    missing: tuple[str, ...]


def _validate_model_capabilities(
    *,
    mode_config: Any,
    model_def: Any,
    preferred_model_id: str,
    router_wired: bool,
) -> _CapabilityGateBlock | None:
    """Validate the resolved model against the mode's required capabilities.

    Returns ``None`` when the turn may proceed. Returns a
    ``_CapabilityGateBlock`` (carrying a user-facing ``TurnResult``) when the
    model is missing a required capability AND a router is wired — the gate
    hard-blocks so a weak model cannot reach for write-effect tools it cannot
    use reliably. When no router is wired, the gate is advisory (returns
    ``None``) because a host without a router has no escalation path.

    Backward compatible: a mode with no ``required_capabilities`` (the default)
    never blocks. A ``model_def`` that does not carry ``capabilities`` (e.g. a
    test stub) is treated as having no declared capabilities, which only
    matters when the mode requires capabilities.
    """
    if mode_config is None:
        return None
    required = getattr(mode_config, "required_capabilities", None) or ()
    if not required:
        return None
    # The gate only hard-blocks when a router is wired (escalation path exists).
    if not router_wired:
        _log.info("capability gate: advisory only for mode=%s model=%s required=%s (router not wired)", getattr(mode_config, 'mode_id', '(unknown)'), preferred_model_id, tuple(required))
        return None
    model_caps = set(getattr(model_def, "capabilities", ()) or ())
    missing = tuple(cap for cap in required if cap not in model_caps)
    if not missing:
        return None
    model_id = getattr(model_def, "model_id", preferred_model_id)
    mode_id = getattr(mode_config, "mode_id", "(unknown)")
    missing_str = ", ".join(sorted(missing))
    message = (
        f"Model '{model_id}' is missing the capability(ies) required by mode "
        f"'{mode_id}' ({missing_str}). Use a model that declares those "
        f"capabilities, or switch to a mode that does not require them. No "
        f"tools were called."
    )
    # Lazy import to avoid a circular dependency: ``kernel.turn_runner``
    # re-exports this module and defines ``TurnResult``.
    from kernel.turn_runner import TurnResult as _TurnResult  # noqa: PLC0415

    _log.warning("capability gate: model=%s missing required capabilities for mode=%s missing=%s", model_id, mode_id, missing)
    return _CapabilityGateBlock(
        result=_TurnResult(
            response_text=message,
            blocked=True,
            block_origin="capability_gate",
            block_diagnostic=BlockDiagnostic(
                reason="capability_gate",
                resolved_model_id=model_id,
                missing_capabilities=tuple(sorted(missing)),
                message=message,
            ),
        ),
        missing=missing,
    )


def _stringify_result(result: OperationResult) -> str:
    """Convert an operation result into plain text for model consumption."""
    output_text = getattr(result, "output_text", "") or ""
    output_json = getattr(result, "output_json", None) or {}
    if output_text.strip():
        return output_text.strip()
    if output_json:
        return json.dumps(output_json)
    return "Done"


def _serialize_tool_json(result: ToolResult) -> str:
    """Serialize a ToolResult's output_json for model consumption."""
    return json.dumps(result.output_json)


def _stringify_pack_tool_result(result: ToolResult) -> str:
    """Render a pack-contributed tool result for model consumption.

    Pack tools (coding_list_directory, coding_find_files, coding_read_file, etc.)
    return BOTH a terse human summary in ``output_text`` (e.g. "Listed 5 entries.")
    AND the structured payload in ``output_json`` (the entry list itself). Emitting
    only the summary starves the model of the data it called the tool to get. When
    both are present, concatenate the summary and the JSON payload.

    When the tool returns an error (``result.error`` is non-empty), the error
    MUST be surfaced to the model. Otherwise the model sees only "Done" and
    cannot understand why the tool produced no data, leading it to report
    that the tool "returned a generic Done message" instead of diagnosing
    the actual failure (e.g. missing pack configuration, bad workspace root).
    """
    summary = result.output_text.strip()
    payload = result.output_json
    if result.error.strip():
        error_text = f"Error: {result.error.strip()}"
        if summary:
            return f"{summary}\n{error_text}"
        if payload:
            return f"{error_text}\n{json.dumps(payload, indent=2, sort_keys=True)}"
        return error_text
    if summary and payload:
        return f"{summary}\n{json.dumps(payload, indent=2, sort_keys=True)}"
    if summary:
        return summary
    if payload:
        return json.dumps(payload, indent=2, sort_keys=True)
    return "Done"


def _to_tool_context(tool_id: str, ctx: Any) -> ToolExecutionContext:
    """Build a ToolExecutionContext from an OperationExecutionContext or similar.

    Carries the ambient identity/scope fields so pack-contributed tools can
    use them without a host-specific context object.
    """
    return ToolExecutionContext(
        tool_id=tool_id,
        workspace_id=getattr(ctx, "workspace_id", "default"),
        user_id=getattr(ctx, "user_id", ""),
        user_slug=getattr(ctx, "user_slug", "default"),
        surface_type=getattr(ctx, "surface_type", "personal_dm"),
        surface_id=getattr(ctx, "surface_id", ""),
        memory_scope=getattr(ctx, "memory_scope", "personal"),
        mode_id=getattr(ctx, "mode_id", ""),
        query=getattr(ctx, "query", ""),
        parameters=getattr(ctx, "parameters", {}) or {},
        pack_config=getattr(ctx, "pack_config", {}),
        pack_configuration_resolver=getattr(ctx, "pack_configuration_resolver", None),
        approval_service=getattr(ctx, "approval_service", None),
        approval_request_id=getattr(ctx, "approval_request_id", ""),
        session_plan_digest=getattr(ctx, "session_plan_digest", ""),
        code_change_artifact_store=getattr(ctx, "code_change_artifact_store", None),
        event_bus=getattr(ctx, "event_bus", None),
        state_provider=getattr(ctx, "state_provider", None),
        data_dir=getattr(ctx, "data_dir", ""),
    )


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _extract_completion_usage(
    completion: Any,
) -> tuple[int, int, int, int, int, float, int]:
    """Extract token, sub-counter, duration, and latency counts from one completion.

    Returns a 7-tuple ``(prompt_tokens, completion_tokens, total_tokens,
    cached_tokens, reasoning_tokens, duration_ms, latency_ms)``.

    The function reads both the flat attributes some hosts set on the
    completion object (``prompt_tokens`` etc.) and the nested ``usage``
    object the kernel gateways populate. Sub-counters (``cached_tokens``,
    ``reasoning_tokens``) and ``duration_ms`` are read defensively so a
    missing field never raises; they default to 0 when the provider did
    not surface them.
    """

    prompt_tokens = _coerce_non_negative_int(getattr(completion, "prompt_tokens", 0))
    completion_tokens = _coerce_non_negative_int(getattr(completion, "completion_tokens", 0))
    total_tokens = _coerce_non_negative_int(getattr(completion, "total_tokens", 0))
    cached_tokens = _coerce_non_negative_int(getattr(completion, "cached_tokens", 0))
    reasoning_tokens = _coerce_non_negative_int(getattr(completion, "reasoning_tokens", 0))
    duration_ms = _coerce_non_negative_float(getattr(completion, "duration_ms", 0.0))
    latency_ms = _coerce_non_negative_int(getattr(completion, "latency_ms", 0))

    usage = getattr(completion, "usage", None)
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_cached = 0
    usage_reasoning = 0
    if isinstance(usage, dict):
        usage_prompt = _coerce_non_negative_int(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )
        usage_completion = _coerce_non_negative_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )
        usage_total = _coerce_non_negative_int(usage.get("total_tokens"))
        usage_cached = _coerce_non_negative_int(usage.get("cached_tokens"))
        usage_reasoning = _coerce_non_negative_int(usage.get("reasoning_tokens"))
    elif usage is not None:
        usage_prompt = _coerce_non_negative_int(
            getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        )
        usage_completion = _coerce_non_negative_int(
            getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        )
        usage_total = _coerce_non_negative_int(getattr(usage, "total_tokens", None))
        usage_cached = _coerce_non_negative_int(getattr(usage, "cached_tokens", None))
        usage_reasoning = _coerce_non_negative_int(getattr(usage, "reasoning_tokens", None))

    prompt = prompt_tokens or usage_prompt
    completion_value = completion_tokens or usage_completion
    total_value = total_tokens or usage_total
    if total_value <= 0:
        total_value = prompt + completion_value
    cached = cached_tokens or usage_cached
    reasoning = reasoning_tokens or usage_reasoning

    return prompt, completion_value, total_value, cached, reasoning, duration_ms, latency_ms
