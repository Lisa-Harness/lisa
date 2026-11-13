"""Post-turn helpers and capability-gate utilities for the turn runner.

Owns the capability-gate check (``_validate_model_capabilities`` +
``_CapabilityGateBlock``), the result-stringification helpers
(``_stringify_result``, ``_serialize_tool_json``,
``_stringify_pack_tool_result``), the tool-context builder
(``_to_tool_context``), and the completion-usage extraction helpers
(``_coerce_non_negative_int``, ``_coerce_non_negative_float``,
``_optional_int``, ``_optional_float``, ``_extract_completion_usage``).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from kernel.models import (
    BlockDiagnostic,
    OperationResult,
    ToolExecutionContext,
    ToolResult,
    resolve_turn_index,
)

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kernel.turn_runner import TurnResult


@dataclass(frozen=True, slots=True)
class CompletionUsageSummary:
    """Normalized per-completion usage, including cache sub-counters.

    Returned by :func:`_extract_completion_usage`. Cache-efficiency v2:
    counters the provider never surfaced stay ``None`` (unknown); a
    present ``0`` is a real measured zero. ``total_tokens`` is derived
    from ``prompt + completion`` only when both are known, and the
    derivation is marked in ``metric_origins``.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    #: Provider-reported prompt-cache *creation/population* tokens
    #: (Anthropic ``cache_creation_input_tokens``). ``None`` when unknown.
    cache_creation_tokens: int | None = None
    duration_ms: float | None = None
    latency_ms: int | None = None
    #: The provider's own reported charge for this call (OpenRouter
    #: ``usage.cost``). Authoritative when present — it already reflects
    #: the provider's real cache read/write rates. ``None`` when
    #: unreported, in which case callers may price the call from manifest
    #: rates; ``None`` is unknown, not a measured free call.
    provider_cost_usd: float | None = None
    #: Per-field origin map (``derived`` / ``host``); absent key =
    #: provider-reported at the source.
    metric_origins: dict[str, str] = field(default_factory=dict)
    #: Counters a routing proxy declared on this call's response headers
    #: (see ``ProxyTelemetrySpec``). Empty when no proxy is in the path or it
    #: declared none.
    proxy_telemetry: dict[str, int] = field(default_factory=dict)

    #: Legacy tuple layout — ``(prompt, completion, total, cached, reasoning,
    #: duration_ms, latency_ms)``. Iterating/unpacking preserves the original
    #: 7-field order so pre-summary tuple consumption keeps working; the new
    #: ``cache_creation_tokens`` counter is read by attribute, not by index.
    #: Unknown counters contribute ``0`` to the legacy tuple (operational
    #: arithmetic only — the durable event keeps ``None``).
    def __iter__(self) -> Any:
        yield self.prompt_tokens or 0
        yield self.completion_tokens or 0
        yield self.total_tokens or 0
        yield self.cached_tokens or 0
        yield self.reasoning_tokens or 0
        yield self.duration_ms or 0.0
        yield self.latency_ms or 0

    def __getitem__(self, index: int) -> Any:
        values = (
            self.prompt_tokens or 0,
            self.completion_tokens or 0,
            self.total_tokens or 0,
            self.cached_tokens or 0,
            self.reasoning_tokens or 0,
            self.duration_ms or 0.0,
            self.latency_ms or 0,
        )
        return values[index]


__all__ = [
    "CompletionUsageSummary",
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
        _log.info(
            "capability gate: advisory only for mode=%s model=%s required=%s (router not wired)",
            getattr(mode_config, "mode_id", "(unknown)"),
            preferred_model_id,
            tuple(required),
        )
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
    from kernel.turn_runner import TurnResult as _TurnResult

    _log.warning(
        "capability gate: model=%s missing required capabilities for mode=%s missing=%s",
        model_id,
        mode_id,
        missing,
    )
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
        return _render_pack_tool_error(result)
    if summary and payload:
        return f"{summary}\n{json.dumps(payload, indent=2, sort_keys=True)}"
    if summary:
        return summary
    if payload:
        return json.dumps(payload, indent=2, sort_keys=True)
    return "Done"


def _render_pack_tool_error(result: ToolResult) -> str:
    """Render a pack tool's error text, optionally with summary or JSON payload."""
    summary = result.output_text.strip()
    payload = result.output_json
    error_text = f"Error: {result.error.strip()}"
    if summary:
        return f"{summary}\n{error_text}"
    if payload:
        return f"{error_text}\n{json.dumps(payload, indent=2, sort_keys=True)}"
    return error_text


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
        thread_store_slug=getattr(ctx, "thread_store_slug", ""),
        surface_type=getattr(ctx, "surface_type", "personal_dm"),
        surface_id=getattr(ctx, "surface_id", ""),
        memory_scope=getattr(ctx, "memory_scope", "personal"),
        mode_id=getattr(ctx, "mode_id", ""),
        query=getattr(ctx, "query", ""),
        parameters=getattr(ctx, "parameters", {}) or {},
        pack_config=getattr(ctx, "pack_config", {}),
        capability_binding_provider=getattr(ctx, "capability_binding_provider", None),
        approval_service=getattr(ctx, "approval_service", None),
        approval_request_id=getattr(ctx, "approval_request_id", ""),
        task_id=getattr(ctx, "task_id", ""),
        session_plan_digest=getattr(ctx, "session_plan_digest", ""),
        code_change_artifact_store=getattr(ctx, "code_change_artifact_store", None),
        event_bus=getattr(ctx, "event_bus", None),
        state_provider=getattr(ctx, "state_provider", None),
        data_dir=getattr(ctx, "data_dir", ""),
        thread_dir=getattr(ctx, "thread_dir", ""),
        turn_index=resolve_turn_index(getattr(ctx, "turn_index", None)),
        mcp_tool_call=getattr(ctx, "mcp_tool_call", None),
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


def _optional_int(value: Any) -> int | None:
    """Coerce to a non-negative int, or ``None`` when the value is absent.

    Cache-efficiency v2 counterpart to :func:`_coerce_non_negative_int`:
    ``None``/absent/malformed stays unknown instead of degrading to a
    measured zero. ``0`` survives as a measured zero.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _optional_float(value: Any) -> float | None:
    """Coerce to a non-negative float, or ``None`` when the value is absent."""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _coerce_telemetry(value: Any) -> dict[str, int]:
    """Normalize proxy-supplied counters to a ``{str: non-negative int}`` map.

    Proxy telemetry arrives as JSON from a gateway, so a malformed payload
    from a third-party proxy must degrade to "no counters" rather than
    raising inside usage extraction. Non-numeric or negative values are
    dropped individually so one bad counter does not discard the rest.
    """

    if not isinstance(value, dict):
        return {}
    clean: dict[str, int] = {}
    for key, raw in value.items():
        name = str(key or "").strip()
        if not name:
            continue
        try:
            clean[name] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return clean


_UsageT = TypeVar("_UsageT")


def _resolve_usage_value(
    flat: _UsageT | None,
    nested: _UsageT | None,
) -> _UsageT | None:
    """Resolve one usage metric: a nonzero flat value wins, flat-0 defers.

    Legacy hosts stamped ``0`` defaults onto the completion object for
    metrics the provider never surfaced, so a flat ``0`` is treated as the
    "not populated" sentinel and the nested provider ``usage`` value wins.
    A nonzero flat value is an explicit host measurement. When neither
    source reports the metric the flat value (``0`` sentinel or ``None``)
    survives, and the kernel's nullable pipeline keeps it unknown when
    unknown is the truth.
    """

    if flat is not None and flat != 0:
        return flat
    if nested is not None:
        return nested
    return flat


def _read_nested_usage(
    usage: Any,
    duration_ms: float,
) -> tuple[
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    float | None,
    dict[str, int],
    float,
]:
    """Read the nested provider ``usage`` object defensively.

    Handles both dict-shaped and object-shaped usage (the kernel gateways
    emit objects; some hosts pass raw provider dicts). Returns the nested
    counters/cost/proxy-telemetry plus the duration fallback applied: flat
    attributes take precedence over the nested ``duration_ms`` so
    time-billed providers keep their duration without the caller
    re-stamping it.
    """

    empty_telemetry: dict[str, int] = {}
    if usage is None:
        return None, None, None, None, None, None, None, empty_telemetry, duration_ms

    def _get(key: str) -> Any:
        return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)

    def _first_present(*keys: str) -> Any:
        """First alias whose value is present — a measured ``0`` wins."""

        for key in keys:
            value = _get(key)
            if value is not None:
                return value
        return None

    prompt = _optional_int(_first_present("prompt_tokens", "input_tokens"))
    completion = _optional_int(_first_present("completion_tokens", "output_tokens"))
    total = _optional_int(_first_present("total_tokens"))
    cached = _optional_int(_first_present("cached_tokens"))
    reasoning = _optional_int(_first_present("reasoning_tokens"))
    cache_creation = _optional_int(
        _first_present("cache_creation_tokens", "cache_creation_input_tokens")
    )
    provider_cost = _optional_float(_get("provider_cost_usd"))
    proxy_telemetry = _coerce_telemetry(_get("proxy_telemetry"))
    if duration_ms <= 0.0:
        duration_ms = _coerce_non_negative_float(_get("duration_ms") or 0.0)
    return (
        prompt,
        completion,
        total,
        cached,
        reasoning,
        cache_creation,
        provider_cost,
        proxy_telemetry,
        duration_ms,
    )


def _extract_completion_usage(
    completion: Any,
) -> CompletionUsageSummary:
    """Extract token, sub-counter, duration, and latency counts from one completion.

    Returns a :class:`CompletionUsageSummary` carrying ``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, ``cached_tokens``,
    ``reasoning_tokens``, ``cache_creation_tokens``, ``duration_ms``, and
    ``latency_ms``.

    The function reads both the flat attributes some hosts set on the
    completion object (``prompt_tokens`` etc.) and the nested ``usage``
    object the kernel gateways populate. Fields are read defensively so a
    missing field never raises. Cache-efficiency v2 resolution per metric:
    a nonzero flat attribute wins; a flat ``0`` (the legacy "host did not
    populate this" sentinel) defers to the nested provider ``usage``; when
    neither reports the metric it stays ``None`` (unknown), and a genuine
    measured ``0`` from the only reporting source survives.
    """

    prompt_tokens = _optional_int(getattr(completion, "prompt_tokens", None))
    completion_tokens = _optional_int(getattr(completion, "completion_tokens", None))
    total_tokens = _optional_int(getattr(completion, "total_tokens", None))
    cached_tokens = _optional_int(getattr(completion, "cached_tokens", None))
    reasoning_tokens = _optional_int(getattr(completion, "reasoning_tokens", None))
    cache_creation_tokens = _optional_int(getattr(completion, "cache_creation_tokens", None))
    provider_cost = _optional_float(getattr(completion, "provider_cost_usd", None))
    duration_ms = _coerce_non_negative_float(getattr(completion, "duration_ms", 0.0))
    latency_ms = _coerce_non_negative_int(getattr(completion, "latency_ms", 0))

    origins: dict[str, str] = {}
    usage = getattr(completion, "usage", None)
    (
        usage_prompt,
        usage_completion,
        usage_total,
        usage_cached,
        usage_reasoning,
        usage_cache_creation,
        usage_provider_cost,
        usage_proxy_telemetry,
        duration_ms,
    ) = _read_nested_usage(usage, duration_ms)

    prompt = _resolve_usage_value(prompt_tokens, usage_prompt)
    completion_value = _resolve_usage_value(completion_tokens, usage_completion)
    total_value = _resolve_usage_value(total_tokens, usage_total)
    cached_value = _resolve_usage_value(cached_tokens, usage_cached)
    reasoning_value = _resolve_usage_value(reasoning_tokens, usage_reasoning)
    cache_creation_value = _resolve_usage_value(cache_creation_tokens, usage_cache_creation)
    provider_cost_value = _resolve_usage_value(provider_cost, usage_provider_cost)
    if (
        (total_value is None or total_value == 0)
        and prompt is not None
        and completion_value is not None
    ):
        # ``None`` is unknown and a flat ``0`` is the legacy host sentinel;
        # both derive from known prompt + completion.
        total_value = prompt + completion_value
        origins["total_tokens"] = "derived"
    if latency_ms <= 0 and duration_ms > 0.0:
        latency_ms = _coerce_non_negative_int(duration_ms)

    return CompletionUsageSummary(
        prompt_tokens=prompt,
        completion_tokens=completion_value,
        total_tokens=total_value,
        cached_tokens=cached_value,
        reasoning_tokens=reasoning_value,
        cache_creation_tokens=cache_creation_value,
        duration_ms=duration_ms,
        latency_ms=latency_ms,
        provider_cost_usd=provider_cost_value,
        metric_origins=origins,
        proxy_telemetry=usage_proxy_telemetry,
    )
