"""Capability dispatch execution and receipt tracking for the turn runner.

This module encapsulates the execution of in-process operations, pack-contributed
tools, and MCP tools during Phase 2 of the turn loop, including within-turn
read caching, effect receipt capture, error-streak tracking, and recovery
settlement.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from kernel.approval import ApprovalRequired, EffectKind
from kernel.models import (
    EffectReceipt,
    ForkChildResult,
    OperationExecutionContext,
    OperationResult,
    ToolResult,
    YieldAction,
)
from kernel.tools.mcp_client import MCPToolPool
from kernel.tools.registry import execute_tool
from kernel.turn_phases._post_turn import (
    _stringify_pack_tool_result,
    _stringify_result,
    _to_tool_context,
)
from kernel.turn_phases._turn_state import (
    _TurnState,
)
from kernel.turn_phases._turn_state import (
    merge_child_usage as _merge_child_usage_fn,
)

if TYPE_CHECKING:
    from kernel.models import CallLedger
    from kernel.turn_runner import TurnHooks

_log = logging.getLogger(__name__)

__all__ = [
    "CapabilityDispatchOutcome",
    "_track_retrieval_call",
    "dispatch_capability",
    "enforce_write_receipt",
    "format_read_abbreviation",
    "read_cache_signature",
]

#: Thread-archive tools whose repeated identical calls signal paging in
#: circles rather than gathering (the 2026-10-02 reviewer burn: 23
#: thread_read_content calls, many repeating earlier pages).
_RETRIEVAL_TOOL_SUFFIXES = (
    "thread_read_content",
    "thread_read_range",
    "thread_read_recent",
    "thread_find_content",
    "thread_search",
    "thread_task_view",
)


def _track_retrieval_call(state: Any, tool_name: str, tool_arguments: Any) -> None:
    """Count a retrieval-tool call and its distinct argument signature.

    Paired fields on ``_TurnState`` give the read-loop nudge its factual
    basis: how many retrieval calls happened and how many repeated
    arguments the model already held.
    """
    if tool_name.rsplit(".", 1)[-1] not in _RETRIEVAL_TOOL_SUFFIXES:
        return
    state.retrieval_calls += 1
    try:
        signature = json.dumps(tool_arguments, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        signature = repr(tool_arguments)
    state.retrieval_distinct_args.add(f"{tool_name}:{signature}")


def read_cache_signature(tool_name: str, tool_arguments: dict[str, Any]) -> str:
    """Build a stable cache key for one read-tool invocation."""
    parts = [tool_name]
    for key in sorted(tool_arguments):
        val = tool_arguments[key]
        if isinstance(val, (dict, list)):
            parts.append(f"{key}={json.dumps(val, sort_keys=True)}")
        else:
            parts.append(f"{key}={val}")
    return "|".join(parts)


def format_read_abbreviation(
    template: str,
    params: dict[str, Any],
    result_json: dict[str, Any],
    result_text: str = "",
) -> str:
    """Apply a tool's ``abbreviation_template`` to params + result JSON."""
    fields: dict[str, Any] = dict(params)
    fields.update(result_json)
    fields["output_text"] = result_text

    class _MissingKeyDict(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_MissingKeyDict(fields))


def enforce_write_receipt(
    *,
    tool_name: str,
    dispatched_effect: str,
    dispatched_succeeded: bool,
    receipts: list[EffectReceipt],
    receipts_before: int,
) -> str | None:
    """Enforce the Phase-8 receipt invariant for one dispatch.

    A capability whose declared effect is ``write`` / ``external_write``
    and whose dispatch succeeded MUST have produced an ``EffectReceipt``
    in this dispatch (``receipts[receipts_before:]``). Returns a loud
    kernel-error marker string when the invariant is violated, else ``None``.
    """
    if dispatched_effect not in ("write", "external_write") or not dispatched_succeeded:
        return None
    if any(r.capability_id == tool_name for r in receipts[receipts_before:]):
        return None
    return (
        f"KERNEL ERROR: write capability '{tool_name}' (effect="
        f"{dispatched_effect}) returned success without producing an "
        "EffectReceipt. The completion contract cannot verify this write; "
        "the kernel counts it as unverified."
    )


@dataclass(frozen=True, slots=True)
class CapabilityDispatchOutcome:
    """Outcome of dispatching one capability."""

    tool_output: str
    duration_ms: int
    is_ok: bool
    operation_result: OperationResult | None = None
    delegate_yield: YieldAction | None = None


async def _dispatch_operation(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    ctx: OperationExecutionContext,
    state: _TurnState,
    exec_op: Callable[..., Awaitable[OperationResult]],
) -> tuple[str, OperationResult | None, YieldAction | None]:
    """Execute an in-process operation via the host's exec_op hook."""
    op_ctx = ctx.model_copy(update={"operation_id": tool_name, "params": tool_arguments})
    op_result = await exec_op(tool_name, op_ctx)
    tool_output = _stringify_result(op_result)

    op_json = getattr(op_result, "output_json", None)
    if isinstance(op_json, dict) and (
        op_json.get("per_model_usage") or op_json.get("per_mode_usage")
    ):
        _merge_child_usage_fn(
            state,
            per_model_usage=op_json.get("per_model_usage"),
            per_mode_usage=op_json.get("per_mode_usage"),
        )

    if (
        isinstance(op_json, dict)
        and tool_name == "fork"
        and isinstance(op_json.get("children"), list)
    ):
        state.fork_results = [
            ForkChildResult.model_validate(child)
            for child in op_json["children"]
            if isinstance(child, dict)
        ]

    delegate_yield = None
    if (
        isinstance(op_json, dict)
        and tool_name == "delegate"
        and op_json.get("status") == "model_yield"
        and isinstance(op_json.get("yield_action"), dict)
    ):
        ya_dict = op_json["yield_action"]
        child_question = str(ya_dict.get("question") or "")
        delegated_query = str(tool_arguments.get("query") or "")
        ya_dict["context"] = (
            f"The delegate was asked: '{delegated_query}'. It asks: '{child_question}'"
        )
        delegate_yield = YieldAction.model_validate(ya_dict)
        tool_output = f"Yielded to user: {delegate_yield.question}"

    return tool_output, op_result, delegate_yield


async def _dispatch_pack_tool(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    tool_def: Any,
    ctx: OperationExecutionContext,
    state: _TurnState,
    pool: MCPToolPool,
    logger: logging.Logger,
) -> tuple[str, ToolResult | None, Any | None]:
    """Execute an in-process pack-contributed tool with optional read caching."""
    cache_template = getattr(tool_def, "abbreviation_template", "") or ""
    cacheable = (
        getattr(tool_def, "effect", "") == "read"
        and bool(cache_template)
        and isinstance(tool_arguments, dict)
    )
    if cacheable:
        cache_key = read_cache_signature(tool_name, tool_arguments)
        cached = state.read_cache.get(cache_key)
    else:
        cache_key = ""
        cached = None

    if cached is not None:
        logger.info(
            "read cache hit (tool=%s, key=%s); returning abbreviation",
            tool_name,
            cache_key[:120],
        )
        return cached, None, None

    if tool_def.implementation_module is not None:
        tool_ctx = _to_tool_context(tool_name, ctx)
        tool_ctx.parameters = tool_arguments
        tool_ctx.stream_output = bool(getattr(tool_def, "stream_output", False))
        tool_result = await execute_tool(tool_name, tool_ctx)
        tool_output = _stringify_pack_tool_result(tool_result)

        if cacheable and isinstance(tool_result, ToolResult):
            try:
                abbrev = format_read_abbreviation(
                    cache_template,
                    tool_arguments,
                    tool_result.output_json or {},
                    result_text=tool_result.output_text or "",
                )
                state.read_cache[cache_key] = abbrev
            except Exception as cache_exc:  # pylint: disable=broad-except
                logger.debug("read cache record failed for %s: %s", tool_name, cache_exc)
        return tool_output, tool_result, None

    mcp_result = await pool.call_tool(tool_name, tool_arguments)
    return mcp_result.output_text or "Done", None, mcp_result


def _resolve_dispatch_metadata(
    tool_name: str,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    op_result: OperationResult | None,
    tool_result: ToolResult | None,
) -> tuple[str, str]:
    """Return (dispatched_effect, dispatched_artifact_id)."""
    dispatched_effect = "read"
    dispatched_artifact_id = ""
    if tool_name in operation_catalog:
        op_def = operation_catalog[tool_name]
        dispatched_effect = str(getattr(op_def, "effect", "read") or "read")
        dispatched_artifact_id = str(
            getattr(op_result, "output_json", {}).get("artifact_id", "")
            or getattr(op_result, "output_json", {}).get("document_id", "")
            or getattr(op_result, "output_json", {}).get("memory_id", "")
            or ""
        )
    elif tool_name in tool_catalog:
        tool_def = tool_catalog[tool_name]
        dispatched_effect = str(getattr(tool_def, "effect", "read") or "read")
        if tool_result is not None and isinstance(tool_result, ToolResult):
            dispatched_artifact_id = str(
                (tool_result.output_json or {}).get("artifact_id", "")
                or (tool_result.output_json or {}).get("document_id", "")
                or (tool_result.output_json or {}).get("memory_id", "")
                or ""
            )
    return dispatched_effect, dispatched_artifact_id


def _returned_error(
    op_result: OperationResult | None,
    tool_result: ToolResult | None,
    mcp_result: Any | None,
) -> str:
    if op_result is not None and not op_result.is_success:
        return op_result.error_detail or op_result.output_text or f"Operation {op_result.status}."
    if tool_result is not None and tool_result.error.strip():
        return tool_result.error.strip()
    if mcp_result is not None and getattr(mcp_result, "is_error", False):
        return str(getattr(mcp_result, "output_text", "") or "MCP tool returned an error.")
    return ""


def _handle_write_receipt(
    *,
    tool_name: str,
    dispatched_effect: str,
    dispatched_artifact_id: str,
    op_result: OperationResult | None,
    tool_result: ToolResult | None,
    tool_name_in_op_catalog: bool,
    tool_name_in_tool_catalog: bool,
    state: _TurnState,
    logger: logging.Logger,
) -> bool:
    """Handle write/external_write effect receipt recording."""
    dispatched_succeeded = True
    if (
        tool_name_in_tool_catalog
        and tool_result is not None
        and isinstance(tool_result, ToolResult)
    ):
        dispatched_succeeded = not tool_result.error.strip()

    if (
        tool_name_in_op_catalog
        and op_result is not None
        and isinstance(op_result, OperationResult)
        and not op_result.is_success
    ):
        dispatched_succeeded = False
        logger.warning(
            "operation '%s' returned status=%s (effect=%s); not recording a write receipt. "
            "detail=%s",
            tool_name,
            op_result.status,
            dispatched_effect,
            op_result.error_detail or op_result.output_text[:200],
        )

    if dispatched_succeeded:
        state.effect_receipts.append(
            EffectReceipt(
                effect=cast(EffectKind, dispatched_effect),
                capability_id=tool_name,
                artifact_id=dispatched_artifact_id,
            )
        )
    return dispatched_succeeded


def _handle_read_receipt(
    *,
    tool_name: str,
    dispatched_effect: str,
    dispatched_artifact_id: str,
    mcp_result: Any | None,
    known_read: bool,
    pool: MCPToolPool,
    state: _TurnState,
) -> bool:
    """Handle read/execute effect receipt recording."""
    mcp_read_only = False
    mcp_spec_fn = getattr(pool, "tool_spec", None)
    if mcp_spec_fn is not None and not known_read:
        mcp_spec = mcp_spec_fn(tool_name)
        if mcp_spec is not None:
            mcp_read_only = bool(getattr(mcp_spec, "read_only_hint", False))
    mcp_succeeded = True
    if mcp_result is not None and hasattr(mcp_result, "is_error"):
        mcp_succeeded = not getattr(mcp_result, "is_error", False)
    if mcp_succeeded:
        if known_read or mcp_read_only:
            state.effect_receipts.append(
                EffectReceipt(
                    effect=cast(EffectKind, dispatched_effect),
                    capability_id=tool_name,
                    artifact_id=dispatched_artifact_id,
                )
            )
        else:
            state.effect_receipts.append(
                EffectReceipt(
                    effect="write",
                    capability_id=tool_name,
                    artifact_id="",
                )
            )
    return mcp_succeeded


def _capture_effect_receipt(
    *,
    tool_name: str,
    tool_output: str,
    op_result: OperationResult | None,
    tool_result: ToolResult | None,
    mcp_result: Any | None,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    pool: MCPToolPool,
    state: _TurnState,
    logger: logging.Logger,
) -> tuple[str, str, bool, str]:
    """Capture effect receipt and enforce write receipt invariant."""
    receipts_before = len(state.effect_receipts)
    dispatched_effect, dispatched_artifact_id = _resolve_dispatch_metadata(
        tool_name, operation_catalog, tool_catalog, op_result, tool_result
    )
    returned_error = _returned_error(op_result, tool_result, mcp_result)

    if returned_error:
        dispatched_succeeded = False
    elif dispatched_effect in ("write", "external_write"):
        dispatched_succeeded = _handle_write_receipt(
            tool_name=tool_name,
            dispatched_effect=dispatched_effect,
            dispatched_artifact_id=dispatched_artifact_id,
            op_result=op_result,
            tool_result=tool_result,
            tool_name_in_op_catalog=tool_name in operation_catalog,
            tool_name_in_tool_catalog=tool_name in tool_catalog,
            state=state,
            logger=logger,
        )
    else:
        dispatched_succeeded = _handle_read_receipt(
            tool_name=tool_name,
            dispatched_effect=dispatched_effect,
            dispatched_artifact_id=dispatched_artifact_id,
            mcp_result=mcp_result,
            known_read=tool_name in operation_catalog or tool_name in tool_catalog,
            pool=pool,
            state=state,
        )

    receipt_error = enforce_write_receipt(
        tool_name=tool_name,
        dispatched_effect=dispatched_effect,
        dispatched_succeeded=dispatched_succeeded,
        receipts=state.effect_receipts,
        receipts_before=receipts_before,
    )
    if receipt_error is not None:
        logger.error("%s", receipt_error)
        tool_output = receipt_error
        returned_error = receipt_error
        dispatched_succeeded = False

    return tool_output, dispatched_effect, dispatched_succeeded, returned_error


def _track_consecutive_errors(
    *,
    tool_name: str,
    dispatched_effect: str,
    dispatched_succeeded: bool,
    op_result: OperationResult | None,
    tool_result: ToolResult | None,
    mcp_result: Any | None,
    state: _TurnState,
) -> None:
    """Track streak of consecutive failures on the same tool."""
    dispatch_failed = False
    if dispatched_effect in ("write", "external_write"):
        dispatch_failed = not dispatched_succeeded
    elif tool_result is not None and isinstance(tool_result, ToolResult):
        dispatch_failed = bool(tool_result.error.strip())
    elif mcp_result is not None and hasattr(mcp_result, "is_error"):
        dispatch_failed = bool(getattr(mcp_result, "is_error", False))
    elif op_result is not None and isinstance(op_result, OperationResult):
        dispatch_failed = not op_result.is_success

    if dispatch_failed:
        if state.last_error_tool == tool_name:
            state.consecutive_tool_errors += 1
        else:
            state.last_error_tool = tool_name
            state.consecutive_tool_errors = 1
    else:
        state.consecutive_tool_errors = 0
        state.last_error_tool = ""


async def _settle_recovery(
    hooks: TurnHooks,
    recovery_item_id: str | None,
    rec_identity: Any,
    *,
    state: str,
    tool_name: str,
    logger: logging.Logger,
    result_preview: str = "",
    last_error: str = "",
    retry_policy: Any = None,
) -> None:
    """Safely settle an active recovery record with fail-loud logging."""
    if not (
        recovery_item_id
        and hooks.services.recovery_service is not None
        and rec_identity is not None
    ):
        return
    try:
        kwargs: dict[str, Any] = {"state": state}
        if result_preview:
            kwargs["result_preview"] = result_preview
        if last_error:
            kwargs["last_error"] = last_error
        if retry_policy is not None:
            kwargs["retry_policy"] = retry_policy
        await hooks.services.recovery_service.settle(
            recovery_item_id,
            rec_identity,
            **kwargs,
        )
    except Exception as recovery_exc:  # pylint: disable=broad-except
        logger.warning(
            "recovery settle %s failed for %s: %s",
            state,
            tool_name,
            recovery_exc,
            exc_info=True,
        )


async def dispatch_capability(
    *,
    tool_name: str,
    tool_arguments: dict[str, Any],
    tool_kind: str,
    ctx: OperationExecutionContext,
    state: _TurnState,
    ledger: CallLedger,
    pool: MCPToolPool,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    exec_op: Callable[..., Awaitable[OperationResult]],
    hooks: TurnHooks,
    recovery_item_id: str | None = None,
    recovery_retry_policy: str = "confirm",
    build_recovery_identity_fn: Callable[[TurnHooks], Any] | None = None,
    log: logging.Logger | None = None,
) -> CapabilityDispatchOutcome:
    """Dispatch an operation, pack tool, or MCP tool and update ledger/receipts."""
    logger = log or _log
    tool_dispatch_started = time.monotonic()
    op_result: OperationResult | None = None
    tool_result: ToolResult | None = None
    mcp_result: Any | None = None
    delegate_yield: YieldAction | None = None
    rec_identity = build_recovery_identity_fn(hooks) if build_recovery_identity_fn else None
    dispatch_error = ""

    try:
        if tool_name in operation_catalog:
            tool_output, op_result, delegate_yield = await _dispatch_operation(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                ctx=ctx,
                state=state,
                exec_op=exec_op,
            )
        elif tool_name in tool_catalog:
            tool_output, tool_result, mcp_result = await _dispatch_pack_tool(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                tool_def=tool_catalog[tool_name],
                ctx=ctx,
                state=state,
                pool=pool,
                logger=logger,
            )
        else:
            mcp_res = await pool.call_tool(tool_name, tool_arguments)
            mcp_result = mcp_res
            tool_output = mcp_res.output_text or "Done"

        state.total_tool_calls += 1
        state.step_tool_calls += 1
        state.called_tool_names.add(tool_name)
        _track_retrieval_call(state, tool_name, tool_arguments)
        ledger.record(tool_name)

        (
            tool_output,
            dispatched_effect,
            dispatched_succeeded,
            dispatch_error,
        ) = _capture_effect_receipt(
            tool_name=tool_name,
            tool_output=tool_output,
            op_result=op_result,
            tool_result=tool_result,
            mcp_result=mcp_result,
            operation_catalog=operation_catalog,
            tool_catalog=tool_catalog,
            pool=pool,
            state=state,
            logger=logger,
        )
        state.call_ledger_counts = dict(ledger.counts)

        _track_consecutive_errors(
            tool_name=tool_name,
            dispatched_effect=dispatched_effect,
            dispatched_succeeded=dispatched_succeeded,
            op_result=op_result,
            tool_result=tool_result,
            mcp_result=mcp_result,
            state=state,
        )

        await _settle_recovery(
            hooks,
            recovery_item_id,
            rec_identity,
            state="succeeded" if dispatched_succeeded else "stalled",
            tool_name=tool_name,
            logger=logger,
            result_preview=str(tool_output)[:240] if dispatched_succeeded else "",
            last_error=dispatch_error,
            retry_policy=cast(Any, recovery_retry_policy) if dispatch_error else None,
        )

    except ApprovalRequired:
        await _settle_recovery(
            hooks,
            recovery_item_id,
            rec_identity,
            state="stalled",
            tool_name=tool_name,
            logger=logger,
            last_error="approval required",
            retry_policy=cast(Any, "confirm"),
        )
        raise

    except Exception as exc:  # pylint: disable=broad-except
        state.total_tool_calls += 1
        state.step_tool_calls += 1
        state.called_tool_names.add(tool_name)
        _track_retrieval_call(state, tool_name, tool_arguments)
        ledger.record(tool_name)
        logger.warning("Tool '%s' failed in turn runner: %s", tool_name, exc)
        tool_output = f"Tool execution failed: {type(exc).__name__}: {exc}"
        dispatch_error = str(exc)
        await _settle_recovery(
            hooks,
            recovery_item_id,
            rec_identity,
            state="stalled",
            tool_name=tool_name,
            logger=logger,
            last_error=str(exc),
            retry_policy=cast(Any, recovery_retry_policy),
        )

    duration_ms = int((time.monotonic() - tool_dispatch_started) * 1000)
    is_ok = not dispatch_error

    return CapabilityDispatchOutcome(
        tool_output=tool_output,
        duration_ms=duration_ms,
        is_ok=is_ok,
        operation_result=op_result,
        delegate_yield=delegate_yield,
    )
