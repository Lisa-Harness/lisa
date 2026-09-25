"""Shared child-turn executor and classifier for fork and delegate paths.

Both the fork fan-out (``kernel.turn_phases._fork._run_fork_child_turn``) and
the default delegate implementation (``kernel.turn_runner._default_delegate_fn``)
recurse into ``run_turn`` as a child turn, then classify the child's
``TurnResult`` into a status. Historically those two paths duplicated the
classifier and diverged:

- The fork classifier never read ``TurnResult.blocked``, so a
  capability-gated child was misreported as ``no_tools`` ("hallucinated
  success") instead of ``blocked`` ("router declined to escalate") — the
  2026-08-04 audit finding 3.1.
- The action-tool set was derived from ``authored_allowed_*`` only, missing
  operations admitted via their own ``mode_allowlist`` — finding 4.1.
- The status Literals and result models (``ForkChildResult`` /
  ``DelegationResult``) drifted apart.

This module owns the single :func:`classify_child_turn` that both paths call.
It produces a :class:`ChildTurnOutcome` (``kernel.models``), which each
adapter maps to its host-facing model. The classification order is fixed:

1. ``blocked`` — capability gate / input guard / output guard.
2. ``model_yield`` — the model called the ``yield`` tool voluntarily.
3. ``interrupted`` — the harness force-yielded (iteration cap / budget / stuck).
4. ``timeout`` / ``error`` / ``cancelled`` — exception-based outcomes.
5. ``complete`` / ``no_tools`` — completion-contract validation via receipts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kernel.models import (
    BlockDiagnostic,
    ChildTurnOutcome,
)

if TYPE_CHECKING:
    from kernel.turn_runner import TurnResult

__all__ = [
    "classify_child_turn",
]


def classify_child_turn(
    *,
    child_result: TurnResult,
    mode_id: str,
    elapsed_ms: int,
    action_tool_ids: set[str],
) -> ChildTurnOutcome:
    """Classify one child ``TurnResult`` into a canonical ``ChildTurnOutcome``.

    Called by both the fork child executor and the default delegate function
    after ``run_turn`` returns. The ``action_tool_ids`` are the mode's
    required side-effecting capability IDs from
    :func:`kernel.operations.registry.mode_completion_effects` — the single
    canonical source. An empty set means the mode has no side-effecting
    completion contract (chat / reasoning modes); any clean exit is
    ``complete``.

    The ``child_result.blocked`` check is first so a capability-gated child
    surfaces ``blocked`` with its diagnostic, never ``no_tools``. This is the
    fix for the 2026-08-04 audit finding 3.1: the old classifier fell through
    to ``tool_call_count == 0 → no_tools`` and the parent reported "model too
    weak to follow the sub-turn contract" when the actual cause was the router
    declining to escalate to a code-capable model.
    """
    _clog = logging.getLogger(__name__)

    # Carry the child's per-model / per-mode token usage onto the outcome so
    # the fork/delegate adapters can propagate it to ``ForkChildResult`` /
    # ``DelegationResult`` and the parent turn can merge it into its own
    # ``_state`` ledger. Without this, ``/tokens`` attributes every fork /
    # delegate child's tokens to the parent's resolved default model — the
    # 2026-08-06 regression: a multi-model turn showed only one model.
    _usage_kwargs: dict[str, Any] = {
        "per_model_usage": tuple(
            dict(entry) for entry in getattr(child_result, "per_model_usage", []) or []
        ),
        "per_mode_usage": tuple(
            dict(entry) for entry in getattr(child_result, "per_mode_usage", []) or []
        ),
    }

    # 1. Blocked (capability gate / input guard / output guard / lifecycle).
    # ``block_origin`` is empty for non-guard lifecycle failures (entry/plan/
    # fork failures that set ``blocked=True`` directly). Default to
    # ``lifecycle_failure`` — not ``output_guard`` — so the parent surfaces
    # the actual cause instead of misattributing an entry-op failure to the
    # output guard (the 2026-08-04 audit finding 4).
    if child_result.blocked:
        _block_reason = child_result.block_origin or "lifecycle_failure"
        if _block_reason == "capability_gate":
            _clog.warning(
                "child turn mode=%s was blocked by the capability gate "
                "(router declined to escalate); classifying as blocked "
                "(not no_tools)",
                mode_id,
            )
        else:
            _clog.warning(
                "child turn mode=%s was blocked by the %s; classifying as blocked",
                mode_id,
                _block_reason,
            )
        return ChildTurnOutcome(
            status="blocked",
            output_text=child_result.response_text,
            error_message=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
            block_diagnostic=child_result.block_diagnostic
            or BlockDiagnostic(
                reason=_block_reason,  # type: ignore[arg-type]
                message=child_result.response_text,
            ),
            **_usage_kwargs,
        )

    # 2. Model yield (the model called the ``yield`` tool voluntarily).
    if child_result.yielded is not None:
        _ya = child_result.yielded
        if _ya.context == "model-emitted yield":
            _clog.debug(
                "child turn mode=%s model-emitted yield; propagating question "
                "to parent (not logged to avoid echoing user content)",
                mode_id,
            )
            return ChildTurnOutcome(
                status="model_yield",
                output_text=child_result.response_text,
                duration_ms=elapsed_ms,
                tool_call_count=child_result.tool_call_count,
                called_tool_names=tuple(sorted(child_result.called_tool_names)),
                effect_receipts=tuple(child_result.effect_receipts),
                yield_action=_ya,
                **_usage_kwargs,
            )
        # 3. Interrupted (harness force-yielded: iteration cap / budget / stuck).
        _clog.warning(
            "child turn mode=%s was force-yielded by the harness "
            "(iteration cap / stuck-in-stage / budget); marking "
            "interrupted (not hallucinated)",
            mode_id,
        )
        return ChildTurnOutcome(
            status="interrupted",
            output_text=child_result.response_text,
            error_message=(
                f"child turn {mode_id} was interrupted by the turn lifecycle "
                f"(force-yielded) before completing its action tools; its "
                f"reported output is partial."
            ),
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
            yield_action=_ya,
            **_usage_kwargs,
        )

    # 4. Completion-contract validation.
    # A mode with no side-effecting completion contract (empty
    # ``action_tool_ids``) is ``complete`` on any clean exit — its output is
    # the response text (the reasoning / opinion), not a side effect.
    if not action_tool_ids:
        _clog.debug(
            "child turn mode=%s complete (no action tools declared)",
            mode_id,
        )
        return ChildTurnOutcome(
            status="complete",
            output_text=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
            **_usage_kwargs,
        )

    # A mode with a completion contract must produce an ``EffectReceipt``
    # proving a side-effecting action tool actually landed its write.
    # Receipts are collected at dispatch time on *successful* dispatches —
    # a failed write (``ToolResult.error`` non-empty or MCP ``is_error``)
    # emits no receipt, so a failed call never satisfies the contract.
    # Read/execute receipts prove real read-only work (the 2026-08-27
    # read-receipt fix): a child that only read did honest non-write work,
    # and a read-only turn never trips the write contract.
    # Opaque MCP write tools (not in the in-process catalog) also emit a
    # receipt on success via the ``action_tool_ids`` membership check at
    # dispatch time, so the receipt is the single source of truth.
    #
    # A bare ``called_tool_names`` match is NOT sufficient: it records the
    # attempted name before success is known, so a failed MCP write would
    # false-positive as completion. Only receipts count.
    # lazy: avoids top-level MCP import
    from kernel.tools.mcp_client import bare_tool_name  # noqa: PLC0415

    # Match the receipt against the action set using BOTH the raw
    # capability id and its bare form. ``bare_tool_name`` strips a
    # ``server_id.`` namespace prefix (``lisa-storage.write_file`` →
    # ``write_file``) so a namespaced MCP write matches its bare authored
    # action id; but it ALSO strips the dot from synthetic dotted tool
    # names (``memory.add`` → ``add``), which would wrongly fail to match
    # the authored ``memory.add``. Including the raw id covers the
    # synthetic case; including the bare form covers the namespaced-MCP
    # case. Mirrors the parent-turn receipt check in ``turn_runner.py``.
    write_receipt_ids = {
        bare_tool_name(r.capability_id)
        for r in child_result.effect_receipts
        if r.effect in ("write", "external_write")
    } | {
        r.capability_id
        for r in child_result.effect_receipts
        if r.effect in ("write", "external_write")
    }
    has_write_effect = bool(write_receipt_ids & action_tool_ids)

    if has_write_effect:
        _clog.debug(
            "child turn mode=%s complete (write-effect receipt verified)",
            mode_id,
        )
        return ChildTurnOutcome(
            status="complete",
            output_text=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
            **_usage_kwargs,
        )

    # No write-effect receipt. Check for honest read-only work before
    # classifying: a child with read/execute receipts did real non-write
    # work and never attempted a write — the write contract does not apply
    # to a read-only turn. Only a write ATTEMPT (name dispatched, receipt
    # absent because the dispatch failed) keeps the strict path: the model
    # may claim success on the failed write.
    read_work = any(r.effect in ("read", "execute") for r in child_result.effect_receipts)
    called_bare = {bare_tool_name(n) for n in child_result.called_tool_names} | set(
        child_result.called_tool_names
    )
    write_attempted = bool(called_bare & action_tool_ids)
    if read_work and not write_attempted:
        _clog.debug(
            "child turn mode=%s complete (read-only work verified via "
            "read/execute receipts; write contract not applicable)",
            mode_id,
        )
        return ChildTurnOutcome(
            status="complete",
            output_text=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
            **_usage_kwargs,
        )

    # No write-effect receipt and the mode requires one. Distinguish
    # zero-call hallucination from failed-work for honest messaging.
    has_dense_content = len(child_result.response_text.strip()) > 150
    if has_dense_content:
        _clog.info(
            "child turn mode=%s completed with no verified write, but "
            "produced dense content (%d chars); accepting as successful.",
            mode_id,
            len(child_result.response_text.strip()),
        )
        return ChildTurnOutcome(
            status="complete",
            output_text=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
            **_usage_kwargs,
        )

    if child_result.tool_call_count == 0:
        _clog.warning(
            "child turn mode=%s completed with 0 tool calls but its mode "
            "declares action tools %s; marking no_tools (hallucinated success)",
            mode_id,
            sorted(action_tool_ids),
        )
    else:
        _clog.warning(
            "child turn mode=%s made %d tool call(s) but none produced a "
            "write-effect receipt for its mode's action tools %s; marking "
            "no_tools (hallucinated success — nothing written/learned)",
            mode_id,
            child_result.tool_call_count,
            sorted(action_tool_ids),
        )
    return ChildTurnOutcome(
        status="no_tools",
        output_text=child_result.response_text,
        error_message=(
            f"child turn {mode_id} did not invoke any of its mode's action "
            f"tools ({', '.join(sorted(action_tool_ids))}); its reported "
            "success is unverified — nothing was written to disk or memory."
        ),
        duration_ms=elapsed_ms,
        tool_call_count=child_result.tool_call_count,
        called_tool_names=tuple(sorted(child_result.called_tool_names)),
        effect_receipts=tuple(child_result.effect_receipts),
        **_usage_kwargs,
    )
