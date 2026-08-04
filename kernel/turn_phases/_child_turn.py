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

DESIGN — re-exported by ``kernel.turn_runner``:
    Existing tests import fork helpers from ``kernel.turn_runner``. That
    module re-exports the symbols below so no test file needs editing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

    # 1. Blocked (capability gate / input guard / output guard).
    if child_result.blocked:
        _block_reason = child_result.block_origin or "output_guard"
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
            block_diagnostic=BlockDiagnostic(
                reason=_block_reason,  # type: ignore[arg-type]
                message=child_result.response_text,
            ),
        )

    # 2. Model yield (the model called the ``yield`` tool voluntarily).
    if child_result.yielded is not None:
        _ya = child_result.yielded
        if _ya.context == "model-emitted yield":
            _clog.info(
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
        )

    # 4. Completion-contract validation.
    # A mode with no side-effecting completion contract (empty
    # ``action_tool_ids``) is ``complete`` on any clean exit — its output is
    # the response text (the reasoning / opinion), not a side effect.
    if not action_tool_ids:
        return ChildTurnOutcome(
            status="complete",
            output_text=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
        )

    # A mode with a completion contract must produce evidence it invoked a
    # side-effecting action tool. Evidence comes from two sources:
    #
    # 1) ``EffectReceipt`` — collected at dispatch time from the capability's
    #    declared ``effect`` on a *successful* write. This is the primary,
    #    strongest evidence: it only fires when a write actually landed
    #    (a failed write emits no receipt), so a read-only or failed call
    #    never satisfies the contract.
    #
    # 2) ``called_tool_names`` matching an action tool — the *fallback* for
    #    opaque MCP tools (e.g. ``lisa-storage.write_file``) that are not
    #    present in the in-process tool catalog and therefore cannot declare
    #    an effect or emit a receipt at dispatch. ``mode_completion_effects``
    #    already restricts the action set to write-effect tools (filtering
    #    out resolvable read tools), so a name match against an action id is
    #    a write-intent match. ``bare_tool_name`` strips the ``server_id.``
    #    prefix so ``lisa-storage.write_file`` matches the authored ``write_file``.
    #
    # Receipt ``capability_id`` is recorded as the dispatched namespaced MCP
    # name; ``bare_tool_name`` normalizes it to match the bare action id.
    # lazy: avoids top-level MCP import
    from kernel.tools.mcp_client import bare_tool_name  # noqa: PLC0415

    called_bare = {bare_tool_name(n) for n in child_result.called_tool_names}
    receipt_ids = {bare_tool_name(r.capability_id) for r in child_result.effect_receipts}
    has_write_effect = bool(receipt_ids & action_tool_ids) or bool(called_bare & action_tool_ids)

    if has_write_effect:
        return ChildTurnOutcome(
            status="complete",
            output_text=child_result.response_text,
            duration_ms=elapsed_ms,
            tool_call_count=child_result.tool_call_count,
            called_tool_names=tuple(sorted(child_result.called_tool_names)),
            effect_receipts=tuple(child_result.effect_receipts),
        )

    # No write-effect receipt and the mode requires one. Distinguish
    # zero-call hallucination from read-only gather for honest messaging.
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
    )