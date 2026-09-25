"""Approval orchestration — the durable-state machinery of host approvals.

This module owns the *flow* of an approval through the host runtime:
resuming a paused approval from a short affirmative, driving the
policy-decided auto-approve path in-process, recording an approved action
in the thread ledger, and settling the paired recovery record so the
workflow unblocks. It does NOT render text (that lives in
``host.approval.presentation``) and it never touches the model — when an
approved action completes it routes the *synthesis* turn through the same
single-turn machinery used for user turns.

The functions here were previously embedded in ``handle_message.py``,
which made that dispatcher a second runtime. They are extracted verbatim
so the dispatcher shrinks back to its documented three-phase contract.

Dependency direction: ``orchestration`` → ``host.runtime``,
``host.approval.presentation``, ``kernel.approval``, ``kernel.recovery``,
``host.policy``, ``host.types``. The command entry point
``handle_approval_command`` takes a ``route_model_turn`` callback so this
module stays testable and free of a circular import on
``host.handle_message``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    ApprovalRequired,
    render_approval_request,
)
from kernel.turn_runner import execute_approved_invocation

from host.approval.presentation import (
    approval_action_receipt,
    approval_result_text,
    redact_parameters,
    same_session_scope,
    unresolved_approval_text,
)
from host.policy import AUTO_APPROVE_TURN_CAP, read_containment, remember_allow_prefix
from host.turn_hooks_builder import build_turn_hooks

if TYPE_CHECKING:
    from host.callbacks import TransportCallbacks
    from host.runtime import HostRuntime
    from host.types import HostCommandResult

logger = logging.getLogger(__name__)

# Callable that routes a continuation query through a fresh model turn in the
# approval's owning mode, returning ``str | HostCommandResult``. Wired by the
# caller (``host.handle_message``) so this module never imports the dispatcher.
RouteModelTurnFn = Callable[["HostRuntime", str, "TransportCallbacks"], Awaitable[Any]]


def approval_identity(runtime: HostRuntime) -> ApprovalIdentity:
    """Build the current host identity used to validate approval ownership."""

    return ApprovalIdentity(
        workspace_id=runtime.session_plan.manifest.name,
        user_id=runtime.config.user_slug,
        user_slug=runtime.config.user_slug,
        surface_type=runtime.config.surface_type,
        surface_id=runtime.config.surface_id,
        memory_scope=runtime.config.memory_scope,
        mode_id=runtime.session.current_mode_id,
        session_plan_digest=runtime.session_plan.plan_digest,
    )


# Short affirmative phrases that, when a single approval is pending, are
# treated as an implicit ``/approve <id>`` rather than a fresh model turn. Kept
# deliberately small and unambiguous — anything longer or question-like is
# routed to the model so the user can still hold a normal conversation
# while an approval is pending.
#
# This set has two tiers:
# - ``_AFFIRMATIVE_EXACT``: conversational affirmatives (``"ok"``, ``"yes"``)
#   that must match the *whole* message. They would false-positive as
#   prefixes (``"ok but…"`` is not a resume).
# - Imperative resume phrases (``"proceed"``, ``"continue"``) live in the
#   shared registry (:func:`kernel.routing_vocabulary.get_resume_phrases`)
#   and are matched as **prefixes** so ``"proceed with the current step in
#   the plan"`` matches ``"proceed"``. The registry is session-authored via
#   ``routing.resume_phrases`` in ``session_context.md``; the historical
#   default is used when no session vocabulary is wired.
_AFFIRMATIVE_EXACT = frozenset(
    {
        "approved",
        "approve",
        "approve it",
        "ok",
        "okay",
        "yes",
        "y",
        "yep",
        "yeah",
        "sure",
        "confirm",
        "confirmed",
    }
)


def is_affirmative_resume(lowered: str) -> bool:
    """Return True when ``lowered`` is a short affirmative resume phrase.

    Matches the whole message against the exact-affirmative set, or the
    message's start against any registered resume phrase (prefix match).
    The resume-phrase vocabulary is session-authored and shared with the
    kernel's continuation-query detector via
    :func:`kernel.routing_vocabulary.get_resume_phrases`.
    """
    if lowered in _AFFIRMATIVE_EXACT:
        return True
    # Prefix match against the shared resume-phrase registry. ``"proceed
    # anyway"`` and ``"continue anyway"`` are historically affirmative —
    # the registry's default list includes ``"proceed"`` and ``"continue"``,
    # so the prefix match covers them without a dedicated entry.
    from kernel.routing_vocabulary import get_resume_phrases

    return any(lowered.startswith(phrase) for phrase in get_resume_phrases())


# Negative resume phrases that, when a single approval is pending, are treated
# as an implicit ``/deny <id>``. Whole-message match only (denial must be
# unambiguous: "no way that breaks" should never deny). Deliberately tiny.
_NEGATIVE_EXACT = frozenset({"no", "n", "nope", "deny", "deny it", "cancel"})

# Approve-and-remember phrases: approve this once AND persist a policy rule so
# future matching runs are auto-approved (the panel's 'a' key sends these).
_REMEMBER_EXACT = frozenset({"a", "always", "approve always", "remember"})


async def try_affirmative_approval_resume(
    runtime: HostRuntime,
    stripped: str,
    lowered: str,
    transport: TransportCallbacks,
    *,
    route_model_turn: RouteModelTurnFn,
) -> HostCommandResult | None:
    """Treat a short decision phrase as an implicit approval decision.

    Maps a one-word reply to the matching command when exactly one approval is
    pending for the current session identity:

    - affirmative ("yes", "proceed", ...) -> ``/approve <id>``
    - negative ("no", "cancel", ...)      -> ``/deny <id>``
    - remember ("a", "always")            -> ``/approve <id> --remember``

    Returns a ``HostCommandResult`` when the message was consumed, or ``None``
    to let normal model-turn routing proceed. When multiple approvals are
    pending the phrase is left for the model and the user must disambiguate
    with an explicit ``/approve <id>`` — guessing would silently execute the
    wrong action.
    """
    if lowered in _REMEMBER_EXACT:
        decision_cmd = "/approve {id} --remember"
    elif lowered in _NEGATIVE_EXACT:
        decision_cmd = "/deny {id}"
    elif is_affirmative_resume(lowered):
        decision_cmd = "/approve {id}"
    else:
        return None
    service = runtime.services.approval_service
    if service is None or not hasattr(service, "list_pending"):
        return None
    try:
        pending = await service.list_pending(approval_identity(runtime))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("approval decision list_pending failed: %s", exc)
        return None
    if len(pending) != 1:
        return None
    request_id = pending[0].invocation.request_id
    final_command = decision_cmd.format(id=request_id)
    logger.debug("approval decision: treating %r as %s", stripped, final_command)
    result_text = await handle_approval_command(
        runtime,
        final_command,
        approve=not decision_cmd.startswith("/deny"),
        transport=transport,
        route_model_turn=route_model_turn,
    )
    from host.types import HostCommandResult as _HostCommandResult

    if isinstance(result_text, _HostCommandResult):
        await transport.on_response(result_text.text)
        return result_text
    await transport.on_response(result_text)
    return _HostCommandResult(text=result_text)


# Parameter keys whose values are withheld from the transport approval payload:
# they are the most likely carriers of secrets or huge bodies, and the panel
# only needs the command itself plus a preview to render.
_APPROVAL_REDACT_KEYS = frozenset({"api_key", "token", "secret", "password", "content", "patch"})


def build_approval_data(runtime: HostRuntime, invocation: ApprovalInvocation) -> dict[str, Any]:
    """Build the structured ``data["approval"]`` payload for one pending action.

    Carries everything a panel transport needs to render a keystroke decision
    without parsing prose, plus the containment the policy engine resolved so
    the panel can badge why it is asking. Non-panel transports see only the
    ``text`` half. Never raises.
    """
    try:
        fact = read_containment(Path(runtime.paths.data_dir), invocation.identity)
        containment = str(fact.get("containment", "unknown") or "unknown")
        reasons = str(fact.get("reasons", "") or "")
    except Exception:  # pylint: disable=broad-except
        containment, reasons = "unknown", ""
    redacted = redact_parameters(getattr(invocation, "parameters", None) or {})
    expires = getattr(invocation, "expires_at", None)
    expires_in_seconds: int | None = None
    if isinstance(expires, datetime):
        with suppress(TypeError):
            expires_in_seconds = max(0, int((expires - datetime.now(UTC)).total_seconds()))
    policy_verdict = ""
    policy_reason = ""
    policy_tier = ""
    try:
        decide = getattr(runtime.services.approval_service, "decide", None)
        if callable(decide):
            decision = decide(invocation)
            policy_verdict = str(getattr(decision, "verdict", "") or "")
            policy_reason = str(getattr(decision, "reason", "") or "")
            policy_tier = str(getattr(decision, "tier", "") or "")
    except Exception:  # pylint: disable=broad-except
        pass
    return {
        "schema": "lisa.approval.v1",
        "request_id": getattr(invocation, "request_id", ""),
        "target_kind": getattr(invocation, "target_kind", ""),
        "target_id": getattr(invocation, "target_id", ""),
        "pack_id": getattr(invocation, "pack_id", ""),
        "effect": getattr(invocation, "effect", ""),
        "preview": getattr(invocation, "safe_preview", "") or "",
        "command": redacted.get("command", ""),
        "parameters": redacted,
        "containment": containment,
        "containment_reasons": reasons,
        "expires_at": expires.isoformat() if hasattr(expires, "isoformat") else str(expires or ""),
        "expires_in_seconds": expires_in_seconds,
        "policy_verdict": policy_verdict,
        "policy_reason": policy_reason,
        "policy_tier": policy_tier,
    }


async def maybe_auto_continue_approval(
    runtime: HostRuntime,
    invocation: ApprovalInvocation,
    transport: TransportCallbacks,
    *,
    route_model_turn: RouteModelTurnFn,
) -> HostCommandResult | None:
    """Drive the policy auto-approved path in-process.

    Returns the continuation ``HostCommandResult`` when the invocation was
    policy-approved (a receipt already minted by ``PolicyApprovalService``) and
    the per-turn cap has room, so the action executes and synthesizes without
    waiting on the operator. Returns ``None`` to fall back to the normal
    still-pending prompt path (ask) or when the cap is exhausted (a runaway
    loop must surface as a prompt, not burn the ledger).
    """
    service = runtime.services.approval_service
    if service is None or not hasattr(service, "get"):
        return None
    request_id = getattr(invocation, "request_id", "")
    try:
        record = await service.get(request_id)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("auto-continue: get(%s) failed: %s", request_id, exc)
        return None
    if record is None or record.state != "approved" or record.receipt is None:
        return None

    # Per-turn cap on unattended auto-approvals. Exceeding it falls back to the
    # prompt so the operator re-engages with a runaway approval loop.
    used = runtime.session.auto_approved_this_turn
    if used >= AUTO_APPROVE_TURN_CAP:
        logger.warning(
            "auto-approve cap (%d) reached; prompting for %s",
            AUTO_APPROVE_TURN_CAP,
            invocation.target_id,
        )
        return None
    runtime.session.auto_approved_this_turn = used + 1

    logger.info(
        "auto-approving policy-contained action %s (receipt %s)",
        invocation.target_id,
        request_id[:8],
    )
    result = await handle_approval_command(
        runtime,
        f"/approve {request_id}",
        approve=True,
        transport=transport,
        route_model_turn=route_model_turn,
    )
    from host.types import HostCommandResult as _HostCommandResult

    if isinstance(result, _HostCommandResult):
        return result
    return _HostCommandResult(text=str(result))


def _approval_action_receipt_for_thread(invocation: ApprovalInvocation) -> dict[str, str]:
    """Thread-form of the approved-invocation effect receipt."""
    return approval_action_receipt(invocation)


async def _record_and_continue_after_approval(
    runtime: HostRuntime,
    invocation: ApprovalInvocation,
    result: Any,
    *,
    transport: TransportCallbacks,
    route_model_turn: RouteModelTurnFn,
) -> HostCommandResult:
    """Record an approved action in the thread, then run a synthesis turn.

    Two durable-state problems with returning the raw action output alone:

    1. No thread record. The paused turn's history ends with the approval
       request; the action's execution is invisible to every later context
       assembly, so the model re-plans from the stale approval prompt and
       re-issues the same command (observed twice on tumbleweed3).
    2. No synthesis. The raw output (e.g. ``root``) lands in the operator's
       terminal, but the model never sees it — the task it announced is
       dropped and the user must re-ask to continue.

    This helper appends a turn entry carrying the executed action and its
    effect receipt, then routes a model turn in the owning mode with the
    action output as the continuation query. The model sees what ran, what
    it produced, and finishes the task it announced. The continuation turn
    reuses the existing single-turn machinery so mode dispatch, persistence,
    and token accounting stay identical to a user-initiated turn.
    """

    from host.thread import (  # lazy: defers host.thread import to approval time
        append_thread_entry,
    )

    action_text = approval_result_text(result)
    owner_mode = invocation.identity.mode_id or runtime.session.current_mode_id or "default"

    # ── Thread record for the approved action ────────────────────────
    # No model call was made (the registry executed the saved invocation
    # directly), so there is no ``turn_input`` snapshot — provenance states
    # the source so /debug does not try to replay a model input that never
    # existed. The receipt makes the write/execute contract countable.
    append_thread_entry(
        thread_path=runtime.paths.thread_path,
        user_slug=runtime.config.user_slug,
        segment_turns=runtime.config.thread_segment_turns,
        mode_id=owner_mode,
        query=f"/approve {invocation.request_id}",
        response_text=action_text,
        provenance={
            "source": "host-approved",
            "finish_reason": "approved-action",
            "model_id": "",
            "approval_request_id": invocation.request_id,
            "target_kind": invocation.target_kind,
            "target_id": invocation.target_id,
        },
        effect_receipts=[_approval_action_receipt_for_thread(invocation)],
        handoff_to=""
        if owner_mode == runtime.session.current_mode_id
        else runtime.session.current_mode_id,
    )

    # ── Continuation turn in the owning mode ─────────────────────────
    # A dedicated thread entry keeps the action output addressable while
    # the synthesis turn carries the query "<target> executed and returned:
    # <output>" so the model treats the output as observed data, not as a
    # fresh user instruction to re-interpret. The continuation is a REPORT,
    # not an invitation to plan more work: this host may be a one-shot CLI
    # process, so any new effectful action the model tries here pauses for
    # another approval the operator must type by hand — the double-approve
    # friction observed on tumbleweed3. Report, then stop.
    continuation_query = (
        f"The approved action `{invocation.target_id}` executed successfully "
        f"(approval {invocation.request_id}) and returned this output:\n\n"
        f"{action_text}\n\n"
        "Report the result to the operator in plain prose. Do not call any "
        "tools in this turn — do not re-run the approved action and do not "
        "start follow-up commands; the operator is watching and can ask for "
        "more work explicitly. End your report with the single most likely "
        "next step as one short prose sentence (no tool calls, no question) "
        'so the operator can say "go" or redirect.'
    )
    runtime.session.current_mode_id = owner_mode
    routed = await route_model_turn(runtime, continuation_query, transport)
    # ``route_model_turn`` returns a ``HostCommandResult`` on the normal
    # turn path and a bare ``str`` on the early-exit paths (budget gate,
    # unsupported mode) — normalize so the command dispatcher always has a
    # string to emit while preserving ``operator_exit``.
    from host.types import HostCommandResult as _HostCommandResult

    if isinstance(routed, _HostCommandResult):
        return routed
    return _HostCommandResult(text=str(routed))


def _command_and_prefix(invocation: ApprovalInvocation, remember: str | None) -> tuple[str, str]:
    """Return (command, allow-rule prefix) for a ``--remember`` write.

    ``remember`` is the explicit ``--remember <prefix>`` value when given; the
    command is the action's command parameter. The default prefix is the
    command's program + first subcommand token (``git status`` from
    ``git status -sb``) — the natural "this thing, repeatedly" scope. Only the
    first two tokens are used so a path/flag never widens the rule.
    """
    params = getattr(invocation, "parameters", None) or {}
    command = ""
    for key in ("command", "cmd", "script"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            command = value.strip()
            break
    argv = params.get("argv")
    if not command and isinstance(argv, (list, tuple)) and argv:
        command = " ".join(str(part) for part in argv).strip()
    if remember and remember.strip():
        return command, remember.strip()
    tokens = command.split()
    prefix = " ".join(tokens[:2]) if tokens else command
    return command, prefix


def _remember_allow_rule(
    runtime: HostRuntime, invocation: ApprovalInvocation, remember: str | None
) -> None:
    """Persist a policy ``allow`` rule for an approved action.

    No-op for actions without a command to remember (no executable prefix to
    scope a rule to) and swallowed on any write error — a failure to persist a
    convenience rule must never break a successful approve.
    """
    command, prefix = _command_and_prefix(invocation, remember)
    if not command or not prefix:
        logger.debug("approve --remember: no command on %s to remember", invocation.target_id)
        return
    try:
        remember_allow_prefix(
            Path(runtime.paths.data_dir),
            tool=str(invocation.target_id or ""),
            command_prefix=prefix,
        )
        logger.info("approve --remember: allow %s prefix %r", invocation.target_id, prefix)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("approve --remember: failed to persist rule: %s", exc)


async def handle_approval_command(
    runtime: HostRuntime,
    command: str,
    *,
    approve: bool,
    transport: TransportCallbacks,
    route_model_turn: RouteModelTurnFn,
) -> str | HostCommandResult:
    """Approve or deny one persisted invocation without asking the model.

    Identity handling is two-stage. The primary path rebuilds the identity
    from the live runtime — the same identity the turn path used, so a
    pending approval inside one process resolves exactly. When that rebuild
    cannot match (mode handoffs reset ``current_mode_id`` across process
    boundaries; the invocation's identity carries the *owning* mode), the
    recorded owner identity on the saved invocation is used instead: the
    record is a host-persisted claim, and the id was minted by the approval
    service that owns this data dir, so the operator's explicit id reference
    is the session-boundary decision. Cross-user references are still
    rejected — the fallback only ever matches the record's own owner.

    Returns a plain ``str`` for the deny and failure paths; a successful
    approve returns a ``HostCommandResult`` from the post-approval
    continuation turn so ``operator_exit`` survives the command dispatch.
    """

    _, _, request_id = command.partition(" ")
    request_id = request_id.strip()
    verb = "approve" if approve else "deny"
    if not request_id:
        return f"Usage: /{verb} <request_id>"

    # `--remember [prefix]` (approve only): on success, persist an allow rule
    # so future runs of this command are auto-approved without a prompt. The
    # host owns the policy write — the CLI only ever sends the command text.
    remember: str | None = None
    if approve and "--remember" in request_id:
        head, _, tail = request_id.partition("--remember")
        request_id = head.strip()
        remember = tail.strip() or None
    if not request_id:
        return f"Usage: /{verb} <request_id>"

    record = await runtime.services.approval_service.get(request_id)
    if record is None:
        return f"No pending approval request named '{request_id}'."

    identity = approval_identity(runtime)
    recorded = getattr(record.invocation, "identity", None)
    identity_fallback_used = recorded is not None and identity != recorded
    if identity_fallback_used:
        if not same_session_scope(identity, recorded):
            return (
                f"Approval request {request_id} belongs to a different session "
                f"(user {record.invocation.identity.user_slug or 'unknown'})."
            )
        logger.info(
            "approval %s: live identity mismatch (mode %r vs owner %r); using recorded owner",
            request_id,
            identity.mode_id,
            recorded.mode_id,
        )
        identity = recorded

    if not approve:
        if await runtime.services.approval_service.deny(request_id, identity):
            # Denying the approval also unblocks the paired recovery record:
            # the original model-turn stall was ``blocks_workflow=True`` and
            # would otherwise re-yield on every subsequent turn. Settle it
            # as cancelled so the workflow can advance.
            await settle_paired_recovery_record(
                runtime,
                record.invocation,
                state="cancelled",
                last_error="denied by operator",
            )
            return f"Denied approval request {request_id}."
        return unresolved_approval_text(request_id, record)

    # Policy ``allow`` has already persisted and approved this exact action
    # so the registry can consume its genuine receipt. Re-approving would
    # reject the record as non-pending and strand an otherwise valid
    # auto-approval. Human approvals remain the normal pending transition.
    if record.state == "approved" and record.receipt is not None:
        receipt = record.receipt
    else:
        receipt = await runtime.services.approval_service.approve(request_id, identity)
        if receipt is None:
            # The service may have mutated the record inside this call (e.g.
            # latched an expired TTL) — re-fetch so the diagnostic reflects the
            # post-attempt state, not the pre-call snapshot.
            fresh = await runtime.services.approval_service.get(request_id)
            return unresolved_approval_text(request_id, fresh or record)

    hooks = build_turn_hooks(runtime, context={}, transport=transport)
    try:
        result = await execute_approved_invocation(invocation=record.invocation, hooks=hooks)
    except ApprovalRequired as required:
        return render_approval_request(required.invocation)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("approved action %s failed: %s", request_id, exc)
        await settle_paired_recovery_record(
            runtime, record.invocation, state="stalled", last_error=f"{type(exc).__name__}: {exc}"
        )
        return f"Approved action failed: {type(exc).__name__}: {exc}"
    # The approved action executed against the real registry, which consumes
    # the receipt. But the *model-turn* path that originally raised
    # ``ApprovalRequired`` also settled a paired recovery record as
    # ``stalled`` with ``blocks_workflow=True`` (turn_runner.py). The approval
    # and recovery stores are independent with no shared id, so approving the
    # action left the stall permanently blocking every subsequent turn. Settle
    # the paired record now so the workflow unblocks.
    await settle_paired_recovery_record(
        runtime,
        record.invocation,
        state="succeeded",
        result_preview=approval_result_text(result)[:240],
    )

    # `--remember`: persist an allow rule so future runs of this command are
    # auto-approved. Default prefix = the executed command's first token (the
    # program / program+subcommand, e.g. "git status" from "git status -sb"),
    # which is the natural "this thing, repeatedly" scope; an explicit flag
    # value overrides it. Host-side write — the CLI never edits policy.yaml.
    if approve and "--remember" in command:
        _remember_allow_rule(runtime, record.invocation, remember)

    # ── Post-approval continuation ───────────────────────────────────
    # Return the raw action output alone and the model never learns the
    # action ran: no thread record exists, the next turn re-plans from the
    # stale approval prompt, and the user must re-ask to continue (the
    # tumbleweed3 re-issue loop). Record the action and run a synthesis
    # turn in the owning mode instead.
    return await _record_and_continue_after_approval(
        runtime,
        record.invocation,
        result,
        transport=transport,
        route_model_turn=route_model_turn,
    )


async def settle_paired_recovery_record(
    runtime: HostRuntime,
    invocation: ApprovalInvocation,
    *,
    state: str,
    last_error: str = "",
    result_preview: str = "",
) -> None:
    """Settle the recovery record paired to an approved invocation.

    The model-turn path raises ``ApprovalRequired`` *after* the recovery
    service has already ``begin``-ed a record for the same target. Approving
    via ``/approve`` re-executes through a fresh registry path that does not
    touch the recovery service, so the original record stays ``stalled`` —
    the stalled record that re-yields on every subsequent turn. This
    correlates by ``target_id`` + identity and settles the first open
    stalled match. Correlation is deliberately by target id, not approval
    request id: the two stores never share an id.
    """

    recovery_service = runtime.services.recovery_service
    if recovery_service is None or not hasattr(recovery_service, "list_open"):
        return
    recovery_identity = _recovery_identity_for_invocation(runtime, invocation)
    if recovery_identity is None:
        return
    try:
        open_records = await recovery_service.list_open(recovery_identity)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("paired-recovery list_open failed for %s: %s", invocation.request_id, exc)
        return
    for open_record in open_records:
        if open_record.target_id != invocation.target_id:
            continue
        if getattr(open_record, "state", "") != "stalled":
            continue
        try:
            await recovery_service.settle(
                open_record.work_item_id,
                recovery_identity,
                state=state,  # type: ignore[arg-type]
                last_error=last_error,
                result_preview=result_preview,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "paired-recovery settle failed for %s (work item %s): %s",
                invocation.request_id,
                open_record.work_item_id,
                exc,
            )
        return


def _recovery_identity_for_invocation(
    runtime: HostRuntime,
    invocation: ApprovalInvocation,
) -> Any | None:
    """Build the recovery identity scoped to an approval invocation.

    ``ApprovalIdentity`` and ``RecoveryIdentity`` carry the same scoping
    fields (``RecoveryIdentity`` dropped ``session_plan_digest`` when the
    checkpoint layer was removed — see CHANGELOG "Trimmed RecoveryRecord").
    The approval invocation still carries a plan digest, so a stale approval
    from a prior session plan returns ``None`` and the caller skips
    settlement safely rather than clearing another session's stall.
    """

    from kernel.recovery import RecoveryIdentity

    if invocation.session_plan_digest != runtime.session_plan.plan_digest:
        return None
    identity = invocation.identity
    return RecoveryIdentity(
        workspace_id=identity.workspace_id,
        user_id=identity.user_id,
        user_slug=identity.user_slug,
        surface_type=identity.surface_type,  # type: ignore[arg-type]
        surface_id=identity.surface_id,
        memory_scope=identity.memory_scope,  # type: ignore[arg-type]
    )


# ── Payload/redaction helpers re-exported for convenience ────────────
# ``redact_parameters`` is defined in ``presentation`` but used by
# ``build_approval_data``; re-exports keep the public surface discoverable.
__all__ = [
    "approval_identity",
    "build_approval_data",
    "handle_approval_command",
    "is_affirmative_resume",
    "maybe_auto_continue_approval",
    "redact_parameters",
    "settle_paired_recovery_record",
    "try_affirmative_approval_resume",
]
