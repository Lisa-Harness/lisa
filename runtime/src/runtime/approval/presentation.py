"""Approval presentation — the text and payload renders for host approvals.

This module owns the pure, side-effect-free rendering of approval state:
the redacted parameter payload a panel transport shows, the result text of
an executed action, the diagnostic replies for an approve/deny that could
not act, and the effect receipt stamped onto the thread entry. It has no
runtime dependencies beyond the kernel approval models and is importable
without the MCP pool, the manifest, or the thread store.

Extracted from ``handle_message.py`` so the presentation vocabulary of the
approval flow is discoverable in one small file.
"""

from __future__ import annotations

import json
from typing import Any

from kernel.approval import ApprovalIdentity, ApprovalInvocation
from kernel.models import OperationResult, ToolResult

# Parameter keys whose values are withheld from the transport approval payload:
# they are the most likely carriers of secrets or huge bodies, and the panel
# only needs the command itself plus a preview to render.
_APPROVAL_REDACT_KEYS = frozenset({"api_key", "token", "secret", "password", "content", "patch"})

#: Maximum characters rendered for one approval parameter value before
#: truncation with an ellipsis.
_APPROVAL_VALUE_MAX_CHARS = 200


def redact_parameters(parameters: dict[str, Any]) -> dict[str, str]:
    """Redact invocation parameters for the transport approval payload.

    Keeps scalar values (truncated) so the panel can show the command being
    proposed, but replaces sensitive/large keys and any long value with a
    placeholder. Never raises — a payload is display-only.
    """
    out: dict[str, str] = {}
    for key, value in (parameters or {}).items():
        lower = key.lower()
        if lower in _APPROVAL_REDACT_KEYS or any(
            token in lower for token in ("key", "token", "secret", "pass")
        ):
            out[key] = "[redacted]"
            continue
        if isinstance(value, (list, tuple)):
            joined = " ".join(str(v) for v in value)
            text = joined
        else:
            text = str(value)
        out[key] = (
            text
            if len(text) <= _APPROVAL_VALUE_MAX_CHARS
            else text[: _APPROVAL_VALUE_MAX_CHARS - 1] + "…"
        )
    return out


def approval_result_text(result: Any) -> str:
    """Render a direct action result without running a model synthesis pass."""

    output_text = str(getattr(result, "output_text", "") or "").strip()
    if output_text:
        return output_text
    output_json = getattr(result, "output_json", {}) or {}
    if output_json:
        return json.dumps(output_json, ensure_ascii=True, sort_keys=True)
    return "Approved action completed."


def same_session_scope(live: ApprovalIdentity, recorded: ApprovalIdentity) -> bool:
    """Return whether two identities name the same operator session.

    The approval fallback matches the *recorded owner* identity when the
    live rebuild differs (process restarts reset ``current_mode_id`` to the
    start mode, so ``mode_id`` legitimately diverges). Mode divergence is
    safe to bridge — the owning mode is exactly what the live rebuild loses.
    User, workspace, and surface must still agree: a request id from another
    user or surface is not approvable from this one.
    """

    return (
        live.user_slug == recorded.user_slug
        and live.workspace_id == recorded.workspace_id
        and live.surface_id == recorded.surface_id
        and live.surface_type == recorded.surface_type
        and live.memory_scope == recorded.memory_scope
    )


def unresolved_approval_text(request_id: str, record: Any) -> str:
    """Render a diagnostic reply for an approve/deny that could not act.

    ``ApprovalService.approve``/``deny`` return ``None``/``False`` for three
    distinct conditions: a non-pending state (decided, consumed, or expired)
    and an identity mismatch. The persisted record distinguishes them, so the
    reply reports the actual state instead of one ambiguous message — the
    no-logs CLI host otherwise leaves the operator guessing between "already
    decided" and "stranded across a process restart".
    """

    return f"Approval request {request_id} cannot be approved — {decision_state_text(record)}."


def decision_state_text(record: Any) -> str:
    """Render the persisted decision state of an approval record.

    Distinguishes the three non-pending outcomes so ``/approve`` failures are
    diagnosable from the operator reply alone — the CLI host is one process
    per turn with no log tail by default, and "expired" and "already decided"
    previously rendered as the same message.
    """

    state = str(getattr(record, "state", "") or "")
    if state == "expired":
        return "it has expired; repeat the request to get a fresh approval"
    decision = getattr(record, "decision", None)
    decision_state = str(getattr(decision, "state", "") or "") if decision is not None else ""
    if state == "approved" or decision_state == "approved":
        return "it was already approved"
    if state == "denied" or decision_state == "denied":
        return "it was already denied"
    if state == "consumed" or decision_state == "consumed":
        return "it was already executed"
    return f"its state is {state or 'unknown'}"


def approval_action_receipt(
    invocation: ApprovalInvocation, result: ToolResult | OperationResult
) -> dict[str, str]:
    """Build the effect-receipt dict for one approved invocation.

    The completion contract counts receipts, not tool names, so an action
    executed via ``/approve`` must land in the thread ledger exactly as it
    would have in the paused turn — otherwise the next model turn reads a
    history that shows the announced action never ran (the re-issue loop
    observed on tumbleweed3: the model re-planned and re-issued ``whoami``
    even though the approval had already executed it).
    """

    output_json = result.output_json or {}
    return {
        "capability_id": str(invocation.target_id or ""),
        "effect": str(invocation.effect or ""),
        "artifact_id": str(
            output_json.get("artifact_id")
            or output_json.get("document_id")
            or output_json.get("memory_id")
            or ""
        ),
    }


__all__ = [
    "approval_action_receipt",
    "approval_result_text",
    "decision_state_text",
    "redact_parameters",
    "same_session_scope",
    "unresolved_approval_text",
]
