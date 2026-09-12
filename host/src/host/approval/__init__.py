"""Host approval flow — transport-neutral approval and resume machinery.

This package owns the host side of the exact-action approval contract.
It was extracted from ``handle_message.py`` so the dispatcher shrinks back
to its documented three-phase pipeline (command dispatch → approval resume
→ model turn) instead of embedding a 600-line approval subsystem.

Dependency direction
--------------------
``approval`` ← ``handle_message.py`` / transport hosts  (callers import
from approval; approval never imports handle_message — it receives
``route_model_turn`` as an injected callable)

``approval`` → ``host.approval.presentation``, ``host.runtime``,
``host.policy``, ``host.thread``, ``kernel.approval``,
``kernel.recovery``  (approval imports these; none of them import approval)

The two modules
---------------
- :mod:`host.approval.presentation` — pure, side-effect-free text and
  payload rendering (:func:`redact_parameters`,
  :func:`unresolved_approval_text`, ...).
- :mod:`host.approval.orchestration` — the durable-state flow: identity
  rebuild, affirmative-resume detection, policy auto-continue, thread
  record + synthesis turn, paired-recovery settlement, and the dispatch
  entry point :func:`handle_approval_command`.

Public surface
--------------
- :func:`approval_identity`
- :func:`build_approval_data`
- :func:`handle_approval_command`
- :func:`is_affirmative_resume`
- :func:`maybe_auto_continue_approval`
- :func:`try_affirmative_approval_resume`
- :func:`settle_paired_recovery_record`
- :func:`redact_parameters`
- :func:`approval_result_text`
- :func:`same_session_scope`
- :func:`unresolved_approval_text`
- :func:`decision_state_text`
- :func:`approval_action_receipt`

.. note::
   ``handle_approval_command``, ``try_affirmative_approval_resume``, and
   ``maybe_auto_continue_approval`` take a ``route_model_turn`` callable.
   ``host.handle_message`` injects its own ``_route_model_turn`` so the
   post-approval synthesis turn reuses the single-turn machinery without
   a circular import.
"""

from __future__ import annotations

from host.approval.orchestration import (
    approval_identity,
    build_approval_data,
    handle_approval_command,
    is_affirmative_resume,
    maybe_auto_continue_approval,
    settle_paired_recovery_record,
    try_affirmative_approval_resume,
)
from host.approval.presentation import (
    approval_action_receipt,
    approval_result_text,
    decision_state_text,
    redact_parameters,
    same_session_scope,
    unresolved_approval_text,
)

__all__ = [
    "approval_action_receipt",
    "approval_identity",
    "approval_result_text",
    "build_approval_data",
    "decision_state_text",
    "handle_approval_command",
    "is_affirmative_resume",
    "maybe_auto_continue_approval",
    "redact_parameters",
    "same_session_scope",
    "settle_paired_recovery_record",
    "try_affirmative_approval_resume",
    "unresolved_approval_text",
]
