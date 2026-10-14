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

``approval`` → ``runtime.approval.presentation``, ``runtime.runtime``,
``runtime.policy``, ``runtime.thread``, ``kernel.approval``,
``kernel.recovery``  (approval imports these; none of them import approval)

The two modules
---------------
- :mod:`runtime.approval.presentation` — pure, side-effect-free text and
  payload rendering (:func:`redact_parameters`,
  :func:`unresolved_approval_text`, ...).
- :mod:`runtime.approval.orchestration` — the durable-state flow: identity
  rebuild, affirmative-resume detection, policy auto-continue, thread
  record + synthesis turn, paired-recovery settlement, the typed core
  :func:`decide_approval`, and its slash-grammar parse adapter
  :func:`handle_approval_command`.

Public surface
--------------
- :func:`approval_identity`
- :func:`build_approval_data`
- :func:`decide_approval`
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
   ``decide_approval``, ``handle_approval_command``,
   ``try_affirmative_approval_resume``, and
   ``maybe_auto_continue_approval`` take a ``route_model_turn`` callable.
   ``runtime.handle_message`` injects its own ``_route_model_turn`` so the
   post-approval synthesis turn reuses the single-turn machinery without
   a circular import. ``decide_approval`` is the typed seam (values in, no
   command string); ``handle_approval_command`` is its slash-grammar parse
   adapter for the string dispatcher.
"""

from __future__ import annotations

from runtime.approval.orchestration import (
    approval_identity,
    build_approval_data,
    decide_approval,
    handle_approval_command,
    is_affirmative_resume,
    maybe_auto_continue_approval,
    settle_paired_recovery_record,
    try_affirmative_approval_resume,
)
from runtime.approval.presentation import (
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
    "decide_approval",
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
