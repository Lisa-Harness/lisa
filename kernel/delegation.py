"""Delegate operation input contract.

The :class:`DelegationSpec` is the structured input the pack-level
``delegate`` operation builds from LLM-supplied parameters and hands to
the injected ``delegate_fn``. The model itself lives in
:mod:`kernel.models` alongside the other Pydantic contracts; this module
re-exports it so pack handlers import from ``kernel.delegation`` (the
spec's canonical location) without depending on the full
:mod:`kernel.models` surface.

The default ``delegate_fn`` implementation (``_default_delegate_fn`` in
:mod:`kernel.turn_runner``) runs ``run_turn`` recursively as a
``sub_turn_kind="delegate"`` child, reusing the fork infrastructure
(budget, status mapping, action-tool validation) without fanning out in
parallel.
"""

from __future__ import annotations

from kernel.models import DelegationResult, DelegationSpec, DelegationStatus

__all__ = [
    "DelegationResult",
    "DelegationSpec",
    "DelegationStatus",
]