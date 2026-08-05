"""Recovery pack for Lisa.

Provides operator-facing slash commands for inspecting and resolving
interrupted dispatch work items recorded by the kernel's recovery
service:

- ``/stalls`` — list open (stalled, indeterminate, running) work items.
- ``/stall <id>`` — show details for one work item.
- ``/retry <id>`` — retry one stalled item (approval-gated for
  effectful/indeterminate actions).
- ``/retry-safe`` — retry all safe (read-only) stalled items.
- ``/skip <id>`` — mark one item as skipped, unblocking the workflow.
- ``/cancel <id>`` — mark one item as cancelled, unblocking the workflow.

The kernel owns the recovery contracts (``kernel.recovery``) and the
write-ahead lifecycle instrumentation in the turn runner. The host
(``host/recovery.py``) owns the durable file-backed store. This
pack owns the operator UX: rendering stall summaries, gating retries
through approval, and deterministic replay of saved invocations.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
