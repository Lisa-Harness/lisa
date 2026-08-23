"""Typed operator signals for host-kernel communication.

Defines the :class:`OperatorSignal` enum that replaces the bare ``str`` on
``HostRuntime.operator_signal_value``. The kernel polls the signal once per
iteration via ``TurnHooks.operator_signal``; the transport host writes it
from key bindings and clears it at the start of each turn.

The enum inherits from ``str`` so all existing ``== "pause"`` / ``== ""``
comparisons in the kernel continue to work without modification. New code
should use ``is OperatorSignal.PAUSE`` for identity-based comparisons.
"""

from __future__ import annotations

from enum import Enum


class OperatorSignal(str, Enum):
    """A structured operator signal sent from the transport host to the turn runner.

    Values:
        NONE: No signal — the turn should proceed normally.
        PAUSE: Freeze the turn in place. The kernel holds the turn open.
            Resume by sending any input or pressing the pause key again.
        STOP: Return the prompt but keep the workflow state. The transport
            returns the prompt so the next message is a fresh turn.
        TERMINATE: Kill the workflow. The kernel marks the WorkflowPosition
            complete and returns the prompt. No resume.
        INTERRUPT: Cooperative cancel requested (Ctrl+C first press). The
            kernel cancels the in-flight turn and returns the prompt. The
            MCP pool survives.
        EOF: End-of-input (Ctrl+D, or Ctrl+Z on Windows). Sent when the
            transport wants to exit the session cleanly.
        SUSPEND: Suspend the process (Ctrl+Z on POSIX). The transport
            flushes and sends SIGTSTP. Ignored on Windows.
    """

    NONE = ""
    PAUSE = "pause"
    STOP = "stop"
    TERMINATE = "terminate"
    INTERRUPT = "interrupt"
    EOF = "eof"
    SUSPEND = "suspend"

    def __bool__(self) -> bool:
        """An ``OperatorSignal`` is truthy when it is not ``NONE``."""
        return self is not OperatorSignal.NONE


__all__ = ["OperatorSignal"]
