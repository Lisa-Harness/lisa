"""Answer replayed tool dispatches from the recorded transcript (Spec 3 slice 3).

Exact-mode replay never re-executes a tool: the recorded transcript's
``role="tool"`` messages already hold the output the model saw, keyed by
``tool_call_id``. :class:`TranscriptToolAnswerer` binds that map onto the
kernel's ``ToolHooks.answer_tool_call`` seam (wired through
``SessionState.tool_call_answerer``), so a replayed dispatch is answered
from the recording and the loop behaves as if the tool had executed.

A dispatch whose ``tool_call_id`` has no recorded result is a divergence —
the model called a tool the recording never produced, or the archive is
incomplete. The gap is recorded and raised; the engine converts it into a
report entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ReplayToolGapError(RuntimeError):
    """A replayed dispatch had no recorded result for its ``tool_call_id``."""

    def __init__(self, tool_name: str, tool_call_id: str) -> None:
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        super().__init__(
            f"replayed dispatch of '{tool_name}' (tool_call_id={tool_call_id}) "
            "has no recorded result; the recording is incomplete"
        )


@dataclass
class TranscriptToolAnswerer:
    """Answer ``answer_tool_call`` dispatches from the recorded transcript."""

    tool_results: dict[str, str]
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def answer(self, tool_name: str, tool_arguments: dict[str, Any], tool_call_id: str) -> str:
        """Return the recorded output for ``tool_call_id``, or record a gap."""
        _ = tool_arguments
        if tool_call_id not in self.tool_results:
            self.gaps.append({"tool": tool_name, "tool_call_id": tool_call_id})
            raise ReplayToolGapError(tool_name, tool_call_id)
        return self.tool_results[tool_call_id]
