"""Trajectory capture for diffable eval regressions.

A trajectory records every tool invocation a run made (id, parameters,
result fields, error) as a JSON-serializable dict. Eval tests assert on
trajectories the way Inspect-style scorers grade transcripts: outcomes and
state transitions, not prose. Writing the JSON to a scratch directory makes
regressions diffable across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataclasses import dataclass
from kernel.models import ToolResult

__all__ = ["ToolRun", "Trajectory", "write_trajectory"]


@dataclass(slots=True)
class ToolRun:
    """Outcome of one in-process tool invocation."""

    tool_id: str
    result: ToolResult
    error: str = ""



class Trajectory:
    """Ordered record of tool invocations from one run."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._entries: list[dict[str, Any]] = []

    def record(self, run: ToolRun) -> None:
        """Append one tool invocation."""

        self._entries.append(
            {
                "tool_id": run.tool_id,
                "error": run.error,
                "result_error": run.result.error,
                "output_text": run.result.output_text,
                "output_json": run.result.output_json,
            }
        )

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    @property
    def tool_ids(self) -> list[str]:
        return [entry["tool_id"] for entry in self._entries]

    def results_for(self, tool_id: str) -> list[dict[str, Any]]:
        """Return every entry recorded for ``tool_id``."""

        return [e for e in self._entries if e["tool_id"] == tool_id]

    def last_result_for(self, tool_id: str) -> dict[str, Any] | None:
        """Return the most recent entry for ``tool_id``, or ``None``."""

        matches = self.results_for(tool_id)
        return matches[-1] if matches else None

    def first_error(self) -> str:
        """Return the first result-level error message, or ``''``."""

        for entry in self._entries:
            if entry["result_error"]:
                return str(entry["result_error"])
        return ""

    def all_errors_absent(self) -> bool:
        """Whether every step completed with no result-level error."""

        return not any(entry["result_error"] for entry in self._entries)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "steps": self._entries}, indent=2, default=str)


def write_trajectory(trajectory: Trajectory, out_dir: Path) -> Path:
    """Persist a trajectory as JSON and return the file path."""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trajectory-{trajectory.name or 'run'}.json"
    path.write_text(trajectory.to_json(), encoding="utf-8")
    return path