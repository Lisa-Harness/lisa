"""GitHub processing loop — anomaly detection from cached snapshots.

Folds the former ``anomalies.py`` anomaly detector into a ``ProcessingLoop``
that runs after signal collection during the bottom-half tick.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.intake.contracts import ProcessContext, ProcessedState
from kernel.models import AnomalyAlert
from lisa_pack_github.models import GitHubStatus


class GitHubProcessingLoop:
    """Anomaly detection for cached GitHub telemetry."""

    loop_id = "github"

    async def process(self, ctx: ProcessContext) -> ProcessedState:
        """Examine the GitHub snapshot and emit alerts when CI is failing."""
        if ctx.state is None:
            return ProcessedState(loop_id=self.loop_id)

        snapshot = ctx.state.signal_snapshots.get("github")
        if snapshot is None:
            return ProcessedState(loop_id=self.loop_id)

        github = GitHubStatus.model_validate(snapshot)
        if github.main_workflow_status != "failure":
            return ProcessedState(loop_id=self.loop_id)

        return ProcessedState(
            loop_id=self.loop_id,
            active_alerts=[
                AnomalyAlert(
                    severity="warn",
                    source="github",
                    message=f"Main branch CI is FAILING. Last commit: {github.latest_commit_message}",
                    detected_at=datetime.now(UTC).isoformat(),
                )
            ],
        )


__all__ = ["GitHubProcessingLoop"]