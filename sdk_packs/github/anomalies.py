"""Extension-owned anomaly detectors for cached GitHub telemetry."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionAnomalyDetector, ExtensionRuntimeContext
from kernel.models import AnomalyAlert, UnifiedState
from sdk_packs.github.models import GitHubStatus


async def detect_github_anomalies(state: UnifiedState) -> list[AnomalyAlert]:
    """Return low-cost anomaly alerts derived from cached GitHub snapshots."""

    snapshot = state.signal_snapshots.get("github")
    if snapshot is None:
        return []

    github = GitHubStatus.model_validate(snapshot)
    if github.main_workflow_status != "failure":
        return []

    return [
        AnomalyAlert(
            severity="warn",
            source="github",
            message=f"Main branch CI is FAILING. Last commit: {github.latest_commit_message}",
            detected_at="",
        )
    ]


def build_anomaly_detectors(
    runtime_context: ExtensionRuntimeContext,
) -> list[ExtensionAnomalyDetector]:
    """Expose the GitHub-owned anomaly detectors to heartbeat."""

    del runtime_context
    return [detect_github_anomalies]


__all__ = ["build_anomaly_detectors", "detect_github_anomalies"]

