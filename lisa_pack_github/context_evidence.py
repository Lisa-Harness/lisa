"""Extension-owned context evidence for GitHub."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.models import ContextEvidence, UnifiedState
from lisa_pack_github.models import GitHubStatus
from sdk.config import settings


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _default_collected_at(state: UnifiedState) -> str:
    return state.compiled_at or datetime.now(UTC).isoformat()


def _freshness_expires_at(collected_at: str) -> str:
    collected_at_dt = _parse_iso(collected_at) or datetime.now(UTC)
    ttl = timedelta(seconds=max(settings.heartbeat_interval_seconds * 2, 60))
    return (collected_at_dt + ttl).isoformat()


def build_context_evidence(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ContextEvidence]:
    """Expose GitHub context evidence without host-owned summary branches."""

    del runtime_context
    raw_snapshot = state.signal_snapshots.get("github")
    if raw_snapshot is None:
        return []

    github = GitHubStatus.model_validate(raw_snapshot)
    collected_at = _default_collected_at(state)

    evidence_collected_at = github.collected_at or collected_at
    return [
        ContextEvidence(
            evidence_id="connector:github",
            source="github",
            connector_id="github",
            evidence_type="connector_status",
            title="GitHub delivery state",
            body=(
                f"Workflow {github.main_workflow_status.upper()} | "
                f"latest={github.latest_commit_sha} {github.latest_commit_message}"
            ).strip(),
            collected_at=evidence_collected_at,
            freshness_expires_at=_freshness_expires_at(evidence_collected_at),
            confidence=0.8,
            salience_score=0.9 if github.main_workflow_status == "failure" else 0.45,
            tags=["github", "ci", github.main_workflow_status],
        )
    ]


__all__ = ["build_context_evidence"]


