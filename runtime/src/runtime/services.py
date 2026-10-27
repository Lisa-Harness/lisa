"""HostServices — the host's durable service handles.

Owns the five host-persisted services every turn and command can reach
without recomputing: the policy-deciding approval service, the recovery
store, the code-change artifact store, and the token-usage and
decision-telemetry appenders.

Extracted from ``runtime.runtime`` so service construction and identity live
in one visible place instead of being assembled inline in the runtime
``__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernel.decision_telemetry import DecisionTelemetryHandler
from kernel.token_usage import TokenUsageTelemetryHandler
from sdk.layout import LOGS_DIR, STATE_DIR

from runtime.approvals import FileApprovalService
from runtime.code_artifacts import FileCodeChangeArtifactStore
from runtime.policy_service import PolicyApprovalService
from runtime.recovery import FileRecoveryService


@dataclass(slots=True)
class HostServices:
    """Host-persisted service handles, constructed once per runtime.

    The approval service is a decorator stack: the durable
    ``FileApprovalService`` wrapped by the policy engine
    (``PolicyApprovalService``), which pre-decides each invocation (allow /
    ask / deny) and mints receipts for auto-approvals. The on-disk receipt
    contract is owned by the inner file store; the policy layer only
    decides.
    """

    approval_service: PolicyApprovalService
    recovery_service: FileRecoveryService
    code_change_artifact_store: FileCodeChangeArtifactStore
    token_usage_handler: TokenUsageTelemetryHandler
    decision_telemetry_handler: DecisionTelemetryHandler

    @classmethod
    def build(cls, data_dir: Path) -> HostServices:
        """Construct the service stack rooted at the host data directory."""
        return cls(
            approval_service=PolicyApprovalService(
                FileApprovalService(data_dir / STATE_DIR / "approvals.json"),
                data_dir=data_dir,
            ),
            recovery_service=FileRecoveryService(data_dir / STATE_DIR / "recovery-items.json"),
            code_change_artifact_store=FileCodeChangeArtifactStore(
                data_dir / STATE_DIR / "code-change-artifacts.json"
            ),
            token_usage_handler=TokenUsageTelemetryHandler(
                data_dir / LOGS_DIR / "token-usage.jsonl"
            ),
            decision_telemetry_handler=DecisionTelemetryHandler(
                data_dir / LOGS_DIR / "decisions.jsonl"
            ),
        )


__all__ = ["HostServices"]
