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
from sdk.filelock import file_lock, sidecar_lock_path
from sdk.layout import LOGS_DIR, STATE_DIR

from runtime.approvals import FileApprovalService
from runtime.code_artifacts import FileCodeChangeArtifactStore
from runtime.policy_service import PolicyApprovalService
from runtime.recovery import FileRecoveryService

#: Telemetry is best-effort: rather than stalling a live turn behind a stuck
#: lock holder for the file-lock default (30s), the ledger drops the line
#: after a short wait. Cross-process contention between two appending hosts
#: resolves in milliseconds; seconds of contention means something is wrong.
_TELEMETRY_LOCK_TIMEOUT_S = 5.0


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
        """Construct the service stack rooted at the host data directory.

        The two telemetry appenders are serialized across processes through
        sidecar advisory locks: multiple Lisa processes sharing one data dir
        append to the same JSONL ledgers, and the lock keeps every line
        intact while holding the kernel free of host-locking imports.
        """
        token_usage_path = data_dir / LOGS_DIR / "token-usage.jsonl"
        decisions_path = data_dir / LOGS_DIR / "decisions.jsonl"
        return cls(
            approval_service=PolicyApprovalService(
                FileApprovalService(
                    data_dir / STATE_DIR / "approvals.json",
                    ledger_path=data_dir / LOGS_DIR / "approvals.jsonl",
                ),
                data_dir=data_dir,
            ),
            recovery_service=FileRecoveryService(data_dir / STATE_DIR / "recovery-items.json"),
            code_change_artifact_store=FileCodeChangeArtifactStore(
                data_dir / STATE_DIR / "code-change-artifacts.json"
            ),
            token_usage_handler=TokenUsageTelemetryHandler(
                token_usage_path,
                lock_factory=lambda: file_lock(
                    sidecar_lock_path(token_usage_path), timeout=_TELEMETRY_LOCK_TIMEOUT_S
                ),
            ),
            decision_telemetry_handler=DecisionTelemetryHandler(
                decisions_path,
                lock_factory=lambda: file_lock(
                    sidecar_lock_path(decisions_path), timeout=_TELEMETRY_LOCK_TIMEOUT_S
                ),
            ),
        )


__all__ = ["HostServices"]
