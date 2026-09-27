"""HostServices — the host's durable service handles.

Owns the four host-persisted services every turn and command can reach
without recomputing: the policy-deciding approval service, the recovery
store, the code-change artifact store, and the token-usage telemetry
appender. Also carries the capability-selected memory provider config
(``memory_config_pack_slug`` names which pack won the memory capability,
and ``memory_config`` is the resolved MCP server config).

Extracted from ``runtime.runtime`` so service construction and identity live
in one visible place instead of being assembled inline in the runtime
``__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    #: The resolved memory MCP server config selected by capability
    #: provider policy, plus the pack slug that won the capability.
    memory_config: Any = None
    memory_config_pack_slug: str | None = None

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
        )


__all__ = ["HostServices"]
