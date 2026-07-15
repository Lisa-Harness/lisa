"""Cloudflare processing loop — alert detection from assessment findings.

Folds the former threshold-based anomaly detector into a ``ProcessingLoop``
that consumes :class:`CloudflareAssessment` findings emitted by
:func:`lisa_pack_cloudflare.assessment.assess`. Only ``critical`` severity
findings produce alerts; ``warn`` findings are noted in triage but do not
trigger the alert pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kernel.intake.contracts import ProcessContext, ProcessedState
from kernel.models import AnomalyAlert
from lisa_pack_cloudflare.assessment import assess
from lisa_pack_cloudflare.models import CloudflareMetrics


class CloudflareProcessingLoop:
    """Alert detection for cached Cloudflare telemetry."""

    loop_id = "cloudflare"

    async def process(self, ctx: ProcessContext) -> ProcessedState:
        """Examine the Cloudflare assessment and emit alerts for critical findings."""
        if ctx.state is None:
            return ProcessedState(loop_id=self.loop_id)

        snapshot = ctx.state.signal_snapshots.get("cloudflare")
        if snapshot is None:
            return ProcessedState(loop_id=self.loop_id)

        cloudflare = CloudflareMetrics.model_validate(snapshot)
        assessment = assess(cloudflare)

        alerts = []
        for finding in assessment.findings:
            if finding.severity != "critical":
                continue
            alerts.append(
                AnomalyAlert(
                    severity="critical",
                    source="cloudflare",
                    message=f"{finding.title}: {finding.body}",
                    detected_at=datetime.now(UTC).isoformat(),
                )
            )

        if not alerts:
            return ProcessedState(loop_id=self.loop_id)

        return ProcessedState(
            loop_id=self.loop_id,
            active_alerts=alerts,
        )


__all__ = ["CloudflareProcessingLoop"]