"""Cloudflare assessment pre-turn operation.

Reads the cached Cloudflare signal snapshot, runs the pure :func:`assess`
projector, and renders a deterministic infrastructure health report as the
pre-turn briefing for ``cloudflare`` chat mode.
"""

from __future__ import annotations

from kernel.models import OperationExecutionContext, OperationResult, UnifiedState
from lisa_pack_cloudflare.assessment import assess
from lisa_pack_cloudflare.models import CloudflareMetrics


def _render_report(assessment) -> str:
    """Render a CloudflareAssessment as a concise pre-turn briefing."""
    lines: list[str] = [f"## Cloudflare Status — {assessment.status.upper()}"]
    lines.append("")

    if assessment.status == "unconfigured":
        lines.append(
            "Cloudflare credentials are not configured. "
            "Set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ZONE_ID` to enable monitoring."
        )
        return "\n".join(lines)

    # Surface degraded/critical findings first
    severity_order = {"critical": 0, "warn": 1, "info": 2}
    sorted_findings = sorted(
        assessment.findings,
        key=lambda f: (severity_order.get(f.severity, 3), f.id),
    )

    for finding in sorted_findings:
        marker = ""
        if finding.severity == "critical":
            marker = "🚨 "
        elif finding.severity == "warn":
            marker = "⚠️ "
        lines.append(f"- {marker}**{finding.title}**: {finding.body}")

    if assessment.hypotheses:
        lines.append("")
        lines.append("### Hypotheses")
        for h in assessment.hypotheses[:8]:
            lines.append(f"- {h}")

    lines.append("")
    lines.append(
        "_Ask me to investigate any finding for a live deep-dive via the Cloudflare API._"
    )
    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the Cloudflare assessment and return the report as pre-turn briefing."""

    state: UnifiedState | None = None
    if ctx.state_provider is not None:
        state = ctx.state_provider()

    if state is None:
        return OperationResult(
            output_text=(
                "⚠️ No cached runtime state available. "
                "The Cloudflare signal snapshot has not been collected yet."
            )
        )

    snapshot = state.signal_snapshots.get("cloudflare")
    if snapshot is None:
        return OperationResult(
            output_text=(
                "⚠️ No Cloudflare snapshot in cached state. "
                "Ensure the Cloudflare signal connector is configured and has run at least once."
            )
        )

    try:
        metrics = CloudflareMetrics.model_validate(snapshot)
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"⚠️ Failed to parse Cloudflare snapshot: {type(exc).__name__}: {exc}"
        )

    assessment = assess(metrics)
    return OperationResult(output_text=_render_report(assessment))