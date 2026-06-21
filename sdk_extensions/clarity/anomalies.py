"""Extension-owned anomaly detectors for cached Clarity live-insights."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionAnomalyDetector, ExtensionRuntimeContext
from kernel.models import AnomalyAlert, UnifiedState
from sdk_extensions.clarity.utils import build_clarity_summary


async def detect_clarity_anomalies(state: UnifiedState) -> list[AnomalyAlert]:
    """Return low-cost anomaly alerts derived from cached Clarity summaries."""

    alerts: list[AnomalyAlert] = []
    clarity_summary = build_clarity_summary(state.intel_collections.get("clarity"))
    if clarity_summary.get("status") != "available":
        return alerts

    signals = clarity_summary.get("signals") or {}
    rage_clicks = int(signals.get("rage_clicks", 0))
    js_errors = int(signals.get("js_errors", 0))
    script_errors = int(signals.get("script_errors", 0))

    if rage_clicks >= 5:
        alerts.append(
            AnomalyAlert(
                severity="warn",
                source="clarity",
                message=f"{rage_clicks} rage clicks detected in the cached Clarity window.",
                detected_at="",
            )
        )

    if js_errors >= 3:
        severity = "critical" if js_errors >= 10 else "warn"
        alerts.append(
            AnomalyAlert(
                severity=severity,
                source="clarity",
                message=f"{js_errors} JavaScript errors detected in the cached Clarity window.",
                detected_at="",
            )
        )

    if script_errors >= 3:
        severity = "critical" if script_errors >= 10 else "warn"
        alerts.append(
            AnomalyAlert(
                severity=severity,
                source="clarity",
                message=f"{script_errors} script errors detected in the cached Clarity window.",
                detected_at="",
            )
        )

    return alerts


def build_anomaly_detectors(
    runtime_context: ExtensionRuntimeContext,
) -> list[ExtensionAnomalyDetector]:
    """Expose the Clarity-owned anomaly detectors to heartbeat."""

    del runtime_context
    return [detect_clarity_anomalies]


__all__ = ["build_anomaly_detectors", "detect_clarity_anomalies"]
