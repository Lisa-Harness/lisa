"""Clarity snapshot summary helpers.

Pure analysis utilities — no I/O.  Can be imported by heartbeat anomaly checks,
skill renderers, and stats rolls without pulling in the full connector.
"""
from __future__ import annotations

import re
from typing import Any

from kernel.models import IntelCollection
from sdk_extensions.clarity.models import ClarityLiveInsightsResponse, ClarityMetricRow


def _coerce_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _metric_name_slug(metric_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", metric_name.lower())


def _is_rage_metric(metric_name: str) -> bool:
    return _metric_name_slug(metric_name).startswith("rage")


def _is_js_error_metric(metric_name: str) -> bool:
    metric_slug = _metric_name_slug(metric_name)
    return metric_slug.startswith("javascript") or metric_slug.startswith("jserror")


def _is_dead_click_metric(metric_name: str) -> bool:
    return _metric_name_slug(metric_name).startswith("dead")


def _is_quick_back_metric(metric_name: str) -> bool:
    return _metric_name_slug(metric_name).startswith("quickback")


def _is_script_error_metric(metric_name: str) -> bool:
    return _metric_name_slug(metric_name).startswith("scripterror")


def _metric_value_keys(metric_name: str) -> tuple[str, ...]:
    normalized_name = metric_name.lower()
    if "popularpage" in normalized_name:
        return ("visitsCount",)
    if normalized_name == "traffic":
        return ("totalSessionCount", "distinctUserCount")
    if "engagementtime" in normalized_name:
        return ("activeTime", "totalTime")
    if "scrolldepth" in normalized_name:
        return ("averageScrollDepth",)
    return (
        "subTotal",
        "visitsCount",
        "sessionsCount",
        "totalSessionCount",
        "distinctUserCount",
        "pagesViews",
        "count",
    )


def _metric_value_from_entry(metric_name: str, entry: dict[str, Any]) -> int:
    for key in _metric_value_keys(metric_name):
        value = _coerce_number(entry.get(key))
        if value is not None:
            return value
    return 0


def _metric_total(metric: ClarityMetricRow) -> int:
    return sum(
        _metric_value_from_entry(metric.metric_name, entry)
        for entry in metric.information
        if isinstance(entry, dict)
    )


def _find_metric(
    snapshot: ClarityLiveInsightsResponse,
    predicate: Any,
) -> ClarityMetricRow | None:
    for metric in snapshot.metrics:
        if predicate(metric.metric_name):
            return metric
    return None


def _metric_preview(metric: ClarityMetricRow) -> str:
    if not metric.information:
        return ""
    first_entry = metric.information[0]
    if isinstance(first_entry, dict):
        preview_keys = ("url", "page", "path", "name", "subTotal", "visitsCount", "sessionsCount")
        values = [
            str(first_entry.get(key))
            for key in preview_keys
            if str(first_entry.get(key, "")).strip()
        ]
        if not values:
            values = [str(value) for value in list(first_entry.values())[:3] if str(value).strip()]
        return ", ".join(values)
    return str(first_entry)


def _matches_query(metric: ClarityMetricRow, query: str) -> bool:
    normalized_query = query.strip().lower()
    if not normalized_query:
        return True
    haystack = " ".join([metric.metric_name.lower(), _metric_preview(metric).lower()])
    terms = [term for term in re.split(r"[^a-z0-9]+", normalized_query) if term]
    if not terms:
        return True
    return all(term in haystack for term in terms)


def _signal_count(snapshot: ClarityLiveInsightsResponse, predicate: Any) -> int:
    total = 0
    for metric in snapshot.metrics:
        if predicate(metric.metric_name):
            total += _metric_total(metric)
    return total


def _extract_page_hint(metric: ClarityMetricRow) -> str:
    if not metric.information:
        return ""
    for entry in metric.information:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            key_text = str(key).lower()
            if key_text not in {"page", "url", "path", "pageurl", "pagetitle"}:
                continue
            candidate = str(value).strip()
            if candidate:
                return candidate
    return ""


def _rank_hotspot_pages(snapshot: ClarityLiveInsightsResponse) -> list[dict[str, Any]]:
    hotspots: dict[str, int] = {}
    for metric in snapshot.metrics:
        if not any(
            matcher(metric.metric_name)
            for matcher in (
                _is_rage_metric,
                _is_dead_click_metric,
                _is_quick_back_metric,
                _is_js_error_metric,
                _is_script_error_metric,
            )
        ):
            continue
        page = _extract_page_hint(metric)
        if not page:
            continue
        hotspots[page] = hotspots.get(page, 0) + _metric_total(metric)
    ranked = sorted(hotspots.items(), key=lambda item: item[1], reverse=True)
    return [{"page": page, "signal_hits": hits} for page, hits in ranked[:3]]


def _extract_traffic_overview(snapshot: ClarityLiveInsightsResponse) -> dict[str, Any]:
    traffic_metric = _find_metric(
        snapshot, lambda metric_name: _metric_name_slug(metric_name) == "traffic"
    )
    sessions_metric = _find_metric(
        snapshot, lambda metric_name: _metric_name_slug(metric_name).startswith("session")
    )
    engagement_metric = _find_metric(
        snapshot, lambda metric_name: _metric_name_slug(metric_name) == "engagementtime"
    )
    scroll_metric = _find_metric(
        snapshot, lambda metric_name: _metric_name_slug(metric_name) == "scrolldepth"
    )

    sessions = 0
    users = 0
    if traffic_metric and traffic_metric.information:
        first_entry = traffic_metric.information[0]
        if isinstance(first_entry, dict):
            sessions = _coerce_number(first_entry.get("totalSessionCount")) or 0
            users = _coerce_number(first_entry.get("distinctUserCount")) or 0
    if not sessions and sessions_metric is not None:
        sessions = _metric_total(sessions_metric)
    if not users:
        users = sessions

    active_time_seconds = 0
    if engagement_metric and engagement_metric.information:
        first_entry = engagement_metric.information[0]
        if isinstance(first_entry, dict):
            active_time_seconds = _coerce_number(first_entry.get("activeTime")) or 0

    average_scroll_depth = 0
    if scroll_metric and scroll_metric.information:
        first_entry = scroll_metric.information[0]
        if isinstance(first_entry, dict):
            average_scroll_depth = _coerce_number(first_entry.get("averageScrollDepth")) or 0

    if sessions < 20:
        sample_size = "low"
        sample_note = "Directional only — the current sample is still small."
    elif sessions < 100:
        sample_size = "medium"
        sample_note = "Useful directional signal, but not enough for hard conclusions."
    else:
        sample_size = "high"
        sample_note = "Sample size is large enough to treat the current patterns as meaningful."

    return {
        "sessions": sessions,
        "users": users,
        "active_time_seconds": active_time_seconds,
        "average_scroll_depth": average_scroll_depth,
        "sample_size": sample_size,
        "sample_note": sample_note,
    }


def _popular_pages(snapshot: ClarityLiveInsightsResponse) -> list[dict[str, Any]]:
    metric = _find_metric(
        snapshot, lambda metric_name: _metric_name_slug(metric_name) == "popularpages"
    )
    if metric is None:
        return []
    pages: list[dict[str, Any]] = []
    for entry in metric.information:
        if not isinstance(entry, dict):
            continue
        page = str(entry.get("url") or entry.get("page") or entry.get("name") or "").strip()
        visits = _coerce_number(entry.get("visitsCount")) or 0
        if page and visits:
            pages.append({"page": page, "visits": visits})
    return sorted(pages, key=lambda page: page["visits"], reverse=True)[:5]


def _build_business_notes(
    *,
    signals: dict[str, int],
    traffic: dict[str, Any],
    popular_pages: list[dict[str, Any]],
    query: str,
    metric_count: int,
) -> list[str]:
    notes: list[str] = []
    if query.strip() and metric_count == 0:
        notes.append(f"No Clarity metrics matched query: {query.strip()}")
        return notes

    sample_note = str(traffic.get("sample_note", "")).strip()
    if sample_note:
        notes.append(sample_note)

    if popular_pages:
        lead_page = popular_pages[0]
        notes.append(
            f"Traffic is concentrated on {lead_page['page']}"
            f" ({lead_page['visits']} visits in the cached window)."
        )

    if signals["dead_clicks"]:
        notes.append(
            f"Dead clicks are the clearest friction signal right now"
            f" ({signals['dead_clicks']} total)."
        )
    if signals["quick_backs"]:
        notes.append(
            f"Quick backs suggest some visitors are abandoning a page quickly"
            f" ({signals['quick_backs']} total)."
        )
    if signals["js_errors"] or signals["script_errors"]:
        notes.append(
            "Technical friction is present, so UX confusion may not be the only problem."
        )
    if not notes:
        notes.append("No standout Clarity spikes were detected in the cached snapshot.")
    return notes[:4]


def _build_next_actions(
    *,
    signals: dict[str, int],
    popular_pages: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    if popular_pages:
        actions.append(
            f"Inspect {popular_pages[0]['page']} first because it is carrying the most visits."
        )
    if signals["dead_clicks"]:
        actions.append(
            "Review buttons, cards, and clickable-looking UI elements for dead-click traps."
        )
    if signals["quick_backs"]:
        actions.append(
            "Check whether the first screen on the affected path matches visitor intent."
        )
    if signals["js_errors"] or signals["script_errors"]:
        actions.append(
            "Check client-side logs and recent frontend changes for runtime failures."
        )
    if not actions:
        actions.append("Keep watching for more sessions before making a product change.")
    return actions[:3]


def _coerce_snapshot(
    snapshot_or_collection: ClarityLiveInsightsResponse | IntelCollection | None,
) -> ClarityLiveInsightsResponse | None:
    """Normalize cached Clarity state into the rich snapshot shape.

    UnifiedState stores Clarity under ``intel_collections['clarity']``. After blob
    round-trips that value may be deserialized as the base ``IntelCollection`` with
    the live-insights rows preserved under ``collection_metadata``.
    """
    if snapshot_or_collection is None:
        return None

    if isinstance(snapshot_or_collection, ClarityLiveInsightsResponse):
        if snapshot_or_collection.metrics:
            return snapshot_or_collection
        collection_metadata = snapshot_or_collection.collection_metadata
        generated_at = snapshot_or_collection.generated_at
    else:
        collection_metadata = snapshot_or_collection.collection_metadata
        generated_at = snapshot_or_collection.generated_at

    metric_rows = collection_metadata.get("metrics", [])
    if not isinstance(metric_rows, list) or not metric_rows:
        return None

    return ClarityLiveInsightsResponse(
        generated_at=generated_at,
        metrics=[
            ClarityMetricRow(
                metric_name=str(metric.get("metric_name", "")),
                information=metric.get("information", []),
            )
            for metric in metric_rows
            if isinstance(metric, dict)
        ],
        raw=collection_metadata.get("raw", {}),
        collection_metadata=collection_metadata,
    )


def build_clarity_summary(
    snapshot_or_collection: ClarityLiveInsightsResponse | IntelCollection | None,
    *,
    query: str = "",
) -> dict[str, Any]:
    """Return one compact summary of the cached Clarity live-insights snapshot."""
    snapshot = _coerce_snapshot(snapshot_or_collection)
    if snapshot is None or not snapshot.metrics:
        return {
            "status": "unavailable",
            "query": query,
            "metric_count_total": 0,
            "metric_count_filtered": 0,
            "signals": {},
            "traffic": {},
            "popular_pages": [],
            "top_metrics": [],
            "next_actions": [],
            "notes": [
                "No cached Clarity data is available yet.",
                "Lisa currently uses aggregated Clarity live-insights, not raw session exports.",
            ],
        }

    filtered_metrics = [metric for metric in snapshot.metrics if _matches_query(metric, query)]
    signal_snapshot = snapshot.model_copy(
        update={"metrics": filtered_metrics if query.strip() else snapshot.metrics}
    )
    ranked_metrics = sorted(
        signal_snapshot.metrics,
        key=lambda metric: (_metric_total(metric), metric.metric_name.lower()),
        reverse=True,
    )
    top_metrics = [
        {
            "name": metric.metric_name,
            "entry_count": _metric_total(metric),
            "preview": _metric_preview(metric),
        }
        for metric in ranked_metrics[:5]
    ]

    signals = {
        "rage_clicks": _signal_count(signal_snapshot, _is_rage_metric),
        "js_errors": _signal_count(signal_snapshot, _is_js_error_metric),
        "dead_clicks": _signal_count(signal_snapshot, _is_dead_click_metric),
        "quick_backs": _signal_count(signal_snapshot, _is_quick_back_metric),
        "script_errors": _signal_count(signal_snapshot, _is_script_error_metric),
    }
    traffic = _extract_traffic_overview(snapshot)
    weighted_signal_points = (
        (30 * signals["rage_clicks"])
        + (35 * signals["dead_clicks"])
        + (20 * signals["quick_backs"])
        + (15 * signals["script_errors"])
    )
    friction_score = int(
        round(weighted_signal_points / max(int(traffic.get("sessions", 0)) or 1, 1))
    )
    hotspot_pages = _rank_hotspot_pages(signal_snapshot)
    popular_pages = _popular_pages(snapshot)
    notes = _build_business_notes(
        signals=signals,
        traffic=traffic,
        popular_pages=popular_pages,
        query=query,
        metric_count=len(signal_snapshot.metrics),
    )
    next_actions = _build_next_actions(signals=signals, popular_pages=popular_pages)

    return {
        "status": "available",
        "query": query,
        "metric_count_total": len(snapshot.metrics),
        "metric_count_filtered": len(signal_snapshot.metrics),
        "signals": signals,
        "traffic": traffic,
        "friction_score": friction_score,
        "hotspot_pages": hotspot_pages,
        "popular_pages": popular_pages,
        "top_metrics": top_metrics,
        "next_actions": next_actions,
        "notes": notes,
    }


__all__ = ["build_clarity_summary"]
