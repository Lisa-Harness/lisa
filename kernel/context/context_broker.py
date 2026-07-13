"""Query-aware context brokerage over the pre-warmed UnifiedState cache.

This module builds a structured evidence catalog from the hot state snapshot
and can select a small ranked ContextPacket for one turn without performing
live third-party polling.
"""

from __future__ import annotations

import math
import os
import re
import socket
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def resolve_host_name() -> str:
    """Resolve the OS hostname for ambient context injection.

    Honors the platform-conventional env vars first so a user can override
    what the model sees (``HOSTNAME`` on POSIX, ``COMPUTERNAME`` on Windows),
    then falls back to :func:`socket.gethostname`. Returns an empty string
    only if every source is unavailable.
    """

    for env_var in ("HOSTNAME", "COMPUTERNAME"):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    try:
        return socket.gethostname() or ""
    except OSError:
        return ""

from kernel.config import settings
from kernel.context.context_frames import render_ranked_evidence_namespace_frame
from kernel.context.extension_evidence import NoOpExtensionEvidenceBuilder
from kernel.extension_runtime import ExtensionRuntimeContext
from kernel.intake.connector_discovery import discover_installed_signal_collectors
from kernel.models import (
    ContextConflictRisk,
    ContextEvidence,
    ContextEvidenceNamespace,
    ContextEvidenceSourceType,
    ContextPacket,
    IntelCollection,
    UnifiedState,
)
from kernel.seams.protocols import ExtensionEvidenceBuilder

# Module-level extension-evidence builder. Hosts override via
# ``set_extension_evidence_builder`` when they need the production builder.
# Callers that need explicit injection (e.g. tests, kernel seams) pass
# ``extension_evidence_builder=`` to ``build_context_evidence_catalog`` or
# ``select_ranked_context``; when omitted, this module global is used.
_extension_evidence_builder: ExtensionEvidenceBuilder = NoOpExtensionEvidenceBuilder()


def set_extension_evidence_builder(builder: ExtensionEvidenceBuilder) -> None:
    """Override the module-level extension-evidence builder."""

    global _extension_evidence_builder
    _extension_evidence_builder = builder


def reset_extension_evidence_builder() -> None:
    """Reset the extension-evidence builder to the NoOp default (for tests)."""

    global _extension_evidence_builder
    _extension_evidence_builder = NoOpExtensionEvidenceBuilder()


_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "at",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
}
_SEVERITY_SALIENCE = {"critical": 1.0, "warn": 0.85, "info": 0.65}
_NAMESPACE_RENDER_ORDER: tuple[ContextEvidenceNamespace, ...] = (
    "operational_state",
    "product_repo_knowledge",
    "long_term_strategic_memory",
    "market_signal",
    "conversational_style_context",
    "general_context",
)
_MARKET_SIGNAL_SOURCES = {"market", "clarity"}
_LONG_TERM_MEMORY_SOURCES = {"strategy", "thinking"}


def build_ambient_facts(
    *,
    timezone: str = "UTC",
    host_os: str = "",
    host_name: str = "",
    now: datetime | None = None,
) -> str:
    """Build compact ambient runtime facts for prompt context injection."""

    now_utc = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    configured_timezone = (timezone or "UTC").strip() or "UTC"
    timezone_used = configured_timezone
    try:
        target_zone = ZoneInfo(configured_timezone)
    except ZoneInfoNotFoundError:
        # ZoneInfo may fail when the IANA tz database is unavailable
        # (e.g. Windows without the ``tzdata`` package).  Fall back to
        # the system local timezone rather than silently reporting UTC,
        # so the model sees the correct local time when the configured
        # zone can't be loaded.  This mirrors the status-line badge
        # fallback in ``hosts/cli/repl.py``.
        target_zone = datetime.now().astimezone().tzinfo or UTC
        timezone_used = str(target_zone) if target_zone is not UTC else "UTC"
    local_now = now_utc.astimezone(target_zone)

    lines = [
        f"- local_date: {local_now.strftime('%Y-%m-%d')}",
        f"- local_time: {local_now.strftime('%H:%M %Z')}",
        f"- local_weekday: {local_now.strftime('%A')}",
        f"- timezone: {timezone_used}",
        f"- utc_now: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if host_os.strip():
        lines.append(f"- host_os: {host_os.strip()}")
    if host_name.strip():
        lines.append(f"- host: {host_name.strip()}")
    return "\n".join(lines)


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]
    return token


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text.lower()):
        token = _normalize_token(match.group(0).lower())
        if token and token not in _STOPWORDS:
            tokens.add(token)
    return tokens


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


def _source_ttl(source: str) -> timedelta:
    if source in discover_installed_signal_collectors() or source in {"alerts", "dev_state"}:
        return timedelta(seconds=max(settings.heartbeat_interval_seconds * 2, 60))
    if source == "domain_knowledge":
        return timedelta(hours=max(settings.repo_context_stale_after_hours, 1))
    if source == "thinking":
        return timedelta(minutes=max(settings.reflection_cadence_minutes * 2, 1))
    if source == "strategy":
        return timedelta(hours=max(settings.dream_hypothesis_ttl_hours, 1))
    return timedelta(minutes=30)


def _build_freshness_expires_at(*, collected_at: str, source: str) -> str:
    collected_at_dt = _parse_iso(collected_at) or datetime.now(UTC)
    return (collected_at_dt + _source_ttl(source)).isoformat()


def _compute_recency_score(collected_at: str, now: datetime) -> float:
    collected_at_dt = _parse_iso(collected_at)
    if collected_at_dt is None:
        return 0.5
    age_hours = max((now - collected_at_dt).total_seconds() / 3600.0, 0.0)
    return round(math.exp(-0.693 * age_hours / 24.0), 4)


def _compute_freshness_score(
    *,
    collected_at: str,
    freshness_expires_at: str,
    now: datetime,
) -> float:
    collected_at_dt = _parse_iso(collected_at)
    expires_at_dt = _parse_iso(freshness_expires_at)
    if collected_at_dt is None or expires_at_dt is None or expires_at_dt <= collected_at_dt:
        return 0.5
    if now <= expires_at_dt:
        return 1.0
    ttl_seconds = max((expires_at_dt - collected_at_dt).total_seconds(), 1.0)
    stale_seconds = (now - expires_at_dt).total_seconds()
    decay = math.exp(-stale_seconds / ttl_seconds)
    return round(max(0.0, min(1.0, decay)), 4)


def _base_relevance_for_evidence(evidence: ContextEvidence) -> tuple[float, str]:
    """Return a baseline relevance score for critical system/extension evidence.

    This ensures that certain evidence types (like alerts or reflection state)
    always have a path to being included even when they don't share lexical
    tokens with the user's specific query.
    """
    if evidence.evidence_type == "alert":
        return 0.85, "baseline: alert"

    if evidence.source == "self_model" or evidence.evidence_type == "self_model_runtime":
        return 0.8, "baseline: self model"

    if evidence.source_type in {"reflection", "strategy_memory"}:
        return 0.75, f"baseline: {evidence.source_type}"

    if evidence.source_type == "domain_knowledge":
        return 0.65, "baseline: domain knowledge"

    if evidence.source_type == "signal_collector":
        return 0.45, "baseline: signal connector"

    return 0.0, ""


def _compute_relevance_score(
    query_tokens: set[str],
    evidence: ContextEvidence,
) -> tuple[float, list[str]]:
    base_score, base_reason = _base_relevance_for_evidence(evidence)
    if not query_tokens:
        return base_score, [base_reason] if base_reason else []

    evidence_text = " ".join([evidence.title, evidence.body, " ".join(evidence.tags)])
    evidence_tokens = _tokenize(evidence_text)
    if not evidence_tokens:
        return base_score, [base_reason] if base_reason else []

    overlap = query_tokens & evidence_tokens
    if not overlap:
        return base_score, [base_reason] if base_reason else []

    overlap_score = len(overlap) / max(len(query_tokens), 1)
    reasons = [f"lexical overlap: {', '.join(sorted(overlap)[:4])}"]

    final_score = max(base_score, overlap_score)
    if base_score > overlap_score and base_reason:
        reasons.append(base_reason)

    return round(min(final_score, 1.0), 4), reasons


def _compute_combined_score(evidence: ContextEvidence) -> float:
    relevance_weight = 0.45
    salience_weight = 0.25
    freshness_weight = 0.15
    recency_weight = 0.15 if settings.context_broker_prefer_recent else 0.05
    score = (
        (evidence.relevance_score * relevance_weight)
        + (evidence.salience_score * salience_weight)
        + (evidence.freshness_score * freshness_weight)
        + (evidence.recency_score * recency_weight)
    )
    return round(min(max(score, 0.0), 1.0), 4)


def _classify_evidence_namespace(evidence: ContextEvidence) -> ContextEvidenceNamespace:
    source = evidence.source.lower()
    evidence_type = evidence.evidence_type.lower()
    tags = {tag.lower() for tag in evidence.tags}
    evidence_id = evidence.evidence_id.lower()
    signal_sources = set(discover_installed_signal_collectors())

    if source == "self_model" or evidence_type == "self_model_runtime":
        return "operational_state"

    if evidence_type == "alert" or source in signal_sources or source.endswith("_ops_trends"):
        return "operational_state"
    if source == "dev_state" or evidence_type in {"recent_work", "domain_knowledge"}:
        return "product_repo_knowledge"
    if source in _LONG_TERM_MEMORY_SOURCES or evidence_type in {
        "active_thoughts",
        "strategy_memory",
    }:
        return "long_term_strategic_memory"
    if (
        source in _MARKET_SIGNAL_SOURCES
        or "market" in source
        or "market" in evidence_id
        or "market" in tags
        or source == "clarity"
    ):
        return "market_signal"
    if "style" in tags or "preference" in tags or evidence_type == "conversation_style":
        return "conversational_style_context"
    return "general_context"


def _classify_evidence_source_type(evidence: ContextEvidence) -> ContextEvidenceSourceType:
    source = evidence.source.lower()
    evidence_type = evidence.evidence_type.lower()
    signal_sources = set(discover_installed_signal_collectors())

    if source == "self_model" or evidence_type == "self_model_runtime":
        return "reflection"

    if evidence_type == "alert":
        return "operational_alert"
    if source in signal_sources or evidence.collector_id:
        return "signal_collector"
    if source == "dev_state":
        return "developer_state"
    if source == "domain_knowledge" or evidence_type == "domain_knowledge":
        return "domain_knowledge"
    if _classify_evidence_namespace(evidence) == "market_signal":
        return "market_intel"
    if source == "thinking" or evidence_type == "active_thoughts":
        return "reflection"
    if source == "strategy" or evidence_type == "strategy_memory":
        return "strategy_memory"
    if evidence.metadata.get("extension_id") or evidence.metadata.get("extension_name"):
        return "extension"
    return "general"


def _apply_evidence_identity(evidence: ContextEvidence) -> ContextEvidence:
    if evidence.namespace == "general_context":
        evidence.namespace = _classify_evidence_namespace(evidence)
    if evidence.source_type == "general":
        evidence.source_type = _classify_evidence_source_type(evidence)
    return evidence


def _compute_conflict_risk(evidence: ContextEvidence) -> ContextConflictRisk:
    if evidence.confidence < 0.45 or evidence.freshness_score < 0.35:
        return "high"
    if evidence.namespace in {"market_signal", "long_term_strategic_memory"}:
        return "medium"
    if evidence.confidence < 0.7 or evidence.freshness_score < 0.8:
        return "medium"
    return "low"


def _new_evidence(
    *,
    evidence_id: str,
    source: str,
    evidence_type: str,
    title: str,
    body: str,
    collected_at: str,
    collector_id: str = "",
    confidence: float = 0.8,
    salience_score: float = 0.5,
    tags: Iterable[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> ContextEvidence:
    evidence = ContextEvidence(
        evidence_id=evidence_id,
        source=source,
        collector_id=collector_id,
        evidence_type=evidence_type,
        title=title,
        body=body,
        collected_at=collected_at,
        freshness_expires_at=_build_freshness_expires_at(collected_at=collected_at, source=source),
        confidence=confidence,
        salience_score=salience_score,
        tags=[tag for tag in (tags or []) if tag],
        metadata=dict(metadata or {}),
    )
    return _apply_evidence_identity(evidence)


def _build_extension_evidence(
    state: UnifiedState,
    *,
    builder: ExtensionEvidenceBuilder | None = None,
) -> list[ContextEvidence]:
    runtime_context = ExtensionRuntimeContext(state_provider=None)
    active_builder = builder or _extension_evidence_builder
    return active_builder.build_extension_context_evidence(runtime_context, state)


def _build_alert_evidence(state: UnifiedState) -> list[ContextEvidence]:
    evidence: list[ContextEvidence] = []
    for index, alert in enumerate(state.active_alerts, start=1):
        evidence.append(
            _new_evidence(
                evidence_id=f"alert:{index}:{alert.source}",
                source=alert.source or "alerts",
                collector_id=alert.source,
                evidence_type="alert",
                title=f"{alert.source.title()} {alert.severity} alert",
                body=alert.message,
                collected_at=alert.detected_at or _default_collected_at(state),
                confidence=0.95,
                salience_score=_SEVERITY_SALIENCE.get(alert.severity, 0.65),
                tags=["alert", alert.source, alert.severity],
            )
        )
    return evidence


def _build_dev_state_evidence(state: UnifiedState) -> list[ContextEvidence]:
    if state.dev_state is None or not state.dev_state.recent_accomplishments:
        return []
    recent = state.dev_state.recent_accomplishments[-3:]
    bullets = [f"[{entry.ts[:10]}] {entry.summary}" for entry in recent]
    return [
        _new_evidence(
            evidence_id="dev_state:recent_accomplishments",
            source="dev_state",
            evidence_type="recent_work",
            title="Recent developer work",
            body="\n".join(bullets),
            collected_at=state.dev_state.last_updated or _default_collected_at(state),
            salience_score=0.4,
            tags=["developer", "outbox", "recent-work"],
        )
    ]


def _build_collection_evidence(
    *,
    source: str,
    collection: IntelCollection | None,
    state: UnifiedState,
    title: str,
    evidence_type: str,
    salience_score: float,
) -> list[ContextEvidence]:
    if collection is None:
        return []
    bullets = collection.summary_bullets[:4]
    if not bullets and not collection.collection_metadata:
        return []
    body = (
        "\n".join(f"- {bullet}" for bullet in bullets)
        if bullets
        else str(collection.collection_metadata)
    )
    collected_at = getattr(collection, "generated_at", "") or _default_collected_at(state)
    return [
        _new_evidence(
            evidence_id=f"collection:{source}",
            source=source,
            evidence_type=evidence_type,
            title=title,
            body=body,
            collected_at=collected_at,
            salience_score=salience_score,
            tags=[source, evidence_type],
        )
    ]


def _collection_title(source: str) -> str:
    """Render one human-readable title for an intel collection source ID."""

    return f"{source.replace('_', ' ').strip().title()} insights"


def _build_all_collection_evidence(
    state: UnifiedState,
    *,
    skipped_sources: set[str],
) -> list[ContextEvidence]:
    """Render ambient evidence for cached intel collections not owned by extensions."""

    evidence: list[ContextEvidence] = []
    for source, collection in state.intel_collections.items():
        if source in skipped_sources:
            continue
        evidence.extend(
            _build_collection_evidence(
                source=source,
                collection=collection,
                state=state,
                title=_collection_title(source),
                evidence_type="intel_collection",
                salience_score=0.5,
            )
        )
    return evidence


def _build_analysis_evidence(state: UnifiedState) -> list[ContextEvidence]:
    evidence: list[ContextEvidence] = []

    if state.thinking_vibe:
        evidence.append(
            _new_evidence(
                evidence_id="thinking:active_vibe",
                source="thinking",
                evidence_type="active_thoughts",
                title="Lisa active thoughts",
                body=state.thinking_vibe,
                collected_at=state.compiled_at,
                salience_score=0.55,
                tags=["thinking", "reflection"],
            )
        )

    if state.strategy_vibe:
        evidence.append(
            _new_evidence(
                evidence_id="strategy:promoted_memory",
                source="strategy",
                evidence_type="strategy_memory",
                title="Promoted strategy memory",
                body=state.strategy_vibe,
                collected_at=state.compiled_at,
                salience_score=0.75,
                tags=["strategy", "memory"],
            )
        )

    return evidence


def _self_model_salience(body: str) -> float:
    body_lower = body.lower()
    if "health=critical" in body_lower or "critical" in body_lower:
        return 0.95
    if "health=degraded" in body_lower or "degraded" in body_lower:
        return 0.8
    return 0.65


def _build_self_model_evidence(state: UnifiedState) -> list[ContextEvidence]:
    if not state.self_model_vibe:
        return []
    return [
        _new_evidence(
            evidence_id="self_model:runtime_vibe",
            source="self_model",
            evidence_type="self_model_runtime",
            title="Live self-model runtime state",
            body=state.self_model_vibe,
            collected_at=state.compiled_at,
            salience_score=_self_model_salience(state.self_model_vibe),
            tags=["self_model", "runtime", "operational"],
        )
    ]


def _build_domain_knowledge_evidence(state: UnifiedState) -> list[ContextEvidence]:
    evidence: list[ContextEvidence] = []

    for artifact_id, artifact in sorted(state.domain_knowledge.items()):
        visible_bullets = [bullet.strip() for bullet in artifact.summary_bullets if bullet.strip()]
        if not visible_bullets:
            continue

        salience_score = 0.65 if artifact.status in {"warning", "error"} else 0.45
        evidence.append(
            _new_evidence(
                evidence_id=f"domain_knowledge:{artifact_id}",
                source="domain_knowledge",
                evidence_type="domain_knowledge",
                title=artifact.display_name or artifact_id.replace("_", " "),
                body=" | ".join(visible_bullets[:4]),
                collected_at=artifact.generated_at or _default_collected_at(state),
                salience_score=salience_score,
                tags=["domain_knowledge", artifact_id, artifact.status],
                metadata={
                    "artifact_id": artifact_id,
                    **artifact.metadata,
                },
            )
        )

    return evidence


def build_context_evidence_catalog(
    state: UnifiedState,
    *,
    extension_evidence_builder: ExtensionEvidenceBuilder | None = None,
) -> list[ContextEvidence]:
    """Build the full structured evidence catalog from the current hot state.

    Args:
        state: The current hot-state snapshot.
        extension_evidence_builder: Optional explicit builder to use for
            extension evidence. When ``None`` (default), the module-level
            builder (set via ``set_extension_evidence_builder``) is used.
    """
    extension_evidence = _build_extension_evidence(
        state,
        builder=extension_evidence_builder,
    )
    extension_owned_collection_sources = {
        item.source for item in extension_evidence if item.evidence_type == "intel_collection"
    }
    catalog: list[ContextEvidence] = []
    catalog.extend(extension_evidence)
    catalog.extend(_build_alert_evidence(state))
    catalog.extend(_build_dev_state_evidence(state))
    catalog.extend(_build_self_model_evidence(state))
    catalog.extend(
        _build_all_collection_evidence(
            state,
            skipped_sources=extension_owned_collection_sources,
        )
    )
    catalog.extend(_build_domain_knowledge_evidence(state))
    catalog.extend(_build_analysis_evidence(state))
    for item in catalog:
        _apply_evidence_identity(item)
    return catalog


def _should_include_for_connectors(evidence: ContextEvidence, connector_set: set[str]) -> bool:
    if not connector_set:
        return True
    if evidence.evidence_type == "alert":
        return True
    return bool(evidence.collector_id and evidence.collector_id in connector_set)


def select_ranked_context(
    *,
    state: UnifiedState,
    query: str,
    mode_id: str = "default",
    stage_id: str = "",
    connectors: list[str] | None = None,
    extension_evidence_builder: ExtensionEvidenceBuilder | None = None,
) -> ContextPacket:
    """Select one ranked ContextPacket for the current turn.

    Args:
        extension_evidence_builder: Optional explicit builder for extension
            evidence. When ``None`` (default), the module-level builder is
            used. Pass this explicitly in tests or kernel seams to avoid
            depending on module-global mutation.
    """
    connector_set = set(connectors or [])
    now = _parse_iso(state.compiled_at) or datetime.now(UTC)
    query_tokens = _tokenize(query)
    candidates = state.context_evidence_catalog or build_context_evidence_catalog(
        state,
        extension_evidence_builder=extension_evidence_builder,
    )
    scored: list[ContextEvidence] = []

    for candidate in candidates:
        if not _should_include_for_connectors(candidate, connector_set):
            continue
        evidence = candidate.model_copy(deep=True)
        relevance_score, match_reasons = _compute_relevance_score(query_tokens, evidence)
        evidence.relevance_score = relevance_score
        evidence.recency_score = _compute_recency_score(
            evidence.collected_at or state.compiled_at,
            now,
        )
        evidence.freshness_score = _compute_freshness_score(
            collected_at=evidence.collected_at or state.compiled_at,
            freshness_expires_at=evidence.freshness_expires_at,
            now=now,
        )
        evidence.match_reasons = match_reasons
        if evidence.freshness_score < 1.0:
            evidence.match_reasons.append("stale-but-usable fallback")
        _apply_evidence_identity(evidence)
        evidence.conflict_risk = _compute_conflict_risk(evidence)
        evidence.combined_score = _compute_combined_score(evidence)
        if evidence.combined_score < settings.context_broker_min_evidence_score:
            continue
        scored.append(evidence)

    scored.sort(
        key=lambda item: (
            item.combined_score,
            item.relevance_score,
            item.salience_score,
            item.freshness_score,
            item.recency_score,
        ),
        reverse=True,
    )

    selected: list[ContextEvidence] = []
    rendered_chars = 0
    truncation_reason = ""
    for evidence in scored:
        evidence_chars = len(evidence.title) + len(evidence.body)
        if len(selected) >= settings.context_broker_max_selected_items:
            truncation_reason = "max selected items reached"
            break
        if rendered_chars + evidence_chars > settings.context_broker_max_markdown_chars:
            truncation_reason = "max markdown chars reached"
            break
        selected.append(evidence)
        rendered_chars += evidence_chars

    return ContextPacket(
        compiled_at=state.compiled_at,
        query=query,
        mode_id=mode_id,
        stage_id=stage_id,
        total_available_evidence=len(scored),
        selected_evidence=selected,
        dropped_evidence_count=max(len(scored) - len(selected), 0),
        truncation_reason=truncation_reason,
        ranking_metadata={
            "connectors_filter": sorted(connector_set),
            "query_tokens": sorted(query_tokens),
            "selected_namespaces": list(dict.fromkeys(item.namespace for item in selected)),
            "deterministic_phase": True,
        },
    )


def _ordered_selected_namespaces(
    selected_evidence: list[ContextEvidence],
) -> list[ContextEvidenceNamespace]:
    selected = set(item.namespace for item in selected_evidence)
    ordered = [namespace for namespace in _NAMESPACE_RENDER_ORDER if namespace in selected]
    ordered.extend(
        namespace
        for namespace in dict.fromkeys(item.namespace for item in selected_evidence)
        if namespace not in ordered
    )
    return ordered


def _render_evidence_insert(evidence: ContextEvidence) -> str:
    reasons = (
        ", ".join(evidence.match_reasons[:3]) if evidence.match_reasons else "ranked by salience"
    )
    return (
        f"- [{evidence.source}] {evidence.title}: {evidence.body}\n"
        f"  - metadata: namespace={evidence.namespace}; source_type={evidence.source_type}; "
        f"relevance={evidence.relevance_score:.2f}; freshness={evidence.freshness_score:.2f}; "
        f"confidence={evidence.confidence:.2f}; conflict_risk={evidence.conflict_risk}; "
        f"score={evidence.combined_score:.2f}; why={reasons}"
    )


def render_context_packet_markdown(packet: ContextPacket) -> str:
    """Render one ContextPacket into prompt-safe markdown."""
    if not packet.selected_evidence:
        return "- No ranked context evidence was selected for this turn."

    lines = [
        f"- Context packet built at: {packet.compiled_at}",
        "- Consumption rule: keep namespace boundaries intact unless the user asks for synthesis.",
    ]
    for namespace in _ordered_selected_namespaces(packet.selected_evidence):
        namespace_evidence = [
            item for item in packet.selected_evidence if item.namespace == namespace
        ]
        body = "\n".join(_render_evidence_insert(evidence) for evidence in namespace_evidence)
        lines.append(
            render_ranked_evidence_namespace_frame(
                namespace,
                body,
            )
        )
    if packet.truncation_reason:
        lines.append(f"- Context packet truncated: {packet.truncation_reason}")
    return "\n".join(lines)
