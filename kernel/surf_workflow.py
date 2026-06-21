"""Durable session helpers for Lisa's iterative surf workflow.

This module owns the non-browser session semantics for surf mode: creating,
loading, trimming, summarising, and deleting page-by-page surf state stored in
host object storage. It is called by host agent loops for restart-safe workflow
rehydration and by surf-mode operations for per-turn updates.

Browser navigation itself remains host-owned and is injected through the
browser automation protocol.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from kernel.config import settings
from kernel.models import (
    SurfaceContext,
    SurfCommentaryEntry,
    SurfLinkCandidate,
    SurfPageCapture,
    SurfSessionState,
)

_LINK_FILLER_WORDS = frozenset(
    {"a", "an", "the", "page", "pages", "link", "links", "tab", "section", "site", "please"}
)


def _now_utc() -> datetime:
    """Return the current UTC timestamp for surf session bookkeeping."""
    return datetime.now(UTC)


def _isoformat(value: datetime) -> str:
    """Return one timezone-aware UTC timestamp as an ISO-8601 string."""
    return value.isoformat()


def _current_page_label(page: SurfPageCapture) -> str:
    """Return a human-readable label for one visited surf page."""
    return (
        page.trace.page_title
        or page.trace.final_url
        or page.trace.requested_url
        or "Untitled page"
    )


def _truncate_text(value: str, limit: int = 180) -> str:
    """Collapse whitespace and cap long commentary or summary fragments."""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _normalize_link_lookup_text(value: str) -> tuple[str, set[str]]:
    """Normalise one free-form link target into comparable tokens."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    tokens = [token for token in cleaned.split() if token not in _LINK_FILLER_WORDS]
    normalized = " ".join(tokens)
    return normalized, set(tokens)


def effective_surface_id(
    surface_context: SurfaceContext | None,
    *,
    user_id: str = "",
    user_slug: str = "default",
) -> str:
    """Return a stable surf-session surface id even in older test harnesses.

    Older unit tests often omit ``SurfaceContext``, so this helper falls back
    to a deterministic personal-chat style id derived from the user identity.
    """
    if surface_context is not None and surface_context.surface_id:
        return surface_context.surface_id

    identity = user_id.strip() or user_slug.strip() or "default"
    return f"personal:{identity}"


def build_surf_session_id(workspace_id: str, surface_id: str) -> str:
    """Build a deterministic surf session id for one workspace/surface pair."""
    digest = hashlib.sha256(f"{workspace_id}::{surface_id}".encode()).hexdigest()[:12]
    return f"surf-{digest}"


def is_surf_session_expired(session: SurfSessionState, *, now: datetime | None = None) -> bool:
    """Return whether one durable surf session is past its TTL."""
    reference = now or _now_utc()
    try:
        expires_at = datetime.fromisoformat(session.expires_at)
    except ValueError:
        return True
    return expires_at <= reference


def build_surf_session_summary(session: SurfSessionState) -> str:
    """Return one short summary string for handoff and history rendering."""
    if not session.pages:
        return "Surf session started but no pages have been loaded yet."

    current_page = session.pages[-1]
    summary = (
        f"Visited {len(session.pages)} page(s). "
        f"Current page: {_current_page_label(current_page)}."
    )

    if len(session.pages) > 1:
        recent_labels = [_current_page_label(page) for page in session.pages[-3:]]
        summary += f" Recent trail: {' -> '.join(recent_labels)}."

    if session.commentary:
        last_commentary = _truncate_text(session.commentary[-1].commentary, limit=160)
        if last_commentary:
            summary += f" Last commentary: {last_commentary}"
    return summary


def build_surf_stage_data(session: SurfSessionState) -> dict[str, object]:
    """Return compact stage data for the active surf workflow session."""
    return {
        "surf_session_id": session.session_id,
        "surf_surface_id": session.surface_id,
        "surf_current_url": session.current_page_url,
        "surf_current_title": session.current_page_title,
        "surf_page_count": len(session.pages),
        "surf_commentary_count": len(session.commentary),
        "surf_last_action": session.last_action,
        "surf_summary": session.summary,
    }


def touch_surf_session(session: SurfSessionState, *, now: datetime | None = None) -> None:
    """Refresh one surf session's timestamps and expiry window."""
    reference = now or _now_utc()
    session.updated_at = _isoformat(reference)
    session.expires_at = _isoformat(
        reference + timedelta(minutes=settings.surf_session_ttl_minutes)
    )


def trim_surf_session(session: SurfSessionState) -> None:
    """Bound stored pages, links, and commentary for one surf session."""
    max_pages = max(1, settings.surf_session_max_pages)
    max_links = max(1, settings.surf_session_max_links)
    max_commentary = max(1, settings.surf_session_max_commentary_entries)

    if len(session.pages) > max_pages:
        session.pages = session.pages[-max_pages:]
    if len(session.commentary) > max_commentary:
        session.commentary = session.commentary[-max_commentary:]

    for page in session.pages:
        if len(page.link_candidates) > max_links:
            page.link_candidates = page.link_candidates[:max_links]

    current_page = get_current_page(session)
    if current_page is not None:
        session.current_page_url = current_page.trace.final_url or current_page.trace.requested_url
        session.current_page_title = current_page.trace.page_title
    else:
        session.current_page_url = ""
        session.current_page_title = ""
    session.summary = build_surf_session_summary(session)


def create_surf_session(
    workspace_id: str,
    surface_context: SurfaceContext,
) -> SurfSessionState:
    """Create a brand-new durable surf workflow session."""
    now = _now_utc()
    return SurfSessionState(
        session_id=build_surf_session_id(workspace_id, surface_context.surface_id),
        workspace_id=workspace_id or settings.workspace_id_fallback,
        surface_id=surface_context.surface_id,
        surface_type=surface_context.surface_type,
        memory_scope=surface_context.memory_scope,
        created_at=_isoformat(now),
        updated_at=_isoformat(now),
        expires_at=_isoformat(now + timedelta(minutes=settings.surf_session_ttl_minutes)),
    )


def load_surf_session(
    workspace_id: str,
    surface_context: SurfaceContext,
    *,
    user_id: str = "",
    user_slug: str = "default",
    blob_io: Any = None,
) -> SurfSessionState | None:
    """Load one durable surf session, deleting it first when expired.

    Args:
        blob_io: Optional blob I/O adapter with ``read_surf_workflow_session``
            and ``delete_surf_workflow_session``. Falls back to no-op when
            None so the caller (app side) is responsible for injection.
    """
    surface_id = effective_surface_id(surface_context, user_id=user_id, user_slug=user_slug)
    if blob_io is None:
        return None
    session = blob_io.read_surf_workflow_session(
        workspace_id or settings.workspace_id_fallback,
        surface_id,
    )
    if session is None:
        return None
    if is_surf_session_expired(session):
        blob_io.delete_surf_workflow_session(
            workspace_id or settings.workspace_id_fallback,
            surface_id,
        )
        return None
    return session


def save_surf_session(session: SurfSessionState, *, blob_io: Any = None) -> bool:
    """Persist one surf session after bounding its retained history.

    Args:
        blob_io: Optional blob I/O adapter with ``write_surf_workflow_session``.
    """
    trim_surf_session(session)
    if blob_io is None:
        return False
    return blob_io.write_surf_workflow_session(session)


def delete_surf_session(
    workspace_id: str,
    surface_context: SurfaceContext,
    *,
    user_id: str = "",
    user_slug: str = "default",
    blob_io: Any = None,
) -> bool:
    """Delete one durable surf workflow session for the active surface.

    Args:
        blob_io: Optional blob I/O adapter with ``delete_surf_workflow_session``.
    """
    surface_id = effective_surface_id(surface_context, user_id=user_id, user_slug=user_slug)
    if blob_io is None:
        return False
    return blob_io.delete_surf_workflow_session(
        workspace_id or settings.workspace_id_fallback,
        surface_id,
    )


def get_current_page(session: SurfSessionState) -> SurfPageCapture | None:
    """Return the current page snapshot for one surf session when available."""
    if not session.pages:
        return None
    return session.pages[-1]


def append_page_capture(
    session: SurfSessionState,
    page_capture: SurfPageCapture,
    *,
    user_query: str,
    action: str,
) -> None:
    """Append one newly visited page to a surf session and refresh its summary."""
    session.pages.append(page_capture)
    session.last_user_query = user_query
    session.last_action = action
    session.current_page_url = page_capture.trace.final_url or page_capture.trace.requested_url
    session.current_page_title = page_capture.trace.page_title
    touch_surf_session(session)
    trim_surf_session(session)


def append_commentary(
    session: SurfSessionState,
    *,
    user_query: str,
    commentary: str,
) -> None:
    """Append one assistant commentary entry to the durable surf session."""
    session.commentary.append(
        SurfCommentaryEntry(
            created_at=_isoformat(_now_utc()),
            page_url=session.current_page_url,
            user_query=user_query,
            commentary=commentary,
        )
    )
    session.last_user_query = user_query
    session.last_action = session.last_action or "commentary"
    touch_surf_session(session)
    trim_surf_session(session)


def resolve_link_candidate(
    session: SurfSessionState,
    target: str,
) -> SurfLinkCandidate | None:
    """Resolve a numeric or fuzzy link target against the current page links."""
    current_page = get_current_page(session)
    if current_page is None:
        return None

    normalized_target = target.strip()
    if not normalized_target:
        return None

    if normalized_target.isdigit():
        index = int(normalized_target)
        if 1 <= index <= len(current_page.link_candidates):
            return current_page.link_candidates[index - 1]
        return None

    lowered_target = normalized_target.casefold()
    normalized_lookup, lookup_tokens = _normalize_link_lookup_text(normalized_target)
    best_match: SurfLinkCandidate | None = None
    best_score = 99
    for candidate in current_page.link_candidates:
        label = candidate.label.casefold()
        url = candidate.url.casefold()
        normalized_label, label_tokens = _normalize_link_lookup_text(candidate.label)
        normalized_url, url_tokens = _normalize_link_lookup_text(candidate.url)
        if lowered_target == label or lowered_target == url:
            return candidate
        if normalized_lookup and normalized_lookup == normalized_label:
            return candidate
        if normalized_lookup and normalized_lookup == normalized_url and best_score > 1:
            best_match = candidate
            best_score = 1
        elif lookup_tokens and lookup_tokens.issubset(label_tokens) and best_score > 2:
            best_match = candidate
            best_score = 2
        elif normalized_lookup and normalized_lookup in normalized_label and best_score > 3:
            best_match = candidate
            best_score = 3
        elif lowered_target in label and best_score > 4:
            best_match = candidate
            best_score = 4
        elif lookup_tokens and lookup_tokens.issubset(url_tokens) and best_score > 5:
            best_match = candidate
            best_score = 5
        elif normalized_lookup and normalized_lookup in normalized_url and best_score > 6:
            best_match = candidate
            best_score = 6
        elif lowered_target in url and best_score > 7:
            best_match = candidate
            best_score = 7
    return best_match
