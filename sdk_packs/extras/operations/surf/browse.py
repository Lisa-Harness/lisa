"""Surf workflow pre-turn operation.

This operation owns the browser-facing half of the iterative surf workflow. On each
turn it loads the durable surf session for the active conversation surface, decides
whether the user's message implies navigation or analysis on the current page,
executes browser actions when needed, and returns a structured pre-turn
briefing for the workflow model.
"""

from __future__ import annotations

import re

from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    SurfaceContext,
    SurfPageCapture,
    SurfTrace,
    ToolExecutionContext,
)
from kernel.surf_workflow import (
    append_page_capture,
    build_surf_stage_data,
    create_surf_session,
    effective_surface_id,
    get_current_page,
    load_surf_session,
    resolve_link_candidate,
    save_surf_session,
    touch_surf_session,
)
from lisa_sdk.config import settings

_FLUSH_QUERIES = {
    "flush",
    "flush session",
    "flush browser session",
    "flush surf session",
}
_HISTORY_QUERIES = {"history", "show history", "show pages", "page history"}
_SUMMARY_QUERIES = {"summary", "recap", "summarize", "where are we"}
_NAVIGATION_PREFIX_RE = re.compile(
    r"^(?:open|go\s+to|goto|follow|visit)\s+(?P<target>.+)$",
    re.IGNORECASE,
)
_URL_LIKE_RE = re.compile(r"^(?:https?://|www\.|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/\S*)?)")


def _is_flush_query(query: str) -> bool:
    """Return whether the current turn requests shared browser-session deletion."""
    return query.strip().lower() in _FLUSH_QUERIES


def _is_history_query(query: str) -> bool:
    """Return whether the user is asking for visited-page history."""
    return query.strip().lower() in _HISTORY_QUERIES


def _is_summary_query(query: str) -> bool:
    """Return whether the user is asking for a recap of the surf session."""
    return query.strip().lower() in _SUMMARY_QUERIES


def _truncate_surf_excerpt(excerpt: str, limit: int = 600) -> str:
    """Collapse and cap extracted page text for the surf workflow briefing."""
    normalized = " ".join(excerpt.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)] + "..."


def _effective_surface_context(ctx: OperationExecutionContext) -> SurfaceContext:
    """Return the best available surface context for one surf operation call."""
    fallback_surface_id = effective_surface_id(
        SurfaceContext(
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
        ),
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
    )
    return SurfaceContext(
        surface_type=ctx.surface_type,
        surface_id=fallback_surface_id,
        memory_scope=ctx.memory_scope,
    )


def _extract_navigation_target(query: str, *, has_session: bool) -> str | None:
    """Return the user's requested navigation target when the turn implies one."""
    stripped = query.strip()
    if not stripped:
        return None

    prefix_match = _NAVIGATION_PREFIX_RE.match(stripped)
    if prefix_match is not None:
        return (prefix_match.group("target") or "").strip()

    if stripped.isdigit():
        return stripped

    if _URL_LIKE_RE.match(stripped):
        return stripped

    if not has_session:
        return stripped

    return None


def _is_url_like_target(target: str) -> bool:
    """Return whether one navigation target already looks like a URL or hostname."""
    return _URL_LIKE_RE.match(target.strip()) is not None


def _render_history_lines(session_pages: list[SurfPageCapture]) -> list[str]:
    """Return recent page history lines for the current surf session."""
    lines: list[str] = []
    start_index = max(1, len(session_pages) - 4)
    for offset, page in enumerate(session_pages[-5:], start=start_index):
        final_url = page.trace.final_url or page.trace.requested_url
        title = page.trace.page_title or final_url
        lines.append(f"{offset}. {title} — {final_url}")
    return lines


def _render_link_lines(page: SurfPageCapture | None) -> list[str]:
    """Return numbered follow-up link lines for the current page."""
    if page is None:
        return []
    return [
        f"{index}. {link.label or link.url} — {link.url}"
        for index, link in enumerate(
            page.link_candidates[: settings.surf_session_max_links], start=1
        )
    ]


def _load_current_page_screenshot(session, *, blob_io=None) -> bytes | None:
    """Load the persisted screenshot for the current surf page when present."""
    current_page = get_current_page(session) if session is not None else None
    if current_page is None or blob_io is None:
        return None
    return blob_io.read_surf_screenshot(current_page.trace.screenshot_blob_path)


def _render_surf_briefing(
    session,
    *,
    action_notes: list[str],
    error_note: str = "",
) -> str:
    """Render one structured pre-turn briefing for the surf workflow model."""
    current_page = get_current_page(session) if session is not None else None
    lines = [
        "Surf workflow briefing",
        f"- Session id: {session.session_id}" if session is not None else "- Session id: none",
        f"- Pages visited: {len(session.pages)}" if session is not None else "- Pages visited: 0",
        (
            f"- Commentary entries: {len(session.commentary)}"
            if session is not None
            else "- Commentary entries: 0"
        ),
        (
            f"- Current URL: {session.current_page_url or 'none'}"
            if session is not None
            else "- Current URL: none"
        ),
        (
            f"- Current page: {session.current_page_title or 'none'}"
            if session is not None
            else "- Current page: none"
        ),
        "",
        "Session summary",
        session.summary if session is not None else "No surf session is active yet.",
    ]

    if action_notes:
        lines.extend(["", "Turn action"])
        for note in action_notes:
            lines.append(f"- {note}")

    if error_note:
        lines.extend(["", "Turn issue", f"- {error_note}"])

    if current_page is not None:
        final_url = current_page.trace.final_url or current_page.trace.requested_url
        excerpt = _truncate_surf_excerpt(current_page.trace.text_excerpt)
        lines.extend(
            [
                "",
                "Current page details",
                f"- Requested URL: {current_page.trace.requested_url}",
                f"- Final URL: {final_url}",
                f"- Page title: {current_page.trace.page_title or 'Untitled page'}",
                f"- HTTP status: {current_page.trace.http_status or 'unknown'}",
                f"- Duration: {current_page.trace.duration_ms}ms",
                (
                    "- Shared session: "
                    f"loaded={'yes' if current_page.trace.used_persisted_session else 'no'}, "
                    f"saved={'yes' if current_page.trace.saved_persisted_session else 'no'}"
                ),
                (
                    "- Screenshot: "
                    f"{settings.surf_container}/{current_page.trace.screenshot_blob_path}"
                    if current_page.trace.screenshot_blob_path
                    else "- Screenshot: not available"
                ),
                "",
                "Current page excerpt",
                (
                    excerpt
                    if excerpt
                    else "The page loaded, but there was not much readable body text to extract."
                ),
                "",
                "Recent page history",
            ]
        )
        lines.extend(_render_history_lines(session.pages))

        link_lines = _render_link_lines(current_page)
        if link_lines:
            lines.extend(["", "Available follow-up links"])
            lines.extend(link_lines)

        lines.extend(["", "Console errors"])
        if current_page.trace.console_errors:
            for entry in current_page.trace.console_errors[:3]:
                lines.append(f"- {entry}")
        else:
            lines.append("- none")
    else:
        lines.extend(
            [
                "",
                "Current page details",
                "- No page is active yet. Start surf mode with a public HTTPS URL.",
            ]
        )
    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run one surf workflow pre-turn and return the page-context briefing.

    The app side must inject `ctx.execute_tool_fn` and `ctx.blob_io`.
    """
    if ctx.execute_tool_fn is None:
        return OperationResult(
            output_text="⛔ Surf browse operation unavailable: execute_tool_fn not injected."
        )

    normalized_query = ctx.query.strip()
    surface_context = _effective_surface_context(ctx)
    workspace_id = ctx.workspace_id or settings.workspace_id_fallback
    session = load_surf_session(
        workspace_id,
        surface_context,
        user_id=ctx.user_id,
        user_slug=ctx.user_slug,
        blob_io=ctx.blob_io,
    )
    action_notes: list[str] = []
    error_note = ""
    screenshot_bytes: bytes | None = None

    if _is_flush_query(normalized_query):
        tool_result = await ctx.execute_tool_fn(
            "surf_session_flush",
            ToolExecutionContext(
                tool_id="surf_session_flush",
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_type=ctx.surface_type,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                query=normalized_query,
            ),
        )
        if tool_result.error:
            error_note = tool_result.error
        else:
            deleted = bool(tool_result.output_json.get("deleted"))
            action_notes.append(
                "Shared browser session was flushed."
                if deleted
                else "Shared browser session was already empty."
            )

    navigation_target = _extract_navigation_target(
        normalized_query,
        has_session=session is not None,
    )
    if navigation_target is not None and not _is_flush_query(normalized_query):
        resolved_url = navigation_target
        matched_link = None
        should_navigate = True
        if session is not None:
            matched_link = resolve_link_candidate(session, navigation_target)
            if matched_link is not None:
                resolved_url = matched_link.url
            elif not _is_url_like_target(navigation_target):
                should_navigate = False
                error_note = (
                    f"Could not match '{navigation_target}' to one of the current page links. "
                    "Use a link number or a label from the current page briefing."
                )
                action_notes.append(
                    "Stayed on the current page because the requested follow-up link was not found."
                )

        if should_navigate:
            tool_result = await ctx.execute_tool_fn(
                "surf_browser",
                ToolExecutionContext(
                    tool_id="surf_browser",
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    user_slug=ctx.user_slug,
                    surface_type=ctx.surface_type,
                    surface_id=ctx.surface_id,
                    memory_scope=ctx.memory_scope,
                    mode_id=ctx.mode_id,
                    query=resolved_url,
                ),
            )
            if tool_result.error:
                error_note = tool_result.error
            else:
                trace = SurfTrace.model_validate(tool_result.output_json["trace"])
                page_payload = tool_result.output_json.get("page") or {}
                page_capture = (
                    SurfPageCapture.model_validate(page_payload)
                    if page_payload
                    else SurfPageCapture(trace=trace)
                )
                if session is None:
                    session = create_surf_session(workspace_id, surface_context)
                append_page_capture(
                    session,
                    page_capture,
                    user_query=normalized_query,
                    action="navigate",
                )
                save_surf_session(session, blob_io=ctx.blob_io)
                screenshot_bytes = tool_result.extra.get("screenshot_bytes")
                if matched_link is not None:
                    action_notes.append(
                        "Opened link "
                        f"'{matched_link.label or matched_link.url}' -> {matched_link.url}."
                    )
                else:
                    action_notes.append(f"Navigated to {trace.final_url or trace.requested_url}.")
        elif session is not None:
            touch_surf_session(session)
            save_surf_session(session, blob_io=ctx.blob_io)
    elif session is None:
        return OperationResult(
            output_text=_render_surf_briefing(
                session,
                action_notes=action_notes,
                error_note=error_note,
            )
        )
    else:
        touch_surf_session(session)
        save_surf_session(session, blob_io=ctx.blob_io)
        if _is_history_query(normalized_query):
            action_notes.append("The user asked for the visited-page history.")
        elif _is_summary_query(normalized_query):
            action_notes.append("The user asked for a recap of the surf session so far.")
        else:
            action_notes.append(
                "No navigation happened this turn; answer using the current page "
                "and prior surf history."
            )

    if screenshot_bytes is None:
        screenshot_bytes = _load_current_page_screenshot(session, blob_io=ctx.blob_io)

    return OperationResult(
        output_text=_render_surf_briefing(
            session, action_notes=action_notes, error_note=error_note
        ),
        extra={
            "screenshot_bytes": screenshot_bytes,
            "stage_data_update": build_surf_stage_data(session),
        },
    )
