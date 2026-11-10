"""Operator commands for the session plan archive.

``/session`` exposes the content-addressed session-source archive
(:mod:`runtime.session_archive`): what digests are on disk, which one the
live boot compiled, which one the latest paused turn saved, and how to
restore an archived source so a stranded pause can resume.

The command exists because the resume guard is fail-closed by design — a
continuation saved under plan A must not resume under plan B — but before
the archive that fail-closed decision could not be *acted on*: the operator
knew the saved digest only from the error text and had no bytes to restore
it from. The archive turns "different turn identity or session plan" from a
dead end into a bounded recovery: restore, resume, then re-apply the edit.

Dependency direction: ``_session`` → ``runtime.session_archive``,
``runtime.thread``, ``kernel.*``. ``_simple`` imports the handler, so this
module must not import ``_simple``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from runtime.runtime import HostRuntime
from runtime.session_archive import list_archived, restore_source, sessions_dir
from runtime.thread import load_recent_thread_entries

logger = logging.getLogger(__name__)

#: How far back the paused-turn digest scan looks. Pauses are recovered from
#: the newest entry first; a generous window keeps a resumed-then-paused
#: chain visible without reading the whole thread.
_PAUSE_SCAN_LIMIT = 50


def _latest_paused_digest(thread_path: Path | None) -> tuple[str, int] | None:
    """Return ``(digest, turn)`` from the newest continuation-bearing entry.

    Scans the recent tail for an entry carrying a ``turn_continuation_ref``
    and reads the saved continuation's ``session_plan_digest``. Unreadable
    references (missing result object, corrupt payload) are skipped, not
    fatal — the operator asked what is resumable, and a broken entry is not.
    """
    from runtime.turn_continuation import load_continuation

    entries = load_recent_thread_entries(thread_path=thread_path, limit=_PAUSE_SCAN_LIMIT)
    for entry in reversed(entries):
        provenance = entry.get("provenance") or {}
        if not provenance.get("turn_continuation_ref"):
            continue
        try:
            saved = load_continuation(thread_path, provenance)
        except (OSError, ValueError, KeyError) as exc:
            logger.debug("paused-turn continuation unreadable in /session scan: %s", exc)
            continue
        if saved is None:
            continue
        digest = str(saved.identity.session_plan_digest or "")
        return digest, int(entry.get("turn") or 0)
    return None


def _live_digest(runtime: HostRuntime) -> str:
    """Return the plan digest the running host booted with."""
    return str(runtime.session_plan.plan_digest)


def _render_status(runtime: HostRuntime) -> str:
    """Render the live digest, the latest paused-turn digest, and the verdict."""
    live = _live_digest(runtime)
    lines = [
        "## Session plan",
        "",
        f"- Live digest: `{live[:12]}` (booted from {runtime.session_plan.source_uri})",
        f"- Archive: {sessions_dir(runtime.paths.data_dir)}",
    ]
    paused = _latest_paused_digest(runtime.paths.thread_path)
    if paused is None:
        lines.append("- Latest paused turn: none — no continuation is in flight.")
    else:
        digest, turn = paused
        lines.append(f"- Latest paused turn: #{turn} — saved digest `{digest[:12] or 'none'}`")
        if digest and digest == live:
            lines.append(
                "- Verdict: **resumable** — the live plan matches the paused turn's digest. "
                'Send "continue" (or an affirmative) to resume it.'
            )
        else:
            lines.append(
                "- Verdict: **blocked** — the live plan differs from the paused turn's "
                "digest (the session file changed since that turn paused). To recover it, "
                "restore the archived source with the matching digest / "
                f"`/session restore {digest[:12]}`, resume the turn, then re-apply your edit."
            )
    return "\n".join(lines)


#: Render cap for the archive list — beyond this the tail is summarized.
_ARCHIVE_RENDER_LIMIT = 50


def _render_archive(runtime: HostRuntime) -> str:
    """Render the archive inventory as a Markdown table."""
    live = _live_digest(runtime)
    items = list_archived(runtime.paths.data_dir)
    if not items:
        return (
            "## Session archive\n\n"
            "The archive is empty. Every boot archives its resolved session source; "
            "one appears after the next launch."
        )
    lines = [
        "## Session archive",
        "",
        "| Digest | Archived (UTC) | Source | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for item in items[:_ARCHIVE_RENDER_LIMIT]:
        digest = str(item.get("digest") or "")
        notes: list[str] = []
        if digest == live:
            notes.append("live digest")
        if not item.get("intact", False):
            notes.append("**source file missing**")
        lines.append(
            f"| `{digest[:12]}` | {item.get('archived_at', '')} | {item.get('source_uri', '')} "
            f"| {', '.join(notes) or (f'restorable: `/session restore {digest[:12]}`')} |"
        )
    if len(items) > _ARCHIVE_RENDER_LIMIT:
        lines.append(f"\n… and {len(items) - _ARCHIVE_RENDER_LIMIT} older entries.")
    return "\n".join(lines)


def _restore(runtime: HostRuntime, digest: str) -> str:
    """Restore an archived source and report the verification result."""
    try:
        restored = restore_source(digest, runtime.paths.data_dir)
    except ValueError as exc:
        return f"## Session restore failed\n\n{exc}"
    return (
        "## Session restore\n\n"
        f"- Restored `{restored}` from the archive.\n"
        "- The restored bytes verified: they compile back to the archived digest.\n"
        "- Your previous session text was snapshotted beside it as `.bak`.\n"
        "- The restore takes effect on the next launch (the host boots per message / "
        'per session). Send "continue" afterward to resume the paused turn under '
        "the restored plan."
    )


def handle_session_command(runtime: HostRuntime, stripped: str, message: str) -> str:
    """Handle ``/session [status|archive|restore <digest>]``.

    ``status`` (the default) compares the live plan digest against the
    latest paused turn's saved digest and states the verdict plainly.
    ``archive`` lists the entries on disk. ``restore`` replays an archived
    source to its recorded path (verified, with a ``.bak`` safety net).
    Unknown subcommands report usage rather than silently defaulting, so a
    typo cannot look like it restored something.
    """
    arg = stripped[len("/session") :].strip()
    subcommand, _, remainder = arg.partition(" ")
    subcommand = subcommand.lower()
    digest = remainder.strip().lstrip("`").rstrip("`").strip()

    if subcommand in ("", "status", "show"):
        return _render_status(runtime)
    if subcommand == "archive":
        return _render_archive(runtime)
    if subcommand == "restore":
        if not digest:
            return (
                "Usage: /session restore <digest> — the digest prefix from "
                "/session archive (e.g. /session restore 441d19b2)."
            )
        return _restore(runtime, digest)
    return (
        "Usage: `/session [status | archive | restore <digest>]`\n\n"
        f"Unrecognized subcommand: {subcommand!r}"
    )


__all__ = ["handle_session_command"]
