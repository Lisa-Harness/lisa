"""Content-addressed archive of the compiled session source.

Every turn continuation stamps
:class:`~kernel.turn_hooks.TurnIdentity.session_plan_digest` — the SHA-256 of
the compiled session plan — and the resume guard
(:func:`kernel.turn_runner._validate_turn_continuation` via
:meth:`TurnIdentity.same_turn`) rejects any saved continuation whose saved
digest differs from the live one. Editing the session file between a pause
and its resume therefore strands the paused turn with

    Turn failed: ValueError: Continuation belongs to a different turn identity
    or session plan (saved digest … != live digest …).

The archive makes that failure recoverable in place: boot writes the exact
resolved session source (host-layer ``from:``/``body:`` directives already
expanded) to ``<data_dir>/state/sessions/<digest>.md``, so the bytes behind
every digest a turn was ever paused under stay on disk. The plan is
content-addressed, so restoring the bytes restores the digest — with one
scope constraint: the digest payload includes the plan's ``source_uri``, so
the restore must land at the same file path the original runtime resolved
the session from. :func:`restore_source` records that URI in the archive
sidecar, restores to it, and verifies the restored bytes still compile to
the same digest.

Archive layout (``state`` is the machine-written, back-up-able category in
``sdk.layout``)::

    <data_dir>/state/sessions/<digest>.md          # resolved session source
    <data_dir>/state/sessions/<digest>.meta.json   # source_uri + timestamps

One digest maps to exactly one compiled plan, so a same-content boot
rewrites the same bytes (an idempotent no-op) and distinct edits are
distinct files. Nothing here prunes: the archive grows one small file per
distinct session edit, and cleanup stays the operator's call.

Dependency direction
--------------------
``session_archive.py`` ← ``runtime.py`` (boot hook), ``commands._session``
(callers import from here).

``session_archive.py`` → ``kernel.session_plan``, ``runtime.session_context``
(stdlib + those; never imports transport or command code).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdk.layout import STATE_DIR

logger = logging.getLogger(__name__)

_SESSIONS_DIR_NAME = "sessions"
_META_SUFFIX = ".meta.json"
_SOURCE_SUFFIX = ".md"
_CONFIG_DIR = "config"
_DEFAULT_SESSION_NAME = "session_context.md"


def sessions_dir(data_dir: Path) -> Path:
    """Return the session archive directory for a host ``data_dir``."""
    return Path(data_dir) / STATE_DIR / _SESSIONS_DIR_NAME


def resolve_boot_source(
    *,
    session_context: str,
    data_dir: Path,
) -> tuple[str, str] | None:
    """Re-resolve the boot session source as ``(source_uri, resolved_text)``.

    Mirrors :func:`runtime.session_context.compile_selected_session_plan`'s
    resolution exactly — same resolver, same ``from:``/``body:`` expansion —
    so the archived bytes are precisely what the compiled plan consumed.
    Returns ``None`` (and logs) when re-resolution fails; archiving is a boot
    convenience and must never break startup.
    """
    from runtime.session_context import resolve_session_context_source
    from runtime.session_from_resolver import resolve_session_from_directives

    try:
        source = resolve_session_context_source(session_context=session_context, data_dir=data_dir)
        resolved_text = resolve_session_from_directives(
            text=source.text,
            source_uri=source.source_uri,
            pack_session_lookup=_boot_pack_session_lookup(data_dir=data_dir),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("session archive skipped: boot source re-resolution failed: %s", exc)
        return None
    return source.source_uri, resolved_text


def _boot_pack_session_lookup(*, data_dir: Path) -> Callable[..., Path]:
    """Build the ``pack_session_lookup`` closure boot resolution needs.

    Matches :func:`runtime.session_context._make_pack_session_lookup`; kept
    separate so pack discovery failures degrade to the archive skip above
    instead of raising from boot.
    """
    from runtime.session_context import _make_pack_session_lookup

    return _make_pack_session_lookup(data_dir=data_dir)


def _plan_digest(resolved_text: str, source_uri: str) -> str:
    """Return the compiled plan digest for ``resolved_text`` at ``source_uri``."""
    from kernel.session_plan import compile_session_plan

    return compile_session_plan(text=resolved_text, source_uri=source_uri).plan_digest


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a same-dir temp file, then rename.

    A crash between writing the source and its meta sidecar would otherwise
    leave an archive entry that claims a digest its bytes do not honor.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def archive_session_source(
    source_uri: str,
    resolved_text: str,
    *,
    data_dir: Path,
) -> Path:
    """Archive the resolved session source under its plan digest.

    Computes the digest :func:`kernel.session_plan.compile_session_plan`
    stamps on the plan, then writes ``<digest>.md`` plus a ``.meta.json``
    sidecar (recording ``source_uri`` — the restore target — and
    provenance) under :func:`sessions_dir` for ``data_dir``. Any source is
    archived: the config default, an env-var override, or an explicit CLI
    ``--session-context`` file all restore the same way — to their
    recorded URI.

    Args:
        source_uri: The plan's resolved source URI. Also the restore target:
            the digest payload includes it, so the same bytes compile to the
            same digest only at the same path. Recorded in the meta sidecar.
        resolved_text: The fully resolved session text — host-layer
            ``from:``/``body:`` directives already expanded — exactly as
            boot compiled it.
        data_dir: The host ``data_dir`` the archive directory lives under.

    Returns:
        The archive path written.

    Raises:
        kernel.session_plan.SessionPlanError: When the text does not
            compile — the archive must only ever hold bytes that honor
            their digest. Callers verify before calling, so production
            cannot trigger this.
    """
    digest = _plan_digest(resolved_text, source_uri)
    archive = sessions_dir(data_dir)
    meta = {
        "digest": digest,
        "source_uri": source_uri,
        "archived_at": datetime.now(UTC).isoformat(),
        "bytes": len(resolved_text.encode("utf-8")),
        "sha256": hashlib.sha256(resolved_text.encode("utf-8")).hexdigest(),
    }
    source_file = archive / f"{digest}{_SOURCE_SUFFIX}"
    _atomic_write(source_file, resolved_text)
    _atomic_write(
        archive / f"{digest}{_META_SUFFIX}",
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
    )
    logger.debug("archived session source digest=%s uri=%s", digest[:12], source_uri)
    return source_file


def read_archived(digest: str, data_dir: Path) -> tuple[str, dict[str, Any]] | None:
    """Return ``(source_text, meta)`` for ``digest``, or ``None`` when absent.

    ``digest`` may be the full 64 hex chars or any unambiguous prefix — an
    ambiguous prefix returns ``None`` rather than guessing.
    """
    directory = sessions_dir(data_dir)
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{digest}*{_SOURCE_SUFFIX}"))
    if len(matches) != 1:
        return None
    source_path = matches[0]
    meta_path = source_path.with_suffix("").with_suffix(_META_SUFFIX)
    meta: dict[str, Any] = {}
    try:
        text = source_path.read_text(encoding="utf-8")
        if meta_path.is_file():
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
    except (OSError, json.JSONDecodeError):
        return None
    return text, meta


def list_archived(data_dir: Path) -> list[dict[str, Any]]:
    """Return the archive inventory as meta dicts, newest first.

    Entries whose source file is missing (partial archive, manual cleanup)
    are reported with ``intact: False`` rather than silently dropped.
    """
    directory = sessions_dir(data_dir)
    if not directory.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for meta_path in sorted(directory.glob(f"*{_META_SUFFIX}")):
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, dict):
            continue
        source_path = meta_path.with_suffix("").with_suffix(_SOURCE_SUFFIX)
        items.append({**loaded, "source_path": str(source_path), "intact": source_path.is_file()})
    items.sort(key=lambda entry: str(entry.get("archived_at") or ""), reverse=True)
    return items


def restore_source(digest: str, data_dir: Path) -> Path:
    """Restore the archived source for ``digest`` to its recorded ``source_uri``.

    The restore replays the archive bytes at the exact path the original
    runtime resolved the session from — the digest is URI-scoped, so any
    other path would compile the same text to a different digest. After
    writing, the restored file is re-compiled and verified against the
    archived digest; a mismatch raises and the caller sees the failure
    instead of a silent, still-broken resume.

    The replaced session file is snapshotted to ``<name>.bak`` beside itself
    first, so the restore is reversible without this module.

    Args:
        digest: Full digest or unambiguous prefix of the entry to restore.
        data_dir: The host ``data_dir`` the archive lives under (the
            fallback restore target when the entry carries no usable
            recorded URI).

    Returns:
        The restored session file path.

    Raises:
        ValueError: When the digest is absent or ambiguous, or when the
            restore fails its compile-back verification.
    """
    from kernel.session_plan import SessionPlanError, compile_session_plan

    archived = read_archived(digest, data_dir)
    if archived is None:
        raise ValueError(
            f"No archived session source matches digest prefix {digest!r}; "
            "run /session archive to list the entries on disk."
        )
    text, meta = archived
    recorded_uri = str(meta.get("source_uri") or "")
    fallback = data_dir / _CONFIG_DIR / _DEFAULT_SESSION_NAME
    target = Path(recorded_uri) if recorded_uri else fallback
    if not target.is_absolute():
        # A relative recorded URI (corrupt sidecar) is un-restorable safely.
        logger.warning(
            "archived source_uri %r is not an absolute path; restoring to %s instead",
            recorded_uri,
            fallback,
        )
        target = fallback
    backup = target.with_suffix(".md.bak")
    if target.exists():
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    _atomic_write(target, text)
    try:
        live_digest = compile_session_plan(text=text, source_uri=str(target)).plan_digest
    except SessionPlanError as exc:
        raise ValueError(
            f"Restored {target} does not compile; the archive entry is corrupt ({exc}). "
            f"Your previous session text is preserved at {backup}."
        ) from exc
    if live_digest != str(meta.get("digest")):
        raise ValueError(
            f"Restored {target} compiles to {live_digest[:12]}, not the archived "
            f"{str(meta.get('digest'))[:12]}; the archive entry is inconsistent. "
            f"Your previous session text is preserved at {backup}."
        )
    logger.info("restored archived session source digest=%s to %s", live_digest[:12], target)
    return target
