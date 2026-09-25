"""Thread excerpt + manifest builders (prompt-injection reads).

Pure reads over the segmented store: ``build_recent_thread_excerpt`` strips
host-only commands, and ``build_thread_manifest`` renders a compact structured
summary. Both delegate to ``sdk.thread.ThreadStore``.
"""

from __future__ import annotations

from pathlib import Path

from host.host_commands import is_host_command_query as _is_host_command_query


def _store_from_package(thread_path: Path | None):
    """Resolve the store through the package namespace (``host.thread``).

    The redaction policy is a module-level global owned by ``host.thread``;
    importing the factory lazily keeps ``_store`` and the excerpt module
    import-cycle-free.
    """
    from host.thread import _thread_store  # noqa: PLC0415

    return _thread_store(thread_path)


# ── Excerpt (prompt injection) ─────────────────────────────────────────


def build_recent_thread_excerpt(
    *,
    thread_path: Path | None,
    limit: int = 3,
) -> str:
    """Build a short human-readable excerpt of recent thread entries.

    Delegates to :meth:`sdk.thread.ThreadStore.build_recent_excerpt`,
    passing the host's ``is_host_command_query`` filter so host-only
    commands are stripped from the transcript.
    """
    store = _store_from_package(thread_path)
    if store is None:
        return ""
    return store.build_recent_excerpt(
        limit=limit,
        is_host_command=_is_host_command_query,
    )


def build_thread_manifest(
    *,
    thread_path: Path | None,
    limit: int = 30,
) -> str:
    """Build a compact structured manifest of recent thread entries.

    Delegates to :meth:`sdk.thread.ThreadStore.build_manifest`.
    """
    store = _store_from_package(thread_path)
    if store is None:
        return ""
    return store.build_manifest(limit=limit)
