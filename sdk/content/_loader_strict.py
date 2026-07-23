"""Strict authored-text loading — raises instead of defaulting to empty string.

This module is the single source of truth for the "required content" contract.
Consumers that depend on authored text being present call
:func:`require_authored_text` instead of calling the loader directly and
tolerating ``""``. The helper raises :class:`MissingAuthoredContentError` on
three failure modes:

1. The loader callable is not wired (``None`` / not callable).
2. The loader raises an exception.
3. The loader returns an empty or whitespace-only string (missing content).

This replaces the ad-hoc ``_load_system_prompt`` / ``_load_text`` duplicates
in pack operations and the silent ``return ""`` / ``or ""`` defaults scattered
across prompt assembly. The transport layer (``ContentReader``,
``MCPContentRegistry``) still returns ``""`` on miss — strictness lives at the
consumer layer so callers that legitimately tolerate missing content
(optional assets, diagnostics) can keep using the raw loader.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

__all__ = [
    "MissingAuthoredContentError",
    "require_authored_text",
    "validate_required_content",
]


class MissingAuthoredContentError(RuntimeError):
    """Raised when required authored content is missing, empty, or unloadable.

    Carries the authored-text ``path`` so callers and logs can pinpoint which
    asset failed. The message is human-readable and actionable (it names the
    path and the failure mode).
    """

    def __init__(self, message: str, *, path: str = "") -> None:
        super().__init__(message)
        self.path = path


def require_authored_text(
    loader: Callable[[str], str] | None,
    path: str,
    *,
    context: str = "",
) -> str:
    """Load authored text, raising on any failure mode.

    Args:
        loader: The ``read_authored_text`` callable (e.g. from
            ``TurnHooks.context_seams.content_registry`` or the SDK
            ``ContentReader``). ``None`` or a non-callable raises.
        path: The content-root-relative path to the authored asset.
        context: Optional human-readable context appended to error messages
            (e.g. ``"synthesize_learning operation"``) to aid debugging.

    Returns:
        The stripped authored text content.

    Raises:
        MissingAuthoredContentError: If the loader is not wired, raises an
            exception, or returns an empty/whitespace-only string.
    """
    suffix = f" ({context})" if context else ""
    if not callable(loader):
        raise MissingAuthoredContentError(
            f"read_authored_text_fn is not wired; cannot load authored content at {path}{suffix}",
            path=path,
        )
    try:
        loaded = str(loader(path) or "").strip()
    except Exception as exc:  # pylint: disable=broad-except
        raise MissingAuthoredContentError(
            f"read_authored_text_fn failed for {path}{suffix}: {exc}",
            path=path,
        ) from exc
    if not loaded:
        raise MissingAuthoredContentError(
            f"Authored text at {path} is empty or missing{suffix}",
            path=path,
        )
    return loaded


def validate_required_content(
    loader: Callable[[str], str],
    paths: Iterable[str],
    *,
    context: str = "startup validation",
) -> None:
    """Eagerly load all required authored paths, failing fast on any miss.

    Called at host startup (from ``setup_framework``) so a broken content
    install fails at boot rather than producing a silently-degraded first
    turn. Each path is loaded via :func:`require_authored_text`; the first
    failure raises and aborts startup.

    Args:
        loader: The ``read_authored_text`` callable.
        paths: Required authored-text paths to validate.
        context: Context label for error messages.

    Raises:
        MissingAuthoredContentError: On the first missing/empty path.
    """
    for path in paths:
        require_authored_text(loader, path, context=context)
