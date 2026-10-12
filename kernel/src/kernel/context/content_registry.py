"""Module-level content-registry holder for authored context assets.

The legacy context-frame system (``ContextFrameId`` / ``CONTEXT_FRAME_IDS``
and the ``read_context_frame`` / ``render_context_frame`` helpers) was
superseded by the session-owned :class:`~kernel.context.context_mapper.ContextMapper`
and removed. What remains is the single module-level content registry that
hosts inject at startup via :func:`set_content_registry` and read back via
:func:`get_content_registry` to populate ``TurnHooks.context_seams.content_registry``.
No disk-based default; callers that read before the registry is set get
``None``.
"""

from __future__ import annotations

from kernel.seams.protocols import ContentRegistry

# Module-level content registry. Hosts MUST inject one via
# ``set_content_registry`` before any authored asset is read. No disk-based
# default.
_content_registry: ContentRegistry | None = None


def set_content_registry(registry: ContentRegistry) -> None:
    """Override the module-level content registry."""

    global _content_registry  # noqa: PLW0603 — module-level singleton seam
    _content_registry = registry


def get_content_registry() -> ContentRegistry | None:
    """Return the injected content registry, or ``None`` before injection."""

    return _content_registry
