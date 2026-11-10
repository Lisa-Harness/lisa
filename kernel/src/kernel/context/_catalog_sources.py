"""Kernel contracts for runtime-injected catalog/content sources (Slice B).

The kernel's synchronous catalog/content adapters have two transports: the
stdio MCP subprocesses (``lisa-catalog``/``lisa-content``) and — since
Slice B of ``docs/SPEC/PROMPT_STARTUP.md`` — runtime-injected in-process
sources that call the shared SDK service directly. The kernel defines the
typed consumer contracts here; runtime wires SDK implementations. The
kernel never imports ``sdk``, ``mcp_server`` or runtime code.

In-process sources must be synchronous: the kernel consumers run on the
host thread and must not synchronously submit work to a blocked event
loop. A failing source must raise — never silently fall back to the
stdio transport.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["CatalogSource", "CatalogSourceConfigLike", "ContentSource"]


@runtime_checkable
class CatalogSourceConfigLike(Protocol):
    """Structural view of the frozen SDK source configuration.

    Typed as ``CatalogSourceConfig`` in the SDK; the kernel sees only this
    structural contract so it never imports the SDK.
    """

    @property
    def packs_venv_site_packages(self) -> object: ...  # pragma: no cover


@runtime_checkable
class CatalogSource(Protocol):
    """Synchronous catalog queries served without the lisa-catalog subprocess."""

    def list_modes(self) -> str:
        """Return the mode catalog as JSON (``lisa-catalog`` response shape)."""
        ...  # pragma: no cover

    def list_operations(self) -> str:
        """Return the operation catalog as JSON."""
        ...  # pragma: no cover

    def list_tools(self) -> str:
        """Return the internal tool catalog as JSON."""
        ...  # pragma: no cover

    def invalidate(self) -> None:
        """Clear the source's catalog caches (explicit, idempotent)."""
        ...  # pragma: no cover

    def list_content_roots(self) -> str:
        """Return the effective root list as a JSON array of path strings."""
        ...  # pragma: no cover


@runtime_checkable
class ContentSource(Protocol):
    """Synchronous authored-text reads served without the lisa-content subprocess."""

    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string."""
        ...  # pragma: no cover

    def invalidate(self) -> None:
        """Clear the source's read cache after content roots change."""
        ...  # pragma: no cover

    def roots_fingerprint(self) -> str:
        """Return a stable string key for the source's effective root set."""
        ...  # pragma: no cover
