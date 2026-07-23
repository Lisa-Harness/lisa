"""Protocol for pack-registered host command handlers.

Packs register slash commands through the ``lisa.extensions.host_commands``
entry-point group. Each entry point returns a ``HostCommandHandler`` instance
that the host dispatch layer consults after its own built-in commands.

This keeps pack-owned behavior out of ``hosts/core`` — the host owns only the
dispatch mechanism and host-intrinsic commands (``/help``, ``/tokens``,
``/debug``, etc.). Pack-owned commands (``/checkpoint``, ``/restore``,
``/checkpoints``, ``/summarize-history``, etc.) live in the pack that owns
the operations they call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hosts.core.runtime import HostRuntime

__all__ = ["HostCommandHandler", "HostCommandSpec", "PackCommandEntry"]


@runtime_checkable
class HostCommandHandler(Protocol):
    """Async callable that handles one slash command for a host.

    The handler receives the live ``HostRuntime``, the stripped command
    text (without the leading prefix), and the original message. It
    returns a plain-text response string.
    """

    async def __call__(
        self,
        runtime: HostRuntime,
        stripped: str,
        message: str,
    ) -> str: ...


@dataclass(frozen=True)
class HostCommandSpec:
    """One pack-registered slash command.

    Attributes:
        prefix: The slash prefix (e.g. ``/checkpoint``).
        handler: Async callable matching :class:`HostCommandHandler`.
        help_text: One-line description for ``/help`` output.
    """

    prefix: str
    handler: HostCommandHandler
    help_text: str = ""


@dataclass(frozen=True)
class PackCommandEntry:
    """One pack's contribution to the host command dispatch table.

    A pack's entry-point function returns a ``list[HostCommandSpec]``.
    This wrapper carries the source pack name alongside the specs so
    the dispatcher can attribute commands and the host can exclude
    commands from disabled packs.
    """

    pack_name: str
    specs: list[HostCommandSpec]