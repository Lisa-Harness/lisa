"""Runtime-neutral extension hooks for context-item injection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kernel.models import ContextItem, UnifiedState
from kernel.seams.protocols import StateProvider


@dataclass(slots=True)
class ExtensionRuntimeContext:
    """Host services injected into extension-owned context-item builders."""

    state_provider: StateProvider | None = None
    http_client: Any = None
    event_bus: Any = None
    openai_client: Any = None


ExtensionContextItemBuilder = Callable[
    [ExtensionRuntimeContext, UnifiedState],
    "list[ContextItem]",
]


__all__ = [
    "ExtensionContextItemBuilder",
    "ExtensionRuntimeContext",
]
