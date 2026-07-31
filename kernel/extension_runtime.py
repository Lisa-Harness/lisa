"""Runtime-neutral extension hooks for context-evidence injection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from kernel.models import AnomalyAlert, ContextEvidence, UnifiedState
from kernel.seams.protocols import StateProvider


@dataclass(slots=True)
class ExtensionRuntimeContext:
    """Host services injected into extension-owned context-evidence builders."""

    state_provider: StateProvider | None = None
    http_client: Any = None
    event_bus: Any = None
    openai_client: Any = None
    complete_with_profile: Callable[..., Awaitable[str]] | None = None


ExtensionAnomalyDetector = Callable[[UnifiedState], Awaitable[list[AnomalyAlert]]]
ExtensionContextEvidenceBuilder = Callable[
    [ExtensionRuntimeContext, UnifiedState],
    "list[ContextEvidence]",
]


__all__ = [
    "ExtensionAnomalyDetector",
    "ExtensionContextEvidenceBuilder",
    "ExtensionRuntimeContext",
]
