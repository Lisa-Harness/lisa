"""Runtime-neutral extension hooks for context-evidence injection.

After the 2026-07-11 strip-down, the extension runtime surface is minimal:
``ExtensionRuntimeContext`` carries the host services a pack's context-evidence
builder needs, and ``ExtensionAnomalyDetector`` is the callable shape for
anomaly detection (now folded into the bottom-half ProcessingLoop contract).

Everything else — scheduled jobs, state hydration, bus consumers, debug
context fragments, heartbeat context fragments, digest sections, synthesis
stage executors — was deleted because it had zero consumers in the public
tree. The bottom half is now ``kernel.intake.contracts.ProcessingLoop``.
"""

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
