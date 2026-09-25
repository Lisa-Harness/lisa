"""ContextSeams + lazy seam defaults for the turn loop.

The 7 ``_lazy_*`` factories defer importing optional concrete classes until
first use so hosts that never wire a guard/policy/router pay no import cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from kernel.seams.protocols import (
    ContentRegistry,
    DataSanitizer,
    EventBus,
    ExtensionItemBuilder,
    InputGuard,
    ModelPolicy,
    ModelRouter,
    OutputGuard,
    ProviderGatewayProvider,
)

logger = logging.getLogger(__name__)


def _lazy_model_policy() -> ModelPolicy:
    from kernel.context import ManifestModelPolicy

    return ManifestModelPolicy()


def _lazy_provider_gateways() -> ProviderGatewayProvider | None:
    # The kernel ships no default gateway provider. Concrete gateway
    # implementations live in the SDK (sdk/providers/) and are discovered
    # via the lisa.extensions.provider_gateways entry-point group. Hosts
    # wire the discovered ProviderGatewayProvider into ContextSeams; when
    # nothing is discovered, this stays None and a model call will fail
    # with a clear error.
    return None


def _lazy_event_bus() -> EventBus:
    from kernel.context import InProcessEventBus

    return InProcessEventBus()


def _lazy_data_sanitizer() -> DataSanitizer:
    from kernel.context import PassThroughSanitizer

    return PassThroughSanitizer()


def _lazy_extension_item_builder() -> ExtensionItemBuilder:
    from kernel.context import NoOpExtensionItemBuilder

    return NoOpExtensionItemBuilder()


def _lazy_input_guard() -> InputGuard:
    from kernel.context import PassThroughInputGuard

    return PassThroughInputGuard()


def _lazy_output_guard() -> OutputGuard:
    from kernel.context import PassThroughOutputGuard

    return PassThroughOutputGuard()


@dataclass
class ContextSeams:
    """Bundle of context-layer seams injected through ``TurnHooks``.

    These six seams invert the private ``app.*`` dependencies that
    ``kernel/context/`` previously took. Defaults wire the public
    implementations shipped under ``kernel.context``; the production app
    overrides any of them with its own without a shim layer.
    """

    content_registry: ContentRegistry = field(default=None)  # type: ignore[assignment]
    model_policy: ModelPolicy = field(default_factory=_lazy_model_policy)
    model_router: ModelRouter | None = None
    """Per-iteration model router. When ``None`` (default), the loop falls
    back to ``ModelPolicy.choose`` if present, then to the statically-resolved
    ``preferred_model_id`` — preserving today's behavior. Hosts wire a
    discovered ``ModelRouter`` (from ``lisa.extensions.model_routers``) here
    to enable per-iteration model selection without extra model calls."""
    provider_gateways: ProviderGatewayProvider | None = field(
        default_factory=_lazy_provider_gateways
    )
    """Provider gateway dispatch. When ``None`` (default), the kernel ships
    no gateways — concrete implementations are discovered via
    ``lisa.extensions.provider_gateways`` (SDK ships the reference backends)
    and wired by the host. A model call with no gateway wired fails with a
    clear error."""
    event_bus: EventBus = field(default_factory=_lazy_event_bus)
    data_sanitizer: DataSanitizer = field(default_factory=_lazy_data_sanitizer)
    extension_item_builder: ExtensionItemBuilder = field(
        default_factory=_lazy_extension_item_builder
    )
    input_guard: InputGuard = field(default_factory=_lazy_input_guard)
    """Validates user input before the model call (Phase 1). Default:
    ``PassThroughInputGuard`` allows everything. Hosts override with a real
    implementation (Azure AI Content Safety, Bedrock Guardrails, regex, etc.)."""
    output_guard: OutputGuard = field(default_factory=_lazy_output_guard)
    """Validates model output before it reaches the user (Phase 3). Default:
    ``PassThroughOutputGuard`` allows everything. Hosts override with a real
    implementation. Blocked outputs are replaced with a safe fallback and
    audited via ``event_bus.publish("guardrail.audit", ...)``."""


# Re-export the grouped ``TurnHooks`` so existing ``from kernel.turn_runner
# import TurnHooks`` construction and annotation sites keep working; the
# definition now lives in ``kernel.turn_hooks``.
from kernel.turn_hooks import TurnHooks as TurnHooks  # noqa: E402
