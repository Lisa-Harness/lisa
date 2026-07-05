"""Public default implementations of the kernel context seams.

These modules replace the private ``app.*`` dependencies that previously
lived under ``kernel/context/``. Hosts inject them through
``TurnHooks.context_seams``; the production app may override any of them
with its own implementation.
"""

from kernel.context.content_registry import DiskContentRegistry
from kernel.context.data_sanitizer import PassThroughSanitizer
from kernel.context.event_bus import InProcessEventBus
from kernel.context.extension_evidence import NoOpExtensionEvidenceBuilder
from kernel.context.input_guard import PassThroughInputGuard
from kernel.context.model_policy import ManifestModelPolicy
from kernel.context.output_guard import PassThroughOutputGuard
from kernel.context.provider_gateway_provider import KernelProviderGatewayProvider

__all__ = [
    "DiskContentRegistry",
    "InProcessEventBus",
    "KernelProviderGatewayProvider",
    "ManifestModelPolicy",
    "NoOpExtensionEvidenceBuilder",
    "PassThroughInputGuard",
    "PassThroughOutputGuard",
    "PassThroughSanitizer",
]