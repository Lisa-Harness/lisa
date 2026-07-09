"""Public default implementations of the kernel context seams.

These modules replace the private ``app.*`` dependencies that previously
lived under ``kernel/context/``. Hosts inject them through
``TurnHooks.context_seams``; the production app may override any of them
with its own implementation.

All content access is mediated through the ``lisa-content`` MCP server.
The kernel never touches the filesystem directly for content loading.
"""

from kernel.context.data_sanitizer import PassThroughSanitizer
from kernel.context.event_bus import InProcessEventBus
from kernel.context.extension_evidence import NoOpExtensionEvidenceBuilder
from kernel.context.input_guard import PassThroughInputGuard
from kernel.context.mcp_catalog import MCPModeCatalog, MCPOperationCatalog
from kernel.context.mcp_content_registry import MCPContentRegistry
from kernel.context.mcp_guard import MCPInputGuard, MCPOutputGuard
from kernel.context.mcp_storage import MCPBlobIO
from kernel.context.model_policy import ManifestModelPolicy
from kernel.context.output_guard import PassThroughOutputGuard
from kernel.context.provider_gateway_provider import KernelProviderGatewayProvider

__all__ = [
    "InProcessEventBus",
    "KernelProviderGatewayProvider",
    "MCPBlobIO",
    "MCPContentRegistry",
    "MCPInputGuard",
    "MCPModeCatalog",
    "MCPOperationCatalog",
    "MCPOutputGuard",
    "ManifestModelPolicy",
    "NoOpExtensionEvidenceBuilder",
    "PassThroughInputGuard",
    "PassThroughOutputGuard",
    "PassThroughSanitizer",
]
