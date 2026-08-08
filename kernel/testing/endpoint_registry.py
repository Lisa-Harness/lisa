"""Declarative registry of every public extension point in the framework.

Each entry asserts: (1) the contract exists, (2) at least one pack produces
it, (3) at least one host/kernel site consumes it. The test in
``test_endpoint_registry.py`` fails when any endpoint is orphaned.

This file is the documentation proxy for external consumers. A pack author
reads this file to learn every available extension point and its required
shape. Because external consumers can't be surveyed, the registry + test
proves every contract has a producer and consumer in the reference
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Endpoint:
    """One public extension point in the framework.

    Attributes:
        group: The entry-point group name (e.g. ``lisa.extensions.signal_connectors``).
        contract_module: The module path where the Protocol/ABC lives.
        contract_name: The Protocol/ABC class name.
        consumer_site: The ``module.attr`` that loads this group at runtime.
        description: What this endpoint does, for pack authors.
    """

    group: str
    contract_module: str
    contract_name: str
    consumer_site: str
    description: str


REGISTRY: list[Endpoint] = [
    Endpoint(
        group="lisa.extensions.content_roots",
        contract_module="pathlib",
        contract_name="Path",
        consumer_site="kernel.content_roots.discover_installed_content_roots",
        description=(
            "Declares a pack's filesystem root for mode, operation, and "
            "content discovery. The return value is a ``Path`` to the pack "
            "root directory."
        ),
    ),
    Endpoint(
        group="lisa.extensions.signal_connectors",
        contract_module="kernel.intake.connectors",
        contract_name="SignalCollector",
        consumer_site="kernel.intake.connector_discovery.discover_installed_signal_collectors",
        description=(
            "Stateless external monitor (e.g. Cloudflare, GitHub, Stripe). "
            "Each collector returns a typed Pydantic snapshot that the "
            "bottom-half tick caches in ``UnifiedState.signal_snapshots``."
        ),
    ),
    Endpoint(
        group="lisa.extensions.context_evidence",
        contract_module="kernel.extension_runtime",
        contract_name="ExtensionContextEvidenceBuilder",
        consumer_site="sdk.context.extension_evidence_builder.EntryPointExtensionEvidenceBuilder",
        description=(
            "Contributes ranked context evidence from installed extensions. "
            "Each callable accepts ``(runtime_context, state)`` and returns "
            "a ``list[ContextEvidence]``. The session-owned context mapper "
            "merges these into ranked, lane-ordered context for each turn."
        ),
    ),
    Endpoint(
        group="lisa.extensions.host_commands",
        contract_module="kernel.host_commands",
        contract_name="HostCommandHandler",
        consumer_site="kernel.host_command_discovery.discover_pack_host_commands",
        description=(
            "Registers pack-owned slash commands (e.g. /checkpoint, /restore). "
            "Each entry point returns a list[HostCommandSpec]. The host "
            "dispatch layer consults discovered commands after built-in "
            "host commands."
        ),
    ),
    Endpoint(
        group="lisa.extensions.processing_loops",
        contract_module="kernel.intake.contracts",
        contract_name="ProcessingLoop",
        consumer_site="host.intake._discover_processing_loops",
        description=(
            "Background processing chain triggered after signal collection. "
            "Each loop takes ``ProcessContext`` (with collection results) and "
            "returns ``ProcessedState`` that the bottom-half tick merges into "
            "``UnifiedState``. Common uses: anomaly detection, trend analysis, "
            "state derivation."
        ),
    ),
]


__all__ = ["REGISTRY", "Endpoint"]
