"""Production ``ExtensionEvidenceBuilder`` that walks installed extensions.

Discovers ``lisa.extensions.context_evidence`` entry points at build time and
calls each extension's ``build_context_evidence(runtime_context, state)``
function to collect :class:`~kernel.models.ContextEvidence` records from
every installed extension (Cloudflare, GitHub, Stripe, etc.).

Hosts call :func:`set_production_extension_evidence_builder` during
``setup_framework`` so the broker's catalog builder picks up extension
evidence automatically. The default :class:`NoOpExtensionEvidenceBuilder`
returns an empty list when no extensions are installed.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "EntryPointExtensionEvidenceBuilder",
    "set_production_extension_evidence_builder",
]

_ENTRY_POINT_GROUP = "lisa.extensions.context_evidence"


class EntryPointExtensionEvidenceBuilder:
    """Aggregate context evidence from all installed extensions.

    Walks the ``lisa.extensions.context_evidence`` entry-point group and
    calls each loaded callable. Each callable must accept
    ``(runtime_context, state)`` and return a ``list[ContextEvidence]``.
    Failures in one extension are logged and swallowed so a broken
    extension does not block context assembly.
    """

    def build_extension_context_evidence(
        self,
        runtime_context: Any,
        state: Any,
    ) -> list[Any]:
        """Return the combined evidence list from all installed extensions."""
        evidence: list[Any] = []
        try:
            eps = entry_points(group=_ENTRY_POINT_GROUP)
        except Exception:  # pylint: disable=broad-except
            logger.debug("No entry-point group %r found", _ENTRY_POINT_GROUP, exc_info=True)
            return evidence

        for ep in eps:
            try:
                builder_fn = ep.load()
            except Exception:  # pylint: disable=broad-except
                logger.warning("Failed to load context_evidence entry %r", ep.name, exc_info=True)
                continue

            try:
                contributed = builder_fn(runtime_context, state)
            except Exception:  # pylint: disable=broad-except
                logger.warning(
                    "Extension %r context_evidence builder raised; continuing.",
                    ep.name,
                    exc_info=True,
                )
                continue

            if contributed:
                evidence.extend(contributed)

        return evidence


def set_production_extension_evidence_builder() -> None:
    """Install the entry-point-based builder as the broker's extension evidence source."""
    from sdk.lisa_context.context_broker import set_extension_evidence_builder

    set_extension_evidence_builder(EntryPointExtensionEvidenceBuilder())
