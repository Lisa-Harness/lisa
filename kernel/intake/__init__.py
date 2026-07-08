"""Kernel intake — external data collection contracts.

This subpackage unifies the three forms of external data collection the
framework supports:

- ``connectors``: stateless, repeatable monitors for external systems
  (Cloudflare, GitHub, Stripe, etc.). Each connector returns a typed
  snapshot that can be cached in ``UnifiedState``.
- ``sources``: stateful data fetchers that may maintain rolling caches
  across execution cycles (e.g. for trend comparison).
- ``knowledge``: pack-owned domain-knowledge providers that contribute
  bounded orientation material (repo atlases, operator guidance) through
  the same intake seam.

All three share the ``DataSourceHooks`` host-injection contract from
``kernel.seams.protocols``.
"""

from kernel.intake.connectors import SignalConnector, SignalRunMode
from kernel.intake.knowledge import DomainKnowledgeProvider
from kernel.intake.sources import DataSource

__all__ = [
    "DataSource",
    "DomainKnowledgeProvider",
    "SignalConnector",
    "SignalRunMode",
]
