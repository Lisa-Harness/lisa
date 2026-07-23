"""Kernel intake — external data collection contracts.

This subpackage defines the bottom-half contracts for passive data
acquisition:

- ``connectors``: stateless, repeatable monitors for external systems
  (Cloudflare, GitHub, Stripe, etc.). Each collector returns a typed
  snapshot that can be cached in ``UnifiedState``.
- ``contracts``: the minimal ``ProcessingLoop`` protocol for background
  processing chains that run after collection and write results back
  into ``UnifiedState``.

The bottom half is driven by a one-shot process (``little-lisa bottom
--tick``) triggered by a systemd timer or cron — no long-running threads.
The top half (turn loop) reads the resulting ``UnifiedState`` via the
host's ``_state_provider()``.
"""

from kernel.intake.connectors import SignalCollector, SignalRunMode
from kernel.intake.contracts import ProcessContext, ProcessedState, ProcessingLoop

__all__ = [
    "ProcessContext",
    "ProcessedState",
    "ProcessingLoop",
    "SignalCollector",
    "SignalRunMode",
]
