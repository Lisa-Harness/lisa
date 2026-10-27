"""Introspect pack for Lisa.

Owns the operational self-awareness operations:

- ``self_model`` — a deterministic producer that walks the wired runtime
  catalogs (modes, operations, tools, models) and emits a
  ``LisaSelfModelArtifact`` with entries across all seven
  ``SelfKnowledgeCategory`` buckets. Invoked by the post-turn slice, not by
  the LLM; no model call, runs entirely within the slice budget.
- ``turn_forensics`` — reads recorded evidence about the model's own past
  turns: the thread entry, the lifecycle decisions that drove it, every
  dispatch that ran, and what it cost, joined on ``turn_index``. Ambient
  (``default_available``), read-effect, and rendered through the shared
  ``sdk.forensics`` formatter so the model's view and the operator's
  ``/forensics`` view cannot diverge.

The kernel owns the contracts (``LisaSelfModelArtifact``,
``OperationExecutionContext``, the operation registry mechanism); this pack
owns the behavior that produces the self-model and the forensic read. The
evidence reading itself lives in :mod:`sdk.forensics` because packs may not
import ``runtime``.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
