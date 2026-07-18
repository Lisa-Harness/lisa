"""Introspect pack for Lisa.

Owns the operational self-awareness operations that were previously bundled
into the core pack:

- ``self_model`` — a deterministic producer that walks the wired runtime
  catalogs (modes, operations, tools, models) and emits a
  ``LisaSelfModelArtifact`` with entries across all seven
  ``SelfKnowledgeCategory`` buckets. Invoked by the post-turn slice, not by
  the LLM; no model call, runs entirely within the slice budget.
- ``reflect_self`` — a declarative operation that forms a holistic
  self-assessment from the deterministic self-model, live introspection
  telemetry, and recent turn context. Available as an allowed operation on
  modes where the model may benefit from self-introspection during
  problem-solving.

The kernel owns the contracts (``LisaSelfModelArtifact``,
``OperationExecutionContext``, the operation registry mechanism); this pack
owns the behavior that produces the self-model and reflective vibe.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent