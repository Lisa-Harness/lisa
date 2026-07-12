"""Checkpoint pack for Lisa.

Provides two runtime operations that enrich and persist checkpoints via
the ``lisa-storage`` MCP server:

- ``checkpoint_create`` — calls the model to summarize the conversation,
  then saves an enriched ``CheckpointBundle`` (overview, work_done,
  important_files, next_steps) to ``lisa-storage``.
- ``checkpoint_restore`` — loads a checkpoint by ID, returns the full
  bundle (including turn history and enrichment) for the host to restore.

The kernel provides the mechanism (``CheckpointBundle``, ``on_checkpoint``
hook, ``lisa-storage`` tools); this pack owns the behavior (LLM
summarization, enrichment, formatting).
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent