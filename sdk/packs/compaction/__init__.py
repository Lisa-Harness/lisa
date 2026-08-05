"""Compaction pack for Lisa.

Provides utilization-triggered, pressure-driven context compaction as a
composable pack. The pack contributes three things:

- A **slice provider** (``slice_provider.py``) registered via the new
  ``lisa.extensions.slice_providers`` entry-point group. Each turn, after
  the host's context-mapper pass, the provider reads the live
  ``ContextBudgetSnapshot`` from ``TurnHooks.last_context_snapshot``,
  computes context-window utilization, and fires the configured
  ``summarize_operation`` (default: ``checkpoint_history_summarize``)
  when utilization crosses ``CompactionTuning.pressure_threshold`` — or
  unconditionally when it crosses ``hard_threshold``. The compaction runs
  in the post-turn slice, paid for out of the existing pressure budget
  in ``kernel.slice``, not the user's turn latency.

- A **``context_compact`` operation** that wraps the configured
  ``summarize_operation`` with utilization-aware framing. The pack
  defaults to reusing ``checkpoint_history_summarize`` rather than
  forking it — one compaction primitive, two triggers (manual
  ``/summarize-history`` and automatic via this pack).

- Two **slash commands** (``/compact-now``, ``/compaction-status``) for
  explicit invocation and observability.

Configuration lives in ``session_context`` under
``context.history.compaction`` (see ``kernel.session_plan.CompactionTuning``).
Default is ``enabled=False``; a host opts in by setting
``enabled: true`` in its base session.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
