"""Transport-neutral state provider and slice compaction.

This module owns:

- :func:`build_unified_state` — compiles a :class:`UnifiedState` from a
  dict-based state snapshot (loaded via ``--state-snapshot``) without
  touching global runtime state.
- :class:`FileBackedStateProvider` — the host's state-provider slot. The
  legacy lossy thread-digest pipeline was removed in the memory × thread
  realignment: the thread store is the lossless ground truth (manifest +
  ``thread_read_range`` address it), and the memory pack's consolidation
  slice owns distillation. Sealing (in ``host.thread``) is storage
  management, not compaction — no slice truncates history.
- :func:`_slice_budget_expired` — a private helper used by slice providers
  to honour the slice time budget.

All functions are transport-neutral: they take explicit ``thread_path`` and
``state_snapshot`` parameters rather than reading global runtime state. A
transport host wires its own ``HostRuntime`` paths and snapshot into these
functions at startup.

Dependency direction
--------------------
``state_provider.py`` ← ``main.py`` / ``turn_hooks_builder.py``  (callers import
from state_provider; state_provider never imports main or any transport
code)

Public surface
--------------
- :func:`build_unified_state`
- :class:`FileBackedStateProvider`
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.models import (
    ContextItem,
    UnifiedState,
)
from kernel.seams.protocols import ExtensionItemBuilder
from kernel.slice import SliceOutcome
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def build_unified_state(
    *,
    state_snapshot: dict[str, Any] | None,
    item_builder: ExtensionItemBuilder | None = None,
    runtime_context: Any = None,
) -> UnifiedState:
    """Return a compiled :class:`UnifiedState` for operations and the context mapper.

    Builds the state from the dict-based ``state_snapshot`` (loaded via
    ``--state-snapshot``). Intake-produced context items (signals, intel,
    alerts) are merged with hook-produced extension items
    (``item_builder.build_extension_context_items(runtime_context, state)``)
    into ``context_items`` — the single feed the session-owned
    context mapper walks to produce ranked, lane-ordered context without a
    live MCP round-trip.

    Context admission and ordering policy is not carried on the state: it
    lives in the session plan's ``context`` block (``ContextPlan``), which the
    host passes to the mapper alongside this state.

    Returns an empty :class:`UnifiedState` when ``state_snapshot`` is
    ``None`` or falsy.
    """
    compiled_at = datetime.now(UTC).isoformat()
    snapshot = state_snapshot
    if not snapshot:
        state = UnifiedState(compiled_at=compiled_at)
    else:
        state = UnifiedState(
            compiled_at=str(
                snapshot.get("compiled_at") or snapshot.get("generated_at") or compiled_at
            ),
        )
        # Intake-produced context items (signals, intel, alerts,
        # self_model) is the single feed. No arbitrary named buckets.
        state.context_items = _coerce_items(
            snapshot.get("context_items")
        )

    # Invoke the extension context-item hook so installed packs contribute
    # ContextItem records (self_model, Cloudflare, GitHub, Stripe, etc.).
    # The hook is the only coupling between packs and the host context path.
    if item_builder is not None:
        try:
            contributed = item_builder.build_extension_context_items(
                runtime_context, state
            )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "extension item builder raised; continuing with intake items only.",
                exc_info=True,
            )
            contributed = []
        if contributed:
            existing_ids = {e.item_id for e in state.context_items}
            for item in contributed:
                if not isinstance(item, ContextItem):
                    continue
                if item.item_id in existing_ids:
                    continue
                state.context_items.append(item)
                existing_ids.add(item.item_id)

    return state


def load_state_snapshot(
    *,
    explicit_path: str | None,
    data_dir: Path,
) -> dict[str, Any] | None:
    """Load the runtime state snapshot dict from disk.

    Resolution order:
    1. ``explicit_path`` (``--state-snapshot``) — must exist, else raise.
    2. ``<data_dir>/state.json`` — the default intake output. When the
       file does not exist, return ``None`` (intake has not run yet) so
       the host boots cleanly on a fresh data dir instead of crashing.

    Returns ``None`` when no snapshot is available. Raises ``ValueError``
    when an explicit path was given but the file is missing or not a JSON
    object.
    """
    configured = (explicit_path or "").strip()
    if configured:
        file_path = Path(configured).expanduser()
        source = "explicit"
    else:
        file_path = data_dir / "state.json"
        source = "default"

    if not file_path.exists():
        if source == "explicit":
            logger.error("state snapshot file not found: %s", file_path)
            raise ValueError(f"State snapshot file not found: {file_path}")
        return None

    try:
        decoded = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("state snapshot file %s is unreadable: %s", file_path, exc)
        raise ValueError(f"State snapshot file {file_path} is unreadable: {exc}") from exc

    if not isinstance(decoded, dict):
        logger.error("state snapshot file must contain a JSON object: %s", file_path)
        raise ValueError("State snapshot file must contain a JSON object.")

    logger.info("loaded state snapshot from %s (%d keys)", file_path, len(decoded))
    return decoded


def _coerce_items(raw_items: Any) -> list[ContextItem]:
    """Coerce a raw list of item dicts into a list of :class:`ContextItem`."""
    if not isinstance(raw_items, list):
        return []
    try:
        return [
            e if isinstance(e, ContextItem) else ContextItem.model_validate(e)
            for e in raw_items
        ]
    except (ValidationError, TypeError):
        return []


def _slice_budget_expired(policy: Any, started: float) -> bool:
    """Return ``True`` if the slice budget has been consumed.

    A budget of zero or negative means "no budget" (never expires).
    """
    budget_ms = getattr(policy, "budget_ms", 0) or 0
    if budget_ms <= 0:
        return False
    return (time.monotonic() - started) * 1000.0 >= budget_ms


class FileBackedStateProvider:
    """Host state-provider slot (post realignment): no-LLM, no-op slice.

    The legacy lossy thread-digest pipeline (``<base>.digest.md`` +
    ``<base>.digest.marker``) is removed. Under the memory × thread split
    the thread store is the lossless record of what happened — addressed by
    the thread manifest and ``thread_read_range`` — and cross-session
    distillation is owned by the memory pack's consolidation slice
    (``sdk/packs/memory/slice_provider.py``), which mines sealed segments
    opportunistically inside the post-turn slice budget.

    This class remains because the host slice wiring and the ``/slice``
    explicit command expect a state-provider object; it simply reports
    zero work so the pack-provided slices own all useful slice labor.
    """

    def __init__(
        self,
        *,
        thread_path: Path | None,
    ) -> None:
        # ``thread_path`` retained for signature stability with callers
        # (turn_hooks_builder, /slice command); nothing is digested.
        del thread_path

    def __call__(self) -> Any:
        return None

    async def slice(self, ctx: Any, policy: Any) -> SliceOutcome:
        """No-op slice: history is never compacted, only sealed and mined."""
        del ctx, policy
        return SliceOutcome(
            work_items_completed=0,
            elapsed_ms=0.0,
            last_item="",
            notes=("no_op_no_compaction",),
        )
