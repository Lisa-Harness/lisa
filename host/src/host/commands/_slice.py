"""``/consolidate`` (``/slice``) and ``/intake`` command handlers.

These run host-side side-effectful work outside the model turn: the explicit
slice and the one-shot intake tick. Both set turn context and
append thread entries to keep the local thread consistent with model turns.

Dependency direction: ``_slice`` → ``_log``, ``host.state_provider``,
``host.intake`` (lazy). No imports from ``_render``/``_packs``/``_lists``/
``_models``/``host.lisa_config``.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import UTC, datetime

from kernel.slice import SliceMode, SliceOutcome, SlicePolicy

from host.commands._log import (
    _append_thread_entry,
    _append_turn_log,
    _set_turn_context,
)
from host.runtime import HostRuntime
from host.state_provider import FileBackedStateProvider

logger = logging.getLogger(__name__)


# ── /consolidate (/slice) ──────────────────────────────────────────


async def run_explicit_slice_command(
    runtime: HostRuntime,
    *,
    message: str,
) -> str:
    """Run the explicit-mode slice as a user-invoked command.

    This is the escape hatch for hosts whose opportunistic pre/post
    slices are disabled (or whose state provider has no work to do).
    Calling ``/consolidate`` (or ``/slice``) invokes the same slice
    machinery the kernel uses, but with a one-shot user-controlled
    budget resolved once at host startup from the ``LISA_SLICE_BUDGET_MS``
    env var into ``HostConfig.slice_config.explicit_budget_ms``.
    """
    explicit_budget_ms = runtime.config.slice_config.explicit_budget_ms
    provider = FileBackedStateProvider(
        thread_path=runtime.paths.thread_path,
    )
    policy = SlicePolicy(
        mode=SliceMode.EXPLICIT,
        budget_ms=explicit_budget_ms,
    )
    started = _time.monotonic()
    outcome: SliceOutcome = await provider.slice(ctx=None, policy=policy)
    elapsed_ms = (_time.monotonic() - started) * 1000.0
    runtime.session.last_slice_finished_at = _time.monotonic()
    timestamp_utc = datetime.now(UTC).isoformat()
    slice_data = {
        "budget_ms": explicit_budget_ms,
        "provider": type(provider).__name__,
        "work_items_completed": outcome.work_items_completed,
        "last_item": outcome.last_item,
        "notes": list(outcome.notes),
        "elapsed_ms": round(elapsed_ms, 3),
        "policy_budget_ms": policy.budget_ms,
    }
    _set_turn_context(
        runtime,
        timestamp_utc=timestamp_utc,
        raw_message=message,
        mode_id="consolidate",
        handoff="explicit slice command",
        turn_input={"explicit_slice": slice_data},
    )
    _append_turn_log(
        context={
            "timestamp_utc": timestamp_utc,
            "raw_message": message,
            "mode_id": "consolidate",
            "handoff": "explicit slice command",
            "explicit_slice": slice_data,
        },
        response_text="",
    )
    _append_thread_entry(
        runtime,
        mode_id="consolidate",
        query=message,
        response_text="",
    )
    runtime.session.current_mode_id = "consolidate"
    runtime.session.current_stage_id = "-"
    runtime.session.next_model_id = "-"
    if outcome.work_items_completed == 0:
        return (
            f"Slice ran in {elapsed_ms:.1f} ms with no work items. "
            "Thread and memory are within their caps; nothing to compact."
        )
    note_text = "; ".join(outcome.notes) if outcome.notes else "ok"
    return (
        f"Slice drained {outcome.work_items_completed} work item(s) in "
        f"{elapsed_ms:.1f} ms (budget {policy.budget_ms} ms). "
        f"Last item: {outcome.last_item or '<none>'}. Notes: {note_text}."
    )


# ── /intake ──────────────────────────────────────────────────────────


async def handle_intake_command(
    runtime: HostRuntime,
    message: str,
) -> str:
    """Handle ``/intake`` — run one intake tick in-process.

    The CLI exposes this as a one-shot ``lisa intake`` process. The slash
    command runs the same :func:`host.intake.run_intake` inside the
    host's live event loop so the running REPL/VsCode/web session can refresh
    ambient state (GitHub PR status, Stripe billing health, Cloudflare zone
    state, etc.) on demand. The next model turn reads the written
    ``state.json`` via the host's state provider.
    """
    try:
        from host.intake import run_intake
    except ImportError:
        return "intake module not available in this host."

    started = _time.monotonic()
    try:
        state = await run_intake(data_dir=runtime.paths.data_dir)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Intake tick failed: %s", exc)
        return f"Intake tick failed: {exc}"
    elapsed_ms = (_time.monotonic() - started) * 1000.0

    timestamp_utc = datetime.now(UTC).isoformat()
    intake_data = {
        "elapsed_ms": round(elapsed_ms, 3),
        "item_count": len(state.context_items),
    }
    _set_turn_context(
        runtime,
        timestamp_utc=timestamp_utc,
        raw_message=message,
        mode_id="intake",
        handoff="intake tick command",
        turn_input={"intake": intake_data},
    )
    _append_turn_log(
        context={
            "timestamp_utc": timestamp_utc,
            "raw_message": message,
            "mode_id": "intake",
            "handoff": "intake tick command",
            "intake": intake_data,
        },
        response_text="",
    )
    _append_thread_entry(
        runtime,
        mode_id="intake",
        query=message,
        response_text="",
    )
    runtime.session.current_mode_id = "intake"
    runtime.session.current_stage_id = "-"
    runtime.session.next_model_id = "-"

    from sdk.layout import STATE_DIR

    state_path = runtime.paths.data_dir / STATE_DIR / "state.json"
    return (
        f"Intake tick complete in {elapsed_ms:.1f} ms. "
        f"{len(state.context_items)} context item(s). "
        f"State written to {state_path}"
    )
