"""Publishing incremental tool output to the turn's operator surface.

Tools that declare ``stream_output: true`` in their ``.tool.yml`` call
:func:`publish_tool_output` while they run — one call per output line or
chunk. The kernel's turn runner subscribes to the ``lisa.tool_output``
event-bus topic for the duration of the turn and re-emits each payload as a
``tool_output`` stage event on the transport; the CLI transport renders those
in a live scrolling region, other transports decide their own treatment, and
headless contexts (no bus, or no subscriber) drop the events silently.

Contract:
    * The publish is **best-effort**: any failure is swallowed. A broken or
      absent transport must never break the tool call.
    * Payloads are operator-facing display data, NOT model context — nothing
      here is appended to ``messages``. The tool's ``ToolResult`` remains the
      only model-visible output.
    * Tools self-gate on ``ctx.stream_output`` (mirrored from the catalog
      declaration) so a tool that loses its declaration stops publishing.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from kernel.models import ToolExecutionContext

__all__ = ["TOOL_OUTPUT_TOPIC", "publish_tool_output"]

#: Event-bus topic the turn runner subscribes to for incremental tool output.
TOOL_OUTPUT_TOPIC = "lisa.tool_output"

#: Strong references to in-flight publish tasks (RUF006). Tasks are
#: fire-and-forget telemetry; the set keeps them garbage-collectable only
#: after completion and is pruned on each publish.
_PENDING_PUBLISH_TASKS: set[asyncio.Task[None]] = set()


def publish_tool_output(
    ctx: ToolExecutionContext,
    *,
    text: str,
    stream: str = "stdout",
    seq: int | None = None,
) -> None:
    """Publish one incremental operator-visible output line, best-effort.

    Args:
        ctx: The executing tool's context. Reads ``ctx.event_bus`` (the
            publisher) and ``ctx.stream_output`` (the opt-in gate).
        text: The raw output fragment (usually one line, possibly partial).
        stream: Which logical stream the fragment came from — ``stdout`` or
            ``stderr``. Transports may style them differently.
        seq: Monotonic per-call sequence number. ``None`` lets the publisher
            leave ordering to arrival order.
    """
    if not text:
        return
    if not getattr(ctx, "stream_output", False):
        return
    bus = getattr(ctx, "event_bus", None)
    if bus is None or not hasattr(bus, "publish"):
        return
    payload: dict[str, Any] = {
        "tool_id": ctx.tool_id,
        "text": text,
        "stream": stream,
    }
    if seq is not None:
        payload["seq"] = seq
    # Fire-and-forget: real tools call this inside the running loop, so
    # ``create_task`` schedules the publish without blocking execution. When
    # no loop is running (sync test contexts, non-async callers) the publish
    # is dropped rather than blocking — this helper is strictly best-effort
    # telemetry.
    with contextlib.suppress(Exception):
        task = asyncio.create_task(bus.publish(TOOL_OUTPUT_TOPIC, payload))
        _PENDING_PUBLISH_TASKS.add(task)
        task.add_done_callback(_PENDING_PUBLISH_TASKS.discard)