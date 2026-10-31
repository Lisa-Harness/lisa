"""Spill-on-arrival: the cache-aware default for large tool results.

The mid-turn governor's epoch (``kernel.turn_phases._compaction``) is the
FALLBACK. This module is the workhorse: a large tool result is ref'd out
at DELIVERY time, before it ever enters the live provider message array,
so the array stays append-only and normal dispatches extend the cached
prefix instead of rewriting it. Because the content never entered the
cached prefix, spilling it later busts nothing.

What lands inline is a bounded structured preview:

- the first excerpt of the result,
- a dereferenceable ``r<sha256>`` live ref produced by the host-supplied
  ``persist_tool_result`` closure (``ToolHooks``), which must use the
  same record shape and store as the host's result projector so the
  preview's ref dereferences via the thread tools AND via the
  projector's own ref-resolution path,
- a closing inventory of up to ``preview_repeat_limit`` distinct
  high-frequency lines, each collapsed to one occurrence (the turn-1116
  edit-batch validator failures each dumped nearly identical stack
  traces; the collapse keeps the signal, drops the repetition).

The full text is persisted BEFORE preview construction, so recovery is a
plain dereference: ``thread_read_content`` with the ref. One-way ref-out
is preserved — spilling replaces the inline body; nothing here re-inlines
content into the array (eviction is one-way by construction).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

#: Minimum body size (bytes) worth spilling. Below this the result stays
#: inline: a preview plus dereference round-trip is not worth its cost for
#: content the model reads in one glance.
SPILL_MIN_BYTES = 8192

_EXCERPT_CHARS = 400

#: A bare ``r<sha256>`` ref is ``r`` + 64 hex chars.
_REF_CHARS = 65

#: Minimum stripped-line length counted for the repeated-lines inventory.
_REPEAT_MIN_CHARS = 16


def _is_ref(content: str) -> bool:
    """True when the content already is a bare ``r<sha256>`` dereferenceable ref."""
    stripped = content.strip()
    return stripped.startswith("r") and len(stripped) == _REF_CHARS and stripped[1:].isalnum()


def build_spill_preview(
    content: str,
    *,
    persist: Callable[[dict[str, Any]], str],
    task_id: str,
    tool_name: str,
    preview_repeat_limit: int,
) -> str:
    """Persist ``content`` and return the bounded preview that replaces it inline.

    The payload is persisted with the host-supplied ``persist`` closure —
    same record shape the host projector persists, so the returned ref is
    resolvable by both the projector's loader and the thread tools. A
    persist failure propagates: the caller keeps the full text inline
    rather than replacing it with a ref that would not dereference.
    """
    ref = persist({"content": content, "task_id": task_id, "tool_name": tool_name})
    lines = content.splitlines()
    head = "\n".join(lines[:10])[:_EXCERPT_CHARS]
    repeats = Counter(line.strip() for line in lines if len(line.strip()) >= _REPEAT_MIN_CHARS)
    inventory = [line for line, count in repeats.most_common() if count >= preview_repeat_limit][
        :preview_repeat_limit
    ]
    if not inventory:
        body = head
    else:
        body = "\n".join([head, "", "repeated lines (collapsed; full text via ref):", *inventory])
    return (
        f"[spilled tool result ({len(content)} chars) — dereference {ref} "
        f"via thread tools for the full text]\n{body}"
    )


def maybe_spill_tool_result(
    *,
    result_message: dict[str, Any],
    persist: Callable[[dict[str, Any]], str] | None,
    task_id: str,
    preview_repeat_limit: int,
) -> bool:
    """Spill one freshly-dispatched tool result in place when it is large.

    Mutates ``result_message["content"]`` only when a spill happened.
    Returns True when the content was replaced by a preview. Already-ref'd
    content passes through untouched (one-way ref-out: never double-persist
    a dereferenceable result).
    """
    if persist is None:
        return False
    content = result_message.get("content")
    if not isinstance(content, str) or not content:
        return False
    if _is_ref(content):
        return False
    if len(content.encode("utf-8")) < SPILL_MIN_BYTES:
        return False
    result_message["content"] = build_spill_preview(
        content,
        persist=persist,
        task_id=task_id,
        tool_name=str(result_message.get("name") or ""),
        preview_repeat_limit=preview_repeat_limit,
    )
    return True
