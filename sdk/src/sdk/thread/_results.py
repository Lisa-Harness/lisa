"""Stable result-object references and provider-only tool-output projection."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sdk.thread.redaction import ThreadRedactionPolicy

_RESULT_REFERENCE = re.compile(r"r[a-f0-9]{64}")
_THREAD_REFERENCE = re.compile(r"t\d{1,20}(?:\.m\d{1,20})?(?:\.c\d{1,20})?(?:\.[a-z_]+)?$")
_INLINE_RESULT_BYTES = 8192
_MAX_RETRIEVAL_BYTES = 32768
_TOOL_CONTEXT_BYTES = 65536
_INVENTORY_RESERVE_BYTES = 2048
_HEAD_CHARS = 600
_TAIL_CHARS = 300
_PRESSURE_HEAD_CHARS = 120
_PRESSURE_TAIL_CHARS = 60
_PRESSURE_SPAN_BYTES = 96 * 1024
_RETRIEVAL_TOOLS = frozenset(
    {
        "thread_read_range",
        "thread_read_recent",
        "thread_search",
        "thread_read_content",
        "thread_find_content",
    }
)


def result_object_path(base: Path, ref: str) -> Path:
    if _RESULT_REFERENCE.fullmatch(ref) is None:
        raise ValueError("Invalid result-object reference")
    directory = Path(f"{base}.results")
    path = directory / f"{ref}.json"
    if directory.is_symlink() or path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError("Result-object reference escapes thread storage")
    return path


def load_result_object(base: Path, ref: str) -> dict[str, Any]:
    path = result_object_path(base, ref)
    try:
        payload = path.read_bytes()
        result = json.loads(payload)
    except (OSError, ValueError) as exc:
        raise ValueError("Result-object reference not found or unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != ref[1:]:
        raise ValueError("Result-object integrity check failed")
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise ValueError("Invalid result-object content")
    return result


def _tool_result_names(messages: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[str, str] = {}
    result_names: dict[int, str] = {}
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            function = call.get("function") or call
            names[str(call.get("id") or "")] = str(function.get("name") or "")
        if message.get("role") == "tool":
            result_names[index] = str(
                message.get("name") or names.get(str(message.get("tool_call_id") or ""), "")
            )
    return result_names


def project_tool_messages(
    messages: list[dict[str, Any]],
    task_id: str,
    persist: Callable[[dict[str, Any]], str],
    redaction: ThreadRedactionPolicy | None = None,
    load: Callable[[str], dict[str, Any]] | None = None,
    *,
    budget_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Abbreviate large tool text without modifying raw messages or tool-call pairing.

    ``budget_bytes`` bounds the total tool-result bytes that may remain
    resident in this projection; ``None`` (default) keeps the module
    ``_TOOL_CONTEXT_BYTES`` ceiling. A lower budget is how the turn-start
    replay allocator recomputes the resident working set — oldest results
    collapse into the evidence inventory, newest stay verbatim.
    """
    budget = _TOOL_CONTEXT_BYTES if budget_bytes is None else max(0, budget_bytes)
    projected = copy.deepcopy(messages)
    tool_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    ]
    result_names = _tool_result_names(messages)
    retrieval_indices = {
        index for index, name in result_names.items() if name.rsplit(".", 1)[-1] in _RETRIEVAL_TOOLS
    }

    # Recall bytes do not shrink work previews, but still count toward the
    # absolute projection budget. Older recall pages must yield room for the
    # latest page instead of accumulating unboundedly.
    tool_bytes = sum(
        len(str(messages[index].get("content") or "").encode("utf-8"))
        for index in tool_indices
        if index not in retrieval_indices
    )
    total_bytes = sum(
        len(str(messages[index].get("content") or "").encode("utf-8")) for index in tool_indices
    )
    pressure = tool_bytes > budget
    over_budget = total_bytes > budget
    latest_retrieval = max(retrieval_indices, default=-1)
    scale = _preview_scale(tool_bytes, budget) if pressure else (_HEAD_CHARS, _TAIL_CHARS)
    refs: dict[int, str] = {}
    # Inline results register lazy persisters instead of eager refs: the
    # absolute-bound path may still have to evict them to the inventory.
    inline_refs: dict[int, Callable[[], str]] = {}
    for index, message in enumerate(projected):
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str):
            continue
        name = result_names[index]
        retrieval_page = index in retrieval_indices
        existing = _existing_result(content, load, redaction)
        if existing is not None:
            refs[index] = existing[0]
            if not pressure and (
                not over_budget or not retrieval_page or index == latest_retrieval
            ):
                continue
            content = existing[1]["content"]
        permitted = redaction.redact(content) if redaction is not None else content
        pointer = (
            _retrieval_pointer(content, permitted)
            if (pressure or over_budget) and retrieval_page and existing is None
            else None
        )
        if pointer is not None:
            # A page of an already-stored value re-expands the original ref;
            # persisting the page again would wrap the ref in another ref.
            refs[index] = pointer["ref"]
            message["content"] = json.dumps(pointer, ensure_ascii=False, separators=(",", ":"))
            continue
        if (
            retrieval_page
            and len(content.encode("utf-8")) <= _MAX_RETRIEVAL_BYTES
            and (not over_budget or index == latest_retrieval)
        ):
            message["content"] = permitted
            continue
        if (
            len(content.encode("utf-8")) <= _INLINE_RESULT_BYTES
            and (not pressure or len(permitted) <= scale[0] + scale[1])
            and (not retrieval_page or not over_budget or index == latest_retrieval)
        ):
            # A result that fits the active head+tail scale has nothing to
            # omit: abbreviating it would replace content with a strictly
            # larger ref envelope (the 2026-10-06 turn-557/558 micro-window
            # loop, where zero-omission previews inflated the context they
            # bound). Keep it inline. ``_compact_groups`` can still evict it
            # when over budget, persisting the raw content on demand, so the
            # absolute bound and the lossless archive both hold.

            def _persist_inline(bound_content: str = content, bound_name: str = name) -> str:
                return persist(
                    {
                        "content": bound_content,
                        "task_id": task_id,
                        "tool_name": bound_name,
                    }
                )

            inline_refs[index] = _persist_inline
            message["content"] = permitted
            continue
        ref = (
            existing[0]
            if existing
            else persist({"content": content, "task_id": task_id, "tool_name": name})
        )
        refs[index] = ref
        message["content"] = _preview(ref, permitted, *scale)
    refs.update({index: make_ref() for index, make_ref in inline_refs.items()})
    return (
        _compact_groups(messages, projected, refs, task_id, persist, budget=budget)
        if pressure or over_budget
        else projected
    )


def _retrieval_pointer(content: str, permitted: str) -> dict[str, Any] | None:
    """Return a pointer to an already-stored value when content is its page.

    Retrieval tools return envelopes like ``{ref, content, offset_chars, …}``.
    When the envelope's ``ref`` addresses an existing stored object or thread
    entry, the page adds nothing the model cannot recover by expanding that
    ref, so projection emits a compact pointer instead of persisting a copy.
    """

    if not content.startswith("{") or '"ref"' not in content:
        return None
    try:
        envelope = json.loads(content)
    except ValueError:
        return None
    if not isinstance(envelope, dict):
        return None
    ref = envelope.get("ref")
    # A mid-value page carries progress the original ref cannot reconstruct.
    if not isinstance(ref, str) or envelope.get("has_more") is True:
        return None
    if _RESULT_REFERENCE.fullmatch(ref) or _THREAD_REFERENCE.match(ref):
        return _page_pointer(ref, permitted)
    return None


def _page_pointer(ref: str, permitted: str) -> dict[str, Any]:
    stored: dict[str, Any] = {"ref": ref, "page_omitted": True, "expand": "thread_read_content"}
    excerpt = permitted[:120]
    if excerpt:
        stored["head"] = excerpt
    return stored


def _existing_pointer(
    preview: dict[str, Any],
    load: Callable[[str], dict[str, Any]],
    redaction: ThreadRedactionPolicy | None,
) -> tuple[str, dict[str, Any]] | None:
    ref = preview["ref"]
    if _RESULT_REFERENCE.fullmatch(ref) is None:
        return None
    result = load(ref)
    permitted = redaction.redact(result["content"]) if redaction else result["content"]
    head = preview.get("head", "")
    if head and not permitted.startswith(head):
        return None
    return ref, result


def _preview_scale(tool_bytes: int, budget: int = _TOOL_CONTEXT_BYTES) -> tuple[int, int]:
    """Map total tool-result bytes to graduated (head, tail) preview sizes.

    Sizes shrink linearly from (_HEAD_CHARS, _TAIL_CHARS) at the pressure
    threshold to (_PRESSURE_HEAD_CHARS, _PRESSURE_TAIL_CHARS) at span end.
    Below the threshold the scale is the full values exactly; at and beyond
    the span end the scale is the floor values exactly. The continuous
    ramp removes the collapse cliff: a small overage costs proportionally
    little preview fidelity instead of 83% of it in one step (the
    2026-10-05 turn-524 thrash, where every re-expansion re-triggered
    full collapse).
    """
    over = max(0, tool_bytes - budget)
    span = max(1, budget // 2) if budget < _TOOL_CONTEXT_BYTES else _PRESSURE_SPAN_BYTES
    fraction = min(1.0, over / span)
    head = int(_HEAD_CHARS - (_HEAD_CHARS - _PRESSURE_HEAD_CHARS) * fraction)
    tail = int(_TAIL_CHARS - (_TAIL_CHARS - _PRESSURE_TAIL_CHARS) * fraction)
    return head, tail


def _preview(ref: str, content: str, head_chars: int, tail_chars: int) -> str:
    head_chars = min(head_chars, len(content))
    tail_chars = min(tail_chars, len(content) - head_chars)
    return json.dumps(
        {
            "ref": ref,
            "abbreviated": True,
            "total_chars": len(content),
            "head": content[:head_chars],
            "tail": content[-tail_chars:] if tail_chars else "",
            "omitted_chars": len(content) - head_chars - tail_chars,
            "expand": "thread_read_content",
            "find": "thread_find_content",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _existing_result(
    content: str,
    load: Callable[[str], dict[str, Any]] | None,
    redaction: ThreadRedactionPolicy | None,
) -> tuple[str, dict[str, Any]] | None:
    if load is None or not content.startswith('{"'):
        return None
    try:
        preview = json.loads(content)
        if (
            isinstance(preview, dict)
            and preview.get("page_omitted") is True
            and isinstance(preview.get("ref"), str)
        ):
            return _existing_pointer(preview, load, redaction)
        ref = preview["ref"]
        if preview.get("abbreviated") is not True or _RESULT_REFERENCE.fullmatch(ref) is None:
            return None
        result = load(ref)
        permitted = redaction.redact(result["content"]) if redaction else result["content"]
        if preview not in (
            json.loads(_preview(ref, permitted, *_preview_scale(0))),
            json.loads(_preview(ref, permitted, _PRESSURE_HEAD_CHARS, _PRESSURE_TAIL_CHARS)),
        ):
            return None
        return ref, result
    except (KeyError, TypeError, ValueError):
        return None


def _complete_groups(messages: list[dict[str, Any]]) -> list[tuple[int, tuple[int, ...]]]:
    outputs = {
        str(message.get("tool_call_id")): index
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
    }
    groups = []
    for index, message in enumerate(messages):
        ids = [str(call.get("id") or "") for call in message.get("tool_calls") or []]
        if ids and all(call_id in outputs and outputs[call_id] > index for call_id in ids):
            groups.append((index, tuple(outputs[call_id] for call_id in ids)))
    return groups


def _compact_groups(
    raw: list[dict[str, Any]],
    projected: list[dict[str, Any]],
    refs: dict[int, str],
    task_id: str,
    persist: Callable[[dict[str, Any]], str],
    *,
    budget: int = _TOOL_CONTEXT_BYTES,
) -> list[dict[str, Any]]:
    remaining = sum(
        len(str(message.get("content") or "").encode("utf-8"))
        for message in projected
        if message.get("role") == "tool"
    )
    omitted: set[int] = set()
    inventory: list[dict[str, Any]] = []
    reserve = min(_INVENTORY_RESERVE_BYTES, max(0, budget // 16))
    for call_index, result_indices in _complete_groups(projected):
        if remaining <= budget - reserve:
            break
        if not all(index in refs for index in result_indices):
            continue
        inventory.append(
            {
                "tool_calls": raw[call_index].get("tool_calls") or [],
                "results": [
                    {"ref": refs[index], "tool_call_id": raw[index].get("tool_call_id")}
                    for index in result_indices
                ],
            }
        )
        omitted.update((call_index, *result_indices))
        remaining -= sum(
            len(str(projected[index].get("content") or "").encode("utf-8"))
            for index in result_indices
        )
    if not inventory:
        return projected
    ref = persist(
        {
            "content": json.dumps(inventory, ensure_ascii=False, separators=(",", ":")),
            "task_id": task_id,
            "tool_name": "thread_evidence_inventory",
        }
    )
    marker = {
        "role": "assistant",
        "content": json.dumps(
            {
                "abbreviated_tool_groups": len(inventory),
                "inventory_ref": ref,
                "expand": "thread_read_content",
                "find": "thread_find_content",
            },
            separators=(",", ":"),
        ),
    }
    first_omitted = min(omitted)
    return [
        marker if index == first_omitted else message
        for index, message in enumerate(projected)
        if index == first_omitted or index not in omitted
    ]
