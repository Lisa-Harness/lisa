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
_TOOL_CONTEXT_BYTES = 65536
_INVENTORY_RESERVE_BYTES = 2048
_HEAD_CHARS = 600
_TAIL_CHARS = 300
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
    """Fit tool text into a resident budget without modifying raw messages or pairing.

    Results are allocated newest-first against ``budget_bytes``, which the
    kernel sizes from the active model's context window (``None`` keeps the
    module ``_TOOL_CONTEXT_BYTES`` ceiling). A result stays verbatim when it
    fits the bytes not yet allocated to newer results, so the evidence the
    model just requested is always readable. There is no per-result cap: a
    fixed 8KB one previewed every mid-sized source file while most of the
    budget sat free, and the model re-read the same files until the
    iteration cap (2026-10-09 turn-622). A result that does not fit is abbreviated: a complete
    page of an already-stored value pointers to that value, anything else
    is persisted and previewed. Oldest groups then collapse into the
    evidence inventory until the absolute bound holds.

    Residency depends only on what stays resident, never on how much raw
    history the turn accumulated: a sticky turn-wide "pressure" flag once
    previewed every new read and pointered every expansion back to its own
    ref after eviction had already freed the budget (2026-10-08 turn-618).
    """
    budget = _TOOL_CONTEXT_BYTES if budget_bytes is None else max(0, budget_bytes)
    reserve = min(_INVENTORY_RESERVE_BYTES, max(0, budget // 16))
    available = budget - reserve
    projected = copy.deepcopy(messages)
    result_names = _tool_result_names(messages)
    # Refs resolve lazily: verbatim results persist only if the inventory
    # actually evicts them.
    refs: dict[int, Callable[[], str]] = {}
    for index in sorted(result_names, reverse=True):
        message = projected[index]
        content = message.get("content")
        if not isinstance(content, str):
            continue
        name = result_names[index]
        retrieval_page = name.rsplit(".", 1)[-1] in _RETRIEVAL_TOOLS
        existing = _existing_result(content, load, redaction)
        if existing is not None:
            refs[index] = _known_ref(existing[0])
            available -= len(content.encode("utf-8"))
            continue
        permitted = redaction.redact(content) if redaction is not None else content
        size = len(permitted.encode("utf-8"))
        # A complete page (has_more false) of an already-stored value never
        # persists a copy: its ref reconstructs the bytes, so a second
        # address would be a stub-of-a-page (Lisa-Harness/lisa#5).
        pointer = _retrieval_pointer(content, permitted) if retrieval_page else None
        # A result no larger than a preview's excerpts stays inline even
        # without room: abbreviating it would inflate the wire (the
        # 2026-10-06 micro-window loop). The inventory can still evict it.
        if size <= available or len(permitted) <= _HEAD_CHARS + _TAIL_CHARS:
            message["content"] = permitted
            available -= size
            refs[index] = (
                _known_ref(pointer["ref"])
                if pointer is not None
                else _lazy_persist(persist, content, task_id, name)
            )
            continue
        if pointer is not None:
            ref = pointer["ref"]
            message["content"] = json.dumps(pointer, ensure_ascii=False, separators=(",", ":"))
        else:
            ref = persist({"content": content, "task_id": task_id, "tool_name": name})
            message["content"] = _preview(ref, permitted)
        refs[index] = _known_ref(ref)
        available -= len(message["content"].encode("utf-8"))
    if available + reserve >= 0:
        return projected
    return _compact_groups(messages, projected, refs, task_id, persist, budget=budget)


def _known_ref(ref: str) -> Callable[[], str]:
    return lambda: ref


def _lazy_persist(
    persist: Callable[[dict[str, Any]], str], content: str, task_id: str, name: str
) -> Callable[[], str]:
    return lambda: persist({"content": content, "task_id": task_id, "tool_name": name})


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


def _preview(
    ref: str, content: str, head_chars: int = _HEAD_CHARS, tail_chars: int = _TAIL_CHARS
) -> str:
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
        # Persisted threads hold stubs minted at older graduated scales
        # (e.g. head 431). Recognizing only the current 600/300 scale would
        # re-persist such a stub as fresh content — a stub-of-a-stub whose
        # ref expands to preview JSON, not original bytes (2026-10-07
        # replay-eviction finding). Rebuild the preview at the stub's own
        # head/tail and compare; that accepts any scale while still
        # rejecting envelopes whose excerpts do not match the store.
        head_chars = preview.get("head")
        tail_chars = preview.get("tail")
        rebuilt = (
            json.loads(_preview(ref, permitted, len(head_chars), len(tail_chars)))
            if isinstance(head_chars, str) and isinstance(tail_chars, str)
            else None
        )
        if rebuilt is None or preview != rebuilt:
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
    refs: dict[int, Callable[[], str]],
    task_id: str,
    persist: Callable[[dict[str, Any]], str],
    *,
    budget: int,
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
                    {"ref": refs[index](), "tool_call_id": raw[index].get("tool_call_id")}
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
