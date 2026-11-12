"""In-turn pack history: record claims, fold stale results, finalize, resolve anchors.

The pure ledger lives in :mod:`kernel.history`; this module wires it into
the turn. Folding follows the governor's cache policy: the live array is
append-only between epochs, and a supersession epoch rewrites it only when
the model's cache prices say the rewrite is repaid by the stale bytes it
stops re-sending on the iterations still expected this turn.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from kernel.history import (
    HistoryRecord,
    existing_result_ref,
    render_stub,
    resolve_history_keys,
)
from kernel.models import ToolResult
from kernel.tools.registry import get_tool_catalog, invoke_history_hook
from kernel.turn_phases._post_turn import _to_tool_context

_ANCHOR = re.compile(r"(?P<tool>[a-z][a-z0-9_]*):\S+")
_DEFAULT_ANCHOR_PAGE_CHARS = 4000
_MAX_ANCHOR_PAGE_CHARS = 16000


def record_dispatch_history(
    *,
    state: Any,
    tool_def: Any,
    tool_name: str,
    tool_call_id: str,
    params: Any,
    tool_result: Any,
    resident_content: str,
    log: logging.Logger,
) -> None:
    """Record one result's history claims (an error's side effects only) on the ledger."""
    policy = getattr(tool_def, "history", None)
    if policy is None or tool_result is None:
        return
    try:
        claims = resolve_history_keys(
            policy,
            getattr(tool_def, "implementation_module", None),
            params if isinstance(params, dict) else {},
            tool_result,
        )
    except Exception as exc:  # pylint: disable=broad-except
        # A pack hook bug must never break the turn; the result just stays resident.
        log.warning("history keys failed for %s; result stays resident: %s", tool_name, exc)
        return
    if claims is None or not (claims.keys or claims.invalidates or claims.scope == "turn"):
        return
    newly_stale = state.history_ledger.record(
        HistoryRecord(
            call_id=tool_call_id,
            tool_id=tool_name,
            owner_id=str(getattr(tool_def, "owner_id", "") or getattr(tool_def, "pack_id", "")),
            keys=list(claims.keys),
            invalidates=list(claims.invalidates),
            scope=claims.scope,
            anchor=claims.anchor,
            chars=len(resident_content),
        )
    )
    if newly_stale:
        log.debug(
            "history: %s made %d result(s) stale (%s)",
            tool_call_id,
            len(newly_stale),
            ", ".join(r.call_id for r in newly_stale),
        )


def _tool_message_index(messages: list[dict[str, Any]], call_id: str, start: int) -> int:
    for index in range(len(messages) - 1, start - 1, -1):
        message = messages[index]
        if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
            return index
    return -1


def _ensure_ref(record: HistoryRecord, content: str, hooks: Any) -> str:
    if record.ref:
        return record.ref
    ref = existing_result_ref(content)
    persist = hooks.tools.persist_tool_result
    if not ref and persist is not None:
        ref = str(
            persist(
                {"content": content, "task_id": hooks.identity.task_id, "tool_name": record.tool_id}
            )
        )
    record.ref = ref
    return ref


def cache_price_shares(model_def: Any, cfg: Any) -> tuple[float, float]:
    """Return ``(read, write)`` cache prices as shares of the uncached input price.

    The manifest's per-model prices win when the provider publishes a
    cached-read price; a published read price without a write price means
    cache creation bills at the base rate. Otherwise the configured
    fallback shares apply.
    """
    prompt = float(getattr(model_def, "prompt_price", 0.0) or 0.0)
    read = float(getattr(model_def, "cache_read_price", 0.0) or 0.0)
    write = float(getattr(model_def, "cache_write_price", 0.0) or 0.0)
    if prompt > 0 and read > 0:
        return read / prompt, (write / prompt if write > 0 else 1.0)
    return float(cfg.cache_read_share), float(cfg.cache_write_share)


def fold_pays_off(
    *,
    removed: int,
    suffix: int,
    later_requests: int,
    read_share: float,
    write_share: float,
) -> float:
    """Return the net saving (in uncached-input units) of folding; > 0 pays off.

    Folding rewrites the cached ``suffix`` from the first stub onward: the
    next request writes ``suffix - removed`` instead of reading ``suffix``
    from cache, and each later request reads ``removed`` fewer cached
    units. Without caching (``read_share == write_share == 1``) the
    rewrite is free and any removal pays off.
    """
    rewrite_cost = write_share * (suffix - removed) - read_share * suffix
    return read_share * removed * later_requests - rewrite_cost


def _message_chars(message: dict[str, Any]) -> int:
    size = len(str(message.get("content") or ""))
    if message.get("tool_calls"):
        size += len(json.dumps(message["tool_calls"], default=str))
    return size


def _later_requests(state: Any) -> int:
    """Estimate the requests left in this turn from the requests made so far."""
    done = max(1, int(getattr(state, "iteration", 0) or 0))
    cap = int(getattr(state, "max_iterations", 0) or 0)
    return max(0, min(done, cap - done)) if cap > 0 else done


def _best_fold_start(
    candidates: list[tuple[int, HistoryRecord]],
    messages: list[dict[str, Any]],
    *,
    later_requests: int,
    read_share: float,
    write_share: float,
) -> int:
    """Return the candidate position to fold from (folding all later ones), or -1."""
    best, best_saving = -1, 0.0
    removed = 0
    suffix = sum(_message_chars(m) for m in messages[candidates[-1][0] :])
    next_index = candidates[-1][0]
    for position in range(len(candidates) - 1, -1, -1):
        index, record = candidates[position]
        suffix += sum(_message_chars(m) for m in messages[index:next_index])
        next_index = index
        removed += record.chars
        saving = fold_pays_off(
            removed=removed,
            suffix=suffix,
            later_requests=later_requests,
            read_share=read_share,
            write_share=write_share,
        )
        if saving > best_saving:
            best, best_saving = position, saving
    return best


def _fold_candidates(state: Any, messages: list[dict[str, Any]]) -> list[tuple[int, HistoryRecord]]:
    """Return stale records still resident, with their message index, in order."""
    candidates: list[tuple[int, HistoryRecord]] = []
    for record in state.history_ledger.foldable(turn_finished=False):
        index = _tool_message_index(messages, record.call_id, state.transcript_start)
        if index < 0:
            record.folded = True
        else:
            candidates.append((index, record))
    return sorted(candidates, key=lambda pair: pair[0])


def maybe_run_supersession_epoch(
    *,
    state: Any,
    hooks: Any,
    messages: list[dict[str, Any]],
    log: logging.Logger,
    model_def: Any = None,
) -> int:
    """Fold stale results when the cache rewrite pays for itself. Returns the fold count.

    Above the ``history_epoch_min_tokens`` floor, the epoch picks the fold
    boundary with the largest net saving under the model's cache prices
    (:func:`fold_pays_off`); a later boundary rewrites less cached prefix,
    so older stale results may wait for a later epoch. Every stub carries
    a persisted ``r<sha256>`` ref; a result whose ref cannot be produced
    (no store wired) stays resident.
    """
    cfg = state.intra_turn_compaction
    ledger = state.history_ledger
    if not cfg.enabled or cfg.history_epoch_min_tokens <= 0:
        return 0
    # Same conservative 3 bytes/token the live-array estimator uses.
    if -(-ledger.pending_dead_chars() // 3) < cfg.history_epoch_min_tokens:
        return 0
    candidates = _fold_candidates(state, messages)
    if not candidates:
        return 0
    read_share, write_share = cache_price_shares(model_def, cfg)
    later = _later_requests(state)
    start = _best_fold_start(
        candidates,
        messages,
        later_requests=later,
        read_share=read_share,
        write_share=write_share,
    )
    if start < 0:
        log.debug(
            "history epoch: %d stale result(s) kept; no fold repays its cache rewrite "
            "(read=%.3g write=%.3g later=%d)",
            len(candidates),
            read_share,
            write_share,
            later,
        )
        return 0
    removed = 0
    folded = 0
    for index, record in candidates[start:]:
        content = str(messages[index].get("content") or "")
        if not _ensure_ref(record, content, hooks):
            continue
        stub = render_stub(record)
        messages[index]["content"] = stub
        # The projection copy is reused as the next request's prefix; fold it
        # too or the stale text comes back from the snapshot.
        projected = state.provider_messages
        if index < len(projected) and projected[index].get("tool_call_id") == record.call_id:
            projected[index]["content"] = stub
        record.folded = True
        folded += 1
        removed += max(0, len(content.encode("utf-8")) - len(stub.encode("utf-8")))
    if folded:
        estimator = state.intra_turn_compaction_estimator
        # Under-subtract (4 bytes/token) so the estimate stays biased high.
        estimator._estimate = max(0, estimator._estimate - removed // 4)
        estimator._live_bytes = max(0, estimator._live_bytes - removed)
        log.info("history epoch: folded %d stale result(s), %d bytes", folded, removed)
    return folded


async def finalize_turn_history(
    *,
    state: Any,
    hooks: Any,
    tool_catalog: dict[str, Any],
    log: logging.Logger,
) -> dict[str, Any]:
    """Persist refs for every recorded result and collect pack snapshots.

    Returns the thread-entry ``history`` block (``{}`` when nothing was
    recorded). Replay folds from these records, so the thread transcript
    itself stays the lossless record.
    """
    ledger = state.history_ledger
    if not ledger.records or state.messages is None:
        return {}
    messages = state.messages
    # A resumed turn restores an earlier segment's ledger; only results in
    # this entry's transcript belong to its block (replay prefixes call ids
    # per entry, so a carried-over record would shadow the real one).
    own: list[HistoryRecord] = []
    for record in ledger.records:
        index = _tool_message_index(messages, record.call_id, state.transcript_start)
        if index < 0:
            continue
        own.append(record)
        if not record.ref:
            _ensure_ref(record, str(messages[index].get("content") or ""), hooks)
    if not own:
        return {}
    history: dict[str, Any] = {"records": [r.to_json() for r in own]}
    snapshots = await _collect_snapshots(state, own, tool_catalog, log)
    if snapshots:
        history["snapshots"] = snapshots
    return history


async def _collect_snapshots(
    state: Any,
    records: list[HistoryRecord],
    tool_catalog: dict[str, Any],
    log: logging.Logger,
) -> dict[str, str]:
    foldable_owners = {r.owner_id for r in records if r.owner_id and (r.dead or r.scope == "turn")}
    if not foldable_owners or state.ctx is None:
        return {}
    snapshots: dict[str, str] = {}
    for tool_id, definition in tool_catalog.items():
        policy = getattr(definition, "history", None)
        owner = str(getattr(definition, "owner_id", "") or getattr(definition, "pack_id", ""))
        if policy is None or not policy.snapshot or owner not in foldable_owners:
            continue
        if owner in snapshots:
            continue
        keys = [key for r in records if r.owner_id == owner for key in r.keys]
        try:
            text = await invoke_history_hook(
                tool_id, "history_snapshot", _to_tool_context(tool_id, state.ctx), keys
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("history snapshot failed for %s: %s", tool_id, exc)
            continue
        if isinstance(text, str) and text.strip():
            snapshots[owner] = text.strip()
    return snapshots


async def maybe_resolve_anchor_read(
    *,
    state: Any,
    tool_name: str,
    tool_arguments: dict[str, Any],
    tool_catalog: dict[str, Any],
    ctx: Any,
    log: logging.Logger,
) -> str | None:
    """Serve ``thread_read_content`` for a pack anchor, or return ``None``.

    An anchor is ``<tool_id>:<payload>`` from a stub. The owning tool's
    ``resolve_history_anchor`` returns the content; when it cannot, the
    call is rewritten to the stub's ``r<sha256>`` ref so the original
    result is still served by the thread store.
    """
    if tool_name.rsplit(".", 1)[-1] != "thread_read_content":
        return None
    anchor = str(tool_arguments.get("ref") or "")
    match = _ANCHOR.fullmatch(anchor)
    definition = tool_catalog.get(match.group("tool")) if match else None
    if match is None or definition is None or getattr(definition, "history", None) is None:
        return None
    if not _anchor_issued(state, anchor):
        return None
    content: Any = None
    try:
        content = await invoke_history_hook(
            match.group("tool"),
            "resolve_history_anchor",
            _to_tool_context(match.group("tool"), ctx),
            anchor,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("history anchor %s failed to resolve: %s", anchor, exc)
    if isinstance(content, str):
        return _anchor_page(anchor, content, tool_arguments)
    fallback = next(
        (r.ref for r in reversed(state.history_ledger.records) if r.anchor == anchor and r.ref),
        "",
    )
    if fallback:
        tool_arguments["ref"] = fallback
        return None
    return json.dumps(
        {
            "ref": anchor,
            "error": "anchor could not be resolved; dereference the r<sha256> ref in its stub",
        }
    )


def approved_invocation_history(
    *,
    tool_id: str,
    request_id: str,
    parameters: dict[str, Any],
    result: Any,
    log: logging.Logger,
) -> dict[str, Any]:
    """History block for a host-approved tool call executed outside a turn.

    The approved result lands in the thread entry's response text, not in a
    tool message, so it never folds; its record exists so the call's
    ``invalidates`` and ``key`` claims still fire at replay (an approved
    edit must stale the reads it changed).
    """
    definition = get_tool_catalog().get(tool_id)
    policy = getattr(definition, "history", None)
    if policy is None or not isinstance(result, ToolResult):
        return {}
    try:
        claims = resolve_history_keys(
            policy, getattr(definition, "implementation_module", None), parameters, result
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("history keys failed for approved %s: %s", tool_id, exc)
        return {}
    if claims is None or not (claims.keys or claims.invalidates):
        return {}
    record = HistoryRecord(
        call_id=f"approved-{request_id}",
        tool_id=tool_id,
        owner_id=str(getattr(definition, "owner_id", "") or getattr(definition, "pack_id", "")),
        keys=list(claims.keys),
        invalidates=list(claims.invalidates),
        scope=claims.scope,
    )
    return {"records": [record.to_json()]}


def _anchor_issued(state: Any, anchor: str) -> bool:
    """True when the kernel handed ``anchor`` out in a stub.

    Anchor resolvers run pack code with anchor-derived parameters, so only
    anchors from this turn's ledger or from a stub in the live messages
    (replay folds) are served; anything else takes the normal ref path.
    """
    if any(r.anchor == anchor for r in state.history_ledger.records):
        return True
    marker = f" or {anchor} with thread_read_content]"
    return any(
        isinstance(m.get("content"), str) and marker in m["content"]
        for m in state.messages or ()
        if m.get("role") == "tool"
    )


def _anchor_page(anchor: str, content: str, tool_arguments: dict[str, Any]) -> str:
    offset = max(0, int(tool_arguments.get("offset_chars") or 0))
    limit = int(tool_arguments.get("limit_chars") or _DEFAULT_ANCHOR_PAGE_CHARS)
    limit = min(max(1, limit), _MAX_ANCHOR_PAGE_CHARS)
    page = content[offset : offset + limit]
    end = offset + len(page)
    return json.dumps(
        {
            "ref": anchor,
            "source": "pack anchor (current state)",
            "content": page,
            "offset_chars": offset,
            "total_chars": len(content),
            "next_offset_chars": end if end < len(content) else None,
            "has_more": end < len(content),
        },
        ensure_ascii=False,
    )


__all__ = [
    "cache_price_shares",
    "finalize_turn_history",
    "fold_pays_off",
    "maybe_resolve_anchor_read",
    "maybe_run_supersession_epoch",
    "record_dispatch_history",
]
