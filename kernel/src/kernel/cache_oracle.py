"""Deterministic cache-prefix oracle (cache-efficiency SPEC §7 criterion 2).

A scripted end-to-end scenario drives model iterations through
:class:`CacheOracle.observe`, which digests the canonical provider
request shape — the typed system-prompt segments and the transcript
entries a gateway would actually place on the wire, plus the ordered
tool schemas — and explains every transition with an explicit break
reason. Reports carry digests, providers, and boundary positions;
never prompt text.

The carried-prefix model matches implicit provider caching: the longest
head of canonical entries that is unchanged since the previous request
is the region a prompt cache could still hold, so

- appending tool results after an unchanged prefix leaves the carried
  digest equal to the previous request's shape digest;
- a history fold or compaction moves the digest only at the boundary it
  rewrites (the oracle records the first divergent entry index); and
- a known-expired cache is reported as ``cache_expired`` — distinguishable
  from a changed prefix even when the shape is identical.

Fold evidence comes from the real :func:`kernel.history.fold_history`
machinery, not a parallel stub renderer.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from kernel.history import fold_history
from kernel.provider_contracts import CompletionRequest

__all__ = [
    "BREAK_CACHE_EXPIRED",
    "BREAK_IDENTICAL",
    "BREAK_NO_PRIOR",
    "BREAK_PREFIX_CHANGED",
    "BREAK_PROVIDER_CHANGED",
    "BREAK_TAIL_APPEND",
    "BREAK_TOOLS_CHANGED",
    "CacheOracle",
    "PrefixObservation",
    "apply_folds",
    "explain",
    "shape_digest",
    "tools_digest",
]

BREAK_NO_PRIOR = "no_prior_request"
BREAK_IDENTICAL = "identical_request"
BREAK_TAIL_APPEND = "tail_append_only"
BREAK_PREFIX_CHANGED = "prefix_changed"
BREAK_CACHE_EXPIRED = "cache_expired"
BREAK_PROVIDER_CHANGED = "provider_or_model_changed"
BREAK_TOOLS_CHANGED = "tool_schemas_changed"

_CACHEABLE_ROLES = frozenset({"system", "user", "assistant", "tool"})


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _entries(request: CompletionRequest) -> list[str]:
    """Canonical carried entries: typed system segments, then cacheable messages."""
    entries = [
        json.dumps({"system": s.text, "stability": s.stability}, sort_keys=True)
        for s in request.system_prompt_segments
    ]
    entries.extend(
        json.dumps(
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "tool_call_id": m.get("tool_call_id"),
                "tool_calls": m.get("tool_calls"),
            },
            sort_keys=True,
        )
        for m in request.messages
        if m.get("role") in _CACHEABLE_ROLES
    )
    return entries


def shape_digest(request: CompletionRequest) -> str:
    """Digest the canonical carried shape (system segments + messages)."""
    return _digest(_entries(request))


def tools_digest(request: CompletionRequest) -> str:
    """Digest the ordered tool schemas; reordering or widening changes it."""
    return _digest(request.tools or [])


@dataclass(frozen=True, slots=True)
class PrefixObservation:
    """One model iteration's cache-relevant facts. Digests, never prompt text."""

    index: int
    label: str
    provider: str
    model_id: str
    shape_digest: str
    tools_digest: str
    #: Digest of the longest still-valid carried head (unchanged entries head).
    carried_digest: str
    break_reason: str
    break_detail: str = ""
    prior_index: int | None = None
    #: First divergent canonical entry; ``None`` when nothing diverged.
    first_divergence: int | None = None
    #: Entries appended beyond an unchanged head (tail-append transitions).
    appended_entries: int = 0
    #: ``fresh``/``expired`` when the scenario configured a TTL; else ``n/a``.
    cache_ttl_state: str = "n/a"

    @property
    def short_digest(self) -> str:
        return self.shape_digest[:12]


def explain(observation: PrefixObservation) -> str:
    """One aggregate-only report line for an observation."""
    line = (
        f"[{observation.index}] {observation.label or 'iteration'} "
        f"{observation.provider}/{observation.model_id}: {observation.break_reason}"
    )
    if observation.break_detail:
        line += f" ({observation.break_detail})"
    if observation.cache_ttl_state != "n/a":
        line += f" ttl={observation.cache_ttl_state}"
    line += f" carried={observation.carried_digest[:12]} shape={observation.short_digest}"
    return line


@dataclass(slots=True)
class _CachedRequest:
    index: int
    entries: list[str]
    tools_digest: str
    provider: str
    model_id: str
    emitted_at: float


@dataclass
class CacheOracle:
    """Observe every model iteration of a scenario; digest, compare, explain."""

    #: Optional cache TTL in seconds. When set, a request issued after the
    #: TTL elapses with an otherwise identical shape reports
    #: ``cache_expired`` — a genuinely changed prefix still reports
    #: ``prefix_changed``, keeping the two causes distinguishable.
    ttl_seconds: float | None = None
    clock: Callable[[], float] = field(default=time.monotonic)
    _observations: list[PrefixObservation] = field(default_factory=list, init=False, repr=False)
    _last: _CachedRequest | None = field(default=None, init=False, repr=False)

    @property
    def observations(self) -> tuple[PrefixObservation, ...]:
        return tuple(self._observations)

    def observe(self, request: CompletionRequest, *, label: str = "") -> PrefixObservation:
        """Digest one canonical request and classify its transition."""
        entries = _entries(request)
        shape = _digest(entries)
        tools = tools_digest(request)
        provider = request.model_def.provider
        model_id = request.model_def.model_id
        now = self.clock()
        index = len(self._observations)
        prior = self._last

        reason = BREAK_NO_PRIOR
        detail = ""
        prior_index: int | None = None
        first_divergence: int | None = None
        appended = 0
        ttl_state = "n/a"
        carried = entries
        if prior is not None:
            prior_index = prior.index
            common = 0
            for old, new in zip(prior.entries, entries, strict=False):
                if old != new:
                    break
                common += 1
            shape_equal = entries == prior.entries
            tools_equal = tools == prior.tools_digest
            expired = self.ttl_seconds is not None and (now - prior.emitted_at) > self.ttl_seconds
            if self.ttl_seconds is not None:
                ttl_state = "expired" if expired else "fresh"
            if provider != prior.provider or model_id != prior.model_id:
                reason = BREAK_PROVIDER_CHANGED
                detail = f"shape_changed={not shape_equal}"
            elif common == len(prior.entries) and not shape_equal:
                reason = BREAK_TAIL_APPEND
                appended = len(entries) - len(prior.entries)
                detail = f"appended {appended} entries"
            elif not shape_equal:
                reason = BREAK_PREFIX_CHANGED
                first_divergence = common
            elif expired:
                reason = BREAK_CACHE_EXPIRED
                detail = (
                    f"identical shape after {now - prior.emitted_at:.1f}s; "
                    f"tools_changed={not tools_equal}"
                )
            elif not tools_equal:
                reason = BREAK_TOOLS_CHANGED
                detail = "carried transcript unchanged; ordered tool schemas differ"
            else:
                reason = BREAK_IDENTICAL
            carried = entries[: len(prior.entries)] if shape_equal else entries[:common]

        self._last = _CachedRequest(
            index=index,
            entries=entries,
            tools_digest=tools,
            provider=provider,
            model_id=model_id,
            emitted_at=now,
        )
        observation = PrefixObservation(
            index=index,
            label=label,
            provider=provider,
            model_id=model_id,
            shape_digest=shape,
            tools_digest=tools,
            carried_digest=_digest(carried),
            break_reason=reason,
            break_detail=detail,
            prior_index=prior_index,
            first_divergence=first_divergence,
            appended_entries=appended,
            cache_ttl_state=ttl_state,
        )
        self._observations.append(observation)
        return observation

    def render_report(self) -> list[str]:
        """Aggregate-only scenario report: one line per observed iteration."""
        lines = ["### Cache prefix oracle"]
        lines.extend(explain(obs) for obs in self._observations)
        return lines


def apply_folds(
    messages: list[dict[str, Any]],
    entries: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Apply real :func:`kernel.history.fold_history` results to a transcript.

    Returns the rewritten message list and the replacement stubs keyed by
    call id. Whole-message tool results are replaced in place; anything
    else is left untouched so scenario authors see exactly what folded.
    Re-applying the same fold entries changes nothing further — a fold
    and a compaction at the same boundary cannot double-rewrite.
    """
    folds = fold_history(entries)
    if not folds.replacements:
        return list(messages), {}
    rewritten: list[dict[str, Any]] = []
    for message in messages:
        call_id = message.get("tool_call_id")
        if message.get("role") == "tool" and call_id in folds.replacements:
            rewritten.append({**message, "content": folds.replacements[call_id]})
        else:
            rewritten.append(message)
    return rewritten, dict(folds.replacements)
