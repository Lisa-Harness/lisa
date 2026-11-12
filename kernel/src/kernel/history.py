"""Pack-declared history semantics: supersession keys, scope, and the fold ledger.

Packs make claims; the kernel stays the sole rewriter of model history. A
tool's ``.tool.yml`` ``history:`` block (or its module's ``history_keys``
hook) names opaque keys for each successful result:

- ``key`` — a later result with an equal key supersedes the earlier one.
- ``invalidates`` — key prefixes this result makes stale (an edit
  invalidates earlier reads of the edited paths).
- ``scope`` — ``task`` results stay resident until superseded; ``turn``
  results fold once their turn has finished.

A folded result is replaced by a one-line stub that always carries the
result's ``r<sha256>`` ref, so folding never loses evidence: the model can
dereference the ref with ``thread_read_content`` (or the pack's anchor,
when the tool declares one). The ledger is shared by the in-turn
supersession epoch and the turn-start replay so both fold identically.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

HistoryScope = Literal["task", "turn"]

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_FIELD_PATH = re.compile(
    r"(?P<result>result\.)?(?P<root>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<fanout>\[\])?(?P<rest>(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
)
_RESULT_REF = re.compile(r"r[a-f0-9]{64}")
_MISSING = object()


class HistoryTemplateError(ValueError):
    """A history template is malformed or names an unknown parameter."""


@dataclass(frozen=True, slots=True)
class _Placeholder:
    from_result: bool
    root: str
    fanout: bool
    rest: tuple[str, ...]


def _parse_template(template: str) -> list[_Placeholder]:
    placeholders: list[_Placeholder] = []
    for raw in _PLACEHOLDER.findall(template):
        match = _FIELD_PATH.fullmatch(raw)
        if match is None:
            raise HistoryTemplateError(f"invalid history placeholder {{{raw}}} in {template!r}")
        rest = tuple(part for part in match.group("rest").split(".") if part)
        placeholders.append(
            _Placeholder(
                from_result=bool(match.group("result")),
                root=match.group("root"),
                fanout=bool(match.group("fanout")),
                rest=rest,
            )
        )
    if sum(1 for p in placeholders if p.fanout) > 1:
        raise HistoryTemplateError(f"history template {template!r} has more than one [] fan-out")
    leftover = _PLACEHOLDER.sub("", template)
    if "{" in leftover or "}" in leftover:
        raise HistoryTemplateError(f"unbalanced braces in history template {template!r}")
    return placeholders


class HistoryPolicy(BaseModel):
    """A tool's declared history semantics (the ``history:`` yml block).

    ``key``/``invalidates``/``anchor`` are templates: ``{param}`` reads the
    validated call parameters, ``{result.field}`` reads the result's
    ``output_json``, ``{name[]}`` / ``{name[].field}`` fans one template out
    over a list. ``hook: true`` delegates key computation to the tool
    module's ``history_keys(params, result)``. ``snapshot: true`` marks the
    tool module's ``history_snapshot(ctx, keys)`` as the pack's turn-end
    state summary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    scope: HistoryScope = "task"
    anchor: str = ""
    hook: bool = False
    snapshot: bool = False

    @field_validator("key", "invalidates", mode="before")
    @classmethod
    def _coerce_templates(cls, value: object) -> object:
        if isinstance(value, str):
            return (value,)
        return value

    @field_validator("key", "invalidates")
    @classmethod
    def _check_templates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for template in value:
            if not template.strip():
                raise HistoryTemplateError("history templates must be non-empty")
            _parse_template(template)
        return value

    @field_validator("anchor")
    @classmethod
    def _check_anchor(cls, value: str) -> str:
        if value:
            _parse_template(value)
            if ":" not in value.split("{", 1)[0]:
                raise HistoryTemplateError(
                    f"history anchor {value!r} must start with a literal '<tool_id>:' scheme"
                )
        return value

    def parameter_roots(self) -> set[str]:
        """Return the call-parameter names the templates read."""
        roots: set[str] = set()
        for template in (*self.key, *self.invalidates, *((self.anchor,) if self.anchor else ())):
            roots.update(p.root for p in _parse_template(template) if not p.from_result)
        return roots

    def validate_parameters(self, tool_id: str, parameter_names: Iterable[str]) -> None:
        """Raise when a template names a parameter the tool does not accept."""
        if self.hook and (self.key or self.invalidates or self.scope != "task"):
            raise HistoryTemplateError(
                f"tool {tool_id!r}: history.hook computes keys; drop key/invalidates/scope"
            )
        if self.anchor and not self.anchor.startswith(f"{tool_id}:"):
            raise HistoryTemplateError(
                f"tool {tool_id!r}: history.anchor must start with '{tool_id}:'"
            )
        unknown = self.parameter_roots() - set(parameter_names)
        if unknown:
            raise HistoryTemplateError(
                f"tool {tool_id!r}: history templates name unknown parameters "
                f"{sorted(unknown)}; use {{result.<field>}} for result fields"
            )


class HistoryKeys(BaseModel):
    """Resolved history claims for one result (template output or hook return)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    scope: HistoryScope = "task"
    anchor: str = ""


def _lookup(source: Any, path: tuple[str, ...]) -> Any:
    value = source
    for part in path:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return _MISSING
    return value


def _render(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _placeholder_values(
    placeholder: _Placeholder, params: Mapping[str, Any], result: Mapping[str, Any]
) -> list[Any] | None:
    source = result if placeholder.from_result else params
    base = _lookup(source, (placeholder.root,))
    if base is _MISSING:
        return None
    if not placeholder.fanout:
        value = _lookup(base, placeholder.rest) if placeholder.rest else base
        return None if value is _MISSING else [value]
    if not isinstance(base, list | tuple):
        return None
    values = [_lookup(item, placeholder.rest) if placeholder.rest else item for item in base]
    return [value for value in values if value is not _MISSING]


def expand_template(
    template: str,
    params: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    none_is_missing: bool = False,
) -> list[str]:
    """Expand one template; a missing field yields no keys rather than a guess."""
    expansions = [""]
    cursor = 0
    for match, placeholder in zip(
        _PLACEHOLDER.finditer(template), _parse_template(template), strict=True
    ):
        literal = template[cursor : match.start()]
        cursor = match.end()
        values = _placeholder_values(placeholder, params, result)
        if values is None or (none_is_missing and any(v is None for v in values)):
            return []
        rendered = [_render(value) for value in values]
        expansions = [prefix + literal + value for prefix in expansions for value in rendered]
    tail = template[cursor:]
    return list(dict.fromkeys(prefix + tail for prefix in expansions))


def keys_from_policy(
    policy: HistoryPolicy, params: Mapping[str, Any], result: Mapping[str, Any]
) -> HistoryKeys:
    """Resolve a declarative policy's templates against one call and result."""
    keys = [key for template in policy.key for key in expand_template(template, params, result)]
    invalidates = [
        key for template in policy.invalidates for key in expand_template(template, params, result)
    ]
    anchors = (
        expand_template(policy.anchor, params, result, none_is_missing=True)
        if policy.anchor
        else []
    )
    return HistoryKeys(
        keys=tuple(dict.fromkeys(keys)),
        invalidates=tuple(dict.fromkeys(invalidates)),
        scope=policy.scope,
        anchor=anchors[0] if len(anchors) == 1 else "",
    )


def _validated_params(module: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the tool's ``Parameters`` defaults so templates see omitted fields."""
    model = getattr(module, "Parameters", None)
    if model is None or not hasattr(model, "model_validate"):
        return dict(params)
    try:
        return dict(model.model_validate(dict(params)).model_dump(mode="json"))
    except ValueError:
        return dict(params)


def resolve_history_keys(
    policy: HistoryPolicy | None,
    module: Any,
    params: Mapping[str, Any],
    result: Any,
) -> HistoryKeys | None:
    """Return the history claims for one successful tool result, or ``None``.

    ``result`` is the tool's ``ToolResult``. ``params`` are the raw call
    arguments; the tool's ``Parameters`` model fills defaults first. Hooked
    tools call the module's ``history_keys(params, result)``; declarative
    tools expand templates over ``params`` and ``result.output_json``.
    """
    if policy is None or result is None or str(getattr(result, "error", "") or "").strip():
        return None
    params = _validated_params(module, params)
    if policy.hook:
        hook = getattr(module, "history_keys", None)
        if hook is None:
            raise HistoryTemplateError("history.hook is set but the module has no history_keys")
        claims = hook(params, result)
        if claims is None:
            return None
        return HistoryKeys.model_validate(claims)
    output = getattr(result, "output_json", None)
    return keys_from_policy(policy, params, output if isinstance(output, Mapping) else {})


@dataclass
class HistoryRecord:
    """One tool result's history claims and fold state.

    A dataclass (not a pydantic model) so ``_TurnState``'s TypeAdapter can
    generate a schema for it; the record serializes to the thread entry via
    :meth:`to_json`.
    """

    call_id: str
    tool_id: str
    owner_id: str = ""
    keys: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)
    scope: str = "task"
    anchor: str = ""
    chars: int = 0
    ref: str = ""
    dead_keys: list[str] = field(default_factory=list)
    stale_by: str = ""
    stale_reason: str = ""
    folded: bool = False

    @property
    def dead(self) -> bool:
        """Every key this result claimed has been superseded or invalidated."""
        return bool(self.keys) and set(self.keys) <= set(self.dead_keys)

    def to_json(self) -> dict[str, Any]:
        """Return the thread-entry form (fold state is recomputed on replay)."""
        record: dict[str, Any] = {
            "call_id": self.call_id,
            "tool_id": self.tool_id,
            "scope": self.scope,
            "chars": self.chars,
        }
        for name in ("owner_id", "anchor", "ref"):
            if value := getattr(self, name):
                record[name] = value
        if self.keys:
            record["keys"] = list(self.keys)
        if self.invalidates:
            record["invalidates"] = list(self.invalidates)
        return record

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, call_prefix: str = "") -> HistoryRecord:
        """Rebuild a record from its thread-entry form."""
        return cls(
            call_id=f"{call_prefix}{data.get('call_id', '')}",
            tool_id=str(data.get("tool_id") or ""),
            owner_id=str(data.get("owner_id") or ""),
            keys=[str(k) for k in data.get("keys") or ()],
            invalidates=[str(k) for k in data.get("invalidates") or ()],
            scope=str(data.get("scope") or "task"),
            anchor=str(data.get("anchor") or ""),
            chars=int(data.get("chars") or 0),
            ref=str(data.get("ref") or ""),
        )


@dataclass
class HistoryLedger:
    """Supersession ledger keyed by tool-call id, in dispatch order."""

    records: list[HistoryRecord] = field(default_factory=list)

    def record(self, record: HistoryRecord) -> list[HistoryRecord]:
        """Append ``record``; return earlier records that just went fully stale."""
        newly_dead: list[HistoryRecord] = []
        for prior in self.records:
            if not prior.keys or prior.dead:
                continue
            for key in prior.keys:
                if key in prior.dead_keys:
                    continue
                if key in record.keys:
                    reason = "superseded"
                elif any(key.startswith(prefix) for prefix in record.invalidates):
                    reason = "invalidated"
                else:
                    continue
                prior.dead_keys.append(key)
                prior.stale_by = record.call_id
                prior.stale_reason = reason
            if prior.dead:
                newly_dead.append(prior)
        self.records.append(record)
        return newly_dead

    def pending_dead_chars(self) -> int:
        """Characters of fully stale results that are still resident."""
        return sum(r.chars for r in self.records if r.dead and not r.folded)

    def foldable(self, *, turn_finished: bool) -> list[HistoryRecord]:
        """Records to fold: stale ones, plus turn-scoped ones once the turn ends."""
        return [
            r
            for r in self.records
            if not r.folded and (r.dead or (turn_finished and r.scope == "turn"))
        ]


def render_stub(record: HistoryRecord) -> str:
    """Render the one-line stub that replaces a folded result.

    Invariant: every stub carries the result's ``r<sha256>`` ref.
    """
    if not _RESULT_REF.fullmatch(record.ref):
        raise ValueError(f"history stub for call {record.call_id} has no result ref")
    if record.dead:
        head = f"[{record.stale_reason or 'superseded'} by call {record.stale_by}"
    else:
        head = "[turn-scoped result folded after its turn"
    anchor = f" or {record.anchor}" if record.anchor else ""
    return (
        f"{head} ({record.chars} chars) — dereference {record.ref}{anchor} "
        "with thread_read_content]"
    )


def existing_result_ref(content: str) -> str:
    """Return the ``r<sha256>`` ref a spilled/previewed result already carries."""
    stripped = content.strip()
    if _RESULT_REF.fullmatch(stripped):
        return stripped
    if not stripped.startswith("{") or '"ref"' not in stripped:
        return ""
    try:
        envelope = json.loads(stripped)
    except ValueError:
        return ""
    if not isinstance(envelope, dict):
        return ""
    ref = envelope.get("ref")
    previewed = envelope.get("abbreviated") is True or envelope.get("page_omitted") is True
    return ref if previewed and isinstance(ref, str) and _RESULT_REF.fullmatch(ref) else ""


def fold_replay(
    entries: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, str]:
    """Return replacement content for replayed tool results, keyed by call id.

    ``entries`` are ``(call_id_prefix, entry_history)`` pairs in thread
    order; replay prefixes call ids per turn, so records are rebuilt with
    the same prefix. Every replayed turn is finished, so turn-scoped
    results fold along with stale ones. The newest snapshot per pack owner
    is appended to that owner's last folded result.
    """
    ledger = HistoryLedger()
    snapshots: dict[str, str] = {}
    for prefix, history in entries:
        for data in history.get("records") or ():
            if isinstance(data, Mapping):
                ledger.record(HistoryRecord.from_json(data, call_prefix=prefix))
        for owner, text in (history.get("snapshots") or {}).items():
            if isinstance(text, str) and text.strip():
                snapshots[str(owner)] = text.strip()
    folded = [r for r in ledger.foldable(turn_finished=True) if _RESULT_REF.fullmatch(r.ref)]
    replacements = {record.call_id: render_stub(record) for record in folded}
    last_by_owner: dict[str, str] = {}
    for record in folded:
        last_by_owner[record.owner_id] = record.call_id
    for owner, call_id in last_by_owner.items():
        if owner in snapshots:
            replacements[call_id] += f"\n[{owner} state] {snapshots[owner]}"
    return replacements


__all__ = [
    "HistoryKeys",
    "HistoryLedger",
    "HistoryPolicy",
    "HistoryRecord",
    "HistoryScope",
    "HistoryTemplateError",
    "existing_result_ref",
    "expand_template",
    "fold_replay",
    "keys_from_policy",
    "render_stub",
    "resolve_history_keys",
]
