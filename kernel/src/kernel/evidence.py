"""Typed evidence payloads and the common envelope for durable ledgers.

Every durable evidence row is self-describing: it names its ``topic``, the
``schema_version`` of that topic's payload, when the fact happened
(``emitted_at``), and which execution, run configuration, and model call
produced it. This module is the single owner of those declarations so the
ledger handlers and the readers cannot drift apart.

Why an allowlist and not a schema-on-write validator
The token ledger already protects itself this way (``_LEDGER_FIELDS`` in
:mod:`kernel.token_usage`): the durable serializer writes exactly the typed
fields it declares and drops everything else. A publisher bug therefore
cannot put prompt, response, or tool-argument content on disk, because no
allowlist entry exists for it to land in. This module generalizes that
posture to every decision-ledger topic instead of inventing a second one.

Dropping is silent by design
A publisher that sends an undeclared key loses it. That is the point: the
allowlist is enforced at write, so the payload models below are the contract
and adding a field is a deliberate act. Removal or retyping is a hard
cutover (bump the topic's version); readers never guess.

Unknown topics pass through
A host that subscribes a topic this module does not declare gets its rows
persisted unchanged, with a warning naming the topic. Dropping every key of
an undeclared topic would turn a missing declaration into lost evidence,
which is the worse failure; the warning makes the gap visible instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

#: Handled-and-stamped by the ledger handler. Publishers never set these:
#: ``logged_at`` is the append time, ``sequence``/``ledger_instance`` are the
#: loss-detection stamps, and ``topic`` names which declaration applies.
HANDLER_STAMPED_FIELDS: frozenset[str] = frozenset(
    {"topic", "logged_at", "sequence", "ledger_instance"}
)

#: Envelope fields a publisher may stamp on any topic. The handler fills the
#: ones a publisher omits so every row carries a complete envelope; ``None``
#: is explicit unknown ("no turn", "no manifest yet"), never a guess.
PUBLISHER_ENVELOPE_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "emitted_at",
        "user_slug",
        "thread_store_slug",
        "turn_index",
        "turn_id",
        "run_ref",
        "trace_id",
        "span_id",
    }
)

#: The complete envelope: handler stamps plus publisher fields.
ENVELOPE_FIELDS: frozenset[str] = HANDLER_STAMPED_FIELDS | PUBLISHER_ENVELOPE_FIELDS

#: JSON Schema dialect the committed schemas declare.
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: Keys every persisted row carries because the handler itself writes them:
#: the append stamps, the topic tag, and the envelope values it fills with
#: explicit nulls. These are the schema's ``required`` set.
_HANDLER_GUARANTEED_FIELDS: frozenset[str] = frozenset(
    {
        "emitted_at",
        "ledger_instance",
        "logged_at",
        "run_ref",
        "schema_version",
        "sequence",
        "span_id",
        "topic",
        "trace_id",
        "turn_id",
    }
)


class _Payload(BaseModel):
    """Base for topic payloads: every field optional, unknown keys dropped.

    ``extra="ignore"`` matters twice: the handler already dropped undeclared
    keys before validation, and a caller validating a bare publisher payload
    must not fail merely because the envelope is absent.
    """

    model_config = ConfigDict(extra="ignore")


class TurnDecisionPayload(_Payload):
    """One ``turn.decision`` row: why the loop continued, finished, or yielded."""

    mode_id: str | None = None
    iteration: int | None = None
    turns_in_stage: int | None = None
    action_type: str | None = None
    action_reason: str | None = None
    total_tool_calls: int | None = None
    write_receipts: int | None = None
    read_receipts: int | None = None
    #: The model call whose output this decision resolved. ``None`` = the
    #: decision was not attributable to a single call (legacy or non-call
    #: decisions such as a workflow-position yield).
    call_id: str | None = None


class ToolTelemetryPayload(_Payload):
    """One ``tool.telemetry`` row: what ran, how long, and with what status.

    Mirrors ``kernel.models.ToolExecutionTelemetry`` field for field; the
    drift test pins the two together so a change to the event cannot silently
    become a silent drop here.
    """

    event_id: str | None = None
    event_type: str | None = None
    idempotency_key: str | None = None
    workspace_id: str | None = None
    execution_id: str | None = None
    target_id: str | None = None
    target_type: str | None = None
    status: str | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    normalized_fields: list[str] | None = None
    timestamp: str | None = None
    #: Join keys minted per dispatch and per requesting model call.
    dispatch_id: str | None = None
    tool_call_id: str | None = None
    call_id: str | None = None
    iteration: int | None = None


class ModelRequestPayload(_Payload):
    """One ``model.request`` row: an off-turn call an operation made."""

    operation_id: str | None = None
    request_ref: str | None = None
    call_id: str | None = None
    dispatch_id: str | None = None
    iteration: int | None = None


class RoutingDecisionPayload(_Payload):
    """One ``routing.decision`` row: the hint shown and the arming decision."""

    mode_id: str | None = None
    query_preview: str | None = None
    iteration: int | None = None
    turn: int | None = None
    hint_kind: str | None = None
    hint_targets: list[str] | None = None
    hint_injected: bool | None = None
    hint_suppressed: bool | None = None
    forced_armed: bool | None = None
    forced_kind: str | None = None
    forced_targets: list[str] | None = None
    forced_exempt: bool | None = None
    model_id: str | None = None
    model_cost_tier: str | None = None
    model_reasoning: bool | None = None
    model_structured_reliability: float | None = None


class RoutingForcedPayload(_Payload):
    """One ``routing.forced`` row: the enforcer's accountability record."""

    mode_id: str | None = None
    iteration: int | None = None
    kind: str | None = None
    targets: list[str] | None = None
    non_compliant_dispatches: int | None = None
    trigger: str | None = None
    expected_tool: str | None = None
    model_id: str | None = None
    model_cost_tier: str | None = None
    model_reasoning: bool | None = None
    model_structured_reliability: float | None = None


@dataclass(frozen=True)
class TopicSpec:
    """One topic's payload contract."""

    #: Payload model whose fields are the topic's write-time allowlist.
    model: type[_Payload]
    #: Current payload schema revision. Bumped when a field is added.
    schema_version: int = 1


#: Every topic this module declares, keyed by ledger topic name.
TOPIC_SPECS: dict[str, TopicSpec] = {
    "turn.decision": TopicSpec(model=TurnDecisionPayload),
    "tool.telemetry": TopicSpec(model=ToolTelemetryPayload),
    "model.request": TopicSpec(model=ModelRequestPayload),
    "routing.decision": TopicSpec(model=RoutingDecisionPayload),
    "routing.forced": TopicSpec(model=RoutingForcedPayload),
}


def payload_fields(topic: str) -> frozenset[str]:
    """Return the payload field names declared for ``topic``.

    Unknown topics declare no payload fields; :func:`allowed_fields` unions in
    the envelope, which is what the handler uses.
    """
    spec = TOPIC_SPECS.get(topic)
    if spec is None:
        return frozenset()
    return frozenset(spec.model.model_fields)


def allowed_fields(topic: str) -> frozenset[str]:
    """Return every key the ledger may persist for ``topic``."""
    return payload_fields(topic) | ENVELOPE_FIELDS


def is_known_topic(topic: str) -> bool:
    """Return whether ``topic`` has a declared payload contract."""
    return topic in TOPIC_SPECS


def drop_unknown_fields(topic: str, record: dict[str, Any]) -> dict[str, Any]:
    """Return ``record`` restricted to ``topic``'s allowlist.

    Explicit ``None`` values are kept: on this ledger a null ``turn_index`` is
    the meaningful statement "this row belongs to no turn", not an absent
    field, so it must survive the filter (unlike the token ledger, whose
    serializer omits nulls outright).

    An undeclared topic is returned unchanged — see the module docstring.
    """
    if not is_known_topic(topic):
        return dict(record)
    allowed = allowed_fields(topic)
    return {key: value for key, value in record.items() if key in allowed}


def stamp_envelope(topic: str, record: dict[str, Any]) -> dict[str, Any]:
    """Fill the envelope keys ``record`` is missing, in place, and return it.

    The handler calls this so every row carries a complete envelope even when
    the publisher is a plain dict that stamped only its payload fields. A
    publisher-supplied value always wins: this never overwrites.
    """
    spec = TOPIC_SPECS.get(topic)
    if spec is not None:
        record.setdefault("schema_version", spec.schema_version)
    record.setdefault("emitted_at", _now_iso())
    for field in ("turn_id", "run_ref", "trace_id", "span_id"):
        record.setdefault(field, None)
    return record


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def topic_schema(topic: str) -> dict[str, Any]:
    """Return the JSON Schema for a persisted row of ``topic``.

    Pydantic generates the payload schema (it already knows how to render
    ``X | None``, arrays, and objects), and this function adds the envelope the
    handler stamps.

    Only the handler-guaranteed keys are ``required``: the append stamps and
    the envelope values it fills with explicit nulls. Payload fields are typed
    but optional, because a publisher may legitimately omit one (a decision
    with no resolved ``call_id`` on a legacy row) and the handler does not
    invent it. Requiring them would make the schema reject rows the handler
    really writes.
    """
    spec = TOPIC_SPECS[topic]
    payload = spec.model.model_json_schema(mode="validation")
    properties: dict[str, Any] = dict(payload.get("properties", {}))
    for name in sorted(PUBLISHER_ENVELOPE_FIELDS):
        properties.setdefault(name, _envelope_property(name))
    properties["topic"] = {"const": topic}
    properties["logged_at"] = {"type": ["number", "string"]}
    properties["sequence"] = {"type": "integer"}
    properties["ledger_instance"] = {"type": "string"}
    return {
        "$schema": SCHEMA_DIALECT,
        "additionalProperties": False,
        "properties": dict(sorted(properties.items())),
        "required": sorted(_HANDLER_GUARANTEED_FIELDS),
        "title": topic,
        "type": "object",
    }


def _envelope_property(name: str) -> dict[str, Any]:
    """Return the schema fragment for one non-payload envelope field."""
    if name == "schema_version":
        return {"type": "integer"}
    if name == "emitted_at":
        return {"type": "string"}
    if name == "turn_index":
        # The turn join key is a count, and null states "no turn" explicitly.
        return {"type": ["integer", "null"]}
    return {"type": ["string", "null"]}


def schema_documents() -> dict[str, dict[str, Any]]:
    """Return each topic's committed JSON Schema, keyed by topic."""
    return {topic: topic_schema(topic) for topic in sorted(TOPIC_SPECS)}


def schema_filename(topic: str) -> str:
    """Return the committed file name for one topic's schema."""
    return f"{topic}.schema.json"


def render_schema(topic: str) -> str:
    """Render one topic's schema as the exact bytes committed to the repo."""
    return json.dumps(topic_schema(topic), indent=2, sort_keys=True) + "\n"


__all__ = [
    "ENVELOPE_FIELDS",
    "HANDLER_STAMPED_FIELDS",
    "PUBLISHER_ENVELOPE_FIELDS",
    "SCHEMA_DIALECT",
    "TOPIC_SPECS",
    "ModelRequestPayload",
    "RoutingDecisionPayload",
    "RoutingForcedPayload",
    "ToolTelemetryPayload",
    "TopicSpec",
    "TurnDecisionPayload",
    "allowed_fields",
    "drop_unknown_fields",
    "is_known_topic",
    "payload_fields",
    "render_schema",
    "schema_documents",
    "schema_filename",
    "stamp_envelope",
    "topic_schema",
]
