"""Tests for the typed evidence envelope and write-time allowlists.

These pin the acceptance gate from EVIDENCE-ENVELOPE.md: every publisher's
payload validates against its exported JSON Schema, and the decision handler
drops keys no declaration names.
"""

from __future__ import annotations

import json

import pytest
from kernel.decision_telemetry import (
    ROUTING_DECISION_TOPIC,
    ROUTING_FORCED_TOPIC,
    TOOL_TELEMETRY_TOPIC,
    TURN_DECISION_TOPIC,
    DecisionTelemetryHandler,
)
from kernel.evidence import (
    ENVELOPE_FIELDS,
    TOPIC_SPECS,
    allowed_fields,
    drop_unknown_fields,
    is_known_topic,
    payload_fields,
    render_schema,
    schema_documents,
    schema_filename,
    stamp_envelope,
    topic_schema,
)
from kernel.models import ToolExecutionTelemetry

#: Real payloads exactly as the publishers emit them, so the schema pin covers
#: the shipped shape rather than a hand-written approximation of it.
_PUBLISHED_PAYLOADS: dict[str, dict[str, object]] = {
    TURN_DECISION_TOPIC: {
        "mode_id": "coding_design",
        "turn_index": 42,
        "user_slug": "local-user",
        "thread_store_slug": "local-user--conv1",
        "iteration": 3,
        "turns_in_stage": 1,
        "action_type": "ContinueAction",
        "action_reason": "tool calls remain",
        "total_tool_calls": 4,
        "write_receipts": 1,
        "read_receipts": 2,
    },
    TOOL_TELEMETRY_TOPIC: ToolExecutionTelemetry(
        target_id="fetch_page",
        target_type="tool",
        status="error",
        duration_ms=812,
        error_message="timeout",
        turn_index=17,
        user_slug="local-user",
        thread_store_slug="local-user--conv1",
        normalized_fields=["url"],
    ).model_dump(mode="json"),
    "model.request": {
        "operation_id": "mine_thread_segment",
        "request_ref": "r" + "e" * 64,
        "turn_index": None,
        "user_slug": "local-user",
        "thread_store_slug": "local-user",
    },
    ROUTING_DECISION_TOPIC: {
        "mode_id": "default",
        "query_preview": "what is the status",
        "turn_index": 11,
        "user_slug": "local-user",
        "thread_store_slug": "local-user",
        "iteration": 1,
        "turn": 1,
        "hint_kind": "handoff",
        "hint_targets": ["coding_design"],
        "hint_injected": True,
        "hint_suppressed": False,
        "forced_armed": True,
        "forced_kind": "transition",
        "forced_targets": ["coding_design"],
        "forced_exempt": False,
        "model_id": "z-ai/glm-5.3-flashx",
        "model_cost_tier": "standard",
        "model_reasoning": False,
        "model_structured_reliability": 0.5,
    },
    ROUTING_FORCED_TOPIC: {
        "mode_id": "default",
        "turn_index": 7,
        "user_slug": "local-user",
        "thread_store_slug": "local-user",
        "iteration": 4,
        "kind": "transition",
        "targets": ["planner"],
        "non_compliant_dispatches": 2,
        "trigger": "dispatch_override",
        "expected_tool": "transition_stage",
        "model_id": "z-ai/glm-flash-latest",
        "model_cost_tier": "budget",
        "model_reasoning": False,
        "model_structured_reliability": 0.0,
    },
}


def _persisted_row(topic: str, payload: dict[str, object]) -> dict[str, object]:
    """Return exactly what the handler writes for one publisher payload."""
    row = stamp_envelope(topic, dict(payload))
    row["topic"] = topic
    row["logged_at"] = 1.0
    row["sequence"] = 1
    row["ledger_instance"] = "abc123"
    return row


def test_every_published_payload_key_is_allowlisted() -> None:
    """No real publisher field is dropped by its topic's own allowlist."""
    for topic, payload in _PUBLISHED_PAYLOADS.items():
        unknown = sorted(set(payload) - allowed_fields(topic))
        assert not unknown, f"{topic} would drop real publisher field(s): {unknown}"


def test_payload_matches_committed_schema() -> None:
    """Every publisher payload validates against its exported JSON Schema."""
    jsonschema = pytest.importorskip("jsonschema")
    for topic, payload in _PUBLISHED_PAYLOADS.items():
        jsonschema.validate(instance=_persisted_row(topic, payload), schema=topic_schema(topic))


def test_rendered_schema_is_stable_bytes() -> None:
    """The committed schema text is deterministic, so regeneration cannot churn."""
    for topic in TOPIC_SPECS:
        assert render_schema(topic) == render_schema(topic)
        assert render_schema(topic).endswith("\n")


def test_schema_rejects_content_shaped_keys() -> None:
    """A prompt-carrying row is invalid: the schema closes the object."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = topic_schema(TURN_DECISION_TOPIC)
    assert schema["additionalProperties"] is False
    row = _persisted_row(TURN_DECISION_TOPIC, {"mode_id": "m", "prompt_text": "secret"})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=row, schema=schema)


def test_drop_unknown_fields_removes_undeclared_keys() -> None:
    """The allowlist keeps declared keys and drops everything else."""
    kept = drop_unknown_fields(
        TURN_DECISION_TOPIC,
        {"mode_id": "m", "action_type": "ContinueAction", "prompt_text": "leak"},
    )
    assert kept == {"mode_id": "m", "action_type": "ContinueAction"}


def test_drop_unknown_fields_preserves_explicit_null() -> None:
    """A null ``turn_index`` is meaningful on this ledger and must survive."""
    kept = drop_unknown_fields("model.request", {"turn_index": None, "operation_id": "op"})
    assert kept == {"turn_index": None, "operation_id": "op"}


def test_drop_unknown_fields_passes_undeclared_topic_through() -> None:
    """An undeclared topic keeps its keys; losing all evidence is worse."""
    record = {"anything": 1, "else": "kept"}
    assert drop_unknown_fields("pack.custom", dict(record)) == record
    assert not is_known_topic("pack.custom")


def test_payload_fields_leave_envelope_keys_to_the_envelope() -> None:
    """Payload declarations never shadow an envelope field.

    Both sets contribute to the allowlist, so an overlap would make the file
    claim two owners for one key; the envelope is the single owner.
    """
    for topic in TOPIC_SPECS:
        assert payload_fields(topic).isdisjoint(ENVELOPE_FIELDS), (
            f"{topic} declares an envelope field as payload"
        )


def test_stamp_envelope_fills_unknowns_without_overwriting() -> None:
    """The handler adds the envelope, and a publisher value always wins."""
    stamped = stamp_envelope(
        TURN_DECISION_TOPIC, {"mode_id": "m", "turn_id": "t-1", "run_ref": "r" + "a" * 64}
    )
    assert stamped["turn_id"] == "t-1"
    assert stamped["run_ref"] == "r" + "a" * 64
    assert stamped["schema_version"] == 1
    assert stamped["emitted_at"]
    # Reserved trace slots exist as explicit unknown until trace export lands.
    assert stamped["trace_id"] is None
    assert stamped["span_id"] is None


@pytest.mark.asyncio
async def test_handler_drops_unknown_keys_at_write(tmp_path) -> None:
    """The decision handler no longer persists arbitrary mappings."""
    path = tmp_path / "decisions.jsonl"
    handler = DecisionTelemetryHandler(path)
    await handler.publish_turn_decision(
        {
            "mode_id": "coding_design",
            "action_type": "ContinueAction",
            "prompt_text": "do not persist me",
            "tool_arguments": {"secret": 1},
        }
    )
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["mode_id"] == "coding_design"
    assert "prompt_text" not in record
    assert "tool_arguments" not in record
    assert record["schema_version"] == 1
    assert record["topic"] == TURN_DECISION_TOPIC
    assert record["turn_id"] is None


@pytest.mark.asyncio
async def test_handler_keeps_undeclared_topic_unfiltered(tmp_path) -> None:
    """A topic with no declaration is persisted unchanged, never emptied."""
    path = tmp_path / "decisions.jsonl"
    handler = DecisionTelemetryHandler(path)
    handler._persist("pack.custom", {"mode_id": "m", "custom_field": 7})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["custom_field"] == 7


def test_schema_documents_cover_every_topic() -> None:
    """Every declared topic exports a document named for it."""
    documents = schema_documents()
    assert set(documents) == set(TOPIC_SPECS)
    for topic, document in documents.items():
        assert document["title"] == topic
        assert document["$schema"].startswith("https://json-schema.org/")


def test_schema_filename_is_derived_from_topic() -> None:
    """The committed file name tracks the topic, so a rename cannot orphan one."""
    for topic in TOPIC_SPECS:
        assert schema_filename(topic) == f"{topic}.schema.json"
