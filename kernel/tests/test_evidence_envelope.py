"""Tests for the typed evidence envelope and write-time allowlists.

These pin the acceptance gate from EVIDENCE-ENVELOPE.md: every publisher's
payload validates against its exported JSON Schema, and the decision handler
drops keys no declaration names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import kernel.operations.registry as operations_registry
import kernel.tools.registry as tools_registry
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
    SCHEMA_WRITE_CONTRACT_COMMENT,
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
from kernel.models import OperationExecutionContext, ToolExecutionContext, ToolExecutionTelemetry
from kernel.telemetry import PENDING_TELEMETRY_PUBLISH_TASKS
from kernel.turn_hooks import TurnIdentity
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import _publish_routing_decision, _publish_routing_forced

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
    assert stamped["schema_version"] == 2
    assert stamped["emitted_at"]
    # Lineage the publisher omitted is filled as explicit unknown, so every
    # persisted row carries the keys the schema requires.
    assert stamped["parent_turn_id"] is None
    assert stamped["parent_dispatch_id"] is None
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
    assert record["schema_version"] == 2
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


#: The committed documents live beside the spec that explains their contract.
_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "SPEC" / "schemas"


def test_committed_schemas_match_the_generator_exactly() -> None:
    """A committed document is byte-for-byte ``render_schema(topic)``.

    Committing generated files only buys review visibility if they cannot
    drift: this test is the gate that makes the diff trustworthy.
    """
    for topic in TOPIC_SPECS:
        path = _SCHEMA_DIR / schema_filename(topic)
        assert path.is_file(), f"missing committed schema for {topic}: {path}"
        assert path.read_text(encoding="utf-8") == render_schema(topic), (
            f"{path.name} is stale; regenerate from kernel.evidence.render_schema"
        )


def test_every_schema_carries_the_write_contract_comment() -> None:
    """Each document warns, in-band, that it is not a historical validator.

    ``additionalProperties: false`` is a write-time allowlist. A future agent
    that wires these documents into the forensic reader would reject legally
    written historical rows, so the rule rides the schema itself and not only
    the neighbouring README.
    """
    for topic in TOPIC_SPECS:
        schema = topic_schema(topic)
        comment = schema.get("$comment")
        assert isinstance(comment, str) and comment, f"{topic} has no $comment"
        assert comment == SCHEMA_WRITE_CONTRACT_COMMENT
        assert "historical" in comment.lower()
        assert "must not validate" in comment.lower()


def test_schema_comment_survives_a_json_round_trip() -> None:
    """``$comment`` is inert data: it round-trips without altering validation."""
    for topic in TOPIC_SPECS:
        rendered = render_schema(topic)
        reparsed = json.loads(rendered)
        assert reparsed["$comment"] == SCHEMA_WRITE_CONTRACT_COMMENT
        # Annotation keywords never change the accept/reject decision.
        assert reparsed["additionalProperties"] is False
        assert reparsed["required"] == topic_schema(topic)["required"]


# ── Envelope contract: what actually reaches disk ──────────────────────────
# The allowlist's job is to *drop* unknown keys, so a field that is declared but
# never populated and a field that is populated but dropped are both invisible to
# a coherence check. Every identity bug found so far (a resume that split one
# execution across two ids, lineage threaded through contexts but never emitted,
# sibling publishers of which only one was wired) was a value-flow failure. These
# tests assert on the row as persisted by the real handler, not on the model.

#: A populated execution identity. Shape matches ``mint_turn_id`` output so a
#: reader joining on the prefix cannot mistake a fixture for an unset value.
_TURN_ID = "turn-" + "a" * 32
_PARENT_TURN_ID = "turn-" + "b" * 32
_PARENT_DISPATCH_ID = "dispatch-" + "c" * 32

#: The lineage a child turn must record, i.e. the edge a reader walks to render
#: the delegation tree promised by the spec.
_CHILD_LINEAGE: dict[str, object] = {
    "parent_turn_id": _PARENT_TURN_ID,
    "parent_dispatch_id": _PARENT_DISPATCH_ID,
}


def _persist_one(tmp_path: Path, topic: str, payload: dict[str, object]) -> dict[str, object]:
    """Persist one publisher payload through the real handler and read it back.

    ``DecisionTelemetryHandler._persist`` is the write path: it drops undeclared
    keys and stamps the envelope before appending. Asserting here rather than on
    ``stamp_envelope`` alone is what makes a publisher stamping ``turn_id=None``
    fail even though ``turn_id`` is declared on the model.
    """
    path = tmp_path / f"{topic}.jsonl"
    handler = DecisionTelemetryHandler(path)
    handler._persist(topic, dict(payload))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1, f"{topic} persisted {len(rows)} rows for one publish"
    return rows[0]


def _assert_populated_identity(row: dict[str, object], topic: str) -> None:
    """Assert one persisted row names its execution and its parent edge."""
    assert row["topic"] == topic
    assert row["turn_id"] == _TURN_ID, f"{topic} lost its turn_id on the way to disk"
    assert row["parent_turn_id"] == _PARENT_TURN_ID, f"{topic} lost parent_turn_id"
    assert row["parent_dispatch_id"] == _PARENT_DISPATCH_ID, f"{topic} lost parent_dispatch_id"


def test_every_topic_persists_a_populated_identity_to_disk(tmp_path: Path) -> None:
    """Every declared topic carries identity and lineage through the write path.

    Sweeps ``TOPIC_SPECS`` rather than one topic: a new topic that forgets to
    emit the envelope is exactly the failure mode this pins.
    """
    for topic in TOPIC_SPECS:
        payload = dict(_PUBLISHED_PAYLOADS[topic], turn_id=_TURN_ID, **_CHILD_LINEAGE)
        _assert_populated_identity(_persist_one(tmp_path, topic, payload), topic)


def test_a_top_level_row_states_absent_lineage_as_explicit_null(tmp_path: Path) -> None:
    """A top-level row carries its turn id and *explicit* null parent keys.

    Absent lineage must be stated as a null rather than left to absence: the
    spec's lineage queries filter on the key, and an omitted key and "no parent"
    would otherwise be indistinguishable from a row written by an older version.
    """
    for topic in TOPIC_SPECS:
        row = _persist_one(tmp_path, topic, dict(_PUBLISHED_PAYLOADS[topic], turn_id=_TURN_ID))
        assert row["turn_id"] == _TURN_ID, f"{topic} lost its turn_id on the way to disk"
        for key in _CHILD_LINEAGE:
            assert key in row, f"{topic} omitted the {key} key instead of stating it null"
            assert row[key] is None, f"{topic} invented a {key} for a top-level row"


def _capture_into(handler: DecisionTelemetryHandler) -> object:
    """Return a ``schedule_telemetry_publish`` stand-in that writes via ``handler``.

    Swapping the scheduler for a direct write keeps the real publisher (which
    builds the payload) and the real handler (which filters it), and drops only
    the fire-and-forget task, whose result a test cannot otherwise await.
    """

    def _capture(_bus: object, topic: str, payload: object, **_kwargs: object) -> None:
        handler._persist(topic, payload)

    return _capture


def _identity_context(cls: type, **extra: object) -> object:
    """Build a dispatch context whose identity is populated, as a turn wires it."""
    return cls(
        user_slug="local-user",
        thread_store_slug="local-user--conv1",
        turn_index=7,
        turn_id=_TURN_ID,
        event_bus=object(),
        **_CHILD_LINEAGE,
        **extra,
    )


def test_every_tool_telemetry_publisher_persists_a_populated_identity(
    tmp_path: Path,
) -> None:
    """Both tool-registry publishers and both operation-registry ones are wired.

    ``tool.telemetry`` has four sibling publishers. Slice 4A found that only one
    of them stamped the identity, so enumerating them is the point: sampling a
    single publisher would have passed while three rows per turn lost their
    execution.
    """
    handler = DecisionTelemetryHandler(tmp_path / "decisions.jsonl")
    tool_ctx = _identity_context(
        ToolExecutionContext, tool_id="coding_commit", normalized_fields=["path"]
    )
    operation_ctx = _identity_context(OperationExecutionContext)
    span = SimpleNamespace(set_attribute=lambda *a, **k: None)

    with (
        patch("kernel.tools.registry.schedule_telemetry_publish", _capture_into(handler)),
        patch("kernel.operations.registry.schedule_telemetry_publish", _capture_into(handler)),
    ):
        tools_registry._publish_tool_telemetry(tool_ctx, "coding_commit", "success", 5, None)
        tools_registry._publish_rejection_telemetry(tool_ctx, "coding_commit", "bad parameters")
        operations_registry._emit_operation_telemetry(
            "commit", operation_ctx, span, time.perf_counter(), status="success", err_msg=None
        )
        operations_registry._publish_rejection_telemetry(operation_ctx, "commit", "bad parameters")

    rows = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 4, f"expected one row per publisher, got {len(rows)}"
    for row in rows:
        _assert_populated_identity(row, TOOL_TELEMETRY_TOPIC)


@pytest.mark.asyncio
async def test_routing_publishers_persist_a_populated_identity(tmp_path: Path) -> None:
    """Both turn-runner routing publishers stamp identity and lineage.

    ``routing.forced`` is the enforcer's accountability record; without the
    execution id a reader cannot tell which turn was overridden, and without the
    parent keys a child turn's overrides detach from the delegation tree.
    """
    handler = DecisionTelemetryHandler(tmp_path / "decisions.jsonl")

    class _PersistBus:
        async def publish(self, topic: str, payload: object) -> None:
            handler._persist(topic, payload)

    hooks = SimpleNamespace(
        context_seams=SimpleNamespace(event_bus=_PersistBus()),
        identity=TurnIdentity(
            turn_id=_TURN_ID,
            turn_index=7,
            user_slug="local-user",
            thread_store_slug="local-user--conv1",
            **_CHILD_LINEAGE,
        ),
    )
    state = _TurnState()
    state.iteration = 2
    state.effective_model = SimpleNamespace(
        model_id="z-ai/glm-5.3-flashx",
        cost_tier="standard",
        reasoning_model=False,
        structured_reliability=0.5,
    )
    log = logging.getLogger(__name__)

    await _publish_routing_decision(
        hooks=hooks,
        state=state,
        mode_id="default",
        query="q",
        hint_text="",
        suppressed=False,
        forced_kind="",
        forced_targets=[],
        forced_exempt=False,
        log=log,
    )
    _publish_routing_forced(
        hooks=hooks,
        state=state,
        mode_id="default",
        forced_kind="transition",
        forced_targets=["planner"],
        non_compliant_dispatches=2,
        trigger="dispatch_override",
        log=log,
    )

    # ``routing.forced`` publishes fire-and-forget, so the row exists only once
    # the scheduled task has run. Draining the module's pending set keeps this
    # test about the payload rather than about scheduler timing.

    await asyncio.gather(*list(PENDING_TELEMETRY_PUBLISH_TASKS))

    rows = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_topic = {row["topic"]: row for row in rows}
    for topic in (ROUTING_DECISION_TOPIC, ROUTING_FORCED_TOPIC):
        assert topic in by_topic, f"{topic} published no row"
        _assert_populated_identity(by_topic[topic], topic)
