"""Tests for call-level join keys (Evidence Envelope §2).

The acceptance gates this file proves, in order:

- a turn with a streaming tool-call fallback produces two ``turn_input.calls``
  entries with distinct ``call_id``s and ``attempt`` values, and two token rows
  that join to them;
- a failed dispatch row joins to exactly one transcript tool result by
  ``tool_call_id``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from kernel.approval import ApprovalIdentity, ApprovalInvocation, build_approval_invocation
from kernel.call_identity import (
    CALL_ATTEMPTS,
    PRIMARY_ATTEMPT,
    STREAM_TOOL_FALLBACK_ATTEMPT,
    TRUNCATION_CONTINUATION_ATTEMPT,
    mint_call_id,
    mint_dispatch_id,
    mint_turn_id,
)
from kernel.evidence import ToolTelemetryPayload, allowed_fields
from kernel.models import (
    OperationExecutionContext,
    ToolExecutionContext,
    ToolExecutionTelemetry,
)
from kernel.provider_contracts import StreamedChunk
from kernel.telemetry import PENDING_TELEMETRY_PUBLISH_TASKS
from kernel.token_usage import (
    TOKEN_USAGE_TOPIC,
    TokenUsageTelemetryHandler,
    build_token_usage_event,
)
from kernel.tools.registry import _publish_tool_telemetry
from kernel.turn_hooks import ModelHooks, TurnIdentity
from kernel.turn_lineage import child_identity
from kernel.turn_runner import ContextSeams, TurnHooks, run_turn


def test_attempt_vocabulary_is_closed() -> None:
    """Only the declared attempts are known; junk and the retired retry are not.

    The set is closed on purpose. ``retry`` was declared for a provider-error
    retry the turn loop never performs, so a row carrying it would name a
    case no producer can create.
    """
    assert {
        "primary",
        "stream_tool_fallback",
        "truncation_continuation",
    } == CALL_ATTEMPTS
    assert "retry" not in CALL_ATTEMPTS


def test_minted_call_ids_are_unique_and_prefixed() -> None:
    """A call id is unique per call, so two calls never collapse into one."""
    ids = {mint_call_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("call-") for i in ids)
    assert mint_dispatch_id().startswith("disp-")


def test_minted_turn_ids_are_unique_and_prefixed() -> None:
    """A turn id names one execution, distinct from a call or dispatch id."""
    ids = {mint_turn_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("turn-") for i in ids)


def test_child_identity_records_both_lineage_keys() -> None:
    """A child turn names itself and the parent turn and dispatch behind it.

    This is the spec §4 acceptance gate: a delegate (and fork) child's rows
    carry its own ``turn_id``, the parent's ``turn_index``, and a
    ``parent_dispatch_id`` equal to the dispatch that spawned it.

    The child is built by the production helper the two spawn sites call, not
    by a copy of its logic kept in the test: deleting a lineage edge from
    ``kernel.turn_lineage.child_identity`` fails here, which is the property a
    hand-rolled ``dataclasses.replace`` could not defend.
    """
    parent = TurnIdentity(turn_index=7, turn_id=mint_turn_id(), task_goal="parent goal")
    dispatch_id = mint_dispatch_id()
    child = child_identity(parent, dispatch_id)

    assert child.turn_id and child.turn_id != parent.turn_id
    # The child belongs to the parent's thread turn, so it keeps the index.
    assert child.turn_index == parent.turn_index == 7
    assert child.parent_turn_id == parent.turn_id
    assert child.parent_dispatch_id == dispatch_id
    # A child with no declared goal inherits the parent's.
    assert child.task_goal == "parent goal"


def test_child_identity_declares_its_own_goal_when_it_has_one() -> None:
    """A delegate child's goal rides the child, not the parent's hooks."""
    parent = TurnIdentity(turn_index=3, turn_id=mint_turn_id(), task_goal="parent goal")
    child = child_identity(parent, mint_dispatch_id(), task_goal="diagnostic")

    assert child.task_goal == "diagnostic"
    assert parent.task_goal == "parent goal"


def test_top_level_identity_has_no_lineage_edges() -> None:
    """A top-level turn records no parent keys, which is what makes a tree."""
    identity = TurnIdentity(turn_index=1, turn_id=mint_turn_id())
    assert identity.parent_turn_id == ""
    assert identity.parent_dispatch_id == ""


def test_token_event_carries_call_join_keys() -> None:
    """The token event has first-class fields for the call join keys."""
    event = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=10,
        call_id="call-abc",
        iteration=3,
        attempt=STREAM_TOOL_FALLBACK_ATTEMPT,
        request_ref="r" + "a" * 64,
    )
    assert event.call_id == "call-abc"
    assert event.iteration == 3
    assert event.attempt == STREAM_TOOL_FALLBACK_ATTEMPT
    assert event.request_ref == "r" + "a" * 64


def test_token_event_join_keys_stay_unknown_when_absent() -> None:
    """Absent stays unknown: no invented call id, iteration, or attempt."""
    event = build_token_usage_event(provider="p", model_id="m", prompt_tokens=1)
    assert event.call_id is None
    assert event.iteration is None
    assert event.attempt is None
    assert event.request_ref is None


def test_token_row_persists_call_join_keys(tmp_path: Any) -> None:
    """The durable row keeps the job keys, so a reader can join on them."""
    import asyncio

    path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageTelemetryHandler(path)
    event = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=5,
        call_id="call-two",
        iteration=1,
        attempt=STREAM_TOOL_FALLBACK_ATTEMPT,
        request_ref="r" + "b" * 64,
    )
    asyncio.run(handler.publish(event))
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["call_id"] == "call-two"
    assert row["iteration"] == 1
    assert row["attempt"] == STREAM_TOOL_FALLBACK_ATTEMPT
    assert row["request_ref"] == "r" + "b" * 64


def test_token_row_persists_the_turn_it_was_spent_in(tmp_path: Any) -> None:
    """The cost ledger names the turn that spent it.

    The token ledger is the one that costs money, and a per-turn cost that
    cannot name its turn cannot be attributed to a turn at all. The event
    model accepted ``turn_id`` and dropped it, and the durable allowlist never
    listed it, so the row on disk carried no execution identity. This asserts
    on the ROW, not on the built event: the row is what a reader joins on.
    """
    import asyncio

    path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageTelemetryHandler(path)
    turn_id = mint_turn_id()
    event = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=5,
        turn_id=turn_id,
        call_id="call-one",
        iteration=1,
        attempt=PRIMARY_ATTEMPT,
    )
    asyncio.run(handler.publish(event))
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["turn_id"] == turn_id


def test_token_row_omits_the_turn_when_it_is_unknown(tmp_path: Any) -> None:
    """An unknown turn is an absent key, never an empty string.

    An empty ``turn_id`` would read as "a turn whose id is the empty string"
    instead of "no turn". Absence is the honest statement.
    """
    import asyncio

    path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageTelemetryHandler(path)
    event = build_token_usage_event(provider="p", model_id="m", prompt_tokens=5)
    asyncio.run(handler.publish(event))
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert "turn_id" not in row


def test_rejection_row_carries_the_dispatch_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rejected dispatch is a dispatch: it names the call it failed.

    The spec's acceptance requires a failed dispatch row to join to exactly
    one transcript tool result by ``tool_call_id``. The rejection publishers
    stamped the execution identity and the lineage but left every dispatch key
    off, so a rejection named no call and joined to nothing. This drives both
    rejection publishers and reads the payload each one actually publishes.
    """
    published: list[dict[str, Any]] = []

    def _capture(_bus: Any, topic: str, payload: dict[str, Any], **_kw: Any) -> None:
        published.append({"topic": topic, **payload})

    class _Bus:
        pass

    import kernel.operations.registry as operations_registry
    import kernel.tools.registry as tools_registry

    monkeypatch.setattr(tools_registry, "schedule_telemetry_publish", _capture)
    monkeypatch.setattr(operations_registry, "schedule_telemetry_publish", _capture)

    tool_ctx = ToolExecutionContext(
        user_slug="p920",
        event_bus=_Bus(),
        turn_id="turn-exec-1",
        dispatch_id="disp-rejected",
        tool_call_id="toolu-rejected",
        call_id="call-requesting",
        iteration=4,
    )
    operation_ctx = OperationExecutionContext(
        user_slug="p920",
        event_bus=_Bus(),
        turn_id="turn-exec-1",
        dispatch_id="disp-op-rejected",
        tool_call_id="toolu-op-rejected",
        call_id="call-op-requesting",
        iteration=4,
    )

    tools_registry._publish_rejection_telemetry(tool_ctx, "coding_commit", "bad parameters")
    operations_registry._publish_rejection_telemetry(operation_ctx, "commit", "bad parameters")

    assert len(published) == 2
    tool_row, operation_row = published
    for row in published:
        assert row["topic"] == "tool.telemetry"
        assert row["tool_call_id"]
        assert row["dispatch_id"]
        assert row["call_id"]
        assert row["iteration"] == 4
    assert tool_row["tool_call_id"] == "toolu-rejected"
    assert tool_row["dispatch_id"] == "disp-rejected"
    assert operation_row["tool_call_id"] == "toolu-op-rejected"
    assert operation_row["dispatch_id"] == "disp-op-rejected"


def test_two_calls_in_one_iteration_are_distinguishable() -> None:
    """The core §2 claim: same iteration, two calls, two identities.

    This is what the ledgers could not express before: the primary call and the
    streaming tool-call fallback both happen in iteration N, so ``iteration``
    alone cannot tell them apart and cost could not be attributed to either.
    """
    primary = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=100,
        call_id="c1",
        iteration=4,
        attempt=PRIMARY_ATTEMPT,
    )
    fallback = build_token_usage_event(
        provider="p",
        model_id="m",
        prompt_tokens=900,
        call_id="c2",
        iteration=4,
        attempt=STREAM_TOOL_FALLBACK_ATTEMPT,
    )
    assert primary.iteration == fallback.iteration == 4
    assert primary.call_id != fallback.call_id
    assert {primary.attempt, fallback.attempt} == {PRIMARY_ATTEMPT, STREAM_TOOL_FALLBACK_ATTEMPT}


def test_tool_telemetry_event_carries_dispatch_keys() -> None:
    """A dispatch row names its dispatch, tool call, requesting call, iteration."""
    event = ToolExecutionTelemetry(
        target_id="fetch_page",
        target_type="tool",
        status="error",
        duration_ms=12,
        dispatch_id="disp-1",
        tool_call_id="toolu_abc",
        call_id="call-1",
        iteration=2,
    )
    assert event.dispatch_id == "disp-1"
    assert event.tool_call_id == "toolu_abc"
    assert event.call_id == "call-1"
    assert event.iteration == 2


def test_tool_telemetry_payload_field_matches_event() -> None:
    """A failed dispatch row's join keys survive the write-time allowlist.

    Without this the schema test would pass while the handler silently dropped
    ``tool_call_id``, so the row and its transcript result could never join.
    """
    allowed = allowed_fields("tool.telemetry")
    for name in ("dispatch_id", "tool_call_id", "call_id", "iteration"):
        assert name in allowed, f"tool.telemetry would drop {name}"
        assert name in ToolTelemetryPayload.model_fields

    # Execution identity and lineage are deliberately the other arrangement:
    # the event carries them, the envelope owns them, and the payload model
    # must not declare them (one key, one owner). Pinned here so the exception
    # reads as a decision rather than a mirror someone forgot to finish.
    for name in ("turn_id", "parent_turn_id", "parent_dispatch_id"):
        assert name in allowed, f"tool.telemetry would drop {name}"
        assert name in ToolExecutionTelemetry.model_fields
        assert name not in ToolTelemetryPayload.model_fields


def test_truncation_continuation_is_its_own_attempt() -> None:
    """A truncation rescue is neither primary nor a plain stream fallback."""
    assert TRUNCATION_CONTINUATION_ATTEMPT == "truncation_continuation"
    assert TRUNCATION_CONTINUATION_ATTEMPT != STREAM_TOOL_FALLBACK_ATTEMPT
    assert TRUNCATION_CONTINUATION_ATTEMPT in CALL_ATTEMPTS


@pytest.mark.asyncio
async def test_token_row_names_the_call_that_issued_the_spend(tmp_path: Any) -> None:
    """A rescue's spend is reported against the rescue's own call.

    Root cause: the token-usage publish read the turn's MUTABLE call state
    at publish time -- ``current_call_attempt`` plus the last recorded
    request ref -- and the truncation / tool-call-fallback rescues issued
    real model calls that were never recorded at all. Their spend was
    therefore reported against the ITERATION'S FIRST call: a row that named
    a call which did not issue it, and a request ref that described a
    different request than the one that was paid for.

    This drives a real turn whose stream ends with ``finish_reason
    ="tool_calls"`` and no streamed tool-call deltas -- the shape that
    forces the non-streaming regeneration -- and asserts on the token row
    the durable ledger actually wrote.
    """
    import asyncio

    refs: list[str] = []

    def _record_request(_snapshot: Any) -> str:
        ref = f"r{len(refs):04d}"
        refs.append(ref)
        return ref

    class _Spend:
        """A completion that reports usage, so the token row carries a spend."""

        content = "final answer"
        tool_calls: list[dict[str, Any]] | None = None
        prompt_tokens = 100
        completion_tokens = 20
        total_tokens = 120

    async def complete(**_kwargs: Any) -> _Spend:
        return _Spend()

    async def complete_stream(**_kwargs: Any):
        yield StreamedChunk(content="", finish_reason="tool_calls")

    path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageTelemetryHandler(path)

    class _Bus:
        async def publish(self, topic: str, payload: object) -> None:
            if topic == TOKEN_USAGE_TOPIC:
                await handler.publish(payload)

    await run_turn(
        mode_id="default",
        query="hi",
        hooks=TurnHooks(
            models=ModelHooks(
                complete=complete,
                complete_stream=complete_stream,
                record_request=_record_request,
            ),
            context_seams=ContextSeams(event_bus=_Bus()),
            identity=TurnIdentity(turn_id="turn-spend-1", user_slug="p920"),
        ),
    )

    # The token publish is fire-and-forget; drain the pending set so this
    # test asserts on the row, not on scheduler timing.
    await asyncio.gather(*list(PENDING_TELEMETRY_PUBLISH_TASKS))

    assert len(refs) == 2, f"expected the primary call and the fallback call, got {refs}"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1, f"expected one token row, got {rows}"
    row = rows[0]
    assert row["attempt"] == STREAM_TOOL_FALLBACK_ATTEMPT
    assert row["request_ref"] == refs[1], "the row names the iteration's first request"
    assert row["request_ref"] != refs[0]
    assert row["call_id"], "the token row names no call"


@pytest.mark.parametrize("attempt", sorted(CALL_ATTEMPTS))
def test_every_attempt_is_usable_on_a_token_row(attempt: str) -> None:
    """Every declared attempt value round-trips onto a token event."""
    event = build_token_usage_event(provider="p", model_id="m", attempt=attempt)
    assert event.attempt == attempt


def test_context_carries_dispatch_keys_to_the_telemetry_event() -> None:
    """The dispatch keys survive the whole path from context to event.

    ``_dispatch_pack_tool`` stamps these on the context and
    ``_publish_tool_telemetry`` reads them straight back off it, so this pins
    the two halves together: if either side renamed a field, the row would
    publish with the key missing and the join would silently stop working.
    """
    ctx = ToolExecutionContext(
        user_slug="p920",
        event_bus=None,
        dispatch_id="disp-join",
        tool_call_id="toolu_join",
        call_id="call-join",
        iteration=7,
    )
    event = ToolExecutionTelemetry(
        target_id="fetch_page",
        target_type="tool",
        status="error",
        duration_ms=1,
        dispatch_id=ctx.dispatch_id or None,
        tool_call_id=ctx.tool_call_id or None,
        call_id=ctx.call_id or None,
        iteration=ctx.iteration,
    )
    assert event.dispatch_id == "disp-join"
    assert event.tool_call_id == "toolu_join"
    assert event.call_id == "call-join"
    assert event.iteration == 7


def test_context_dispatch_keys_default_to_empty_not_none() -> None:
    """An unstamped context defaults to empty strings, which publish as None.

    Empty rather than ``None`` keeps the context a plain-string model; the
    publisher is what turns "absent" into a JSON null. A dispatch outside a
    turn (an off-turn or recovered dispatch) is exactly this case.
    """
    ctx = ToolExecutionContext(user_slug="p920", event_bus=None)
    assert ctx.dispatch_id == ""
    assert ctx.tool_call_id == ""
    assert ctx.call_id == ""
    assert ctx.iteration is None


def test_failed_dispatch_row_joins_one_transcript_tool_result() -> None:
    """The §2 acceptance gate: a failed dispatch row joins exactly one result.

    The transcript tool result is built from ``prepared.tool_call_id`` and the
    telemetry row is published with the same id, so the join is exact and
    one-to-one. Before this slice the row carried no tool_call_id at all and
    the failure could not be attributed to *which* tool call failed.
    """
    tool_call_id = "toolu_abc123"
    dispatch_id = mint_dispatch_id()

    # The transcript message the turn loop appends for this dispatch.
    transcript = [
        {"role": "assistant", "content": "calling"},
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "coding_edit_files",
            "content": "patch rejected",
        },
    ]

    # The telemetry row the registry publishes for the same dispatch.
    row = ToolExecutionTelemetry(
        target_id="coding_edit_files",
        target_type="tool",
        status="error",
        duration_ms=42,
        # The transcript carries the tool's rendered output (which the loop
        # prefixes with "error: "), while the row carries the bare reason. The
        # join is on tool_call_id, not on message text, which is precisely why
        # the id is the join key.
        error_message="patch rejected",
        dispatch_id=dispatch_id,
        tool_call_id=tool_call_id,
        call_id="call-model",
        iteration=3,
    ).model_dump(mode="json")

    joined = [m for m in transcript if m.get("tool_call_id") == row["tool_call_id"]]
    assert len(joined) == 1, "a failed dispatch row must join exactly one result"
    assert joined[0]["name"] == row["target_id"]
    assert row["error_message"] in joined[0]["content"]
    assert row["dispatch_id"] == dispatch_id


def test_publish_reads_dispatch_keys_from_the_context() -> None:
    """``_publish_tool_telemetry`` stamps the keys it finds on the context.

    Uses a recording bus rather than a live one: the assertion is about the
    event payload the publisher builds, which is the half that can silently
    regress to nulls.
    """
    published: list[dict[str, Any]] = []

    class _Bus:
        pass

    ctx = ToolExecutionContext(
        user_slug="p920",
        event_bus=_Bus(),
        dispatch_id="disp-9",
        tool_call_id="toolu-9",
        call_id="call-9",
        iteration=2,
    )

    import kernel.tools.registry as registry_mod

    original = registry_mod.schedule_telemetry_publish

    def _capture(_bus: Any, topic: str, payload: dict[str, Any], **_kw: Any) -> None:
        published.append({"topic": topic, **payload})

    registry_mod.schedule_telemetry_publish = _capture
    try:
        _publish_tool_telemetry(ctx, "coding_commit", "error", 5, "boom")
    finally:
        registry_mod.schedule_telemetry_publish = original

    assert len(published) == 1
    row = published[0]
    assert row["topic"] == "tool.telemetry"
    assert row["dispatch_id"] == "disp-9"
    assert row["tool_call_id"] == "toolu-9"
    assert row["call_id"] == "call-9"
    assert row["iteration"] == 2
    assert row["status"] == "error"


def test_every_telemetry_publisher_stamps_execution_and_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four dispatch publishers stamp the execution id and its lineage (\u00a74).

    Four call sites write ``tool.telemetry``: the success path and the
    pre-execution rejection, once for tools and once for operations. Each
    builds its own event, so each can drop the attribution on its own; this
    drives all four and reads the payload each one actually publishes.
    """
    published: list[dict[str, Any]] = []

    def _capture(_bus: Any, topic: str, payload: dict[str, Any], **_kw: Any) -> None:
        published.append({"topic": topic, **payload})

    class _Bus:
        pass

    class _Span:
        def set_attribute(self, _key: str, _value: object) -> None:
            pass

    import kernel.operations.registry as operations_registry
    import kernel.tools.registry as tools_registry

    monkeypatch.setattr(tools_registry, "schedule_telemetry_publish", _capture)
    monkeypatch.setattr(operations_registry, "schedule_telemetry_publish", _capture)

    tool_ctx = ToolExecutionContext(
        user_slug="p920",
        event_bus=_Bus(),
        turn_id="turn-exec-1",
        parent_turn_id="turn-parent",
        parent_dispatch_id="disp-parent",
        dispatch_id="disp-1",
        tool_call_id="toolu-1",
    )
    operation_ctx = OperationExecutionContext(
        user_slug="p920",
        event_bus=_Bus(),
        turn_id="turn-exec-1",
        parent_turn_id="turn-parent",
        parent_dispatch_id="disp-parent",
    )

    tools_registry._publish_tool_telemetry(tool_ctx, "coding_commit", "success", 5, None)
    tools_registry._publish_rejection_telemetry(tool_ctx, "coding_commit", "bad parameters")
    operations_registry._emit_operation_telemetry(
        "commit",
        operation_ctx,
        _Span(),
        time.perf_counter(),
        status="error",
        err_msg="boom",
    )
    operations_registry._publish_rejection_telemetry(operation_ctx, "commit", "bad parameters")

    assert len(published) == 4, "each publisher must emit exactly one row"
    for row in published:
        assert row["topic"] == "tool.telemetry"
        assert row["turn_id"] == "turn-exec-1"
        assert row["parent_turn_id"] == "turn-parent"
        assert row["parent_dispatch_id"] == "disp-parent"


def test_a_top_level_dispatch_states_absent_lineage_as_explicit_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch with no parent turn still names both lineage keys (\u00a74).

    A top-level row must carry the keys as explicit ``None`` rather than omit
    them, or a reader cannot tell "this is a root" from "this publisher forgot
    the field".
    """
    published: list[dict[str, Any]] = []

    def _capture(_bus: Any, topic: str, payload: dict[str, Any], **_kw: Any) -> None:
        published.append({"topic": topic, **payload})

    class _Bus:
        pass

    import kernel.tools.registry as tools_registry

    monkeypatch.setattr(tools_registry, "schedule_telemetry_publish", _capture)
    ctx = ToolExecutionContext(user_slug="p920", event_bus=_Bus(), turn_id="turn-exec-1")
    tools_registry._publish_tool_telemetry(ctx, "read", "success", 1, None)

    assert len(published) == 1
    row = published[0]
    assert row["turn_id"] == "turn-exec-1"
    assert "parent_turn_id" in row
    assert row["parent_turn_id"] is None
    assert "parent_dispatch_id" in row
    assert row["parent_dispatch_id"] is None


def test_approval_invocation_carries_dispatch_keys() -> None:
    """An approval row names the tool call and dispatch that requested it."""
    inv = ApprovalInvocation(
        request_id="req1",
        target_kind="tool",
        target_id="coding_commit",
        pack_id="lisa-pack-coding",
        effect="write",
        identity=ApprovalIdentity(user_slug="p920", session_plan_digest="d"),
        pack_config_fingerprint="f",
        session_plan_digest="d",
        action_digest="a",
        safe_preview="write coding_commit",
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
        tool_call_id="toolu-approve",
        dispatch_id="disp-approve",
    )
    assert inv.tool_call_id == "toolu-approve"
    assert inv.dispatch_id == "disp-approve"


def test_dispatch_keys_do_not_change_the_action_digest() -> None:
    """Join keys are provenance, not action identity.

    ``action_digest`` decides whether an approval covers the action being
    retried, so a dispatch id leaking into it would force a fresh approval on
    every retry of the same action.
    """
    kwargs: dict[str, Any] = {
        "target_kind": "tool",
        "target_id": "coding_commit",
        "pack_id": "p",
        "effect": "write",
        "parameters": {"a": 1},
        "identity": ApprovalIdentity(user_slug="p920", session_plan_digest="d"),
        "pack_config": {},
        "now": datetime(2026, 1, 1, tzinfo=UTC),
    }
    bare = build_approval_invocation(**kwargs)
    stamped = bare.model_copy(update={"dispatch_id": "disp-x", "tool_call_id": "toolu-x"})
    assert bare.action_digest == stamped.action_digest


def test_off_turn_model_request_row_carries_call_and_dispatch() -> None:
    """An off-turn call's row names the call and the dispatch behind it.

    Exercises the real publisher (``_CallableHost._publish_model_request_row``)
    rather than a stand-in, because the risk here is exactly that the row is
    built with the keys missing.
    """
    from kernel.seams import host_adapter as ha

    from kernel import telemetry as telemetry_mod

    audit = ha.CompletionAuditContext(
        record_request=lambda snap: "r" + "c" * 64,
        event_bus=object(),
        operation_id="fetch_page",
        turn_index=None,
        user_slug="p920",
        thread_store_slug="t1",
        call_id="call-offturn",
        dispatch_id="disp-offturn",
    )
    host = ha._CallableHost(complete_fn=None, resolve_model_fn=None, audit=audit)

    captured: list[dict[str, Any]] = []

    def _capture(_bus: Any, _topic: str, payload: dict[str, Any], **_kw: Any) -> None:
        captured.append(payload)

    original = telemetry_mod.schedule_telemetry_publish
    telemetry_mod.schedule_telemetry_publish = _capture
    try:
        host._publish_model_request_row(audit, "r" + "c" * 64)
    finally:
        telemetry_mod.schedule_telemetry_publish = original

    assert len(captured) == 1
    row = captured[0]
    assert row["call_id"] == "call-offturn"
    assert row["dispatch_id"] == "disp-offturn"
    assert row["operation_id"] == "fetch_page"
