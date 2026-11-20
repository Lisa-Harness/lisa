"""Tests for the exact-mode replay engine core (Spec 3 slice 3)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import CompletionRequest
from runtime.replay import (
    RecordedTurn,
    ReplayDivergence,
    ReplayGatewayError,
    ReplayGatewayProvider,
    ReplayReport,
    ReplayToolGapError,
    TranscriptToolAnswerer,
)
from runtime.replay._engine import (
    _compare_receipts,
    _newest_scratch_entry,
    _project_gateway_errors,
    _replay_divergences_from_gaps,
    _replay_divergences_from_provider,
    _seed_scratch_config,
)
from runtime.testing.archive import ArchivedCall, ArchiveGatewayProvider, ArchiveOutcomeError
from runtime.testing.scripted import ScriptError

# ── TranscriptToolAnswerer ────────────────────────────────────────────────


def test_answerer_returns_recorded_output_for_known_id() -> None:
    answerer = TranscriptToolAnswerer(tool_results={"call_recorded_1": "ok"})
    assert answerer.answer("t", {"x": 1}, "call_recorded_1") == "ok"
    assert answerer.gaps == []


def test_answerer_records_gap_and_raises_for_unknown_id() -> None:
    answerer = TranscriptToolAnswerer(tool_results={})
    with pytest.raises(ReplayToolGapError, match="call_recorded_1"):
        answerer.answer("t", {"x": 1}, "call_recorded_1")
    assert answerer.gaps == [{"tool": "t", "tool_call_id": "call_recorded_1"}]


def test_answerer_status_marks_error_text_and_json_errors() -> None:
    answerer = TranscriptToolAnswerer(
        tool_results={
            "text-error": "Error: provider unavailable",
            "json-error": '{"error": "provider unavailable"}',
            "malformed-json": '{"error":',
            "http-error": "HTTP 403 (Forbidden)",
            "ok": "provider answered",
        }
    )

    assert answerer.answer_with_status("t", {}, "text-error")[1] is True
    assert answerer.answer_with_status("t", {}, "json-error")[1] is True
    assert answerer.answer_with_status("t", {}, "malformed-json")[1] is False
    assert answerer.answer_with_status("t", {}, "http-error")[1] is False
    assert answerer.answer_with_status("t", {}, "ok")[1] is False


# ── Divergence projection ────────────────────────────────────────────────


def test_provider_divergences_project_onto_the_report() -> None:
    provider = ArchiveGatewayProvider(calls=[])
    provider.divergences.append(
        {"call_id": "call-1", "field": "model_id", "recorded": "a", "replayed": "b"}
    )
    divergences = _replay_divergences_from_provider(provider)
    assert len(divergences) == 1
    assert divergences[0].kind == "model_request"
    assert divergences[0].field == "model_id"
    assert divergences[0].call_id == "call-1"
    assert divergences[0].recorded == "a"
    assert divergences[0].replayed == "b"


def test_gap_divergences_project_onto_the_report() -> None:
    answerer = TranscriptToolAnswerer(tool_results={})
    answerer.gaps.append({"tool": "t", "tool_call_id": "call_recorded_1"})
    divergences = _replay_divergences_from_gaps(answerer)
    assert len(divergences) == 1
    assert divergences[0].kind == "tool_result"
    assert divergences[0].tool_call_id == "call_recorded_1"
    assert divergences[0].detail == "t"


# ── Receipts comparison ──────────────────────────────────────────────────


def test_receipts_match_structurally() -> None:
    recorded = [{"effect": "read", "capability_id": "t", "artifact_id": "", "mode_id": "default"}]
    replayed = [{"effect": "read", "capability_id": "t", "artifact_id": "", "mode_id": "replay"}]
    assert _compare_receipts(recorded, replayed) is True


def test_receipts_mismatch_on_capability_or_effect() -> None:
    recorded = [{"effect": "read", "capability_id": "t", "artifact_id": ""}]
    replayed = [{"effect": "write", "capability_id": "t", "artifact_id": ""}]
    assert _compare_receipts(recorded, replayed) is False
    replayed2 = [{"effect": "read", "capability_id": "other", "artifact_id": ""}]
    assert _compare_receipts(recorded, replayed2) is False


def test_receipts_mismatch_on_length() -> None:
    recorded = [{"effect": "read", "capability_id": "t", "artifact_id": ""}]
    assert _compare_receipts(recorded, []) is False


# ── Scratch entry discovery ──────────────────────────────────────────────


def test_newest_scratch_entry_reads_the_last_line(tmp_path: Path) -> None:
    threads = tmp_path / "threads" / "user"
    threads.mkdir(parents=True)
    active = threads / "thread.current.jsonl"
    active.write_text(
        json.dumps({"turn": 0, "text": "first"})
        + "\n"
        + json.dumps({"turn": 1, "text": "second"})
        + "\n",
        encoding="utf-8",
    )
    entry = _newest_scratch_entry(tmp_path)
    assert entry is not None
    assert entry["turn"] == 1
    assert entry["text"] == "second"


def test_newest_scratch_entry_returns_none_without_a_store(tmp_path: Path) -> None:
    assert _newest_scratch_entry(tmp_path) is None


# ── ReplayGatewayProvider routing ─────────────────────────────────────────────


def _gateway_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    call_kind: str = "",
) -> CompletionRequest:
    """One replayed gateway request, classified by its owning seam.

    ``call_kind`` mirrors the explicit origin the production seams stamp
    (``"turn"`` at the turn loop, ``"auxiliary"`` at OperationHost and the
    runtime off-turn/slice seam); the empty default is the unclassified
    request the routing provider must refuse.
    """
    return CompletionRequest(
        messages=messages,
        model_def=SimpleNamespace(provider="ollama", model_id="qwen3:8b", display_name="Qwen3 8B"),
        max_tokens=1024,
        temperature=0.0,
        tools=tools,
        call_kind=call_kind,
    )


def _archived_call(content: str, tools: list[dict[str, Any]] | None = None) -> ArchivedCall:
    return ArchivedCall(
        call_id="call-1",
        iteration=1,
        attempt="primary",
        request={
            "messages": [{"role": "user", "content": "go"}],
            "tools": tools or [],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
        response={
            "call_id": "call-1",
            "outcome": "ok",
            "content": content,
            "provider": "ollama",
            "model_id": "qwen3:8b",
        },
    )


def test_routing_provider_answers_turn_kind_from_the_turn_queue() -> None:
    """A ``call_kind="turn"`` request consumes the in-turn archive queue."""
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("turn reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    result = provider.answer(
        _gateway_request([{"role": "user", "content": "go"}], tools=tools, call_kind="turn")
    )
    assert result.content == "turn reply"
    assert provider.turn_calls.position == 1
    assert provider.auxiliary_calls.position == 0


def test_routing_provider_answers_zero_tool_turn_calls_from_the_turn_queue() -> None:
    """A zero-tool in-turn request still consumes ``turn_calls`` in order.

    Routing must not key on the tool list: a mode with an empty tool
    surface makes in-turn calls with ``tools=None``/``[]``, and a
    tools-based heuristic would misroute them onto the auxiliary queue
    and desync both. The explicit ``call_kind="turn"`` origin — not the
    request shape — selects the queue, and the auxiliary queue stays
    untouched.
    """
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(
            calls=[_archived_call("first"), _archived_call("second")]
        ),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    first = provider.answer(
        _gateway_request([{"role": "user", "content": "go"}], tools=None, call_kind="turn")
    )
    second = provider.answer(
        _gateway_request([{"role": "user", "content": "go"}], tools=[], call_kind="turn")
    )
    assert [first.content, second.content] == ["first", "second"]
    assert provider.turn_calls.position == 2
    assert provider.auxiliary_calls.position == 0


def test_routing_provider_answers_auxiliary_kind_from_the_auxiliary_queue() -> None:
    """A ``call_kind="auxiliary"`` request consumes the auxiliary queue."""
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("turn reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    result = provider.answer(
        _gateway_request([{"role": "user", "content": "go"}], call_kind="auxiliary")
    )
    assert result.content == "planner reply"
    assert provider.turn_calls.position == 0
    assert provider.auxiliary_calls.position == 1


def test_routing_provider_answers_auxiliary_requests_carrying_tools() -> None:
    """An auxiliary request with a tool surface still consumes the auxiliary queue.

    The origin — not the request shape — routes: an off-turn operation
    that passes tools (a distiller with a tool schema, say) must not be
    misrouted onto the turn-call queue by a tools-based heuristic.
    """
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("turn reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    result = provider.answer(
        _gateway_request([{"role": "user", "content": "go"}], tools=tools, call_kind="auxiliary")
    )
    assert result.content == "planner reply"
    assert provider.turn_calls.position == 0
    assert provider.auxiliary_calls.position == 1


def test_routing_provider_fails_closed_on_unclassified_requests() -> None:
    """A request with no ``call_kind`` must fail loudly, not guess.

    Silently guessing a queue would desync both archives on the first
    misroute; the routing provider refuses the request so the unclassified
    seam surfaces at its owner.
    """
    from runtime.testing.scripted import ScriptError

    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("turn reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    with pytest.raises(ScriptError, match="call_kind"):
        provider.answer(_gateway_request([{"role": "user", "content": "go"}]))
    assert provider.turn_calls.position == 0
    assert provider.auxiliary_calls.position == 0


def test_routing_provider_fails_closed_on_unknown_call_kind() -> None:
    """An unknown ``call_kind`` value must fail loudly, not guess."""
    from runtime.testing.scripted import ScriptError

    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("turn reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    with pytest.raises(ScriptError, match="call_kind"):
        provider.answer(_gateway_request([{"role": "user", "content": "go"}], call_kind="surprise"))
    assert provider.turn_calls.position == 0
    assert provider.auxiliary_calls.position == 0


def test_routing_provider_projects_both_queues_divergences() -> None:
    """Divergences from either queue land on the report in turn-first order."""
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    provider.turn_calls.divergences.append({"field": "model_id", "recorded": "a", "replayed": "b"})
    provider.auxiliary_calls.divergences.append(
        {"field": "message_count", "recorded": 1, "replayed": 2}
    )
    divergences = _replay_divergences_from_provider(provider)
    assert [d.field for d in divergences] == ["model_id", "message_count"]


# ── FIX-3: structured gateway errors ─────────────────────────────────────


def test_turn_queue_script_error_records_a_structured_gateway_error() -> None:
    """A turn-queue ``ScriptError`` records one structured gateway error.

    The kernel converts the raised error into the turn's response text, so
    without the structured record the failure would surface only as a
    response divergence with the message buried in replayed text.
    """
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[_archived_call("planner reply")]),
    )
    with pytest.raises(ScriptError, match="archive exhausted"):
        provider.answer(_gateway_request([{"role": "user", "content": "go"}], call_kind="turn"))
    assert len(provider.gateway_errors) == 1
    error = provider.gateway_errors[0]
    assert error.kind == "gateway_script_error"
    assert error.queue == "turn"
    assert "archive exhausted" in error.message
    assert error.call_id == ""


def test_auxiliary_queue_script_error_records_a_structured_gateway_error() -> None:
    """An auxiliary-queue ``ScriptError`` records the auxiliary queue's error."""
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("turn reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    with pytest.raises(ScriptError, match="archive exhausted"):
        provider.answer(
            _gateway_request([{"role": "user", "content": "go"}], call_kind="auxiliary")
        )
    assert len(provider.gateway_errors) == 1
    error = provider.gateway_errors[0]
    assert error.kind == "gateway_script_error"
    assert error.queue == "auxiliary"
    assert "archive exhausted" in error.message


def test_archive_outcome_error_is_not_a_gateway_error() -> None:
    """An intentionally recorded non-ok outcome re-raises unrecorded.

    ``ArchiveOutcomeError`` means the archived call itself failed and the
    replay must reproduce that failure — it is recorded model behavior,
    not a gateway malfunction, so it must not populate ``gateway_errors``.
    """
    failed_call = ArchivedCall(
        call_id="call-failed",
        iteration=1,
        attempt="primary",
        request={
            "messages": [{"role": "user", "content": "go"}],
            "tools": [],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
        response={
            "call_id": "call-failed",
            "outcome": "error",
            "error": {"message": "provider 500"},
            "content": "",
            "provider": "ollama",
            "model_id": "qwen3:8b",
        },
    )
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[failed_call]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    with pytest.raises(ArchiveOutcomeError, match="provider 500"):
        provider.answer(_gateway_request([{"role": "user", "content": "go"}], call_kind="turn"))
    assert provider.gateway_errors == []


def test_gateway_errors_survive_alongside_divergences() -> None:
    """A gateway error does not displace the queues' divergence notes."""
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    provider.turn_calls.divergences.append({"field": "model_id", "recorded": "a", "replayed": "b"})
    with pytest.raises(ScriptError, match="archive exhausted"):
        provider.answer(_gateway_request([{"role": "user", "content": "go"}], call_kind="turn"))
    divergences = _replay_divergences_from_provider(provider)
    assert [d.field for d in divergences] == ["model_id"]
    assert len(provider.gateway_errors) == 1
    assert provider.gateway_errors[0].queue == "turn"


def test_gateway_error_projection_populates_report_error() -> None:
    """The projection copies the structured errors and summarizes the first.

    The kernel converts a gateway error into response text, so the run's
    own ``except`` may not have set ``report.error`` — the projection must
    still make the failure loud for operators reading the plain field.
    """
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    provider.gateway_errors.append(
        ReplayGatewayError(
            kind="gateway_script_error",
            queue="turn",
            message="archive exhausted after 1 calls; the replayed turn made an extra call",
        )
    )
    report = ReplayReport(turn_index=0, turn_id="turn-abc")
    _project_gateway_errors(report, provider)
    assert len(report.gateway_errors) == 1
    assert report.gateway_errors[0].queue == "turn"
    assert "archive exhausted" in report.error
    assert report.ok is False


def test_gateway_error_projection_keeps_an_existing_error() -> None:
    """A louder run failure (e.g. the replay itself raised) wins the summary."""
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    provider.gateway_errors.append(
        ReplayGatewayError(kind="gateway_script_error", queue="auxiliary", message="boom")
    )
    report = ReplayReport(turn_index=0, turn_id="turn-abc", error="replay run failed: boom")
    _project_gateway_errors(report, provider)
    assert report.error == "replay run failed: boom"
    assert len(report.gateway_errors) == 1


# ── ReplayReport.ok ──────────────────────────────────────────────────────


def test_report_ok_requires_no_error_no_divergences_and_matches() -> None:
    report = ReplayReport(turn_index=0, turn_id="turn-abc")
    assert report.ok is True
    report.error = "boom"
    assert report.ok is False
    report.error = ""
    report.divergences.append(ReplayDivergence(kind="response"))
    assert report.ok is False
    report.divergences.clear()
    report.response_match = False
    assert report.ok is False
    report.response_match = True
    report.receipts_match = False
    assert report.ok is False


def test_report_ok_is_false_on_gateway_errors() -> None:
    """FIX-3: a structured gateway error fails the report even with no
    response divergence — the kernel may have converted the error into
    text that happens to match, so ``ok`` must not depend on that."""
    report = ReplayReport(turn_index=0, turn_id="turn-abc")
    report.gateway_errors.append(
        ReplayGatewayError(
            kind="gateway_script_error",
            queue="turn",
            message="archive exhausted after 1 calls; the replayed turn made an extra call",
        )
    )
    assert report.ok is False
    report.gateway_errors.clear()
    assert report.ok is True


# ── replay_turn error paths (no runtime needed) ──────────────────────────


class _Store:
    """Minimal ThreadStore double: ``load_turns`` + ``base_path``."""

    def __init__(self, base: Path, entries: dict[int, dict[str, Any]]) -> None:
        self.base_path = base
        self._entries = entries

    def load_turns(self, turns: Any) -> dict[int, dict[str, Any]]:
        return {t: self._entries[t] for t in turns if t in self._entries}


@pytest.mark.asyncio
async def test_replay_turn_missing_turn_sets_report_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing recorded turn surfaces as the report's error, not a raise."""
    import runtime.replay._engine as engine

    monkeypatch.setattr(engine, "ThreadStore", lambda _path: _Store(tmp_path, {}))
    from runtime.replay import replay_turn

    report = await replay_turn(
        source_thread_path=tmp_path / "thread.jsonl",
        turn_index=7,
        scratch_data_dir=tmp_path / "scratch",
    )
    assert report.error == "no recorded turn at index 7"
    assert report.ok is False


@pytest.mark.asyncio
async def test_replay_gateway_complete_delegates_to_owner_answer(
    tmp_path: Path,
) -> None:
    provider = ReplayGatewayProvider(
        turn_calls=ArchiveGatewayProvider(calls=[_archived_call("reply")]),
        auxiliary_calls=ArchiveGatewayProvider(calls=[]),
    )
    gateway = provider.gateway_for("ollama")

    result = await gateway.complete(
        _gateway_request([{"role": "user", "content": "go"}], call_kind="turn")
    )

    assert result.content == "reply"
    assert provider.turn_calls.position == 1


def test_seed_scratch_config_copies_models_and_overlays(tmp_path: Path) -> None:
    source = tmp_path / "source"
    scratch = tmp_path / "scratch"
    (source / "config" / "overlays").mkdir(parents=True)
    (source / "config" / "models.yaml").write_text("active: gpt\n", encoding="utf-8")
    (source / "config" / "overlays" / "lisa.yaml").write_text("mode: test\n", encoding="utf-8")

    _seed_scratch_config(source, scratch)

    assert (scratch / "config" / "models.yaml").read_text(encoding="utf-8") == "active: gpt\n"
    assert (scratch / "config" / "overlays" / "lisa.yaml").read_text(
        encoding="utf-8"
    ) == "mode: test\n"


def test_newest_scratch_entry_skips_blank_and_invalid_lines(tmp_path: Path) -> None:
    threads = tmp_path / "threads"
    threads.mkdir()
    active = threads / "thread.current.jsonl"
    active.write_text("\nnot-json\n{}\n", encoding="utf-8")

    entry = _newest_scratch_entry(tmp_path)

    assert entry == {}


def test_newest_scratch_entry_treats_stat_errors_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads = tmp_path / "threads"
    threads.mkdir()
    (threads / "thread.current.jsonl").write_text("{}\n", encoding="utf-8")
    original_stat = Path.stat

    def _stat_error(_path: Path, *args: object, **kwargs: object) -> os.stat_result:
        if _path.name == "thread.current.jsonl":
            raise OSError("stat failed")
        return original_stat(_path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _stat_error)

    assert _newest_scratch_entry(tmp_path) is None


@pytest.mark.asyncio
async def test_replay_turn_success_path_collects_no_divergences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.bootstrap
    import runtime.handle_message
    import runtime.replay._engine as engine

    recorded = RecordedTurn(
        turn_index=3,
        turn_id="turn-3",
        mode_id="exact",
        query="hello",
        response_text="hello",
    )
    monkeypatch.setattr(engine, "ThreadStore", lambda _path: _Store(tmp_path, {}))
    monkeypatch.setattr(engine, "load_recorded_turn", lambda _store, _index: recorded)
    monkeypatch.setattr(engine, "load_auxiliary_calls", lambda **_kwargs: [])

    class _Session:
        provider_gateways_override: object = None
        tool_call_answerer: object = None

    class _Runtime:
        def __init__(self) -> None:
            self.session = _Session()

        async def start(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    fake_runtime = _Runtime()
    monkeypatch.setattr(runtime.bootstrap, "build_runtime", lambda **_kwargs: fake_runtime)

    async def _handle(_runtime: object, _query: str, _transport: object) -> SimpleNamespace:
        return SimpleNamespace(text="hello")

    monkeypatch.setattr(runtime.handle_message, "handle_message", _handle)
    from runtime.replay import replay_turn

    report = await replay_turn(
        source_thread_path=tmp_path / "source.jsonl",
        turn_index=3,
        scratch_data_dir=tmp_path / "scratch",
    )

    assert report.turn_id == "turn-3"
    assert report.response_match is True
    assert report.receipts_match is True
    assert report.ok is True
    assert fake_runtime.session.provider_gateways_override is None
    assert fake_runtime.session.tool_call_answerer is None


@pytest.mark.asyncio
async def test_replay_turn_runtime_failure_is_reported_and_cleaned_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.bootstrap
    import runtime.handle_message
    import runtime.replay._engine as engine

    recorded = RecordedTurn(
        turn_index=4,
        turn_id="turn-4",
        mode_id="exact",
        query="hello",
        response_text="recorded",
    )
    monkeypatch.setattr(engine, "ThreadStore", lambda _path: _Store(tmp_path, {}))
    monkeypatch.setattr(engine, "load_recorded_turn", lambda _store, _index: recorded)
    monkeypatch.setattr(engine, "load_auxiliary_calls", lambda **_kwargs: [])

    class _Session:
        provider_gateways_override: object = None
        tool_call_answerer: object = None

    class _Runtime:
        def __init__(self) -> None:
            self.session = _Session()

        async def start(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    fake_runtime = _Runtime()
    monkeypatch.setattr(runtime.bootstrap, "build_runtime", lambda **_kwargs: fake_runtime)

    async def _handle_failure(_runtime: object, _query: str, _transport: object) -> SimpleNamespace:
        raise RuntimeError("runtime exploded")

    monkeypatch.setattr(runtime.handle_message, "handle_message", _handle_failure)
    from runtime.replay import replay_turn

    report = await replay_turn(
        source_thread_path=tmp_path / "source.jsonl",
        turn_index=4,
        scratch_data_dir=tmp_path / "scratch",
    )

    assert report.error == "replay run failed: runtime exploded"
    assert report.response_match is False
    assert report.ok is False
    assert fake_runtime.session.provider_gateways_override is None
    assert fake_runtime.session.tool_call_answerer is None


@pytest.mark.asyncio
async def test_replay_turn_unexpected_load_error_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.replay._engine as engine
    from runtime.replay import replay_turn

    class _BrokenStore:
        base_path = tmp_path

        def load_turns(self, _turns: object) -> dict:
            raise RuntimeError("load exploded")

    monkeypatch.setattr(engine, "ThreadStore", lambda _path: _BrokenStore())
    monkeypatch.setattr(
        engine,
        "load_recorded_turn",
        lambda _store, _index: (_ for _ in ()).throw(RuntimeError("loader exploded")),
    )

    report = await replay_turn(
        source_thread_path=tmp_path / "source.jsonl",
        turn_index=1,
        scratch_data_dir=tmp_path / "scratch",
    )

    assert report.error == "failed to load recorded turn 1: loader exploded"
    assert report.ok is False


@pytest.mark.asyncio
async def test_replay_turn_receipt_mismatch_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import runtime.bootstrap
    import runtime.handle_message
    import runtime.replay._engine as engine
    from runtime.replay import replay_turn

    recorded = RecordedTurn(
        turn_index=5,
        turn_id="turn-5",
        mode_id="exact",
        query="hello",
        response_text="hello",
        effect_receipts=[{"effect": "read", "capability_id": "tool"}],
    )
    monkeypatch.setattr(engine, "ThreadStore", lambda _path: _Store(tmp_path, {}))
    monkeypatch.setattr(engine, "load_recorded_turn", lambda _store, _index: recorded)
    monkeypatch.setattr(engine, "load_auxiliary_calls", lambda **_kwargs: [])

    class _Session:
        provider_gateways_override: object = None
        tool_call_answerer: object = None

    class _Runtime:
        def __init__(self) -> None:
            self.session = _Session()

        async def start(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    fake_runtime = _Runtime()
    monkeypatch.setattr(runtime.bootstrap, "build_runtime", lambda **_kwargs: fake_runtime)

    async def _handle(_runtime: object, _query: str, _transport: object) -> SimpleNamespace:
        return SimpleNamespace(text="hello")

    monkeypatch.setattr(runtime.handle_message, "handle_message", _handle)

    report = await replay_turn(
        source_thread_path=tmp_path / "source.jsonl",
        turn_index=5,
        scratch_data_dir=tmp_path / "scratch",
    )

    assert report.receipts_match is False
    assert any(divergence.kind == "receipts" for divergence in report.divergences)
