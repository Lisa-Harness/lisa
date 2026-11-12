"""Pack-declared history semantics: policy, ledger, stubs, replay folds, turn wiring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from kernel.capability_binding import CapabilityBinding, CapabilityRef
from kernel.context.mcp_catalog import _deserialize_tool
from kernel.history import (
    HistoryKeys,
    HistoryLedger,
    HistoryPolicy,
    HistoryRecord,
    HistoryTemplateError,
    existing_result_ref,
    expand_template,
    fold_replay,
    render_stub,
    resolve_history_keys,
)
from kernel.ipsar_fsm import WorkflowValidationError, _parse_working_set_tokens
from kernel.models import ToolDefinition, ToolExecutionContext, ToolResult
from kernel.turn_phases._compaction import IntraTurnCompactionState, LiveArrayEstimator
from kernel.turn_phases._history import (
    approved_invocation_history,
    finalize_turn_history,
    maybe_resolve_anchor_read,
    maybe_run_supersession_epoch,
    record_dispatch_history,
)
from pydantic import BaseModel


def _ref(text: str) -> str:
    return "r" + hashlib.sha256(text.encode()).hexdigest()


def _quiet_log() -> SimpleNamespace:
    return SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )


# ── Templates and policy validation ───────────────────────────────────


def test_expand_template_reads_params_results_and_fans_out() -> None:
    params = {"artifact_id": "w1", "paths": ["a.py", "b.py"], "opts": {"mode": "x"}}
    result = {"revision": "abc", "files": [{"path": "c.py"}, {"path": "d.py"}]}
    assert expand_template("k:{artifact_id}@{result.revision}", params, result) == ["k:w1@abc"]
    assert expand_template("k:{paths[]}@", params, result) == ["k:a.py@", "k:b.py@"]
    assert expand_template("k:{result.files[].path}", params, result) == ["k:c.py", "k:d.py"]
    assert expand_template("k:{opts.mode}", params, result) == ["k:x"]


def test_expand_template_missing_yields_no_key_and_none_renders_dash() -> None:
    assert expand_template("k:{result.absent}", {}, {}) == []
    assert expand_template("k:{paths[]}", {"paths": []}, {}) == []
    assert expand_template("k:{base}", {"base": None}, {}) == ["k:-"]
    assert expand_template("k:{base}", {"base": None}, {}, none_is_missing=True) == []


def test_policy_rejects_malformed_templates() -> None:
    with pytest.raises(ValueError, match="more than one"):
        HistoryPolicy(key=("k:{a[]}:{b[]}",))
    with pytest.raises(ValueError, match="invalid history placeholder"):
        HistoryPolicy(key=("k:{a-b}",))
    with pytest.raises(ValueError, match="scheme"):
        HistoryPolicy(anchor="{path}")
    with pytest.raises(ValueError, match="non-empty"):
        HistoryPolicy(key=(" ",))


def test_policy_validates_parameters_against_tool() -> None:
    policy = HistoryPolicy(key="k:{artifact_id}:{result.path}", anchor="read:{artifact_id}")
    policy.validate_parameters("read", {"artifact_id"})
    with pytest.raises(HistoryTemplateError, match="unknown parameters"):
        policy.validate_parameters("read", {"path"})
    with pytest.raises(HistoryTemplateError, match="must start with 'other:'"):
        policy.validate_parameters("other", {"artifact_id"})
    with pytest.raises(HistoryTemplateError, match="drop key"):
        HistoryPolicy(hook=True, scope="turn").validate_parameters("t", ())


class _Params(BaseModel):
    artifact_id: str


def _tool_data(history: dict[str, Any], module: str) -> dict[str, Any]:
    return {"tool_id": "read", "module_path": module, "history": history}


def test_catalog_load_fails_on_unknown_parameter_or_missing_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = SimpleNamespace(Parameters=_Params, execute=lambda ctx: None)
    monkeypatch.setattr("kernel.context.mcp_catalog._import_tool_module", lambda path: module)
    ok = _deserialize_tool(_tool_data({"key": "k:{artifact_id}"}, "m"))
    assert ok.history == HistoryPolicy(key=("k:{artifact_id}",))
    with pytest.raises(HistoryTemplateError, match="unknown parameters"):
        _deserialize_tool(_tool_data({"key": "k:{path}"}, "m"))
    with pytest.raises(HistoryTemplateError, match="history_keys"):
        _deserialize_tool(_tool_data({"hook": True}, "m"))
    with pytest.raises(HistoryTemplateError, match="history_snapshot"):
        _deserialize_tool(_tool_data({"key": "k", "snapshot": True}, "m"))


def test_resolve_keys_skips_errors_and_calls_hooks() -> None:
    policy = HistoryPolicy(key="k:{artifact_id}", scope="turn")
    ok = ToolResult(tool_id="t", output_json={})
    assert resolve_history_keys(policy, None, {"artifact_id": "w"}, ok) == HistoryKeys(
        keys=("k:w",), scope="turn"
    )
    failed = ToolResult(tool_id="t", error="boom")
    assert resolve_history_keys(policy, None, {"artifact_id": "w"}, failed) is None
    hooked = SimpleNamespace(history_keys=lambda params, result: {"keys": [params["cmd"]]})
    assert resolve_history_keys(HistoryPolicy(hook=True), hooked, {"cmd": "ls"}, ok) == HistoryKeys(
        keys=("ls",)
    )


def test_resolve_keys_fills_parameter_defaults() -> None:
    class Parameters(BaseModel):
        path: str
        mode: str = "full"

    module = SimpleNamespace(Parameters=Parameters)
    ok = ToolResult(tool_id="t", output_json={})
    policy = HistoryPolicy(key="k:{path}:{mode}")
    assert resolve_history_keys(policy, module, {"path": "a"}, ok).keys == ("k:a:full",)
    # Invalid arguments fall back to the raw call (that result errored anyway).
    assert resolve_history_keys(policy, module, {"mode": "x"}, ok).keys == ()


# ── Ledger ─────────────────────────────────────────────────────────────


def test_ledger_supersedes_equal_keys_and_invalidates_prefixes() -> None:
    ledger = HistoryLedger()
    read_a = HistoryRecord(call_id="c1", tool_id="read", keys=["r:a@1"], chars=900)
    read_b = HistoryRecord(call_id="c2", tool_id="read", keys=["r:b@1"], chars=300)
    assert ledger.record(read_a) == []
    assert ledger.record(read_b) == []
    reread = HistoryRecord(call_id="c3", tool_id="read", keys=["r:a@1"], chars=900)
    assert ledger.record(reread) == [read_a]
    assert (read_a.stale_reason, read_a.stale_by) == ("superseded", "c3")
    edit = HistoryRecord(call_id="c4", tool_id="edit", invalidates=["r:b@"], scope="turn")
    assert ledger.record(edit) == [read_b]
    assert read_b.stale_reason == "invalidated"
    assert ledger.pending_dead_chars() == 1200
    assert ledger.foldable(turn_finished=False) == [read_a, read_b]
    assert ledger.foldable(turn_finished=True) == [read_a, read_b, edit]


def test_record_with_several_keys_stays_live_until_all_are_dead() -> None:
    ledger = HistoryLedger()
    both = HistoryRecord(call_id="c1", tool_id="status", keys=["s:a", "s:b"])
    ledger.record(both)
    ledger.record(HistoryRecord(call_id="c2", tool_id="status", keys=["s:a"]))
    assert not both.dead
    ledger.record(HistoryRecord(call_id="c3", tool_id="status", keys=["s:b"]))
    assert both.dead


def test_stub_always_carries_ref() -> None:
    record = HistoryRecord(
        call_id="c1", tool_id="read", keys=["k"], dead_keys=["k"], chars=10, anchor="read:x"
    )
    record.stale_by, record.stale_reason = "c2", "superseded"
    with pytest.raises(ValueError, match="no result ref"):
        render_stub(record)
    record.ref = _ref("x")
    stub = render_stub(record)
    assert record.ref in stub and "read:x" in stub and "superseded by call c2" in stub


def test_existing_result_ref_reads_bare_and_preview_refs() -> None:
    ref = _ref("body")
    assert existing_result_ref(ref) == ref
    assert existing_result_ref(json.dumps({"ref": ref, "abbreviated": True})) == ref
    assert existing_result_ref(json.dumps({"ref": ref, "page_omitted": True})) == ref
    assert existing_result_ref(json.dumps({"ref": ref})) == ""
    assert existing_result_ref("plain text") == ""


def test_fold_replay_uses_prefixes_turn_scope_and_latest_snapshot() -> None:
    turn1 = {
        "records": [
            {"call_id": "a", "tool_id": "read", "keys": ["r:x"], "ref": _ref("1")},
            {"call_id": "b", "tool_id": "ls", "scope": "turn", "ref": _ref("2"), "owner_id": "p"},
            {"call_id": "c", "tool_id": "ls", "scope": "turn"},
        ],
        "snapshots": {"p": "old"},
    }
    turn2 = {
        "records": [{"call_id": "a", "tool_id": "read", "keys": ["r:x"], "ref": _ref("3")}],
        "snapshots": {"p": "cwd=/w"},
    }
    folds = fold_replay([("t1_", turn1), ("t2_", turn2)])
    # t1_c has no ref, so it can never be stubbed; the live re-read stays resident.
    assert set(folds) == {"t1_a", "t1_b"}
    assert "superseded by call t2_a" in folds["t1_a"]
    assert folds["t1_b"].endswith("[p state] cwd=/w")


# ── Turn wiring ────────────────────────────────────────────────────────


def _turn_state(min_tokens: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        history_ledger=HistoryLedger(),
        intra_turn_compaction=IntraTurnCompactionState(history_epoch_min_tokens=min_tokens),
        intra_turn_compaction_estimator=LiveArrayEstimator(),
        provider_messages=[],
        transcript_start=1,
        messages=None,
        ctx=None,
    )


def _hooks(store: dict[str, str]) -> SimpleNamespace:
    def persist(record: dict[str, Any]) -> str:
        ref = _ref(record["content"])
        store[ref] = record["content"]
        return ref

    return SimpleNamespace(
        tools=SimpleNamespace(persist_tool_result=persist),
        identity=SimpleNamespace(task_id="task-1"),
    )


_READ_DEF = SimpleNamespace(
    history=HistoryPolicy(key="r:{path}", anchor="read:{path}"),
    implementation_module=None,
    owner_id="lisa-pack-x",
)


def _dispatch(state: Any, messages: list[dict[str, Any]], call_id: str, path: str) -> None:
    content = f"contents of {path} " * 20
    messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    record_dispatch_history(
        state=state,
        tool_def=_READ_DEF,
        tool_name="read",
        tool_call_id=call_id,
        params={"path": path},
        tool_result=ToolResult(tool_id="read"),
        resident_content=content,
        log=_quiet_log(),
    )


def test_supersession_epoch_folds_after_threshold_and_resyncs() -> None:
    state = _turn_state(min_tokens=10)
    state.intra_turn_compaction_estimator.observe_request_tokens(5000)
    store: dict[str, str] = {}
    messages: list[dict[str, Any]] = [{"role": "system", "content": "s"}]
    _dispatch(state, messages, "c1", "a.py")
    original = messages[1]["content"]
    state.provider_messages = [dict(m) for m in messages]
    assert (
        maybe_run_supersession_epoch(
            state=state, hooks=_hooks(store), messages=messages, log=_quiet_log()
        )
        == 0
    )
    _dispatch(state, messages, "c2", "a.py")
    folded = maybe_run_supersession_epoch(
        state=state, hooks=_hooks(store), messages=messages, log=_quiet_log()
    )
    assert folded == 1
    stub = messages[1]["content"]
    assert stub.startswith("[superseded by call c2") and "read:a.py" in stub
    assert state.provider_messages[1]["content"] == stub
    # Lossless: the stub's ref dereferences to the exact folded text.
    assert store[existing_ref_in(stub)] == original
    assert messages[2]["content"] != stub
    assert state.intra_turn_compaction_estimator._estimate < 5000
    assert state.history_ledger.pending_dead_chars() == 0


def existing_ref_in(stub: str) -> str:
    return next(word for word in stub.split() if word.startswith("r") and len(word) == 65)


def test_supersession_epoch_waits_for_threshold_and_needs_a_store() -> None:
    state = _turn_state(min_tokens=100_000)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "s"}]
    _dispatch(state, messages, "c1", "a.py")
    _dispatch(state, messages, "c2", "a.py")
    hooks = _hooks({})
    assert (
        maybe_run_supersession_epoch(state=state, hooks=hooks, messages=messages, log=_quiet_log())
        == 0
    )
    state.intra_turn_compaction.history_epoch_min_tokens = 1
    no_store = SimpleNamespace(
        tools=SimpleNamespace(persist_tool_result=None), identity=SimpleNamespace(task_id="t")
    )
    assert (
        maybe_run_supersession_epoch(
            state=state, hooks=no_store, messages=messages, log=_quiet_log()
        )
        == 0
    )
    assert messages[1]["content"].startswith("contents of a.py")


def test_record_dispatch_history_survives_hook_failure() -> None:
    def broken(params: dict[str, Any], result: ToolResult) -> None:
        raise RuntimeError("bug")

    state = _turn_state()
    record_dispatch_history(
        state=state,
        tool_def=SimpleNamespace(
            history=HistoryPolicy(hook=True),
            implementation_module=SimpleNamespace(history_keys=broken),
            owner_id="p",
        ),
        tool_name="bash_run",
        tool_call_id="c1",
        params={},
        tool_result=ToolResult(tool_id="bash_run"),
        resident_content="x",
        log=_quiet_log(),
    )
    assert state.history_ledger.records == []


class _Bindings:
    def binding_for(self, capability: CapabilityRef) -> CapabilityBinding:
        return CapabilityBinding(
            owner_id=capability.owner_id, configuration=MappingProxyType({"root": "/w"})
        )


def _snapshot_catalog(seen: dict[str, Any]) -> dict[str, ToolDefinition]:
    async def history_snapshot(ctx: ToolExecutionContext, keys: list[str]) -> str:
        seen["config"] = dict(ctx.pack_config)
        seen["keys"] = keys
        return "cwd=/w"

    async def resolve_history_anchor(ctx: ToolExecutionContext, anchor: str) -> str | None:
        return "fresh text" if anchor.endswith("a.py") else None

    module = SimpleNamespace(
        execute=None,
        history_snapshot=history_snapshot,
        resolve_history_anchor=resolve_history_anchor,
    )
    return {
        "read": ToolDefinition(
            tool_id="read",
            owner_id="lisa-pack-x",
            implementation_module=module,
            history=HistoryPolicy(key="r:{path}", anchor="read:{path}", snapshot=True),
        )
    }


@pytest.mark.asyncio
async def test_finalize_persists_refs_filters_own_records_and_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    catalog = _snapshot_catalog(seen)
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", catalog)
    state = _turn_state()
    state.ctx = SimpleNamespace(capability_binding_provider=_Bindings(), mode_id="m")
    messages: list[dict[str, Any]] = [{"role": "user", "content": "q"}]
    state.messages = messages
    _dispatch(state, messages, "c1", "a.py")
    _dispatch(state, messages, "c2", "a.py")
    # A record carried over from a resumed segment has no message in this transcript.
    state.history_ledger.records.insert(
        0, HistoryRecord(call_id="old", tool_id="read", keys=["r:z"])
    )
    store: dict[str, str] = {}
    history = await finalize_turn_history(
        state=state, hooks=_hooks(store), tool_catalog=catalog, log=_quiet_log()
    )
    assert [r["call_id"] for r in history["records"]] == ["c1", "c2"]
    assert all(r["ref"] in store for r in history["records"])
    assert history["snapshots"] == {"lisa-pack-x": "cwd=/w"}
    assert seen == {"config": {"root": "/w"}, "keys": ["r:a.py", "r:a.py"]}
    folds = fold_replay([("t0_", history)])
    assert folds["t0_c1"].endswith("[lisa-pack-x state] cwd=/w")


@pytest.mark.asyncio
async def test_finalize_is_empty_without_records() -> None:
    state = _turn_state()
    state.messages = []
    assert (
        await finalize_turn_history(
            state=state, hooks=_hooks({}), tool_catalog={}, log=_quiet_log()
        )
        == {}
    )


@pytest.mark.asyncio
async def test_anchor_read_resolves_pages_and_falls_back_to_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _snapshot_catalog({})
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", catalog)
    state = _turn_state()
    ctx = SimpleNamespace(capability_binding_provider=_Bindings())
    common = {"state": state, "tool_catalog": catalog, "ctx": ctx, "log": _quiet_log()}
    # An anchor the kernel never issued is not served: it takes the normal ref path.
    forged = {"ref": "read:a.py"}
    assert (
        await maybe_resolve_anchor_read(
            tool_name="thread_read_content", tool_arguments=forged, **common
        )
        is None
    )
    assert forged == {"ref": "read:a.py"}
    # Replay folds issue anchors through stubs in the live messages.
    state.messages = [
        {"role": "tool", "tool_call_id": "c0", "content": _stub("read:a.py")},
        {"role": "tool", "tool_call_id": "c9", "content": _stub("read:c.py")},
    ]
    served = await maybe_resolve_anchor_read(
        tool_name="thread_read_content",
        tool_arguments={"ref": "read:a.py", "offset_chars": 2, "limit_chars": 3},
        **common,
    )
    assert served is not None
    page = json.loads(served)
    assert (page["content"], page["total_chars"], page["has_more"]) == ("esh", 10, True)
    # Unresolvable anchor: rewrite the call to the recorded ref and let the store serve it.
    state.history_ledger.records.append(
        HistoryRecord(call_id="c1", tool_id="read", anchor="read:b.py", ref=_ref("b"))
    )
    arguments = {"ref": "read:b.py"}
    assert (
        await maybe_resolve_anchor_read(
            tool_name="thread_read_content", tool_arguments=arguments, **common
        )
        is None
    )
    assert arguments["ref"] == _ref("b")
    unknown = await maybe_resolve_anchor_read(
        tool_name="thread_read_content", tool_arguments={"ref": "read:c.py"}, **common
    )
    assert unknown is not None and "could not be resolved" in json.loads(unknown)["error"]
    # Plain refs and other tools pass straight through.
    for name, ref in (("thread_read_content", _ref("x")), ("other", "read:a.py")):
        assert (
            await maybe_resolve_anchor_read(tool_name=name, tool_arguments={"ref": ref}, **common)
            is None
        )


def test_approved_invocation_history_carries_invalidations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = {
        "edit": ToolDefinition(
            tool_id="edit",
            owner_id="lisa-pack-x",
            history=HistoryPolicy(scope="turn", invalidates="r:{path}"),
        ),
        "plain": ToolDefinition(tool_id="plain"),
    }
    monkeypatch.setattr("kernel.tools.registry._TOOL_CATALOG", catalog)
    common = {"request_id": "approval-1", "parameters": {"path": "a.py"}, "log": _quiet_log()}
    history = approved_invocation_history(
        tool_id="edit", result=ToolResult(tool_id="edit"), **common
    )
    assert history == {
        "records": [
            {
                "call_id": "approved-approval-1",
                "tool_id": "edit",
                "scope": "turn",
                "chars": 0,
                "owner_id": "lisa-pack-x",
                "invalidates": ["r:a.py"],
            }
        ]
    }
    failed = ToolResult(tool_id="edit", error="boom")
    assert approved_invocation_history(tool_id="edit", result=failed, **common) == {}
    plain = ToolResult(tool_id="plain")
    assert approved_invocation_history(tool_id="plain", result=plain, **common) == {}


def _stub(anchor: str) -> str:
    return render_stub(
        HistoryRecord(
            call_id="c", tool_id="read", keys=["k"], dead_keys=["k"], anchor=anchor, ref=_ref("s")
        )
    )


def test_mode_working_set_tokens_must_be_positive(tmp_path: Path) -> None:
    path = tmp_path / "_mode.md"
    assert _parse_working_set_tokens({}, path) == 0
    assert _parse_working_set_tokens({"working_set_tokens": "80000"}, path) == 80000
    with pytest.raises(WorkflowValidationError, match="working_set_tokens"):
        _parse_working_set_tokens({"working_set_tokens": 0}, path)
