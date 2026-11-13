"""Pack-declared history semantics: policy, ledger, stubs, history folds, turn wiring."""

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
    HistoryFolds,
    HistoryKeys,
    HistoryLedger,
    HistoryPolicy,
    HistoryRecord,
    HistoryTemplateError,
    existing_result_ref,
    expand_template,
    fold_history,
    pack_state_message,
    render_stub,
    resolve_history_keys,
    split_pack_state,
)
from kernel.ipsar_fsm import WorkflowValidationError, _parse_working_set_tokens
from kernel.models import ToolDefinition, ToolExecutionContext, ToolResult
from kernel.turn_phases._compaction import IntraTurnCompactionState, LiveArrayEstimator
from kernel.turn_phases._history import (
    _later_requests,
    approved_invocation_history,
    cache_price_shares,
    finalize_turn_history,
    fold_pays_off,
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
    nested = {"sync": {"paths": ["e.py"], "files": [{"meta": {"p": "f.py"}}]}}
    assert expand_template("k:{result.sync.paths[]}", params, nested) == ["k:e.py"]
    assert expand_template("k:{result.sync.files[].meta.p}", params, nested) == ["k:f.py"]
    assert expand_template("k:{result.sync.absent[]}", params, nested) == []


def test_expand_template_missing_yields_no_key_and_none_renders_dash() -> None:
    assert expand_template("k:{result.absent}", {}, {}) == []
    assert expand_template("k:{paths[]}", {"paths": []}, {}) == []
    assert expand_template("k:{base}", {"base": None}, {}) == ["k:-"]
    assert expand_template("k:{base}", {"base": None}, {}, none_is_missing=True) == []


def test_policy_rejects_malformed_templates() -> None:
    with pytest.raises(ValueError, match="more than one"):
        HistoryPolicy(key=("k:{a[]}:{b[]}",))
    with pytest.raises(ValueError, match="more than one"):
        HistoryPolicy(key=("k:{a[].b[]}",))
    with pytest.raises(ValueError, match="invalid history placeholder"):
        HistoryPolicy(key=("k:{a.[]}",))
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


def test_error_results_keep_only_result_side_effect_invalidations() -> None:
    policy = HistoryPolicy(
        key="k:{artifact_id}",
        scope="turn",
        invalidates=[
            "r:{artifact_id}:{edits[].path}",
            "r:{result.artifact_id}:{result.sync.upstream_paths[]}@-:",
            "s:{artifact_id}",
            "constant",
        ],
    )
    params = {"artifact_id": "w", "edits": [{"path": "a.py"}]}
    rebased = ToolResult(
        tool_id="t",
        error="stale hash",
        output_json={"artifact_id": "w", "sync": {"upstream_paths": ["b.py", "c.py"]}},
    )
    assert resolve_history_keys(policy, None, params, rebased) == HistoryKeys(
        invalidates=("r:w:b.py@-:", "r:w:c.py@-:")
    )
    # No side effect reported: nothing to claim.
    bare = ToolResult(tool_id="t", error="stale hash", output_json={"artifact_id": "w"})
    assert resolve_history_keys(policy, None, params, bare) is None
    # Hooks never run for errors.
    hooked = SimpleNamespace(history_keys=lambda params, result: pytest.fail("hook ran"))
    assert resolve_history_keys(HistoryPolicy(hook=True), hooked, params, rebased) is None


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


def test_fold_history_uses_prefixes_turn_scope_and_latest_snapshot() -> None:
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
    folds = fold_history([("t1_", turn1), ("t2_", turn2)])
    # t1_c has no ref, so it can never be stubbed; the live re-read stays resident.
    assert set(folds.replacements) == {"t1_a", "t1_b"}
    assert "superseded by call t2_a" in folds.replacements["t1_a"]
    # Snapshots never ride on a folded result: that would rewrite a cached message.
    assert all("state]" not in stub for stub in folds.replacements.values())
    assert folds.snapshots == {"p": "cwd=/w"}


def test_fold_history_drops_snapshots_of_owners_without_folds() -> None:
    turn = {
        "records": [{"call_id": "a", "tool_id": "read", "keys": ["r:x"], "ref": _ref("1")}],
        "snapshots": {"p": "cwd=/w"},
    }
    assert fold_history([("t1_", turn)]) == HistoryFolds()


def test_pack_state_message_round_trips_through_split() -> None:
    assert pack_state_message({}) is None
    state = pack_state_message({"q": "b", "p": "a"})
    assert state is not None and state["role"] == "user"
    rehydrate = [{"role": "user", "content": "hi"}, state]
    kept, text = split_pack_state(rehydrate)
    assert kept == [{"role": "user", "content": "hi"}]
    assert text == "[p state] a\n[q state] b"


# ── Turn wiring ────────────────────────────────────────────────────────


def _turn_state(
    min_tokens: int = 10, *, read_share: float = 1.0, write_share: float = 1.0
) -> SimpleNamespace:
    # Default shares model a provider without prompt caching: every fold pays off.
    return SimpleNamespace(
        history_ledger=HistoryLedger(),
        intra_turn_compaction=IntraTurnCompactionState(
            history_epoch_min_tokens=min_tokens,
            cache_read_share=read_share,
            cache_write_share=write_share,
        ),
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


def _dispatch(
    state: Any, messages: list[dict[str, Any]], call_id: str, path: str, repeat: int = 20
) -> None:
    content = f"contents of {path} " * repeat
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


def test_cache_price_shares_prefer_manifest_prices() -> None:
    cfg = IntraTurnCompactionState(cache_read_share=0.2, cache_write_share=1.5)
    anthropic = SimpleNamespace(prompt_price=3e-6, cache_read_price=3e-7, cache_write_price=3.75e-6)
    assert cache_price_shares(anthropic, cfg) == pytest.approx((0.1, 1.25))
    # A published read price with NO write price keeps the configured write
    # fallback (1.5 here): cache creation may still bill at a premium.
    openai = SimpleNamespace(prompt_price=2e-6, cache_read_price=5e-7, cache_write_price=0.0)
    assert cache_price_shares(openai, cfg) == pytest.approx((0.25, 1.5))
    assert cache_price_shares(SimpleNamespace(prompt_price=1e-6), cfg) == (0.2, 1.5)
    assert cache_price_shares(None, cfg) == (0.2, 1.5)


def test_fold_pays_off_weighs_rewrite_against_later_reads() -> None:
    # No caching: the rewrite is free, any removal pays off.
    assert (
        fold_pays_off(removed=1, suffix=100_000, later_requests=0, read_share=1.0, write_share=1.0)
        > 0
    )
    cached = {"read_share": 0.1, "write_share": 1.25}
    # 12K stale under a 60K suffix: about 45 later requests to break even.
    assert fold_pays_off(removed=12_000, suffix=60_000, later_requests=40, **cached) < 0
    assert fold_pays_off(removed=12_000, suffix=60_000, later_requests=70, **cached) > 0
    # Stale bytes at the tail cost almost nothing to fold.
    assert fold_pays_off(removed=12_000, suffix=13_000, later_requests=2, **cached) > 0


def test_turn_1240_fold_parameters_stay_unprofitable() -> None:
    # Regression witness for the 2026-11-12 turn 1240 cache-collapse. The
    # epoch folded ~7K-token stale reads under a ~70K cached suffix on
    # glm-5.3-flashx (read share 0.2, no published write price) eight
    # times in one turn, each rewrite re-purchasing the prefix at full
    # input price. With the honest amortization window (about half the
    # requests made so far; observed gaps between folds were 1-16
    # requests) the fold must never look profitable.
    later = _later_requests(SimpleNamespace(iteration=26, max_iterations=150))
    assert later == 13
    # OpenRouter publishes a read price but no write price: write share
    # stays at the configured fallback (1.25), not the old 1.0 guess.
    read, write = cache_price_shares(
        SimpleNamespace(prompt_price=3.7e-7, cache_read_price=7.5e-8, cache_write_price=0.0),
        IntraTurnCompactionState(cache_read_share=0.1, cache_write_share=1.25),
    )
    assert read == pytest.approx(0.2027, abs=1e-3)
    assert write == pytest.approx(1.25)
    assert (
        fold_pays_off(
            removed=7_000, suffix=70_000, later_requests=later, read_share=read, write_share=write
        )
        < 0
    )
    # ... and even the most optimistic old estimate (later=26) loses once
    # the write share is honest.
    assert (
        fold_pays_off(
            removed=7_000, suffix=70_000, later_requests=26, read_share=read, write_share=write
        )
        < 0
    )


def test_supersession_epoch_skips_unprofitable_folds_under_caching() -> None:
    state = _turn_state(read_share=0.1, write_share=1.25)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "s"}]
    _dispatch(state, messages, "c1", "a.py")
    messages.append({"role": "user", "content": "x" * 20_000})
    _dispatch(state, messages, "c2", "a.py")
    hooks = _hooks({})
    state.iteration = 3
    assert (
        maybe_run_supersession_epoch(state=state, hooks=hooks, messages=messages, log=_quiet_log())
        == 0
    )
    assert messages[1]["content"].startswith("contents of a.py")
    # The same fold pays off once enough requests are expected to follow ...
    # (later = iteration // 2 must exceed this geometry's ~2K-request
    # break-even; iteration=5000 gives later=2500.)
    state.iteration = 5000
    state.max_iterations = 0
    folded = maybe_run_supersession_epoch(
        state=state, hooks=hooks, messages=messages, log=_quiet_log()
    )
    assert folded == 1 and messages[1]["content"].startswith("[superseded by call c2")


def test_supersession_epoch_respects_the_iteration_cap() -> None:
    state = _turn_state(read_share=0.1, write_share=1.25)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "s"}]
    _dispatch(state, messages, "c1", "a.py")
    messages.append({"role": "user", "content": "x" * 20_000})
    _dispatch(state, messages, "c2", "a.py")
    # ... but not when the iteration cap leaves no requests to repay it.
    state.iteration, state.max_iterations = 1000, 1001
    assert (
        maybe_run_supersession_epoch(
            state=state, hooks=_hooks({}), messages=messages, log=_quiet_log()
        )
        == 0
    )


def test_supersession_epoch_picks_the_boundary_with_the_largest_saving() -> None:
    state = _turn_state(read_share=0.1, write_share=1.25)
    state.iteration = 10
    messages: list[dict[str, Any]] = [{"role": "system", "content": "s"}]
    _dispatch(state, messages, "a1", "a.py")
    messages.append({"role": "user", "content": "x" * 50_000})
    _dispatch(state, messages, "b1", "b.py", repeat=400)
    _dispatch(state, messages, "a2", "a.py")
    _dispatch(state, messages, "b2", "b.py")
    folded = maybe_run_supersession_epoch(
        state=state, hooks=_hooks({}), messages=messages, log=_quiet_log()
    )
    # Folding b1 rewrites only the short tail; also folding a1 would rewrite
    # the 50K message between them for 340 chars, so a1 stays resident.
    assert folded == 1
    assert messages[1]["content"].startswith("contents of a.py")
    assert messages[3]["content"].startswith("[superseded by call b2")
    assert [r.call_id for r in state.history_ledger.foldable(turn_finished=False)] == ["a1"]


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


def test_record_dispatch_history_records_error_side_effects() -> None:
    state = _turn_state()
    common = {
        "state": state,
        "tool_def": SimpleNamespace(
            history=HistoryPolicy(
                scope="turn", invalidates=["r:{path}", "r:{result.sync.upstream_paths[]}"]
            ),
            implementation_module=None,
            owner_id="p",
        ),
        "tool_name": "edit",
        "params": {"path": "a.py"},
        "resident_content": "x",
        "log": _quiet_log(),
    }
    record_dispatch_history(
        tool_call_id="c1", tool_result=ToolResult(tool_id="edit", error="boom"), **common
    )
    assert state.history_ledger.records == []
    record_dispatch_history(
        tool_call_id="c2",
        tool_result=ToolResult(
            tool_id="edit", error="boom", output_json={"sync": {"upstream_paths": ["b.py"]}}
        ),
        **common,
    )
    [record] = state.history_ledger.records
    assert (record.call_id, record.keys, record.invalidates) == ("c2", [], ["r:b.py"])
    assert record.scope != "turn"


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
    folds = fold_history([("t0_", history)])
    assert "t0_c1" in folds.replacements
    assert folds.snapshots == {"lisa-pack-x": "cwd=/w"}


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
    # Rehydration folds issue anchors through stubs in the live messages.
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
    catalog["edit"] = ToolDefinition(
        tool_id="edit",
        owner_id="lisa-pack-x",
        history=HistoryPolicy(scope="turn", invalidates=["r:{path}", "r:{result.paths[]}"]),
    )
    rebased = ToolResult(tool_id="edit", error="boom", output_json={"paths": ["b.py"]})
    assert approved_invocation_history(tool_id="edit", result=rebased, **common) == {
        "records": [
            {
                "call_id": "approved-approval-1",
                "tool_id": "edit",
                "scope": "task",
                "chars": 0,
                "owner_id": "lisa-pack-x",
                "invalidates": ["r:b.py"],
            }
        ]
    }
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
