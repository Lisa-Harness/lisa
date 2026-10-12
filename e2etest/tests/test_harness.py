from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from e2etest.driver import DriverOutcome, Exchange, approval_payload
from e2etest.environment import RunPaths
from e2etest.oracles import Evidence, evaluate
from e2etest.scenario import OracleSpec, ScenarioError, load_scenario
from e2etest.scripted import ScriptError, ScriptTurn, load_script
from kernel.provider_contracts import CompletionRequest
from runtime.types import HostCommandResult


def _write_scenario(tmp_path: Path, body: str) -> Path:
    (tmp_path / "ctx.md").write_text("---\nname: t\n---\n", encoding="utf-8")
    path = tmp_path / "s.scenario.yml"
    path.write_text(body, encoding="utf-8")
    return path


def test_scenario_parses_oracles_and_approvals(tmp_path: Path) -> None:
    path = _write_scenario(
        tmp_path,
        """
id: pack/x
session_context: ctx.md
turns: ["hello"]
approvals: {approve: ["*"], deny: [coding_push]}
oracles:
  hygiene:
    - no_turn_errors: {}
    - receipt: {capability: coding_read_file, min: 2}
""",
    )
    scenario = load_scenario(path)
    assert scenario.scenario_id == "pack/x"
    assert scenario.approvals.decide("coding_commit") == "approve"
    assert scenario.approvals.decide("coding_push") == "deny"
    assert [o.name for o in scenario.oracles] == ["no_turn_errors", "receipt"]
    assert scenario.oracles[1].args == {"capability": "coding_read_file", "min": 2}


def test_unlisted_approval_target_is_denied(tmp_path: Path) -> None:
    path = _write_scenario(
        tmp_path, "session_context: ctx.md\nturns: [x]\napprovals: {approve: [a]}\n"
    )
    assert load_scenario(path).approvals.decide("b") == "deny"


def test_scenario_resolves_data_files_fixture(tmp_path: Path) -> None:
    (tmp_path / "seed").mkdir()
    path = _write_scenario(
        tmp_path, "session_context: ctx.md\nturns: [x]\nfixture: {data_files: seed}\n"
    )
    scenario = load_scenario(path)
    assert scenario.fixture_data_files == (tmp_path / "seed").resolve()
    assert scenario.fixture_repo is None
    missing = _write_scenario(
        tmp_path, "session_context: ctx.md\nturns: [x]\nfixture: {data_files: nope}\n"
    )
    with pytest.raises(ScenarioError, match=r"fixture\.data_files"):
        load_scenario(missing)


def test_data_check_runs_in_storage_root_with_placeholders(tmp_path: Path) -> None:
    files = tmp_path / "data" / "data" / "files"
    files.mkdir(parents=True)
    (files / "ledger.txt").write_text("balanced\n", encoding="utf-8")
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "check.sh").write_text('grep -q "$1" ledger.txt\n', encoding="utf-8")
    paths = RunPaths(tmp_path, tmp_path / "data", None, None, "", scenario_dir)
    evidence = Evidence(paths, DriverOutcome())
    argv = ["sh", "${SCENARIO_DIR}/check.sh"]
    results = evaluate(
        (_spec("data_check", argv=[*argv, "balanced"]), _spec("data_check", argv=[*argv, "x"])),
        evidence,
    )
    assert [r.passed for r in results] == [True, False]
    assert "exit=1" in results[1].detail
    empty = Evidence(RunPaths(tmp_path, tmp_path / "none", None, None, ""), DriverOutcome())
    (missing,) = evaluate((_spec("data_check", argv=["true"]),), empty)
    assert not missing.passed and "storage sandbox missing" in missing.detail


def test_scenario_rejects_missing_turns(tmp_path: Path) -> None:
    path = _write_scenario(tmp_path, "session_context: ctx.md\nturns: []\n")
    with pytest.raises(ScenarioError, match="turns"):
        load_scenario(path)


def test_approval_payload_accepts_flat_and_nested_shapes() -> None:
    flat = {"schema": "lisa.approval.v1", "request_id": "r1", "target_id": "t"}
    assert approval_payload(HostCommandResult(text="", data=flat)) == flat
    nested = HostCommandResult(text="", data={"approval": {"request_id": "r2"}})
    assert approval_payload(nested) == {"request_id": "r2"}
    assert approval_payload(HostCommandResult(text="done")) is None


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def committed_run(tmp_path: Path) -> Evidence:
    workspace = tmp_path / "workspace" / "repo"
    workspace.mkdir(parents=True)
    _git(workspace, "init", "-q", "-b", "main")
    _git(workspace, "config", "user.email", "t@t")
    _git(workspace, "config", "user.name", "t")
    (workspace / "f.txt").write_text("bad\n", encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-qm", "init")
    initial = _git(workspace, "rev-parse", "HEAD")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(workspace, "remote", "add", "origin", str(remote))
    _git(workspace, "push", "-q", "origin", "main")
    worktree = tmp_path / "wt"
    _git(workspace, "worktree", "add", "-q", "-b", "lisa/x", str(worktree))
    (worktree / "f.txt").write_text("good\n", encoding="utf-8")
    _git(worktree, "commit", "-qam", "fix")
    commit = _git(worktree, "rev-parse", "HEAD")
    tree = _git(worktree, "rev-parse", "HEAD^{tree}")

    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    (data / "state" / "code-change-artifacts.json").write_text(
        json.dumps(
            {
                "a1": {
                    "artifact_id": "a1",
                    "status": "committed",
                    "branch": "lisa/x",
                    "commit_sha": commit,
                    "worktree_path": str(worktree),
                    "revision": 1,
                    "required_check_ids": ["test"],
                    "checks": {
                        "test": {"status": "passed", "revision": 1, "checked_tree_sha": tree}
                    },
                    "updated_at": "2026-10-01T00:00:00Z",
                }
            }
        ),
        encoding="utf-8",
    )
    threads = data / "data" / "threads"
    threads.mkdir(parents=True)
    (threads / "t.current.jsonl").write_text(
        json.dumps({"turn": 1, "effect_receipts": [{"capability_id": "coding_read_file"}]}) + "\n",
        encoding="utf-8",
    )
    paths = RunPaths(tmp_path, data, workspace, remote, initial)
    outcome = DriverOutcome(exchanges=[Exchange(index=0, kind="turn", message="hi", text="ok")])
    return Evidence(paths, outcome)


def _spec(name: str, **args: object) -> OracleSpec:
    return OracleSpec(phase="p", name=name, args=dict(args))


def test_code_change_oracles_pass_on_a_real_committed_worktree(committed_run: Evidence) -> None:
    check = ["grep", "-q", "good", "f.txt"]
    results = evaluate(
        (
            _spec("not_aborted"),
            _spec("no_turn_errors"),
            _spec("receipt", capability="coding_read_file"),
            _spec("workproduct_committed"),
            _spec("branch_prefix", prefix="lisa/"),
            _spec("main_untouched"),
            _spec("diff_touches", paths=["f.txt"]),
            _spec("diff_touches", paths=["f.txt"], exact=True),
            _spec("checks_current"),
            _spec("worktree_clean"),
            _spec("baseline_fails", argv=check),
            _spec("independent_check", argv=check),
        ),
        committed_run,
    )
    assert all(r.passed for r in results), [(r.name, r.detail) for r in results if not r.passed]


def test_oracles_fail_with_evidence(committed_run: Evidence) -> None:
    worktree = Path(committed_run.artifacts[0]["worktree_path"])
    (worktree / "f.txt").write_text("dirty\n", encoding="utf-8")
    committed_run.outcome.exchanges[0].operator_exit = "pause"
    results = {
        r.name: r
        for r in evaluate(
            (
                _spec("worktree_clean"),
                _spec("no_operator_exit"),
                _spec("diff_touches", paths=["other.txt"]),
                _spec("nope"),
            ),
            committed_run,
        )
    }
    assert not any(r.passed for r in results.values())
    assert "dirty" in results["worktree_clean"].detail
    assert "pause" in results["no_operator_exit"].detail
    assert "other.txt" in results["diff_touches"].detail
    assert results["nope"].detail == "unknown oracle"


def test_diff_allow_and_check_restore_judge_source_against_original_tests(
    committed_run: Evidence,
) -> None:
    """``allow`` admits a test edit; ``restore`` rejudges with the baseline file."""
    worktree = Path(committed_run.artifacts[0]["worktree_path"])
    (worktree / "t.txt").write_text("extra regression test\n", encoding="utf-8")
    _git(worktree, "add", "t.txt")
    _git(worktree, "commit", "-qm", "add test")
    committed_run.artifacts[0]["commit_sha"] = _git(worktree, "rev-parse", "HEAD")
    results = evaluate(
        (
            _spec("diff_touches", paths=["f.txt"], allow=["t.txt"], exact=True),
            _spec("diff_touches", paths=["f.txt"], exact=True),
            _spec("independent_check", argv=["grep", "-q", "bad", "f.txt"], restore=["f.txt"]),
            _spec("independent_check", argv=["grep", "-q", "good", "f.txt"], restore=["f.txt"]),
        ),
        committed_run,
    )
    assert [r.passed for r in results] == [True, False, True, False], [
        (r.name, r.detail) for r in results
    ]
    assert "unexpected=['t.txt']" in results[1].detail
    assert "restored=['f.txt']" in results[3].detail


def test_checkout_oracles_expand_scenario_dir(committed_run: Evidence, tmp_path: Path) -> None:
    """Hidden acceptance checks live beside the scenario, outside the fixture."""
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "accept.sh").write_text("grep -q good f.txt\n", encoding="utf-8")
    ev = Evidence(replace(committed_run.paths, scenario_dir=scenario_dir), committed_run.outcome)
    argv = ["sh", "${SCENARIO_DIR}/accept.sh"]
    results = evaluate(
        (_spec("baseline_fails", argv=argv), _spec("independent_check", argv=argv)), ev
    )
    assert all(r.passed for r in results), [(r.name, r.detail) for r in results]


def test_recovery_oracles_read_store_and_exchange_evidence(committed_run: Evidence) -> None:
    state_dir = committed_run.paths.data_dir / "state"
    (state_dir / "recovery-items.json").write_text(
        json.dumps(
            {
                "w1": {
                    "work_item_id": "w1",
                    "state": "skipped",
                    "retry_policy": "confirm",
                    "target_id": "bash_run",
                }
            }
        ),
        encoding="utf-8",
    )
    committed_run.outcome.exchanges[0].text = "Open recovery work items: w1 state=stalled"
    results = evaluate(
        (
            _spec("exchange_contains", text="state=stalled"),
            _spec("recovery_record", retry_policy="confirm"),
            _spec("recovery_settled", state="skipped"),
        ),
        committed_run,
    )
    assert all(result.passed for result in results), [
        (r.name, r.detail) for r in results if not r.passed
    ]


def _request(messages: list[dict[str, Any]], **kwargs: Any) -> CompletionRequest:
    return CompletionRequest(
        messages=messages, model_def=SimpleNamespace(), max_tokens=1, temperature=0.0, **kwargs
    )


def _write_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model.py"
    path.write_text("from e2etest.scripted import Reply\n" + body, encoding="utf-8")
    return path


def test_script_turn_pairs_tool_calls_with_results() -> None:
    call = {"id": "c1", "function": {"name": "coding_read_file", "arguments": '{"path": "a"}'}}
    turn = ScriptTurn(
        _request(
            [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "c1", "content": 'Read a {"sha": "x"}'},
            ],
            tools=[{"type": "function", "function": {"name": "coding_read_file"}}],
        ),
        index=0,
    )
    assert turn.tool_names == {"coding_read_file"} and not turn.auxiliary
    assert turn.user_text == "fix it"
    last = turn.last("coding_read_file")
    assert last is not None and last.arguments == {"path": "a"} and last.data == {"sha": "x"}
    assert turn.count("coding_read_file") == 1 and turn.last("coding_edit_files") is None


def test_scripted_provider_answers_with_tool_calls_and_default_auxiliary(
    tmp_path: Path,
) -> None:
    provider = load_script(
        _write_script(tmp_path, "def respond(turn):\n    return Reply.call('t', x=1)\n")
    )
    tools = [{"type": "function", "function": {"name": "t"}}]
    result = provider.answer(_request([{"role": "user", "content": "go"}], tools=tools))
    assert result.finish_reason == "tool_calls"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"x": 1}
    assert provider.answer(_request([], response_format=dict)).content == "{}"
    assert provider.answer(_request([])).content == ""
    assert [entry["auxiliary"] for entry in provider.log] == [False, True, True]


def test_load_script_requires_respond_and_scenario_resolves_script(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="respond"):
        load_script(_write_script(tmp_path, "X = 1\n"))
    _write_script(tmp_path, "def respond(turn):\n    return Reply.text('ok')\n")
    path = _write_scenario(
        tmp_path, "session_context: ctx.md\nturns: [x]\nmodel: {script: model.py}\n"
    )
    assert load_scenario(path).model_script == (tmp_path / "model.py").resolve()
