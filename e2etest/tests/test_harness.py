from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from e2etest.driver import (
    DriverOutcome,
    Exchange,
    _BudgetExceededError,
    _Conversation,
    approval_payload,
)
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


def test_scenario_parses_surface_identity(tmp_path: Path) -> None:
    path = _write_scenario(
        tmp_path,
        "session_context: ctx.md\nturns: [x]\nsurface: {type: teams_channel, id: chan-42}\n",
    )
    scenario = load_scenario(path)
    assert scenario.surface.type == "teams_channel"
    assert scenario.surface.id == "chan-42"
    # Omitted surface block defaults to the session-plan fallback (empty).
    plain = _write_scenario(tmp_path, "session_context: ctx.md\nturns: [x]\n")
    default = load_scenario(plain)
    assert default.surface.type == ""
    assert default.surface.id == ""


def _write_memory_db(data_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    """Create a minimal memory.db with (content, memory_scope, surface_id) rows."""
    import sqlite3

    db = data_dir / "data" / "memory.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE memories ("
            "memory_kind TEXT, status TEXT, content TEXT, "
            "memory_scope TEXT, surface_id TEXT);"
        )
        conn.executemany("INSERT INTO memories VALUES ('episodic', 'active', ?, ?, ?)", rows)


def test_driver_forwards_surface_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_conversation`` must pass scenario.surface into ``build_runtime``."""
    captured: dict[str, Any] = {}

    def _fake_build(**kwargs: Any) -> Any:
        captured.update(kwargs)
        # Return just enough for run_conversation's boot checks; the script
        # below runs zero turns so _drive never touches the fake.
        return SimpleNamespace(session=SimpleNamespace(restart_required=False))

    import e2etest.driver as driver_mod

    monkeypatch.setattr(driver_mod, "build_runtime", _fake_build)
    monkeypatch.setattr(driver_mod, "inject_packs_venv_site_packages", lambda _p: Path("/x"))
    monkeypatch.setattr(driver_mod, "load_script", lambda _p: SimpleNamespace(log=[]))

    async def _noop_drive(runtime: Any, scenario: Any, outcome: Any) -> None:
        return None

    monkeypatch.setattr(driver_mod, "_drive", _noop_drive)

    path = _write_scenario(
        tmp_path,
        "session_context: ctx.md\nturns: [x]\n"
        "surface: {type: teams_channel, id: chan-42}\n"
        "model: {script: m.py}\n",
    )
    (tmp_path / "m.py").write_text("def respond(*a, **k): ...\n", encoding="utf-8")
    scenario = load_scenario(path)
    paths = RunPaths(tmp_path, tmp_path / "data", None, None, "", tmp_path)

    driver_mod.run_conversation(scenario, paths, tmp_path / "venv")

    assert captured["surface_type"] == "teams_channel"
    assert captured["surface_id"] == "chan-42"


def test_memory_scope_oracle_matches_scenario_surface(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_memory_db(
        data_dir,
        [
            ("prefers dark mode", "personal", "chan-42"),
            ("uses UTC times", "personal", "chan-42"),
        ],
    )
    paths = RunPaths(tmp_path, data_dir, None, None, "", tmp_path)
    evidence = Evidence(paths, DriverOutcome())

    (passed,) = evaluate((_spec("memory_scope", surface_id="chan-42"),), evidence)
    assert passed.passed, passed.detail
    (failed,) = evaluate((_spec("memory_scope", surface_id="lobby"),), evidence)
    assert not failed.passed and "0/2 memories" in failed.detail


def test_memory_scope_oracle_fails_without_records(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_memory_db(data_dir, [])
    paths = RunPaths(tmp_path, data_dir, None, None, "", tmp_path)
    (result,) = evaluate(
        (_spec("memory_scope", surface_id="chan-42"),), Evidence(paths, DriverOutcome())
    )
    assert not result.passed and "no memories recorded" in result.detail


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
    # A throwaway fixture repo must not inherit the developer's ambient
    # ``commit.gpgsign=true``. Signing would launch an interactive pinentry
    # that no test process can answer, so every fixture commit would fail
    # with exit 128. Signatures carry no meaning for a disposable repo.
    _git(workspace, "config", "commit.gpgsign", "false")
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


# ── offline base data + usage roll-up ────────────────────────────────


def test_bundled_base_data_has_offline_manifest() -> None:
    """The bundled fixture satisfies the kernel manifest schema (offline CI)."""
    from e2etest.environment import bundled_base_data_dir

    base = bundled_base_data_dir()
    models = base / "config" / "models.yaml"
    assert models.is_file()
    import yaml

    manifest = yaml.safe_load(models.read_text(encoding="utf-8"))
    assert manifest["mode_models"], "mode_models must be non-empty for the host manifest"
    assert manifest["models"], "at least one model is required"
    assert (base / "config" / "overlays" / "tools" / "web_search" / "web_search.tool.yml").is_file()


def test_prepare_run_seeds_overlays_from_base(tmp_path: Path) -> None:
    """The run dir gets the base overlays tree, so ambient flips survive."""
    from e2etest.environment import prepare_run

    base = tmp_path / "base"
    (base / "config" / "overlays" / "tools" / "web").mkdir(parents=True)
    (base / "config" / "models.yaml").write_text(
        "mode_models: {default: m}\nmodels:\n- model_id: m\n  provider: openai\n",
        encoding="utf-8",
    )
    (base / "config" / "overlays" / "tools" / "web" / "w.tool.yml").write_text(
        "tool_id: w\n", encoding="utf-8"
    )
    scenario = load_scenario(_smoke_scenario_path())
    run_dir = tmp_path / "run"
    paths = prepare_run(scenario, run_dir, tmp_path / "venv", base=base)
    assert (paths.data_dir / "config" / "overlays" / "tools" / "web" / "w.tool.yml").is_file()
    assert (paths.data_dir / "config" / "models.yaml").is_file()
    assert (paths.data_dir / "config" / "policy.yaml").is_file()


def _smoke_scenario_path() -> Path:
    return Path(__file__).resolve().parent / "e2e" / "harness_smoke.scenario.yml"


def test_smoke_scenario_is_hermetic() -> None:
    """The offline smoke scenario parses and only names in-repo packs."""
    from e2etest.packs_venv import _manifest

    scenario = load_scenario(_smoke_scenario_path())
    assert scenario.model_script is not None, "smoke must be scripted (zero-credential)"
    _framework, sources = _manifest(Path(__file__).resolve().parents[1] / "packs.yml")
    unknown = [p for p in scenario.packs if p not in sources]
    assert not unknown, f"smoke packs must be registered: {unknown}"


def test_usage_summary_aggregates_the_durable_log(tmp_path: Path) -> None:
    from e2etest.usage import usage_summary

    log = tmp_path / "token-usage.jsonl"
    rows = [
        {
            "model_id": "gpt-x",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cached_tokens": 40,
            "reasoning_tokens": 5,
            "total_tokens": 120,
            "cost_usd": 0.0015,
        },
        {"model_id": "gpt-x", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"model_id": "glm-5", "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    summary = usage_summary(log)
    assert summary["calls"] == 3
    assert summary["billed_calls"] == 2
    assert summary["prompt_tokens"] == 110
    assert summary["completion_tokens"] == 23
    assert summary["cached_tokens"] == 40
    assert summary["reasoning_tokens"] == 5
    assert summary["total_tokens"] == 133
    assert summary["cost_usd"] == 0.0015
    assert summary["by_model"]["gpt-x"]["calls"] == 2
    assert summary["by_model"]["glm-5"]["total_tokens"] == 13


def test_usage_summary_absent_log_is_zero(tmp_path: Path) -> None:
    """A run without usage evidence reports zero counters, not an error."""
    from e2etest.cli import _usage_for

    paths = SimpleNamespace(data_dir=tmp_path / "no-such")
    summary = _usage_for(paths)
    assert summary == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_usage_summary_tolerates_torn_lines(tmp_path: Path) -> None:
    """A torn final JSONL line is skipped, not a crash (crash-survivable log)."""
    from e2etest.usage import usage_summary

    log = tmp_path / "token-usage.jsonl"
    log.write_text(
        json.dumps(
            {"model_id": "m1", "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        )
        + "\n"
        + '{"torn": ',  # killed mid-append
        encoding="utf-8",
    )
    summary = usage_summary(log)
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 5


# ── CLI: base-data resolution and repeat validation ──────────────────


def test_resolve_base_data_dir_explicit_override_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``base_data_dir`` honors the env override; the default is the live dir."""
    from e2etest.environment import base_data_dir, default_live_base_data_dir

    monkeypatch.setenv("LISA_E2E_BASE_DATA_DIR", "/tmp/custom")
    assert base_data_dir() == Path("/tmp/custom")
    monkeypatch.delenv("LISA_E2E_BASE_DATA_DIR", raising=False)
    assert base_data_dir() == default_live_base_data_dir()


def test_copy_base_data_requires_models_yaml(tmp_path: Path) -> None:
    """A base dir without models.yaml is a loud setup error."""
    from e2etest.environment import _copy_base_data

    config_dir = tmp_path / "run" / "config"
    config_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"models\.yaml"):
        _copy_base_data(tmp_path, config_dir)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(
        "mode_models: {default: m}\n", encoding="utf-8"
    )
    (tmp_path / "config" / "overlays").mkdir()
    _copy_base_data(tmp_path, config_dir)
    assert (config_dir / "overlays").is_dir()


def test_resolve_base_data_dir_scripted_uses_bundled_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from e2etest.cli import _resolve_base_data_dir
    from e2etest.environment import bundled_base_data_dir, default_live_base_data_dir

    monkeypatch.delenv("LISA_E2E_BASE_DATA_DIR", raising=False)
    assert _resolve_base_data_dir(scenario_scripted=True) == bundled_base_data_dir()
    assert _resolve_base_data_dir(scenario_scripted=False) == default_live_base_data_dir()


def test_repeat_rejects_min_pass_above_runs(capfd: pytest.CaptureFixture[str]) -> None:
    from types import SimpleNamespace as NS

    from e2etest.cli import _EXIT_SETUP, _repeat

    args = NS(scenario=Path("x"), runs=2, min_pass=3, runs_dir=Path("."), rebuild_venv=False)
    assert _repeat(args) == _EXIT_SETUP
    err = capfd.readouterr().err
    assert "exceeds" in err


def test_repeat_rejects_zero_min_pass(capfd: pytest.CaptureFixture[str]) -> None:
    from types import SimpleNamespace as NS

    from e2etest.cli import _EXIT_SETUP, _repeat

    args = NS(scenario=Path("x"), runs=2, min_pass=0, runs_dir=Path("."), rebuild_venv=False)
    assert _repeat(args) == _EXIT_SETUP
    assert "at least 1" in capfd.readouterr().err


def _repeat_args(tmp_path: Path, runs: int, min_pass: int | None) -> Any:
    from types import SimpleNamespace as NS

    return NS(
        scenario=Path("unused.yml"),
        runs=runs,
        min_pass=min_pass,
        runs_dir=tmp_path / "runs",
        rebuild_venv=False,
    )


def test_repeat_counts_passes_across_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Each run is its own subprocess; min-pass decides the verdict."""
    import subprocess as sp

    from e2etest import cli

    codes = [0, 1, 0]

    def fake_run(argv: Any, **kwargs: Any) -> Any:
        return sp.CompletedProcess(argv, codes.pop(0), stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    code = cli._repeat(_repeat_args(tmp_path, runs=3, min_pass=2))
    assert code == 0
    out = capfd.readouterr().out
    assert "run 1/3: PASS" in out and "run 2/3: FAIL" in out
    assert "repeat verdict: 2/3 passed (required 2)" in out


def test_repeat_fails_below_min_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    from e2etest import cli

    def fake_run(argv: Any, **kwargs: Any) -> Any:
        return sp.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._repeat(_repeat_args(tmp_path, runs=2, min_pass=1)) == 1


def test_repeat_stops_on_setup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 from a child is a deterministic setup error: stop and surface."""
    import subprocess as sp

    from e2etest import cli

    def fake_run(argv: Any, **kwargs: Any) -> Any:
        return sp.CompletedProcess(argv, 2, stdout="", stderr="no models.yaml")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._repeat(_repeat_args(tmp_path, runs=3, min_pass=3)) == 2
    assert "setup error" in capfd.readouterr().err


def test_driver_uses_override_seam_not_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver binds the script through ``provider_gateways_override``."""
    captured: dict[str, Any] = {}

    class _Session:
        restart_required = False

    class _Runtime:
        session = _Session()

        async def ensure_pool_ready(self) -> None: ...

    def _fake_build(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _Runtime()

    import e2etest.driver as driver_mod

    seen_override: list[Any] = []

    async def _fake_drive(runtime: Any, scenario: Any, outcome: Any) -> None:
        seen_override.append(runtime.session.provider_gateways_override)

    monkeypatch.setattr(driver_mod, "build_runtime", _fake_build)
    monkeypatch.setattr(driver_mod, "inject_packs_venv_site_packages", lambda _p: Path("/x"))
    fake_script = SimpleNamespace(log=[])
    monkeypatch.setattr(driver_mod, "load_script", lambda _p: fake_script)
    monkeypatch.setattr(driver_mod, "_drive", _fake_drive)

    path = _write_scenario(
        tmp_path,
        "session_context: ctx.md\nturns: [x]\nmodel: {script: m.py}\n",
    )
    (tmp_path / "m.py").write_text("def respond(*a, **k): ...\n", encoding="utf-8")
    scenario = load_scenario(path)
    paths = RunPaths(tmp_path, tmp_path / "data", None, None, "", tmp_path)

    driver_mod.run_conversation(scenario, paths, tmp_path / "venv")

    assert captured["surface_type"] == ""  # no surface declared
    # The override rode the seam during the drive and is cleared afterwards.
    assert seen_override == [fake_script]
    assert _Runtime.session.provider_gateways_override is None


def test_main_routes_repeat_subcommand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    from e2etest import cli

    seen: list[Any] = []
    monkeypatch.setattr(cli, "_repeat", lambda args: seen.append(args) or 0)
    code = cli.main(["repeat", "s.yml", "--runs", "2", "--min-pass", "1"])
    assert code == 0 and len(seen) == 1
    assert seen[0].runs == 2 and seen[0].min_pass == 1


# ── driver conversation mechanics ────────────────────────────────────


def _outcome() -> DriverOutcome:
    return DriverOutcome()


def _scenario_with(tmp_path: Path, body: str) -> Any:
    path = _write_scenario(tmp_path, body)
    return load_scenario(path)


def _fake_runtime(tmp_path: Path, script: list[Any]) -> SimpleNamespace:
    """A runtime double whose ``handle_message`` replays a scripted result list."""

    class _Paths(SimpleNamespace):
        pass

    async def _handle(runtime: Any, message: str, transport: Any) -> HostCommandResult:
        index = len(runtime.handled)
        runtime.handled.append(message)
        return script[min(index, len(script) - 1)]

    runtime = SimpleNamespace(handled=[], paths=_Paths(data_dir=tmp_path / "data"))
    runtime.handle = lambda message, transport: _handle(runtime, message, transport)
    return runtime


def test_approval_payload_reads_nested_and_typed_shapes() -> None:
    request_id = "req-1"
    nested = HostCommandResult(
        text="", data={"approval": {"request_id": request_id, "target_id": "coding_commit"}}
    )
    flat = HostCommandResult(text="", data={"schema": "lisa.approval.v1", "request_id": request_id})
    wrong_schema = HostCommandResult(text="", data={"schema": "other.v1", "request_id": "x"})
    assert approval_payload(nested) == {"request_id": request_id, "target_id": "coding_commit"}
    assert approval_payload(flat) == {"schema": "lisa.approval.v1", "request_id": request_id}
    assert approval_payload(wrong_schema) is None
    assert approval_payload(HostCommandResult(text="plain")) is None


@pytest.mark.asyncio
async def test_send_records_exchange_and_respects_max_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each send appends one exchange; the budget aborts past max_messages."""
    scenario = _scenario_with(
        tmp_path, "session_context: ctx.md\nturns: [a, b]\nbudgets: {max_messages: 1}\n"
    )
    runtime = _fake_runtime(tmp_path, [HostCommandResult(text="ok")])
    monkeypatch.setattr(
        "e2etest.driver.handle_message",
        lambda rt, message, transport: runtime.handle(message, transport),
    )
    conversation = _Conversation(runtime, scenario, _outcome())

    exchange = await conversation.send("turn", "a")
    assert exchange.index == 0 and exchange.text == "ok"
    with pytest.raises(_BudgetExceededError, match="max_messages"):
        await conversation.send("turn", "b")


@pytest.mark.asyncio
async def test_follow_up_answers_approval_from_scenario_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval-pending exchange is answered per the scenario's decisions."""
    scenario = _scenario_with(
        tmp_path,
        "session_context: ctx.md\nturns: [go]\napprovals: {approve: [ok_tool], deny: [bad_tool]}\n",
    )
    runtime = _fake_runtime(tmp_path, [HostCommandResult(text="done")])
    monkeypatch.setattr(
        "e2etest.driver.handle_message",
        lambda rt, message, transport: runtime.handle(message, transport),
    )
    conversation = _Conversation(runtime, scenario, _outcome())
    pending = SimpleNamespace(
        text="",
        is_error=False,
        operator_exit="",
        yielded_question="",
        approval={"request_id": "r1", "target_id": "ok_tool"},
    )
    follow = await conversation.follow_up(pending)
    assert follow is not None
    assert conversation.decisions == 1
    assert runtime.handled[-1] == "/approve r1"

    denied = SimpleNamespace(
        text="",
        is_error=False,
        operator_exit="",
        yielded_question="",
        approval={"request_id": "r2", "target_id": "bad_tool"},
    )
    follow = await conversation.follow_up(denied)
    assert runtime.handled[-1] == "/deny r2"

    # Over max_decisions the driver aborts instead of silently deciding.
    scenario2 = _scenario_with(
        tmp_path, "session_context: ctx.md\nturns: [go]\napprovals: {max_decisions: 0}\n"
    )
    conversation2 = _Conversation(runtime, scenario2, _outcome())
    with pytest.raises(_BudgetExceededError, match="max_decisions"):
        await conversation2.follow_up(pending)


@pytest.mark.asyncio
async def test_follow_up_honors_yield_and_stop_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Yield replies and continues are strictly capped (0 = fail loudly)."""
    scenario = _scenario_with(
        tmp_path,
        "session_context: ctx.md\nturns: [x]\nbudgets: {max_continues: 1, max_yield_replies: 1}\n",
    )
    runtime = _fake_runtime(tmp_path, [HostCommandResult(text="ok")])
    monkeypatch.setattr(
        "e2etest.driver.handle_message",
        lambda rt, message, transport: runtime.handle(message, transport),
    )

    conversation = _Conversation(runtime, scenario, _outcome())
    yielded = SimpleNamespace(
        text="q?", is_error=False, operator_exit="", yielded_question="q?", approval=None
    )
    assert await conversation.follow_up(yielded) is not None  # yield_reply sent
    assert conversation.yield_replies == 1
    # Budget spent: no more yield replies.
    assert await conversation.follow_up(yielded) is None

    # A pause with continue budget left sends the continue message.
    conversation2 = _Conversation(runtime, scenario, _outcome())
    paused = SimpleNamespace(
        text="", is_error=False, operator_exit="paused", yielded_question="", approval=None
    )
    followed = await conversation2.follow_up(paused)
    assert followed is not None
    assert followed.kind == "continue"
    assert conversation2.continues == 1
    assert await conversation2.follow_up(paused) is None  # continue budget spent


@pytest.mark.asyncio
async def test_recovery_open_id_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``$RECOVERY_OPEN_ID`` reads the first open record at send time."""
    scenario = _scenario_with(
        tmp_path, "session_context: ctx.md\nturns: ['/skip $RECOVERY_OPEN_ID']\n"
    )
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    (state / "recovery-items.json").write_text(
        json.dumps(
            {
                "a": {"state": "skipped", "work_item_id": "resolved-one"},
                "b": {"state": "stalled", "work_item_id": "open-two"},
            }
        ),
        encoding="utf-8",
    )
    runtime = _fake_runtime(tmp_path, [HostCommandResult(text="ok")])
    monkeypatch.setattr(
        "e2etest.driver.handle_message",
        lambda rt, message, transport: runtime.handle(message, transport),
    )
    conversation = _Conversation(runtime, scenario, _outcome())
    exchange = await conversation.send("turn", "/skip $RECOVERY_OPEN_ID")
    assert exchange.message == "/skip open-two"

    # No open record: the placeholder stays verbatim.
    (state / "recovery-items.json").write_text(
        json.dumps({"a": {"state": "skipped", "work_item_id": "x"}}), encoding="utf-8"
    )
    exchange = await conversation.send("turn", "/skip $RECOVERY_OPEN_ID")
    assert exchange.message == "/skip $RECOVERY_OPEN_ID"

    # Malformed store: left verbatim, never raises.
    (state / "recovery-items.json").write_text("not-json{", encoding="utf-8")
    exchange = await conversation.send("turn", "/skip $RECOVERY_OPEN_ID")
    assert exchange.message == "/skip $RECOVERY_OPEN_ID"
