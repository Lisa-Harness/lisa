"""Spec 3 e2e determinism gate (TURN-REPLAY.md §Acceptance).

The gate records the deterministic ``harness_smoke`` scenario through the
real e2e driver, then replays turn 0 through the exact-mode replay engine
(``runtime.replay``) and asserts a faithful reproduction:

- **positive control** — ``report.ok`` with zero divergences, a matching
  response, and matching effect receipts (the "recorded once and then
  replayed with exact" acceptance);
- **isolation control** — the source run's data root is byte-identical
  before and after replay (the "replay never writes the source data root"
  acceptance);
- **three negative controls** — each corrupts the recording before the
  replay and pins the exact failure mode (a response divergence, a
  tool_result gap divergence, and an archive-exhausted error), so the gate
  cannot rot into a vacuous pass.

The recorded scenario is deterministic (a scripted model), so
record-then-replay in one test is the spec's "recorded once" without a
brittle committed binary fixture. The replay runs in a scratch data root
under the pytest tmp dir, outside the source run, so the isolation test can
hash the entire source data root with nothing to exclude.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from runtime.replay import ReplayReport
from runtime.thread._results import _persist_result
from sdk.thread import ThreadStore, load_result_object

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_SCENARIO = _HERE / "e2e" / "harness_smoke.scenario.yml"


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Record the smoke scenario once via the real ``lisa-e2e`` CLI.

    The recording runs in a subprocess — the same boundary the CLI's own
    ``repeat`` command uses — because process-global registries (host
    commands, catalog caches) make one process unable to host consecutive
    independent runs. The CLI is the exact path CI's e2e-smoke job drives,
    so the recorded artifact is the one production produces.

    Returns the run dir (``<runs>/harness__smoke-scripted/<stamp>``).
    """
    runs_dir = tmp_path_factory.mktemp("gate-runs")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "e2etest.cli",
            "run",
            str(_SCENARIO),
            "--runs-dir",
            str(runs_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the recorded smoke run failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    bundle_dir = runs_dir / "harness__smoke-scripted"
    stamps = sorted(p for p in bundle_dir.iterdir() if p.is_dir())
    assert stamps, f"the CLI wrote no run dir under {bundle_dir}"
    return stamps[-1]


def _source_paths(run_dir: Path) -> tuple[Path, ThreadStore]:
    """Locate the recorded run's data dir and thread store.

    The run dir holds ``data/`` (the runtime's data root) and the thread
    store lives at ``data/data/threads/<session-name>``. The session name
    is runtime-derived, so locate the active segment by glob.
    """
    data_dir = run_dir / "data"
    threads_dir = data_dir / "data" / "threads"
    actives = sorted(threads_dir.rglob("*.current.jsonl"))
    assert actives, f"the recorded run wrote no thread segment under {threads_dir}"
    base = actives[0].with_name(actives[0].name[: -len(".current.jsonl")])
    return data_dir, ThreadStore(base)


def _copy_run(run_dir: Path, tmp_path: Path) -> Path:
    """Copy the recorded run so a control can mutate its copy freely."""
    copy_dir = tmp_path / "run-copy"
    shutil.copytree(run_dir, copy_dir)
    return copy_dir


def _hash_data_root(data_dir: Path) -> dict[str, str]:
    """Sha256 every file under ``data_dir`` in a stable (sorted) order."""
    snapshot: dict[str, str] = {}
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot[str(path.relative_to(data_dir))] = digest
    return snapshot


def _replay(data_dir: Path, store: ThreadStore, tmp_path: Path) -> ReplayReport:
    """Replay turn 0 from the recorded run, in a fresh subprocess.

    The replay boots a full runtime, and process-global registries (host
    commands, catalog caches) make one process unable to host consecutive
    independent replays — the same limitation the CLI's ``repeat`` command
    documents for scenario runs. Each replay therefore runs in its own
    process via ``e2etest.replay_runner``, so every control sees a clean
    runtime exactly as CI would.

    The scratch data root lives under the pytest tmp dir, outside the
    source run, so the isolation test can hash the whole source data root
    with nothing to exclude. The engine seeds the scratch config from the
    source data dir, so the replayed runtime resolves the same models and
    overlays.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "e2etest.replay_runner",
            str(store.base_path),
            "0",
            str(tmp_path / "replay-scratch" / "data"),
            str(data_dir),
            str(data_dir / "config" / "session_context.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the replay subprocess failed (exit {proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return _report_from_json(proc.stdout)


def _report_from_json(raw: str) -> ReplayReport:
    """Rebuild a :class:`ReplayReport` from the runner's JSON dump."""
    from runtime.replay import ReplayDivergence, ReplayGatewayError

    payload = json.loads(raw)
    divergences = [
        ReplayDivergence(
            kind=str(d["kind"]),
            field=str(d.get("field", "")),
            call_id=str(d.get("call_id", "")),
            recorded=d.get("recorded"),
            replayed=d.get("replayed"),
            detail=str(d.get("detail", "")),
        )
        for d in payload.get("divergences", [])
    ]
    gateway_errors = [
        ReplayGatewayError(
            kind=str(g.get("kind", "gateway_script_error")),
            queue=str(g.get("queue", "")),
            message=str(g.get("message", "")),
            call_id=str(g.get("call_id", "")),
        )
        for g in payload.get("gateway_errors", [])
    ]
    return ReplayReport(
        turn_index=int(payload["turn_index"]),
        turn_id=str(payload.get("turn_id", "")),
        mode=str(payload.get("mode", "exact")),
        divergences=divergences,
        gateway_errors=gateway_errors,
        response_match=bool(payload.get("response_match")),
        receipts_match=bool(payload.get("receipts_match")),
        error=str(payload.get("error", "")),
    )


def _assert_report_shape(report: ReplayReport) -> None:
    """Freeze the report's structural contract for every control."""
    assert isinstance(report, ReplayReport)
    assert report.turn_index == 0
    assert report.mode == "exact"


def test_harness_smoke_replays_without_divergence(recorded_run: Path, tmp_path: Path) -> None:
    """The recorded smoke turn replays faithfully: ok, zero divergences."""
    data_dir, store = _source_paths(recorded_run)
    report = _replay(data_dir, store, tmp_path)
    _assert_report_shape(report)
    assert report.error == "", f"replay failed: {report.error}"
    assert report.divergences == [], f"unexpected divergences: {report.divergences}"
    assert report.response_match is True
    assert report.receipts_match is True
    assert report.ok is True


def test_replay_never_writes_the_source_data_root(recorded_run: Path, tmp_path: Path) -> None:
    """The source run's data root is byte-identical before and after replay."""
    data_dir, store = _source_paths(recorded_run)
    before = _hash_data_root(data_dir)
    assert before, "the recorded run produced no files to hash"
    report = _replay(data_dir, store, tmp_path)
    after = _hash_data_root(data_dir)
    assert report.ok is True, (
        f"replay must be faithful to test isolation; "
        f"divergences: {[(d.kind, d.field, d.call_id) for d in report.divergences]}; "
        f"error: {report.error!r}"
    )
    assert before == after, "the replay wrote into the source data root"


def test_corrupted_archived_response_diverges(recorded_run: Path, tmp_path: Path) -> None:
    """Altering the archived response text must surface a response divergence."""
    copy_dir = _copy_run(recorded_run, tmp_path)
    data_dir, store = _source_paths(copy_dir)
    _corrupt_archived_response(store)
    report = _replay(data_dir, store, tmp_path)
    _assert_report_shape(report)
    assert report.response_match is False
    assert report.ok is False
    assert any(d.kind == "response" for d in report.divergences), (
        f"expected a response divergence, got: {report.divergences}"
    )


def test_missing_tool_result_leaves_a_gap(recorded_run: Path, tmp_path: Path) -> None:
    """Losing a recorded tool result must surface a tool_result gap divergence."""
    copy_dir = _copy_run(recorded_run, tmp_path)
    data_dir, store = _source_paths(copy_dir)
    _drop_tool_result_from_transcript(store)
    report = _replay(data_dir, store, tmp_path)
    _assert_report_shape(report)
    assert report.ok is False
    assert any(d.kind == "tool_result" for d in report.divergences), (
        f"expected a tool_result gap, got: {report.divergences}"
    )


def test_truncated_archive_exhausts(recorded_run: Path, tmp_path: Path) -> None:
    """Dropping the last archived call must surface a structured gateway error.

    The kernel converts a gateway error into the turn's response text, so
    exhaustion previously landed only as a response divergence with the
    message buried in the replayed text. FIX-3 pins the structured
    surfacing: the report carries a ``gateway_script_error`` naming the
    turn queue and the exhaustion message, ``report.error`` summarizes
    it, and ``ok`` is false.
    """
    copy_dir = _copy_run(recorded_run, tmp_path)
    data_dir, store = _source_paths(copy_dir)
    _truncate_archive(store)
    report = _replay(data_dir, store, tmp_path)
    _assert_report_shape(report)
    assert report.gateway_errors, (
        f"the truncated archive must surface a structured gateway error; "
        f"report: {json.dumps(dataclasses.asdict(report), default=str)}"
    )
    exhaustion = [g for g in report.gateway_errors if "archive exhausted" in g.message]
    assert exhaustion, f"expected an archive-exhausted gateway error, got: {report.gateway_errors}"
    assert all(g.kind == "gateway_script_error" for g in report.gateway_errors)
    assert any(g.queue == "turn" for g in exhaustion), (
        f"the exhaustion must name the turn queue, got: {report.gateway_errors}"
    )
    assert report.error, "the structured gateway error must also populate report.error"
    assert "archive exhausted" in report.error
    assert report.ok is False


def _corrupt_archived_response(store: ThreadStore) -> None:
    """Rewrite the final archived response's content to something else.

    The turn's response text comes from the *last* archived call (the
    first call's content is empty — it carries the tool calls that drive
    the turn), so corrupting the first call would leave the surfaced text
    untouched and the control would pass vacuously. Loads the persisted
    ``{"content": json.dumps(envelope)}`` snapshot, appends a marker to
    the model text, persists it as a *new* result object (the store's
    integrity check rejects in-place edits), and repoints the recorded
    call's ``response_ref`` at it.
    """
    entry = store.load_turns({0})[0]
    calls = entry["turn_input"]["calls"]
    assert calls, "the recorded turn archived no model calls"
    victim = calls[-1]
    response_ref = victim["response_ref"]
    snapshot = load_result_object(store.base_path, response_ref)
    envelope = json.loads(snapshot["content"])
    envelope["content"] = envelope["content"] + " [corrupted for gate]"
    new_ref = _persist_result(
        store.base_path, {"content": json.dumps(envelope, ensure_ascii=False)}
    )
    victim["response_ref"] = new_ref
    _rewrite_entry(store, entry)


def _drop_tool_result_from_transcript(store: ThreadStore) -> None:
    """Erase the first recorded tool result from the transcript.

    The replay's transcript answerer consumes the persisted ``role="tool"``
    transcript rows; removing the row whose ``tool_call_id`` the second
    model call carries leaves its dispatch unanswerable and must surface
    as a tool_result gap divergence.
    """
    entry = store.load_turns({0})[0]
    tool_rows = [m for m in entry.get("transcript") or [] if m.get("role") == "tool"]
    assert tool_rows, "the recorded turn produced no tool results"
    victim_id = tool_rows[0]["tool_call_id"]
    entry["transcript"] = [m for m in entry["transcript"] if m is not tool_rows[0]]
    _rewrite_entry(store, entry)
    assert victim_id not in {m.get("tool_call_id") for m in entry["transcript"]}


def _truncate_archive(store: ThreadStore) -> None:
    """Remove the last archived call from the recorded turn.

    With one fewer archived call than the replayed turn makes, the archive
    gateway must exhaust on the replayed turn's extra call.
    """
    entry = store.load_turns({0})[0]
    calls = entry["turn_input"]["calls"]
    assert len(calls) >= 2, "the gate needs at least two archived calls"
    entry["turn_input"]["calls"] = calls[:-1]
    _rewrite_entry(store, entry)


def _rewrite_entry(store: ThreadStore, entry: dict) -> None:
    """Persist a mutated turn entry back into the store.

    The store is append-only JSONL; the negative controls rewrite the
    active segment in place: load the whole segment, replace the turn-0
    line with the mutated entry, and rewrite the file. The chain hash is
    recomputed at append time by the runtime, so the rewritten segment
    stays verifiable.
    """
    current = store.paths.current
    lines = current.read_text(encoding="utf-8").splitlines()
    assert lines, "the recorded thread segment is empty"
    replacement = json.dumps(entry, ensure_ascii=False)
    current.write_text("\n".join([*lines[:-1], replacement]) + "\n", encoding="utf-8")
