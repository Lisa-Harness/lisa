"""Outcome assertions evaluated against a finished run's on-disk evidence."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from string import Template
from typing import Any

from e2etest.driver import DriverOutcome
from e2etest.environment import RunPaths, git
from e2etest.scenario import OracleSpec


@dataclass(frozen=True)
class OracleResult:
    phase: str
    name: str
    passed: bool
    detail: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Lazy views over everything a run left behind."""

    paths: RunPaths
    outcome: DriverOutcome

    @cached_property
    def thread_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted((self.paths.data_dir / "data" / "threads").glob("*.current.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    @cached_property
    def receipts(self) -> list[dict[str, Any]]:
        return [r for e in self.thread_entries for r in e.get("effect_receipts") or ()]

    @cached_property
    def artifacts(self) -> list[dict[str, Any]]:
        path = self.paths.data_dir / "state" / "code-change-artifacts.json"
        if not path.is_file():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.values() if isinstance(raw, dict) else raw
        return sorted(items, key=lambda a: str(a.get("updated_at", "")))

    @cached_property
    def memories(self) -> list[dict[str, Any]]:
        db = self.paths.data_dir / "data" / "memory.db"
        if not db.is_file():
            return []
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT memory_kind, status, content FROM memories").fetchall()
        return [dict(r) for r in rows]

    def committed_artifact(self) -> dict[str, Any] | None:
        committed = [a for a in self.artifacts if a.get("commit_sha")]
        return committed[-1] if committed else None


Check = Callable[[Evidence, dict[str, Any]], tuple[bool, str]]
ORACLES: dict[str, Check] = {}


def oracle(name: str) -> Callable[[Check], Check]:
    def register(fn: Check) -> Check:
        ORACLES[name] = fn
        return fn

    return register


def _run_argv(argv: list[str], cwd: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def _tail(text: str, lines: int = 8) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _expand(ev: Evidence, argv: list[str], **extra: str) -> list[str]:
    """Expand ``${SCENARIO_DIR}``/``${RUN_DIR}`` (plus ``extra``) in ``argv``."""
    values = {
        "SCENARIO_DIR": str(ev.paths.scenario_dir or ""),
        "RUN_DIR": str(ev.paths.run_dir),
        **extra,
    }
    return [Template(str(a)).safe_substitute(values) for a in argv]


def _run_at(
    ev: Evidence,
    sha: str,
    argv: list[str],
    timeout_s: float,
    restore: list[str] | None = None,
) -> tuple[int, str]:
    """Run ``argv`` in a throwaway detached checkout of ``sha`` (never Lisa's worktree).

    ``restore`` paths are reset to the fixture's initial commit first, so the
    committed source is judged against the original tests, not edited ones.
    """
    workspace = ev.paths.workspace
    assert workspace is not None
    with tempfile.TemporaryDirectory(dir=ev.paths.run_dir, prefix="verify-") as tmp:
        checkout = Path(tmp) / "tree"
        git(workspace, "worktree", "add", "--quiet", "--detach", str(checkout), sha)
        try:
            if restore:
                git(checkout, "checkout", ev.paths.initial_main_sha, "--", *restore)
            proc = _run_argv(_expand(ev, argv), checkout, timeout_s)
        finally:
            git(workspace, "worktree", "remove", "--force", str(checkout))
    return proc.returncode, _tail(proc.stdout + "\n" + proc.stderr)


# ── conversation hygiene ────────────────────────────────────────────────


@oracle("not_aborted")
def _not_aborted(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    return not ev.outcome.aborted, ev.outcome.aborted or "completed within budgets"


@oracle("no_turn_errors")
def _no_turn_errors(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    bad = [e for e in ev.outcome.exchanges if e.is_error or e.errors]
    detail = "; ".join(f"#{e.index}: {(e.errors or [e.text])[0][:200]}" for e in bad)
    return not bad, detail or "no errored exchanges"


@oracle("no_operator_exit")
def _no_operator_exit(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    bad = [e for e in ev.outcome.exchanges if e.operator_exit]
    detail = "; ".join(f"#{e.index}: {e.operator_exit}: {e.text[:160]}" for e in bad)
    return not bad, detail or "no pause/stop/terminate exits"


@oracle("no_unanswered_yield")
def _no_unanswered_yield(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    last = ev.outcome.exchanges[-1] if ev.outcome.exchanges else None
    if last and last.yielded_question:
        return False, f"ended asking: {last.yielded_question[:200]}"
    return True, "conversation did not end on a question"


@oracle("max_exchanges")
def _max_exchanges(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    count = len(ev.outcome.exchanges)
    return count <= int(args["n"]), f"{count} exchanges (limit {args['n']})"


@oracle("no_runtime_errors")
def _no_runtime_errors(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    """Error-level runtime log records; failed operations often surface only here."""
    ignore = [re.compile(p) for p in args.get("ignore") or ()]
    log = ev.paths.data_dir / "logs" / "runtime.jsonl"
    hits: list[str] = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            msg = str(record.get("msg", ""))
            if record.get("level") == "error" and not any(p.search(msg) for p in ignore):
                hits.append(f"{record.get('ts', '')[11:19]} {record.get('name')}: {msg[:300]}")
    return not hits, "\n".join(hits) or "no error-level runtime log records"


# ── capability + knowledge evidence ─────────────────────────────────────


@oracle("receipt")
def _receipt(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    wanted = (
        set(args["capability"]) if isinstance(args["capability"], list) else {args["capability"]}
    )
    hits = [r for r in ev.receipts if r.get("capability_id") in wanted]
    need = int(args.get("min", 1))
    seen = sorted({str(r.get("capability_id")) for r in ev.receipts})
    return len(hits) >= need, f"{len(hits)} receipt(s) for {sorted(wanted)}; seen: {seen}"


@oracle("memory_recorded")
def _memory_recorded(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    rows = ev.memories
    terms = [str(t).lower() for t in args.get("mentions_any") or ()]
    matching = [r for r in rows if not terms or any(t in str(r["content"]).lower() for t in terms)]
    need = int(args.get("min", 1))
    sample = "; ".join(str(r["content"])[:120] for r in matching[:3])
    return len(matching) >= need, f"{len(matching)}/{len(rows)} memories match {terms}. {sample}"


# ── code-change outcome (coding pack WorkProduct + git) ─────────────────


@oracle("workproduct_committed")
def _workproduct_committed(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    art = ev.committed_artifact()
    states = [f"{a.get('artifact_id', '')[:8]}={a.get('status')}" for a in ev.artifacts]
    if art is None:
        return False, f"no artifact with a commit_sha; artifacts: {states or 'none'}"
    return (
        True,
        f"{art['artifact_id'][:8]} status={art.get('status')} commit={art['commit_sha'][:10]}",
    )


@oracle("branch_prefix")
def _branch_prefix(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    art = ev.committed_artifact()
    if art is None:
        return False, "no committed artifact"
    branch = str(art.get("branch", ""))
    return branch.startswith(str(args["prefix"])), f"branch={branch}"


@oracle("main_untouched")
def _main_untouched(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    workspace = ev.paths.workspace
    if workspace is None:
        return False, "scenario has no fixture repo"
    head = git(workspace, "rev-parse", "main")
    remote = git(workspace, "ls-remote", "origin", "refs/heads/main").split()[0]
    initial = ev.paths.initial_main_sha
    ok = head == initial and remote == initial
    return ok, f"local main={head[:10]} remote main={remote[:10]} initial={initial[:10]}"


@oracle("diff_touches")
def _diff_touches(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    art = ev.committed_artifact()
    if art is None or ev.paths.workspace is None:
        return False, "no committed artifact"
    changed = git(
        ev.paths.workspace, "diff", "--name-only", ev.paths.initial_main_sha, art["commit_sha"]
    ).splitlines()
    wanted = [str(p) for p in args["paths"]]
    allowed = set(wanted) | {str(p) for p in args.get("allow", ())}
    missing = [p for p in wanted if p not in changed]
    unexpected = [p for p in changed if p not in allowed] if args.get("exact") else []
    ok = not missing and not unexpected
    return ok, f"changed={changed} missing={missing} unexpected={unexpected}"


@oracle("checks_current")
def _checks_current(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    art = ev.committed_artifact()
    if art is None or ev.paths.workspace is None:
        return False, "no committed artifact"
    tree = git(ev.paths.workspace, "rev-parse", f"{art['commit_sha']}^{{tree}}")
    problems: list[str] = []
    for check_id in art.get("required_check_ids") or ():
        rec = (art.get("checks") or {}).get(check_id)
        if rec is None:
            problems.append(f"{check_id}: missing")
        elif rec.get("status") != "passed":
            problems.append(f"{check_id}: {rec.get('status')}")
        elif rec.get("revision") != art.get("revision"):
            problems.append(f"{check_id}: stale revision {rec.get('revision')}")
        elif rec.get("checked_tree_sha") != tree:
            problems.append(
                f"{check_id}: tree {str(rec.get('checked_tree_sha'))[:10]} != {tree[:10]}"
            )
    if not art.get("required_check_ids"):
        problems.append("no required_check_ids configured")
    return not problems, "; ".join(problems) or f"required checks passed on tree {tree[:10]}"


@oracle("worktree_clean")
def _worktree_clean(ev: Evidence, _args: dict[str, Any]) -> tuple[bool, str]:
    art = ev.committed_artifact()
    if art is None:
        return False, "no committed artifact"
    worktree = Path(str(art.get("worktree_path", "")))
    if not worktree.is_dir():
        return False, f"worktree missing: {worktree}"
    status = git(worktree, "status", "--porcelain")
    head = git(worktree, "rev-parse", "HEAD")
    ok = not status and head == art["commit_sha"]
    return ok, f"HEAD={head[:10]} commit={art['commit_sha'][:10]} dirty={status[:200]!r}"


@oracle("baseline_fails")
def _baseline_fails(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    code, out = _run_at(
        ev, ev.paths.initial_main_sha, list(args["argv"]), args.get("timeout_s", 120)
    )
    return code != 0, f"exit={code} on fixture baseline\n{out}"


@oracle("independent_check")
def _independent_check(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    art = ev.committed_artifact()
    if art is None:
        return False, "no committed artifact"
    restore = [str(p) for p in args.get("restore", ())]
    code, out = _run_at(
        ev, art["commit_sha"], list(args["argv"]), args.get("timeout_s", 120), restore
    )
    restored = f" restored={restore}" if restore else ""
    return code == 0, f"exit={code} at {art['commit_sha'][:10]}{restored}\n{out}"


# ── data-dir outcome (packs whose work product lives in lisa-storage) ───


@oracle("data_check")
def _data_check(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    """Run ``argv`` in the storage sandbox root; ``${SCENARIO_DIR}``/``${FILES}`` expand."""
    files = ev.paths.files_root
    if not files.is_dir():
        return False, f"storage sandbox missing: {files}"
    argv = _expand(ev, list(args["argv"]), FILES=str(files))
    proc = _run_argv(argv, files, args.get("timeout_s", 120))
    return proc.returncode == 0, f"exit={proc.returncode}\n{_tail(proc.stdout + proc.stderr)}"


def _recovery_records(ev: Evidence) -> dict[str, dict[str, Any]]:
    """Load the run's ``recovery-items.json``; empty when the run wrote none."""
    path = ev.paths.data_dir / "state" / "recovery-items.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _matching_recovery_records(ev: Evidence, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter recovery records by ``state``/``retry_policy``/``target_id`` args."""
    records = _recovery_records(ev).values()
    state = args.get("state")
    retry_policy = args.get("retry_policy")
    target_id = args.get("target_id")
    matched = []
    for record in records:
        if state is not None and record.get("state") != state:
            continue
        if retry_policy is not None and record.get("retry_policy") != retry_policy:
            continue
        if target_id is not None and record.get("target_id") != target_id:
            continue
        matched.append(record)
    return matched


@oracle("recovery_record")
def _recovery_record(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    """At least ``min`` recovery records match the given state filters."""
    matched = _matching_recovery_records(ev, args)
    minimum = int(args.get("min", 1))
    if len(matched) >= minimum:
        summary = ", ".join(
            f"{r.get('work_item_id', '?')}[{r.get('state')}/{r.get('retry_policy')}]"
            for r in matched[:3]
        )
        return True, f"{len(matched)} matching record(s): {summary}"
    total = len(_recovery_records(ev))
    filters = {k: args[k] for k in ("state", "retry_policy", "target_id") if args.get(k)}
    return (
        False,
        f"{len(matched)} matching (need {minimum}); {total} total records; filters={filters}",
    )


@oracle("recovery_settled")
def _recovery_settled(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    """At least ``min`` records sit in the given terminal state.

    Distinct from ``recovery_record`` semantically: a settlement assertion
    (the operator resolved the stall), not an existence assertion. Reads
    the same evidence.
    """
    return _recovery_record(ev, args)


@oracle("exchange_contains")
def _exchange_contains(ev: Evidence, args: dict[str, Any]) -> tuple[bool, str]:
    """Some exchange's response text contains the given ``text``.

    Asserts from the driver-recorded conversation. The right evidence for
    state that is real-time (``/stalls`` output shows a stalled record even
    though the final ``recovery-items.json`` shows it settled).
    """
    needle = str(args.get("text") or "")
    if not needle:
        return False, "no 'text' arg given"
    for exchange in ev.outcome.exchanges:
        if needle in exchange.text:
            return True, f"found in exchange[{exchange.index}] {exchange.kind}: {needle[:60]}..."
    return False, f"no exchange contains {needle[:80]!r}; {len(ev.outcome.exchanges)} exchange(s)"


def evaluate(specs: tuple[OracleSpec, ...], evidence: Evidence) -> list[OracleResult]:
    """Run every oracle; an oracle that raises fails with the exception as its detail."""
    results: list[OracleResult] = []
    for spec in specs:
        check = ORACLES.get(spec.name)
        if check is None:
            results.append(OracleResult(spec.phase, spec.name, False, "unknown oracle", spec.args))
            continue
        try:
            passed, detail = check(evidence, spec.args)
        except Exception as exc:  # pylint: disable=broad-except
            passed, detail = False, f"oracle raised {type(exc).__name__}: {exc}"
        results.append(OracleResult(spec.phase, spec.name, passed, detail, spec.args))
    return results
