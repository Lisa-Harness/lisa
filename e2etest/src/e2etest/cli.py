"""``lisa-e2e``: run one scenario end to end and write a verdict bundle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from e2etest.bundle import verdict, write_bundle
from e2etest.driver import run_conversation
from e2etest.environment import prepare_run
from e2etest.oracles import Evidence, evaluate
from e2etest.packs_venv import ensure_packs_venv
from e2etest.scenario import load_scenario

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXIT_FAIL = 1
_EXIT_SETUP = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lisa-e2e", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a scenario (live model, or scripted when declared).")
    run.add_argument("scenario", type=Path)
    run.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / ".e2e" / "runs")
    run.add_argument("--rebuild-venv", action="store_true")
    repeat = sub.add_parser(
        "repeat",
        help="Run one live scenario N times in isolated processes; require --min-pass successes.",
    )
    repeat.add_argument("scenario", type=Path)
    repeat.add_argument("--runs", type=int, default=3, help="Total independent runs (default 3).")
    repeat.add_argument(
        "--min-pass", type=int, default=None, help="Required passes (default: all runs)."
    )
    repeat.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / ".e2e" / "runs")
    repeat.add_argument("--rebuild-venv", action="store_true")
    return parser


def _resolve_base_data_dir(scenario_scripted: bool) -> Path:
    """The base data dir for one run.

    ``LISA_E2E_BASE_DATA_DIR`` always wins. Scripted-model scenarios fall
    back to the bundled offline fixture, so zero-credential runs (CI) work
    out of the box; live scenarios need the operator's roster (``~/.lisa``).
    """
    override = os.getenv("LISA_E2E_BASE_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if scenario_scripted:
        from e2etest.environment import bundled_base_data_dir

        return bundled_base_data_dir()
    from e2etest.environment import default_live_base_data_dir

    return default_live_base_data_dir()


def _usage_for(paths: object) -> dict[str, object]:
    """Summarize the run's durable token-usage log; zero counters when absent."""
    from e2etest.usage import usage_summary

    usage_path = paths.data_dir / "logs" / "token-usage.jsonl"  # type: ignore[attr-defined]
    if not isinstance(usage_path, Path) or not usage_path.is_file():
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    return usage_summary(usage_path)


def _run_scenario_once(
    scenario_path: Path,
    *,
    runs_dir: Path,
    rebuild_venv: bool,
    tag_suffix: str = "",
) -> int:
    scenario = load_scenario(scenario_path)
    scripted = scenario.model_script is not None
    base = _resolve_base_data_dir(scenario_scripted=scripted)
    venv = ensure_packs_venv(scenario.packs, _REPO_ROOT / ".e2e" / "cache", rebuild=rebuild_venv)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bundle_id = scenario.scenario_id.replace("/", "__") + tag_suffix
    run_dir = (runs_dir / bundle_id / stamp).resolve()
    paths = prepare_run(scenario, run_dir, venv, base=base)
    print(f"run dir: {run_dir}", flush=True)
    outcome = run_conversation(scenario, paths, venv)
    evidence = Evidence(paths, outcome)
    payload = verdict(scenario, evidence, evaluate(scenario.oracles, evidence))
    usage = _usage_for(paths)
    payload["usage"] = usage
    path = write_bundle(run_dir, payload, evidence)
    for result in payload["results"]:
        mark = "PASS" if result["passed"] else "FAIL"
        first = result["detail"].splitlines()[0] if result["detail"] else ""
        print(f"{mark} {result['phase']}.{result['name']}: {first}")
    print(
        "usage: {calls} model call(s), {prompt} prompt / {completion} completion tokens"
        " (cost {cost})".format(
            calls=usage["calls"],
            prompt=usage["prompt_tokens"],
            completion=usage["completion_tokens"],
            cost=usage["cost_usd"],
        )
    )
    print(f"verdict: {'PASS' if payload['passed'] else 'FAIL'} → {path}")
    return 0 if payload["passed"] else _EXIT_FAIL


def _run(args: argparse.Namespace) -> int:
    try:
        return _run_scenario_once(
            args.scenario, runs_dir=args.runs_dir, rebuild_venv=args.rebuild_venv
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_SETUP


def _repeat(args: argparse.Namespace) -> int:
    """Run one live scenario N times, each in its own subprocess.

    Process-global registries (host commands, catalog caches) make one
    process unable to host consecutive independent runs; the subprocess
    boundary also gives every attempt a fresh run dir.
    """
    min_pass = args.min_pass if args.min_pass is not None else args.runs
    if min_pass > args.runs:
        print(f"--min-pass {min_pass} exceeds --runs {args.runs}", file=sys.stderr)
        return _EXIT_SETUP
    if min_pass < 1:
        print("--min-pass must be at least 1", file=sys.stderr)
        return _EXIT_SETUP
    passes = 0
    runs: list[dict[str, int]] = []
    for index in range(1, args.runs + 1):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "e2etest.cli",
                "run",
                str(args.scenario),
                "--runs-dir",
                str(args.runs_dir),
                *(["--rebuild-venv"] if args.rebuild_venv else []),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == _EXIT_SETUP:
            # A setup error would repeat identically (bad scenario, missing
            # roster, venv failure); stop and surface it instead of burning
            # the remaining runs.
            print(f"run {index}: setup error (exit 2); stopping repeat", file=sys.stderr)
            print(proc.stderr.rstrip(), file=sys.stderr)
            return _EXIT_SETUP
        runs.append({"run": index, "exit": proc.returncode})
        passed = proc.returncode == 0
        if passed:
            passes += 1
        print(f"run {index}/{args.runs}: {'PASS' if passed else 'FAIL'}")
    print(f"repeat verdict: {passes}/{args.runs} passed (required {min_pass})")
    return 0 if passes >= min_pass else _EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "repeat":
        try:
            return _repeat(args)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _EXIT_SETUP
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
