"""``lisa-e2e``: run one scenario end to end and write a verdict bundle."""

from __future__ import annotations

import argparse
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
    run = sub.add_parser("run", help="Run a scenario against a live model.")
    run.add_argument("scenario", type=Path)
    run.add_argument("--runs-dir", type=Path, default=_REPO_ROOT / ".e2e" / "runs")
    run.add_argument("--rebuild-venv", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    venv = ensure_packs_venv(
        scenario.packs, _REPO_ROOT / ".e2e" / "cache", rebuild=args.rebuild_venv
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.runs_dir / scenario.scenario_id.replace("/", "__") / stamp).resolve()
    paths = prepare_run(scenario, run_dir, venv)
    print(f"run dir: {run_dir}", flush=True)
    outcome = run_conversation(scenario, paths, venv)
    evidence = Evidence(paths, outcome)
    payload = verdict(scenario, evidence, evaluate(scenario.oracles, evidence))
    path = write_bundle(run_dir, payload, evidence)
    for result in payload["results"]:
        mark = "PASS" if result["passed"] else "FAIL"
        first = result["detail"].splitlines()[0] if result["detail"] else ""
        print(f"{mark} {result['phase']}.{result['name']}: {first}")
    print(f"verdict: {'PASS' if payload['passed'] else 'FAIL'} → {path}")
    return 0 if payload["passed"] else _EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_SETUP


if __name__ == "__main__":
    sys.exit(main())
