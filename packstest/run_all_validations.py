"""Run packstest validate_pack over every pack in the given pack repos.

Each pack runs in an isolated subprocess so per-pack ``lisa_packs`` namespace
packages cannot shadow each other inside one interpreter.

Usage:
    python run_all_validations.py                     # default repo set
    python run_all_validations.py --repo /path ...    # explicit repos
    python run_all_validations.py --pack /path ...    # single-pack checks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_REPOS = [Path("../../lisa-packs"), Path("../../packs-wip")]

WORKER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from packstest.catalog import validate_pack

pack = Path(sys.argv[2])
try:
    report = validate_pack(pack)
except Exception as exc:  # noqa: BLE001
    print(f"CRASH {type(exc).__name__}: {exc}")
    raise SystemExit(2)
print(f"provider_id={report.provider_id!r} tools={len(report.tools)} ok={report.ok}")
for issue in report.issues:
    print(f"ISSUE {issue}")
raise SystemExit(0 if report.ok else 1)
"""


def _packstest_src() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "src" / "packstest").is_dir():
        return here / "src"
    import packstest  # installed distribution

    return Path(packstest.__file__).resolve().parent.parent


def _python_for(packstest_src: Path) -> str:
    """Prefer the workspace venv python (has kernel/sdk deps); else this interpreter."""
    workspace_venv = packstest_src.parents[1] / ".venv" / "bin" / "python"
    if workspace_venv.is_file():
        return str(workspace_venv)
    return sys.executable


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", action="append", type=Path, help="Repo dir to validate every pack under."
    )
    parser.add_argument("--pack", action="append", type=Path, help="One pack checkout.")
    args = parser.parse_args(argv)

    targets: list[Path] = []
    repos = [Path(r) for r in args.repo or ()]
    if repos or not args.pack:
        resolved = [r if r.is_absolute() else here / r for r in (repos or DEFAULT_REPOS)]
        for repo in resolved:
            if not repo.is_dir():
                print(f"repo not found: {repo}", file=sys.stderr)
                return 2
            targets.extend(
                d
                for d in sorted(repo.iterdir())
                if (d / "pyproject.toml").is_file() and d.name != "docs"
            )
    targets.extend(Path(p) for p in args.pack or ())

    packstest_src = _packstest_src()
    python = _python_for(packstest_src)
    all_ok = True
    with tempfile.TemporaryDirectory(prefix="packstest-worker-") as tmp:
        worker_file = Path(tmp) / "packstest_worker.py"
        worker_file.write_text(WORKER, encoding="utf-8")
        for pack in targets:
            proc = subprocess.run(
                [python, str(worker_file), str(packstest_src), str(pack)],
                capture_output=True,
                text=True,
                check=False,
            )
            print(f"\n===== {pack}  rc={proc.returncode}")
            out = (proc.stdout + proc.stderr).strip()
            for line in out.splitlines():
                if line.startswith(("ISSUE", "CRASH", "provider_id")):
                    print(f"  {line}")
            if proc.returncode != 0:
                all_ok = False

    print("\n===== OVERALL:", "OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
