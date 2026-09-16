"""Run packstest validate_pack over every pack in both pack repos.

Each pack runs in an isolated subprocess so per-pack ``lisa_packs`` namespace
packages cannot shadow each other inside one interpreter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOS = [Path("/home/adam/Code/lisa-packs"), Path("/home/adam/Code/lisa-packs-wip")]

WORKER = r'''
import sys
from pathlib import Path
sys.path.insert(0, "/home/adam/Code/lisa/packstest/src")
from packstest.catalog import validate_pack

pack = Path(sys.argv[1])
try:
    report = validate_pack(pack)
except Exception as exc:  # noqa: BLE001
    print(f"CRASH {type(exc).__name__}: {exc}")
    raise SystemExit(2)
print(f"provider_id={report.provider_id!r} tools={len(report.tools)} ok={report.ok}")
for issue in report.issues:
    print(f"ISSUE {issue}")
raise SystemExit(0 if report.ok else 1)
'''

WORKER_FILE = Path("/tmp/packstest_worker.py")
WORKER_FILE.write_text(WORKER, encoding="utf-8")

PY = "/home/adam/Code/lisa/.venv/bin/python"

all_ok = True
for repo in REPOS:
    for pack_dir in sorted(repo.iterdir()):
        if not (pack_dir / "pyproject.toml").is_file() or pack_dir.name == "docs":
            continue
        proc = subprocess.run(
            [PY, str(WORKER_FILE), str(pack_dir)],
            capture_output=True,
            text=True,
        )
        print(f"\n===== {pack_dir.name}  ({repo.name}) rc={proc.returncode}")
        out = (proc.stdout + proc.stderr).strip()
        for line in out.splitlines():
            if (
                line.startswith("ISSUE")
                or line.startswith("CRASH")
                or line.startswith("provider_id")
            ):
                print(f"  {line}")
        if proc.returncode != 0:
            all_ok = False

print("\n===== OVERALL:", "OK" if all_ok else "FAILURES PRESENT")