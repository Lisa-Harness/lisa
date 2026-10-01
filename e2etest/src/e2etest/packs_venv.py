"""Build an isolated packs venv with editable installs of the packs under test."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parents[2] / "packs.yml"
_STAMP = ".e2e-packs"


class PacksVenvError(RuntimeError):
    """Raised when the packs venv cannot be built."""


def _manifest(path: Path) -> tuple[list[Path], dict[str, Path]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    framework = [(base / p).resolve() for p in raw.get("framework") or ()]
    packs = {str(k): (base / v).resolve() for k, v in (raw.get("packs") or {}).items()}
    return framework, packs


def ensure_packs_venv(packs: tuple[str, ...], cache_dir: Path, *, rebuild: bool = False) -> Path:
    """Return a packs venv containing exactly ``packs`` (editable), building it if needed."""
    framework, sources = _manifest(MANIFEST)
    unknown = [p for p in packs if p not in sources]
    if unknown:
        raise PacksVenvError(f"packs not in {MANIFEST}: {', '.join(unknown)}")
    selected = sorted(packs)
    spec = "\n".join([*map(str, framework), *(f"{p}={sources[p]}" for p in selected)])
    key = hashlib.sha256(spec.encode()).hexdigest()[:12]
    venv = cache_dir / f"packs-venv-{key}"
    stamp = venv / _STAMP
    if stamp.is_file() and not rebuild:
        return venv
    if venv.exists():
        shutil.rmtree(venv)
    missing = [str(p) for p in [*framework, *(sources[p] for p in selected)] if not p.is_dir()]
    if missing:
        raise PacksVenvError(f"source dirs missing: {', '.join(missing)}")
    _run(["uv", "venv", "--quiet", "--python", sys.executable, str(venv)])
    editable: list[str] = []
    for path in [*framework, *(sources[p] for p in selected)]:
        editable += ["-e", str(path)]
    _run(["uv", "pip", "install", "--quiet", "--python", str(venv / "bin" / "python"), *editable])
    stamp.write_text(spec + "\n", encoding="utf-8")
    return venv


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise PacksVenvError(f"{' '.join(argv[:3])} failed:\n{proc.stderr.strip()}")
