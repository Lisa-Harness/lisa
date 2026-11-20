#!/usr/bin/env python3
"""Validate coherent workspace release metadata and an optional release tag."""

from __future__ import annotations

import argparse
import tomllib
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGES = {
    "kernel": "lisa-harness-kernel",
    "sdk": "lisa-harness-sdk",
    "mcp_server": "lisa-harness-mcp-servers",
    "runtime": "lisa-harness-runtime",
    "sdk/packs/core": "lisa-pack-core",
    "sdk/packs/introspect": "lisa-pack-introspect",
    "sdk/packs/memory": "lisa-pack-memory",
    "sdk/packs/recovery": "lisa-pack-recovery",
    "packstest": "lisa-harness-packstest",
    "e2etest": "lisa-harness-e2e",
}


class ReleaseMetadataError(ValueError):
    """Raised when release metadata is not safe to build or publish."""


def _project(root: Path, directory: Path) -> dict[str, object]:
    with (root / directory / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _has_dated_release_heading(changelog: list[str], version: str) -> bool:
    prefix = f"## [{version}] - "
    for line in changelog:
        if not line.startswith(prefix):
            continue
        try:
            date.fromisoformat(line.removeprefix(prefix))
        except ValueError:
            continue
        return True
    return False


def validate(root: Path, *, tag: str | None = None) -> str:
    """Return the validated version, or raise with every metadata mismatch."""
    version = str(_project(root, Path("."))["version"])
    errors: list[str] = []
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    if not _has_dated_release_heading(changelog, version):
        errors.append(f"CHANGELOG.md has no dated release section for {version}")

    for directory, expected_name in _PACKAGES.items():
        project = _project(root, Path(directory))
        name = project["name"]
        package_version = project["version"]
        if name != expected_name:
            errors.append(f"{directory}: expected distribution {expected_name!r}, found {name!r}")
        if package_version != version:
            errors.append(
                f"{expected_name}: version {package_version!r} "
                f"does not match workspace version {version!r}"
            )

    if tag is not None and tag != f"v{version}":
        errors.append(f"release tag {tag!r} does not match workspace version {version!r}")

    if errors:
        raise ReleaseMetadataError("\n".join(errors))
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT, help="Repository root")
    parser.add_argument("--tag", help="Expected v<version> release tag")
    args = parser.parse_args(argv)
    try:
        version = validate(args.root, tag=args.tag)
    except (OSError, KeyError, ReleaseMetadataError, tomllib.TOMLDecodeError) as exc:
        parser.exit(1, f"release metadata: {exc}\n")
    print(f"release metadata valid for {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
