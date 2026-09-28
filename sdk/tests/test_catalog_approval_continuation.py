from __future__ import annotations

import json
from pathlib import Path

import pytest
from kernel.context.mcp_catalog import _deserialize_mode
from sdk.content import build_mode_catalog_json, resolve_mode_roots


def _write_mode(root: Path, extra_line: str = "") -> None:
    mode_dir = root / "modes" / "demo"
    mode_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "mode_id: demo",
        "start_stage_id: none",
        "mode_kind: chat",
        'description: "demo"',
    ]
    if extra_line:
        lines.append(extra_line)
    lines += ["---", "", "# Demo", ""]
    (mode_dir / "_mode.md").write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize(
    ("extra_line", "expected"),
    [("", "report"), ("approval_continuation: resume", "resume")],
)
def test_approval_continuation_survives_catalog_round_trip(
    tmp_path: Path, extra_line: str, expected: str
) -> None:
    _write_mode(tmp_path, extra_line)

    catalog = json.loads(build_mode_catalog_json(resolve_mode_roots([tmp_path])))

    assert _deserialize_mode(catalog["demo"]).approval_continuation == expected
