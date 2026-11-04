"""Pack-validation eval: the memory pack stays structurally healthy.

Runs under the framework's ``eval`` marker (deterministic, no model calls).
Validates the pack from source via ``packstest.catalog.validate_pack``:
distribution metadata, entry points, content subdirs, and in-process
importability. First eval any pack edit should run — it fails fast and
names the exact file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from packstest.catalog import validate_pack

PACK_PATH = Path(__file__).resolve().parents[2]  # .../sdk/packs/memory

EXPECTED_OPERATIONS = {
    "synthesize_and_store_learning",
    "recall_memory",
    "dispute_memory",
    "mine_thread_segment",
}


def _operation_ids() -> set[str]:
    ops_dir = PACK_PATH / "src" / "sdk" / "packs" / "memory" / "operations" / "memory"
    ids: set[str] = set()
    for path in ops_dir.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        if "operation_id:" not in text:
            continue
        ids.add(text.split("operation_id:")[1].split()[0])
    return ids


@pytest.mark.eval
def test_memory_pack_validates_from_source() -> None:
    """Entry points load, content subdirs present, modules import in-process."""

    report = validate_pack(PACK_PATH)
    assert report.ok, report.summary()


@pytest.mark.eval
def test_memory_pack_operation_catalog_complete() -> None:
    """Every expected operation is authored; the retired
    ``consolidate_semantic`` operation must not resurface."""

    operation_ids = _operation_ids()
    assert "consolidate_semantic" not in operation_ids, (
        "consolidate_semantic was removed with the pressure-consolidation "
        "path (write-time dedup superseded it)"
    )
    missing = EXPECTED_OPERATIONS - operation_ids
    assert not missing, f"operations missing from catalog: {sorted(missing)}"
    unexpected = operation_ids - EXPECTED_OPERATIONS
    assert not unexpected, f"unexpected operations: {sorted(unexpected)}"


@pytest.mark.eval
def test_memory_pack_entry_points_load() -> None:
    """The content-root and slice-provider entry points resolve from source."""

    report = validate_pack(PACK_PATH)
    assert "lisa.extensions.content_roots" in report.entry_points
    for group, entries in report.entry_points.items():
        for name, value in entries.items():
            assert value is not None, f"{group}:{name} failed to load"
