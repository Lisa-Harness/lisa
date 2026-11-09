"""Tests for file-backed recovery service and /stalls host command."""

from __future__ import annotations

from pathlib import Path

import pytest
from kernel.recovery import RecoveryIdentity, RecoveryRecord
from runtime.handle_message import handle_message
from runtime.recovery import FileRecoveryService, RecoveryStoreCorruptError

from runtime import HostConfig, HostRuntime, NullTransport


def _identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        memory_scope="personal",
    )


def _record(*, target_id: str) -> RecoveryRecord:
    return RecoveryRecord(
        identity=_identity(),
        mode_id="default",
        target_kind="operation",
        target_id=target_id,
    )


def _make_runtime(tmp_path: Path) -> HostRuntime:
    manifest_dir = tmp_path / "config"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "models.yaml").write_text(
        "\n".join(
            [
                "version: 1",
                "mode_models: {default: gemma3:latest}",
                "models:",
                "  - model_id: gemma3:latest",
                "    provider: ollama",
            ]
        ),
        encoding="utf-8",
    )
    return HostRuntime(
        HostConfig.from_env(
            data_dir=str(tmp_path),
            user_slug="user",
            surface_type="personal_dm",
            surface_id="surface",
        )
    )


@pytest.mark.asyncio
async def test_file_recovery_service_begin_and_list_open(tmp_path: Path) -> None:
    service = FileRecoveryService(tmp_path / "state" / "recovery-items.json")

    record = await service.begin(_record(target_id="op_a"))
    assert record.state == "running"

    open_items = await service.list_open(_identity())
    assert len(open_items) == 1
    assert open_items[0].target_id == "op_a"


@pytest.mark.asyncio
async def test_file_recovery_service_settle_succeeded_removes_from_open(tmp_path: Path) -> None:
    service = FileRecoveryService(tmp_path / "state" / "recovery-items.json")
    record = await service.begin(_record(target_id="op_b"))

    settled = await service.settle(
        record.work_item_id,
        _identity(),
        state="succeeded",
        result_preview="ok",
    )
    assert settled is not None
    assert settled.state == "succeeded"

    open_items = await service.list_open(_identity())
    assert open_items == []


@pytest.mark.asyncio
async def test_mark_running_indeterminate_promotes_orphans(tmp_path: Path) -> None:
    service = FileRecoveryService(tmp_path / "state" / "recovery-items.json")
    record = await service.begin(_record(target_id="op_c"))

    changed = await service.mark_running_indeterminate(_identity())
    assert changed == 1

    promoted = await service.get(record.work_item_id, _identity())
    assert promoted is not None
    assert promoted.state == "indeterminate"
    assert promoted.retry_policy == "confirm"


# ---------------------------------------------------------------------------
# Corruption fail-closed (review 2026-11-08, finding 3): the store must
# distinguish file-absent from file-unreadable, and must never let a
# mutation persist a misread (empty/subset) view over the original bytes.
# ---------------------------------------------------------------------------


async def _write_corrupt_store(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("{not json", id="malformed-json"),
        pytest.param("[]", id="not-an-object"),
        pytest.param('{"wi-1": {"identity": {"workspace_id": 4}}}', id="invalid-record"),
    ],
)
async def test_read_corrupt_recovery_store_fails_closed(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "state" / "recovery-items.json"
    await _write_corrupt_store(path, payload)
    service = FileRecoveryService(path)

    with pytest.raises(RecoveryStoreCorruptError):
        await service.list_open(_identity())


@pytest.mark.asyncio
async def test_recovery_mutation_on_corrupt_store_preserves_original(
    tmp_path: Path,
) -> None:
    """A mutation over a corrupt store raises and never rewrites the file."""
    path = tmp_path / "state" / "recovery-items.json"
    original = '{"wi-1": {"identity": {"workspace_id": 4}}}'
    await _write_corrupt_store(path, original)
    service = FileRecoveryService(path)

    with pytest.raises(RecoveryStoreCorruptError):
        await service.begin(_record(target_id="op_d"))

    assert path.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_missing_recovery_store_is_clean_slate(tmp_path: Path) -> None:
    service = FileRecoveryService(tmp_path / "state" / "recovery-items.json")

    assert await service.list_open(_identity()) == []


@pytest.mark.asyncio
async def test_stalls_command_lists_open_recovery_items(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    record = await runtime.services.recovery_service.begin(
        RecoveryRecord(
            identity=RecoveryIdentity(
                workspace_id=runtime.session_plan.manifest.name,
                user_id=runtime.config.user_slug,
                user_slug=runtime.config.user_slug,
                surface_type=runtime.config.surface_type,
                surface_id=runtime.config.surface_id,
                memory_scope=runtime.config.memory_scope,
            ),
            mode_id="default",
            target_kind="operation",
            target_id="test_read_op",
            state="stalled",
            last_error="timeout",
            retry_policy="confirm",
        )
    )
    await runtime.services.recovery_service.settle(
        record.work_item_id,
        RecoveryIdentity(
            workspace_id=runtime.session_plan.manifest.name,
            user_id=runtime.config.user_slug,
            user_slug=runtime.config.user_slug,
            surface_type=runtime.config.surface_type,
            surface_id=runtime.config.surface_id,
            memory_scope=runtime.config.memory_scope,
        ),
        state="stalled",
        last_error="timeout",
        retry_policy="confirm",
    )

    result = await handle_message(runtime, "/stalls", NullTransport())
    assert "Open recovery work items:" in result.text
    assert "test_read_op" in result.text
    assert "state=stalled" in result.text
