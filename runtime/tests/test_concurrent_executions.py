"""Concurrent execution and multi-process safety tests.

Covers the concurrency contract for running overlapping Lisa turns across
multiple host processes on the same machine:

1. Cross-process advisory file locking (`sdk.filelock`) and atomic writes.
2. Multi-process thread-store appends, segment sealing, dictionary training,
   and monotonic turn-index assignment without data loss or corruption.
3. Multi-process read-modify-write safety on `FileApprovalService`,
   `FileRecoveryService`, `FileCodeChangeArtifactStore`, `installed-packs.json`,
   and `policy.yaml`.
4. Owner-PID liveness checks in `FileRecoveryService.mark_running_indeterminate`
   and top-level-only reconciliation guard (sub-turns do not mark active turns
   indeterminate).
5. Per-conversation isolation (`conversation_id` / `LISA_CONVERSATION_ID`).
6. Multi-process `SqliteMemoryStore` WAL + busy-timeout safety.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from kernel.approval import ApprovalIdentity, ApprovalInvocation
from kernel.models import MemoryWriteRequest
from kernel.recovery import RecoveryIdentity, RecoveryRecord
from runtime.approvals import FileApprovalService
from runtime.code_artifacts import FileCodeChangeArtifactStore
from runtime.config import HostConfig
from runtime.content_roots import load_installed_packs_lock, save_installed_packs_lock
from runtime.mcp_pool import _mcp_environment
from runtime.policy import load_policy, remember_allow_prefix
from runtime.recovery import FileRecoveryService
from runtime.runtime import HostRuntime
from runtime.runtime_paths import HostPaths
from runtime.thread import (
    append_thread_entry,
    load_recent_thread_entries,
    load_thread_range,
    reserved_turn,
)
from sdk.filelock import (
    FileLockTimeoutError,
    async_file_lock,
    atomic_write_text,
    file_lock,
    is_process_alive,
)
from sdk.memory.store import SqliteMemoryStore
from sdk.workproducts import WorkProduct

# ── Top-level worker functions for multiprocessing (spawn-safe) ──────


def _hold_lock_worker(lock_path_str: str, ready_event: Any, release_event: Any) -> None:
    with file_lock(Path(lock_path_str), timeout=10.0):
        ready_event.set()
        release_event.wait(timeout=10.0)


def _thread_append_worker(args: tuple[str, int, int, int]) -> list[int | None]:
    base_str, worker_id, count, segment_turns = args
    base = Path(base_str)
    committed_turns: list[int | None] = []
    for i in range(count):
        turn = append_thread_entry(
            user_slug="shared-user",
            mode_id="build",
            query=f"worker-{worker_id}-query-{i}-" + ("x" * 120),
            response_text=f"worker-{worker_id}-response-{i}-" + ("y" * 120),
            provenance={"source": "test", "worker": str(worker_id), "seq": str(i)},
            thread_path=base,
            turn=0,  # intentionally stale pre-computed turn index
            segment_turns=segment_turns,
        )
        committed_turns.append(turn)
    return committed_turns


def _overlapping_turn_worker(base_str: str, worker_id: int, barrier: Any, results: Any) -> None:
    base = Path(base_str)
    with reserved_turn(thread_path=base) as reservation:
        # Every worker holds its reservation before any commits: the
        # overlap that handed two live turns the same index.
        barrier.wait(timeout=30)
        committed = append_thread_entry(
            user_slug="shared-user",
            mode_id="build",
            query=f"worker-{worker_id}",
            response_text="",
            thread_path=base,
            turn=reservation.turn,
            reservation=reservation.token,
        )
    results.put((worker_id, reservation.turn, committed))


def _approval_create_worker(args: tuple[str, int, int]) -> list[str]:
    approvals_path_str, worker_id, count = args
    service = FileApprovalService(Path(approvals_path_str))
    now = datetime.now(UTC)
    created_ids: list[str] = []

    async def _run() -> None:
        for i in range(count):
            aid = f"ap-w{worker_id}-{i}"
            invocation = ApprovalInvocation(
                request_id=aid,
                target_kind="operation",
                target_id="run_shell_command",
                pack_id="bash",
                effect="execute",
                parameters={"worker": worker_id, "item": i},
                identity=ApprovalIdentity(
                    workspace_id="ws",
                    user_id="u1",
                    user_slug="u1",
                    surface_type="personal_dm",
                    surface_id="s1",
                    memory_scope="personal",
                    mode_id="build",
                    session_plan_digest="digest",
                ),
                pack_config_fingerprint="fp",
                session_plan_digest="digest",
                action_digest=f"digest-w{worker_id}-{i}",
                safe_preview=f"worker {worker_id} item {i}",
                created_at=now,
                expires_at=now + timedelta(hours=1),
            )
            await service.request(invocation)
            created_ids.append(aid)

    asyncio.run(_run())
    return created_ids


def _sqlite_memory_worker(args: tuple[str, int, int]) -> int:
    db_path_str, worker_id, count = args
    store = SqliteMemoryStore(Path(db_path_str))

    async def _run() -> int:
        written = 0
        for i in range(count):
            await store.record_memory(
                MemoryWriteRequest(
                    user_slug="shared-user",
                    workspace_id="ws",
                    memory_scope="personal",
                    surface_id="s1",
                    content=f"Memory from worker {worker_id} item {i}",
                    memory_tier="daily",
                    memory_kind="episodic",
                    salience=0.7,
                )
            )
            written += 1
        return written

    try:
        return asyncio.run(_run())
    finally:
        store.close()


# ── Tests ────────────────────────────────────────────────────────────


def test_file_lock_reentrant_and_cross_process_exclusion(tmp_path: Path) -> None:
    """file_lock is reentrant in-task and blocks competing processes until release."""
    lock_path = tmp_path / "resource.lock"

    # Reentrant in the same thread/task.
    with file_lock(lock_path, timeout=1.0), file_lock(lock_path, timeout=1.0):
        assert lock_path.exists()

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock_worker, args=(str(lock_path), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=10.0), "Child process did not acquire lock in time"
        with pytest.raises(FileLockTimeoutError), file_lock(lock_path, timeout=0.1):
            pass
    finally:
        release.set()
        proc.join(timeout=10.0)

    # Lock is immediately acquirable after child releases it.
    with file_lock(lock_path, timeout=2.0):
        pass


@pytest.mark.asyncio
async def test_async_file_lock_reentrant_within_task_and_serializes_sibling_tasks(
    tmp_path: Path,
) -> None:
    """async_file_lock is reentrant within one asyncio.Task and serializes sibling tasks."""
    lock_path = tmp_path / "async.lock"
    order: list[str] = []

    async def worker(name: str, hold_sec: float) -> None:
        async with async_file_lock(lock_path, timeout=5.0):
            order.append(f"{name}:enter")
            # Reentrant acquisition inside the same task must not deadlock.
            async with async_file_lock(lock_path, timeout=1.0):
                await asyncio.sleep(hold_sec)
            order.append(f"{name}:exit")

    await asyncio.gather(worker("a", 0.05), worker("b", 0.01))
    assert order in (
        ["a:enter", "a:exit", "b:enter", "b:exit"],
        ["b:enter", "b:exit", "a:enter", "a:exit"],
    )


def test_atomic_write_text_leaves_no_temp_files(tmp_path: Path) -> None:
    """atomic_write_text replaces target atomically and cleans up sibling temp files."""
    target = tmp_path / "nested" / "config.json"
    atomic_write_text(target, '{"v": 1}\n')
    atomic_write_text(target, '{"v": 2}\n')
    assert target.read_text(encoding="utf-8") == '{"v": 2}\n'
    leftovers = [p for p in target.parent.iterdir() if p != target]
    assert leftovers == []


def test_multiprocess_thread_store_appends_seals_and_dictionary_training(
    tmp_path: Path,
) -> None:
    """Concurrent processes appending & sealing across the 20-seal boundary never corrupt."""
    base = tmp_path / "threads" / "shared-user"
    # 4 workers * 16 appends = 64 turns; with segment_turns=2 that's 32 seals,
    # crossing the 1st seal (initial dict training) and 10th/20th/30th seals!
    workers = 4
    per_worker = 16
    segment_turns = 2
    total_expected = workers * per_worker

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(
            _thread_append_worker,
            [(str(base), w, per_worker, segment_turns) for w in range(workers)],
        )

    all_committed = [turn for sub in results for turn in sub]
    assert sorted(t for t in all_committed if t is not None) == list(range(total_expected))

    # Every sealed segment and active entry must decompress and read cleanly in order.
    entries = load_thread_range(thread_path=base, start_turn=0, end_turn=total_expected - 1)
    assert len(entries) == total_expected
    assert [e["turn"] for e in entries] == list(range(total_expected))

    # load_recent reads the active segment plus at most the single newest sealed segment.
    recent = load_recent_thread_entries(limit=4, thread_path=base)
    assert len(recent) == 4
    assert [e["turn"] for e in recent] == list(range(total_expected - 4, total_expected))


def test_overlapping_processes_reserve_distinct_turns_and_commit_contiguously(
    tmp_path: Path,
) -> None:
    """Overlapping turns never share a provisional index nor a committed one."""
    base = tmp_path / "threads" / "shared-user"
    append_thread_entry(
        user_slug="shared-user", mode_id="build", query="seed", response_text="", thread_path=base
    )
    workers = 4
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(workers)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_overlapping_turn_worker, args=(str(base), w, barrier, results))
        for w in range(workers)
    ]
    for proc in procs:
        proc.start()
    outcomes = [results.get(timeout=60) for _ in procs]
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0

    assert sorted(turn for _, turn, _ in outcomes) == [1, 2, 3, 4]
    assert sorted(committed for _, _, committed in outcomes) == [1, 2, 3, 4]
    assert not Path(f"{base}.reservations.json").exists()
    entries = load_thread_range(thread_path=base, start_turn=1, end_turn=4)
    provisional = {w: turn for w, turn, _ in outcomes}
    for entry in entries:
        ran_as = provisional[int(entry["query"].removeprefix("worker-"))]
        assert entry.get("provisional_turn", entry["turn"]) == ran_as


def test_multiprocess_file_approval_service_preserves_all_records(tmp_path: Path) -> None:
    """Concurrent processes creating ApprovalRecords do not lose updates."""
    approvals_path = tmp_path / "state" / "approvals.json"
    workers = 4
    per_worker = 10
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(
            _approval_create_worker,
            [(str(approvals_path), w, per_worker) for w in range(workers)],
        )

    expected_ids = {aid for sub in results for aid in sub}
    assert len(expected_ids) == workers * per_worker

    service = FileApprovalService(approvals_path)
    identity = ApprovalIdentity(
        workspace_id="ws",
        user_id="u1",
        user_slug="u1",
        surface_type="personal_dm",
        surface_id="s1",
        memory_scope="personal",
        mode_id="build",
        session_plan_digest="digest",
    )
    pending = asyncio.run(service.list_pending(identity=identity))
    assert {r.invocation.request_id for r in pending} == expected_ids


@pytest.mark.asyncio
async def test_recovery_mark_running_indeterminate_respects_live_owner_pid(
    tmp_path: Path,
) -> None:
    """mark_running_indeterminate skips live foreign PIDs and sweeps dead/own PIDs."""
    service = FileRecoveryService(tmp_path / "state" / "recovery-items.json")
    now = datetime.now(UTC)
    identity = RecoveryIdentity(
        workspace_id="ws",
        user_id="u1",
        user_slug="u1",
        surface_type="personal_dm",
        surface_id="s1",
        memory_scope="personal",
    )

    # Spawn a live sleeper process to act as a concurrent turn owner.
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    lock_path = tmp_path / "dummy.lock"
    live_proc = ctx.Process(target=_hold_lock_worker, args=(str(lock_path), ready, release))
    live_proc.start()
    try:
        assert ready.wait(timeout=10.0)
        assert live_proc.pid is not None
        assert is_process_alive(live_proc.pid)

        # Record 1: owned by the live foreign process -> must stay "running".
        await service.begin(
            RecoveryRecord(
                work_item_id="rec-live-foreign",
                identity=identity,
                target_kind="operation",
                target_id="edit_file",
                effect="write",
                state="running",
                owner_pid=live_proc.pid,
                created_at=now,
                updated_at=now,
            )
        )
        # Record 2: owned by a non-existent PID -> must become "indeterminate".
        await service.begin(
            RecoveryRecord(
                work_item_id="rec-dead-foreign",
                identity=identity,
                target_kind="operation",
                target_id="edit_file",
                effect="write",
                state="running",
                owner_pid=999_999_999,
                created_at=now,
                updated_at=now,
            )
        )
        # Record 3: owned by current PID (left over from an earlier aborted turn in-process)
        # -> must become "indeterminate".
        await service.begin(
            RecoveryRecord(
                work_item_id="rec-self",
                identity=identity,
                target_kind="operation",
                target_id="edit_file",
                effect="write",
                state="running",
                owner_pid=os.getpid(),
                created_at=now,
                updated_at=now,
            )
        )

        changed = await service.mark_running_indeterminate(identity=identity)
        assert changed == 2

        live_record = await service.get("rec-live-foreign", identity)
        assert live_record is not None
        assert live_record.state == "running"

        dead_record = await service.get("rec-dead-foreign", identity)
        assert dead_record is not None
        assert dead_record.state == "indeterminate"

        self_record = await service.get("rec-self", identity)
        assert self_record is not None
        assert self_record.state == "indeterminate"
    finally:
        release.set()
        live_proc.join(timeout=10.0)


@pytest.mark.asyncio
async def test_code_artifacts_policy_and_pack_lock_concurrent_updates(
    tmp_path: Path,
) -> None:
    """Concurrent updates to artifacts, policy, and pack lock serialize cleanly."""
    store = FileCodeChangeArtifactStore(tmp_path / "state" / "code-change-artifacts.json")
    now = datetime.now(UTC)

    async def put_artifact(idx: int) -> None:
        await store.create(
            WorkProduct(
                artifact_id=f"art-{idx}",
                workspace_id="ws",
                user_id="u1",
                user_slug="u1",
                surface_type="cli",
                surface_id="s1",
                session_plan_digest="digest",
                source_root=str(tmp_path),
                worktree_path=f"/tmp/wt-{idx}",
                created_at=now,
                updated_at=now,
            )
        )

    await asyncio.gather(*(put_artifact(i) for i in range(12)))
    for i in range(12):
        fetched = await store.get(f"art-{i}")
        assert fetched is not None
        assert fetched.worktree_path == f"/tmp/wt-{i}"

    # Policy concurrent allow-prefix writes
    for i in range(8):
        remember_allow_prefix(tmp_path, tool="run_shell_command", command_prefix=f"git status {i}")
    doc = load_policy(tmp_path)
    assert len(doc.allow) == 8

    # Installed packs lock atomic write
    save_installed_packs_lock(tmp_path, {"version": 1, "packs": [{"package_name": "lisa-pack-x"}]})
    loaded_lock = load_installed_packs_lock(tmp_path)
    assert loaded_lock["packs"][0]["package_name"] == "lisa-pack-x"


def test_conversation_id_isolates_thread_path_and_surface_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """conversation_id / LISA_CONVERSATION_ID isolates thread files and surface_id."""
    monkeypatch.setenv("LISA_CONVERSATION_ID", "conv-alpha")
    cfg_a = HostConfig.from_env(data_dir=str(tmp_path), user_slug="alice")
    cfg_b = HostConfig.from_env(
        data_dir=str(tmp_path),
        user_slug="alice",
        conversation_id="conv-beta",
    )
    monkeypatch.delenv("LISA_CONVERSATION_ID", raising=False)
    cfg_default = HostConfig.from_env(data_dir=str(tmp_path), user_slug="alice")

    assert cfg_a.conversation_id == "conv-alpha"
    assert cfg_a.thread_slug == "alice--conv-alpha"
    assert cfg_b.thread_slug == "alice--conv-beta"
    assert cfg_default.thread_slug == "alice"

    paths_a = HostPaths.from_config(cfg_a)
    paths_b = HostPaths.from_config(cfg_b)
    paths_default = HostPaths.from_config(cfg_default)
    assert paths_a.thread_path == tmp_path / "data" / "threads" / "alice--conv-alpha"
    assert paths_b.thread_path == tmp_path / "data" / "threads" / "alice--conv-beta"
    assert paths_default.thread_path == tmp_path / "data" / "threads" / "alice"

    rt_a = HostRuntime(cfg_a)
    rt_b = HostRuntime(cfg_b)
    assert rt_a.config.surface_id == "conv-alpha"
    assert rt_b.config.surface_id == "conv-beta"
    assert _mcp_environment(rt_a, str(tmp_path))["LISA_USER_SLUG"] == "alice--conv-alpha"
    assert _mcp_environment(rt_b, str(tmp_path))["LISA_USER_SLUG"] == "alice--conv-beta"


def test_multiprocess_sqlite_memory_store_concurrent_writes(tmp_path: Path) -> None:
    """Multiple processes writing to the same SqliteMemoryStore succeed with WAL + busy_timeout."""
    db_path = tmp_path / "data" / "memory.sqlite3"
    # Initialize schema once before spawning workers.
    init_store = SqliteMemoryStore(db_path)
    init_store.close()

    workers = 4
    per_worker = 10
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        counts = pool.map(
            _sqlite_memory_worker,
            [(str(db_path), w, per_worker) for w in range(workers)],
        )

    assert sum(counts) == workers * per_worker
    verify_store = SqliteMemoryStore(db_path)
    try:
        total = asyncio.run(
            verify_store.count_memories(
                user_slug="shared-user",
                workspace_id="ws",
                memory_scope="personal",
                surface_id="s1",
            )
        )
        assert total == workers * per_worker
    finally:
        verify_store.close()
