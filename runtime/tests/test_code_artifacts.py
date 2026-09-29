"""Tests for host-owned code change artifact persistence."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel.turn_hooks import TaskCompletionRequest
from packstest.git import git_repo
from runtime.code_artifacts import FileCodeChangeArtifactStore, verify_checked_commit
from runtime.handle_message import _resumable_task_id
from runtime.runtime import HostRuntime
from runtime.testing import build_test_runtime
from runtime.thread import append_thread_entry, load_thread_range, next_turn_index
from sdk.workproducts import WorkProduct, WorkProductCheck


@pytest.mark.asyncio
async def test_file_store_round_trips_updated_artifact(tmp_path: Path) -> None:
    """The host store persists an artifact independently of either pack."""

    store = FileCodeChangeArtifactStore(tmp_path / "code-changes.json")
    artifact = WorkProduct(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="plan",
        source_root="C:/repo",
        branch="lisa/change",
        required_check_ids=["unit"],
    )

    await store.create(artifact)
    artifact.revision = 1
    artifact.changed_paths = ["src/main.py"]
    artifact.checks["unit"] = WorkProductCheck(
        check_id="unit",
        revision=1,
        status="passed",
        argv=["uv", "run", "pytest"],
    )
    await store.save(artifact)

    loaded = await store.get(artifact.artifact_id)
    assert loaded is not None
    assert loaded.revision == 1
    assert loaded.checks["unit"].status == "passed"
    assert await store.delete(artifact.artifact_id) is True
    assert await store.get(artifact.artifact_id) is None


@pytest.mark.asyncio
async def test_checked_commit_requires_this_task_and_clean_checked_tree(tmp_path: Path) -> None:
    repo = await git_repo(tmp_path, with_remote=False)
    worktree = tmp_path / "owned-worktree"
    await repo._run("worktree", "add", "-b", "lisa/task", str(worktree))
    (worktree / "app.txt").write_text("changed\n", encoding="utf-8")
    await repo._run("add", "app.txt", cwd=worktree)
    await repo._run("commit", "-m", "implement", cwd=worktree)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(worktree), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit_sha = git("rev-parse", "HEAD")
    tree_sha = git("rev-parse", "HEAD^{tree}")
    store = FileCodeChangeArtifactStore(tmp_path / "code-changes.json")
    artifact = WorkProduct(
        task_id="task-one",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="plan",
        source_root=str(repo.root),
        worktree_path=str(worktree),
        base_sha=git("rev-parse", "HEAD^"),
        head_sha=commit_sha,
        commit_sha=commit_sha,
        branch="lisa/task",
        status="committed",
        revision=1,
        changed_paths=["app.txt"],
        required_check_ids=["unit"],
        checks={
            "unit": WorkProductCheck(
                check_id="unit", revision=1, status="passed", checked_tree_sha=tree_sha
            )
        },
    )
    await store.create(artifact)
    thread_path = tmp_path / "thread.jsonl"
    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        segment_turns=30,
        mode_id="executor",
        query="implement",
        response_text="worktree created",
        provenance={"source": "observed", "task_id": "task-one"},
        effect_receipts=[
            {
                "capability_id": "create_worktree",
                "effect": "write",
                "artifact_id": artifact.artifact_id,
            }
        ],
    )
    identity = SimpleNamespace(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="plan",
    )

    async def verify(task_id: str):
        return await verify_checked_commit(
            TaskCompletionRequest(task_id=task_id, origin_tool_id="create_worktree"),
            store=store,
            thread_path=thread_path,
            identity=identity,
        )

    assert (await verify("task-one")).verified is True
    assert (await verify("old-task")).verified is False
    assert (
        await verify_checked_commit(
            TaskCompletionRequest(
                task_id="other-task",
                origin_tool_id="create_worktree",
                receipt_artifact_ids=(artifact.artifact_id,),
            ),
            store=store,
            thread_path=thread_path,
            identity=identity,
        )
    ).verified is False

    (worktree / "unchecked.txt").write_text("dirty\n", encoding="utf-8")
    assert (await verify("task-one")).verified is False
    (worktree / "unchecked.txt").unlink()

    artifact.checks["unit"].checked_tree_sha = ""
    await store.save(artifact)
    assert (await verify("task-one")).verified is False

    artifact.checks["unit"].checked_tree_sha = tree_sha
    artifact.changed_paths = ["unrelated.txt"]
    await store.save(artifact)
    assert (await verify("task-one")).verified is False

    artifact.changed_paths = ["app.txt"]
    artifact.checks["unit"].check_id = "unrelated"
    await store.save(artifact)
    assert (await verify("task-one")).verified is False

    artifact.required_check_ids = []
    artifact.checks = {}
    await store.save(artifact)
    assert (await verify("task-one")).verified is True


def test_task_identity_resumes_only_recorded_unfinished_work(tmp_path: Path) -> None:
    thread_path = tmp_path / "thread.jsonl"
    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        mode_id="executor",
        query="fix bug",
        response_text="paused",
        provenance={
            "source": "observed",
            "task_id": "task-one",
            "pending_approval_request_id": "request-one",
        },
    )
    assert _resumable_task_id(thread_path, "continue") == "task-one"
    assert _resumable_task_id(thread_path, "fix a different bug") == ""

    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        mode_id="executor",
        query="continue",
        response_text="needs a checked commit",
        provenance={"source": "observed", "task_id": "task-one", "block_origin": "task_completion"},
    )
    assert _resumable_task_id(thread_path, "continue") == "task-one"

    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        mode_id="executor",
        query="continue",
        response_text="completed",
        provenance={"source": "observed", "task_id": "task-one"},
    )
    assert _resumable_task_id(thread_path, "continue") == ""

    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        mode_id="executor",
        query="/approve request-two",
        response_text="approved action executed",
        provenance={"source": "host-approved", "task_id": "task-two"},
    )
    assert _resumable_task_id(thread_path, "continue") == "task-two"
    assert _resumable_task_id(thread_path, "implement another feature") == ""

    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        mode_id="executor",
        query="clarify request",
        response_text="Which path?",
        provenance={"source": "observed", "task_id": "task-three"},
        yielded=True,
        yield_question="Which path?",
    )
    assert _resumable_task_id(thread_path, "continue") == "task-three"
    assert _resumable_task_id(thread_path, "implement another feature") == ""


def _host_with_declared_bwrap(data_dir: Path) -> HostRuntime:
    runtime = build_test_runtime(
        data_dir, user_slug="user", surface_type="personal_dm", surface_id="surface"
    )
    manifest = runtime.session_plan.manifest
    packs = manifest.resources.packs
    runtime.session_plan = runtime.session_plan.model_copy(
        update={
            "manifest": manifest.model_copy(
                update={
                    "resources": manifest.resources.model_copy(
                        update={
                            "packs": packs.model_copy(
                                update={"include": (*packs.include, "lisa-pack-bwrap")}
                            )
                        }
                    )
                }
            )
        }
    )
    return runtime


def _coding_factory_for_host(runtime: HostRuntime, repo: object):
    from kernel.capability_binding import ExecutionFloorView
    from kernel.models import ToolResult
    from packstest.context import ToolContextFactory

    factory = ToolContextFactory(
        root=repo.root,
        pack_config={
            "workspace_root": str(repo.root),
            "checks": [{"id": "smoke", "argv": ["true"], "timeout_seconds": 30}],
            "required_check_ids": ["smoke"],
        },
        identity={
            "workspace_id": runtime.session_plan.manifest.name,
            "user_id": runtime.config.user_slug,
            "user_slug": runtime.config.user_slug,
            "surface_type": runtime.config.surface_type,
            "surface_id": runtime.config.surface_id,
            "memory_scope": runtime.config.memory_scope,
            "session_plan_digest": runtime.session_plan.plan_digest,
            "task_id": "task-original",
        },
        artifact_store=runtime.services.code_change_artifact_store,
    )
    factory.execution_floor_for = lambda _tool_id: ExecutionFloorView(
        execution="sandbox",
        persistence="direct-write",
        paths={str(repo.root.parent): "rw"},
    )

    async def contained(tool_id: str, parameters: object) -> ToolResult:
        assert tool_id == "bwrap_run" and parameters["argv"] == ["true"]
        worktree = str(parameters["cwd"])
        return ToolResult(
            output_json={
                "sandboxed": True,
                "persistence": "direct-write",
                "cwd": worktree,
                "writable_paths": [worktree],
                "return_code": 0,
            }
        )

    factory.delegated_tool_dispatch = contained
    return factory


def _coding_catalog(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> dict[str, dict[str, object]]:
    from packstest.catalog import build_tool_catalog

    pack_path = Path(__file__).resolve().parents[3] / "lisa-packs-wip" / "coding"
    if not pack_path.is_dir():
        pytest.skip("Coding pack checkout is not available")
    loaded_packs = set(sys.modules)

    def restore_pack_imports() -> None:
        for name in set(sys.modules) - loaded_packs:
            if name == "lisa_packs" or name.startswith("lisa_packs.coding"):
                sys.modules.pop(name, None)

    request.addfinalizer(restore_pack_imports)
    monkeypatch.syspath_prepend(str(pack_path))
    return build_tool_catalog(pack_path, ensure_importable=False)


@pytest.mark.asyncio
async def test_host_turn_allocates_and_persists_task_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kernel.models import WorkflowModeConfig
    from kernel.turn_runner import TurnResult
    from runtime.callbacks import NullTransport

    runtime = _host_with_declared_bwrap(tmp_path)
    turn_module = importlib.import_module("runtime.handle_message")
    mode = WorkflowModeConfig(mode_id="default", start_stage_id="none", mode_kind="chat")
    monkeypatch.setattr(turn_module, "get_workflow_catalog", lambda: {"default": mode})
    seen_task_ids: list[str] = []

    async def fake_run_turn(*, hooks: object, **_kwargs: object) -> TurnResult:
        seen_task_ids.append(hooks.identity.task_id)
        return TurnResult(
            response_text="continue the task",
            blocked=True,
            block_origin="task_completion",
        )

    monkeypatch.setattr(turn_module, "run_turn", fake_run_turn)
    for query in ("fix the code", "continue", "implement another change"):
        await turn_module._execute_mode_turn(
            runtime, query, "default", query, NullTransport()
        )

    entries = load_thread_range(
        thread_path=runtime.paths.thread_path,
        start_turn=0,
        end_turn=next_turn_index(thread_path=runtime.paths.thread_path),
    )
    assert len(entries) == 3
    assert len(seen_task_ids[0]) == 32
    assert [entry["provenance"]["task_id"] for entry in entries] == seen_task_ids
    assert seen_task_ids[0] == seen_task_ids[1]
    assert seen_task_ids[2] != seen_task_ids[0]


@pytest.mark.asyncio
async def test_approval_restart_checks_and_commits_only_its_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    from kernel.approval import build_approval_invocation
    from kernel.models import ToolResult
    from packstest.runner import run_tool
    from runtime.approval.orchestration import approval_identity, handle_approval_command
    from runtime.callbacks import NullTransport
    from runtime.turn_hooks_builder import build_turn_hooks
    from runtime.types import HostCommandResult

    repo = await git_repo(tmp_path, with_remote=False)
    catalog = _coding_catalog(monkeypatch, request)
    host_dir = tmp_path / "host"

    original = _host_with_declared_bwrap(host_dir)
    original.session.current_mode_id = "executor"
    invocation = build_approval_invocation(
        target_kind="tool",
        target_id="coding_create_worktree",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"summary": "fix bug"},
        identity=approval_identity(original),
        pack_config={"workspace_root": str(repo.root)},
    )
    await original.services.approval_service.request(invocation)
    append_thread_entry(
        thread_path=original.paths.thread_path,
        user_slug=original.config.user_slug,
        mode_id="executor",
        query="fix bug",
        response_text="needs approval",
        provenance={
            "source": "observed",
            "task_id": "task-original",
            "pending_approval_request_id": invocation.request_id,
        },
    )

    restarted = _host_with_declared_bwrap(host_dir)
    factory = _coding_factory_for_host(restarted, repo)
    approval_module = importlib.import_module("runtime.approval.orchestration")

    async def approved_action(*, invocation: object, hooks: object) -> ToolResult:
        assert hooks.identity.task_id == "task-original"
        created = await run_tool(catalog, "coding_create_worktree", factory, {"summary": "fix bug"})
        assert created.result.error == ""
        return created.result

    monkeypatch.setattr(approval_module, "execute_approved_invocation", approved_action)
    committed_artifact_id = ""

    async def continue_task(runtime: object, _query: str, _transport: object, *, task_id: str):
        nonlocal committed_artifact_id
        assert task_id == "task-original"
        approved = load_thread_range(
            thread_path=restarted.paths.thread_path,
            start_turn=0,
            end_turn=next_turn_index(thread_path=restarted.paths.thread_path),
        )[-1]
        committed_artifact_id = approved["effect_receipts"][0]["artifact_id"]
        hooks = build_turn_hooks(restarted, {"task_id": task_id})
        request = TaskCompletionRequest(task_id=task_id, origin_tool_id="coding_create_worktree")
        verify = hooks.services.verify_task_completion
        assert verify is not None and not (await verify(request)).verified

        patch = "--- a/app.txt\n+++ b/app.txt\n@@ -1 +1 @@\n-hello\n+fixed\n"
        for tool_id, parameters in (
            ("coding_apply_patch", {"artifact_id": committed_artifact_id, "patch": patch}),
            ("coding_run_check", {"artifact_id": committed_artifact_id, "check_id": "smoke"}),
            ("coding_commit", {"artifact_id": committed_artifact_id, "message": "Fix bug"}),
        ):
            step = await run_tool(catalog, tool_id, factory, parameters)
            assert step.result.error == "", step.result.error

        assert (await verify(request)).verified is True
        assert (
            await verify(
                TaskCompletionRequest(
                    task_id="other-task",
                    origin_tool_id="coding_create_worktree",
                    receipt_artifact_ids=(committed_artifact_id,),
                )
            )
        ).verified is False
        artifact = await restarted.services.code_change_artifact_store.get(committed_artifact_id)
        assert artifact is not None and artifact.pushed_sha == "" and artifact.pr_url == ""
        return HostCommandResult(text="work completed")

    result = await handle_approval_command(
        restarted,
        f"/approve {invocation.request_id}",
        approve=True,
        transport=NullTransport(),
        route_model_turn=continue_task,
    )
    assert result.text == "work completed"
    assert committed_artifact_id
