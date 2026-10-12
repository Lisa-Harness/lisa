"""Tests for host-owned code change artifact persistence."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel.turn_hooks import TaskCompletionRequest
from packstest.context import ToolContextFactory
from packstest.git import git_repo
from runtime.code_artifacts import FileCodeChangeArtifactStore, verify_checked_commit
from runtime.handle_message import _resumable_task_id
from runtime.runtime import HostRuntime
from runtime.testing import build_test_runtime
from runtime.thread import append_thread_entry, load_thread_range, next_turn_index
from runtime.turn_hooks_builder import _derive_view
from sdk.workproducts import WorkProduct, WorkProductCheck, WorkProductPreparation


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
    artifact.preparation = WorkProductPreparation(
        status="ready",
        recipe_id="python-uv",
        recipe_digest="recipe",
        dependency_manifest_digest="manifest",
        environment_path="C:/repo/.venv",
        python_path="C:/repo/.venv/Scripts/python.exe",
        baseline_tree_sha="tree",
    )
    await store.save(artifact)

    loaded = await store.get(artifact.artifact_id)
    assert loaded is not None
    assert loaded.revision == 1
    assert loaded.checks["unit"].status == "passed"
    assert loaded.preparation is not None
    assert loaded.preparation.status == "ready"
    assert loaded.preparation.python_path.endswith("python.exe")
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
            identity=identity,
        )

    assert (await verify("task-one")).verified is True
    assert (await verify("old-task")).reason == (
        "This task has not created or worked on a worktree."
    )
    assert (
        await verify_checked_commit(
            TaskCompletionRequest(
                task_id="other-task",
                origin_tool_id="create_worktree",
                receipt_artifact_ids=(artifact.artifact_id,),
            ),
            store=store,
            identity=identity,
        )
    ).verified is False

    artifact.status = "patched"
    await store.save(artifact)
    assert (await verify("task-one")).reason == (
        f"Worktree {artifact.artifact_id} is patched, not committed."
    )
    artifact.status = "committed"
    await store.save(artifact)

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

    # A later task that advances the same worktree takes over its completion.
    assert artifact.claim_for_task(SimpleNamespace(task_id="task-two")) is True
    await store.save(artifact)
    assert (await verify("task-two")).verified is True
    assert (await verify("task-one")).verified is False


@pytest.mark.asyncio
async def test_checked_commit_accepts_stacked_commits_on_one_worktree(tmp_path: Path) -> None:
    """A reused worktree verifies when commits stack on the recorded base."""
    repo = await git_repo(tmp_path, with_remote=False)
    worktree = tmp_path / "owned-worktree"
    await repo._run("worktree", "add", "-b", "lisa/task", str(worktree))
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(worktree), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    # First commit: one file. Second commit: another file on top.
    (worktree / "app.txt").write_text("changed\n", encoding="utf-8")
    await repo._run("add", "app.txt", cwd=worktree)
    await repo._run("commit", "-m", "first change", cwd=worktree)
    (worktree / "notes.txt").write_text("stacked\n", encoding="utf-8")
    await repo._run("add", "notes.txt", cwd=worktree)
    await repo._run("commit", "-m", "second change", cwd=worktree)

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
        base_sha=base,
        head_sha=commit_sha,
        commit_sha=commit_sha,
        branch="lisa/task",
        status="committed",
        revision=2,
        changed_paths=["app.txt", "notes.txt"],
        required_check_ids=["unit"],
        checks={
            "unit": WorkProductCheck(
                check_id="unit", revision=2, status="passed", checked_tree_sha=tree_sha
            )
        },
    )
    await store.create(artifact)
    identity = SimpleNamespace(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="plan",
    )

    evidence = await verify_checked_commit(
        TaskCompletionRequest(task_id="task-one", origin_tool_id="create_worktree"),
        store=store,
        identity=identity,
    )
    assert evidence.verified is True, evidence.reason

    # A history that no longer contains the recorded base fails verification.
    await repo._run("checkout", "--orphan", "lisa/rebased", cwd=worktree)
    (worktree / "app.txt").write_text("changed\n", encoding="utf-8")
    await repo._run("add", "app.txt", cwd=worktree)
    await repo._run("commit", "-m", "unrelated history", cwd=worktree)
    artifact.commit_sha = git("rev-parse", "HEAD")
    artifact.head_sha = artifact.commit_sha
    artifact.branch = "lisa/rebased"
    await store.save(artifact)
    evidence = await verify_checked_commit(
        TaskCompletionRequest(task_id="task-one", origin_tool_id="create_worktree"),
        store=store,
        identity=identity,
    )
    assert evidence.verified is False
    assert "no longer contains the recorded base commit" in evidence.reason


@pytest.mark.asyncio
async def test_checked_commit_reports_ambiguous_task_worktrees(tmp_path: Path) -> None:
    store = FileCodeChangeArtifactStore(tmp_path / "code-changes.json")
    identity = SimpleNamespace(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )
    for status in ("patched", "committed", "cleaned"):
        await store.create(
            WorkProduct(
                task_id="task",
                session_plan_digest="plan",
                source_root=str(tmp_path),
                status=status,
                **vars(identity),
            )
        )

    evidence = await verify_checked_commit(
        TaskCompletionRequest(task_id="task", origin_tool_id="create_worktree"),
        store=store,
        identity=identity,
    )

    assert evidence.verified is False
    assert evidence.reason.startswith("This task is working on 2 worktrees")


def test_claim_for_task_ignores_missing_or_unchanged_task() -> None:
    artifact = WorkProduct(
        task_id="task-one",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="plan",
        source_root="/repo",
    )

    assert artifact.claim_for_task(SimpleNamespace(task_id="")) is False
    assert artifact.claim_for_task(SimpleNamespace()) is False
    assert artifact.claim_for_task(SimpleNamespace(task_id="task-one")) is False
    assert artifact.task_id == "task-one"


@pytest.mark.parametrize("terminal_response", ["completed", "still working", ""])
def test_task_identity_resumes_only_recorded_unfinished_work(
    tmp_path: Path, terminal_response: str
) -> None:
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

    for detour_task_id in ("task-detour-one", "task-detour-two"):
        append_thread_entry(
            thread_path=thread_path,
            user_slug="user",
            mode_id="default",
            query="continue",
            response_text=terminal_response,
            provenance={"source": "observed", "task_id": detour_task_id},
        )
    assert _resumable_task_id(thread_path, "continue") == "task-one"

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
        response_text=terminal_response,
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


@pytest.mark.parametrize("padding_turns", [35, 270])
@pytest.mark.parametrize(
    "ending", ["unfinished", "invalid", "complete", "new-plan", "new-plan-complete"]
)
def test_restart_recovers_plan_and_receipts_outside_recent_tail(
    tmp_path: Path, padding_turns: int, ending: str
) -> None:
    original = _host_with_declared_bwrap(tmp_path)
    thread_path = original.paths.thread_path
    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        segment_turns=32,
        mode_id="executor",
        query="fix bug",
        response_text=(
            "## Plan\n"
            "1. executor - Edit the owned worktree\n_writes: coding_edit_files_\n"
            "2. executor - Verify and commit\n_writes: coding_commit_\n"
        ),
        provenance={"source": "observed", "task_id": "task-original"},
    )
    append_thread_entry(
        thread_path=thread_path,
        user_slug="user",
        segment_turns=32,
        mode_id="executor",
        query="continue",
        response_text="Applied the edit, awaiting verification.",
        provenance={"source": "observed", "task_id": "task-original"},
        effect_receipts=[
            {"capability_id": "coding_edit_files", "effect": "write", "artifact_id": "owned"}
        ],
    )
    for _ in range(padding_turns):
        append_thread_entry(
            thread_path=thread_path,
            user_slug="user",
            segment_turns=32,
            mode_id="executor",
            query="continue",
            response_text="Waiting for verification.",
            provenance={
                "source": "observed",
                "task_id": "task-original",
                "block_origin": "task_completion",
            },
        )
    if ending != "unfinished":
        append_thread_entry(
            thread_path=thread_path,
            user_slug="user",
            segment_turns=32,
            mode_id="executor",
            query="continue",
            response_text=(
                "## Plan\n1. executor - Publish a different task\n_writes: coding_push_"
                if ending in {"new-plan", "new-plan-complete"}
                else "## Plan\nNo parseable steps"
            ),
            provenance={
                "source": "observed",
                "task_id": "task-original",
                **({"block_origin": "task_completion"} if ending == "invalid" else {}),
            },
            effect_receipts=(
                [{"capability_id": "coding_commit", "effect": "write", "artifact_id": "owned"}]
                if ending == "complete"
                else []
            ),
        )
    if ending == "new-plan-complete":
        append_thread_entry(
            thread_path=thread_path,
            user_slug="user",
            segment_turns=32,
            mode_id="executor",
            query="continue",
            response_text="The newer plan is finished.",
            provenance={"source": "observed", "task_id": "task-original"},
            effect_receipts=[
                {"capability_id": "coding_push", "effect": "external_write", "artifact_id": "owned"}
            ],
        )
    restarted = _host_with_declared_bwrap(tmp_path)
    view = _derive_view(restarted)
    if ending in {"complete", "new-plan-complete"}:
        assert view is None
        return
    assert view is not None
    if ending == "new-plan":
        assert view.completed_steps == ()
        assert view.declared_steps[0].write_capability == "coding_push"
        return
    assert view.completed_steps == ("executor_0",)
    assert view.current_step_id == "executor_1"
    assert view.declared_steps[0].completing_turn == 1
    assert _resumable_task_id(thread_path, "continue") == "task-original"


def test_derive_view_does_not_rescan_unchanged_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: _derive_view re-parsed the whole thread on every
    # workflow_position() call when no plan existed (~3s per call on a 110MB
    # thread), and the kernel calls it many times per turn.
    import runtime.thread as thread_module

    runtime = build_test_runtime(
        tmp_path, user_slug="user", surface_type="personal_dm", surface_id="surface"
    )
    thread_path = runtime.paths.thread_path

    def append(response_text: str, receipts: list[dict[str, str]] | None = None) -> None:
        append_thread_entry(
            thread_path=thread_path,
            user_slug="user",
            segment_turns=64,
            mode_id="executor",
            query="continue",
            response_text=response_text,
            provenance={"source": "observed"},
            effect_receipts=receipts,
        )

    for _ in range(300):
        append("No plan here.")

    loaded: list[int] = []
    original_load = thread_module.load_thread_range

    def counting_load(**kwargs: object) -> list[dict[str, object]]:
        entries = original_load(**kwargs)  # type: ignore[arg-type]
        loaded.append(len(entries))
        return entries

    monkeypatch.setattr(thread_module, "load_thread_range", counting_load)

    assert _derive_view(runtime) is None
    assert sum(loaded) == 300

    loaded.clear()
    assert _derive_view(runtime) is None
    assert _derive_view(runtime) is None
    assert sum(loaded) == 0

    append("Still no plan.")
    loaded.clear()
    assert _derive_view(runtime) is None
    assert sum(loaded) == 1

    append("## Plan\n1. executor - Edit the owned worktree\n_writes: coding_edit_files_\n")
    view = _derive_view(runtime)
    assert view is not None
    assert view.declared_steps[0].write_capability == "coding_edit_files"
    assert view.completed_steps == ()

    append("Edited.", [{"capability_id": "coding_edit_files", "effect": "write"}])
    loaded.clear()
    assert _derive_view(runtime) is None
    assert sum(loaded) < 300


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
    return build_tool_catalog(pack_path)


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
        await turn_module._execute_mode_turn(runtime, query, "default", query, NullTransport())

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


async def _restart_edited_candidate(
    data_dir: Path,
    repo: object,
    catalog: dict[str, dict[str, object]],
    artifact_id: str,
    *,
    prior_factory: ToolContextFactory,
) -> ToolContextFactory:
    from packstest.runner import run_tool

    patch = "--- a/app.txt\n+++ b/app.txt\n@@ -1 +1 @@\n-hello\n+fixed\n"
    for tool_id, parameters in (
        ("coding_apply_patch", {"artifact_id": artifact_id, "patch": patch}),
        ("coding_run_check", {"artifact_id": artifact_id, "check_id": "smoke"}),
    ):
        step = await run_tool(catalog, tool_id, prior_factory, parameters)
        assert step.result.error == "", step.result.error
    edited = await prior_factory.artifact_store.get(artifact_id)
    assert edited is not None and edited.checks["smoke"].status == "passed"
    (Path(edited.worktree_path) / "app.txt").write_text("fixed after restart\n", encoding="utf-8")
    restarted = _host_with_declared_bwrap(data_dir)
    assert _resumable_task_id(restarted.paths.thread_path, "continue") == "task-original"
    factory = _coding_factory_for_host(restarted, repo)
    artifact = await factory.artifact_store.get(artifact_id)
    assert artifact is not None and artifact.revision == 1
    worktree = Path(artifact.worktree_path)
    assert (worktree / "app.txt").read_text(encoding="utf-8") == "fixed after restart\n"
    read = await run_tool(catalog, "coding_read_file", factory, {"path": "app.txt"})
    assert read.result.error == ""
    assert read.result.output_json["scope"] == "worktree"
    assert read.result.output_json["artifact_id"] == artifact_id
    assert read.result.output_json["content_sha256"]
    blocked = await run_tool(
        catalog, "coding_commit", factory, {"artifact_id": artifact_id, "message": "unchecked"}
    )
    assert "Every required check must pass" in blocked.result.error
    factory.identity["session_plan_digest"] = "changed-session-plan"
    survived = await run_tool(
        catalog, "coding_read_file", factory, {"artifact_id": artifact_id, "path": "app.txt"}
    )
    assert survived.result.error == "", survived.result.error
    assert len(await factory.artifact_store.list_owned(factory.context())) == 1
    factory.identity["session_plan_digest"] = restarted.session_plan.plan_digest
    assert len(await factory.artifact_store.list_owned(factory.context())) == 1
    return factory


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
    source_head = await repo.head()
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
        receipt = await restarted.services.approval_service.consume(
            invocation.request_id, invocation
        )
        assert receipt is not None
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

        resumed_factory = await _restart_edited_candidate(
            host_dir, repo, catalog, committed_artifact_id, prior_factory=factory
        )
        for tool_id, parameters in (
            ("coding_run_check", {"artifact_id": committed_artifact_id, "check_id": "smoke"}),
            ("coding_commit", {"artifact_id": committed_artifact_id, "message": "Fix bug"}),
        ):
            step = await run_tool(catalog, tool_id, resumed_factory, parameters)
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
    repeated = await handle_approval_command(
        _host_with_declared_bwrap(host_dir),
        f"/approve {invocation.request_id}",
        approve=True,
        transport=NullTransport(),
        route_model_turn=continue_task,
    )
    assert isinstance(repeated, str) and "already executed" in repeated
    assert len(await factory.artifact_store.list_owned(factory.context())) == 1
    assert await repo.head() == source_head
    assert (repo.root / "app.txt").read_text(encoding="utf-8") == "hello\n"
