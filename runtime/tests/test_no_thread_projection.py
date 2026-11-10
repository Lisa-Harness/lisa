"""PROMPT_STARTUP Slice D regression tests: ``--no-thread`` result projection.

The spec's Slice D contract: projecting oversized tool results, replaying
context, and recording a request snapshot must all succeed without a thread
path, create no thread artifacts, keep ``r<sha256>`` refs dereferenceable
through the same result-object contract, and change nothing when a durable
thread path exists.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from runtime.config import HostConfig
from runtime.runtime_paths import HostPaths
from runtime.thread import build_open_task_messages, project_thread_messages
from runtime.turn_continuation import persist_request
from sdk.thread import result_object_path
from sdk.thread._results import load_result_object


def _paths(tmp_path: Path, *, no_thread: bool) -> HostPaths:
    return HostPaths.from_config(HostConfig.from_env(data_dir=str(tmp_path), no_thread=no_thread))


def _oversized(name: str, size: int = 200_000) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": name, "name": name, "content": "x" * size}


def _expected_ref(content: str, *, task_id: str = "task-1", tool_name: str = "call-0") -> str:
    """The projector persists ``{"content", "task_id", "tool_name"}`` records."""
    record = {"content": content, "task_id": task_id, "tool_name": tool_name}
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "r" + hashlib.sha256(payload).hexdigest()


def _thread_dirs(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_dir() and p.name == "threads"]


def test_projection_persists_to_the_ephemeral_store_without_thread_artifacts(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path, no_thread=True)
    assert paths.thread_path is None
    assert paths.ephemeral_results_path is not None

    result = _oversized("call-0")
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {"name": "call-0", "arguments": "{}"},
                }
            ],
        },
        result,
    ]
    project_thread_messages(
        messages,
        "task-1",
        thread_path=paths.thread_path,
        ephemeral_results_path=paths.ephemeral_results_path,
    )
    assert result_object_path(
        paths.ephemeral_results_path, _expected_ref(result["content"])
    ).is_file()
    assert _thread_dirs(tmp_path) == []


def test_ephemeral_refs_dereference_with_identical_content(tmp_path: Path) -> None:
    paths = _paths(tmp_path, no_thread=True)
    result = _oversized("call-0")
    project_thread_messages(
        [result],
        "task-1",
        thread_path=None,
        ephemeral_results_path=paths.ephemeral_results_path,
    )
    loaded = load_result_object(paths.ephemeral_results_path, _expected_ref(result["content"]))
    assert loaded["content"] == "x" * 200_000


def test_projection_without_any_base_still_raises() -> None:
    with pytest.raises(ValueError, match="thread storage path"):
        project_thread_messages([_oversized("call-0")], "task-1", thread_path=None)


def test_threaded_projection_is_unchanged(tmp_path: Path) -> None:
    thread_path = tmp_path / "threads" / "user"
    thread_path.parent.mkdir(parents=True, exist_ok=True)
    result = _oversized("call-0")
    project_thread_messages(
        [result],
        "task-1",
        thread_path=thread_path,
        ephemeral_results_path=None,
    )
    assert result_object_path(thread_path, _expected_ref(result["content"])).is_file()


def test_empty_replay_projection_succeeds_under_no_thread(tmp_path: Path) -> None:
    paths = _paths(tmp_path, no_thread=True)
    assert (
        build_open_task_messages(
            thread_path=paths.thread_path,
            ephemeral_results_path=paths.ephemeral_results_path,
            budget_bytes=4096,
        )
        == []
    )


def test_persist_request_archives_to_the_ephemeral_store(tmp_path: Path) -> None:
    paths = _paths(tmp_path, no_thread=True)
    request = {"model": "stub", "messages": [{"role": "user", "content": "hi"}]}
    ref = persist_request(paths.thread_path, request, paths.ephemeral_results_path)
    assert ref.startswith("r")
    assert result_object_path(paths.ephemeral_results_path, ref).is_file()
    assert _thread_dirs(tmp_path) == []


def test_persist_request_still_requires_a_base_without_ephemeral() -> None:
    with pytest.raises(ValueError, match="thread storage path"):
        persist_request(None, {"model": "stub"})
