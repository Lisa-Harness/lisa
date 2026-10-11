"""Calls in one conversation share an opaque provider cache key."""

from __future__ import annotations

from types import SimpleNamespace

from kernel.turn_hooks import TurnIdentity
from kernel.turn_runner import _prompt_cache_key


def _hooks(conversation_id: str, task_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        identity=TurnIdentity(
            workspace_id="w", principal_key="p", conversation_id=conversation_id, task_id=task_id
        )
    )


def test_key_is_stable_per_conversation_and_ignores_task() -> None:
    key = _prompt_cache_key(_hooks("c1", "task-a"))  # type: ignore[arg-type]
    assert key == _prompt_cache_key(_hooks("c1", "task-b"))  # type: ignore[arg-type]
    assert key != _prompt_cache_key(_hooks("c2"))  # type: ignore[arg-type]
    assert key.startswith("lisa-") and "c1" not in key and len(key) <= 64
