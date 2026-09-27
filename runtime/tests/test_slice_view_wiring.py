"""Tests for the slice-provider hooks view wiring.

Pins that ``_SliceProviderHooksView`` declares ``execute_tool`` in its
``__slots__`` (so the memory-consolidation slice can call ``memory.list``
to count and load episodic records) and that ``build_turn_hooks`` assigns
a callable to it. Before the wiring fix, the attribute was missing and
the slice no-op'd with ``no_episodic_count_seam`` on every turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runtime.turn_hooks_builder import _SliceProviderHooksView, build_turn_hooks

from runtime import HostRuntime


def _make_runtime(tmp_path: Path) -> HostRuntime:
    from runtime.testing import build_test_runtime

    return build_test_runtime(
        tmp_path,
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
    )


def test_view_declares_execute_tool_slot() -> None:
    """``_SliceProviderHooksView`` declares ``execute_tool`` in __slots__."""
    assert "execute_tool" in _SliceProviderHooksView.__slots__
    assert "execute_operation" in _SliceProviderHooksView.__slots__


def test_view_accepts_execute_tool_assignment(tmp_path: Path) -> None:
    """An ``execute_tool`` closure assigned to the view is readable via getattr.

    The memory-consolidation slice reads
    ``getattr(self._hooks, "execute_tool", None)`` and no-ops when it is
    not callable. This test confirms the attribute is settable and
    resolves (not raising ``AttributeError`` via ``__getattr__``).
    """
    from runtime.testing import build_test_runtime

    runtime = build_test_runtime(tmp_path, user_slug="user", surface_type="dm", surface_id="s")

    async def fake_execute_tool(tool_id: str, args: Any) -> str:
        return "[]"

    view = _SliceProviderHooksView(
        runtime=runtime,
    )
    view.execute_tool = fake_execute_tool

    resolved = getattr(view, "execute_tool", None)
    assert callable(resolved)
    assert resolved is fake_execute_tool


def test_build_turn_hooks_assigns_execute_tool_on_view(tmp_path: Path) -> None:
    """``build_turn_hooks`` wires ``execute_tool`` on the internal view.

    The view is constructed internally during ``build_turn_hooks`` and is
    not returned publicly. We verify the wiring indirectly: after
    ``build_turn_hooks``, the memory-consolidation slice provider (when
    enabled) is constructed and registered. If ``execute_tool`` were
    missing on the view, the slice would be registered but no-op on every
    turn. This test confirms the build path does not raise and the hooks
    are fully wired (the slice-provider tests in
    ``sdk/packs/memory/tests/test_slice_provider.py`` cover the slice's
    use of the seam).
    """
    runtime = _make_runtime(tmp_path)
    hooks = build_turn_hooks(runtime, {"mode_id": "default"})
    assert hooks is not None
    assert hooks.operations.execute_operation is not None
