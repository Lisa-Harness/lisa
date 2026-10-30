"""Approval invocation turn stamping.

The execution registries stamp each approval invocation with the dispatch's
``turn_index``/``thread_store_slug`` so the durable approval evidence joins
to the thread — and the stamp deliberately does NOT change
``action_digest`` (re-approving the same action in a later turn must still
dedup against the pending request). The runtime-side ledger tests live in
``runtime/tests/test_approval_ledger.py`` (kernel tests must not import the
runtime).
"""

from __future__ import annotations

import pytest
from kernel.approval import (
    ApprovalIdentity,
    ApprovalInvocation,
    build_approval_invocation,
)
from kernel.models import ToolExecutionContext


def _identity() -> ApprovalIdentity:
    return ApprovalIdentity(user_slug="adam", mode_id="coding")


def _invocation(turn_index: int | None = None, **overrides: object) -> ApprovalInvocation:
    inv = build_approval_invocation(
        target_kind="tool",
        target_id="apply_patch",
        pack_id="lisa-pack-coding",
        effect="write",
        parameters={"path": "src/x.py"},
        identity=_identity(),
        pack_config={},
        **overrides,  # type: ignore[arg-type]
    )
    return inv.model_copy(
        update={"turn_index": turn_index, "thread_store_slug": "adam"}
    )


# ── Turn stamping ──────────────────────────────────────────────────────


def test_turn_stamp_does_not_change_action_digest() -> None:
    """Turn position is evidence, not action identity."""
    unstamped = build_approval_invocation(
        target_kind="tool",
        target_id="apply_patch",
        pack_id="p",
        effect="write",
        parameters={"path": "a"},
        identity=_identity(),
        pack_config={},
    )
    stamped = unstamped.model_copy(
        update={"turn_index": 7, "thread_store_slug": "adam--conv"}
    )
    assert stamped.turn_index == 7
    assert stamped.action_digest == unstamped.action_digest


def test_turn_index_is_none_by_default() -> None:
    """Older callers (and off-turn requests) keep ``None``, never ``0``."""
    inv = build_approval_invocation(
        target_kind="tool",
        target_id="t",
        pack_id="p",
        effect="read",
        parameters={},
        identity=_identity(),
        pack_config={},
    )
    assert inv.turn_index is None
    assert inv.thread_store_slug == ""


@pytest.mark.asyncio
async def test_registry_stamps_invocation_from_context() -> None:
    """A tool dispatch stamps its invocation with the dispatch's turn.

    Patched at the dispatch site's own module so the test follows the
    kernel test-seam rule (patch the imported binding where behavior is
    defined).
    """
    from kernel.tools import registry as tool_registry

    captured: dict[str, object] = {}

    class _RecordingService:
        async def request(self, invocation: ApprovalInvocation) -> ApprovalInvocation:
            captured["turn_index"] = invocation.turn_index
            captured["stem"] = invocation.thread_store_slug
            return invocation

    class _Definition:
        tool_id = "apply_patch"
        owner_id = "p"
        effect = "write"
        approval_required = True
        approval_preview = "apply a patch"
        stream_output = False

    ctx = ToolExecutionContext(
        tool_id="apply_patch",
        parameters={"path": "a"},
        approval_service=_RecordingService(),
        approval_request_id="",
        turn_index=5,
        thread_store_slug="adam--conv",
        user_slug="adam",
    )

    with pytest.raises(Exception) as raised:
        await tool_registry._require_approval(_Definition(), ctx)  # type: ignore[arg-type]
    assert "Approval required" in str(raised.value)
    assert captured["turn_index"] == 5
    assert captured["stem"] == "adam--conv"
