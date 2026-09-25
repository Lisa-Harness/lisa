"""Unified parent-turn completion contract resolver.

This module owns the post-loop decision that validates whether a parent turn
in a side-effecting mode (a mode declaring a write contract via
``action_tool_ids``) legitimately completed its work or reported hallucinated
success.

It mirrors the fork-child classifier (:func:`kernel.turn_phases._child_turn.classify_child_turn`)
by evaluating ``EffectReceipt``s rather than tool names or prose claims.

Design constraints (mirroring the codebase conventions):

- Pure: takes the turn's final state, receipts, called tools, prose content,
  and mode configuration; returns a tagged :class:`CompletionContractDecision`.
  No I/O, no mutation. The turn loop applies the decision.
- Tag-blind and pack-blind: the resolver evaluates opaque capability IDs and
  their declared effects (``write``, ``external_write``, ``read``, ``execute``).
  It never checks pack names or tool prefixes.
- Deterministic: all rules are strictly ordered and pure functions of the
  provided arguments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kernel.tools.mcp_client import bare_tool_name

if TYPE_CHECKING:
    from kernel.models import EffectReceipt

_log = logging.getLogger(__name__)

__all__ = [
    "AcceptTurn",
    "BlockNoVerifiedWrite",
    "ChatReadFootnote",
    "ChatSoftAccept",
    "CompletionContractDecision",
    "PreexistingFailureBlock",
    "SurfaceProseWithFootnote",
    "resolve_completion_contract",
]


# ── Decision tagged union ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AcceptTurn:
    """The turn satisfies the completion contract or is exempt.

    Content is accepted as-is; ``turn_blocked`` remains False.
    """

    reason: str = ""


@dataclass(frozen=True, slots=True)
class PreexistingFailureBlock:
    """The turn was blocked by an upstream failure (entry op, plan, or fork child).

    ``turn_blocked`` is set to True; ``block_origin`` remains empty because the
    block originates from a lifecycle failure rather than a guard.
    """

    reason: str = ""


@dataclass(frozen=True, slots=True)
class SurfaceProseWithFootnote:
    """The turn aborted a synthesized action dispatch on a zero-dispatch turn.

    The model's prose represents an answer to a conversational query that was
    misrouted into a write-contract mode. The prose is preserved and a
    no-write warning footnote is appended; ``turn_blocked`` remains False.
    """

    action_tool_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class ChatSoftAccept:
    """Chat-mode softening: the chat mode did read-only work and produced dense content.

    For chat modes the conversational response is the artifact. Accepted as
    successful without a warning footnote; ``turn_blocked`` remains False.
    """

    content_len: int


@dataclass(frozen=True, slots=True)
class ChatReadFootnote:
    """Chat-mode read-only work with substantive but not dense content.

    The chat mode did real work (read-only tool calls) and produced content
    below the dense threshold. A warning footnote is appended to the prose;
    ``turn_blocked`` remains False.
    """

    action_tool_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class BlockNoVerifiedWrite:
    """The turn produced no verified write for its action tools (hallucinated success).

    The model reported success without landing a required write. The prose is
    replaced with a deterministic failure message; ``turn_blocked`` is set to
    True with ``block_origin="no_tools"``.
    """

    mode_id: str
    action_tool_ids: frozenset[str]

    @property
    def block_message(self) -> str:
        """The deterministic unverified-write failure message."""
        tools_str = ", ".join(sorted(self.action_tool_ids))
        return (
            f"The `{self.mode_id}` turn produced no verified write: none of the "
            f"mode's action tools ({tools_str}) "
            "completed successfully. The reported result is unverified — "
            "nothing was written to disk or memory. Re-issue the request, "
            "or narrow it so the mode can complete its work."
        )


CompletionContractDecision = (
    AcceptTurn
    | PreexistingFailureBlock
    | SurfaceProseWithFootnote
    | ChatSoftAccept
    | ChatReadFootnote
    | BlockNoVerifiedWrite
)


# ── Pure contract resolver ─────────────────────────────────────────────────


def _check_exemption(
    *,
    preexisting_block: bool,
    sub_turn_kind: str | None,
    yielded: bool,
    action_set: frozenset[str],
) -> CompletionContractDecision | None:
    """Return an Accept/Block decision if the turn is exempt or blocked upstream."""
    if preexisting_block:
        return PreexistingFailureBlock(reason="pre-existing entry/plan/fork failure")
    if sub_turn_kind:
        return AcceptTurn(reason="sub-turn validated by child classifier")
    if yielded:
        return AcceptTurn(reason="turn yielded to operator")
    if not action_set:
        return AcceptTurn(reason="no write contract declared")
    return None


def _check_receipt_satisfaction(
    *,
    action_set: frozenset[str],
    effect_receipts: Sequence[EffectReceipt],
    called_tool_names: frozenset[str] | set[str],
) -> tuple[bool, bool, bool]:
    """Return (write_landed, read_work, write_attempted)."""
    write_receipt_ids = {
        bare_tool_name(r.capability_id)
        for r in effect_receipts
        if r.effect in ("write", "external_write")
    } | {r.capability_id for r in effect_receipts if r.effect in ("write", "external_write")}
    write_landed = bool(write_receipt_ids & action_set)
    read_work = any(r.effect in ("read", "execute") for r in effect_receipts)
    called_bare = {bare_tool_name(n) for n in called_tool_names} | set(called_tool_names)
    write_attempted = bool(called_bare & action_set)
    return write_landed, read_work, write_attempted


def _resolve_unmet_contract(
    *,
    mode_id: str,
    mode_kind: str,
    stripped_content: str,
    action_set: frozenset[str],
    write_attempted: bool,
    routing_misfire_prose: bool,
    total_tool_calls: int,
    dense_content_min_chars: int,
) -> CompletionContractDecision:
    """Resolve an unmet write contract across routing-misfire, chat-softening, and block."""
    if routing_misfire_prose and stripped_content and not write_attempted:
        return SurfaceProseWithFootnote(action_tool_ids=action_set)

    is_chat = mode_kind == "chat"
    did_work = total_tool_calls > 0
    has_dense_content = len(stripped_content) >= dense_content_min_chars

    if is_chat and has_dense_content and did_work:
        return ChatSoftAccept(content_len=len(stripped_content))
    if is_chat and did_work and not write_attempted:
        return ChatReadFootnote(action_tool_ids=action_set)
    return BlockNoVerifiedWrite(mode_id=mode_id, action_tool_ids=action_set)


def resolve_completion_contract(
    *,
    mode_id: str,
    mode_kind: str = "chat",
    content: str = "",
    action_tool_ids: frozenset[str] | set[str] = frozenset(),
    effect_receipts: Sequence[EffectReceipt] = (),
    called_tool_names: frozenset[str] | set[str] = frozenset(),
    total_tool_calls: int = 0,
    sub_turn_kind: str | None = None,
    yielded: bool = False,
    routing_misfire_prose: bool = False,
    dense_content_min_chars: int = 1500,
    preexisting_block: bool = False,
) -> CompletionContractDecision:
    """Evaluate the post-loop completion contract for a turn.

    Priority order:
    1. **Preexisting lifecycle block**: entry op, plan, or fork child failed.
    2. **Exemptions**: sub-turn child (validated by ``classify_child_turn``),
       yielded turn (paused for user input), or mode with no write contract
       (empty ``action_tool_ids``).
    3. **Write contract verification**:
       - Landed write: write/external_write receipt matching ``action_tool_ids``.
       - Honest read-only: read/execute receipts present with NO write attempt.
    4. **Parent no-tools resolution** (write unmet and not honest read-only):
       - Routing-misfire prose surfacing: aborted synthesized action dispatch
         on a pure prose turn with no write attempt.
       - Chat-mode softening: chat mode with tool calls and dense content.
       - Chat-mode read footnote: chat mode with tool calls and substantive prose.
       - Strict block: unverified success -> ``BlockNoVerifiedWrite``.
    """
    action_set = frozenset(action_tool_ids)
    exemption = _check_exemption(
        preexisting_block=preexisting_block,
        sub_turn_kind=sub_turn_kind,
        yielded=yielded,
        action_set=action_set,
    )
    if exemption is not None:
        return exemption

    write_landed, read_work, write_attempted = _check_receipt_satisfaction(
        action_set=action_set,
        effect_receipts=effect_receipts,
        called_tool_names=called_tool_names,
    )
    if write_landed:
        return AcceptTurn(reason="write receipt verified")
    if read_work and not write_attempted:
        return AcceptTurn(reason="read-only work verified without write attempts")

    return _resolve_unmet_contract(
        mode_id=mode_id,
        mode_kind=mode_kind,
        stripped_content=content.strip(),
        action_set=action_set,
        write_attempted=write_attempted,
        routing_misfire_prose=routing_misfire_prose,
        total_tool_calls=total_tool_calls,
        dense_content_min_chars=dense_content_min_chars,
    )
