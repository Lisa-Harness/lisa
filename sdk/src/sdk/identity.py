"""Shared execution-scope identity for durable, host-owned resources.

Several subsystems need to answer one question: *does this durable resource
belong to the execution context in front of me?* A worktree, a bash session,
its scratch directory, and its sandbox containment fact are all long-lived
artifacts keyed to "who and where" — a workspace, a user, and a surface.

Three places were deriving that key independently, and all three accidentally
folded ``session_plan_digest`` into it: ``WorkProduct.belongs_to``, the host's
``sandbox_session_key``, and the bash pack's ``_session_key`` (the latter two
duplicated the hash across repositories, kept in sync only by a comment). This
module is the single definition they share.

``session_plan_digest`` is deliberately excluded. It is a content hash of the
session manifest *and the full session instruction text*, so it changes every
time a user edits session context. That makes it a configuration version, not
an identity: folding it into a scope key silently orphans live worktrees and
sandbox facts on a benign prompt edit, and an unreachable containment fact
fails closed into extra approval prompts. Compare it separately when staleness
genuinely matters (an approval minted under a different plan should not
auto-execute); never use it to decide ownership.
"""

from __future__ import annotations

import hashlib

__all__ = ["execution_scope", "execution_scope_key"]

_SCOPE_FIELDS = ("workspace_id", "user_id", "user_slug", "surface_type", "surface_id")


def execution_scope(source: object) -> tuple[str, ...]:
    """Return the ordered identity tuple that scopes durable resources.

    ``source`` is anything exposing the scope attributes — a
    ``kernel.models.ToolExecutionContext``, a ``kernel.approval.ApprovalIdentity``,
    or a stored ``sdk.workproducts.WorkProduct``. Typed as ``object`` because
    the SDK must not import the kernel's higher-level models; the shared
    attribute names are the contract.
    """
    return tuple(str(getattr(source, field, "") or "") for field in _SCOPE_FIELDS)


def execution_scope_key(source: object) -> str:
    """Return a stable short digest of :func:`execution_scope` for path naming."""
    raw = "/".join(part for part in execution_scope(source) if part) or "default"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
