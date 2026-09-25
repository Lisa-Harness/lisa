"""Tests for the policy-decorating approval service (host/policy_service.py).

These drive ``PolicyApprovalService.request()`` against the real
``FileApprovalService`` store on disk, asserting the three decision outcomes:

- **ask**  -> the record persists as ``pending`` (today's unchanged path);
- **allow** -> the record persists and a real receipt is minted (auto-approve
  is auditable, never a silent skip);
- **deny** -> a ``ToolExecutionError`` is raised so the model sees a tool
  error and reroutes (loud, never silent).

They also cover the ``--remember`` write path: the service's decision is
re-evaluated from disk each request, so a rule written after construction
takes effect without a restart.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from host.approvals import FileApprovalService
from host.policy import (
    PolicyDocument,
    PolicyMode,
    PolicyRule,
    remember_allow_prefix,
    sandbox_session_key,
)
from host.policy_service import PolicyApprovalService
from kernel.approval import build_approval_invocation
from kernel.tools.registry import ToolExecutionError


def _identity() -> object:
    from kernel.approval import ApprovalIdentity

    return ApprovalIdentity(
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        mode_id="shell",
        session_plan_digest="plan-digest",
    )


def _make_service(tmp_path: Path, policy: PolicyDocument | None) -> PolicyApprovalService:
    store = FileApprovalService(tmp_path / "state" / "approvals.json")
    return PolicyApprovalService(store, policy=policy, data_dir=tmp_path)


def _bash_invocation(command: str, *, identity: object | None = None) -> object:
    return build_approval_invocation(
        target_kind="tool",
        target_id="bash_run",
        pack_id="lisa-pack-bash",
        effect="execute",
        parameters={"command": command},
        identity=identity or _identity(),
        pack_config={},
        safe_preview="run a shell command",
    )


def _write_containment(tmp_path: Path, identity: object, state: str) -> None:
    key = sandbox_session_key(identity)
    path = tmp_path / "run" / key / "sandbox-facts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps({"containment": state}), encoding="utf-8")


@pytest.mark.asyncio
async def test_ask_decision_persists_pending_no_receipt(tmp_path: Path) -> None:
    """ask mode (the default) keeps the unchanged prompt path."""
    service = _make_service(tmp_path, PolicyDocument(mode=PolicyMode.ASK))
    invocation = _bash_invocation("ls -la")
    await service.request(invocation)

    record = await service.get(invocation.request_id)
    assert record is not None
    assert record.state == "pending"
    assert record.receipt is None


@pytest.mark.asyncio
async def test_allow_decision_mints_receipt(tmp_path: Path) -> None:
    """An allow-matched action is persisted AND self-approved (receipt on disk)."""
    policy = PolicyDocument(allow=[PolicyRule(tool="bash_run", command_prefix="git status")])
    service = _make_service(tmp_path, policy)
    invocation = _bash_invocation("git status")
    await service.request(invocation)

    record = await service.get(invocation.request_id)
    assert record is not None
    assert record.state == "approved"
    assert record.receipt is not None
    assert record.receipt.action_digest == invocation.action_digest


@pytest.mark.asyncio
async def test_contained_execute_is_auto_approved(tmp_path: Path) -> None:
    """The headline: a sandboxed command mints a receipt, never a prompt."""
    identity = _identity()
    _write_containment(tmp_path, identity, "contained")
    service = _make_service(tmp_path, PolicyDocument())  # default ask + downgrade on
    invocation = _bash_invocation("ls", identity=identity)
    await service.request(invocation)

    record = await service.get(invocation.request_id)
    assert record is not None
    assert record.state == "approved"
    assert record.receipt is not None


@pytest.mark.asyncio
async def test_uncontained_execute_stays_pending(tmp_path: Path) -> None:
    """No containment fact -> the unsandboxed command prompts as before."""
    service = _make_service(tmp_path, PolicyDocument())
    invocation = _bash_invocation("rm -rf /tmp/x")
    await service.request(invocation)

    record = await service.get(invocation.request_id)
    assert record is not None
    assert record.state == "pending"


@pytest.mark.asyncio
async def test_deny_rule_raises_tool_error_loudly(tmp_path: Path) -> None:
    """A deny rule raises ToolExecutionError so the model reroutes."""
    policy = PolicyDocument(deny=[PolicyRule(tool="bash_run", command_prefix="rm")])
    service = _make_service(tmp_path, policy)
    with pytest.raises(ToolExecutionError, match="denied by policy"):
        await service.request(_bash_invocation("rm -rf /tmp/x"))


@pytest.mark.asyncio
async def test_deny_leaves_no_pending_record(tmp_path: Path) -> None:
    """A denied action is never persisted as pending (it was never offered)."""
    policy = PolicyDocument(deny=[PolicyRule(effect="execute")])
    service = _make_service(tmp_path, policy)
    invocation = _bash_invocation("anything")
    with pytest.raises(ToolExecutionError):
        await service.request(invocation)
    assert await service.get(invocation.request_id) is None


@pytest.mark.asyncio
async def test_passthrough_methods_delegate(tmp_path: Path) -> None:
    """approve/deny/consume/get/list_pending behave exactly like the store."""
    store = FileApprovalService(tmp_path / "state" / "approvals.json")
    service = PolicyApprovalService(
        store, policy=PolicyDocument(mode=PolicyMode.ASK), data_dir=tmp_path
    )
    invocation = _bash_invocation("ls")
    await service.request(invocation)
    ident = invocation.identity

    # list_pending surfaces the single pending record.
    pending = await service.list_pending(ident)
    assert len(pending) == 1

    # approve + consume flow through to a real receipt, single-use.
    receipt = await service.approve(invocation.request_id, ident)
    assert receipt is not None
    consumed = await service.consume(
        invocation.request_id,
        build_approval_invocation(
            target_kind="tool",
            target_id="bash_run",
            pack_id="lisa-pack-bash",
            effect="execute",
            parameters={"command": "ls"},
            identity=ident,
            pack_config={},
            safe_preview="x",
        ),
    )
    assert consumed is not None

    # deny on a fresh request returns True and persists the decision.
    second = _bash_invocation("pwd")
    await service.request(second)
    assert await service.deny(second.request_id, ident, reason="not now") is True
    record = await service.get(second.request_id)
    assert record is not None and record.state == "denied"


@pytest.mark.asyncio
async def test_remember_rule_takes_effect_immediately(tmp_path: Path) -> None:
    """--remember write flips a later identical command from ask to allow."""
    identity = _identity()
    # Policy reloads from disk each request (no pinned static policy).
    service = _make_service(tmp_path, None)

    first = _bash_invocation("make build", identity=identity)
    await service.request(first)
    assert (await service.get(first.request_id)).state == "pending"

    # The host-side --remember write persists an allow rule for the prefix.
    remember_allow_prefix(tmp_path, tool="bash_run", command_prefix="make build")

    # The next identical command auto-approves without a restart.
    second = _bash_invocation("make build --verbose", identity=identity)
    await service.request(second)
    record = await service.get(second.request_id)
    assert record is not None
    assert record.state == "approved"


@pytest.mark.asyncio
async def test_request_id_is_idempotent_for_pending(tmp_path: Path) -> None:
    """Re-requesting the same still-pending action reuses the request id."""
    service = _make_service(tmp_path, PolicyDocument(mode=PolicyMode.ASK))
    first = await service.request(_bash_invocation("ls"))
    second = await service.request(_bash_invocation("ls"))
    assert first.request_id == second.request_id
