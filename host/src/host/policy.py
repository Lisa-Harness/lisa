"""Host-owned approval policy engine.

This module answers one question: *does this action need operator approval?*
It is the only place that decision lives — the mode dial, the allow/ask/deny
rules, the effect-tier ladder, and the sandbox-as-boundary downgrade all
converge on :func:`decide`. Keeping it here (not in the kernel, not in a
pack, not in the CLI) is what keeps the layers cohesive:

- The **kernel** owns the receipt (exact-action, single-use, TTL). Unchanged.
- The **host** owns this decision. The kernel and transports only see an
  :class:`~kernel.approval.ApprovalService`.
- **Packs** publish *facts* (e.g. "this command is contained in a sandbox");
  they never decide policy.
- The **CLI** renders a :class:`PolicyDecision`; it never decides.

Boundary rules enforced by this module:

- **Deny is loud.** A denied action raises (the registry surfaces it to the
  model as a tool error so it can reroute); it is never silently skipped.
- **Fail closed.** Unknown or degraded containment is treated as *not*
  contained, so an unsandboxed ``rm`` can never be auto-approved by mistake.
  A missing or corrupt ``policy.yaml`` resolves to ``mode: ask`` with no
  rules — toward prompting, never toward allowing.
- **`external_write` is never auto-approved** by the sandbox downgrade: the
  sandbox is a *filesystem/process* boundary, and by definition an
  ``external_write`` (cloud APIs, PRs, email) escapes it.

The sandbox fact is read from a neutral JSON file a pack writes under
``<data_dir>/sandbox-facts/<session_key>.json`` — the host does not import
pack code. See :func:`read_containment` for the contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_POLICY_PATH",
    "VERDICT_ALLOW",
    "VERDICT_ASK",
    "VERDICT_DENY",
    "PolicyDecision",
    "PolicyDocument",
    "PolicyMode",
    "PolicyRule",
    "decide",
    "load_policy",
    "read_containment",
    "remember_allow_prefix",
    "sandbox_session_key",
]

#: The policy file lives beside the other durable host state.
DEFAULT_POLICY_PATH = "policy.yaml"

#: Decision verdicts (string constants so ``PolicyDecision`` stays a plain
#: dataclass that transports can serialize without importing the enum).
VERDICT_ALLOW = "allow"
VERDICT_ASK = "ask"
VERDICT_DENY = "deny"


class PolicyMode(StrEnum):
    """The approval posture for actions not settled by a rule or containment.

    - ``ASK``   - prompt for any effectful action (today's behavior).
    - ``TRUST`` - allow ``write``/``execute`` without asking; ``external_write``
      still prompts. The "I'm actively driving and trust the workspace" mode.
    - ``PLAN``  - deny all effectful actions: read-only exploration only.
    """

    ASK = "ask"
    TRUST = "trust"
    PLAN = "plan"


# Effect kinds in rising-risk order. The ladder is the fallback when no rule
# and no containment applies; higher index = higher risk.
_EFFECT_TIER: tuple[str, ...] = ("read", "write", "execute", "external_write")

#: Effects the mode dial may auto-allow in TRUST (external_write never).
_TRUST_AUTO_EFFECTS: frozenset[str] = frozenset({"write", "execute"})

#: Cap on how many actions one turn may auto-approve before the host stops
#: pre-empting the prompt. A runaway loop should surface as a prompt, not a
#: burn through the ledger. (Enforced by the caller; the constant lives here.)
#: See ``host.policy_service`` / ``handle_message``.
AUTO_APPROVE_TURN_CAP = 10

#: Only these effects may be downgraded by sandbox containment. ``write`` and
#: below never gate in ``ask`` mode anyway, and ``external_write`` escapes the
#: sandbox by definition, so neither is eligible.
_DOWNGRADE_ELIGIBLE_EFFECTS: frozenset[str] = frozenset({"execute"})

#: Containment states that permit the downgrade. Anything else — "none"
#: (no sandbox), "degraded" (sandbox present but cannot run), or a missing /
#: unreadable fact — is treated as *not* contained (fail closed).
_CONTAINED_STATES: frozenset[str] = frozenset({"contained"})

#: Parameter keys that carry the shell command for ``execute``-effect tools.
#: Checked in order; the first present non-empty string wins. ``command`` is
#: bash_run's parameter; ``argv`` is bwrap_run's (a list).
_COMMAND_PARAM_KEYS: tuple[str, ...] = ("command", "argv", "cmd", "script")


class PolicyRule(BaseModel):
    """One allow/ask/deny rule. Any subset of matchers; all present must match.

    Matchers (v1, deliberately small):

    - ``tool``           — exact ``target_id`` (e.g. ``bash_run``).
    - ``effect``         — exact effect kind (``read|write|execute|external_write``).
    - ``pack``           — exact ``pack_id``.
    - ``command_prefix`` — the action's shell command starts with this string
      (normalized leading whitespace). Only meaningful for ``execute`` tools
      that carry a command parameter.

    A rule with no matchers matches nothing (defensive: an empty rule is not
    a wildcard).
    """

    tool: str = ""
    effect: str = ""
    pack: str = ""
    command_prefix: str = ""

    def describe(self) -> str:
        """A compact human-readable label for diagnostics and /permissions."""
        parts = [
            f"{key}={value}"
            for key, value in (
                ("tool", self.tool),
                ("effect", self.effect),
                ("pack", self.pack),
                ("command_prefix", self.command_prefix),
            )
            if value
        ]
        return " ".join(parts) or "(no matchers)"


class PolicyDocument(BaseModel):
    """The parsed ``policy.yaml`` document.

    ``version`` is forward-compat scaffolding; v1 is the only schema today.
    Lists are evaluated in a fixed order — deny, then ask, then allow — with
    first match winning, independent of the order keys appear in the file.
    """

    version: int = 1
    mode: PolicyMode = PolicyMode.ASK
    sandbox_downgrade: bool = True
    allow: list[PolicyRule] = Field(default_factory=list)
    ask: list[PolicyRule] = Field(default_factory=list)
    deny: list[PolicyRule] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The engine's verdict for one invocation.

    ``verdict`` is one of the ``VERDICT_*`` constants. ``reason`` is a
    human-readable explanation for the transport (and for ``/approve <id>
    why``). ``matched_rule`` is the rule that fired, when one did; ``tier``
    and ``containment`` carry the resolved ladder tier and the sandbox state
    the decision used, so a transport can badge them without re-deriving.
    """

    verdict: str
    reason: str
    tier: str = ""
    containment: str = ""
    matched_rule: PolicyRule | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"{self.verdict}: {self.reason}"


# ── Loading ─────────────────────────────────────────────────────────────


def load_policy(data_dir: Path) -> PolicyDocument:
    """Load ``<data_dir>/policy.yaml``, failing closed to the ask-mode default.

    Never raises: a missing, unreadable, or invalid file resolves to the
    default document (``mode: ask``, ``sandbox_downgrade: true``, no rules),
    so the safe behavior — prompt — is the behavior on any error.
    """
    path = Path(data_dir) / DEFAULT_POLICY_PATH
    try:
        if not path.is_file():
            return PolicyDocument()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("policy: failed to load %s (%s); using ask-mode default", path, exc)
        return PolicyDocument()
    if raw is None:
        return PolicyDocument()
    if not isinstance(raw, dict):
        logger.warning("policy: %s is not a mapping; using ask-mode default", path)
        return PolicyDocument()
    try:
        return PolicyDocument.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError plus any mapping errors
        logger.warning("policy: %s failed validation (%s); using ask-mode default", path, exc)
        return PolicyDocument()


def remember_allow_prefix(data_dir: Path, *, tool: str, command_prefix: str) -> Path:
    """Append an ``allow`` rule for ``tool`` + ``command_prefix`` to policy.yaml.

    This is the host-side write behind ``/approve <id> --remember`` and the
    'a' approval key. The host owns policy writes so the CLI never touches the
    file directly — the CLI sends the command, the host persists the rule.

    Creates the file with a default header when absent. The write appends to
    the ``allow`` block, preserving whatever is already there. Returns the
    path written.
    """
    path = Path(data_dir) / DEFAULT_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = load_policy(data_dir)
    prefix = command_prefix.strip()
    rule = PolicyRule(tool=tool, command_prefix=prefix)
    if not any(r.tool == tool and r.command_prefix == prefix for r in doc.allow):
        doc.allow.append(rule)
    _write_policy(path, doc)
    logger.info("policy: remembered allow rule %s", rule.describe())
    return path


def _write_policy(path: Path, doc: PolicyDocument) -> None:
    """Serialize ``doc`` to ``path`` as a clean, commented YAML document."""
    payload = {
        "version": doc.version,
        "mode": doc.mode.value,
        "sandbox_downgrade": doc.sandbox_downgrade,
        "allow": [r.model_dump(exclude_defaults=True) for r in doc.allow],
        "ask": [r.model_dump(exclude_defaults=True) for r in doc.ask],
        "deny": [r.model_dump(exclude_defaults=True) for r in doc.deny],
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    header = (
        "# Lisa approval policy. Decide which actions need operator approval.\n"
        "# mode: ask|trust|plan   sandbox_downgrade: auto-approve sandboxed commands\n"
        "# Rules: allow/ask/deny lists, evaluated deny -> ask -> allow, first match.\n"
        "# Matchers: tool, effect (read|write|execute|external_write), pack, command_prefix.\n"
    )
    path.write_text(header + body, encoding="utf-8")


# ── Containment fact ─────────────────────────────────────────────────────


def sandbox_session_key(identity: Any) -> str:
    """Recompute the bash session key from an approval identity.

    The bash pack keys its sessions by
    ``sha256(workspace_id/user_slug/session_plan_digest/surface_id)[:16]``. All
    four inputs are present on :class:`~kernel.approval.ApprovalIdentity`, so
    the engine can find the pack's containment fact without importing pack
    code. This function is the single shared definition of that key — keep it
    in lockstep with the pack's ``_session_key`` (same inputs, same order).
    """
    raw = (
        "/".join(
            part
            for part in (
                getattr(identity, "workspace_id", "") or "",
                getattr(identity, "user_slug", "") or "",
                getattr(identity, "session_plan_digest", "") or "",
                getattr(identity, "surface_id", "") or "",
            )
            if part
        )
        or "default"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def read_containment(data_dir: Path, identity: Any) -> dict[str, str]:
    """Read the sandbox containment fact for ``identity``, failing closed.

    Returns a dict with a ``containment`` key of ``contained`` / ``degraded``
    / ``none`` / ``unknown``. ``unknown`` means no fact file exists (the pack
    never wrote one) — treated as *not* contained by the decision ladder.
    ``degraded`` carries a ``reasons`` list the transport can surface.
    """
    key = sandbox_session_key(identity)
    path = Path(data_dir) / "sandbox-facts" / f"{key}.json"
    try:
        if not path.is_file():
            return {"containment": "unknown"}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("policy: containment fact unreadable (%s); failing closed", exc)
        return {"containment": "unknown"}
    if not isinstance(payload, dict):
        return {"containment": "unknown"}
    state = str(payload.get("containment", "") or "unknown").strip() or "unknown"
    reasons = payload.get("reasons")
    out = {"containment": state}
    if isinstance(reasons, list) and reasons:
        out["reasons"] = ", ".join(str(r) for r in reasons)
    return out


# ── Rule matching ────────────────────────────────────────────────────────


def _command_from_parameters(parameters: dict[str, Any]) -> str:
    """Extract the shell command string from an invocation's parameters.

    Checks the known command-carrying keys in order; for ``argv``-style list
    values joins with spaces so a ``command_prefix`` rule matches the program.
    """
    for key in _COMMAND_PARAM_KEYS:
        value = parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            return " ".join(str(part) for part in value).strip()
    return ""


def _rule_matches(rule: PolicyRule, *, invocation: Any, command: str) -> bool:
    """Return True when every present matcher on ``rule`` matches ``invocation``."""
    present = False
    # pyre-ignore[16] — duck-typed kernel ApprovalInvocation
    target_id = str(getattr(invocation, "target_id", "") or "")
    effect = str(getattr(invocation, "effect", "") or "")
    pack_id = str(getattr(invocation, "pack_id", "") or "")

    if rule.tool:
        present = True
        if rule.tool != target_id:
            return False
    if rule.effect:
        present = True
        if rule.effect != effect:
            return False
    if rule.pack:
        present = True
        if rule.pack != pack_id:
            return False
    if rule.command_prefix:
        present = True
        if not command or not command.startswith(rule.command_prefix):
            return False
    # A rule with no matchers matches nothing (an empty rule is not a wildcard).
    return present


def _match(rules: list[PolicyRule], *, invocation: Any, command: str) -> PolicyRule | None:
    """Return the first matching rule in ``rules``, or None."""
    for rule in rules:
        if _rule_matches(rule, invocation=invocation, command=command):
            return rule
    return None


def _tier_of(effect: str) -> int:
    """Return the ladder index for ``effect`` (0 = read, highest = external_write)."""
    try:
        return _EFFECT_TIER.index(effect)
    except ValueError:
        # Unknown effects rank as the riskiest known tier (fail toward prompting).
        return len(_EFFECT_TIER) - 1


# ── The decision ─────────────────────────────────────────────────────────


def _match_rule(
    policy: PolicyDocument, *, invocation: Any, command: str, tier: str
) -> PolicyDecision | None:
    """Evaluate explicit rules, deny -> ask -> allow, first match wins.

    Returns the settled decision, or None when no rule matches and the caller
    should continue down the ladder.
    """
    deny = _match(policy.deny, invocation=invocation, command=command)
    if deny is not None:
        return PolicyDecision(
            verdict=VERDICT_DENY,
            reason=f"denied by policy rule: {deny.describe()}",
            tier=tier,
            matched_rule=deny,
        )
    ask = _match(policy.ask, invocation=invocation, command=command)
    if ask is not None:
        return PolicyDecision(
            verdict=VERDICT_ASK,
            reason=f"ask rule matched: {ask.describe()}",
            tier=tier,
            matched_rule=ask,
        )
    allow = _match(policy.allow, invocation=invocation, command=command)
    if allow is not None:
        return PolicyDecision(
            verdict=VERDICT_ALLOW,
            reason=f"allow rule matched: {allow.describe()}",
            tier=tier,
            matched_rule=allow,
        )
    return None


def _containment_decision(
    policy: PolicyDocument, *, invocation: Any, effect: str, tier: str, data_dir: Path | None
) -> tuple[PolicyDecision | None, str]:
    """Apply the sandbox-as-boundary downgrade.

    Returns (decision-or-None, resolved-containment). Only ``execute`` actions
    are eligible, and only when the pack reports a genuinely contained state:
    external_write escapes the sandbox by definition, and degraded/unknown
    facts fail closed (so an unsandboxed ``rm`` can never be auto-approved).
    A ``degraded`` result short-circuits to ``ask`` so the operator is told the
    sandbox could not run rather than being silently downgraded.
    """
    containment = "unknown"
    eligible = effect in _DOWNGRADE_ELIGIBLE_EFFECTS
    if not policy.sandbox_downgrade or not eligible or data_dir is None:
        return None, containment
    fact = read_containment(data_dir, getattr(invocation, "identity", None))
    containment = str(fact.get("containment", "unknown") or "unknown")
    if containment in _CONTAINED_STATES:
        return (
            PolicyDecision(
                verdict=VERDICT_ALLOW,
                reason="sandboxed command (contained); auto-approved by sandbox boundary",
                tier=tier,
                containment=containment,
            ),
            containment,
        )
    if containment == "degraded":
        return (
            PolicyDecision(
                verdict=VERDICT_ASK,
                reason=f"sandbox degraded ({fact.get('reasons', 'unknown')}); prompting",
                tier=tier,
                containment=containment,
            ),
            containment,
        )
    return None, containment


def decide(
    *,
    policy: PolicyDocument,
    invocation: Any,
    data_dir: Path | None = None,
) -> PolicyDecision:
    """Decide whether ``invocation`` needs operator approval.

    The evaluation order is fixed and total. Each step is short-circuiting:

    1. A ``deny`` rule matches -> ``deny`` (loud; the caller raises).
    2. An ``ask`` rule matches -> ``ask``.
    3. An ``allow`` rule matches -> ``allow``.
    4. Containment downgrade: a sandboxed ``execute`` action is auto-allowed
       when ``sandbox_downgrade`` is on. This is the "approval only when not
       sandboxed" rule.
    5. Mode fallback - ``plan`` denies all effectful actions, ``trust`` allows
       ``write``/``execute``, ``ask`` prompts. ``read`` never prompts.

    ``data_dir`` is only consulted for the containment fact in step 4; pass it
    (the host data dir) whenever the invocation could be sandbox-contained.
    """
    effect = str(getattr(invocation, "effect", "") or "")
    tier = _EFFECT_TIER[min(_tier_of(effect), len(_EFFECT_TIER) - 1)]
    command = _command_from_parameters(getattr(invocation, "parameters", None) or {})

    # 1-3. Explicit rules, deny -> ask -> allow, first match wins.
    settled = _match_rule(policy, invocation=invocation, command=command, tier=tier)
    if settled is not None:
        return settled

    # 4. Sandbox-as-boundary downgrade.
    downgraded, containment = _containment_decision(
        policy, invocation=invocation, effect=effect, tier=tier, data_dir=data_dir
    )
    if downgraded is not None:
        return downgraded

    # 5. Mode fallback.
    if effect == "read":
        return PolicyDecision(
            verdict=VERDICT_ALLOW,
            reason="read-only action",
            tier=tier,
            containment=containment,
        )
    if policy.mode is PolicyMode.PLAN:
        return PolicyDecision(
            verdict=VERDICT_DENY,
            reason=f"plan mode: {effect} actions are denied (read-only exploration only)",
            tier=tier,
            containment=containment,
        )
    if policy.mode is PolicyMode.TRUST and effect in _TRUST_AUTO_EFFECTS:
        return PolicyDecision(
            verdict=VERDICT_ALLOW,
            reason=f"trust mode: {effect} actions allowed",
            tier=tier,
            containment=containment,
        )
    # Default / ASK mode / trust-mode external_write.
    return PolicyDecision(
        verdict=VERDICT_ASK,
        reason=f"{effect} action requires approval",
        tier=tier,
        containment=containment,
    )
