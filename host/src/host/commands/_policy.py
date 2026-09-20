"""``/policy`` — inspect and manage the host approval policy.

The approval policy engine (``host.policy``) reads ``<data_dir>/policy.yaml``
on every request, so an operator editing the file mid-session takes effect
immediately — but there was no in-host way to *read* the effective policy or
*mint* an edit without hand-authoring YAML. This command is that surface:

- ``/policy``           — render the active document: mode dial, sandbox
  downgrade/bare-mode guards, the allow/ask/deny rule lists, and any load
  errors (a non-compiling ``when`` expression fails the whole document to
  ask-mode defaults — the render is the discovery surface for that).
- ``/policy mode <m>``  — set the fallback dial (``ask|trust|plan``).
- ``/policy bare on|off`` — toggle ``bare_requires_approval`` (the guard that
  keeps unsandboxed ``execute`` actions prompting even under ``trust``).
- ``/policy rule add <allow|ask|deny> [--tool T] [--effect E] [--pack P]
  [--prefix PX] [--when EXPR]`` — append a rule. The expression is
  compile-checked BEFORE the write so a bad add is rejected instead of
  poisoning the document into ask-mode defaults.
- ``/policy rule remove <list>:<n>`` — drop rule ``n`` (1-based) from the
  named list (``allow``/``ask``/``deny``).
- ``/policy rule list`` — render the rules with their remove addresses.

The host owns every policy write (the same rule ``--remember`` obeys): a
transport never edits ``policy.yaml`` itself, it sends command text. Mode,
bare, and rule edits preserve the other rule lists and rewrite the document
with the standard commented header via :func:`host.policy.write_policy` (the
same serializer ``remember_allow_prefix`` uses), so hand-authored comments on
other keys survive an operator's next manual edit but a mode flip never
silently drops rules.

Dependency direction: ``_policy`` → ``host.policy``, ``host.runtime``. It is
imported lazily by ``host.handle_message`` so the dispatcher's import cost
stays flat.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any

from host.policy import (
    DEFAULT_POLICY_PATH,
    VERDICT_ALLOW,
    VERDICT_ASK,
    VERDICT_DENY,
    PolicyMode,
    PolicyRule,
    load_policy,
    load_policy_with_errors,
    validate_rule_expressions,
    write_policy,
)
from host.runtime import HostRuntime

logger = logging.getLogger(__name__)

__all__ = ["handle_policy_command"]

#: Token index of the first argument after the verb ("/policy mode trust" ->
#: parts[2] == "trust").
_ARG_TOKEN_INDEX = 2

#: Token index of the rule verb in "/policy rule <verb> ..." (parts[2]) and of
#: the case-preserving remainder after it (parts[3], maxsplit=3).
_RULE_VERB_INDEX = 2
_RULE_REMAINDER_INDEX = 3

#: Valid rule lists, in the order the render shows them.
_RULE_LISTS = (VERDICT_DENY, VERDICT_ASK, VERDICT_ALLOW)


def _rule_line(rule: Any, address: str = "") -> str:
    """Render one policy rule as a compact Markdown bullet."""
    matchers = [
        f"{key}: `{value}`"
        for key, value in (
            ("tool", getattr(rule, "tool", "")),
            ("effect", getattr(rule, "effect", "")),
            ("pack", getattr(rule, "pack", "")),
            ("command_prefix", getattr(rule, "command_prefix", "")),
            ("when", getattr(rule, "when", "")),
        )
        if value
    ]
    label = f"`{address}` " if address else ""
    return "- " + label + (", ".join(matchers) if matchers else "(no matchers)")


def render_policy_document(runtime: HostRuntime) -> str:
    """Render the effective policy document as Markdown for ``/policy``."""
    policy, load_errors = load_policy_with_errors(Path(runtime.paths.data_dir))
    lines: list[str] = ["## Approval Policy"]
    lines.append("")
    lines.append(f"- source: `{Path(runtime.paths.data_dir) / DEFAULT_POLICY_PATH}`")
    lines.append(f"- mode: **{policy.mode.value}**")
    lines.append(f"- sandbox_downgrade: {str(policy.sandbox_downgrade).lower()}")
    lines.append(
        f"- bare_requires_approval: {str(policy.bare_requires_approval).lower()} "
        "(unsandboxed execute always prompts when on)"
    )

    if load_errors:
        lines.append("")
        lines.append("### ⚠ load errors (ask-mode defaults active)")
        lines.extend(f"- {error}" for error in load_errors)

    for label in _RULE_LISTS:
        rules = getattr(policy, label)
        lines.append("")
        lines.append(f"### {label} rules ({len(rules)})")
        if rules:
            lines.extend(
                _rule_line(rule, address=f"{label}:{index}")
                for index, rule in enumerate(rules, start=1)
            )
        else:
            lines.append("- (none)")
    lines.append("")
    lines.append(
        "Set the dial: `/policy mode ask|trust|plan` — "
        "trust auto-allows write/execute (bare-mode guard still applies), "
        "plan denies every effectful action. "
        "Toggle the guard: `/policy bare on|off`. "
        "Manage rules: `/policy rule add|remove|list`."
    )
    return "\n".join(lines)


def _set_policy_fields(
    runtime: HostRuntime,
    *,
    mode: PolicyMode | None = None,
    bare_requires_approval: bool | None = None,
) -> None:
    """Persist a policy edit, preserving the existing rule lists.

    Writes through :func:`host.policy.write_policy` (the shared serializer
    with the commented header) so a mode flip never drops rules and the file
    stays hand-editable.
    """
    data_dir = Path(runtime.paths.data_dir)
    policy = load_policy(data_dir)
    if mode is not None:
        policy.mode = mode
    if bare_requires_approval is not None:
        policy.bare_requires_approval = bare_requires_approval
    write_policy(data_dir, policy)
    logger.info(
        "policy: updated (mode=%s bare=%s)",
        policy.mode.value,
        policy.bare_requires_approval,
    )


def _parse_rule_add_flags(raw: str) -> tuple[PolicyRule | None, str]:
    """Parse ``rule add`` flags into a rule, or return an error message.

    ``--when`` consumes the RAW remainder after its token (never shell-
    tokenized): a CEL expression's double-quoted string literals
    (``action.argv[0] == "rm"``) must survive verbatim, and shlex would
    strip the quotes. ``--when`` is therefore last-flag-wins-tail — the
    same grammar ``/approve --remember <prefix>`` uses. Flags before it
    (``--tool``/``--effect``/``--pack``/``--prefix``) are shlex-split, so
    single-token values may be quoted.

    At least one matcher is required — an empty rule matches nothing, so
    an accidental bare add must be rejected loudly instead of silently
    inert.
    """
    head, sep, when_expr = raw.partition("--when")
    fields: dict[str, str] = {}
    if sep:
        when_expr = when_expr.strip()
        if not when_expr:
            return None, "--when requires an expression."
        # The raw remainder keeps REPL-style whole-expression quoting
        # (``'a == b'``), and a single-quoted string is itself valid — but
        # ALWAYS-TRUTHY — CEL: persisted verbatim it silently widens an
        # allow rule to match everything. When the whole remainder is one
        # quoted token (shlex probe: exactly one token from a quote-led
        # string), unquote it; multi-token remainders keep their inner
        # string literals verbatim (``'x' == 'y'`` is three tokens).
        if when_expr[0] in "'\"":
            try:
                probe = shlex.split(when_expr)
            except ValueError:
                probe = []
            if len(probe) == 1:
                when_expr = probe[0]
        fields["when"] = when_expr
    try:
        tokens = shlex.split(head)
    except ValueError as exc:
        return None, f"Could not parse rule flags ({exc})."
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        if flag in {"--tool", "--effect", "--pack", "--prefix"} and i + 1 < len(tokens):
            key = {"--prefix": "command_prefix"}.get(flag, flag.lstrip("-"))
            fields[key] = tokens[i + 1].strip()
            i += 2
            continue
        return None, f"Unknown or incomplete flag {flag!r}."
    rule = PolicyRule(**fields)  # type: ignore[arg-type]
    if not any((rule.tool, rule.effect, rule.pack, rule.command_prefix, rule.when)):
        return None, "A rule needs at least one matcher (--tool/--effect/--pack/--prefix/--when)."
    return rule, ""


def _handle_rule_add(runtime: HostRuntime, args: str) -> str:
    """Handle ``/policy rule add <list> [flags]``."""
    parts = args.split(None, 1)
    list_name = parts[0].lower() if parts else ""
    if list_name not in _RULE_LISTS:
        return f"Unknown rule list {list_name!r}. Use: {'|'.join(_RULE_LISTS)}."
    rule, error = _parse_rule_add_flags(parts[1] if len(parts) > 1 else "")
    if rule is None:
        return error
    if rule.when:
        # Compile-check BEFORE the write: a non-compiling expression would
        # fail the whole document to ask-mode defaults on the next load —
        # the add must be rejected, not poison the policy.
        errors = validate_rule_expressions([rule])
        if errors:
            return f"Rejected: {errors[0]}"
    data_dir = Path(runtime.paths.data_dir)
    policy = load_policy(data_dir)
    getattr(policy, list_name).append(rule)
    write_policy(data_dir, policy)
    logger.info("policy: rule added to %s: %s", list_name, rule.describe())
    return f"Rule added to **{list_name}**: {rule.describe()}. Takes effect immediately."


def _handle_rule_remove(runtime: HostRuntime, args: str) -> str:
    """Handle ``/policy rule remove <list>:<n>``."""
    address = args.strip().lower()
    list_name, sep, raw_index = address.partition(":")
    if not sep or list_name not in _RULE_LISTS or not raw_index.isdigit():
        return "Usage: /policy rule remove <list>:<n> (e.g. allow:2). See /policy rule list."
    index = int(raw_index)
    data_dir = Path(runtime.paths.data_dir)
    policy = load_policy(data_dir)
    rules = getattr(policy, list_name)
    if not 1 <= index <= len(rules):
        return f"No rule {list_name}:{index} — the list has {len(rules)} rule(s)."
    removed = rules.pop(index - 1)
    write_policy(data_dir, policy)
    logger.info("policy: rule removed from %s: %s", list_name, removed.describe())
    return f"Removed {list_name}:{index} ({removed.describe()}). Takes effect immediately."


def _handle_rule_list(runtime: HostRuntime) -> str:
    """Handle ``/policy rule list`` — rules with their remove addresses."""
    policy, load_errors = load_policy_with_errors(Path(runtime.paths.data_dir))
    lines: list[str] = ["## Policy rules"]
    if load_errors:
        lines.extend(["", "### ⚠ load errors (ask-mode defaults active)"])
        lines.extend(f"- {error}" for error in load_errors)
    for label in _RULE_LISTS:
        rules = getattr(policy, label)
        lines.append("")
        lines.append(f"### {label} ({len(rules)})")
        if rules:
            lines.extend(
                _rule_line(rule, address=f"{label}:{index}")
                for index, rule in enumerate(rules, start=1)
            )
        else:
            lines.append("- (none)")
    lines.append("")
    lines.append(
        "Remove with `/policy rule remove <list>:<n>`, e.g. `/policy rule remove allow:1`."
    )
    return "\n".join(lines)


_VERB_ADD = "add"
_VERB_REMOVE = "remove"
_VERB_LIST = "list"


def _handle_rule(runtime: HostRuntime, stripped: str) -> str:
    """Dispatch ``/policy rule ...`` subcommands from the original-cased text."""
    # Parse verbs case-insensitively but keep the original casing for flag
    # values (a CEL expression is case-sensitive; the dispatch below never
    # lowercases the remainder). "/policy rule add ..." splits as
    # ["/policy", "rule", "add", "<flags remainder>"] via maxsplit=3.
    parts = stripped.split(None, _RULE_REMAINDER_INDEX)
    verb = parts[_RULE_VERB_INDEX].lower() if len(parts) > _RULE_VERB_INDEX else ""
    if verb not in {_VERB_ADD, _VERB_REMOVE, _VERB_LIST}:
        return (
            "Usage: /policy rule add <allow|ask|deny> [--tool T] [--effect E] "
            "[--pack P] [--prefix PX] [--when EXPR] | remove <list>:<n> | list."
        )
    if verb == _VERB_LIST:
        return _handle_rule_list(runtime)
    remainder = parts[_RULE_REMAINDER_INDEX] if len(parts) > _RULE_REMAINDER_INDEX else ""
    if verb == _VERB_ADD:
        return _handle_rule_add(runtime, remainder)
    return _handle_rule_remove(runtime, remainder)


def _handle_mode(runtime: HostRuntime, parts: list[str]) -> str:
    """Handle ``/policy mode <m>``."""
    raw_mode = parts[_ARG_TOKEN_INDEX].strip() if len(parts) > _ARG_TOKEN_INDEX else ""
    try:
        mode = PolicyMode(raw_mode)
    except ValueError:
        known = "|".join(m.value for m in PolicyMode)
        return f"Unknown mode {raw_mode!r}. Use: {known}."
    _set_policy_fields(runtime, mode=mode)
    return f"Policy mode set to **{mode.value}**. Takes effect immediately."


def handle_policy_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/policy`` and its subcommands.

    Returns the reply text; the dispatcher owns transport emission and the
    thread entry. Deliberately synchronous: every policy read/write on this
    path is local file I/O, and the dispatcher awaits nothing here (mirrors
    ``handle_simple_command``'s shape).
    """
    # "/policy" | "/policy mode trust" | "/policy bare off" | "/policy rule ..."
    # The rule branch parses the ORIGINAL string (case-preserving) below;
    # the simple branches only need the lowercased tokens.
    lowered = stripped.lower()
    parts = lowered.split()
    sub = parts[1] if len(parts) > 1 else ""

    if sub == "rule":
        return _handle_rule(runtime, stripped)

    if sub == "mode":
        return _handle_mode(runtime, parts)

    if sub == "bare":
        raw = parts[_ARG_TOKEN_INDEX].strip() if len(parts) > _ARG_TOKEN_INDEX else ""
        if raw in {"on", "true", "1", "yes"}:
            value = True
        elif raw in {"off", "false", "no"}:
            value = False
        else:
            current = load_policy(Path(runtime.paths.data_dir)).bare_requires_approval
            return f"Usage: /policy bare on|off (currently {str(current).lower()})."
        _set_policy_fields(runtime, bare_requires_approval=value)
        return f"bare_requires_approval set to **{str(value).lower()}**. Takes effect immediately."

    if sub in {"", "show", "status"}:
        return render_policy_document(runtime)

    return (
        "Usage: /policy [mode ask|trust|plan | bare on|off | rule add|remove|list]. "
        "Bare /policy renders the active document."
    )
