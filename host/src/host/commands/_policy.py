"""``/policy`` — inspect and manage the host approval policy.

The approval policy engine (``host.policy``) reads ``<data_dir>/policy.yaml``
on every request, so an operator editing the file mid-session takes effect
immediately — but there was no in-host way to *read* the effective policy or
*mint* an edit without hand-authoring YAML. This command is that surface:

- ``/policy``           — render the active document: mode dial, sandbox
  downgrade/bare-mode guards, and the allow/ask/deny rule lists.
- ``/policy mode <m>``  — set the fallback dial (``ask|trust|plan``).
- ``/policy bare on|off`` — toggle ``bare_requires_approval`` (the guard that
  keeps unsandboxed ``execute`` actions prompting even under ``trust``).

The host owns every policy write (the same rule ``--remember`` obeys): a
transport never edits ``policy.yaml`` itself, it sends command text. Mode and
bare edits preserve the existing rule lists and rewrite the document with the
standard commented header via :func:`host.policy.write_policy_document` (the
same serializer ``remember_allow_prefix`` uses), so hand-authored comments on
other keys survive an operator's next manual edit but a mode flip never
silently drops rules.

Dependency direction: ``_policy`` → ``host.policy``, ``host.runtime``. It is
imported lazily by ``host.handle_message`` so the dispatcher's import cost
stays flat.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from host.policy import (
    DEFAULT_POLICY_PATH,
    VERDICT_ALLOW,
    VERDICT_ASK,
    VERDICT_DENY,
    PolicyMode,
    load_policy,
    write_policy,
)
from host.runtime import HostRuntime

logger = logging.getLogger(__name__)

__all__ = ["handle_policy_command"]

#: Token index of the first argument after the verb ("/policy mode trust" ->
#: parts[2] == "trust").
_ARG_TOKEN_INDEX = 2


def _rule_line(rule: Any) -> str:
    """Render one policy rule as a compact Markdown bullet."""
    matchers = [
        f"{key}: `{value}`"
        for key, value in (
            ("tool", getattr(rule, "tool", "")),
            ("effect", getattr(rule, "effect", "")),
            ("pack", getattr(rule, "pack", "")),
            ("command_prefix", getattr(rule, "command_prefix", "")),
        )
        if value
    ]
    return "- " + (", ".join(matchers) if matchers else "(no matchers)")


def render_policy_document(runtime: HostRuntime) -> str:
    """Render the effective policy document as Markdown for ``/policy``."""
    policy = load_policy(Path(runtime.paths.data_dir))
    lines: list[str] = ["## Approval Policy"]
    lines.append("")
    lines.append(f"- source: `{Path(runtime.paths.data_dir) / DEFAULT_POLICY_PATH}`")
    lines.append(f"- mode: **{policy.mode.value}**")
    lines.append(f"- sandbox_downgrade: {str(policy.sandbox_downgrade).lower()}")
    lines.append(
        f"- bare_requires_approval: {str(policy.bare_requires_approval).lower()} "
        "(unsandboxed execute always prompts when on)"
    )

    for label, rules in (
        (VERDICT_DENY, policy.deny),
        (VERDICT_ASK, policy.ask),
        (VERDICT_ALLOW, policy.allow),
    ):
        lines.append("")
        lines.append(f"### {label} rules ({len(rules)})")
        if rules:
            lines.extend(_rule_line(rule) for rule in rules)
        else:
            lines.append("- (none)")
    lines.append("")
    lines.append(
        "Set the dial: `/policy mode ask|trust|plan` — "
        "trust auto-allows write/execute (bare-mode guard still applies), "
        "plan denies every effectful action. "
        "Toggle the guard: `/policy bare on|off`."
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


def handle_policy_command(
    runtime: HostRuntime,
    stripped: str,
    message: str,
) -> str:
    """Handle ``/policy`` and its ``mode``/``bare`` subcommands.

    Returns the reply text; the dispatcher owns transport emission and the
    thread entry. Deliberately synchronous: every policy read/write on this
    path is local file I/O, and the dispatcher awaits nothing here (mirrors
    ``handle_simple_command``'s shape).
    """
    # "/policy" | "/policy mode trust" | "/policy bare off"
    parts = stripped.lower().split()
    sub = parts[1] if len(parts) > 1 else ""

    if sub == "mode":
        raw_mode = parts[_ARG_TOKEN_INDEX].strip() if len(parts) > _ARG_TOKEN_INDEX else ""
        try:
            mode = PolicyMode(raw_mode)
        except ValueError:
            known = "|".join(m.value for m in PolicyMode)
            return f"Unknown mode {raw_mode!r}. Use: {known}."
        _set_policy_fields(runtime, mode=mode)
        return f"Policy mode set to **{mode.value}**. Takes effect immediately."

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
        "Usage: /policy [mode ask|trust|plan | bare on|off]. "
        "Bare /policy renders the active document."
    )
