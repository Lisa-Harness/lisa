# Coding Workflow

This guide describes the **coding/GitHub packs' agent workflow**, not the
framework contributor loop. Contributors should start with
[CONTRIBUTING.md](../CONTRIBUTING.md) and [testing](TESTING.md).

`lisa-pack-coding` owns local workspace inspection and Git changes. The
separate `lisa-pack-github` owns credentials, repository reads, and pull
request creation. They share the host-owned `WorkProduct` contract.

## Workflow

1. Inspect the configured workspace with bounded read tools.
2. Create an isolated worktree on an owned feature branch.
3. Apply a validated unified diff in that worktree.
4. Run named, session-configured checks inside the bwrap sandbox with the owned
   worktree bound writable (`direct-write`). Missing or degraded containment
   fails the check; it never falls back to host execution.
5. Commit only after every required check passes for the current revision and
   candidate Git tree. The host verifies a real changed commit at `HEAD`, the
   required check fingerprints, the task-owned artifact, and a clean worktree
   before the executor can report completion.
6. Push the branch, open a pull request, or clean up only on a separate
   operator request. None is part of the default finish line.

Each patch increments the artifact revision and clears prior check records.
Checks therefore cannot authorize a later patch. Pull-request publication also
requires the recorded `head_sha`, `commit_sha`, and `pushed_sha` to agree.
Existing artifacts without a creation-task ID or check-tree fingerprint cannot
certify a new task.

A task with no configured required checks may finish with a real changed commit
and a clean owned worktree, and that finish certifies nothing about whether the
change is correct. Completion verification confirms that the required set
passed; with an empty set there is no correctness signal to confirm. An empty
set is the ordinary state, not a misconfiguration: `WorkProduct.required_check_ids`
defaults to empty and no completion path asserts otherwise. A caller that
requires checks to have run must declare a non-empty set in the pack's
[session policy](#session-policy), because the verifier does not infer or require
one. Only the end-to-end `checks_current` oracle rejects an empty set, and it
guards the scenario suite rather than production completion.

## Session Policy

The session plan supplies both packs with their own immutable configuration.
Commands are argv vectors, never shell strings.

```yaml
resources:
  packs:
    include: [lisa-pack-coding, lisa-pack-bwrap, lisa-pack-github]
    configuration:
      lisa-pack-coding:
        workspace_root: "."
        checks:
          - id: focused-tests
            argv: [uv, run, pytest, -q]
            timeout_seconds: 600
        required_check_ids: [focused-tests]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
        push_remote: origin
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: sandbox
            persistence: direct-write
            network: deny
            paths:
              # Worktree parent: the coding pack defaults worktrees to
              # <data-dir>/git-worktrees/<workspace-name> (set worktree_dir in
              # the coding pack configuration to override).
              <data-dir>/git-worktrees/<workspace-name>: rw
      lisa-pack-github:
        base_branch: main
        allowed_base_branches: [main]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
```

The GitHub token and repository remain extension-owned settings (`GITHUB_TOKEN`
and `GITHUB_REPO`); they are not session-plan values. Point the bwrap `rw`
grant at the resolved data-dir worktree parent (or the configured
`worktree_dir`) so checks can write into owned worktrees; a narrower `ro`
bind on one worktree takes precedence and prevents checks from running.

## Test Assurances

The shared verification loop and test receipts are standing repository
assurances — they apply to every code change on `main`.

### What every code change is held to

- **Regression tests.** A fix lands with a test that reproduces the bug
  (it fails before the fix and passes after, ideally in the same commit).
  The test is the fix's receipt. The red-before, green-after form is
  preferred but not required: a commit naming and covering the fixed
  behavior with a passing test satisfies the assurance.

- **Tests welcome.** New tests for existing behavior are always welcome;
  they raise the cost of future regressions.

- **Docs and comments** must only describe existing behavior.

### Enforcement for agents

1. Before editing code, the agent must be able to name the behavior it
   changes and the test that covers it. If it cannot, it must stop and
   surface the gap to the operator instead of proceeding.
2. A fix commit's diff to `src/` must be accompanied by a diff to `tests/`
   in the same commit.
3. The verification loop in `.github/instructions/code-quality.md` must pass
   *in its entirety*, not just for touched files. A commit that breaks an
   untouched test is out of scope by definition.
4. Keep "while I'm here" edits out of fix commits. If you spot an adjacent
   issue, file it as a follow-up; commit the fix-sized diff alone.

### Standing suite guards

The kernel and sdk test suites carry autouse settings-hermeticity guards
(`kernel/tests/conftest.py`, `sdk/tests/conftest.py`) that fail any test
leaking mutations to the `kernel.config.settings` or `sdk.config.settings`
singletons. Suite hermeticity — no passes-in-isolation,
fails-in-full-suite flakiness — is a standing gate, not a convention.

### Verification evidence

Run the current [quality/test/coverage gates](TESTING.md#ci-gates) on each
candidate revision. A dated baseline, file count, or test count does not
certify the current tree. Historical baseline repairs remain in
Git/change history; they are not current release blockers.

## Task goal

A checked-commit task ends in one of two ways: a real commit through the mode's
artifact producer, or a declared diagnostic. The declaration is `task_goal`,
carried on `TurnIdentity` (`kernel/src/kernel/turn_hooks.py`) with values
`implementation` and `diagnostic`.

**Undeclared tasks default to `implementation` and fail closed.** A mode with
`task_completion: checked_commit` that makes no worktree and never invokes its
producer blocks, rather than accepting the turn on prose. The block message
reads:

```plaintext
This task declares `task_goal: implementation` (the default), so it must commit
through `<producer>` but the turn never invoked it. If this task is
investigation only, append `[task_goal: diagnostic]` to your message and re-run
(spacing and case are flexible).
```

### Declaring a diagnostic task

There are two surfaces, and they apply at different levels.

**An operator opening a task** uses the inline marker on the message:

```plaintext
diagnose the flaky retry test [task_goal: diagnostic]
```

The runtime consumes the marker (`_TASK_GOAL_MARKER_RE`,
`runtime/src/runtime/handle_message.py`), strips it from the query before the
model sees it, and stamps the resolved goal on the turn. The match is
case-insensitive and tolerates whitespace anywhere inside the brackets, so
`[task_goal:diagnostic]`, `[task_goal:  diagnostic]` and
`[ task_goal : Diagnostic ]` are equivalent to the canonical form.

Detection and stripping share that one pattern, so a declaration can never be
removed from the message without also being read — a property a regression test
asserts directly. Only the value `diagnostic` changes the contract; any other
value is stripped and logged, and the turn keeps the `implementation` default
rather than accepting a declaration it does not understand.

This is the only operator-facing syntax — there is no CLI flag for it.

**A parent turn delegating a child** sets the `task_goal` parameter on the
`delegate` operation, which defaults to `implementation`. The model can set it,
which is intended: the parent scope is choosing what the child is for, and the
mutation guard below still applies to the child. A child goal never widens the
parent's own completion contract.

What neither surface can do is bypass the write check, so a diagnostic
declaration cannot launder a task that mutated the workspace.

### What a diagnostic goal does and does not permit

A diagnostic task is accepted without a producer only when the turn attempted
no producer **and** recorded zero `write`/`external_write` effect receipts. Any
write blocks it, so the worst a mistaken declaration can cause is a loud
false-block, never a false-pass with changes on disk. Accepting the turn stamps
`completion_contract="diagnostic_soft_accept"` on the result for audit; the
stamp is informational in v1 and gates nothing.

Declare a diagnostic for investigation. Do not declare one to avoid committing
work that should have been committed — the mutation guard will block it, and
the turn will have spent its budget.

## Approval

Worktree creation, patching, checks, commits, pushes, cleanup, and pull-request
creation require exact host approval. An approval digest binds the target,
validated parameters, session identity, plan digest, and target-pack policy.

Use the request-specific command returned by the host:

```plaintext
/approve <request_id>
```

The host resumes only that saved invocation and consumes the receipt once.
`/deny <request_id>` leaves it unusable.

## Repository Navigation

The GitHub `repo` mode is read-only. `repo_search_code` finds candidates on the
repository default branch. `repo_lookup` fetches content at the requested ref
and returns evidence only when that target-ref content still matches the query.
Use `repo_read_file` or `repo_list_tree` for non-default ref exploration.
