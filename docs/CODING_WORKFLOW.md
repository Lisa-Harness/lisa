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
