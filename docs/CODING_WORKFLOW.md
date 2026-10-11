# Coding Workflow

> **Stabilization freeze (declared 2026-10-04):** `kernel/`, `runtime/`,
> `sdk/`, and `mcp_server/` are frozen to behavior-changing work. Within those
> trees, the only acceptable commits are bug **fixes** (each with a regression
> test proving the bug) and **tests**. No new features, no refactors, no
> signature changes to public APIs. New capability lands on unmerged
> branches until the freeze lifts. See the Freeze section below for the full
> rules agents must follow.

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
certify a new task. A task with no configured required checks may finish with a
real changed commit and a clean owned worktree.

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
              /path/to/worktree-parent/.lisa-worktrees: rw
      lisa-pack-github:
        base_branch: main
        allowed_base_branches: [main]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
```

The GitHub token and repository remain extension-owned settings (`GITHUB_TOKEN`
and `GITHUB_REPO`); they are not session-plan values.
Replace the bwrap path with the directory adjacent to `workspace_root` where
the coding pack creates its owned worktrees. A narrower `ro` bind there takes
precedence and prevents checks from running.

## Freeze

**Operator exception (2026-10-10):** The freeze is lifted for context
preservation: cache-stable model requests, kernel-owned approval/cap
continuations, consistent evidence residency, and bounded-memory archive
access. Public contract changes required for this work are authorized.
Regression tests and the shared verification loop remain required. Other
work remains subject to the stabilization freeze.

Declared 2026-10-04. Applies to `kernel/`, `runtime/`, `sdk/`, and
`mcp_server/` (including `sdk/packs/`). Rationale: sustained churn at
~3,000 changed lines/day with a ~47% fix-commit ratio over September was
regenerating bugs faster than tests could catch them; the only lever that
reliably restores trust is reducing the rate of behavior change to zero for
a stabilization window.

### What is allowed in frozen trees

- **Bug fixes.** Each fix must land with a test that reproduces the bug
  (fails before the fix, passes after). The test is the fix's receipt.
- **Tests.** New tests for existing behavior are always welcome; they raise
  the cost of future regressions.
- **Docs and comments** that only describe existing behavior.

### What is not allowed

- New features, new modules, new operations, new routing rules.
- Refactors (renames, extractions, file moves, signature changes to public
  APIs, behavior-preserving rewrites). If a refactor is required to land a
  fix, the fix is out of scope for the freeze — escalate instead.
- New dependencies or dependency bumps in frozen trees.
- Configuration changes that alter turn-loop, routing, or lifecycle
  behavior. (Adding a *new* knob is a feature; changing a default is a
  behavior change.)

Unfrozen work (docs outside code, e2etest scenarios, new packs, hosts,
tooling) proceeds normally.

### Enforcement for agents

1. Before editing a file under a frozen tree, the agent must be able to
   name the bug it is fixing and the test that will prove it. If it cannot,
   it must stop and surface the freeze to the operator instead of proceeding.
2. A fix commit's diff to `src/` must be accompanied by a diff to `tests/`
   in the same commit.
3. The verification loop in `.github/instructions/code-quality.md` must pass
   *in its entirety*, not just for touched files. A freeze commit that
   breaks an untouched test is out of scope by definition.
4. Agents must not propose "while I'm here" follow-up refactors. The
   fix-sized diff that leaves the surrounding code alone is the correct
   diff, even if the surrounding code has obvious problems.

### Freeze window activity log (2026-10-05)

Compliance review of `86a803f9..HEAD` (declared freeze `86a803f9` through
`bd95268b`) recorded the following. Carve-outs are known only from commit
messages — the operator's decisions were not recorded here when made;
treat the pushed violations below as unratified until the operator says
otherwise.

**Allowed work (compliant):** every fix commit from `04335b92` through
`cdec3067` ships its `src/` + `tests/` diff in one commit with a
red/green receipt; the four `docs:`, `test:`, and `fixup!` test-infra
commits and all packstest/e2etest work are explicitly unfrozen. Full
verification loop re-verified green at review time (2955 passed,
1 xfailed; coverage floors 7/7).

**Commits claiming an operator carve-out in their message (unpushed, on
`main`):** `668cf86b` (`feat(pack-recovery)` bulk dismiss commands),
`9c2711ad` (idle-deadline wrapper redesign), `6a169483` (dispute scope
repair), `144c022e` (digest live-state reconciliation), `cdec3067`
(state-settling command thread entries).

**Pushed violations recorded after the fact (on `origin/main`):**

- `a4ae5420` `refactor(kernel,runtime)!: move operator command vocabulary
  to the runtime seam` — new public kernel API
  (`YieldRemediation`/`YieldRemediationAction`), public function removed
  (`step_blocks_instance`), BREAKING CHANGE to `render_approval_request`
  output. Per the freeze these required an operator escalation to
  authorize; the commit message claims none. Kept on `main` and now
  documented in the changelog; the operator may reclassify it as a
  post-hoc carve-out or plan a revert.
- `181314db` `feat(sdk-core): take in planner + explainer modes from
  lisa-pack-personas` — new modes with routing front matter inside frozen
  `sdk/packs/core` (new modules + routing rules). No carve-out claimed;
  content-only move with no `mode_id` collisions (verified against the
  external pack's remaining modes).
- `7aa7453e` `docs(core): widen default-mode clarification guidance` —
  changes model-steering prompt behavior in a frozen pack.
- `c9c8b9e6` `chore(deps): refresh uv.lock` — transitive bumps inside
  frozen-tree behavior surfaces, notably `pydantic` 2.3.1 → 2.4.0 (no
  direct dependency changes; lock maintenance mid-freeze).

**Freeze-baseline repairs:** `04335b92` repaired one over-length line
`86a803f9` left in `sdk/src/sdk/thread/_readout.py`; the freeze-declaration
commit's `ruff format --check` baseline was therefore red, and the
"100%-green baseline" claim in its message was inaccurate for that gate.
Post-remediation fixups `bd95268b`, `9c74544f`, and `865b84f7` repair
findings from the 2026-10-05 compliance review (format-gate drift,
`pending`-state misannotation, timed-child relay clobber), all red/
green verified.

### Declared carve-outs

The operator may open a named, bounded exception for work that cannot
reasonably ship as a fix with regression test. A carve-out is recorded
here and is the carve-out's receipt: it scopes the exception to named
modules, bounds it by commit scope, expires with the stated condition,
and is closed by updating this section (same doc, same history).

**Thread-context durability (opened 2026-10-05, authorized by the
operator).** Scope: `sdk/src/sdk/thread/` (readout + results projection
bounds), `runtime/thread/`, `runtime/thread_digest.py`,
`runtime/approval/orchestration.py` (approved-fact attachment),
`runtime/context_assembly.py` (context rendering),
`kernel/session_plan.py` (context/digest knobs only),
`kernel/turn_runner/` (digest consumption points), and the
`lisa-thread` tools in `mcp_server/src/mcp_server/thread/`. Purpose:
repair the hot/cold context boundary so a deep verification turn does
not collapse into re-reading its own prior tool results at 180-char
teaser fidelity. Any commit under this heading must (1) name the
turn-loop behavior it changes, (2) ship a test that fails before and
passes after (same-commit, red/green), and (3) not widen scope to
unrelated signatures. The carve-out closes when these trees settle on a
behavior the operator ratifies, or explicitly on operator order — record
the closure here.

**Context residency by allocation (opened 2026-10-07, authorized by the
operator).** Scope: `sdk/src/sdk/thread/_results.py` (stub envelope
projection), `sdk/src/sdk/thread/_readout.py` and `_search.py` (ref
surfaces the stub row aligns to), `runtime/src/runtime/thread/_replay.py`
and `runtime/thread/` (turn-boundary replay working set),
`runtime/src/runtime/context_assembly.py` (replay lane wired into the
allocator), `kernel/src/kernel/context/context_budget.py` (the
replay/residency lane budget), `kernel/src/kernel/turn_runner/__init__.py`
(turn-start working-set assembly, the per-iteration standing state ledger
via the existing dynamic-segment path, the append-only within-turn
invariant), `kernel/src/kernel/turn_phases/_dispatch.py` (receipt capture
feeding the ledger and the `active_refs` source), and
`mcp_server/src/mcp_server/thread/server.py` (the result shapes the stub
points at). Amended 2026-10-07 (operator-ratified budget design) to add:
`kernel/src/kernel/session_plan.py` (the
`context.history.replay_max_bytes` knob only — schema addition, same
category as the thread-context carve-out's context/digest-knob exception;
no other session-plan surface), `runtime/src/runtime/turn_hooks_builder.py`
(the `prior_messages` replay-budget wiring only), and
`sdk/packs/core/src/sdk/packs/core/sessions/default.md` (the
`context.history.replay_max_bytes` default only).
`kernel/src/kernel/execution_ledger.py` and
`kernel/src/kernel/workflow_position.py` remain *read sources only* for
the ledger — no signature changes to anything another package imports.
Purpose: make context residency bounded by recomputing a budgeted resident
working set at each turn boundary (last-N events verbatim, one stub row
per older/over-threshold event, a standing receipts-derived state ledger),
while every byte stays persisted and recoverable by content/turn address.
No compression, no model-authored summary, no in-place rewrite of any
in-flight message. Design record: `docs/SPEC/RESIDENCY_RFC.md`. Any commit
under this heading must (1) name the turn-loop behavior it changes,
(2) ship a test that fails before and passes after (same-commit,
red/green), (3) not widen scope to unrelated signatures, and (4) keep the
full verification loop green in its entirety. The carve-out closes when
these trees settle on a behavior the operator ratifies, or explicitly on
operator order — record the closure here.

The full suite (`uv run pytest -m "not live"`) is green as of this
declaration (2909 passed, 1 xfailed). The kernel and
sdk test suites carry autouse settings-hermeticity guards
(`kernel/tests/conftest.py`, `sdk/tests/conftest.py`) that fail any test
leaking mutations to the `kernel.config.settings` or `sdk.config.settings`
singletons.

### Frozen baseline (verified 2026-10-04, post-remediation)

All six quality gates pass on commit containing this file:

- `uv run ruff format --check .` — 567 files formatted.
- `uv run ruff check .` — all checks passed.
- `uv run python scripts/check_direct_io_policy.py` — zero in-tree findings
  (allowlist updated to match the post-2026-09-19 thread/storage refactor
  call sites; only out-of-tree `lisa-packs/` advisory entries remain).
- `uv run mypy --explicit-package-bases --exclude '(^|/)(tests|test_[^/]*\.py$)' kernel/src/kernel sdk/src/sdk mcp_server/src/mcp_server`
  — Success: no issues found in 174 source files.
- `uv run vulture kernel/src/kernel sdk/src/sdk mcp_server/src/mcp_server runtime/src/runtime vulture_allowlist.py`
  — no findings.
- `uv run pytest -m "not live" --ignore=e2etest` — 2909 passed, 1 xfailed.

> Test-baseline refresh (2026-10-04, test-suite remediation): the six
> colocated in-src test packages are now wired into `testpaths` (~300
> recovered tests), the hollow xfail stubs and `migrated…` skip-only tests
> in `runtime/tests/test_core_services.py` were removed, and
> `xfail_strict = true` is set — hence zero skips and zero xpassed. A
> per-package coverage gate now runs in CI (`uv run coverage run -m pytest …`
> then `uv run python scripts/check_coverage.py`); `fail_under = 76`
> (branch, source-only) is the suite floor. See `docs/tests-review.md`.

Mypy remediated 12 errors across 4 files by fixing genuine variable-shadowing
bugs (`match` reused for both cursor regex and search results in
`_readout.py`), a missing type parameter on `EMPTY_PACK_CONFIGURATION`, and
narrowing failures on `result`/`projected` dict types. No behavior change;
existing `sdk/tests/test_thread_search.py` (47 references) confirms the
refactor.

### Lifting the freeze

The freeze lifts when all of the following hold for two consecutive weeks:

- Fix-commit ratio (commits whose type is `fix:`) is below 0.20.
- No full-suite-only failure (passes in isolation, fails in the full suite)
  has been observed.
- CI is green on `main` on every Python version/OS in the matrix.

Track the fix-commit ratio with
`git log --oneline --since="14 days ago" | grep -c '^[0-9a-f]* fix'` divided
by total commits in the window.

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
