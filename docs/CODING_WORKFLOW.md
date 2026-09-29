# Coding Workflow

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
