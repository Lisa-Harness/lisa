# Coding Workflow

`lisa-pack-coding` owns local workspace inspection and Git changes. The
separate `lisa-pack-github` owns credentials, repository reads, and pull
request creation. They share only the host-owned `CodeChangeArtifact` contract.

## Workflow

1. Inspect the configured workspace with bounded read tools.
2. Create an isolated worktree on an owned feature branch.
3. Apply a validated unified diff in that worktree.
4. Run named, session-configured command vectors.
5. Commit explicit changed paths only after required checks pass for the current revision.
6. Push the owned branch without any force option.
7. Open a pull request from the pushed artifact.
8. Remove the clean owned worktree when it is no longer needed.

Each patch increments the artifact revision and clears prior check records.
Checks therefore cannot authorize a later patch. Pull-request publication also
requires the recorded `head_sha`, `commit_sha`, and `pushed_sha` to agree.

## Session Policy

The session plan supplies both packs with their own immutable configuration.
Commands are argv vectors, never shell strings.

```yaml
resources:
  packs:
    include: [lisa-pack-coding, lisa-pack-github]
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
      lisa-pack-github:
        base_branch: main
        allowed_base_branches: [main]
        branch_prefix: "lisa/"
        protected_branches: [main, master, develop, prod, production]
```

The GitHub token and repository remain extension-owned settings (`GITHUB_TOKEN`
and `GITHUB_REPO`); they are not session-plan values.

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
