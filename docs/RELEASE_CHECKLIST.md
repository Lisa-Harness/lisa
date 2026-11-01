# Framework release checklist

Use this checklist for **each framework release**. It is not a claim about
the current working tree, repository visibility, or CI state. Record the
actual release evidence in the release/PR, not by checking off this template
forever.

## PR-based releases and protected main

All changes, including release preparation, reach `main` through pull requests.
Main protection requires the GitHub Actions **Merge gate** check, with the branch
up to date before merging. It covers quality, documentation, and the full
OS/Python test and coverage matrix, including docs-only changes. Failed,
cancelled, or skipped dependencies cannot produce a successful merge gate.

Require PR conversations to be resolved, block force pushes and branch deletion,
and enforce protection for administrators. No approving review is mandatory so a
solo maintainer can merge their own passing PR. Configure these repository
settings in GitHub; workflow files alone do not enable branch protection.

Merge policy is **linear history**: enable **rebase and merge** and **squash and
merge**, disable **merge commits**, and enable required linear history. GitHub
enforces this at the server — a merge commit cannot land and no workflow check
duplicates it. Rebase a branch onto `main` instead of merging `main` into it,
and fold `fixup!` commits with `git rebase --autosquash` before merging.

1. Create a `release/vX.Y.Z` branch from current `main`.
2. Update the root workspace and all ten package versions, refresh `uv.lock`,
   and move the changelog's Unreleased entries into a dated version section.
   Leave an empty Unreleased heading for subsequent changes.
3. Open a release PR with migration notes and complete this checklist. Wait
   for **Merge gate** and all other checks to pass on its latest commit.
4. Run `Publish` manually against the release branch. This runs the release
   validation and distribution build without uploading to PyPI. Verify that
   the run succeeded on the PR's latest commit; rerun after changes.
5. Rebase the release branch if `main` moved, then land the PR with **rebase and
   merge** or **squash and merge** — never a merge commit. Fetch the resulting
   `main`, wait for its checks to pass, then create and push an annotated
   `vX.Y.Z` tag on that verified main commit. Never tag an unmerged release
   branch.
6. Monitor `Publish` through all ten uploads and GitHub release creation.
   Do not move a release tag or reuse a published version.

Host and external-pack releases belong to their own repositories. Do not
gate a framework release on a hardcoded list or historical count of external
packs.

## Scope and compatibility

- [ ] Choose the release commit, version, and intended audience. Confirm the
  [test assurances](CODING_WORKFLOW.md#test-assurances) hold for the tree.
- [ ] Review [CHANGELOG.md](../CHANGELOG.md), including breaking API,
  configuration, data-format, and pack/host compatibility changes.
- [ ] Document required upgrade steps and tested host/pack revisions.
  Do not imply a compatibility matrix that has not been exercised.
- [ ] Confirm which workspace members are actually published by
  [publish.yml](../.github/workflows/publish.yml).

## Public-project readiness

- [ ] README, setup/configuration, authoring, host integration, troubleshooting,
  and contribution links work without a maintainer's sibling checkouts.
- [ ] License and distribution metadata match the release.
- [ ] Enable and test GitHub private vulnerability reporting as required by
  [SECURITY.md](../SECURITY.md).
- [ ] Publish and test the confidential conduct contact in
  [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md); the setup notice is unresolved
  until an actual channel is available.
- [ ] Review repository visibility, description, topics, issue settings,
  branch protection, and issue/PR templates.
- [ ] Check the tree and history for secrets, private data, generated artifacts,
  and environment-specific paths using appropriate scanning tools.
- [ ] Verify external documentation/installation links as a non-maintainer.
  Do not assume access to a private host or pack repo.

## Verification

- [ ] Run the [CI quality/test/coverage gates](TESTING.md#ci-gates) at the
  release commit on the supported matrix; investigate failures rather than
  relying on a dated green baseline.
- [ ] Validate bundled packs from source with `packstest`.
- [ ] Run selected opt-in live/sandbox scenarios needed for the changed
  integrations, recording models, configuration, and outcomes.
- [ ] Verify documentation examples and local links.
- [ ] Build and inspect wheels/sdists: namespace layout, Python modules,
  entry points, authored Markdown/YAML, manifests, and required package data.
- [ ] Test clean-environment imports and entry points from the built artifacts,
  not only editable source. Check optional dependencies independently.
- [ ] Smoke-test a compatible external host separately. Its `lisa verify`
  is not a framework CI gate or a substitute for live workflow evidence.

## Build and publication contract

The current publish workflow runs reusable CI, requires a tag's commit to be on
`main`, validates each target's name and version against the pushed `v<version>`
tag and the workspace version, requires a dated changelog section for that
version, builds with `uv build --no-sources --all-packages`, checks the output
counts, and publishes through PyPI trusted publishing. Manual dispatch validates
and builds without publishing.

Its current ten targets are:

| Workspace path | Distribution |
| --- | --- |
| `kernel/` | `lisa-harness-kernel` |
| `sdk/` | `lisa-harness-sdk` |
| `mcp_server/` | `lisa-harness-mcp-servers` |
| `runtime/` | `lisa-harness-runtime` |
| `sdk/packs/core/` | `lisa-pack-core` |
| `sdk/packs/introspect/` | `lisa-pack-introspect` |
| `sdk/packs/memory/` | `lisa-pack-memory` |
| `sdk/packs/recovery/` | `lisa-pack-recovery` |
| `packstest/` | `lisa-harness-packstest` |
| `e2etest/` | `lisa-harness-e2e` |

`packstest` and `e2etest` are workspace members that are current publish
targets. The root workspace is not a packaged application.

- [ ] Update target versions coherently and refresh the lockfile.
- [ ] Confirm the proposed tag equals every publish target's version.
- [ ] Confirm PyPI trusted-publisher settings and required approvals.
- [ ] Push only the deliberately chosen release tag after maintainer sign-off.
  A `v*` tag starts publication; do not use it as a build experiment.
- [ ] Verify all expected wheels and sdists reached the registry.
- [ ] Publish release notes with migration guidance and known limitations.
- [ ] Test a fresh installation against the registry and a compatible host.

If targets or workflow behavior change, update this checklist in the same
change. Do not leave one-off findings or an old working-tree diff here.
