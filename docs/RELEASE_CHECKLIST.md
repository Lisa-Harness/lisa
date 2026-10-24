# Framework release checklist

Use this checklist for **each framework release**. It is not a claim about
the current working tree, repository visibility, or CI state. Record the
actual release evidence in the release/PR, not by checking off this template
forever.

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

The current publish workflow runs reusable CI, validates each target's name
and version against the pushed `v<version>` tag, builds with `uv build
--no-cache`, checks the output counts, and publishes through PyPI trusted
publishing.

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
