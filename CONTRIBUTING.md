# Contributing to Lisa Harness

Bug reports, documentation improvements, tests, and focused fixes are welcome.
Read the [Code of Conduct](CODE_OF_CONDUCT.md) and use the
[security policy](SECURITY.md) for private vulnerability reports.

## Before starting

- Search [issues](https://github.com/Lisa-Harness/lisa/issues) for existing work.
  For a substantial API or architecture change, discuss scope before coding.
- Check the active [stabilization freeze and exceptions](docs/CODING_WORKFLOW.md#freeze).
  Core features and refactors are not currently open-ended contribution areas.
  Documentation describing existing behavior is allowed.
- Follow the [package boundaries](docs/BOUNDARIES.md). Application workflows
  normally belong in packs; transport rendering belongs in hosts.

For bugs, include expected/actual behavior, a minimal reproduction, OS/Python,
framework/host/pack versions, and sanitized diagnostics. Do not attach a live
data directory, credentials, or private conversation logs.

## Set up a checkout

Fork/clone the repository, create a feature branch, and run from its root:

```sh
uv sync --frozen --extra dev
uv run pytest packstest/tests -m "not live" -q
```

The workspace does not install the `lisa` CLI. An independently installed
CLI does not automatically use local framework changes; see
[getting started](docs/GETTING_STARTED.md).

## Development loop

1. Start with a narrow reproducer or failing test.
2. Make the smallest complete fix, keeping unrelated refactors out.
3. Run focused tests, then the required [quality and coverage gates](docs/TESTING.md#ci-gates).
4. Review the diff for unintended behavior, secrets, artifacts, and churn.
5. Commit and open a pull request with the problem, solution, verification,
   and any compatibility impact.

Code fixes under the freeze need a regression test that fails before the
fix and passes after, in the same commit. Do not weaken tests, coverage
floors, or the direct-I/O allowlist to manufacture a passing result.

## Pack changes

Use [packstest](packstest/README.md) from source before an installed-package
test. Keep deterministic evals under the `eval` marker; new user-facing
workflows also need an opt-in [whole-runtime scenario](e2etest/README.md).
Do not use a live provider as the first debugging loop.

See [pack authoring](docs/PACK_AUTHORING_GUIDE.md),
[operation authoring](sdk/AUTHORING.md), and the
[external pack registry](https://github.com/Lisa-Harness/lisa-packs).

## Documentation changes

Use the [documentation index](docs/README.md) to find the canonical guide.
Document current behavior and concrete tasks; link rather than copy long
contracts across READMEs. Keep proposals and dated narratives explicitly
separate from instructions.

Validate local links, complete examples, paths, and ownership of commands.
For prose-only changes, no live model test is required. Do not add working-tree
audits or one-off incident reports to the maintained corpus.

## Pull request checklist

- [ ] Scope respects the active freeze and package boundaries.
- [ ] Relevant tests and required CI gates pass; skipped/unrun checks are stated.
- [ ] A fix includes its regression test; a behavior change has explicit approval.
- [ ] Public APIs have typed contracts, docstrings, and co-maintained tests.
- [ ] Errors are observable; no silent success-shaped fallback or broad catch
  outside a justified boundary.
- [ ] No secrets, generated runtime data, local paths, or unrelated formatting.
- [ ] User-facing docs/examples are updated.
- [ ] Breaking changes and material behavior changes are recorded in
  [CHANGELOG.md](CHANGELOG.md).

Python style and test configuration live in [pyproject.toml](pyproject.toml);
see [Python development](docs/python-BCP.md). Repository agent guidance is
described in [the customization guide](docs/COPILOT_CUSTOMIZATION_GUIDE.md).

## Commit conventions

Use concise subjects with a type and optional scope:

- `feat:` new behavior; `fix:` a bug fix; `imp:` non-breaking improvement.
- `docs:` documentation; `test:` tests; `infra:` build/CI/tooling.
- `copilot:` agent customization.

Describe compatibility breaks in the body and changelog. Keep already shared
history intact; prefer a follow-up or `fixup!` commit over rewriting it.

## Licensing and review

Submit only material you have permission to contribute under the project's
[MIT license](LICENSE). Maintainers review correctness, tests, boundaries,
documentation, and maintenance cost; a passing CI run is necessary but does
not guarantee acceptance. There is no guaranteed review response time.
