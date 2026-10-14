# Roadmap

This page describes project priorities, not committed delivery dates.
Current work and acceptance criteria belong in
[GitHub issues](https://github.com/Lisa-Harness/lisa/issues); delivered work
belongs in the [changelog](../CHANGELOG.md).

## Current constraint: stabilization

Kernel, runtime, SDK, and first-party MCP behavior remains under the
[stabilization freeze](CODING_WORKFLOW.md#freeze), with recorded scoped
exceptions. Fixes need reproducing regression tests. New capabilities or
refactors in those trees require an explicit exception or an unmerged branch.
Documentation, test coverage, and unfrozen tooling can progress independently.

## Priorities

1. **Reliable, observable completion.** Preserve approval ownership, honest
   effect receipts, checked-commit gates, bounded retries, recovery semantics,
   and diagnostic exits. Extend regressions for real failures rather than
   adding model-specific prose workarounds.
2. **Reproducible integration.** Make framework/host/pack versions, model
   capabilities, source refs, configuration, and build artifacts verifiable
   in clean environments.
3. **Context correctness and economics.** Validate request fitting, reference
   expansion, cache reuse, provider usage normalization, and memory scope
   across representative workflows. Known allocator limitations are listed
   in [context lifecycle](CONTEXT_LIFECYCLE.md#limits-and-follow-up-work).
4. **Contributor and release readiness.** Keep runnable examples, source-based
   pack tests, whole-runtime scenarios, and [release evidence](RELEASE_CHECKLIST.md)
   current. Configure the pending private security/conduct reporting channels
   before public release.

## Proposing work

Open an issue describing the user need, current evidence, proposed boundary,
testable outcome, compatibility impact, and whether the freeze permits it.
Separate existing bugs from optional future designs. A past RFC or a fixed
incident is not an outstanding roadmap commitment.

Additional transports, alternative backends, and product-specific workflows
normally start in host or pack repositories rather than expanding the kernel.
No delivery timeline or support commitment is implied.
