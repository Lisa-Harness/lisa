# Roadmap

This page describes project priorities, not committed delivery dates.
Current work and acceptance criteria belong in
[GitHub issues](https://github.com/Lisa-Harness/lisa/issues); delivered work
belongs in the [changelog](../CHANGELOG.md).

## Stability

Kernel, runtime, SDK, and first-party MCP behavior are held to standing
[test assurances](CODING_WORKFLOW.md#test-assurances): the full verification
loop and regression coverage for fixes, plus scoped, receipted changes when
work cannot ship in a single commit. Application workflows normally belong
in packs; transports belong in hosts.

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
testable outcome, and compatibility impact.
Separate existing bugs from optional future designs. A past RFC or a fixed
incident is not an outstanding roadmap commitment.

Additional transports, alternative backends, and product-specific workflows
normally start in host or pack repositories rather than expanding the kernel.
No delivery timeline or support commitment is implied.
