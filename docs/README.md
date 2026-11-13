# Documentation

This is the maintained reference for Lisa Harness. Start with the guide for
your task; module READMEs describe package ownership and implementation details.
The [project README](../README.md) explains the project and its current status.

## Use and operate Lisa

- [Getting started](GETTING_STARTED.md): framework checkout versus CLI
  installation, first model turn, and an offline development check.
- [Configuration](CONFIGURATION.md): data directories, profiles, models,
  provider access, pack activation, and configuration ownership.
- [Session context authoring](SESSION_CONTEXT_AUTHORING_GUIDE.md): strict
  v2 schema, block inheritance, resource selection, and validation.
- [Approval policy](POLICY_GUIDE.md): allow/ask/deny rules, containment,
  and exact-action decisions.
- [Troubleshooting](TROUBLESHOOTING.md): setup failures, catalog collisions,
  blocked work, approvals, and useful diagnostics.
- [Storage and data handling](STORAGE.md): persistence, backups, privacy,
  and the direct-I/O policy.
- [Logging](LOGGING.md): observability configuration and contributor standards.
- [Turn forensics](FORENSICS.md): how to read the evidence for a past turn —
  the four sources, the `turn_index` join, the three failure shapes, and what
  the reader deliberately will not do.

## Understand and extend the framework

- [Architecture and terminology](BOUNDARIES.md): package boundaries, core
  concepts, a turn's lifecycle, and host responsibilities.
- [Host integration](HOST_INTEGRATION.md): composition, async ownership,
  transport callbacks, and the integration checklist.
- [Host SDK architecture](HOST_SDK_ARCHITECTURE.md): planned language-neutral
  hosting, package ownership, process boundaries, and development gates.
- [Host Protocol specification](SPEC/HOST_PROTOCOL.md): proposed v1 messages,
  lifecycle, outcomes, security, and cross-language conformance.
- [Host SDK design primer](HOST_SDK_PRIMER.md): rationale, trade-offs, and
  rejected alternatives for the planned Host SDK.
- [MCP interoperability](MCP_INTEROP.md): running Lisa's first-party MCP
  servers without the turn loop, which servers to expose, identity limits, and
  the client-config export.
- [Pack authoring](PACK_AUTHORING_GUIDE.md): modes, tools, extension entry
  points, packaging, and a local walkthrough.
- [Operation authoring](../sdk/AUTHORING.md): Python versus declarative operations.
- [Mode and workflow routing](ROUTER_DESIGN.md): handoffs, forks, delegation,
  routing hints, and completion contracts.
- [Workflow system cleanup](WORKFLOW_CLEANUP.md): what the workflow machinery
  measures against real use, what still needs fixing, and in what order.
- [Mode taxonomy](MODE_TAXONOMY.md): target-state mode design, pack layering,
  and the migration phases behind it.
- [Model router](../sdk/src/sdk/router/README.md): per-iteration CEL rules
  and provider-independent model selection.
- [Model data layers specification](SPEC/MODEL-DATA-LAYERS.md): separate
  provider facts, measured data, and operator-owned model settings.
- [Cache efficiency specification](SPEC/CACHE-EFFICIENCY.md): telemetry,
  history folding, cache-preserving handoffs, and before/after measurement.
- [Context routing](CONTEXT_ROUTING_GUIDE.md): evidence selectors, lanes,
  ranking, required candidates, and admission budgets.
- [Context lifecycle](CONTEXT_LIFECYCLE.md): lossless references, resident
  evidence, request fitting, prompt caching, and approval continuation.
- [Token budgets](TOKEN_BUDGET.md): per-call, turn, and session limits;
  processed-token accounting versus monetary cost.
- [Memory architecture](MEMORY_ARCHITECTURE.md): scoped recall, learning,
  consolidation, and thread provenance.
- [Thread store specification](SPEC/THREAD_STORE.md): files, records,
  result references, and bounded read paths.
- Observability and replay program (proposed, in order):
  1. [Evidence envelope](SPEC/EVIDENCE-ENVELOPE.md): versioned ledger rows,
     call-level join keys, run provenance, and delegation lineage.
  2. [Model call archive](SPEC/MODEL-CALL-ARCHIVE.md): archived response for
     every in-turn and off-turn model call.
  3. [Turn replay](SPEC/TURN-REPLAY.md): re-run a recorded turn, detect
     divergence, and export it as a regression scenario.
  4. [Trace export and timeline](SPEC/TRACE-EXPORT-AND-TIMELINE.md): span
     tree, OpenTelemetry export, propagation, timeline, and retention. Can
     start after step 1.
- [Prompt startup proposal](SPEC/PROMPT_STARTUP.md): deferred cache/lifecycle,
  in-process content/catalog, import and CLI correctness work.
- [Go Connector SDK contract](GO_CONNECTOR_SDK.md): the language-neutral
  first-party MCP server contract (manifest schema, CLI flags, logging env,
  spawn shape) implemented by `lisa-go`.
- [Turn-loop invariants](lessons/turn-loop-invariants.md): safeguards and
  regression evidence.

## Contribute and maintain

- [Contributing](../CONTRIBUTING.md) and [Python development](python-BCP.md).
- [Testing](TESTING.md), [packstest](../packstest/README.md), and
  [whole-runtime scenarios](../e2etest/README.md).
- [Coding workflow](CODING_WORKFLOW.md), including the standing test
  assurances. The coding-pack workflow is distinct from contributing to this
  framework.
- [Release checklist](RELEASE_CHECKLIST.md) and [roadmap](ROADMAP.md).
- [Security policy](../SECURITY.md) and [Code of Conduct](../CODE_OF_CONDUCT.md).
- [Copilot customization guide](COPILOT_CUSTOMIZATION_GUIDE.md).
- [Changelog](../CHANGELOG.md): release and change history.

## Package references

[Kernel](../kernel/README.md) · [SDK](../sdk/README.md) ·
[Runtime](../runtime/README.md) · [MCP servers](../mcp_server/README.md) ·
[Bundled packs](../sdk/packs/README.md)

## Documentation lifecycle

Guides describe shipped behavior; specifications describe maintained
contracts. Proposals must say what is not implemented and must not be used
as setup instructions.

Keep one canonical reference for each concept and link to it from module
READMEs. Put current work in issues and change history in the changelog/Git,
not working-tree audits, command transcripts, private paths, or incident
reports in the user guide corpus. When a report contains a durable rule,
move that rule into the relevant guide or regression test before removing
the report. Do not retain redirect stubs for superseded documents.
