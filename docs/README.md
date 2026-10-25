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

## Understand and extend the framework

- [Architecture and terminology](BOUNDARIES.md): package boundaries, core
  concepts, a turn's lifecycle, and host responsibilities.
- [Host integration](HOST_INTEGRATION.md): composition, async ownership,
  transport callbacks, and the integration checklist.
- [Pack authoring](PACK_AUTHORING_GUIDE.md): modes, tools, extension entry
  points, packaging, and a local walkthrough.
- [Operation authoring](../sdk/AUTHORING.md): Python versus declarative operations.
- [Mode and workflow routing](ROUTER_DESIGN.md): handoffs, forks, delegation,
  routing hints, and completion contracts.
- [Model router](../sdk/src/sdk/router/README.md): per-iteration CEL rules
  and provider-independent model selection.
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
