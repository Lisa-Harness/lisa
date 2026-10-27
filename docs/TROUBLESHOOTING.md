# Troubleshooting

Start with the smallest failing case. Record the framework/host/pack versions,
OS, Python version, command, and first error before changing configuration.
Do not fix an authorization or schema error by raising token limits.

## Setup and startup

| Symptom | Check | Next action |
| --- | --- | --- |
| `lisa` is not found after `uv sync` | This workspace does not ship that command | Install the separate [CLI](GETTING_STARTED.md#run-the-terminal-host); check its tool PATH |
| Checkout edits have no effect in the CLI | `uv tool` and workspace `.venv` are different environments | Follow the CLI's local-source installation instructions |
| Models cannot load or resolve | Active data root and `<data-dir>/config/models.yaml`, or session `context.models.path` | Review and apply a valid proposal; ensure `mode_models.default` resolves |
| Session compilation fails | Unknown keys, missing body, undeclared lanes, overlapping selectors | Use the strict [session schema](SESSION_CONTEXT_AUTHORING_GUIDE.md) |
| Required pack installation fails | Source ref, trust policy, credentials, dependencies | Repair the declared source or install through the host's pack workflow; do not ignore a required-pack failure |
| Duplicate mode/operation/tool ID | Catalog owners and duplicated installs | Set the session `providers` owner; remove unintended duplicate installations |
| An installed pack is absent | Activation filter, enabled lock state, host-mandatory packs | Inspect `/pack` and session selection; installation and activation are different |
| MCP startup fails | Resolved executable/environment and server logs | First-party adapters require explicit `--data-dir`; required warm-server failures block startup |
| Async cancel-scope/task error | Where resources are started and closed | Keep startup, turns, and shutdown in the [same task/event loop](HOST_INTEGRATION.md#async-ownership) |

Provider credentials are host-owned. Confirm the selected provider's settings
without printing secrets. `lisa verify` diagnoses installation, not every
live capability. Check `lisa --help` for the external host's syntax.

## Turns, approvals, and incomplete work

- **Capability block:** compare the mode's `required_capabilities` with the
  resolved manifest model. Configure a suitable model/router rather than
  disabling the gate.
- **Pending approval:** inspect the exact action and use the request-specific
  approval/denial command. Do not replay a write manually just because the
  agent paused; it may already have an effect.
- **Policy denial:** inspect `/policy`, its rules, and the containment fact.
  `trust` does not make an unsandboxed execution contained.
- **Pause or budget exit:** distinguish request-window pressure, per-call
  output truncation, cumulative turn limits, and session limits using
  [token budgets](TOKEN_BUDGET.md). Preserve the continuation evidence.
- **Claimed completion without an artifact:** inspect effect receipts and
  task gates, not just the final text. A checked-commit task also needs one
  owned artifact, current checks, the real commit, and a clean worktree.
- **Stalled/indeterminate dispatch:** use the active
  [recovery commands](../sdk/packs/recovery/src/sdk/packs/recovery/README.md).
  Verify external effects before retrying an indeterminate write.
- **Missing old evidence:** thread context is a bounded projection. Use
  thread range/search/content tools to read addressed evidence; don't assume
  absence from the prompt means deletion from storage.
- **A fetched page was refused:** the fetch error names the HTTP status
  (`403`, `401`, `429`). A refusal is the site's bot protection, not a
  Lisa failure — try a different source rather than retrying the same URL.
  If the same host fails for every URL, check whether the site requires
  JavaScript rendering: `fetch_page` reads the served HTML and does not
  execute scripts.

## Useful diagnostics

The active runtime exposes `/help`, `/settings`, `/models`, `/modes`,
`/operations`, `/tools`, `/tokens`, `/last`, and `/debug`.
Available pack commands depend on activation. CLI command equivalents belong
to the CLI's own documentation.

Under the resolved data root:

- `logs/runtime.jsonl`: structured failures, routing and lifecycle decisions.
  Rotates at 5 MiB with 3 backups, so a window read must span
  `runtime.jsonl.1`..`.3` — readers that open only the active file silently
  lose older evidence.
- `logs/token-usage.jsonl`: provider-reported usage and cache counts.
- `logs/decisions.jsonl`: one line per lifecycle decision and per dispatch,
  with the `turn_index` that joins them to the thread and the cost log. See
  [decision telemetry](LOGGING.md#decision-telemetry).
- `data/threads/`: turn records, transcripts, sealed segments, request/result
  references. Use the [thread readout](SPEC/THREAD_STORE.md) to expand refs.
- `state/`: approval, recovery, and work-product state.

Logs, thread objects, debug output, and live scenario bundles can contain
private content. Share minimal sanitized excerpts, not the whole data directory
or `.env`. Follow [logging](LOGGING.md) and [security reporting](../SECURITY.md).

## Reproduce before fixing

1. For a pack tool or catalog failure, use [packstest](../packstest/README.md)
   from source with the failing arguments and a temporary data directory.
2. For runtime wiring, use injected fakes and the closest runtime regression
   test. Run targeted non-live tests from the workspace root.
3. For whole-workflow failures, inspect the
   [scenario bundle](../e2etest/README.md#debugging-a-failed-run), reproduce the
   failing layer deterministically, then rerun an opt-in scenario.
4. File a [bug](https://github.com/Lisa-Harness/lisa/issues) with expected
   versus actual behavior, versions, the minimal reproduction, and sanitized
   diagnostics. Use private reporting for exploitable vulnerabilities.
