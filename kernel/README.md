# Kernel

`lisa-harness-kernel` owns runtime-neutral typed contracts and the model/tool
turn loop. It does not depend on another in-repository package. Hosts inject
wiring through `TurnHooks` and `ContextSeams`; packs provide concrete
capabilities.

Start with [architecture and terminology](../docs/BOUNDARIES.md).
Most application authors should use the [shared runtime](../runtime/README.md)
rather than build a new composition root directly against kernel internals.

## Module map

Paths below are relative to [src/kernel](src/kernel/).

| Module | Responsibility |
| --- | --- |
| [turn_runner](src/kernel/turn_runner/) | Shared completion/dispatch loop, transitions, continuations, outcomes |
| [turn_phases](src/kernel/turn_phases/) | Lifecycle, prompts/schemas, routing hints, fork/delegate and completion contracts |
| [turn_hooks](src/kernel/turn_hooks.py) | Host-injected callbacks and context seams |
| [models](src/kernel/models.py) | Execution contexts/results, modes/stages, receipts, state/evidence, budgets |
| [ipsar](src/kernel/ipsar.py) / [ipsar_fsm](src/kernel/ipsar_fsm.py) | Mode command parsing, catalogs, workflow/routing transitions |
| [session_plan](src/kernel/session_plan.py) | Strict v2 schema and deterministic compilation |
| [context](src/kernel/context/) | Evidence mapping, lanes, provenance, request fitting, public seam defaults |
| [operations](src/kernel/operations/) | Runtime/declarative operation discovery and execution |
| [tools](src/kernel/tools/) | Pack-tool registry, MCP pool, schema translation and synthetic aliases |
| [seams](src/kernel/seams/) | Host-facing protocols and adapters |
| [model_manifest](src/kernel/model_manifest.py) | Model roster validation and resolution |
| [provider_contracts](src/kernel/provider_contracts.py) | Completion/embedding contracts and provider-neutral usage |
| [execution_policy](src/kernel/execution_policy.py) / [execution_ledger](src/kernel/execution_ledger.py) | Resource bounds and consumed-work accounting |
| [approval](src/kernel/approval.py) / [turn_continuation](src/kernel/turn_continuation.py) | Exact-action decisions and saved dispatch state |
| [recovery](src/kernel/recovery.py) | Write-ahead work-item state and retry policy |
| [workflow_position](src/kernel/workflow_position.py) / [workflow_view](src/kernel/workflow_view.py) | Typed planned/ad-hoc progress and thread-derived workflow state |
| [slice](src/kernel/slice.py) | Pressure-derived opportunistic housekeeping budgets |
| [intake](src/kernel/intake/) | Signal collector and processing contracts |
| [testing](src/kernel/testing/) | Public stubs, extension harness, import-boundary support |

Provider implementations live in the SDK; the kernel defines/discovers
their gateway contracts. Content scanning lives in the SDK/catalog adapter;
the kernel consumes catalogs and imports selected implementation modules.

## Turn contract

`run_turn` accepts a query, resolved mode/model context, budgets, hooks,
and identity. The runtime normally prepares those inputs.

1. Guard input and establish mode/workflow context.
2. Run configured entry/pre-turn behavior, assemble instructions/evidence,
   and prepare the allowed tool schemas.
3. Resolve each iteration's model, fit the complete request, call it, and
   account for provider usage.
4. Dispatch authorized operations, pack tools, MCP tools, or intercepted
   lifecycle actions; record outcomes and effect receipts.
5. Evaluate explicit lifecycle policy: continue, transition, fork/delegate,
   ask/yield, approval, pause/terminate, or complete.
6. Guard output and run completion/persistence/post-turn hooks as applicable.

Do not equate an answer string with successful action completion.
`TurnResult` carries blocks, handoffs, stage transitions, child outcomes,
approvals/yields, usage, receipts, and evidence needed by the host.

See [mode/workflow routing](../docs/ROUTER_DESIGN.md),
[context lifecycle](../docs/CONTEXT_LIFECYCLE.md),
[token budgets](../docs/TOKEN_BUDGET.md), and the
[invariant registry](../docs/lessons/turn-loop-invariants.md).

## TurnHooks and ContextSeams

[TurnHooks](src/kernel/turn_hooks.py) supplies model resolution/completion,
prompt assembly, authorized operation/tool execution, catalogs/pool access,
state/services/identity, workflow updates, persistence, and slices.
Callbacks are integration seams, not permission to bypass registry policy.

[`ContextSeams`](src/kernel/turn_runner/_seams.py) groups content registry, model policy, provider gateways,
event bus, extension-item builder, and input/output guards.
Public defaults live in [kernel.context](src/kernel/context/).
Default content guards are **pass-through**. Applications needing filtering
must install appropriate implementations.

Input blocks prevent model work. Output guards can replace unsafe output;
hosts must persist/deliver the permitted result, not leak the rejected text.
Privacy also depends on authenticated identity, storage/retrieval filters,
and delivery routing; it cannot be implemented by a prompt alone.

For a custom composition, follow [host integration](../docs/HOST_INTEGRATION.md).
Keep MCP lifecycle startup/use/shutdown in one task and event loop.

## Dispatch and operation contracts

Operations use `OperationExecutionContext` and return `OperationResult`.
They can be Python (`.yml` + `.py`) or declarative (`.operation.md`).
Pack tools use `ToolExecutionContext` / `ToolResult`.
Operations interact with the host through `ctx.host`; do not use retired
`ctx.runtime` examples.

The kernel authorizes catalog capabilities, validates effects/limits, and
uses declared tool surfaces for operation dispatch. Exposing a capability
as an ambient default does not make it a required completion effect.
Markdown operation definitions take precedence over a colocated Python
definition; the Python file can supply the response model.

External tools use the MCP pool; catalog/content access uses the content
seams. In-process Python tools remain trusted code, not sandboxed merely
because they were discovered through MCP.

Use [operation authoring](../sdk/AUTHORING.md),
[pack authoring](../docs/PACK_AUTHORING_GUIDE.md), and
[approval policy](../docs/POLICY_GUIDE.md) for metadata and examples.

## Slices and intake

Slices are opportunistic, budgeted work. Budget grows with the elapsed gap
since the previous slice:

```text
base_budget_ms + max_deferred_ms * (1 - 0.5^(gap_seconds / pressure_half_life_seconds))
```

The result is capped. Policies select pre-turn, post-turn, explicit, or
disabled execution. This is not a continuously running scheduler.
Hosts can separately schedule the one-shot intake process for ambient
signal collection and processing; no cron/timer belongs inside the turn loop.

## Testing and contribution

From the workspace root:

```sh
uv run pytest kernel/tests kernel/src/kernel/context/tests -m "not live" -q
```

The kernel test support is public for pack/extension authors. Use injected
fakes and isolated configuration; do not import private app modules or live
external packs into kernel unit tests.

See [testing](../docs/TESTING.md) for the full CI matrix/gates and
[contributing](../CONTRIBUTING.md).
