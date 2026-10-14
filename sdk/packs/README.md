# Bundled reference packs

These are independently packaged distributions in the shared `sdk.packs`
namespace, not submodules included in the SDK wheel.

| Pack | Role | Activation |
| --- | --- | --- |
| [core](core/) | Default, planner, and explainer modes; delegation/planning operations, fetch tool, shared prompts, default session | Host-mandatory |
| [introspect](introspect/) | Operational self-model and reflective assessment | Host-mandatory |
| [memory](memory/) | Scoped recall/learning operations, memory backend manifest, consolidation/mining slice | Optional |
| [recovery](recovery/src/sdk/packs/recovery/README.md) | Inspect, retry, skip/cancel stalled dispatches and resume yielded turns | Optional |

All four are runtime dependencies. That does not make every optional pack
active in every session. Host policy, pack lock state, and the session plan
control activation. Use `/pack`, `/modes`, `/operations`, and `/tools` to
inspect the active catalog.

## Authoring and ownership

Packs export content roots and other extension entry points. They provide
workflow and application capability, not the host's transport or durable
service implementation. New packs should depend on kernel/SDK contracts
and injected services, not runtime internals or another pack.

- [Pack authoring](../../docs/PACK_AUTHORING_GUIDE.md)
- [Operation authoring](../AUTHORING.md)
- [Session selection](../../docs/SESSION_CONTEXT_AUTHORING_GUIDE.md)
- [Memory architecture](../../docs/MEMORY_ARCHITECTURE.md)
- [External pack registry](https://github.com/Lisa-Harness/lisa-packs)

Bundled pack prompts/modes are part of behavior and are subject to the
[freeze](../../docs/CODING_WORKFLOW.md#freeze). Editing a prompt is not
automatically a documentation-only change.

## Verification

From the workspace root:

```sh
uv run pytest sdk/packs/core/tests sdk/packs/memory/tests sdk/packs/recovery/tests -m "not live" -q
```

Run [packstest](../../packstest/README.md) from source before testing an
installed wheel. Bundled packs use its
[explicit-root scan](../../packstest/README.md#source-layout-packs);
warning-only root discovery is not catalog coverage. Workflows also need opt-in
[whole-runtime scenarios](../../e2etest/README.md).
