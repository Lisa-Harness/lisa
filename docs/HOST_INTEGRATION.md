# Host integration

A host is an application that owns transport, authentication, deployment,
and the runtime's async lifecycle. Most hosts should compose
`lisa-harness-runtime`; use kernel seams directly only when deliberately
replacing that composition layer.

See [architecture](BOUNDARIES.md) for ownership and [runtime](../runtime/README.md)
for the module map.

## Use the composition factory

`HostRuntime(config)` is a state object, not the complete bootstrap sequence.
[build_runtime](../runtime/src/runtime/bootstrap.py) resolves configuration,
compiles the session, discovers content and providers, wires services, and
sets MCP/proxy configurations. It does **not** enter async resources.

This headless example accepts an authored session and a configured model
manifest; it is a live integration, not an offline smoke test:

```python
import asyncio

from runtime import NullTransport, handle_message
from runtime.bootstrap import build_runtime


async def main() -> None:
    runtime = build_runtime(
        host_kind="headless",
        data_dir="./agent-data",
        user_slug="local-user",
        session_context="./session_context.md",
    )
    transport = NullTransport()
    try:
        await runtime.start()
        response = await handle_message(runtime, "/help", transport)
        print(response)
    finally:
        await runtime.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

`/help` is a control command, not a model request. Substitute conversational
text only after configuring provider access and a valid roster.
`NullTransport` discards callback events, including non-fatal errors; production
hosts should implement an observable transport rather than rely on it.

## Async ownership

Start resources, run turns, and close resources **in the same task and event
loop**. MCP clients use task-bound async contexts. Do not start the pool in
one `asyncio.run` and use or close it in another, or close it from an unrelated
shielded task.

Serialize messages for a runtime/session. Do not treat one mutable
`HostRuntime` as an arbitrary concurrent request server. Framework discovery
and some registries are process-global; distinct runtime objects alone do
not establish strong multi-tenant isolation.

Startup can request a host restart after installing required packs. Honor
that lifecycle instead of trying to refresh partially initialized imports.

## Transport callbacks

Implement [TransportCallbacks](../runtime/src/runtime/callbacks.py) or wrap a
transport with `DelegatingTransport`.

| Callback | Host responsibility |
| --- | --- |
| `on_stage` | Render structured activity and lifecycle updates |
| `on_text_chunk` | Handle streaming deltas; keep reasoning distinct from visible output |
| `on_response` | Deliver the final response |
| `on_debug_context` | Show diagnostics only to authorized recipients |
| `on_error` | Surface non-fatal errors explicitly |

Callbacks are awaited; a slow transport slows the turn. Fatal startup errors
still raise. Do not derive successful completion solely from displayed prose:
the runtime also records blocks, yields, pauses, approvals, and receipts.

## Integration checklist

- **Identity:** derive workspace, user, surface, conversation, and principal
  from authenticated host context, not model-supplied text. The single-user
  defaults are not suitable for a shared service.
- **Configuration:** resolve secrets at the host boundary. Compile local
  sessions and model manifests before serving turns.
- **Tools:** activate only reviewed packs and MCP servers. Keep domain
  behavior in packs/SDK; do not let transports bypass catalog authorization.
- **Approvals:** expose the exact request and preserve its identity.
  `/approve <id>` resumes only that saved invocation; policy-approved model-loop
  actions continue in the kernel. See [policy](POLICY_GUIDE.md).
- **Persistence:** preserve thread/result objects, approvals, recovery, and
  work-product state. They represent different contracts, not one checkpoint.
- **Privacy:** enforce memory filters, scope isolation, and delivery routing.
  Default input/output guards pass through; install an appropriate guard
  implementation where your application needs one.
- **Scheduling:** opportunistic housekeeping uses bounded slices. If ambient
  intake needs wall-clock polling, a host-owned scheduler runs the one-shot
  intake process; do not add timers inside the turn loop.
- **Shutdown:** stop accepting messages, settle/cancel owned work, and close
  resources through the original task.
- **Testing:** use [runtime test factories](../runtime/src/runtime/testing/runtime_factory.py)
  and injected fakes before a live integration. Test scope and approval
  failures as well as the happy path.

## Lower-level kernel integration

The kernel accepts [TurnHooks](../kernel/src/kernel/turn_hooks.py) and
[ContextSeams](../kernel/src/kernel/turn_runner/_seams.py). A custom
composition must supply model resolution/completion, content/catalog access,
authorized dispatch, state and identity, persistence callbacks, and lifecycle
services. See the [kernel reference](../kernel/README.md#turnhooks-and-contextseams) for the
individual seams and defaults. Reuse the runtime when you do not need to
replace these mechanisms.
