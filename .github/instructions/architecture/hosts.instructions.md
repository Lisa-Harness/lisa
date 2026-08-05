---
name: Hosts Architecture
applyTo: "hosts/**"
description: Architecture boundaries for the hosts package.
---

# Hosts architecture

- Hosts own runtime integration, process boundaries, and environment-specific wiring. Keep domain logic in kernel or SDK modules.
- `host` is transport-neutral: it owns typed configuration, runtime state, callbacks, provider dispatch, MCP compilation, content roots, thread state, slices, and command dispatch.
- Do not import transport UI packages or call rendering and delivery APIs from `host`; transports implement `TransportCallbacks` at their edge.
- Keep configuration resolution pure and move a new responsibility out of `handle_message` before that dispatcher becomes a second runtime.

## Runtime invariants

- Start and close MCP resources in the same task and event loop that runs turns. Do not enter async resources during synchronous initialization or close them from a shielded or unrelated task.
- Housekeeping is slice-budgeted. Do not add `cron`, `sched`, `threading.Timer`, or other wall-clock background schedulers.
- Test through `HostConfig` and `HostRuntime` with injected fakes rather than monkeypatching module-level core interfaces.
- A test `HostRuntime` needs a seeded model manifest with `mode_models`, not only a `models` list.

## Context, telemetry, and capability rules

- Allocate `recent_conversation` before the lossy prior-conversation digest; budget allocation order is a correctness property.
- Update the current stage from a transition before applying turn usage. Use per-mode usage from nested handoffs when present instead of collapsing it into the entry mode.
- Configure MCP-native default tools through `default_available_tools` and include those resolved IDs in mode-catalog cache keys.
- Resolve the packs-venv site-packages path from `pyvenv.cfg`, then guarded directory fallbacks; a missing first-launch venv must not make startup fail.
- Deduplicate packs installed in both the main and packs virtual environments by normalized distribution name before building the catalog.
