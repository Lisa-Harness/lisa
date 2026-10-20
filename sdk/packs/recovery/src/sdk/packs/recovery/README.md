# Recovery Pack

Operator-facing slash commands for inspecting and resolving interrupted dispatch work items recorded by the kernel's recovery service.

This optional pack must be active to register its commands. See
[bundled packs](../../../../../README.md),
[troubleshooting](../../../../../../../docs/TROUBLESHOOTING.md), and
[context lifecycle](../../../../../../../docs/CONTEXT_LIFECYCLE.md).

## What this pack does

The kernel records write-ahead lifecycle events for every dispatched operation, pack tool, and MCP tool in the turn runner (`kernel/turn_runner.py`). When a dispatch fails, times out, or is interrupted by an approval requirement, the work item is persisted in a `stalled` or `indeterminate` state. When the host process restarts after a crash, orphaned `running` items are promoted to `indeterminate`.

This pack provides the operator UX for inspecting and resolving those records:

| Command | Purpose |
| --- | --- |
| `/stalls` | List all open (stalled, indeterminate, running) work items for this session. |
| `/stall <id>` | Show full details for one work item. |
| `/retry <id>` | Retry one stalled item. Safe (read-only) items settle immediately; effectful or indeterminate items require explicit approval. |
| `/retry-safe` | Retry all safe (read-only) stalled items in one command. |
| `/skip <id>` | Mark one item as skipped, unblocking dependent workflow steps. |
| `/cancel <id>` | Mark one item as cancelled, unblocking dependent workflow steps. |
| `/skip-all` | Mark every open (stalled, indeterminate) item as skipped in one command. Running items are left untouched. |
| `/cancel-all` | Mark every open (stalled, indeterminate) item as cancelled in one command. Running items are left untouched. |
| `/resume [reply]` | Resume from the most recent yielded turn. |

Slash prefixes are matched by the host with leading-prefix matching in
registration order, so every bulk form (`/retry-safe`, `/skip-all`,
`/cancel-all`) is registered before its shorter shadow (`/retry`, `/skip`,
`/cancel`).

## Architecture boundary

| Layer | Owns | Does not own |
| --- | --- | --- |
| `kernel/recovery.py` | Recovery contracts (`RecoveryRecord`, `RecoveryIdentity`, `RecoveryService` protocol), state/retry/scope enums | Persistence, operator UX |
| `kernel/turn_runner.py` | Write-ahead lifecycle instrumentation, startup reconciliation | Replay execution |
| `runtime/recovery.py` | File-backed durable store (`FileRecoveryService`) | Command vocabulary, rendering |
| `sdk/packs/recovery/` (this pack) | Operator commands, rendering, retry gating | Durable state, kernel contracts |

## Retry safety

- **Safe** (`retry_policy: "safe"`): read-only items. `/retry` settles them as succeeded.
- **Confirm** (`retry_policy: "confirm"`): effectful or indeterminate items. `/retry` surfaces a prompt and does NOT auto-execute — the operator must explicitly approve replay.
- **Never** (`retry_policy: "never"`): items that cannot be retried. `/retry` rejects them; use `/skip` or `/cancel`.

Indeterminate items (recovered after a crash while `running`) are always `confirm` — the effect may or may not have happened, so silent replay risks duplication.

## Relationship to the thread store

The thread captures conversation and workflow evidence; recovery records
capture individual dispatch outcomes or uncertainty after a crash.
Other durable contracts include approvals, continuation objects, memories,
and work products. `/resume` of a yielded turn is not the exact-action
approval continuation mechanism. The command scans the thread for the most
recent yielded turn and surfaces its question, optionally alongside stalled
recovery records.
