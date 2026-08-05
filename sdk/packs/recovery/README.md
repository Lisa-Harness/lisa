# Recovery Pack

Operator-facing slash commands for inspecting and resolving interrupted dispatch work items recorded by the kernel's recovery service.

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

## Architecture boundary

| Layer | Owns | Does not own |
| --- | --- | --- |
| `kernel/recovery.py` | Recovery contracts (`RecoveryRecord`, `RecoveryIdentity`, `RecoveryService` protocol), state/retry/scope enums | Persistence, operator UX |
| `kernel/turn_runner.py` | Write-ahead lifecycle instrumentation, startup reconciliation | Replay execution |
| `host/recovery.py` | File-backed durable store (`FileRecoveryService`) | Command vocabulary, rendering |
| `sdk/packs/recovery/` (this pack) | Operator commands, rendering, retry gating | Durable state, kernel contracts |

## Retry safety

- **Safe** (`retry_policy: "safe"`): read-only items. `/retry` settles them as succeeded.
- **Confirm** (`retry_policy: "confirm"`): effectful or indeterminate items. `/retry` surfaces a prompt and does NOT auto-execute — the operator must explicitly approve replay.
- **Never** (`retry_policy: "never"`): items that cannot be retried. `/retry` rejects them; use `/skip` or `/cancel`.

Indeterminate items (recovered after a crash while `running`) are always `confirm` — the effect may or may not have happened, so silent replay risks duplication.

## Relationship to compaction

Recovery is orthogonal to compaction. Compaction manages what conversation material reaches the model; recovery manages executable truth and operator decisions. Recovery records live outside turn history and are never summarized away by compaction.

## Relationship to checkpointing

Checkpoints capture stable workflow boundaries. Recovery records capture in-flight, failed, and indeterminate work since the last boundary. On restart, the host restores stable state first, then reconciles outstanding recovery items — never auto-replaying indeterminate writes.