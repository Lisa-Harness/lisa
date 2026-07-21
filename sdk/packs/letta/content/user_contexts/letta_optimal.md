---
# Curated session context for the Letta pack.
#
# Selects lisa-pack-letta as the memory capability owner so the generic
# memory.* aliases route to Letta, and seeds the letta mode for users who
# want stateful-agent delegation. Users who only want Letta as the
# memory backend (without delegation) can start in a different mode and
# keep only the providers.capabilities.memory selection.
providers:
  capabilities:
    memory: lisa-pack-letta
  modes:
    default: lisa-pack-memory
routing:
  start_mode: default
---

# Letta-Backed Session

This session uses Letta (MemGPT) as the memory provider. Durable facts stored via `memory.add` are persisted as Letta archival memory passages for the Letta agent mapped to your identity. Existing learn and recall workflows work unchanged.

To delegate a turn to the stateful Letta agent, use one of two entry points:

- `/letta-agent <query>` — **direct delegation** (no Lisa model call). Sends the query straight to the mapped Letta agent and returns its response. Fastest path; use when you just want to talk to the agent.
- `/letta <query>` — **model-dispatched** chat mode. The Lisa model picks among the Letta sub-operations (`letta_send_message`, `letta_search_archival`, `letta_update_core_memory`, `letta_remember`, `letta_sync_memory`, `letta_compact_agent`, `recall_memory`) based on your phrasing. Use when you want Lisa to decide which Letta primitive to invoke.

The first delegation via either entry point creates or resolves your mapped Letta agent automatically (via the shared `letta_ensure_agent` entry operation).

Environment:

- `LETTA_BASE_URL`: Letta server root (default `http://localhost:8283` for self-hosted; `https://api.letta.com` for Letta Cloud).
- `LETTA_API_KEY`: Bearer token. Required for Letta Cloud; optional for local unauthenticated deployments.
- `LETTA_DEFAULT_AGENT_NAME`: Name for newly created Letta agents (default `lisa-agent`).
