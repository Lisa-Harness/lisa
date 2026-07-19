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
routing:
  start_mode: default
---

# Letta-Backed Session

This session uses Letta (MemGPT) as the memory provider. Durable facts stored via `memory.add` are persisted as Letta archival memory passages for the Letta agent mapped to your identity. Existing learn and recall workflows work unchanged.

To delegate a turn to the stateful Letta agent, switch to the `letta` mode with `/letta <query>` or `/ask-letta <query>`. The first delegation creates or resolves your mapped Letta agent automatically.

Environment:

- `LETTA_BASE_URL`: Letta server root (default `http://localhost:8283` for self-hosted; `https://api.letta.com` for Letta Cloud).
- `LETTA_API_KEY`: Bearer token. Required for Letta Cloud; optional for local unauthenticated deployments.
- `LETTA_DEFAULT_AGENT_NAME`: Name for newly created Letta agents (default `lisa-agent`).
