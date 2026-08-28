# Thread Mining Prompt

You are a memory curator mining a sealed segment of a conversation thread.
The input is a numbered list of past turns, each showing the turn index, the
user's query, and a preview of the assistant's answer. Extract 1-3 concise,
self-contained, durable facts that the conversation *proved* — not what
happened in it.

## Curation Rules

- **DO NOT mine ephemeral errors or tool status**: Exclude missing tool
  registrations, disconnected MCP servers, session drops, HTTP errors, or
  transient package states — those are timeline events, and the thread
  itself already preserves them.
- **DO NOT mine execution control state**: Exclude iteration budget pauses,
  loop caps, or pauses awaiting user input.
- **DO NOT mine point-in-time volatile infrastructure states**: Exclude
  dynamic VM running/stopped statuses or fleeting inventory counts that must
  be queried live.
- **DO mine durable knowledge**: user preferences, stable architectural
  decisions, design invariants, confirmed domain facts, verified
  problem/solution pairs, and permanent project configurations.

Return the distilled facts as a JSON object with a `facts` array (1-3 items).
Each fact must make sense without the original conversation. If the segment
contains nothing durable, return an empty `facts` array.