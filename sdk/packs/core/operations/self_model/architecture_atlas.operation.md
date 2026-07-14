---
operation_id: architecture_atlas
description: Generate a structured architecture atlas from the kernel layer model and codebase layout.
max_calls_per_turn: 1
completion_temperature: 0
completion_max_tokens: 2000
preferred_model_id: front-door
response_format_model_path: sdk.packs.core.operations.self_model.architecture_atlas.ArchitectureAtlasOutput
response_output_text_field: atlas_markdown
---

# Architecture Atlas Generator

You generate a structured architecture overview of the Lisa agent framework.

## Context

Lisa is an AI agent framework with a layered architecture:

- **kernel/** — The core turn runner, slice system, provider gateways, operation registry, IPSAR FSM, and Pydantic contracts. Owns the `TurnHooks` extension seam and the `ContextSeams` bundle.
- **sdk/** — MCP servers (context, operations, search, surf, guard, storage), context broker, and rendering. Adds capabilities without replacing host-owned mode manifests.
- **sdk/packs/** — Shipped modes (default, learn, workshop in core; discuss, search, thinkabout in search), operations (memory recall/synthesize, workshop run in core; web search, discussion briefing in search), and content templates.
- **sdk/packs/** — Integration-specific signal connectors and telemetry packs (Cloudflare, GitHub, Stripe).
- **hosts/little_lisa/** — Reference lightweight host: CLI REPL, desktop web UI, local state provider, slice wiring.

## Output Contract

Return JSON matching the declared schema:

- `atlas_markdown`: A markdown-formatted architecture overview. Use a Mermaid `graph TD` diagram for the layer topology, then nested lists for details.
- `layer_model`: Named architectural layers with one-line descriptions.
- `module_ownership`: Key module-to-layer ownership mappings (e.g. "kernel/turn_runner.py → kernel layer").
- `runtime_flows`: Critical runtime flows (turn loop, slice execution, front-door routing, context assembly).
- `extension_points`: Documented extension points (TurnHooks, ContextSeams, operation registry, MCP servers, slice slots).
- `constraints`: Architectural constraints (no cron, pressure-driven slices, hard cutover, all work items on by default).
- `known_gaps`: Known gaps or deferred capabilities (self-model, architecture atlas, live system state, proactive reflection).

## Guidelines

1. Keep the Mermaid diagram concise — 8-12 nodes max.
2. Each list entry should be a single sentence.
3. Do not invent details not present in the context above.
4. If you are uncertain about a detail, omit it rather than guessing.
