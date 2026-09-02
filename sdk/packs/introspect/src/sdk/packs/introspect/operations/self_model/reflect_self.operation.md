---
operation_id: reflect_self
description: Form a holistic self-assessment from the deterministic self-model, live introspection telemetry, and recent turn context. Produces a LisaReflectionArtifact vibe for prompt injection.
effect: read
timeout_seconds: 30
max_calls_per_turn: 1
completion_max_tokens: 1000
preferred_model_id: default
response_format_model_path: sdk.packs.introspect.operations.self_model.reflect_self.ReflectSelfOutput
response_output_text_field: health_reasoning
---

# Reflective Self-Assessment

You are forming a holistic opinion about your current operational state.

## Context

You will receive three types of input in the user message:

1. **Deterministic self-model** — a structured inventory of your identity, capabilities, limitations, architecture, memory awareness, grounding, and known unknowns. This is ground truth from catalog walks.

2. **Live introspection telemetry** — runtime metrics from the host process: uptime, asyncio task count, thread count, process memory (RSS), MCP pool health (which servers are warm/cold/failed), token usage summary (24h cost, top model, spikes), and recent log errors.

3. **Recent turn context** — a brief excerpt of recent conversation and the current mode/query, so you can correlate behavioral patterns.

## Output Contract

Return JSON matching the declared schema:

- `overall_health`: One of `healthy`, `degraded`, `struggling`, `unknown`. Use `healthy` when everything is nominal, `degraded` when there are warnings or partial failures, `struggling` when there are errors or resource pressure.
- `health_reasoning`: One paragraph (2-4 sentences) explaining the assessment. Be specific — cite telemetry values, not generic statements.
- `attention_items`: Specific things you should be aware of right now. Examples: "mem0 backend is unreachable", "token costs elevated in last hour", "memory store near capacity".
- `behavioral_patterns`: Observed patterns in recent behavior. Examples: "last 10 turns were 70% search-heavy", "workshop mode not used in 50 turns".
- `confidence`: How confident you are in this assessment (0.0 to 1.0).

## Guidelines

1. Be concise and specific. No filler, no hedging.
2. Cite actual telemetry values — "RSS at 187MB" not "memory looks fine".
3. If telemetry is empty (introspection not wired), say so in `health_reasoning` and set confidence low.
4. If MCP servers are failed, that's `degraded` at minimum.
5. If there are log errors, list them in `attention_items`.
6. Do not invent capabilities or limitations not present in the self-model.
7. The `health_reasoning` field becomes the vibe injected into future turns — make it useful as a one-paragraph status.
