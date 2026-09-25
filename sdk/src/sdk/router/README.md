# Model Router

The Lisa Harness supports a pluggable model-router mechanism that selects
a model **per turn-loop iteration** based on turn state, rather than pinning
one model for the entire turn. This is layered on top of the existing static
`mode_models` resolution — it does not replace it.

> **What's new.** A `ModelRouter` Protocol (`kernel/seams/protocols.py`) and
> `lisa.extensions.model_routers` entry-point group let packs contribute
> router backends. The built-in `lisa-router` backend ships a **CEL-based**
> rules engine (`sdk/router/router.py`, powered by `cel-python`) that makes
> zero extra model calls. External packs (e.g. `lisa-pack-aoi-router` in
> `../lisa-packs/`) can register vendor-backed routers under the same group.
>
> Rules are written in [CEL](https://github.com/google/cel-spec) — boolean
> expressions with `&&`, `||`, `!`, comparisons, and `in` (list/map
> membership). CEL has no side effects, no I/O, and no arbitrary code
> execution, so router rules are safe to evaluate at every iteration without
> a sandbox. Each rule's `when` expression is compiled eagerly at router
> construction time, so an invalid manifest fails at startup rather than
> silently mis-routing turns.

---

## How it works

### Three-tier fallback

The turn loop (`kernel/turn_runner.py:_resolve_iteration_model`) resolves the
model for each iteration with this precedence:

1. **`ContextSeams.model_router`** — the formal `ModelRouter` Protocol with
   a widened `RouterRequest` carrying `query`, `last_tool_result`,
   `required_capabilities`, `mode_id`, `budget_remaining`,
   `default_model_id`, `iteration`, `tool_calls_so_far`, and an
   `effective_model` snapshot of the model used in the preceding
   iteration. This is the pluggable router path.
2. **`ModelPolicy.choose`** — an **opt-in, host-wired** duck-typed method
   with the narrow `(default_model_id, iteration, tool_calls_so_far,
   budget_remaining)` keyword signature. The shipped default
   (`ManifestModelPolicy`) does **not** implement `choose`, so this tier is
   only reached when a host installs a custom `ModelPolicy` that does. The
   turn loop probes for it with `getattr(policy, "choose", None)`.
3. **Static `model_def`** — the once-resolved `preferred_model_id`. This is
   the default behavior when no router and no `choose` are wired.

When no router is wired and the default `ManifestModelPolicy` is in place
(the common case), the loop behaves exactly as before — tier 2 is inert.

### Data flow

```plaintext
models.yaml
  → load_model_manifest()               # kernel/model_manifest.py
  → ModelManifest(models, mode_models, router)

run_turn()
  → resolve_model(preferred_model_id)  # mode_models → find_model
  → per iteration:
      → ModelRouter.choose(RouterRequest) → RouterDecision
          # decision carries exactly one action:
          #   model_id       → resolve_model() → complete()
          #   finish         → strip tools, one final no-tools completion
          #   transition_to  → workflow stage hand-off, then closing summary
      → resolve_model(decision.model_id) → ManifestModelEntry   (for `use`)
      → complete(model_def=...) → ProviderGateway.complete()
```

The router's output is a `model_id` string that flows back through the
existing `resolve_model()` → provider-gateway path. **No new completion code
path.** The router only picks the model; completion still flows through the
existing gateway.

---

## Configuration

Add an optional `router:` block to `models.yaml`. Each rule's `when` is a
CEL expression; each rule carries exactly **one** action — `use`, `finish`,
or `transition_to`:

```yaml
mode_models: {default: 'gpt-5.4'}

router:
  backend: lisa-router          # or "aoi-router" (external pack)
  rules:
    # Capability gate — escalate code-capable turns to the code model
    # for the whole turn. Fires first so capability-needing modes never hit
    # the budget or phase gates below.
    - when: '"code" in route.required_capabilities'
      use: gpt-5.2-codex
    # Budget gate — downgrade long tool-heavy turns before paying for
    # frontier synthesis.
    - when: turn.tool_calls_so_far >= 5
      use: gpt-5.4-mini
    # Phase gate — synthesize on the frontier model after the first tool result.
    # Only reached on plain (non-capability) modes because the capability
    # gate above already captured code turns.
    - when: turn.iteration > 1 && turn.has_tool_result
      use: gpt-5.4
    # Terminal gate — stop runaway turns after the tool budget is exhausted.
    - when: turn.tool_calls_so_far >= 8
      finish: tool_budget_exhausted
```

There is **no `fallback` key**. When no rule matches, the router returns
nothing and the statically-resolved `mode_models` pin is used — that pin is
the required default. To make a specific model the "when nothing matched"
choice, add a trailing catch-all rule:

```yaml
router:
  rules:
    # ...specific rules first...
    - when: 'true'
      use: gpt-5.4-mini   # catch-all: replaces the removed fallback key
```

- **`backend`** — names the active router backend. When empty or absent, no
  router is active and static `mode_models` resolution is the only
  path.
- **`rules`** — ordered list of `{when, <action>}` entries evaluated
  first-match-wins. `<action>` is exactly one of:
  - `use: <model_id>` — route this iteration to the named model. In a
    hand-authored `models.yaml` this must be a **concrete model id**
    (the manifest loader validates `use` against the known model set and
    rejects unknown refs at load time). The turn loop then feeds the id
    back through `resolve_model()`. Symbolic mode-id refs (e.g.
    `"default"`) are only supported in pack-declared default configs and
    are resolved to concrete model ids by `/models discover` *before* the
    YAML is written — they never reach the loader.
  - `finish: <reason_slug>` — stop the tool loop and force one final
    no-tools completion. The value is a short slug for telemetry/audit
    (e.g. `tool_budget_exhausted`); it is **not** a model id.
  - `transition_to: <stage_id>` — move the active workflow to a named
    terminal stage, then produce its closing summary. Only valid when a
    workflow is active; otherwise the action is logged and skipped.
  A rule with zero or more than one action is rejected at manifest load
  time by `_parse_router_config` (`kernel/model_manifest.py`) with
  `"must include one of"` / `"exactly one"`. `RouterRule.action_kind()`
  is the selector the engine uses at `choose()` time to build the
  matching decision — it assumes exactly one action is set.

> **Removed: `fallback`.** The `router.fallback` key was removed (the
> manifest loader rejects it with a migration hint). The statically-resolved
> `mode_models` pin — already required at load time — is the only
> no-rule-match default. Use a trailing `when: 'true'` rule when you want an
> explicit catch-all.

> **YAML quoting.** CEL string literals use double-quotes (e.g.
> `"code" in route.required_capabilities`), so the whole `when` value must
> be **single-quoted** in YAML to avoid the flow-mapping parser treating
> the inner double-quotes as string delimiters. `use` values containing
> `:` (e.g. `glm-5.2:cloud`) are also single-quoted so PyYAML parses them
> as strings, not mappings. The `/models discover` dumper handles this
> automatically; hand-edited blocks must follow the same convention.

### Rule ordering matters

Order rules so earlier rules never shadow later ones:

1. **Capability gate**: a mode that needs `code` gets the code model
   for the whole turn. Fires first so capability-needing modes never hit
   the budget, phase, or terminal gates below.
2. **Budget gate**: a long, tool-heavy turn on a plain mode downgrades to
   the cheap model before we pay for frontier synthesis.
3. **Phase gate**: after the first tool returns, synthesize on the frontier
   model. Only reached on plain (non-capability) modes because the
   capability gate already captured code turns.
4. **Terminal gate**: if the tool budget is truly exhausted, `finish` the
   turn rather than letting it drift. Terminal actions should sit last so
   model-selection rules get a chance to escalate first.

### Presets

The seeded default `router:` block (from `sdk/router/default_config.py`) is
**intentionally minimal** — one capability-escalation rule, nothing else —
so every generated manifest is immediately understandable. Richer recipes
live below as copy-paste presets rather than cluttering every generated
manifest.

`/models discover` writes the active `router:` block plus four commented
alternative presets to `models.proposed.yaml`. Each preset uses resolved
concrete model ids so they're uncomment-and-go, and every rule line carries
a trailing `# → <target> when <reason>` annotation so you can see *why* each
rule exists without flipping back to this doc:

- **cost-conscious**: cheap default, escalate to frontier only on code or
  when the turn drags, downgrade to cheap model early on tool-heavy turns.
- **quality-first**: frontier everywhere, downgrade to cheap model only on
  very long tool-heavy turns.
- **coding-focused**: code model for code turns, frontier for synthesis,
  code model when nothing else matches.
- **terminal-actions**: model selection plus `finish` and `transition_to` to
  stop runaway turns and close out workflow stages. Demonstrates all three
  action kinds in one block.

To switch presets, uncomment the desired block in `models.proposed.yaml`,
comment out the active `router:` block, and run `/models apply`. The apply
step strips all YAML comments (including the preset annotations) so the
active `models.yaml` stays machine-clean.

### Relationship to `mode_models`

This is **not replaced**. It becomes the router's input default:

- `mode_models[mode_id]` sets the **starting model** for
  the turn (`preferred_model_id`).
- The router receives this as `RouterRequest.default_model_id` and can
  override it per iteration.
- When the router is absent or returns `None`, the static default is used.

Remove the `router:` block (or set `backend: ""`) and you're back to pure
static selection — no code path changes.

### Relationship to kernel lifecycle guards

The router's terminal actions (`finish` / `transition_to`) and the kernel's
built-in turn guards (`stage_nudge_threshold` / `stage_nudge_cap`,
`TurnBudget`, `max_tool_iterations`) are complementary layers, not
alternatives:

- **The router is optional deployment config.** It runs only when a backend
  is wired (`router.backend` non-empty) and only evaluates rules the
  deployment authored. The kernel guards are framework invariants that hold
  for every deployment, including no-router hosts and vendor ML backends
  that know nothing about Lisa lifecycle.
- **Counters differ.** CEL rules see per-turn state only
  (`turn.tool_calls_so_far`, `turn.iteration`); `RouterRequest` carries no
  workflow position. The stuck-in-stage guard counts durable, cross-turn
  stage occupancy (`turns_in_stage` in `kernel/turn_phases/_lifecycle.py`),
  which no CEL binding exposes — a model stalled across several turns with
  few tool calls per turn is invisible to any `tool_calls_so_far` rule.
- **Graded vs. terminal.** The guards escalate: a soft nudge at
  `stage_nudge_threshold`, a pause/terminate at `stage_nudge_cap`. A CEL
  `finish` is one hard cliff authored per deployment — it can end a turn,
  not rescue it.
- **Authority layering.** The lifecycle resolver
  (`resolve_lifecycle_action`) ranks a router decision above every kernel
  cap; only an operator pause/stop/terminate signal outranks it. A matching
  `finish` preempts the guards. When no rule matches — or a rule's
  evaluation errors (a non-fatal skip) — the guards fire as usual.

Rule of thumb: author `finish` rules for deployment-specific budget policy;
leave the kernel guards enabled as the framework-wide backstop. They are
defense-in-depth, not duplication.

### Operation-scoped model selection

The router is also consulted when an **operation** executes (not just the
turn loop). This lets you pin a model for a specific operation in
`models.yaml` without editing the pack's `.yml` or `.md` file:

```yaml
router:
  rules:
    - when: 'route.operation_id == "consolidate_semantic"'
      use: gpt-5.2-codex
    - when: 'route.operation_id in ["mine_thread_segment", "synthesize_learning"]'
      use: gpt-5.4-mini
```

 CEL bindings for operation-scoped requests:

- ``route.operation_id`` — the operation's catalog id (e.g.
  ``"consolidate_semantic"``). Empty for turn-loop iterations.
- ``route.mode_id`` — the mode that dispatched the operation.
- ``route.query`` — the operation's query text.

Semantics:

- Only ``use:`` actions apply; ``finish`` and ``transition_to`` are
  turn-loop lifecycle actions and are ignored for operation-scoped
  requests.
- When no rule matches, the operation uses its pack-declared
  ``preferred_model_id`` (or the mode default if the pack omitted it) —
  operation-scoped consultations have no fallback, so turn-loop rules
  never silently change an operation's model.
- The consultation happens once per operation execution, not per
  internal tool-loop iteration inside the operation.

### Pack-seeded router config

Installing a router pack can also **seed** the `router:` block
automatically — you don't have to hand-write the YAML:

- Packs declare a default router config through the
  `lisa.extensions.model_router_configs` entry-point group.
- `/models discover` calls `discover_pack_router_config()` (the first
  pack-declared config wins) and merges it into the proposed manifest.
- Symbolic `use` references (mode names like `"default"`)
  are resolved to concrete model ids against the discovered model set.
  Rules referencing models not in the discovered set are stripped, so the
  proposal always loads cleanly.
- `/models discover` regenerates `models.proposed.yaml` from scratch on
  every run, seeding the pack-declared `router:` block (with symbolic
  refs resolved to concrete model ids). A user who wants to keep their
  hand-edited `router:` block should edit the active `models.yaml`
  directly, or edit the proposal and run `/models apply` without
  re-running `/models discover --force` (which overwrites the proposal,
  including any hand-edited `router:` block, with the pack default).

The built-in `lisa-router` backend ships a minimal default config — one
rule escalating code-capable turns to the `default` model. After
`/models discover --force`, the generated
`models.proposed.yaml` includes that ready-to-use `router:` block with
symbolic refs resolved to concrete model ids — no manual editing required.
Richer recipes (budget gates, phase gates, `finish`, `transition_to`) are
emitted as the four commented presets described above.

---

## CEL bindings (lisa-router)

The built-in `lisa-router` engine exposes the turn state to rule
expressions through three CEL namespaces. Bindings are populated from
`RouterRequest` (`kernel/seams/protocols.py`) and the effective model's
`ManifestModelEntry` metadata.

### `turn.*` — per-iteration turn state

| Binding | Type | Value |
| --- | --- | --- |
| `turn.iteration` | int | 1-based iteration number |
| `turn.tool_calls_so_far` | int | tool calls dispatched so far this turn |
| `turn.budget_remaining` | int | remaining tool-call budget, or `-1` when unlimited/unknown (CEL has no nullable int; `-1` is the sentinel) |
| `turn.last_tool_result` | string | last tool result text, or `""` when none |
| `turn.has_tool_result` | bool | `true` when at least one tool has returned a result this turn (turn-level state, distinct from `turn.last_tool_result`, which is only the *most recent* result text and may be empty even when earlier tools returned results) |

### `route.*` — the route/request that started the turn

| Binding | Type | Value |
| --- | --- | --- |
| `route.query` | string | the user query text |
| `route.mode_id` | string | the active mode id |
| `route.required_capabilities` | list of string | capability names the current mode's tool surface requires (e.g. `["code", "text"]`) |

### `model.*` — metadata of the effective model

The **effective model** is the model actually selected for the preceding
iteration (or the statically-resolved preferred model on iteration 1).
This lets rules inspect what ran last without coupling to the manifest:

| Binding | Type | Value |
| --- | --- | --- |
| `model.model_id` | string | the effective model's id |
| `model.provider` | string | provider slug (e.g. `azure_openai`) |
| `model.cost_tier` | string | `standard` / `frontier` / … |
| `model.latency_tier` | string | `low` / `medium` / `high` |
| `model.capabilities` | list of string | capability names advertised by the model |
| `model.reasoning_model` | bool | `true` if the model is a reasoning model |
| `model.thinking_model` | bool | `true` if the model is a thinking model |
| `model.structured_reliability` | double | confidence (0.0–1.0) the effective model reliably honors structured output; 0.0 = untrusted (e.g. small local Ollama models), 1.0 = provider-enforced (API models) |
| `model.needs_guidance` | bool | `true` when the user declared this model needs stronger kernel guidance (forced-action backstops, see [Model attributes](#model-attributes)) |

When no effective model is available (e.g. iteration 1 with no static
preferred model), `model.*` is populated with empty defaults
(`model_id = default_model_id`, empty strings, empty lists, false bools)
so rules never crash on missing fields.

### Model attributes

Per-model metadata lives under the `models:` list in `models.yaml`. Each
entry is a `ManifestModelEntry` (`kernel/model_manifest.py`). The
attributes the router and turn loop read:

| Attribute | Type | Default | Purpose |
| --- | --- | --- | --- |
| `model_id` | string | required | unique model identifier (e.g. `gpt-5.4`) |
| `provider` | string | required | provider slug (e.g. `azure_openai`, `ollama`) |
| `capabilities` | list of string | `[]` | capability names advertised by the model (e.g. `[reasoning, tools, text]`) |
| `cost_tier` | string | `standard` | cost band: `standard` / `premium` / `frontier` |
| `latency_tier` | string | `medium` | latency band: `low` / `medium` / `high` |
| `context_window` | int | `0` | max context tokens; `0` = unknown/unlimited |
| `reasoning_model` | bool | `false` | the model is a reasoning model (affects trust-aware enforcer cap) |
| `thinking_model` | bool | `false` | the model is a thinking model |
| `structured_reliability` | float | `0.0` | confidence (0.0–1.0) the model honors `response_format` structured output; 0.0 = untrusted, 1.0 = provider-enforced |
| `needs_guidance` | bool | `false` | the model needs stronger kernel guidance backstops. **Off by default** so strong models stay unhinted; set `needs_guidance: true` per model to opt a weak or finicky model into the forced-action backstops (see below). |

#### `needs_guidance` — forced-action backstops

A model with `needs_guidance: true` receives two kernel-side backstops that
strong models do not:

1. **Do-the-work backstop (self-handoff)** — when the routing hint and the
   plan-driven hint both correctly go silent (the plan's next step is the
   current mode), a guidance-needing model is left with no directive and
   may return an empty completion, ending the turn blocked as `no_tools`
   (hallucinated success). The backstop arms a mode action tool and the
   empty-completion enforcer synthesizes a call to it so the turn does real
   work instead of claiming it. One-shot: cleared as soon as any tool
   dispatches, so it cannot loop.

2. **Trust-aware enforcer cap** — the forced-action enforcer's
   `non_compliant_cap` is bypassed for frontier, reasoning, or
   high-reliability models. A `needs_guidance` model is *not* bypassed —
   it receives the full cap treatment so the kernel takes over when it
   refuses to call the expected tool.

These backstops do not change routing, model selection, or the tool
surface — they only change when the kernel synthesizes a tool call itself
vs. accepting the model's empty prose. The `needs_guidance` flag is also
exposed to router rules via `model.needs_guidance` so a custom backend can
factor it into routing decisions.

Example `models.yaml` entry:

```yaml
- model_id: deepseek-v4-flash:preview-cloud
  provider: ollama
  needs_guidance: true
  capabilities: [reasoning, structured_output, tools, text]
  structured_reliability: 0.85
```

### Operator reference

CEL supports the standard boolean and comparison operators:

- `&&`, `||`, `!` — boolean and / or / not
- `==`, `!=`, `<`, `<=`, `>`, `>=` — comparisons
- `in` — list/map membership (e.g. `"code" in route.required_capabilities`)
- `&&` / `||` short-circuit, so you can guard field access:
  `model.reasoning_model && turn.iteration >= 3`.

### Examples

Capability gate — escalate code-capable turns to the code model:

```yaml
rules:
  - when: '"code" in route.required_capabilities'
    use: gpt-5.2-codex
```

Compound condition — synthesize on the frontier model after the first tool
result on iteration 2+:

```yaml
rules:
  - when: turn.iteration > 1 && turn.has_tool_result
    use: gpt-5.4
```

Compound `||` — code capability *or* a long turn:

```yaml
rules:
  - when: '"code" in route.required_capabilities || turn.iteration > 5'
    use: gpt-5.2-codex
```

Effective-model gate — downgrade only when the *current* model is frontier
*and* the turn is dragging:

```yaml
rules:
  - when: 'model.cost_tier == "frontier" && turn.tool_calls_so_far >= 4'
    use: gemma4:cloud
```

Structured-output gate — when the current model is not reliable at honoring
`response_format` (e.g. a small local Ollama model), escalate to one that is,
so JSON-bound entry operations (like `web_search`) get a schema-compliant
reply instead of prose:

```yaml
rules:
  - when: 'model.structured_reliability < 0.8'
    use: deepseek-v4-pro:cloud
```

Effective-model capability gate — escalate when the current model lacks
vision:

```yaml
rules:
  - when: '"vision" in model.capabilities'
    use: gpt-5.4
```

Terminal action — stop the tool loop once the budget is exhausted:

```yaml
rules:
  - when: turn.tool_calls_so_far >= 8
    finish: tool_budget_exhausted
```

Terminal action — hand off to a workflow's `summary` stage after too many
iterations:

```yaml
rules:
  - when: turn.iteration >= 10
    transition_to: summary
```

### Compilation & evaluation semantics

- **Eager compilation.** Every `when` expression is compiled to a CEL
  program at `LisaRouter` construction time. An invalid expression raises
  `ValueError("failed to compile CEL expression ...")` immediately, so an
  invalid manifest fails at startup rather than silently mis-routing turns.
- **Runtime errors are non-fatal.** A CEL evaluation error at `choose()`
  time (e.g. referencing a missing field that survives compilation) is
  logged at `WARNING` and the rule is treated as a non-match; evaluation
  continues with the next rule. A broken rule never breaks the turn.
- **First-match-wins.** Rules are evaluated in declared order; the first
  matching rule's action is returned. Order your rules so earlier rules
  don't shadow later ones.
- **Manifest validation.** A `RouterRule` with zero or more than one action
  (`use` / `finish` / `transition_to`) is rejected by
  `load_model_manifest()` with `"must include one of"`.

Complex routing needs that CEL can't express cleanly should use a vendor
backend (e.g. `lisa-pack-aoi-router`).

---

## Pluggable backends

### Built-in: `lisa-router`

- **Module**: `sdk/router/` (part of the `lisa-harness-sdk` distribution)
- **Dependency**: `cel-python>=0.5.0` (declared in `sdk/pyproject.toml`)
- **Entry points** (declared in `sdk/pyproject.toml`):
  - `lisa.extensions.model_routers` → `sdk.router:get_router_factory`
    (runtime factory — the `ModelRouter` instance)
  - `lisa.extensions.model_router_configs` →
    `sdk.router.default_config:get_default_router_config`
    (static config — the default `router:` block for `/models discover`)
- **Zero model calls**: the engine evaluates CEL rules locally against
  `RouterRequest` fields and the effective model's `ManifestModelEntry`
  metadata (`cost_tier`, `latency_tier`, `capabilities`, `reasoning_model`,
  `thinking_model`, `needs_guidance`). No completion call is made.
- **Active when**: `router.backend == "lisa-router"` in the manifest.
- **Default config**: the backend declares a **minimal** default `router:`
  block (one capability-escalation rule) with symbolic refs
  (`"default"`) that `/models discover` resolves to concrete model ids
  from the discovered set. See `sdk/router/default_config.py`.

> The `lisa-router` engine is **not a pack** — it ships no modes,
> operations, or authored content. It is a runtime `ModelRouter` backend
> bundled with the SDK, kin to `kernel.model_router_discovery` (which owns
> the discovery contract). External router backends (e.g. `lisa-pack-aoi-router`)
> still register under the same `lisa.extensions.model_routers` group from
> a real pack.

### External: `lisa-pack-aoi-router` (deferred)

- **Pack**: `../lisa-packs-wip/aoi-router/` (independent repo)
- **Entry point**: same `lisa.extensions.model_routers` group
- **Model call**: the vendor router is itself a completion call (using
  `kernel.provider_core.AzureOpenAIGateway` with `response_format` for
  structured output), which returns a `model_id`. This is the latency cost
  you accept for ML-based routing.
- **Active when**: `router.backend == "aoi-router"` in the manifest.

### Writing a custom router

1. Create a pack with a `pyproject.toml` entry point:

   ```toml
   [project.entry-points."lisa.extensions.model_routers"]
   my-router = "my_pack.router:get_router_factory"

   [project.entry-points."lisa.extensions.model_router_configs"]
   my-router = "my_pack.router.default_config:get_default_router_config"
   ```

2. Implement the factory. The factory receives `(manifest, session_plan)`;
   return `None` when your backend is not active so the host skips it and
   other factories get a chance. The first non-`None` factory result wins —
   ordering is deterministic by entry-point discovery order.

   ```python
   from kernel.model_manifest import ModelManifest
   from kernel.seams.protocols import (
       ModelRouter,
       RouterDecision,
       RouterRequest,
   )


   def get_router_factory():
       return _factory


   def _factory(manifest: ModelManifest | None, session_plan) -> ModelRouter | None:
       if manifest is None or manifest.router is None:
           return None
       if manifest.router.backend != "my-router":
           return None
       return MyRouter(manifest)


   class MyRouter(ModelRouter):
       def choose(self, req: RouterRequest) -> RouterDecision | None:
           # ... your routing logic ...
           # Return one of:
           #   RouterDecision(model_id="...",      reason="...", routed_by="my-router")
           #   RouterDecision(finish="reason_slug", reason="...", routed_by="my-router")
           #   RouterDecision(transition_to="stage", reason="...", routed_by="my-router")
           # or None to fall back to the statically-resolved preferred_model_id.
           return RouterDecision(model_id="...", reason="...", routed_by="my-router")
   ```

---

## Telemetry

Each `RouterDecision` carries exactly one action plus audit fields:

- `model_id` — the chosen model (when the action is `use`)
- `finish` — the reason slug (when the action is `finish`)
- `transition_to` — the target stage id (when the action is `transition_to`)
- `reason` — human-readable explanation, including the matched rule text
  and the iteration, e.g.
  `"lisa-router: rule 'turn.iteration > 1 && turn.has_tool_result' matched at iteration 2"`.
  For terminal actions the reason is suffixed with
  `(finish: <slug>)` or `(transition_to: <stage>)`.
- `routed_by` — which router made the decision (e.g. `"lisa-router"`)

The turn loop logs each terminal action at `INFO` (`router finish action
...; forcing final` / `router transition_to action ...`) and emits
`model_call` stage events carrying the chosen `model_id` and `provider`.
A `transition_to` that fails (no active workflow, or the target stage is
unknown) is logged at `WARNING` and the turn continues with model
selection — a broken transition never breaks the turn. Router decisions
are auditable via the `reason` and `routed_by` fields.

---

## Design principles

- **Zero extra model calls (rules-based).** The `lisa-router` engine is
  pure-Python local CEL evaluation — no model call. This preserves the
  "zero extra model calls" principle from `docs/ROUTER_DESIGN.md`. Vendor
  routers (e.g. `aoi-router`) are opt-in and accept the latency cost.
- **Safe by construction.** CEL has no side effects, no I/O, and no
  arbitrary code execution, so router rules are safe to evaluate at every
  iteration without a sandbox. Expressions are compiled eagerly so an
  invalid manifest fails at startup.
- **Three actions, one per rule.** `use` selects a model; `finish` stops
  the tool loop; `transition_to` hands off to a workflow stage. Exactly
  one is set per `RouterDecision`/`RouterRule`, enforced at manifest load.
- **No replacement, layering.** `mode_models` is demoted from
  "pinned model" to "starting model + no-rule-match default" but is not
  removed. It remains the default when the router is absent or declines,
  and it is the *only* default — there is no `fallback` key.
- **First-non-None wins.** Router discovery uses the same first-non-None
  pattern as `lisa.extensions.slice_providers`, keeping ordering deterministic.
- **Graceful failure.** Router exceptions are caught and logged; the loop
  falls back to static resolution. A CEL evaluation error skips the rule;
  a broken `transition_to` is logged and skipped; a broken router never
  breaks the turn.

### Capability gate (router safety net)

When a mode declares `required_capabilities` in its `_mode.md` front matter
(pack-authored `ModelCapability` strings like `tools`, `code`), the turn
runner validates the resolved model's `ManifestModelEntry.capabilities`
**before** the tool loop starts. When a router is wired and the model is
missing a required capability, the turn returns a clear user-facing block
instead of letting a weak model reach for write-effect tools it cannot use
reliably — e.g. `gemma4:cloud` in `coding` mode calling
`coding_create_worktree` for a "write a roadmap" request it can't complete.

The gate is the safety net **behind** the router: a router rule like
`"code" in route.required_capabilities → gpt-5.2-codex` should escalate to a
capable model first; the gate catches the cases where the router declined or
returned `None`. When no router is wired the gate is advisory (no block) — a
host without a router has no escalation path, so blocking would dead-end.

A mode with no `required_capabilities` (the default) never blocks — every
model is admitted, preserving the pre-field behavior. See
`../README.md` for authoring.
