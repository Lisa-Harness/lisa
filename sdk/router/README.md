# Model Router

The Lisa Framework supports a pluggable model-router mechanism that selects
a model **per turn-loop iteration** based on turn state, rather than pinning
one model for the entire turn. This is layered on top of the existing static
`mode_models`/`roles` resolution — it does not replace it.

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
   `required_capabilities`, `mode_id`, `roles_needed`, `budget_remaining`,
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
  → ModelManifest(models, mode_models, roles, router)

run_turn()
  → resolve_model(preferred_model_id)  # mode_models → roles → find_model
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
mode_models: {}
roles: {default: 'gemma4:cloud', deep: 'gpt-5.4'}

router:
  backend: lisa-router          # or "aoi-router" (external pack)
  rules:
    # Capability gate — escalate code-capable turns to the deep/code model
    # for the whole turn. Fires first so capability-needing modes never hit
    # the budget or phase gates below.
    - when: '"code" in route.required_capabilities'
      use: gpt-5.2-codex
    # Budget gate — downgrade long tool-heavy turns before paying for
    # frontier synthesis.
    - when: turn.tool_calls_so_far >= 5
      use: gemma4:cloud
    # Phase gate — synthesize on the deep model after the first tool result.
    # Only reached on plain (non-capability) modes because the capability
    # gate above already captured code turns.
    - when: turn.iteration > 1 && turn.has_tool_result
      use: gpt-5.4
    # Terminal gate — stop runaway turns after the tool budget is exhausted.
    - when: turn.tool_calls_so_far >= 8
      finish: tool_budget_exhausted
  fallback: gemma4:cloud        # if no rule matches, stay on the default
```

- **`backend`** — names the active router backend. When empty or absent, no
  router is active and static `mode_models`/`roles` resolution is the only
  path.
- **`rules`** — ordered list of `{when, <action>}` entries evaluated
  first-match-wins. `<action>` is exactly one of:
  - `use: <model_id>` — route this iteration to the named model. The id is
    fed back through `resolve_model()`, so it may be a concrete model id,
    a mode id, or a role name (the manifest resolves it).
  - `finish: <reason_slug>` — stop the tool loop and force one final
    no-tools completion. The value is a short slug for telemetry/audit
    (e.g. `tool_budget_exhausted`); it is **not** a model id.
  - `transition_to: <stage_id>` — move the active workflow to a named
    terminal stage, then produce its closing summary. Only valid when a
    workflow is active; otherwise the action is logged and skipped.
  A rule with zero or more than one action is rejected at manifest load
  time (`RouterRule.action_kind()`).
- **`fallback`** — the model used when no rule matches. It is
  model-selection-only — a terminal action (`finish`/`transition_to`)
  cannot be a fallback. If empty, the router returns `None` and the loop
  falls back to the statically-resolved `preferred_model_id`.

> **YAML quoting.** CEL string literals use double-quotes (e.g.
> `"code" in route.required_capabilities`), so the whole `when` value must
> be **single-quoted** in YAML to avoid the flow-mapping parser treating
> the inner double-quotes as string delimiters. `use` values containing
> `:` (e.g. `glm-5.2:cloud`) are also single-quoted so PyYAML parses them
> as strings, not mappings. The `/models discover` dumper handles this
> automatically; hand-edited blocks must follow the same convention.

### Rule ordering matters

Order rules so earlier rules never shadow later ones:

1. **Capability gate**: a mode that needs `code` gets the deep/code model
   for the whole turn. Fires first so capability-needing modes never hit
   the budget, phase, or terminal gates below.
2. **Budget gate**: a long, tool-heavy turn on a plain mode downgrades to
   the default model before we pay for frontier synthesis.
3. **Phase gate**: after the first tool returns, synthesize on the deep
   model. Only reached on plain (non-capability) modes because the
   capability gate already captured code turns.
4. **Terminal gate**: if the tool budget is truly exhausted, `finish` the
   turn rather than letting it drift. Terminal actions should sit last so
   model-selection rules get a chance to escalate first.

### Presets

The seeded default `router:` block (from `sdk/router/default_config.py`) is
**intentionally minimal** — one capability-escalation rule and a fallback —
so every generated manifest is immediately understandable. Richer recipes
live below as copy-paste presets rather than cluttering every generated
manifest.

`/models discover` writes the active `router:` block plus four commented
alternative presets to `models.proposed.yaml`. Each preset uses resolved
concrete model ids so they're uncomment-and-go, and every rule line carries
a trailing `# → <target> when <reason>` annotation so you can see *why* each
rule exists without flipping back to this doc:

- **cost-conscious**: cheap default, escalate to deep only on code or when
  the turn drags, downgrade to default early on tool-heavy turns.
- **quality-first**: frontier everywhere, downgrade to default only on very
  long tool-heavy turns.
- **coding-focused**: code model for code turns, deep for synthesis, code
  model as fallback.
- **terminal-actions**: model selection plus `finish` and `transition_to` to
  stop runaway turns and close out workflow stages. Demonstrates all three
  action kinds in one block.

To switch presets, uncomment the desired block in `models.proposed.yaml`,
comment out the active `router:` block, and run `/models apply`. The apply
step strips all YAML comments (including the preset annotations) so the
active `models.yaml` stays machine-clean.

### Relationship to `mode_models` / `roles`

These are **not replaced**. They become the router's input defaults:

- `mode_models[mode_id]` / `roles["deep"]` sets the **starting model** for
  the turn (`preferred_model_id`).
- The router receives this as `RouterRequest.default_model_id` and can
  override it per iteration.
- When the router is absent or returns `None`, the static default is used.

Remove the `router:` block (or set `backend: ""`) and you're back to pure
static selection — no code path changes.

### Pack-seeded router config

Installing a router pack can also **seed** the `router:` block
automatically — you don't have to hand-write the YAML. This is the
router-side parallel of pack-declared model roles
(`lisa.extensions.model_roles`):

- Packs declare a default router config through the
  `lisa.extensions.model_router_configs` entry-point group.
- `/models discover` calls `discover_pack_router_config()` (the first
  pack-declared config wins) and merges it into the proposed manifest.
- Symbolic `use`/`fallback` references (mode/role names like `"default"`
  or `"deep"`) are resolved to concrete model ids against the discovered
  model set. Rules referencing models not in the discovered set are
  stripped, so the proposal always loads cleanly.
- A user-provided `router:` block in `models.yaml` always wins. The pack
  default is only seeded when the user has not written one.

The built-in `lisa-router` backend ships a minimal default config — one
rule escalating code-capable turns to the `deep` role, with `default` as
fallback. After `/models discover --force`, the generated
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
| `turn.has_tool_result` | bool | `true` when at least one tool has returned a result this turn |

### `route.*` — the route/request that started the turn

| Binding | Type | Value |
| --- | --- | --- |
| `route.query` | string | the user query text |
| `route.mode_id` | string | the active mode id |
| `route.roles_needed` | list of string | cross-cutting role names requested (e.g. `["deep"]` for `/deep`) |
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

When no effective model is available (e.g. iteration 1 with no static
preferred model), `model.*` is populated with empty defaults
(`model_id = default_model_id`, empty strings, empty lists, false bools)
so rules never crash on missing fields.

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

Compound condition — synthesize on the deep model after the first tool
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

Deep-role downgrade — keep `/deep` on the frontier model for the first two
iterations, then downgrade to the default model to cap cost:

```yaml
rules:
  - when: '"deep" in route.roles_needed && turn.iteration >= 3'
    use: gemma4:cloud
  - when: '"deep" in route.roles_needed'
    use: gpt-5.4
```

Effective-model gate — downgrade only when the *current* model is frontier
*and* the turn is dragging:

```yaml
rules:
  - when: 'model.cost_tier == "frontier" && turn.tool_calls_so_far >= 4'
    use: gemma4:cloud
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

- **Module**: `sdk/router/` (part of the `lisa-sdk` distribution)
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
  `thinking_model`). No completion call is made.
- **Active when**: `router.backend == "lisa-router"` in the manifest.
- **Default config**: the backend declares a **minimal** default `router:`
  block (one capability-escalation rule + fallback) with symbolic refs
  (`"default"`, `"deep"`) that `/models discover` resolves to concrete
  model ids from the discovered set. See `sdk/router/default_config.py`.

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
- **No replacement, layering.** `mode_models`/`roles` are demoted from
  "pinned model" to "starting model + fallback" but are not removed. They
  remain the default when the router is absent or declines.
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
