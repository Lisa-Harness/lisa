# SPEC: Browser capability (`lisa-pack-browser`)

> **Status:** SPEC ONLY — nothing here is implemented. The pack lands as a
> submodule of `lisa-packs` (`lisa-packs/browser`, repo
> `Lisa-Harness/lisa-pack-browser`); the two framework items in §5 land
> here. Updated 2026-11-09.

## 1. Goal and non-goals

Give Lisa a real browser for what the
[fetcher](../../sdk/src/sdk/content/_fetcher.py) cannot do: JavaScript-rendered
pages, multi-step interaction (logins, forms, pagination), verifying web UIs
the coding pack builds, and screenshots as evidence.

Non-goals:

- Replacing `fetch_page` for static reads. It stays the cheap default.
- A kernel browser seam. Browsing is a pack capability over MCP; the stale
  seam mention and the dead `mode_kind: browser` check are removed.
- Vision-first or coordinate-click browsing. Vision is an opt-in fallback.

## 2. Shape: thin pack over `@playwright/mcp`

The pack declares Microsoft's
[Playwright MCP](https://github.com/microsoft/playwright-mcp) server in
`mcp_servers.yaml` (`runtime_kind: npx`, headless, `--snapshot-mode none`,
`--output-dir` under the pack's data subdir) and **does not expose the raw
server tools to the model**. The model sees pack tools (`*.tool.yml` + thin
handlers) that call the server through the host tool-execution seam, the
pattern `lisa-pack-cloudservers` already uses.

Wrapping is required, not stylistic:

- **Approval.** MCP-native tools carry no `effect` / `approval_required`;
  pack tools do, so the write tier is gated by the existing policy engine
  (`config/policy.yaml`, CEL `--when` rules for per-origin auto-approval).
- **Classification.** Playwright's `readOnlyHint` is too strict to define the
  read tier: `browser_navigate`, `browser_navigate_back`, `browser_tabs`,
  `browser_wait_for`, and `browser_press_key` are all `readOnly: false`.
- **Context.** Observation shaping (§4) lives in the handlers.

Rejected alternative: `playwright-cli` + skills through `bash_run`. Microsoft
recommends it for coding agents (fewer tokens), and it would reuse bwrap
unchanged, but each `bash_run` sandbox is per-call while the browser must
outlive calls, and every action would collapse into one `execute` approval
with no per-action effect. Revisit only for the coding pack's "run the
Playwright tests" path, which already works through `bash_run`.

## 3. Modes and tool tiers

Mirrors the coding pack's read/write split
(`coding_review` / `coding_execute`).

| Mode | Routable | Tools | Default profile |
| --- | --- | --- | --- |
| `browse` | yes | read tier | ephemeral |
| `browse_act` | yes, narrow intents | read + act tiers | any (§6) |

| Tier | Pack tools (wrapping) | `effect` | Approval |
| --- | --- | --- | --- |
| Read | `browse_open` (navigate, back, tabs), `browse_snapshot` (scoped `target`/`depth`), `browse_find`, `browse_screenshot`, `browse_extract`, console/network reads | `read` | no |
| Act | `browse_act` (batched click / type / fill_form / select / press / hover / drag / dialog), `browse_upload` | `external_write` | yes |
| Excluded | `browser_evaluate`, `browser_run_code_unsafe`, cookie/storage setters, `route`, `--caps=devtools` | — | not exposed |

Read is not side-effect free: a GET can exfiltrate data in its URL, and
"safe" links can mutate state. The read tier is therefore bound to the
**ephemeral profile** — no cookies, nothing authenticated to leak. Only
`browse_act`, behind approval, may attach a persistent or committed profile.
That rule, not the tool list, is what makes `browse` effectively read-only.

## 4. Context management

Research summary (sources in §9):

- Accessibility-tree snapshots are ~90% smaller than DOM and are the right
  default; complex apps still reach 50k–114k tokens per snapshot.
- Accumulating snapshots is the dominant cost and failure driver: one
  production study measured >43k tokens after 15 actions with full history
  vs. a flat ~12.6k when only the latest snapshot is kept.
- Simple **observation masking** of stale tool outputs matches LLM
  summarization on solve rate at about half the cost (JetBrains,
  "The Complexity Trap", 2025).
- A cheap-model "trim/extract" pass cut primary-model input per step from
  ~10k to 0.5–1k tokens and total cost ~57%, despite ~34% more tool calls.
- Batching independent actions cut tokens ~41% on a 28-field form.

Design:

1. **No automatic snapshots.** Server runs `--snapshot-mode none`; handlers
   request what they need.
2. **Spill, preview, reference.** Every full snapshot is written to
   `lisa-storage` (sandbox-relative path). The tool returns a capped preview
   (initial cap ~8k tokens, tune on the hardest pages) plus the path and
   explicit `[refs X–Y trimmed]` markers; `browse_snapshot(target=…)` and
   `browse_find` re-fetch narrowly. This reuses the kernel's existing
   spill-at-delivery mechanism ([token budget](../TOKEN_BUDGET.md#intra-turn-compaction)).
3. **Read path returns content, not structure.** `browse_open` for read
   intents returns reader-mode markdown of the page, not the ARIA tree.
4. **`browse_extract(question)`** answers a question over the spilled
   snapshot with a cheap model alias and returns only the answer and the
   relevant refs. Structured-output recovery follows the packs rules.
5. **Batch actions.** `browse_act` takes an ordered action list and returns
   one scoped post-action observation.
6. **Supersede stale page state** (kernel work, §5.2). Only the latest page
   observation per tab stays live in context.
7. **Untrusted-content framing.** Page-derived text is delivered as data with
   origin provenance, never as instructions. Real enforcement is code: the
   approval gate on the act tier and the ephemeral-profile rule on the read
   tier. `--allowed-origins` is explicitly *not* a security boundary per the
   Playwright docs; an origin allowlist in the handlers is defense in depth,
   not containment.

## 5. Framework changes in this repository

Neither item is browser-specific.

### 5.1 Sandboxed MCP launch (kernel `mcp_launch` + sandbox capability)

Today `RuntimeKind` launches servers directly; nothing wraps them in the
sandbox capability. Add an optional `sandbox:` block to `MCPServerSpec`
(a floor-vocabulary profile) and a `wrap_argv(profile, argv) -> argv`
member on `SandboxImplementation`. The compiler asks the capability owner to
wrap the launch; with no owner and a sandbox block present, compilation fails
closed. `lisa-pack-bwrap` implements `wrap_argv` with its existing argv
builder (`--die-with-parent`, long-lived process).

The browser pack contributes its floor needs through
`lisa.extensions.sandbox_floors` (`network: allow`, the profile path, the
output dir). Open risk to prototype first: Chromium's own sandbox needs a
nested user namespace inside bwrap; running Chromium with `--no-sandbox`
inside bwrap is a weaker posture and must be an explicit floor decision, not
a default.

### 5.2 Observation supersede (kernel turn compaction)

A tool may declare `observation_key` (template, e.g. `browser:{tab}`). When a
newer result with the same key lands, older results with that key become
eviction-preferred and are ref'd out **at the next compaction epoch**, not
immediately — rewriting old messages per call would defeat the provider
prefix cache the epoch hysteresis exists to protect. Reusable for any
stateful observation (job status, terminal screens).

No runtime change is needed for either item beyond what MCP compilation
already does. No SDK change is required.

## 6. Isolation and profiles

Three profile postures, mapped onto the existing bwrap floor vocabulary
(`persistence: direct-write < overlay < ephemeral`):

| Posture | Playwright flags | Sandbox floor | Commit path |
| --- | --- | --- | --- |
| Ephemeral (default) | `--isolated` | `persistence: ephemeral` (tmpfs) | none |
| Save / commit | `--isolated --storage-state <file>`; save via `browser_storage_state` | profile dir `capture`, `persistence: overlay` | reviewed apply |
| Direct | `--user-data-dir <profile>` | profile dir `rw`, `persistence: direct-write` | n/a |

For save/commit, prefer **storage-state JSON** (cookies + localStorage) over
capturing a whole Chromium profile: it is reviewable as a diff and applies
cleanly through `layers_export` / `layers_apply`, while a captured profile is
binary SQLite/LevelDB churn including lock files. Full-profile overlay
capture remains available for fidelity.

Profiles are pack-config-selected (hosts inject resolved configuration),
never model-selected. A persistent profile allows one browser instance at a
time; concurrent sessions need distinct profiles. bwrap is Linux-only; on
other platforms a session that requires the sandbox fails at startup,
consistent with `lisa-pack-bwrap`'s prerequisites check.

## 7. Hosts: what a TypeScript host would and would not own

Nothing in §2–§6 is host-specific; the pack works in `lisa-cli` unchanged.
What a host can uniquely add is **a user-visible browser**: an embedded
webview or the user's own Chrome, exposed to the pack as a CDP endpoint
(`--cdp-endpoint`, injected as pack configuration). That enables live view,
user takeover for logins and CAPTCHAs, and handing control back. That is the
legitimate "TS host has it, the Python CLI does not" feature; headless stays
the CLI default.

A TypeScript host cannot embed `HostRuntime` in-process. Today the only
language-neutral contract is the MCP *server* contract
([Go connector](../GO_CONNECTOR_SDK.md)); there is no host wire protocol.
A TS host therefore needs a Python process that serves `HostRuntime` over a
wire protocol (turns, streaming `TransportCallbacks`, approvals with
preserved identity). That adapter is a transport and belongs in
`lisa-hosts`, not `runtime` — `runtime` stays transport-neutral. The
protocol itself (custom JSON-RPC vs. an existing agent-client protocol) is a
separate spec.

## 8. Testing

- `packstest`: `validate_pack`, `run_tool` for each read-tier tool against a
  local fixture server; approval-gate assertions for the act tier.
- One e2e scenario per workflow: read (JS-rendered page → answer), act
  (form submit on a fixture, approval observed), commit (storage-state saved,
  exported, applied), and coding-pack verification of a dev server.
- Snapshot-cap evals on a deliberately heavy fixture page.

## 9. Open questions

1. Supersede granularity: per tab, or per tab + URL?
2. Should `browse_extract` use the session's cheap alias or a dedicated
   `mode_models` entry?
3. Is Chromium-sandbox-inside-bwrap viable on target distros, or do we accept
   `--no-sandbox` under a bwrap floor by explicit policy?
4. Do write-tier approvals need an origin-scoped "remember" by default?
5. Host wire protocol choice for non-Python hosts (separate spec).

Sources:
[Playwright MCP](https://github.com/microsoft/playwright-mcp) ·
[snapshots](https://playwright.dev/mcp/snapshots) ·
[Building Browser Agents (2025)](https://arxiv.org/html/2511.19477v1) ·
[The Complexity Trap (2025)](https://arxiv.org/abs/2508.21433) ·
[CaMeL (2025)](https://arxiv.org/abs/2503.18813) ·
[browser-use context growth](https://github.com/browser-use/browser-use/issues/5747)
