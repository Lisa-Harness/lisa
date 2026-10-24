# SPEC: Prompt startup latency

Status: **SPEC ONLY for the follow-up work below; implementation deferred.**
Updated 2026-10-24.

## 1. Scope and implemented baseline

Reduce time from invoking `lisa prompt` to the first model request without
changing content, catalog, provider-selection, policy or failure semantics.
This work belongs to the Python framework and CLI repositories. It does not
depend on the optional
[Go broker](https://github.com/Lisa-Harness/lisa-go/blob/main/PHASE5_HOSTSIDE.md).

Framework commit `1a4d19bf` replaces per-call catalog/content subprocesses with
shared long-lived stdio sessions and prewarms the two servers concurrently.
It preserves a Python subprocess boundary, including its configuration
environment. The shared host is process-scoped, not runtime-scoped; process
exit closes it, and `close_shared_sessions()` allows explicit global teardown.
Runtime `aclose()` does not currently own those sessions.

Diagnostic profiles recorded approximately 24.8 seconds in the old adapters
and 4.4 seconds after session reuse. Most remaining adapter time waited for
the initial catalog process. These are observations from individual runs,
not controlled benchmark results. The runs used different interpreter
environments and model outcomes; the observed total wall times are not a
valid end-to-end speedup claim.

The installed CLI may use packaged copies rather than an editable checkout.
Release verification must confirm loaded module paths and package versions.
Measuring a development checkout does not upgrade the installed CLI.

## 2. Ordered delivery slices

| Slice | Owner | Deliverable | Prerequisite |
| --- | --- | --- | --- |
| A | Runtime, kernel | Cache-refresh and explicit session-lifecycle correctness | Existing shared sessions |
| B | Runtime, SDK, kernel | In-process catalog/content access through injected contracts | A and parity fixtures |
| C | CLI, runtime | Import trimming and measured bootstrap deduplication | Instrumentation; independent of B |
| D | CLI, kernel | Correct `--no-thread` result projection | Independent correctness work |
| E | Framework, CLI | Controlled benchmarks and installed-package verification | Relevant implementation slices |

Implement A before adding more cache/session reuse. B is the main remaining
startup optimization; C must be driven by its own profile. D is not evidence
of a startup improvement and must ship as a separate correctness change.

## 3. Slice A: Refresh and lifecycle ownership

Post-commit review identified a correctness gap: constructing another
`MCPContentRegistry` with equal launch parameters reuses the previous server's
cached hits. Editing an existing prompt at the same path is not visible to
the rebuilt framework until the content cache is invalidated.

Requirements:

1. Define a framework build/content-refresh boundary that invalidates content
   and catalog caches before required-content validation and catalog loading.
   Clear the kernel's local catalog snapshots as well as server/SDK caches.
   Ordinary reads within one unchanged runtime retain hit caching.
2. Give framework sessions an explicit runtime owner. Retain the owner from
   bootstrap through shutdown, including partial construction failure.
   Close from both normal and deferred-start shutdown paths. Keep `atexit`
   only as a last-resort cleanup, not the primary lifecycle.
3. Prefer runtime-scoped sessions over a process-global cache. If sharing is
   retained, specify leases/reference counts: closing one runtime must not
   terminate another runtime's active session. Do not assume the kernel's
   existing module-global registries provide concurrent-runtime isolation.
4. Bound initialization, calls, cancellation and shutdown using existing
   configuration/deadline conventions. Propagate failures with server/tool
   context; shutdown failures must be visible rather than success-shaped.
5. Track evicted owners until teardown completes. Closing must stop admission,
   settle in-flight calls, await owner tasks in their original task scope,
   terminate owned children, and join the loop thread.
6. Restrict replay to explicitly safe reads/idempotent invalidations. Never
   generalize the current transport retry to arbitrary writes or retry an MCP
   application error as though it were a disconnect.

Acceptance tests must cover same-config rebuild after a file edit, missing
content added after a miss, catalog refresh, failed initialization, a hung
server, cancellation during connect/call/close, concurrent close/admission,
and a child dying during a read. Assert no owned tasks, threads or children
remain after explicit teardown, including failed framework setup.

## 4. Slice B: Eliminate catalog/content subprocess startup

Use the existing host in-process selection to choose a runtime-injected
catalog caller/content registry. Do not import `sdk`, `mcp_server`, runtime or
host code into the kernel. The kernel owns small typed consumer contracts;
runtime wires SDK implementations.

Prefer synchronous SDK operations behind those contracts for synchronous
kernel consumers. Do not synchronously submit work to the same event loop
that is blocked waiting for it. A memory-transport implementation is acceptable
only if it has an explicit owner and proves equivalent behavior without
introducing a second framework pool.

The SDK/server code must share the same discovery/catalog/content logic:
do not duplicate front-matter parsing or Python entry-point discovery.
Extract a shared SDK service if necessary. The Go broker is not that service.

### Configuration and semantic parity

- Resolve roots, disabled packs, session provider selections, overlay root
  and default MCP tool IDs once and pass an immutable typed configuration.
  Preserve precedence, ordering, duplicate handling and absent/empty values.
- Replace relevant ambient-environment reads in the shared SDK path with
  explicit inputs. Do not temporarily mutate `os.environ` for each call;
  a lock around mutations does not isolate other threads.
- Preserve pack importability, runtime operation/tool implementation loading,
  collision errors, diagnostics, authored-text path safety, missing-content
  behavior, hit-only read caching and explicit invalidation.
- Preserve observable kernel model types, owner/provider metadata, mode
  capability merging and required-system-prompt validation. A broken content
  install must still fail startup before the first model request.
- Honor explicitly selected stdio operation and third-party servers. An
  in-process failure is reported, not silently retried through a subprocess.
- Inventory the warm pool's catalog/content entries. Reuse the same service
  when its tools must remain published; do not independently start an unused
  duplicate or remove tools without checking their consumers and manifests.

Run the same fixture corpus through stdio and in-process implementations.
Compare full catalog structures, model types, metadata, errors and authored
text. Include competing providers, disabled packs, overlays, ambient tools,
Unicode and spaced paths, traversal attempts, missing required content, and
two differently configured services without cross-contamination.

## 5. Slice C: Imports and bootstrap work

Profile CLI help separately from prompt readiness. Earlier import traces
pointed to `kernel.models`, `runtime.config` and presentation imports; these
are investigation targets, not authorization to restructure them blindly.

- Defer model/runtime/network initialization and rich rendering imports until
  their command actually needs them. Help and version must not start servers,
  discover providers over the network or initialize a model.
- Instrument framework discovery, setup, pool readiness and model submission
  with monotonic timestamps and invocation IDs. Log durations/counts, not
  prompt text, credentials, raw environment values or private content.
- Investigate apparent repeated bootstrap/setup calls with traces. A cProfile
  call count alone can count coroutine resumes and does not prove two runtime
  constructions. Remove duplicate work only after identifying actual distinct
  invocations, preserving required reload and sub-runtime behavior.
- Verify remaining warm servers individually, including memory. Optimize the
  critical path only after distinguishing mandatory readiness from safely
  deferred work; never skip required server failures to improve a benchmark.

## 6. Slice D: `--no-thread` correctness

Reproduce `ValueError: Tool-result projection requires a thread storage path`
in the CLI and identify which projection requests durable storage. Define an
explicit ephemeral result path for no-thread mode using the existing result
contract, without creating a persistent thread or weakening normal durable
projection.

Test no-tools and tool-using prompts, errors, oversized results and teardown.
Assert no persistent thread artifacts are created, model-visible results
remain usable, and ordinary threaded prompts retain their existing behavior.

## 7. Measurement and release gates

Use identical interpreter/dependency versions, content fixtures, session
settings, transport choices and machine conditions for before/after runs.
Verify module paths in separate processes. Report first-run executable-scan
effects separately; do not disable endpoint protection.

Use a deterministic local model stub to isolate framework startup. Measure
real-provider completion separately, since model latency is not startup.
Collect at least ten fresh-process samples and ten warm reads per variant;
publish p50/p95, subprocess counts, setup invocation counts and teardown time.
Separate CLI imports, framework setup, required-server readiness, first model
submission and complete command wall time.

Proposed acceptance targets on the same reference machine and fixture:

- In-process catalog/content: zero catalog/content child processes and
  p50 total catalog/content work below one second.
- Help/version: no framework sessions or model calls, with at least a 25%
  p50 improvement over the controlled import baseline.
- Warm reads: no interpreter starts and no more than 10% p95 latency
  regression versus the shared-stdio baseline.
- Lifecycle: bounded teardown and zero leaked owned children/tasks/threads.
- All cross-path parity fixtures and focused kernel/SDK/runtime/CLI tests
  pass. Performance never excuses changed catalogs or suppressed errors.

If targets are missed, publish the measured blocker and revisit the design;
do not mark the slice complete based on fewer spawns alone. Run the final
smoke tests from built/installed framework and CLI packages, not just editable
sources. Coordinate releases so the installed `lisa` executable actually
loads the new implementation.

## 8. Non-goals and completion

Do not port catalog/content to Go wrappers around Python, move Python policy
into a broker, create a resident machine-wide daemon, or add persistent
catalog caches before invalidation ownership is solved.

The optional Go broker remains a separate execution-backend project. Its
coverage currently excludes these synchronous framework adapters; an extra
IPC hop alone does not remove Python imports or content discovery costs.

Each slice is complete only with its own implementation commit, regression
tests, lifecycle verification and measured acceptance evidence. This spec
does not claim those enhancements are implemented or authorize implementing
the broker as part of prompt-startup work.
