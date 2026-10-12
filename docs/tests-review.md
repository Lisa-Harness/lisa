# Test-suite review — coverage, efficacy, redundancy

> **Historical snapshot (2026-10-04).** The drifted tests it names and the
> file it referenced (`sdk/src/sdk/context/tests/test_context_broker.py`,
> since deleted with the dead context plumbing) no longer exist; retained
> only as a record of the 2026-10-04 remediation work.

**Date:** 2026-10-04 · **Method:** full-suite run under coverage (`pytest -m "not live"`), collection audit, assertion-density scan, near-duplicate detection, CI wiring cross-check.

**Baseline:** 2318 passed, 9 skipped, 1 xfailed, 9 xpassed in 4m13s.

## Summary

The suite is **strong on redundancy and breadth, weak on wiring and coverage discipline**. There is essentially no redundancy, but coverage is uneven and — more serious — several test files are invisible to CI, and some that *do* run are hollow placeholders.

| Package | Source line coverage | Verdict |
|---|---|---|
| kernel | **85%** | Strong; best-in-repo |
| sdk | **80%** | Decent, with three areas orphaned (§2) |
| mcp_server | **82%** | Fine |
| packs (core/memory/recovery) | **79%** | OK; `dispute.py` at 23% |
| packstest | 68% | Thin for a harness |
| e2etest | 66% | The e2e driver/environment under-tested themselves |
| runtime | **72%** | Weakest — intake/bootstrap/commands badly under-covered |

> Note: the bare `coverage report` TOTAL of **88%** is inflated — it counts ~13k statements of *test* code as covered. Real source-only total is ~80%.

## 1. Coverage — gaps that matter

**a) A whole subsystem is effectively blind.** `runtime/src/runtime/model_discovery/` sits at **17–59%**:

- `redact.py` 16%, `probes/azure.py` 17%, `probes/ollama.py` 19%, `probes/{google,openai,openrouter}.py` ~20%, `probes/anthropic.py` 22%, `discover.py` 37%, `classify.py` 40%, `manifest.py` 59%.

**b) runtime commands/intake — the biggest real holes (no test fallback):**

- `commands/_mcp.py` **17%**
- `intake.py` **18%**
- `bootstrap.py` **21%**
- `turn_logging.py` **24%**
- `state_provider.py` **28%**
- `commands/__init__.py` **33%** (dispatch table, lines 271–373, 420–506)
- `gateway_provider.py` **45%**

**c) e2etest and packstest under-tested as harnesses:** `e2etest/driver.py` 38%, `e2etest/environment.py` 43%, `packstest/runner.py` 57%. Thin for tooling the rest of the repo relies on.

**d) No branch coverage and no `fail_under` gate** — nothing stops regressions.

## 2. The big one — tests CI never runs

`testpaths` in `pyproject.toml` omits the six *in-src, colocated* test packages, and CI (`ci.yml:70`, `uv run pytest -m "not live"`) never executes them:

- `sdk/src/sdk/memory/tests/` (45 tests) → `sdk/memory/backup.py` shows 30%
- `sdk/src/sdk/storage/tests/` (39 tests)
- `sdk/src/sdk/guard/tests/` (10 tests)
- `sdk/src/sdk/context/tests/` — partially failing now
- `kernel/src/kernel/context/tests/` → `kernel/context/provenance.py` 44%
- `runtime/src/runtime/model_discovery/tests/` (~170 tests) → drives §1(a)

Run by hand: **300 passed, 2 failed.** Both failures are real drift CI cannot see:

| Failing test | Cause |
|---|---|
| `sdk/src/sdk/context/tests/test_context_broker.py:39` | `TypeError: timezone() argument 1 must be datetime.timedelta, not int` — test still calls `timezone(int)`; drifted from source. |
| `runtime/.../model_discovery/tests/test_openrouter_probe.py:120` | `AssertionError` — enrichment now emits a `'code'` capability the fixture doesn't expect (left over from `07a61b4 fix(azure)`). |

~270 tests exist and rot in place while their source shows misleadingly low coverage. **This is the highest-impact finding.**

## 3. Efficacy — hollow and dead tests

- **9 "xpassed" are empty stubs.** Nine xfail-decorated tests in `runtime/tests/test_core_services.py` have body `pass` (e.g. `:669`, reasons like *"needs deep migration"*). They trivially pass → xpassed, and cover nothing. **Coverage holes masquerading as pending tests.**
- **14 permanently-skipped tests.** `test_core_services.py` has 9 more `pytest.skip("migrated: … now in hosts/cli …")` (`:85`, `:90`, `:1373`, `:1727`, `:1731`, `:1737`, `:1814`, `:1818`, `:2090`). The `hosts/cli` package no longer exists — dead weight, not pending work.
- **~38 assert-free tests.** Mostly honest *"must not raise"* smoke contracts (`test_tool_output_stream.py` failure-is-swallowed cases, import-boundary tests). Legitimate pattern here, not a smell.
- **`xfail` lacks `strict=True`.** One legit xfail in `kernel/tests/test_turn_runner.py:3554`; non-strict xfail means a fix flips it to xpassed silently.

## 4. Redundancy — clean

The suite's strongest axis. Exact-body hashing + shingle near-duplicate detection across the whole corpus (incl. the copy-paste-prone families: 4 gateway suites, 7 model-discovery probe suites, 13 `turn_runner` files):

- **Zero** identical test bodies across files.
- **Zero** same-named test functions in different files.
- **No near-duplicates** above a 0.35 Jaccard threshold — parallel families share fixtures via `conftest.py` rather than cloned assertions.

Per-file counts are proportional to subsystem size (kernel 71 files, runtime 42). **No consolidation is warranted.**

## 5. Not covered at all (by design, but worth stating)

- **Live-provider paths** — `-m "not live"` in CI and no `@pytest.mark.live` marks in-tree; gateway streaming, real `httpx`, and provider probing are exercised only by hand.
- **The e2e scenario layer** — `sdk/packs/recovery/tests/e2e/session_context.md` + `stall_inspect_skip.scenario.yml` are fixtures for the `lisa-e2e` runner (needs a live model, writes `.e2e/runs/`). Unit-tested only at harness-mechanics level (`e2etest/tests/test_harness.py`); executed end-to-end only manually. The recovery pack's real stall→skip behavior has no automated guard.

## Recommendations (priority order)

1. **Wire the six in-src test packages into `testpaths`** (or relocate under each package's top-level `tests/`). Pick one layout convention. Fixes §1(a), §2, part of §1(b).
2. **Fix the 2 drifted failing tests**: `test_context_broker.py:39` (stale `timezone()` call) and `test_openrouter_probe.py:120` (accept the `'code'` capability).
3. **Set `xfail_strict = true`**, resolve the 9 xpassed empty stubs (migrate or delete), and delete the 14 "migrated" skips pointing at a gone `hosts/cli` package.
4. **Add a coverage gate** (`fail_under` per package + `--branch`), starting at current floors so it only ratchets up.
5. **Back-fill runtime**: `commands/_mcp.py`, `intake.py`, `bootstrap.py`, `turn_logging.py`, `state_provider.py`, `gateway_provider.py`, and the `commands/__init__` dispatch table.
6. **Optionally** give the recovery scenario a deterministic packstest path so stall→skip is guarded without a live model.

The one thing this suite does **not** need is de-duplication — it's already tight there.
