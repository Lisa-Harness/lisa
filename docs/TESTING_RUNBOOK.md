# Testing runbook

Use this runbook to choose and run the smallest useful check, then escalate to
the required repository gate. For test-selection policy and coverage details,
see [testing and quality gates](TESTING.md).

## Prepare the checkout

Run all commands from the framework repository root:

```bash
uv sync --frozen --extra dev
```

## Choose a test layer

| Need | Layer | Command |
| --- | --- | --- |
| Reproduce one source-level behavior | Unit | `uv run pytest <test-file> -m "not live" -q` |
| Verify a pack from source without a model call | `packstest` | `uv run pytest <pack-testpaths> -m "eval or not live" -q` |
| Verify one whole runtime workflow with scripted or live models | `lisa-e2e` | `uv run lisa-e2e run <scenario.yml>` |
| Verify all tests affected by the Git change set | Package | `uv run python scripts/check_tests.py` |
| Run the complete offline pre-push suite | Full | `uv run pytest -m "not live" -q` |

`packstest` and `lisa-e2e` are harness layers, not pytest markers. Use
[`packstest`](../packstest/README.md) for deterministic tool and catalog
regressions. Use [`lisa-e2e`](../e2etest/README.md) when routing, handoffs,
policy, persistence, approvals, or model tool use must work together.

## Fast edit-test loop

1. Run the narrowest reproducing test:

   ```bash
   uv run pytest path/to/test_file.py -m "not live" -q
   ```

2. Preview the package tests selected from Git state:

   ```bash
   uv run python scripts/check_tests.py --dry-run
   ```

3. Run that package selection:

   ```bash
   uv run python scripts/check_tests.py
   ```

4. Before pushing, run the full offline suite:

   ```bash
   uv run pytest -m "not live" -q
   ```

For an intentionally small feedback loop while iterating:

```bash
uv run python scripts/check_tests.py --max-tier package
```

This capped selection is not a substitute for the uncapped package command
before commit because cross-cutting changes may require consumer suites.

## Run deterministic pack tests

From a pack or the framework checkout, run its ordinary tests and evals:

```bash
uv run pytest <pack>/tests -m "not live" -q
```

Validate a pack directly from source:

```bash
uv run python -c "
from pathlib import Path
from packstest.catalog import validate_pack

report = validate_pack(Path('../lisa-packs/serper'))
print(report.summary())
assert report.ok, report.summary()
"
```

The eval contract is deterministic: no model calls and no I/O beyond throwaway
temporary directories. See the
[pack test harness guide](../packstest/README.md) for scripted tool-call plans,
state assertions, and worktree fixtures.

## Run offline end-to-end scenarios

No provider credentials are required:

```bash
uv run lisa-e2e run \
  e2etest/tests/e2e/harness_smoke.scenario.yml

uv run lisa-e2e run \
  sdk/packs/memory/tests/e2e/dedup_recall.scenario.yml

uv run lisa-e2e run \
  e2etest/tests/e2e/compaction_spill_low_pressure.scenario.yml

uv run lisa-e2e run \
  e2etest/tests/e2e/compaction_spill_high_pressure.scenario.yml
```

These are the same zero-credential scenarios run by the `e2e-smoke` CI job.

## Run the low-cost live-model canaries

Live runs use the operator's `~/.lisa/config/models.yaml` and `~/.lisa/.env`
unless `LISA_E2E_BASE_DATA_DIR` points elsewhere. The research scenarios also
require:

- a configured provider credential for the selected model;
- `SERPER_SEARCH_API_KEY`;
- the `personas`, `markdown`, `memory`, and `serper` pack sources registered
  in [e2etest/packs.yml](../e2etest/packs.yml).

Select a concrete model ID already present in the base manifest:

```bash
MODEL=openai/gpt-5.4-mini
```

Run the basic no-tool case:

```bash
uv run lisa-e2e run \
  e2etest/tests/e2e/live_direct_answer.scenario.yml \
  --model "$MODEL"
```

Run the cheapest end-to-end research case:

```bash
uv run lisa-e2e run \
  e2etest/tests/e2e/live_web_research.scenario.yml \
  --model "$MODEL"
```

Check reliability with isolated repeat processes:

```bash
uv run lisa-e2e repeat \
  e2etest/tests/e2e/live_web_research.scenario.yml \
  --model "$MODEL" \
  --runs 2 \
  --min-pass 2
```

Verify a durable sourced artifact:

```bash
uv run lisa-e2e run \
  e2etest/tests/e2e/live_web_research_artifact.scenario.yml \
  --model "$MODEL"
```

The `--model` option pins every mode to one model and removes router overrides
for a controlled comparison; it does not edit the operator's manifest.

## Read an e2e result

Every run prints its `verdict`, usage, and bundle path. Inspect evidence in this
order:

1. `verdict.json`: the first failed oracle.
2. `transcript.md`: turns, responses, handoffs, and effect receipts.
3. `data/logs/runtime.jsonl`: routing, lifecycle, provider, and tool errors.
4. `data/logs/token-usage.jsonl`: calls, prompt/completion tokens, and cost.
5. `data/data/threads/*.current.jsonl`: exact model tool calls and results.
6. Scenario-specific state such as `data/state/` or the storage sandbox.

Bundles are under:

```text
.e2e/runs/<scenario-id>/<UTC timestamp>/
```

Exit codes are `0` for pass, `1` for an oracle failure, and `2` for setup
failure. Do not raise stage caps before identifying the failed route, finish
reason, completion usage, and receipt evidence in the bundle.

## Run pre-push quality gates

After targeted tests pass:

```bash
uv run ruff format --check .
uv run ruff check .
uv run python scripts/check_direct_io_policy.py
uv run mypy
uv run vulture
uv run lint-imports
uv run python scripts/check_coverage.py --run
```

The combined coverage command runs the complete offline suite and enforces the
package and changed-line gates. It is the closest local equivalent to the
required test and coverage checks in [CI](../.github/workflows/ci.yml).

For documentation changes:

```bash
uv run python scripts/check_docs.py
```

## CI mapping

| Local check | CI job |
| --- | --- |
| Ruff, policy, mypy, vulture, import-linter | `quality` |
| Offline suite plus package coverage | `test` on Linux/macOS/Windows and Python 3.11/3.12 |
| Four scripted scenarios above | `e2e-smoke` |
| Changed executable-line coverage | `changed-coverage` |
| Documentation links and formatting | Documentation workflow |

The live-model canaries are intentionally not CI jobs: credentials and model
spending are operator-owned. Run them when changing routing, prompts, provider
tool dispatch, completion contracts, or default execution budgets.
