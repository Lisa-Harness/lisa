# Logging Standards for Lisa

This is the canonical logging contract for the Lisa framework. Every production Python module must follow these rules.

## Module-level logger

Every Python module with logic (not purely a type/protocol file or `__init__` re-export) must define:

```python
import logging
_log = logging.getLogger(__name__)
```

No exceptions. A logger must be present even if a module initially has no logging calls — it signals that the module is expected to emit diagnostics.

## Message format

- Use lazy `%s` positional arguments: `_log.info("routing: mode=%s hint=%s", mode_id, hint_text)`
- Never f-strings: `_log.info(f"mode={mode_id}")` ❌
- Phrasing: `"<noun>: <lowercase clause, no trailing period>"`. Examples:
  - `"routing hint: no targets for gap intents %s"` ✓
  - `"lifecycle: no transition pending"` ✓
  - `"decision made"` ❌ (no noun)
  - `"Routing: Made a decision."` ❌ (mixed case, period)

## Logging levels and what goes where

### INFO — decisions and their reasons (the production default)

Record **every decision** at `INFO`, including no-ops and silent routes. The reason must be clear from the message or structured fields. This ensures production logs are never blind to decision paths.

**Examples:**

- `_log.info("routing hint: selected %s to cover %s (reason: first_single_match)", target_id, gap_intents)`
- `_log.info("lifecycle: no transition pending (current_stage=%s)", current_stage)`
- `_log.info("tool schema: empty catalog (expected for mode=%s)", mode_id)`

**Do NOT log at DEBUG:**

- Routing decisions, handoff targets, or deterministic routing outcomes
- Lifecycle state transitions, yield decisions, no-transition reasons
- Capability availability or gate decisions (even when denying)
- Security-relevant fallbacks or policy decisions
- "No X found" outcomes where X is an orchestration boundary (no mode, no model, no capability)

### DEBUG — internal logic traces

Use `DEBUG` for per-item iterations, pattern-matching traces, and keyword/regex evaluation details. These are scoped globally via `LISA_LOG_DEBUG` environment variable, so they don't spam production logs by default.

**Examples:**

- `_log.debug("intent match: keyword '%s' matched tag %s", keyword, tag)` — per-keyword detail
- `_log.debug("candidate=%s; checking fork_allowed=%s", target_id, has_fork_allowed)` — per-candidate iteration
- `_log.debug("regex pattern %r matched", pattern)` — regex evaluation

### WARNING — degraded or fallback paths

Use `WARNING` for caught exceptions, missing optional dependencies, and non-fatal resource unavailability.

**Always include `exc_info=True` when logging an exception:**

```python
try:
    result = expensive_operation()
except ValueError as exc:
    _log.warning("operation failed; using fallback: %s", exc, exc_info=True)
    result = fallback_result()
```

Or use `_log.exception()`:

```python
try:
    result = expensive_operation()
except ValueError:
    _log.exception("operation failed; using fallback")
    result = fallback_result()
```

### ERROR — unhandled or fatal errors

Use `ERROR` with `exc_info=True` or `_log.error(..., exc_info=True)`. Re-raising should typically include a log call at this level:

```python
except ValueError as exc:
    _log.error("failed to parse configuration: %s", exc, exc_info=True)
    raise
```

## Exception handling contract

Every `except` block must choose one of:

1. **Log and handle** (most common):

```python
except ValueError as exc:
    _log.warning("parsing failed; using default: %s", exc, exc_info=True)
    value = default_value
```

1. **Log and re-raise**:

```python
except ImportError as exc:
    _log.error("required module unavailable: %s", exc, exc_info=True)
    raise
```

1. **Bare `pass` is only acceptable for**:
   - Known, benign control-flow exceptions (e.g., `except StopIteration: pass` in a generator chain)
   - Operations explicitly documented to fail-silent (rare; document the exception and reason inline)

**Never do this:**

```python
except ValueError:
    pass  # ❌ Silent, unobservable failure
```

## Structured fields and event names

Use `extra={"event": "namespace.junction", ...fields}` to emit structured audit records. The `event` key is namespaced and descriptive, e.g., `"routing.hint_injected"`, `"guard.parse_failed"`, `"provider.request_succeeded"`.

**Use the helper from `sdk.logging_setup`:**

```python
from sdk.logging_setup import emit_structured_event

emit_structured_event(
    _log,
    level=logging.INFO,
    event="routing.hint_injected",
    message="routing hint: mode=%s hint=%s targets=%s",
    mode_id=mode_id,
    hint_text=hint[:200],  # truncate for safe emission
    targets=targets,
)
```

Structured fields are emitted as top-level JSON keys in the output and indexed by observability tools.

## What NOT to log

- **User prompts and conversational content** — Summarize with intent tags, mode, or bounded summaries only. If you must reference prompt text, truncate and use quotes to bound it.
- **Credentials, API keys, or bearer tokens** — Ever. Use provider/backend identifiers and safe substitutions.
- **Raw HTTP headers, response bodies, or file contents** — Use summaries (size, hash, status code, line count, error message only).
- **Memory content, embeddings, or serialized context** — Use counts and IDs only.
- **File paths from untrusted sources** — Sanitize or use generic descriptions.

**Examples of safe logging:**

- `_log.info("fetched page: url=%s size=%d status=%d", url, size_bytes, status_code)`
- `_log.warning("provider returned %d tokens; limit=%d", tokens_used, limit)`
- `_log.info("memory: %d records recalled, %d dropped", recalled, dropped)`

**Examples of unsafe logging:**

- `_log.info(f"user query: {query}")` ❌ Entire prompt exposed
- `_log.debug(f"response: {response_body}")` ❌ Entire HTTP body, possibly with data
- `_log.info(f"API key: {api_key}")` ❌ Credentials
- `_log.debug(f"memory record: {record_dict}")` ❌ Raw record content

## Repository audit and enforcement

The repository includes an AST-based audit script (`scripts/audit_logging.py`) that validates every production file against this contract. The script checks:

1. Module logger presence (`_log = logging.getLogger(__name__)`)
2. Exception handlers with silent fallbacks (bare `pass` after caught exceptions)
3. Logging calls in exception handlers without `exc_info=True` or `_log.exception()`
4. Decision-point logging below `INFO` (including in exception handlers)

Run the audit:

```bash
python scripts/audit_logging.py --check
```

To fix simple errors automatically (add missing loggers, add `exc_info=True` to exception handlers):

```bash
python scripts/audit_logging.py --fix
```

Intentional exceptions (e.g., control-flow `except StopIteration`) can be allowlisted in `scripts/audit_logging_allowlist.yaml` with a documented reason per entry.

## Examples

### Decision-point logging (routing)

**Bad:**

```python
def _compute_routing_hint(query, targets):
    intents = _detect_query_intents(query, targets)
    gap = intents - current_satisfied
    if not gap:
        return ""
```

**Good:**

```python
def _compute_routing_hint(query, targets):
    intents = _detect_query_intents(query, targets)
    gap = intents - current_satisfied
    if not gap:
        _log.info("routing hint: query satisfied by current mode (intents=%s)", intents)
        return ""
```

### Exception handling (security)

**Bad:**

```python
def validate_guard_response(response_json):
    try:
        parsed = json.loads(response_json)
    except json.JSONDecodeError:
        return GuardResult(allowed=True)  # Silent default-allow on parse failure
```

**Good:**

```python
def validate_guard_response(response_json):
    try:
        parsed = json.loads(response_json)
    except json.JSONDecodeError as exc:
        _log.warning("guard response parse failed; using default_allowed=%s: %s",
                     default_allowed, exc, exc_info=True)
        return GuardResult(allowed=default_allowed)
```

### No-op decision logging (lifecycle)

**Bad:**

```python
def resolve_lifecycle_action(state):
    if state.iteration < 3:
        return None  # No transition yet
```

**Good:**

```python
def resolve_lifecycle_action(state):
    if state.iteration < 3:
        _log.info("lifecycle: no transition pending (iteration=%d < threshold=3)", state.iteration)
        return None
```

## References

- `sdk/logging_setup.py` — the centralized logging configuration and `JSONFormatter` implementation.
- `sdk/router/README.md` — decision-point logging example in router (logged at `INFO` with `reason` and `routed_by` fields).
- `docs/COMPACTION_GUIDE.md` — no-op logging principle: *"you will never see a silent no-op."*
