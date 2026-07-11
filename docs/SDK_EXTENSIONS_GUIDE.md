# SDK Integrations Guide

Extensions add integration-specific capabilities through typed packages and entry points. They should be deterministic by default, host-neutral, and easy to validate without live services.

## Architectural Role: Extensions vs. MCP vs. Packs

As the kernel transitions to an MCP-native architecture, extensions complement MCP and Packs by serving a fundamentally different purpose:

- **Passive Intake vs. Active RPC**: MCP is active and client-driven (the model explicitly calls an MCP tool or reads an MCP resource during a turn). Extensions (`SignalConnectors` and `DataSources`) are passive and opportunistic. They run in the background via the pressure-driven `slice.py` budget, pulling ambient state (e.g., GitHub PRs, Stripe billing) without waiting for a user query, and injecting it into the `heartbeat_context` or `digest_sections`. MCP cannot natively orchestrate this passive background loop.
- **Deep Pipeline Hooks**: MCP stops at the tool and resource boundary. Extensions hook directly into the framework's internal intelligence pipeline (`lisa.extensions.context_evidence`, `lisa.extensions.anomaly_detectors`, `lisa.extensions.heartbeat_context`). These require strongly-typed Python objects and tight coupling with the kernel's state loops, which isn't suited for generic JSON-RPC over stdio.
- **Separation of Concerns**:
  - **MCP Servers**: Own *actions* and *storage* (client-invoked tools, file access, search, memory).
  - **Packs**: Own *workflow* and *personality* (`_mode.md` routing, allowed operations, prompt templates).
  - **Extensions**: Own *ambient telemetry* (background data collection and context injection).

## Package Shape

Each extension package should include:

- `config.py`: package-local settings adapter that reads from `kernel.config.settings`.
- `models.py`: Pydantic output contracts.
- One runtime owner module:
  - `connector.py` for signal connectors.
  - `data_source.py` for data sources.
- Optional contribution modules such as `refresh_summary.py`, `digest.py`, or `context_evidence.py`.
- `pyproject.toml`: package metadata and `lisa.extensions.*` entry points.
- `tests/test_import_boundaries.py`: verifies no private app imports.
- Focused config and contract tests.

Author-facing rich attachment and host-rendered summary DTOs live in `sdk.rendering`, not in a host app module.

## Entry Point Groups

Common public entry-point groups:

- `lisa.extensions.signal_connectors`
- `lisa.extensions.data_sources`
- `lisa.extensions.refresh_summary_sections`
- `lisa.extensions.digest_sections`
- `lisa.extensions.context_evidence`
- `lisa.extensions.heartbeat_context`
- `lisa.extensions.anomaly_detectors`
- `lisa.extensions.content_roots`

Hosts decide which groups they discover and how often extension outputs are collected.

## Public Integration Packs

- `lisa_pack_cloudflare`: Cloudflare zone, worker, and edge infrastructure status pack (published as `lisa-pack-cloudflare`). Includes modes, operations, content, signal connectors, anomaly detectors, and all SDK extension kinds.
- `lisa_pack_github`: GitHub repository, CI/CD, and PR health status pack (published as `lisa-pack-github`). Includes modes, operations, content, signal connectors, anomaly detectors, and all SDK extension kinds.
- `lisa_pack_stripe`: Stripe subscription, trial, and billing health status pack (published as `lisa-pack-stripe`). Includes modes, operations, content, signal connectors, anomaly detectors, and all SDK extension kinds.

Provider packs:

- `lisa_pack_mem0` (published as `lisa-pack-mem0`): external memory provider pack that contributes Mem0 MCP memory ownership metadata.
- `lisa_pack_brave_search` (published as `lisa-pack-brave-search`): external `web_search` provider implementation pack that can replace the bundled search stub.

These packs live as top-level directories in the workspace and are installable via `/pack install` from PyPI or Git (`github.com/the-migus-group`). They can also be installed from a local folder for development.

CloudServers operations and market-intel extensions are intentionally not included in this public staged tree.

## Deterministic Tests

Run package tests without live API calls by default:

```powershell
uv run pytest sdk_packs/github/tests -m "not live"
uv run pytest sdk_packs/cloudflare/tests -m "not live"
uv run pytest sdk_packs/stripe/tests -m "not live"
```

Live tests must be explicit, opt-in, and secret-safe. Do not make live tests part of default local or CI validation.

## Hello World Signal Connector

For a minimal extension, start with one signal connector package and validate it through the public extension harness before wiring it into a fuller host.

```plaintext
C:\Code\lisa-pack-hello-world\
  sdk_packs/
    hello_world/
      __init__.py
      connector.py
      models.py
  pyproject.toml
  tests/
  test_import_boundaries.py
```

`C:\Code\lisa-pack-hello-world\sdk_packs\hello_world\models.py`

```python
from pydantic import BaseModel


class HelloWorldSnapshot(BaseModel):
  collected_at: str
  status: str = "ok"
  message: str = "hello world"
```

`C:\Code\lisa-pack-hello-world\sdk_packs\hello_world\connector.py`

```python
import httpx

from kernel.intake.connectors import SignalConnector

from .models import HelloWorldSnapshot


class HelloWorldSignalConnector(SignalConnector[HelloWorldSnapshot]):
  connector_id = "hello_world"
  display_name = "Hello World"

  async def collect(
    self,
    client: httpx.AsyncClient,
    collected_at: str,
  ) -> HelloWorldSnapshot:
    del client
    return HelloWorldSnapshot(collected_at=collected_at, status="ok", message="hello world")

  def build_unconfigured_snapshot(self, collected_at: str) -> HelloWorldSnapshot:
    return HelloWorldSnapshot(
      collected_at=collected_at,
      status="unconfigured",
      message="missing configuration",
    )

  def build_error_snapshot(self, collected_at: str) -> HelloWorldSnapshot:
    return HelloWorldSnapshot(
      collected_at=collected_at,
      status="error",
      message="connector failed",
    )
```

`C:\Code\lisa-pack-hello-world\pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["sdk_packs.hello_world*"]

[project]
name = "lisa-pack-hello-world"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["lisa-kernel", "httpx>=0.27.0"]

[project.entry-points."lisa.extensions.signal_connectors"]
hello_world = "sdk_packs.hello_world.connector:HelloWorldSignalConnector"
```

From `lisa_framework/` on Windows:

```powershell
uv pip install -e "C:\Code\lisa-pack-hello-world"
uv run python -c "from kernel.testing.extension_harness import list_extension_entries; print([entry.model_dump() for entry in list_extension_entries() if entry.entry_id == 'hello_world'])"
uv run python -c "from kernel.testing.extension_harness import build_connector_contract_preview, load_signal_connector; connector = load_signal_connector('hello_world'); print(build_connector_contract_preview(connector))"
```

Those commands prove entry-point discovery and contract shape without relying on the private app, Teams, Azure, or a live external API.

## Author Checklist

1. Define settings and Pydantic contracts first.
2. Implement one clear runtime owner module.
3. Register entry points in the package `pyproject.toml`.
4. Keep imports limited to `kernel`, `sdk`, and package-local modules.
5. Add deterministic unit tests with fake transports or hook stubs.
6. Add import-boundary tests.
7. Add opt-in live tests only when a real service contract needs them.
8. Document required host callbacks, settings, and permissions.

The public `kernel.testing.extension_harness` module is the intended first validation surface for hello-world connectors and data sources.

## Host Boundary

Extensions should return typed snapshots, contribution fragments, or evidence records. They should not send chat messages, choose proactive recipients, own production storage, or import private host code. A host scheduler or runtime adapter owns those responsibilities.
