"""In-process pack test/eval harness for Lisa content packs.

The harness lets pack authors and framework developers test pack behavior
without installing anything or spawning the catalog MCP subprocess:

- :mod:`packstest.context` — fully-wired ``ToolExecutionContext`` factory.
- :mod:`packstest.store` — in-memory ``WorkProductStore`` double.
- :mod:`packstest.catalog` — in-process pack activation and validation.
- :mod:`packstest.runner` — deterministic tool / scenario drivers.
- :mod:`packstest.case` — declarative eval cases (pytest ``eval`` marker).
- :mod:`packstest.report` — trajectory capture for diffable regressions.
- :mod:`packstest.git` — throwaway Git repository fixtures.

Boundary: imports ``kernel`` and ``sdk`` only, never hosts or packs.
"""

from __future__ import annotations

__all__ = ["context", "case", "catalog", "git", "report", "runner", "store"]