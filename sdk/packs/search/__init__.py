"""Search and reasoning pack for Lisa.

A self-contained, externalizable research toolset providing the web search
capability plus conversational and reasoning modes that consume it:

- Modes: ``discuss`` (conversational research companion), ``search``
  (one-shot grounded answer), and ``ponder`` (reasoned-opinion workflow).
- Operations: the ``web_search`` research loop, the ``discussion_briefing``
  pre-turn enrichment for discuss, and (planned) ``deep_research`` for
  multi-hop fetch-and-synthesize.
- Tools: ``web_search`` (generic provider stub) and ``fetch_page`` (URL fetch
  + main-content extraction) so research loops synthesize from full page
  text, not snippets.

The concrete public web provider lives in separate provider packs (for
example ``lisa-pack-brave-search``), which can override this stub via
host provider preferences.

``discuss`` was absorbed from the core pack so every research/reasoning mode
is co-located with its enrichment operations. The pack is discovered via the
``lisa.extensions.content_roots`` entry point.

The pack reads configuration from environment variables only and never
imports ``sdk`` or ``kernel`` config, so it stays self-contained for
externalization.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
