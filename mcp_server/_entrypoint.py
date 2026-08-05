"""Shared stdio entrypoint shim for first-party MCP servers.

Composes two things every stdio-launched first-party server needs, in one
import line:

1. Suppress the benign ``runpy`` ``RuntimeWarning`` emitted by the
   console-script launcher (see the rationale block that used to live in
   ``catalog/server.py``, ``content/server.py``, and ``fetch/server.py``).
   Narrowly scoped: one message pattern, one category, so real
   ``RuntimeWarning``s still surface.
2. Activate the packs venv site-packages via
   :func:`mcp_server._packs_venv_site.activate_packs_venv_site` so
   editable-install ``.pth`` files register the meta-path finders that map
   ``lisa_packs.<name>`` to source directories in the child process.

Import this module side-effect-first at the top of each server module that
needs the packs venv (``content``, ``catalog``, ``fetch``):

    import mcp_server._entrypoint  # noqa: F401 — shared stdio entrypoint shim
"""

from __future__ import annotations

import warnings

# Suppress the benign runpy RuntimeWarning emitted by the console-script
# launcher. ``lisa-*`` is entered via ``runpy.run_module``, which imports the
# package (e.g. ``mcp_server.catalog``) to locate ``server`` and then execs
# ``server`` as ``__main__``. The side-effect import of
# ``mcp_server._packs_venv_site`` re-enters the package path, and runpy warns
# that the package is in ``sys.modules`` before ``server`` finished executing.
# The "unpredictable behaviour" it warns about only occurs if ``server``
# re-imports itself as ``__main__`` mid-execution — it does not — so the
# warning is a false alarm for this entry-point pattern. Suppressed narrowly
# (one message, one category) so real RuntimeWarnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

# Activate on import so child servers get the packs venv as soon as they
# start, before any `from sdk.content import ...` triggers catalog loading.
# Imported after the warning suppression above so the runpy warning it may
# trigger is already filtered.
from mcp_server._packs_venv_site import activate_packs_venv_site  # noqa: E402

activate_packs_venv_site()

__all__ = ["activate_packs_venv_site"]