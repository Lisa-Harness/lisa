"""Content-root discovery for the Lisa SDK package.

Provides :func:`get_content_root`, registered as the ``sdk`` entry-point in the
``lisa.extensions.content_roots`` group so the app layer can discover mode and
operation YAML definitions without hard-coded paths.

Lives in a dedicated module (rather than ``sdk/__init__.py``) because ``sdk``
is a PEP 420 implicit namespace package — it has no ``__init__.py``. The entry
point ``sdk:get_content_root`` resolves here via the editable install's
namespace assembly.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the filesystem root of this SDK package."""
    return Path(__file__).parent