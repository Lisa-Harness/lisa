"""Lisa framework content-pack namespace under sdk/packs/.

Uses :func:`pkgutil.extend_path` so that ``sdk.packs`` spans multiple
distributions (``lisa-sdk`` owns this namespace init; each pack distribution
ships its own ``sdk/packs/<name>/`` subtree). ``extend_path`` searches every
entry on ``sys.path`` for a ``sdk/packs/`` directory and merges them into
``__path__``, making ``sdk.packs.core``, ``sdk.packs.memory``, etc. importable
even though they live in separate editable installs or installed wheels.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)