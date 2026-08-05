"""Custom YAML emitter for model manifests.

Two readability tweaks layered on top of ``yaml.safe_dump``:

1. **Blank line after each top-level sequence item.** The ``models`` list is
   the only top-level block sequence in a manifest, and each item is a
   multi-line mapping. A blank line after each item makes the file scannable
   without touching any other block sequence.

2. **Short ``capabilities`` lists rendered inline.** ``capabilities`` is a
   short list of bare strings. Rolling it onto a single flow line
   (``[text, tools, structured_output]``) saves ~4 lines per model with zero
   semantic change. Only lists whose serialized flow form fits within
   ``_MAX_FLOW_LEN`` characters are rolled up; longer lists stay block style.
   ``designations`` and ``mode_models`` are intentionally *not* rolled up so
   they remain diff-stable and easy to edit by hand.

Round-trip safety: ``yaml.safe_load`` parses flow and block sequences
identically, so the manifest loader and every existing round-trip test are
unaffected.
"""

from __future__ import annotations

import yaml
from yaml.emitter import Emitter
from yaml.events import SequenceEndEvent

# Max serialized length (incl. brackets, excl. trailing newline) for a list to
# be rendered as a single flow line. Lists longer than this stay block style.
_MAX_FLOW_LEN = 60

# Width passed to ``yaml.dump``. Wide enough to keep ``mode_models`` (a short
# scalar mapping) on one flow line; block items are unaffected by width.
_WIDTH = 120


class _BlankLineEmitter(Emitter):
    """Emitter that inserts a blank line after each top-level sequence item.

    The ``models`` list is the manifest's only root-level block sequence, but
    PyYAML renders it *indentless* (the ``-`` markers sit at column 0), so
    ``self.indent == 0`` identifies those list items. Nested block sequences
    never reach indent 0, so their items stay adjacent.
    """

    def expect_block_sequence_item(self, first: bool = False) -> None:
        if not first and not isinstance(self.event, SequenceEndEvent) and self.indent == 0:
            # Write an extra line break, then clear ``indention`` so the base
            # ``write_indent`` (called next by ``expect_block_sequence_item``)
            # still emits its own line break. Without this reset, ``write_line_break``
            # sets ``indention = True``, which makes ``write_indent`` skip its
            # break — collapsing our extra newline back to a single one.
            self.write_line_break()
            self.indention = False
        super().expect_block_sequence_item(first=first)


class ManifestDumper(_BlankLineEmitter, yaml.Dumper):
    """Dumper combining the blank-line emitter with the compact-list representer."""


def _represent_compact_list(dumper: yaml.Dumper, data: _CompactList) -> yaml.Node:
    """Render a short ``capabilities`` list as a flow sequence; else block style.

    Only ``_CompactList`` instances (used for ``capabilities``) are rolled up.
    Plain ``list`` instances (``designations`` etc.) stay block style via
    :func:`_represent_block_list` so they remain diff-stable and easy to edit.

    The width probe uses ``yaml.safe_dump`` on the raw list, which matches what
    the final emitter will produce for a flow-style scalar list (commas and
    spaces, no quotes for safe strings).
    """

    if data and all(isinstance(item, str) and item for item in data):
        probe = yaml.safe_dump(list(data), default_flow_style=True).rstrip()
        if len(probe) <= _MAX_FLOW_LEN:
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


def _represent_block_list(dumper: yaml.Dumper, data: list[object]) -> yaml.Node:
    """Render a plain list in block style (never rolled up inline)."""

    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)


class _CompactList(list):
    """List subclass rendered inline (flow style) by the emitter.

    Used for ``capabilities`` only. ``designations`` and any other list stay
    plain ``list`` so they render block style.
    """


ManifestDumper.add_representer(_CompactList, _represent_compact_list)
ManifestDumper.add_representer(list, _represent_block_list)


def compact_list(items: list[str]) -> _CompactList:
    """Flag a list for inline (flow) rendering in the emitted manifest."""

    return _CompactList(items)


def dump_manifest(payload: object) -> str:
    """Serialize ``payload`` with the manifest readability tweaks applied."""
    return yaml.dump(
        payload,
        Dumper=ManifestDumper,
        sort_keys=False,
        default_flow_style=None,
        allow_unicode=True,
        width=_WIDTH,
    )
