"""Custom YAML emitter for model manifests.

Two readability tweaks layered on top of ``yaml.safe_dump``:

1. **Blank line between top-level sequence items.** The ``models`` list is the
   only top-level block sequence in a manifest, and each item is a multi-line
   mapping. A blank line between items makes the file scannable without
   touching any other block sequence (``capabilities``, ``designations`` are
   rolled up — see below).

2. **Short string lists rendered inline.** ``capabilities`` and
   ``designations`` are short lists of bare strings. Rolling them onto a single
   flow line (``[text, tools, structured_output]``) saves ~4 lines per model
   with zero semantic change. Only lists whose serialized flow form fits within
   ``_MAX_FLOW_LEN`` characters are rolled up; longer lists stay block style.

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
    """Emitter that inserts a blank line before non-first top-level items.

    ``self.indent == 0`` identifies the root-level block sequence (the
    ``models`` list). Nested block sequences never reach indent 0, so their
    items stay adjacent.
    """

    def expect_block_sequence_item(self, first: bool = False) -> None:
        if (
            not first
            and not isinstance(self.event, SequenceEndEvent)
            and self.indent == 0
        ):
            self.write_line_break()
        super().expect_block_sequence_item(first=first)


class ManifestDumper(yaml.Dumper):
    """Dumper combining the blank-line emitter with the compact-list representer."""


ManifestDumper.emitter = _BlankLineEmitter  # type: ignore[attr-defined]


def _represent_compact_list(
    dumper: yaml.Dumper, data: list[object]
) -> yaml.Node:
    """Render short all-string lists as flow sequences; otherwise block style.

    The width probe uses ``yaml.safe_dump`` on the raw list, which matches what
    the final emitter will produce for a flow-style scalar list (commas and
    spaces, no quotes for safe strings).
    """

    if data and all(isinstance(item, str) and item for item in data):
        probe = yaml.safe_dump(list(data), default_flow_style=True).rstrip()
        if len(probe) <= _MAX_FLOW_LEN:
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=False
    )


ManifestDumper.add_representer(list, _represent_compact_list)


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