"""Turn phase submodules extracted from ``kernel.turn_runner``.

Each submodule owns one cohesive cluster of the turn pipeline. The
``kernel.turn_runner`` module re-exports these helpers so existing test
imports (``from kernel.turn_runner import _compute_capability_routing_hint``)
keep working without edits.
"""
