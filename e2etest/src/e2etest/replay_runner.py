"""Run one exact-mode turn replay in this process and print the report.

The e2e determinism gate replays recorded turns through the production
replay engine, but the engine boots a full runtime — and process-global
registries (host commands, catalog caches) make one process unable to host
consecutive independent replays, exactly as the CLI's own ``repeat``
command documents for scenario runs. This module is the subprocess entry
the gate shells out to, so every replay gets a fresh process.

Usage::

    python -m e2etest.replay_runner <thread-base> <turn-index> \
        <scratch-data-dir> <source-data-dir> <session-context>

The report is printed to stdout as JSON (``dataclasses.asdict`` shape);
setup errors print to stderr and exit non-zero.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

from runtime.replay import replay_turn

_ARG_COUNT = 5


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != _ARG_COUNT:
        print(
            "usage: python -m e2etest.replay_runner "
            "<thread-base> <turn-index> <scratch-data-dir> <source-data-dir> <session-context>",
            file=sys.stderr,
        )
        return 2
    thread_base, turn_index, scratch, source, session_context = args
    try:
        report = asyncio.run(
            replay_turn(
                source_thread_path=Path(thread_base),
                turn_index=int(turn_index),
                scratch_data_dir=Path(scratch),
                source_data_dir=Path(source),
                session_context=session_context,
            )
        )
    except Exception as exc:  # the gate surfaces the message
        print(f"replay runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(dataclasses.asdict(report), default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
