"""IPSAR route resolution.

parse_route() detects which user-facing mode a message targets.  All mode
routing is driven by the workflow catalog loaded from content/instructions/ at
startup.  Adding a new folder with a valid _mode.md makes that mode detectable
here without changing Python code.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# RouteMode is a plain str alias so that catalog-discovered mode names are
# accepted without requiring a Literal update for each new mode folder.
RouteMode = str

# Compiled regex cache for mode command patterns loaded from the workflow
# catalog.  Built lazily on first call to parse_route() to avoid a circular
# import at module-level (ipsar_fsm imports models; models is imported by ipsar).
_mode_route_cache: list[tuple[str, re.Pattern[str]]] | None = None


def _get_mode_routes() -> list[tuple[str, re.Pattern[str]]]:
    """Return a lazily-built list of (mode_id, compiled_pattern) pairs."""
    global _mode_route_cache  # noqa: PLW0603
    if _mode_route_cache is not None:
        return _mode_route_cache

    # Imported here to avoid a module-level circular import.
    from kernel.ipsar_fsm import get_workflow_catalog  # noqa: PLC0415

    routes: list[tuple[str, re.Pattern[str]]] = []
    for mode_id, mode_cfg in get_workflow_catalog().items():
        if mode_cfg.command_pattern:
            try:
                compiled = re.compile(mode_cfg.command_pattern, re.IGNORECASE)
                routes.append((mode_id, compiled))
            except re.error as exc:
                logger.error(
                    "Mode '%s' has invalid command_pattern '%s': %s",
                    mode_id,
                    mode_cfg.command_pattern,
                    exc,
                )

    _mode_route_cache = routes
    return _mode_route_cache


def parse_route(user_message: str) -> tuple[RouteMode, str]:
    """Detect the intended mode for a user message and extract the query text.

     Detection order:
     1. All catalog-defined modes (workflow, chat, and action; patterns come from
         command_pattern in each mode's _mode.md front-matter).
     2. 'default' fallback: full message is treated as the query.

    Args:
        user_message: Raw message text from the host user.

    Returns:
        A tuple of (mode_id, query_text).  mode_id is 'default' or any catalog
        mode name.  query_text is the portion of the message after the
        command prefix, or the full message for 'default'.
    """
    text = user_message.strip()

    for mode_id, pattern in _get_mode_routes():
        match = pattern.match(text)
        if match:
            query = ""
            group_dict = match.groupdict()
            if group_dict:
                for group_name, group_value in group_dict.items():
                    if not group_name.startswith("query") or not group_value:
                        continue
                    query = group_value.strip()
                    break
            return mode_id, query

    # --- Default conversational mode -----------------------------------------
    return "default", text
