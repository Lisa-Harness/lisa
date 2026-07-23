"""Discovery of pack-contributed model roles.

Packs declare model roles (cross-cutting model handles like ``deep``,
``vision``, ``audio``) through the ``lisa.extensions.model_roles`` entry-point
group. Each entry point returns a mapping ``role_name -> default_model_id``
where ``default_model_id`` may be an empty string to declare only that the
role name exists (the user then pins it in ``models.yaml``).

The host merges pack-declared roles with the user's manifest at load time:

- Roles present in the manifest YAML override pack defaults (user wins).
- Roles declared by packs but absent from the YAML are seeded with the pack's
  default model id (validated against the manifest's ``models`` list by the
  shared loader).
- Unknown roles in the YAML that no pack declares are still accepted — the
  loader validates that each role references a known model, but it does not
  gate role *names* on pack declarations. This lets users pin roles ahead of
  a pack being installed.

The kernel itself has zero hardcoded role names; the one built-in role
(``deep``) is declared by the models pack via this entry point.
"""

from __future__ import annotations

import importlib.metadata
import logging

logger = logging.getLogger(__name__)

MODEL_ROLES_ENTRY_POINT_GROUP = "lisa.extensions.model_roles"


def discover_pack_roles() -> dict[str, str]:
    """Return the merged ``role_name -> default_model_id`` map from packs.

    Later entry points override earlier ones on key collision (last-wins),
    which mirrors the import-order convention for other ``lisa.extensions.*``
    groups. Failures loading a single entry point are logged and skipped so
    one broken pack does not poison role discovery for the rest.
    """

    roles: dict[str, str] = {}
    for ep in importlib.metadata.entry_points(group=MODEL_ROLES_ENTRY_POINT_GROUP):
        try:
            factory = ep.load()
            declared = factory() if callable(factory) else factory
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to load model-roles entry-point %r",
                ep.name,
                exc_info=True,
            )
            continue
        if isinstance(declared, dict):
            for role_name, model_ref in declared.items():
                normalized_role = str(role_name or "").strip().lower()
                normalized_model_ref = str(model_ref or "").strip()
                if not normalized_role:
                    continue
                roles[normalized_role] = normalized_model_ref
        else:
            logger.warning(
                "model-roles entry-point %r returned %r, expected a mapping; skipping",
                ep.name,
                type(declared).__name__,
            )
    return roles


__all__ = [
    "MODEL_ROLES_ENTRY_POINT_GROUP",
    "discover_pack_roles",
]
