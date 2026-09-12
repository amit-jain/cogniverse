"""The entity-type vocabulary every layer validates against."""

from __future__ import annotations

from typing import Literal, get_args

EntityType = Literal[
    "CONCEPT", "EVENT", "ORGANIZATION", "PERSON", "PLACE", "TECHNOLOGY"
]

ENTITY_TYPES = frozenset(get_args(EntityType))
"""The entity types the agent emits; every GLiNER label maps into this set."""
