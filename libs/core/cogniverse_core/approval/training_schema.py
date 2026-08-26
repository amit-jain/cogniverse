"""Canonical value contracts for approved synthetic training examples."""

from __future__ import annotations

from typing import Any, Literal, Mapping, get_args

APPROVED_SYNTHETIC_AGENT_TYPES = frozenset(
    {
        "entity_extraction",
        "profile_selection",
        "query_enhancement",
        "routing",
    }
)

APPROVED_SYNTHETIC_OUTPUT_FIELDS = {
    "profile_selection": "selected_profile",
    "query_enhancement": "enhanced_query",
    "routing": "chosen_agent",
}

PROFILE_TRAINING_MODALITIES = frozenset(
    {"audio", "code", "document", "image", "text", "video", "wiki"}
)

ProfileQueryIntent = Literal[
    "multi_modal_search",
    "video_search",
    "image_search",
    "text_search",
    "audio_search",
    "document_search",
    "relationship_aware_search",
    "ensemble_search",
    "code_search",
    "wiki_search",
]

PROFILE_QUERY_INTENT_VALUES = get_args(ProfileQueryIntent)


def _non_empty_string(values: Mapping[str, Any], field: str, context: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty {field} string")
    if value != value.strip():
        raise ValueError(f"{context} {field} must not contain surrounding whitespace")
    return value


def _string_list(values: Mapping[str, Any], field: str, context: str) -> list[str]:
    value = values.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{context} {field} must be a list of non-empty strings")
    if any(item != item.strip() for item in value):
        raise ValueError(f"{context} {field} contains surrounding whitespace")
    return value


def _validate_profile(values: Mapping[str, Any], context: str) -> None:
    available = _non_empty_string(values, "available_profiles", context)
    raw_profiles = available.split(",")
    if any(profile != profile.strip() for profile in raw_profiles):
        raise ValueError(
            f"{context} available_profiles contains surrounding whitespace"
        )
    profiles = raw_profiles
    if any(not profile for profile in profiles) or len(set(profiles)) != len(profiles):
        raise ValueError(
            f"{context} available_profiles must contain distinct non-empty names"
        )
    selected = _non_empty_string(values, "selected_profile", context)
    if selected not in profiles:
        raise ValueError(
            f"{context} selected_profile {selected!r} is absent from available_profiles"
        )
    _non_empty_string(values, "reasoning", context)
    query_intent = _non_empty_string(values, "query_intent", context)
    if query_intent not in PROFILE_QUERY_INTENT_VALUES:
        raise ValueError(f"{context} has unsupported query_intent {query_intent!r}")
    modality = _non_empty_string(values, "modality", context)
    if modality not in PROFILE_TRAINING_MODALITIES:
        raise ValueError(f"{context} has unsupported modality {modality!r}")
    complexity = _non_empty_string(values, "complexity", context)
    if complexity not in {"simple", "medium", "complex"}:
        raise ValueError(f"{context} has unsupported complexity {complexity!r}")


def _validate_entities(values: Mapping[str, Any], context: str) -> None:
    entities = values.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError(f"{context} entities must be a non-empty list")
    entity_texts: set[str] = set()
    ordered_types: list[str] = []
    for position, entity in enumerate(entities):
        if (
            not isinstance(entity, dict)
            or set(entity) != {"text", "type"}
            or not all(
                isinstance(entity[field], str) and entity[field].strip()
                for field in ("text", "type")
            )
        ):
            raise ValueError(
                f"{context} entity at position {position} requires exactly "
                "non-empty text and type strings"
            )
        text = entity["text"].strip()
        entity_type = entity["type"].strip()
        if text != entity["text"] or entity_type != entity["type"]:
            raise ValueError(
                f"{context} entity at position {position} contains surrounding whitespace"
            )
        if text in entity_texts:
            raise ValueError(f"{context} contains duplicate entity text {text!r}")
        entity_texts.add(text)
        if entity_type not in ordered_types:
            ordered_types.append(entity_type)

    relationships = values.get("relationships")
    if not isinstance(relationships, list):
        raise ValueError(f"{context} relationships must be a list")
    seen_relationships: set[tuple[str, str, str]] = set()
    for position, relationship in enumerate(relationships):
        if (
            not isinstance(relationship, dict)
            or set(relationship) != {"source", "target", "type"}
            or not all(
                isinstance(relationship[field], str) and relationship[field].strip()
                for field in ("source", "target", "type")
            )
        ):
            raise ValueError(
                f"{context} relationship at position {position} requires exactly "
                "non-empty source, target, and type strings"
            )
        for endpoint in ("source", "target"):
            value = relationship[endpoint].strip()
            if value != relationship[endpoint]:
                raise ValueError(
                    f"{context} relationship at position {position} {endpoint} "
                    "contains surrounding whitespace"
                )
            if value not in entity_texts:
                raise ValueError(
                    f"{context} relationship at position {position} {endpoint} "
                    f"{value!r} is absent from entities"
                )
        if relationship["type"] != relationship["type"].strip():
            raise ValueError(
                f"{context} relationship at position {position} type contains "
                "surrounding whitespace"
            )
        identity = (
            relationship["source"],
            relationship["target"],
            relationship["type"],
        )
        if identity in seen_relationships:
            raise ValueError(f"{context} contains duplicate relationship {identity!r}")
        seen_relationships.add(identity)


def _validate_query_enhancement(
    values: Mapping[str, Any], context: str, query: str
) -> None:
    enhanced = _non_empty_string(values, "enhanced_query", context)
    if enhanced == query:
        raise ValueError(f"{context} enhanced_query must differ from query")
    _string_list(values, "expansion_terms", context)
    _string_list(values, "synonyms", context)
    query_context = values.get("context")
    if not isinstance(query_context, str):
        raise ValueError(f"{context} context must be a string")
    if query_context != query_context.strip():
        raise ValueError(f"{context} context contains surrounding whitespace")
    _non_empty_string(values, "reasoning", context)


def validate_approved_training_values(
    values: Mapping[str, Any],
    agent_type: str,
    *,
    context: str,
) -> None:
    """Validate the exact supervision values consumed by an optimizer."""
    if not isinstance(values, Mapping):
        raise ValueError(f"{context} must be an object")
    if agent_type not in APPROVED_SYNTHETIC_AGENT_TYPES:
        raise ValueError(f"{context} has unsupported agent_type {agent_type!r}")
    query = _non_empty_string(values, "query", context)
    if agent_type == "entity_extraction":
        _validate_entities(values, context)
    elif agent_type == "profile_selection":
        _validate_profile(values, context)
    elif agent_type == "query_enhancement":
        _validate_query_enhancement(values, context, query)
    else:
        _non_empty_string(values, "chosen_agent", context)
