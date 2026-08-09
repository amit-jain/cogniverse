"""Canonical output projection for supervised training and evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

AgentType = Literal["routing", "profile_selection", "entity_extraction"]

_OUTPUT_KEYS = {
    "routing": frozenset({"recommended_agent"}),
    "profile_selection": frozenset({"selected_profile"}),
    "entity_extraction": frozenset({"entities", "relationships"}),
}


def _require_agent_type(agent_type: str, *, context: str) -> AgentType:
    if agent_type not in _OUTPUT_KEYS:
        raise ValueError(f"{context} has unsupported agent_type {agent_type!r}")
    return agent_type


def _non_empty_string(values: Mapping[str, Any], field: str, context: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty {field} string")
    if value != value.strip():
        raise ValueError(f"{context} {field} must not contain surrounding whitespace")
    return value


def _canonical_entities(
    values: Mapping[str, Any], context: str
) -> list[dict[str, str]]:
    entities = values.get("entities")
    if not isinstance(entities, list):
        raise ValueError(f"{context} entities must be a list")

    canonical: list[dict[str, str]] = []
    for position, entity in enumerate(entities):
        entity_context = f"{context} entity at position {position}"
        if not isinstance(entity, Mapping) or set(entity) != {"text", "type"}:
            raise ValueError(f"{entity_context} requires exactly text and type fields")
        canonical.append(
            {
                "text": _non_empty_string(entity, "text", entity_context),
                "type": _non_empty_string(entity, "type", entity_context),
            }
        )
    return canonical


def _canonical_relationships(
    values: Mapping[str, Any], context: str
) -> list[dict[str, str]]:
    relationships = values.get("relationships")
    if not isinstance(relationships, list):
        raise ValueError(f"{context} relationships must be a list")

    canonical: list[dict[str, str]] = []
    for position, relationship in enumerate(relationships):
        relationship_context = f"{context} relationship at position {position}"
        if not isinstance(relationship, Mapping) or set(relationship) != {
            "source",
            "target",
            "type",
        }:
            raise ValueError(
                f"{relationship_context} requires exactly source, target, and type fields"
            )
        canonical.append(
            {
                "source": _non_empty_string(
                    relationship, "source", relationship_context
                ),
                "target": _non_empty_string(
                    relationship, "target", relationship_context
                ),
                "type": _non_empty_string(relationship, "type", relationship_context),
            }
        )
    return canonical


def canonical_output(
    agent_type: str,
    values: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate and return the exact model-facing output object."""
    canonical_agent_type = _require_agent_type(agent_type, context=context)
    if not isinstance(values, Mapping):
        raise ValueError(f"{context} must be a JSON object")

    expected_keys = _OUTPUT_KEYS[canonical_agent_type]
    if set(values) != expected_keys:
        fields = " and ".join(sorted(expected_keys))
        raise ValueError(f"{context} requires exactly the {fields} field")

    if canonical_agent_type == "routing":
        return {
            "recommended_agent": _non_empty_string(values, "recommended_agent", context)
        }
    if canonical_agent_type == "profile_selection":
        return {
            "selected_profile": _non_empty_string(values, "selected_profile", context)
        }
    return {
        "entities": _canonical_entities(values, context),
        "relationships": _canonical_relationships(values, context),
    }


def canonical_output_json(
    agent_type: str,
    values: Mapping[str, Any],
    *,
    context: str,
) -> str:
    """Serialize an exact canonical output with deterministic JSON formatting."""
    projected = canonical_output(agent_type, values, context=context)
    return json.dumps(projected, ensure_ascii=False, separators=(",", ":"))


def project_training_output(
    agent_type: str,
    source_values: Mapping[str, Any],
    *,
    context: str,
) -> str:
    """Project operational telemetry values onto the model-facing output."""
    canonical_agent_type = _require_agent_type(agent_type, context=context)
    if not isinstance(source_values, Mapping):
        raise ValueError(f"{context} output must be a JSON object")
    missing_fields = _OUTPUT_KEYS[canonical_agent_type] - set(source_values)
    if missing_fields:
        fields = " and ".join(sorted(_OUTPUT_KEYS[canonical_agent_type]))
        raise ValueError(f"{context} requires exactly the {fields} field")

    if canonical_agent_type == "routing":
        projected = {"recommended_agent": source_values.get("recommended_agent")}
    elif canonical_agent_type == "profile_selection":
        projected = {"selected_profile": source_values.get("selected_profile")}
    else:
        projected = {
            "entities": source_values.get("entities"),
            "relationships": source_values.get("relationships"),
        }
    return canonical_output_json(canonical_agent_type, projected, context=context)


def parse_canonical_output(
    agent_type: str,
    output: str,
    *,
    context: str,
) -> dict[str, Any]:
    """Parse a model output and enforce the exact canonical object shape."""
    if not isinstance(output, str) or not output.strip():
        raise ValueError(f"{context} must be a non-empty JSON string")
    try:
        values = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} must be valid JSON: {error.msg}") from error
    return canonical_output(agent_type, values, context=context)


def training_example_identity(agent_type: str, prompt: str, output: str) -> str:
    """Hash an agent prompt and canonical response for held-out exclusion."""
    canonical_agent_type = _require_agent_type(
        agent_type, context="training example identity"
    )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("training example identity requires a non-empty prompt")
    canonical_response = parse_canonical_output(
        canonical_agent_type,
        output,
        context="training example identity output",
    )
    payload = json.dumps(
        {
            "agent_type": canonical_agent_type,
            "prompt": prompt,
            "output": canonical_response,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
