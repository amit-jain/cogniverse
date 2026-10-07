"""Reviewer corrections to synthetic examples awaiting approval.

A pending item's ``data`` is a record of one synthetic example schema; a
reviewer who rejects it may correct the fields that schema lets a reviewer
change. These helpers name the schema, prefill the correctable fields, and
validate corrections before they reach the approval agent.
"""

from pydantic import BaseModel, ValidationError

from cogniverse_core.approval.training_schema import (
    validate_approved_training_values,
)
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_SCHEMA
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    QueryEnhancementExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

SCHEMA_CORRECTION_FIELDS: dict[type[BaseModel], tuple[str, ...]] = {
    ProfileSelectionExampleSchema: tuple(ProfileSelectionExampleSchema.model_fields),
    QueryEnhancementExampleSchema: tuple(QueryEnhancementExampleSchema.model_fields),
    EntityExtractionExampleSchema: ("entities", "relationships"),
    RoutingExperienceSchema: ("entities", "relationships", "chosen_agent"),
    WorkflowExecutionSchema: tuple(WorkflowExecutionSchema.model_fields),
}


def schema_for_item_data(data: dict) -> type[BaseModel]:
    """The synthetic example schema ``data`` is a record of."""
    if "workflow_id" in data:
        return WorkflowExecutionSchema
    if "available_profiles" in data or "selected_profile" in data:
        return ProfileSelectionExampleSchema
    if "chosen_agent" in data:
        return RoutingExperienceSchema
    if "entities" in data or "relationships" in data:
        return EntityExtractionExampleSchema
    if "enhanced_query" in data:
        return QueryEnhancementExampleSchema
    raise ValueError("item data does not match an advertised synthetic example schema")


def review_reasoning(data: dict) -> str:
    """The generator's reasoning for ``data``, or an empty string."""
    schema = schema_for_item_data(data)
    if schema in {ProfileSelectionExampleSchema, QueryEnhancementExampleSchema}:
        reasoning = data.get("reasoning", "")
    else:
        metadata = data.get("metadata", {})
        generation_metadata = metadata.get("_generation_metadata", {})
        reasoning = generation_metadata.get("reasoning", "")
    return reasoning if isinstance(reasoning, str) else ""


def _validate_schema_record(schema: type[BaseModel], data: dict) -> None:
    unknown_fields = sorted(set(data) - set(schema.model_fields))
    if unknown_fields:
        raise ValueError(
            f"{schema.__name__} unsupported item fields: " + ", ".join(unknown_fields)
        )
    try:
        schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid {schema.__name__} record: {exc}") from exc


def _canonical_entities(value) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("entities must be a non-empty list of entity objects")

    entities = []
    for index, entity in enumerate(value):
        if not isinstance(entity, dict) or set(entity) != {"text", "type"}:
            raise ValueError(
                f"entities[{index}] must contain only text and type strings"
            )
        text = entity["text"]
        entity_type = entity["type"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(entity_type, str)
            or not entity_type.strip()
        ):
            raise ValueError(
                f"entities[{index}] must contain only text and type strings"
            )
        entities.append({"text": text.strip(), "type": entity_type.strip()})
    return entities


def _canonical_relationships(
    value,
    *,
    entity_texts: list[str],
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("relationships must be a list of relationship objects")

    relationships = []
    for index, relationship in enumerate(value):
        if not isinstance(relationship, dict) or set(relationship) != {
            "source",
            "target",
            "type",
        }:
            raise ValueError(
                f"relationships[{index}] must contain only source, target, "
                "and type strings"
            )
        canonical = {}
        for field in ("source", "target", "type"):
            field_value = relationship[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"relationships[{index}] must contain only source, target, "
                    "and type strings"
                )
            canonical[field] = field_value.strip()
        for endpoint in ("source", "target"):
            if canonical[endpoint] not in entity_texts:
                raise ValueError(
                    f"relationships[{index}].{endpoint} {canonical[endpoint]!r} "
                    f"is not one of the corrected entity texts {entity_texts!r}"
                )
        relationships.append(canonical)
    return relationships


def parse_corrections(item_data: dict, corrections) -> dict:
    """Validate a reviewer's corrections to ``item_data`` against its schema.

    Returns the corrections with entities and relationships canonicalised;
    raises ``ValueError`` naming the schema for anything the schema, the
    entity graph or the approved-training contract refuses.
    """
    schema = schema_for_item_data(item_data)
    _validate_schema_record(schema, item_data)
    if not isinstance(corrections, dict) or not corrections:
        raise ValueError(
            f"{schema.__name__} corrections must be a non-empty JSON object"
        )
    corrections = dict(corrections)

    allowed_fields = SCHEMA_CORRECTION_FIELDS[schema]
    unsupported_fields = sorted(set(corrections) - set(allowed_fields))
    if unsupported_fields:
        raise ValueError(
            f"{schema.__name__} unsupported correction fields: "
            + ", ".join(unsupported_fields)
        )

    candidate = item_data | corrections
    if schema in {EntityExtractionExampleSchema, RoutingExperienceSchema}:
        entities = _canonical_entities(candidate["entities"])
        relationships = _canonical_relationships(
            candidate.get("relationships", []),
            entity_texts=[entity["text"] for entity in entities],
        )
        candidate["entities"] = entities
        candidate["relationships"] = relationships
        if "entities" in corrections:
            corrections["entities"] = entities
        if "relationships" in corrections:
            corrections["relationships"] = relationships

    _validate_schema_record(schema, candidate)
    agent_type = APPROVED_TRAINING_AGENT_BY_SCHEMA.get(schema)
    if agent_type is not None:
        validate_approved_training_values(
            candidate,
            agent_type,
            context=f"{schema.__name__} corrected record",
        )
    return corrections


def correction_template(item_data: dict) -> tuple[str, dict]:
    """The schema name and the correctable fields of ``item_data``, prefilled."""
    schema = schema_for_item_data(item_data)
    _validate_schema_record(schema, item_data)
    template = {
        field: item_data[field]
        for field in SCHEMA_CORRECTION_FIELDS[schema]
        if field in item_data
    }
    if schema in {EntityExtractionExampleSchema, RoutingExperienceSchema}:
        template["entities"] = [
            {"text": entity.get("text"), "type": entity.get("type")}
            for entity in item_data.get("entities", [])
            if isinstance(entity, dict)
        ]
        template["relationships"] = item_data.get("relationships", [])
    return schema.__name__, template
