"""
Synthetic Data Feedback Handler

Handle rejection feedback and regenerate synthetic data with DSPy.
"""

import asyncio
import copy
import logging
import math
from typing import Any

from pydantic import BaseModel, ValidationError

from cogniverse_core.approval.interfaces import (
    ApprovalStatus,
    FeedbackHandler,
    ReviewDecision,
    ReviewItem,
)
from cogniverse_core.approval.training_schema import (
    validate_approved_training_values,
)
from cogniverse_synthetic.dspy_modules import ValidatedSyntheticExampleRegenerator
from cogniverse_synthetic.generators.routing import _enhance_entity_query
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_SCHEMA
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    QueryEnhancementExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

logger = logging.getLogger(__name__)


def _canonical_entities(
    value: Any,
    field_name: str,
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of entity objects")

    records = []
    for index, entity in enumerate(value):
        if not isinstance(entity, dict):
            raise ValueError(
                f"{field_name}[{index}] must contain non-empty text and type strings"
            )
        text = entity.get("text")
        entity_type = entity.get("type")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(entity_type, str)
            or not entity_type.strip()
        ):
            raise ValueError(
                f"{field_name}[{index}] must contain non-empty text and type strings"
            )
        records.append({"text": text.strip(), "type": entity_type.strip()})

    return (
        records,
        [record["text"] for record in records],
        [record["type"] for record in records],
    )


def _canonical_topics(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(topic, str) or not topic.strip() for topic in value
    ):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return [topic.strip() for topic in value]


def _schema_for(data: dict[str, Any]) -> type[BaseModel]:
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


def _validate_schema_data(
    schema: type[BaseModel],
    data: dict[str, Any],
    *,
    item_id: str,
    label: str,
) -> None:
    unknown_fields = sorted(set(data) - set(schema.model_fields))
    if unknown_fields:
        raise ValueError(
            f"item={item_id} schema={schema.__name__} unsupported {label} fields: "
            + ", ".join(unknown_fields)
        )
    try:
        schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"item={item_id} schema={schema.__name__} invalid {label}: {exc}"
        ) from exc


def _canonical_relationships(
    value: Any,
    *,
    entity_texts: list[str],
    item_id: str,
    schema_name: str,
    field_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of relationship objects")

    relationships: list[dict[str, Any]] = []
    for index, relationship in enumerate(value):
        if not isinstance(relationship, dict):
            raise ValueError(
                f"{field_name}[{index}] must contain non-empty source, target, "
                "and type strings"
            )
        for endpoint in ("source", "target"):
            endpoint_value = relationship.get(endpoint)
            if not isinstance(endpoint_value, str) or not endpoint_value.strip():
                raise ValueError(
                    f"{field_name}[{index}] must contain non-empty source, target, "
                    "and type strings"
                )
            if endpoint_value not in entity_texts:
                raise ValueError(
                    f"item={item_id} schema={schema_name} relationship_index={index} "
                    f"endpoint={endpoint} value={endpoint_value!r} is not one of the "
                    f"regenerated entity texts {entity_texts!r}"
                )
        relationship_type = relationship.get("type")
        if not isinstance(relationship_type, str) or not relationship_type.strip():
            raise ValueError(
                f"{field_name}[{index}] must contain non-empty source, target, "
                "and type strings"
            )
        relationships.append(dict(relationship))
    return relationships


class SyntheticDataFeedbackHandler(FeedbackHandler):
    """
    Handle rejection feedback for synthetic data

    When humans reject synthetic queries:
    1. Log rejection reason and corrections
    2. Regenerate query with DSPy using feedback
    3. Return new ReviewItem with regenerated data

    Example feedback corrections:
    {
        "entities": [
            {"text": "PyTorch", "type": "TECHNOLOGY"},
            {"text": "Tutorial", "type": "CONTENT_TYPE"},
        ],
        "topics": ["beginner tutorials"],
    }
    """

    def __init__(
        self,
        generator: ValidatedSyntheticExampleRegenerator,
        generation_timeout_seconds: float,
        max_regeneration_attempts: int = 2,
    ):
        """
        Initialize feedback handler

        Args:
            generator: Schema-aware DSPy regenerator with an explicitly bound LM
            generation_timeout_seconds: Configured deadline for each LM call
            max_regeneration_attempts: Maximum regeneration attempts per item
        """
        if not isinstance(generator, ValidatedSyntheticExampleRegenerator):
            raise TypeError("generator must be a ValidatedSyntheticExampleRegenerator")
        if generator.lm is None:
            raise ValueError("generator.lm must be explicitly configured")
        if (
            isinstance(generation_timeout_seconds, bool)
            or not isinstance(generation_timeout_seconds, (int, float))
            or not math.isfinite(generation_timeout_seconds)
            or generation_timeout_seconds <= 0
        ):
            raise ValueError("generation_timeout_seconds must be finite and positive")
        if (
            isinstance(max_regeneration_attempts, bool)
            or not isinstance(max_regeneration_attempts, int)
            or max_regeneration_attempts < 1
        ):
            raise ValueError("max_regeneration_attempts must be a positive integer")
        self.generator = generator
        self.generation_timeout_seconds = float(generation_timeout_seconds)
        self.max_attempts = max_regeneration_attempts

        logger.info(
            f"Initialized SyntheticDataFeedbackHandler "
            f"(max_attempts: {max_regeneration_attempts})"
        )

    async def process_rejection(
        self, item: ReviewItem, decision: ReviewDecision
    ) -> ReviewItem:
        """Regenerate one rejected synthetic item from its exact review context."""
        logger.info("Processing rejection for %s: %s", item.item_id, decision.feedback)

        original_data = copy.deepcopy(item.data)
        original_query = original_data.get("query")
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("item data query must be a non-empty string")
        if "_generation_metadata" in original_data:
            raise ValueError(
                "Top-level _generation_metadata is invalid; use "
                "metadata._generation_metadata"
            )
        schema = _schema_for(original_data)
        _validate_schema_data(
            schema,
            original_data,
            item_id=item.item_id,
            label="item data",
        )
        if schema is WorkflowExecutionSchema:
            return self._apply_schema_corrections(
                item=item,
                decision=decision,
                schema=schema,
                original_data=original_data,
            )

        instruction = decision.feedback
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"item={item.item_id} schema={schema.__name__} reviewer feedback "
                "must be a non-empty string"
            )
        instruction = instruction.strip()
        corrections = copy.deepcopy(decision.corrections)
        supported_fields = set(schema.model_fields)
        if schema in {EntityExtractionExampleSchema, RoutingExperienceSchema}:
            supported_fields.add("topics")
        unsupported_fields = sorted(set(corrections) - supported_fields)
        if unsupported_fields:
            raise ValueError(
                f"item={item.item_id} schema={schema.__name__} unsupported "
                "correction fields: " + ", ".join(unsupported_fields)
            )
        if "topics" in corrections:
            _canonical_topics(corrections["topics"], "corrections topics")
        if "entities" in corrections:
            _canonical_entities(corrections["entities"], "corrections entities")

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.generator,
                        schema_name=schema.__name__,
                        source_context=copy.deepcopy(original_data),
                        reviewer_instruction=instruction,
                        corrections=copy.deepcopy(corrections),
                        schema_contract=schema.model_json_schema(),
                    ),
                    timeout=self.generation_timeout_seconds,
                )
            except TimeoutError as exc:
                last_error = TimeoutError(
                    "synthetic feedback regeneration timed out after "
                    f"{self.generation_timeout_seconds:g} seconds for "
                    f"item={item.item_id} schema={schema.__name__} "
                    f"attempt={attempt + 1}/{self.max_attempts}"
                )
                last_error.__cause__ = exc
                logger.warning("%s", last_error)
                continue
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Regeneration attempt %s failed for %s: %s",
                    attempt + 1,
                    item.item_id,
                    exc,
                )
                continue

            try:
                regenerated_data, generation = self._validated_regeneration(
                    schema=schema,
                    item_id=item.item_id,
                    original_data=original_data,
                    corrections=corrections,
                    instruction=instruction,
                    result=result,
                    attempt=attempt,
                )
                return ReviewItem(
                    item_id=f"{item.item_id}_regen_{attempt}",
                    data=regenerated_data,
                    confidence=0.0,
                    metadata={
                        "original_item_id": item.item_id,
                        "regeneration_attempt": attempt + 1,
                        "feedback": instruction,
                        "generation": generation,
                    },
                    status=ApprovalStatus.REGENERATED,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Regeneration attempt %s failed validation for %s: %s",
                    attempt + 1,
                    item.item_id,
                    exc,
                )

        raise RuntimeError(
            f"Failed to regenerate {item.item_id} after "
            f"{self.max_attempts} regeneration attempts"
        ) from last_error

    def _validated_regeneration(
        self,
        *,
        schema: type[BaseModel],
        item_id: str,
        original_data: dict[str, Any],
        corrections: dict[str, Any],
        instruction: str,
        result: Any,
        attempt: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        generated_updates = getattr(result, "updates", None)
        if not isinstance(generated_updates, dict) or not generated_updates:
            raise ValueError("regeneration returned no field updates")
        updates = copy.deepcopy(generated_updates)
        if "topics" in updates:
            generated_topics = _canonical_topics(
                updates.pop("topics"),
                "regenerated data topics",
            )
            if "topics" not in corrections or generated_topics != corrections["topics"]:
                raise ValueError(
                    f"item={item_id} schema={schema.__name__} generator returned "
                    "topics that do not match the prompt-only correction"
                )
        if not updates:
            raise ValueError("regeneration returned no schema-field updates")
        unknown_updates = sorted(set(updates) - set(schema.model_fields))
        if unknown_updates:
            raise ValueError(
                f"item={item_id} schema={schema.__name__} generator returned "
                "unsupported fields: " + ", ".join(unknown_updates)
            )
        reasoning = getattr(result, "reasoning", None)
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("regeneration returned empty reasoning")
        retry_count = getattr(result, "_retry_count", None)
        max_retries = getattr(result, "_max_retries", None)
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 1
            or retry_count >= max_retries
        ):
            raise ValueError("regeneration returned inconsistent retry metadata")

        regenerated_data = copy.deepcopy(original_data)
        regenerated_data.update(copy.deepcopy(updates))
        for field, value in corrections.items():
            if field == "topics":
                continue
            if regenerated_data.get(field) != value:
                raise ValueError(
                    f"item={item_id} schema={schema.__name__} regenerated data "
                    f"does not apply correction {field}={value!r}"
                )

        if schema in {EntityExtractionExampleSchema, RoutingExperienceSchema}:
            entities, entity_texts, _entity_types = _canonical_entities(
                regenerated_data.get("entities"),
                "regenerated data entities",
            )
            regenerated_data["entities"] = entities
            regenerated_data["relationships"] = _canonical_relationships(
                regenerated_data.get("relationships", []),
                entity_texts=entity_texts,
                item_id=item_id,
                schema_name=schema.__name__,
                field_name="regenerated data relationships",
            )
            if schema is not EntityExtractionExampleSchema:
                regenerated_data["enhanced_query"] = _enhance_entity_query(
                    regenerated_data["query"], entities
                )

        meaningful_fields = {
            EntityExtractionExampleSchema: ("query", "entities", "relationships"),
            RoutingExperienceSchema: (
                "query",
                "entities",
                "relationships",
                "enhanced_query",
                "chosen_agent",
            ),
            ProfileSelectionExampleSchema: tuple(
                ProfileSelectionExampleSchema.model_fields
            ),
            QueryEnhancementExampleSchema: tuple(
                QueryEnhancementExampleSchema.model_fields
            ),
        }[schema]
        if all(
            regenerated_data.get(field) == original_data.get(field)
            for field in meaningful_fields
        ):
            raise ValueError(
                f"item={item_id} schema={schema.__name__} regeneration did not "
                "change any training value"
            )

        generation = {
            "retry_count": retry_count,
            "max_retries": max_retries,
            "reasoning": reasoning.strip(),
        }
        if schema is RoutingExperienceSchema:
            metadata_value = regenerated_data.get("metadata", {})
            if not isinstance(metadata_value, dict):
                raise ValueError("regenerated data metadata must be an object")
            metadata = copy.deepcopy(metadata_value)
            metadata["_outcome_metadata"] = {
                "observed": False,
                "required_field_semantics": {
                    "routing_confidence": "unobserved_zero_sentinel",
                    "search_quality": "unobserved_zero_sentinel",
                    "agent_success": "unobserved_false_sentinel",
                    "processing_time": "unobserved_zero_sentinel",
                },
            }
            metadata["_generation_metadata"] = {
                **generation,
                "regeneration_attempt": attempt + 1,
                "max_regeneration_attempts": self.max_attempts,
                "regeneration": True,
                "original_query": original_data["query"],
                "human_feedback": instruction,
                "corrections_applied": corrections,
            }
            regenerated_data.update(
                {
                    "routing_confidence": 0.0,
                    "search_quality": 0.0,
                    "agent_success": False,
                    "user_satisfaction": None,
                    "processing_time": 0.0,
                    "reward": None,
                    "metadata": metadata,
                }
            )

        _validate_schema_data(
            schema,
            regenerated_data,
            item_id=item_id,
            label="regenerated data",
        )
        agent_type = APPROVED_TRAINING_AGENT_BY_SCHEMA.get(schema)
        if agent_type is not None:
            validate_approved_training_values(
                regenerated_data,
                agent_type,
                context=f"item={item_id} schema={schema.__name__} regenerated data",
            )
        return regenerated_data, generation

    def _apply_schema_corrections(
        self,
        *,
        item: ReviewItem,
        decision: ReviewDecision,
        schema: type[BaseModel],
        original_data: dict[str, Any],
    ) -> ReviewItem:
        corrections = decision.corrections
        if not corrections:
            raise ValueError(
                f"item={item.item_id} schema={schema.__name__} requires at least "
                "one correction"
            )
        unsupported_fields = sorted(set(corrections) - set(schema.model_fields))
        if unsupported_fields:
            raise ValueError(
                f"item={item.item_id} schema={schema.__name__} unsupported "
                "correction fields: " + ", ".join(unsupported_fields)
            )

        regenerated_data = original_data | corrections
        _validate_schema_data(
            schema,
            regenerated_data,
            item_id=item.item_id,
            label="regenerated data",
        )
        agent_type = APPROVED_TRAINING_AGENT_BY_SCHEMA.get(schema)
        if agent_type is not None:
            validate_approved_training_values(
                regenerated_data,
                agent_type,
                context=(
                    f"item={item.item_id} schema={schema.__name__} regenerated data"
                ),
            )
        return ReviewItem(
            item_id=f"{item.item_id}_regen_0",
            data=regenerated_data,
            confidence=0.0,
            metadata={
                "original_item_id": item.item_id,
                "regeneration_attempt": 1,
                "feedback": decision.feedback,
            },
            status=ApprovalStatus.REGENERATED,
        )
