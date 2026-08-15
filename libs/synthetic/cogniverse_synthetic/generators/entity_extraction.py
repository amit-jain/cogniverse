"""Entity-extraction training examples labelled by the production agent."""

import asyncio
import logging
import math
import re
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from cogniverse_synthetic.generators.base import (
    DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
    BaseGenerator,
    GenerationTracker,
    entity_candidate_text_fields,
    is_content_hash_topic,
    normalize_text,
)
from cogniverse_synthetic.schemas import EntityExtractionExampleSchema

logger = logging.getLogger(__name__)

EntityExtractor = Callable[[str, str], Awaitable[Any]]
DEFAULT_ENTITY_EXTRACTION_TIMEOUT_SECONDS = 300.0


class EntityExtractionGenerator(BaseGenerator):
    """Generate EntityExtractionExample data from sampled content."""

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs,
    ) -> List[BaseModel]:
        """Generate EntityExtractionExample data.

        Args:
            sampled_content: Backend-sampled content used as source texts.
            target_count: Number of examples to generate.
        """
        self.validate_inputs(sampled_content, target_count)

        tenant_id = kwargs.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required for entity extraction")
        generation_tracker = kwargs.get("generation_tracker")
        floor_count = self._generation_floor_count(
            kwargs.get(
                "generation_floor_count",
                DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
            )
        )

        logger.info("Generating %d EntityExtractionExample examples", target_count)

        candidate_texts = self._candidate_texts(sampled_content)
        labelled_examples: List[EntityExtractionExampleSchema] = []
        skipped_without_entities = 0
        last_validation_error: Exception | None = None
        for text in candidate_texts:
            if len(labelled_examples) == target_count:
                break
            try:
                extraction = await asyncio.wait_for(
                    self.entity_extractor(text, tenant_id),
                    timeout=self.extraction_timeout_seconds,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "entity extraction timed out after "
                    f"{self.extraction_timeout_seconds:g} seconds for source text "
                    f"{text!r}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"entity extraction failed for source text {text!r}"
                ) from exc

            try:
                example = self._to_example(text, extraction)
            except (ValueError, ValidationError) as exc:
                last_validation_error = exc
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(text, exc)
                continue

            if example is not None:
                labelled_examples.append(example)
            else:
                skipped_without_entities += 1
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(
                        text, "entity extractor returned no entities"
                    )

        self.require_exact_target_count(
            labelled_examples,
            target_count,
            source_context=(
                f"{len(candidate_texts)} unique source texts, "
                f"{skipped_without_entities} without entities"
            ),
            floor_count=floor_count,
            generation_tracker=generation_tracker
            if isinstance(generation_tracker, GenerationTracker)
            else None,
            cause=last_validation_error,
        )

        examples: List[BaseModel] = list(labelled_examples)

        logger.info("Generated %d EntityExtractionExample examples", len(examples))
        return examples

    def _candidate_texts(self, sampled_content: List[Dict[str, Any]]) -> List[str]:
        texts: List[str] = []
        seen_texts = set()
        for item in sampled_content:
            for field in entity_candidate_text_fields():
                text = item.get(field)
                if not isinstance(text, str) or not text.strip():
                    continue
                normalized = normalize_text(text)
                if is_content_hash_topic(normalized):
                    continue
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    texts.append(normalized)
        return texts

    @staticmethod
    def _to_mapping(value: Any, *, field: str) -> Dict[str, Any]:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            return value
        raise ValueError(f"entity extractor {field} must be an object")

    @classmethod
    def _to_example(
        cls, text: str, extraction: Any
    ) -> Optional[EntityExtractionExampleSchema]:
        payload = cls._to_mapping(extraction, field="result")
        if payload.get("query") != text:
            raise ValueError("entity extractor result query must match the source text")

        raw_entities = payload.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError("entity extractor result entities must be a list")
        if not raw_entities:
            return None

        entities: List[Dict[str, str]] = []
        entity_types_by_text: Dict[str, str] = {}
        for index, raw_entity in enumerate(raw_entities):
            entity = cls._to_mapping(raw_entity, field=f"entities[{index}]")
            entity_text = entity.get("text")
            entity_type = entity.get("type")
            if not isinstance(entity_text, str) or not entity_text.strip():
                raise ValueError(
                    f"entity extractor entities[{index}].text must be non-empty"
                )
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise ValueError(
                    f"entity extractor entities[{index}].type must be non-empty"
                )
            if entity_text != entity_text.strip():
                raise ValueError(
                    f"entity extractor entities[{index}].text contains "
                    "surrounding whitespace"
                )
            if entity_type != entity_type.strip():
                raise ValueError(
                    f"entity extractor entities[{index}].type contains "
                    "surrounding whitespace"
                )
            if (
                re.search(
                    rf"(?<!\w){re.escape(entity_text)}(?!\w)",
                    text,
                )
                is None
            ):
                raise ValueError(
                    f"entity extractor entities[{index}].text must be an exact "
                    "complete source span"
                )
            prior_type = entity_types_by_text.get(entity_text)
            if prior_type == entity_type:
                continue
            if prior_type is not None:
                raise ValueError(
                    "entity extractor result contains conflicting types for "
                    f"duplicate entity text {entity_text!r}: {prior_type!r} and "
                    f"{entity_type!r}"
                )
            entity_types_by_text[entity_text] = entity_type
            entities.append({"text": entity_text, "type": entity_type})

        raw_relationships = payload.get("relationships")
        if not isinstance(raw_relationships, list):
            raise ValueError("entity extractor result relationships must be a list")
        relationships: List[Dict[str, str]] = []
        entity_texts = {entity["text"] for entity in entities}
        for index, raw_relationship in enumerate(raw_relationships):
            relationship = cls._to_mapping(
                raw_relationship, field=f"relationships[{index}]"
            )
            subject = relationship.get("subject")
            relation = relationship.get("relation")
            object_ = relationship.get("object")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (subject, relation, object_)
            ):
                raise ValueError(
                    "entity extractor relationships"
                    f"[{index}] requires subject, relation, and object"
                )
            for field_name, value in (
                ("subject", subject),
                ("relation", relation),
                ("object", object_),
            ):
                if value != value.strip():
                    raise ValueError(
                        f"entity extractor relationships[{index}].{field_name} "
                        "contains surrounding whitespace"
                    )
            if subject not in entity_texts or object_ not in entity_texts:
                raise ValueError(
                    f"entity extractor relationships[{index}] references "
                    "an entity absent from the result"
                )
            relationships.append(
                {"source": subject, "target": object_, "type": relation}
            )

        entity_types = ",".join(dict.fromkeys(e["type"] for e in entities))
        return EntityExtractionExampleSchema(
            query=text,
            entities=entities,
            entity_types=entity_types,
            relationships=relationships,
        )

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[Any] = None,
        extraction_timeout_seconds: float = DEFAULT_ENTITY_EXTRACTION_TIMEOUT_SECONDS,
    ):
        super().__init__(pattern_extractor, agent_inferrer)
        if not callable(entity_extractor):
            raise ValueError("entity_extractor is required")
        if (
            isinstance(extraction_timeout_seconds, bool)
            or not isinstance(extraction_timeout_seconds, (int, float))
            or not math.isfinite(extraction_timeout_seconds)
            or extraction_timeout_seconds <= 0
        ):
            raise ValueError("extraction_timeout_seconds must be finite and positive")
        self.entity_extractor = entity_extractor
        self.extraction_timeout_seconds = float(extraction_timeout_seconds)
        self.optimizer_config = optimizer_config
