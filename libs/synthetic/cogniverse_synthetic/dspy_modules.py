"""
Custom DSPy Modules for Synthetic Data Generation

Provides validated query generation modules that ensure output quality.
"""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import dspy

from cogniverse_synthetic.dspy_signatures import (
    GenerateEntityQuery,
    RegenerateSyntheticExample,
)

logger = logging.getLogger(__name__)


def _strict_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ValidatedSyntheticExampleRegenerator(dspy.Module):
    """Generate strict schema-field updates from one human review decision."""

    def __init__(self, max_retries: int):
        super().__init__()
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 1
        ):
            raise ValueError("max_retries must be at least 1")
        self.max_retries = max_retries
        self.regenerate = dspy.ChainOfThought(RegenerateSyntheticExample)
        self.lm = None

    def forward(
        self,
        *,
        schema_name: str,
        source_context: Mapping[str, Any],
        reviewer_instruction: str,
        corrections: Mapping[str, Any],
        schema_contract: Mapping[str, Any],
    ) -> dspy.Prediction:
        if not isinstance(schema_name, str) or not schema_name.strip():
            raise ValueError("schema_name must be a non-empty string")
        if not isinstance(source_context, Mapping) or not source_context:
            raise ValueError("source_context must be a non-empty object")
        if (
            not isinstance(reviewer_instruction, str)
            or not reviewer_instruction.strip()
        ):
            raise ValueError("reviewer_instruction must be a non-empty string")
        if not isinstance(corrections, Mapping):
            raise ValueError("corrections must be an object")
        if not isinstance(schema_contract, Mapping) or not schema_contract:
            raise ValueError("schema_contract must be a non-empty object")

        inputs = {
            "schema_name": schema_name.strip(),
            "source_context_json": _strict_json(source_context),
            "reviewer_instruction": reviewer_instruction.strip(),
            "corrections_json": _strict_json(corrections),
            "schema_contract_json": _strict_json(schema_contract),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                context = (
                    dspy.context(lm=self.lm) if self.lm is not None else nullcontext()
                )
                with context:
                    result = self.regenerate(**inputs)
                raw_updates = getattr(result, "updates_json", None)
                if not isinstance(raw_updates, str) or not raw_updates.strip():
                    raise ValueError("regenerator returned empty updates_json")
                updates = json.loads(raw_updates)
                if not isinstance(updates, dict) or not updates:
                    raise ValueError(
                        "regenerator updates_json must be a non-empty object"
                    )
                reasoning = getattr(result, "reasoning", None)
                if not isinstance(reasoning, str) or not reasoning.strip():
                    raise ValueError("regenerator returned empty reasoning")
                result.updates = updates
                result.reasoning = reasoning.strip()
                result._retry_count = attempt
                result._max_retries = self.max_retries
                return result
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc

        raise RuntimeError(
            "Synthetic example regeneration returned no strict JSON update after "
            f"{self.max_retries} attempts"
        ) from last_error


class ValidatedEntityQueryGenerator(dspy.Module):
    """
    Entity query generator that requires every complete entity in the query.

    Uses ChainOfThought for better quality outputs - LLM reasons about which entities to include.
    Validates output and retries if needed.
    """

    def __init__(
        self,
        max_retries: int,
    ):
        super().__init__()
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.max_retries = max_retries
        self.generate = dspy.ChainOfThought(GenerateEntityQuery)
        self.lm = None

    def forward(
        self,
        topics: str,
        entities: Sequence[str],
        entity_types: Sequence[str],
    ) -> dspy.Prediction:
        """
        Generate entity-rich query with validation.

        Args:
            topics: Comma-separated topics
            entities: Ordered entity names
            entity_types: Entity types aligned one-to-one with ``entities``

        Returns:
            Prediction with validated query and generation metadata

        Raises:
            RuntimeError: If no generated query passes validation
        """
        if isinstance(entities, (str, bytes)) or not isinstance(entities, Sequence):
            raise TypeError("entities must be a sequence of strings")
        if isinstance(entity_types, (str, bytes)) or not isinstance(
            entity_types, Sequence
        ):
            raise TypeError("entity_types must be a sequence of strings")

        entity_list = list(entities)
        type_list = list(entity_types)
        if not entity_list:
            raise ValueError("entities must contain at least one item")
        if len(type_list) != len(entity_list):
            raise ValueError("entity_types must align one-to-one with entities")
        for index, entity in enumerate(entity_list):
            if not isinstance(entity, str) or not entity.strip():
                raise ValueError(f"entities[{index}] must be a non-empty string")
            if entity != entity.strip():
                raise ValueError(f"entities[{index}] must not contain outer whitespace")
        for index, entity_type in enumerate(type_list):
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise ValueError(f"entity_types[{index}] must be a non-empty string")
            if entity_type != entity_type.strip():
                raise ValueError(
                    f"entity_types[{index}] must not contain outer whitespace"
                )

        serialized_entities = json.dumps(entity_list, ensure_ascii=False)
        serialized_types = json.dumps(type_list, ensure_ascii=False)

        entity_patterns = [
            re.compile(
                rf"(?<!\w){re.escape(entity)}(?!\w)",
                flags=re.IGNORECASE,
            )
            for entity in entity_list
        ]

        for attempt in range(self.max_retries):
            context = dspy.context(lm=self.lm) if self.lm is not None else nullcontext()
            with context:
                result = self.generate(
                    topics=topics,
                    entities=serialized_entities,
                    entity_types=serialized_types,
                )

            # Every supplied entity must appear as a complete, case-insensitive span.
            # This keeps labels aligned with the exact text the annotator can mark.
            # DSPy may return None when the LM fails to produce the output field.
            if not result.query:
                logger.debug(
                    f"Attempt {attempt + 1}/{self.max_retries}: query is empty"
                )
                continue
            if entity_patterns and all(
                pattern.search(result.query) for pattern in entity_patterns
            ):
                logger.debug(
                    f"Generated valid query on attempt {attempt + 1}: {result.query}"
                )

                # Add generation metadata to result
                result._retry_count = attempt
                result._max_retries = self.max_retries

                return result

            logger.debug(
                f"Attempt {attempt + 1}/{self.max_retries}: "
                f"Query '{result.query}' does not contain every complete entity from "
                f"{entity_list}"
            )

        attempt_label = "attempt" if self.max_retries == 1 else "attempts"
        raise RuntimeError(
            "Entity query validation failed after "
            f"{self.max_retries} {attempt_label} for entities: "
            + ", ".join(entity_list)
        )
