"""
Custom DSPy Modules for Synthetic Data Generation

Provides validated query generation modules that ensure output quality.
"""

import logging

import dspy

from cogniverse_synthetic.dspy_signatures import GenerateEntityQuery

logger = logging.getLogger(__name__)


class ValidatedEntityQueryGenerator(dspy.Module):
    """
    Entity query generator with validation that at least one entity appears in the query.

    Uses ChainOfThought for better quality outputs - LLM reasons about which entities to include.
    Validates output and retries if needed.
    """

    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries
        self.generate = dspy.ChainOfThought(GenerateEntityQuery)

    def forward(self, topics: str, entities: str, entity_types: str) -> dspy.Prediction:
        """
        Generate entity-rich query with validation.

        Args:
            topics: Comma-separated topics
            entities: Comma-separated entity names
            entity_types: Comma-separated entity types

        Returns:
            Prediction with validated query and generation metadata

        Raises:
            ValueError: If unable to generate valid query after max_retries
        """
        entity_list = [e.strip() for e in entities.split(",") if e.strip()]

        # Build a flat list of meaningful words from all entities (length > 3 to exclude
        # stop words like "the", "and", "for"). A query is valid if any of these words
        # appears in it — an entity "Neural Networks Tutorial" is mentioned when any of
        # "Neural", "Networks", or "Tutorial" appears.
        entity_words = [
            word.lower()
            for entity in entity_list
            for word in entity.split()
            if len(word) > 3
        ]
        if not entity_words:
            # All entity words are short (e.g. single-char abbreviations) — fall back to
            # exact entity substring matching.
            entity_words = [e.lower() for e in entity_list]

        for attempt in range(self.max_retries):
            # Generate query
            result = self.generate(
                topics=topics, entities=entities, entity_types=entity_types
            )

            # Validate: at least one meaningful entity word must appear in query
            # (case-insensitive). Multi-word entities ("Neural Networks") match when
            # any constituent word is present.
            query_lower = result.query.lower()
            if any(word in query_lower for word in entity_words):
                logger.debug(
                    f"Generated valid query on attempt {attempt + 1}: {result.query}"
                )

                # Add generation metadata to result
                result._retry_count = attempt
                result._max_retries = self.max_retries

                return result

            logger.debug(
                f"Attempt {attempt + 1}/{self.max_retries}: "
                f"Query '{result.query}' does not contain any entities from {entity_list}"
            )

        # After max retries, raise error instead of using fallback
        raise ValueError(
            f"Failed to generate query containing entities after {self.max_retries} attempts. "
            f"Entities: {entity_list}, Last query: {result.query}"
        )
