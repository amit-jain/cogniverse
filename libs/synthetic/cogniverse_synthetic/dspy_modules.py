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

        for attempt in range(self.max_retries):
            # Generate query
            result = self.generate(
                topics=topics, entities=entities, entity_types=entity_types
            )

            # Validate: at least one entity must appear in query (case-insensitive)
            query_lower = result.query.lower()
            if any(entity.lower() in query_lower for entity in entity_list):
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
