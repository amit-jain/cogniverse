"""
Synthetic Data Confidence Extractor

Extract confidence scores from DSPy-generated synthetic data.
Uses retry count and entity validation as confidence signals.
"""

import logging
from typing import Any, Dict

from cogniverse_agents.approval.interfaces import ConfidenceExtractor

logger = logging.getLogger(__name__)


class SyntheticDataConfidenceExtractor(ConfidenceExtractor):
    """
    Extract confidence from DSPy synthetic data generation

    Confidence signals:
    - Retry count: Fewer retries = higher confidence
    - Entity presence: Entities in query = confidence boost
    - Query length: Reasonable length = confidence boost
    - Reasoning quality: Present reasoning = confidence boost

    Returns normalized 0-1 score where:
    - 0.9-1.0: High confidence (first attempt success, entities present)
    - 0.75-0.9: Medium confidence (1-2 retries)
    - 0.0-0.75: Low confidence (3+ retries, no entities)
    """

    def __init__(
        self,
        min_query_length: int = 10,
        max_query_length: int = 200,
        retry_penalty: float = 0.15,
    ):
        """
        Initialize confidence extractor

        Args:
            min_query_length: Minimum expected query length
            max_query_length: Maximum expected query length
            retry_penalty: Penalty per retry attempt (0-1)
        """
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.retry_penalty = retry_penalty

        logger.info(
            f"Initialized SyntheticDataConfidenceExtractor "
            f"(retry_penalty: {retry_penalty})"
        )

    def extract(self, data: Dict[str, Any]) -> float:
        """
        Extract confidence from synthetic data item

        Expected data format:
        {
            "query": "find TensorFlow tutorial",
            "entities": ["TensorFlow", "Tutorial"],
            "reasoning": "Including TensorFlow as primary entity",
            "_generation_metadata": {
                "retry_count": 0,
                "max_retries": 3,
                "confidence_signals": {...}
            }
        }

        Args:
            data: Synthetic data item dictionary

        Returns:
            Confidence score 0-1
        """
        confidence = 1.0  # Start with perfect confidence

        # Signal 1: Retry count (most important)
        metadata = data.get("_generation_metadata", {})
        retry_count = metadata.get("retry_count", 0)

        if retry_count > 0:
            # Penalty for each retry
            confidence -= retry_count * self.retry_penalty
            logger.debug(
                f"Retry penalty: {retry_count} retries -> "
                f"confidence={confidence:.2f}"
            )

        # Signal 2: Entity presence in query
        query = data.get("query", "")
        entities = data.get("entities", [])

        # Handle entities being a dict (entity_types mapping) or list
        if isinstance(entities, dict):
            # Extract entity names from dict values
            entity_list = []
            for entity_values in entities.values():
                if isinstance(entity_values, list):
                    entity_list.extend(entity_values)
                elif isinstance(entity_values, str):
                    entity_list.append(entity_values)
        else:
            entity_list = entities if isinstance(entities, list) else []

        if entity_list:
            # Check if at least one entity appears in query
            query_lower = query.lower()
            has_entity = any(str(entity).lower() in query_lower for entity in entity_list)

            if has_entity:
                # Small boost for entity presence
                confidence = min(1.0, confidence * 1.05)
                logger.debug(f"Entity present boost -> confidence={confidence:.2f}")
            else:
                # Penalty for missing entities
                confidence *= 0.7
                logger.debug(
                    f"Missing entity penalty -> confidence={confidence:.2f}"
                )

        # Signal 3: Query length (reasonable length = good quality)
        query_len = len(query)
        if query_len < self.min_query_length:
            confidence *= 0.8
            logger.debug(f"Short query penalty -> confidence={confidence:.2f}")
        elif query_len > self.max_query_length:
            confidence *= 0.9
            logger.debug(f"Long query penalty -> confidence={confidence:.2f}")

        # Signal 4: Reasoning presence (indicates LLM thought process)
        reasoning = data.get("reasoning", "")
        if reasoning and len(reasoning) > 20:
            confidence = min(1.0, confidence * 1.02)
            logger.debug(f"Reasoning present boost -> confidence={confidence:.2f}")

        # Clamp to 0-1
        confidence = max(0.0, min(1.0, confidence))

        logger.debug(
            f"Extracted confidence={confidence:.2f} for query: '{query[:50]}...'"
        )

        return round(confidence, 2)

    def get_confidence_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed confidence breakdown for debugging

        Args:
            data: Synthetic data item

        Returns:
            Dictionary with confidence factors
        """
        metadata = data.get("_generation_metadata", {})
        query = data.get("query", "")
        entities = data.get("entities", [])
        reasoning = data.get("reasoning", "")

        # Handle entities being a dict (entity_types mapping) or list
        if isinstance(entities, dict):
            entity_list = []
            for entity_values in entities.values():
                if isinstance(entity_values, list):
                    entity_list.extend(entity_values)
                elif isinstance(entity_values, str):
                    entity_list.append(entity_values)
        else:
            entity_list = entities if isinstance(entities, list) else []

        retry_count = metadata.get("retry_count", 0)
        has_entity = any(str(entity).lower() in query.lower() for entity in entity_list) if entity_list else False

        return {
            "final_confidence": self.extract(data),
            "retry_count": retry_count,
            "retry_penalty_applied": retry_count * self.retry_penalty,
            "has_entity": has_entity,
            "query_length": len(query),
            "has_reasoning": len(reasoning) > 20,
            "entities_provided": len(entity_list),
        }
