"""
Pattern Extraction Utilities

Extract patterns from content for synthetic data generation.
Includes topic extraction, entity recognition, and temporal pattern detection.
Uses field mappings for schema-agnostic pattern extraction.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from cogniverse_foundation.config.unified_config import FieldMappingConfig

logger = logging.getLogger(__name__)


class PatternExtractor:
    """
    Extract patterns from sampled content for synthetic query generation

    Extracts:
    - Topics (bigrams and trigrams from titles/descriptions)
    - Entities (capitalized terms, technical names)
    - Temporal patterns (years, recency indicators)
    - Content types (tutorial, guide, overview, etc.)

    Uses field mappings to work with any backend schema.
    """

    # Common content type indicators
    CONTENT_TYPE_KEYWORDS = [
        "tutorial",
        "guide",
        "overview",
        "introduction",
        "walkthrough",
        "demo",
        "demonstration",
        "explanation",
        "course",
        "lecture",
        "presentation",
        "workshop",
        "webinar",
        "talk",
        "review",
        "analysis",
        "deep dive",
        "beginner",
        "advanced",
        "intermediate",
    ]

    def __init__(self, field_mappings: Optional[FieldMappingConfig] = None):
        """
        Initialize pattern extractor with field mappings

        Args:
            field_mappings: Field mapping configuration for extracting fields (uses default if None)
        """
        self.field_mappings = field_mappings or FieldMappingConfig()
        logger.info("Initialized PatternExtractor with field mappings")

    def extract(self, content_samples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Extract all patterns from content samples

        Args:
            content_samples: List of content items from Vespa

        Returns:
            Dictionary with extracted patterns:
            {
                'topics': ['machine learning', 'neural networks', ...],
                'entities': ['TensorFlow', 'PyTorch', ...],
                'temporal': ['2023', '2024', 'recent', ...],
                'content_types': ['tutorial', 'guide', ...]
            }
        """
        if not content_samples:
            logger.warning("No content samples provided for pattern extraction")
            return {
                "topics": [],
                "entities": [],
                "temporal": ["2024", "2023"],
                "content_types": ["tutorial", "guide"],
            }

        patterns = {
            "topics": self.extract_topics(content_samples),
            "entities": self.extract_entities(content_samples),
            "temporal": self.extract_temporal_patterns(content_samples),
            "content_types": self.extract_content_types(content_samples),
        }

        logger.info(
            f"Extracted patterns from {len(content_samples)} items: "
            f"{len(patterns['topics'])} topics, {len(patterns['entities'])} entities, "
            f"{len(patterns['temporal'])} temporal, {len(patterns['content_types'])} content types"
        )

        return patterns

    def _get_text_from_sample(
        self, sample: Dict[str, Any], field_types: List[str]
    ) -> str:
        """
        Extract text from sample using field mappings

        Args:
            sample: Content sample
            field_types: Types of fields to extract ('topic', 'description', 'transcript')

        Returns:
            Combined text from specified field types
        """
        texts = []

        for field_type in field_types:
            if field_type == "topic":
                # Try all configured topic fields
                for field_name in self.field_mappings.topic_fields:
                    if field_name in sample and sample[field_name]:
                        texts.append(str(sample[field_name]))
                        break
            elif field_type == "description":
                # Try all configured description fields
                for field_name in self.field_mappings.description_fields:
                    if field_name in sample and sample[field_name]:
                        texts.append(str(sample[field_name]))
                        break
            elif field_type == "transcript":
                # Try all configured transcript fields
                for field_name in self.field_mappings.transcript_fields:
                    if field_name in sample and sample[field_name]:
                        texts.append(str(sample[field_name]))
                        break

        return " ".join(texts)

    def extract_topics(self, content_samples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract topics from content titles and descriptions

        Uses bigrams and trigrams to find meaningful multi-word topics.

        Args:
            content_samples: List of content items

        Returns:
            List of extracted topics
        """
        topics: Set[str] = set()

        for sample in content_samples:
            # Get text from configured fields
            text = self._get_text_from_sample(
                sample, ["topic", "description", "transcript"]
            ).lower()

            # Extract words
            words = re.findall(r"\b[a-z]+\b", text)

            # Create bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 10:  # Filter short bigrams
                    topics.add(bigram)

            # Create trigrams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(trigram) > 15:  # Filter short trigrams
                    topics.add(trigram)

        # Return top 50 most relevant topics
        return list(topics)[:50] if topics else []

    def extract_entities(self, content_samples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract named entities (capitalized terms, brands, technical names)

        Args:
            content_samples: List of content items

        Returns:
            List of extracted entities
        """
        entities: Set[str] = set()

        for sample in content_samples:
            # Get text from configured fields
            text = self._get_text_from_sample(
                sample, ["topic", "description", "transcript"]
            )

            # Extract capitalized words and phrases (likely entities)
            # Matches: "TensorFlow", "Neural Networks", "Deep Learning"
            capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
            entities.update(capitalized)

            # Extract technical terms (CamelCase, snake_case, etc.)
            technical = re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", text)
            entities.update(technical)

        # Return top 30 entities
        return list(entities)[:30] if entities else []

    def extract_temporal_patterns(
        self, content_samples: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract temporal patterns from content

        Includes years, recency indicators, and time-based modifiers.

        Args:
            content_samples: List of content items

        Returns:
            List of temporal patterns
        """
        temporal: Set[str] = set()

        for sample in content_samples:
            # Get text from configured fields
            text = self._get_text_from_sample(sample, ["topic", "description"])

            # Extract years (2020-2029)
            years = re.findall(r"\b(202\d)\b", text)
            temporal.update(years)

            # Check timestamp for recency
            timestamp = sample.get("creation_timestamp", sample.get("timestamp"))
            if timestamp:
                try:
                    # Handle different timestamp formats
                    if isinstance(timestamp, (int, float)):
                        content_date = datetime.fromtimestamp(timestamp)
                    else:
                        content_date = datetime.fromisoformat(
                            str(timestamp).replace("Z", "+00:00")
                        )

                    days_old = (datetime.now() - content_date.replace(tzinfo=None)).days

                    # Add recency modifiers
                    if days_old < 30:
                        temporal.add("recent")
                        temporal.add("latest")
                    elif days_old < 90:
                        temporal.add("from this quarter")
                    elif days_old < 365:
                        temporal.add("from this year")
                    else:
                        year = content_date.year
                        temporal.add(f"from {year}")

                except Exception as e:
                    logger.debug(f"Error parsing timestamp: {e}")

        # Add default temporal markers if none found
        if not temporal:
            temporal = {"2024", "2023", "recent", "latest"}

        return list(temporal)

    def extract_content_types(self, content_samples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract content type indicators from titles and descriptions

        Args:
            content_samples: List of content items

        Returns:
            List of content type indicators
        """
        content_types: Set[str] = set()

        for sample in content_samples:
            # Get text from configured fields
            text = self._get_text_from_sample(sample, ["topic", "description"]).lower()

            # Check for content type keywords
            for keyword in self.CONTENT_TYPE_KEYWORDS:
                if keyword in text:
                    content_types.add(keyword)

        # Return content types or defaults
        return (
            list(content_types) if content_types else ["tutorial", "guide", "overview"]
        )

    def extract_relationships(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities

        Simple heuristic-based relationship extraction.
        For production, use NLP models like spaCy dependency parsing.

        Args:
            entities: List of extracted entities with metadata

        Returns:
            List of relationships between entities
        """
        relationships = []

        # Simple co-occurrence based relationships
        # This is a placeholder - real implementation would use NLP
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # Create relationship based on proximity or common context
                relationship = {
                    "source": entity1.get("text", ""),
                    "target": entity2.get("text", ""),
                    "type": "RELATED_TO",  # Simple default relationship
                    "confidence": 0.5,  # Placeholder confidence
                }
                relationships.append(relationship)

                # Limit to avoid explosion
                if len(relationships) >= 10:
                    return relationships

        return relationships
