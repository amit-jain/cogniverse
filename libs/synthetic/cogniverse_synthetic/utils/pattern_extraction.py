"""
Pattern Extraction Utilities

Extract patterns from content for synthetic data generation.
Includes topic extraction, entity recognition, and temporal pattern detection.
Uses field mappings for schema-agnostic pattern extraction.
"""

import logging
import re
from datetime import datetime, timezone
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
    NON_ENTITY_CAPITALIZED_WORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "into",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
            "without",
        }
    )

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
            return {
                "topics": [],
                "entities": [],
                "temporal": [],
                "content_types": [],
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
        semantic_fields = {
            "topic": ["topic", *self.field_mappings.topic_fields],
            "description": [
                "description",
                *self.field_mappings.description_fields,
            ],
            "transcript": ["transcript", *self.field_mappings.transcript_fields],
        }

        for field_type in field_types:
            seen_fields: Set[str] = set()
            for field_name in semantic_fields.get(field_type, []):
                if field_name in seen_fields:
                    continue
                seen_fields.add(field_name)
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
                bigram = f"{words[i]} {words[i + 1]}"
                if len(bigram) > 10:  # Filter short bigrams
                    topics.add(bigram)

            # Create trigrams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                if len(trigram) > 15:  # Filter short trigrams
                    topics.add(trigram)

        # Return top 50 most relevant topics
        return sorted(topics)[:50] if topics else []

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
        named_entities = {
            entity
            for entity in entities
            if entity.casefold() not in self.NON_ENTITY_CAPITALIZED_WORDS
        }
        return sorted(named_entities)[:30] if named_entities else []

    @staticmethod
    def _parse_timestamp(timestamp: Any) -> datetime:
        if isinstance(timestamp, bool):
            raise ValueError("boolean content timestamp is invalid")
        if isinstance(timestamp, (int, float)):
            seconds = float(timestamp)
            magnitude = abs(seconds)
            if magnitude >= 1e17:
                seconds /= 1e9
            elif magnitude >= 1e14:
                seconds /= 1e6
            elif magnitude >= 1e11:
                seconds /= 1e3
            return datetime.fromtimestamp(seconds, tz=timezone.utc)

        content_date = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if content_date.utcoffset() is None:
            raise ValueError("content timestamp must include a timezone offset")
        return content_date.astimezone(timezone.utc)

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
            timestamp = sample.get("creation_timestamp")
            if timestamp is None:
                timestamp = sample.get("timestamp")
            if timestamp is not None:
                try:
                    content_date = self._parse_timestamp(timestamp)
                except (OSError, OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid content timestamp {timestamp!r}"
                    ) from exc

                days_old = (
                    datetime.now(timezone.utc) - content_date
                ).total_seconds() / 86400

                if 0 <= days_old < 30:
                    temporal.add("recent")
                    temporal.add("latest")
                elif 30 <= days_old < 90:
                    temporal.add("from this quarter")
                elif 90 <= days_old < 365:
                    temporal.add("from this year")
                else:
                    year = content_date.year
                    temporal.add(f"from {year}")

        return sorted(temporal)

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

        return sorted(content_types)

    def extract_relationships(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities

        Entity names alone do not establish a relationship, so this method
        returns no relationships until source evidence is supplied.

        Args:
            entities: List of extracted entities with metadata

        Returns:
            List of relationships between entities
        """
        return []
