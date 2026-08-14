"""
Base Generator Interface

Abstract base class for all synthetic data generators.
Defines common interface and shared functionality.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

CANONICAL_TOPIC_FIELDS = (
    "description",
    "segment_description",
    "transcript",
    "topic",
    "title",
    "video_title",
)
_CONTENT_HASH_TOPIC_RE = re.compile(r"^[0-9a-f]{32,}(?:_seg_\d+)?$", re.IGNORECASE)


def is_content_hash_topic(value: str) -> bool:
    """Return True when a topic is only an identifier-like content hash."""
    return bool(_CONTENT_HASH_TOPIC_RE.fullmatch(value.strip()))


def extract_topic(
    item: Mapping[str, Any],
    *,
    field_order: Sequence[str] = CANONICAL_TOPIC_FIELDS,
    max_words: int | None = None,
) -> str | None:
    """Return the first descriptive topic-like value from ``item``."""
    for field_name in field_order:
        value = item.get(field_name)
        if not isinstance(value, str):
            continue
        topic = " ".join(value.split())
        if not topic or is_content_hash_topic(topic):
            continue
        if max_words is not None:
            topic = " ".join(topic.split()[:max_words])
        return topic
    return None


class BaseGenerator(ABC):
    """
    Abstract base class for optimizer-specific synthetic data generators

    All generators must implement the generate() method which produces
    synthetic training examples from sampled backend content.
    """

    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
    ):
        """
        Initialize base generator

        Args:
            pattern_extractor: Utility for extracting patterns from content
            agent_inferrer: Utility for inferring correct agents (optional)
        """
        self.pattern_extractor = pattern_extractor
        self.agent_inferrer = agent_inferrer
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    async def generate(
        self, sampled_content: List[Dict[str, Any]], target_count: int, **kwargs
    ) -> List[BaseModel]:
        """
        Generate synthetic data from sampled content

        Args:
            sampled_content: Content sampled from backend (Vespa)
            target_count: Number of examples to generate
            **kwargs: Generator-specific parameters

        Returns:
            List of generated examples conforming to optimizer schema

        Raises:
            ValueError: If target_count is invalid or sampled_content is empty
        """
        pass

    def validate_inputs(
        self, sampled_content: List[Dict[str, Any]], target_count: int
    ) -> None:
        """
        Validate generator inputs

        Args:
            sampled_content: Content sampled from backend
            target_count: Number of examples to generate

        Raises:
            ValueError: If inputs are invalid
        """
        if type(target_count) is not int or target_count <= 0:
            raise ValueError(
                f"target_count must be a positive integer, got {target_count!r}"
            )

        if (
            not isinstance(sampled_content, list)
            or not sampled_content
            or any(not isinstance(record, dict) for record in sampled_content)
        ):
            raise ValueError("sampled_content must be a non-empty list of dict records")

    def require_exact_target_count(
        self,
        examples: List[Any],
        target_count: int,
        *,
        source_context: str,
    ) -> None:
        """Reject partial training sets when grounded sources are insufficient."""
        if len(examples) != target_count:
            raise ValueError(
                f"{self.__class__.__name__} generated {len(examples)} unique "
                f"grounded examples but target_count={target_count}; "
                f"source_context={source_context}"
            )

    def get_generator_info(self) -> Dict[str, Any]:
        """
        Get information about this generator

        Returns:
            Dictionary with generator metadata
        """
        return {
            "name": self.__class__.__name__,
            "has_pattern_extractor": self.pattern_extractor is not None,
            "has_agent_inferrer": self.agent_inferrer is not None,
        }
