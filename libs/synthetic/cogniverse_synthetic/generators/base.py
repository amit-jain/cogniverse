"""
Base Generator Interface

Abstract base class for all synthetic data generators.
Defines common interface and shared functionality.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_foundation.config.unified_config import FieldMappingConfig

logger = logging.getLogger(__name__)

_ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\ufeff\u200b\u200c\u200d\u2060")
_SCHEMA_DIR = Path(
    os.environ.get(
        "COGNIVERSE_SCHEMAS_DIR",
        Path(__file__).resolve().parents[4] / "configs" / "schemas",
    )
)
_CONTENT_HASH_TOPIC_RE = re.compile(r"^[0-9a-f]{32,}(?:_seg_\d+)?$", re.IGNORECASE)


def normalize_text(value: str) -> str:
    """Strip invisible Unicode markers and collapse whitespace."""
    return " ".join(value.translate(_ZERO_WIDTH_TRANSLATION).split())


def is_content_hash_topic(value: str) -> bool:
    """Return True when a topic is only an identifier-like content hash."""
    return bool(_CONTENT_HASH_TOPIC_RE.fullmatch(normalize_text(value)))


def _base_text_fields() -> tuple[str, ...]:
    field_mappings = FieldMappingConfig()
    ordered_fields = (
        *field_mappings.description_fields,
        *field_mappings.transcript_fields,
        "topic",
        *field_mappings.topic_fields,
    )
    seen: set[str] = set()
    canonical_fields: list[str] = []
    for field_name in ordered_fields:
        if field_name not in seen:
            canonical_fields.append(field_name)
            seen.add(field_name)
    return tuple(canonical_fields)


@lru_cache(maxsize=1)
def _schema_text_extras() -> tuple[str, ...]:
    if not _SCHEMA_DIR.exists():
        return ()

    base_fields = set(_base_text_fields())
    extras: list[str] = []
    seen: set[str] = set()
    for schema_path in sorted(_SCHEMA_DIR.glob("*_schema.json")):
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fieldsets = schema.get("fieldsets") or []
        default_fieldset = next(
            (fieldset for fieldset in fieldsets if fieldset.get("name") == "default"),
            None,
        )
        if default_fieldset is None:
            continue
        fields = default_fieldset.get("fields")
        if not isinstance(fields, list):
            continue
        for field_name in fields:
            if (
                isinstance(field_name, str)
                and field_name.strip()
                and field_name not in base_fields
                and field_name not in seen
            ):
                extras.append(field_name)
                seen.add(field_name)
    return tuple(extras)


@lru_cache(maxsize=1)
def canonical_topic_fields() -> tuple[str, ...]:
    """Return the canonical text field order used by synthetic generators."""
    return _base_text_fields() + _schema_text_extras()


CANONICAL_TOPIC_FIELDS = canonical_topic_fields()


@lru_cache(maxsize=1)
def entity_candidate_text_fields() -> tuple[str, ...]:
    """Return the candidate text order used by entity extraction."""
    field_mappings = FieldMappingConfig()
    ordered_fields = (
        *field_mappings.topic_fields,
        *field_mappings.description_fields,
        *field_mappings.transcript_fields,
        "topic",
    )
    seen: set[str] = set()
    canonical_fields: list[str] = []
    for field_name in ordered_fields:
        if field_name not in seen:
            canonical_fields.append(field_name)
            seen.add(field_name)
    return tuple(canonical_fields) + _schema_text_extras()


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
        topic = normalize_text(value)
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
