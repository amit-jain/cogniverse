"""
Base Generator Interface

Abstract base class for all synthetic data generators.
Defines common interface and shared functionality.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_foundation.config.unified_config import FieldMappingConfig

logger = logging.getLogger(__name__)

_ZERO_WIDTH_TRANSLATION = str.maketrans("", "", "\ufeff\u200b\u200c\u200d\u2060")
_SCHEMA_DIR = Path(__file__).resolve().parents[4] / "configs" / "schemas"
_CONTENT_HASH_TOPIC_RE = re.compile(r"^[0-9a-f]{32,}(?:_seg_\d+)?$", re.IGNORECASE)
_TEXT_DOCUMENT_MAPPING_ROLES = (
    "title",
    "description",
    "content",
    "text_content",
    "transcript",
)
DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT = 1


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
        document_mapping = schema.get("document_mapping")
        if not isinstance(document_mapping, dict):
            continue
        # Document mappings declare the text-bearing source fields. Default
        # fieldsets also carry metadata, so only mapping roles widen the
        # candidate list.
        for role in _TEXT_DOCUMENT_MAPPING_ROLES:
            field_name = document_mapping.get(role)
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


@dataclass(slots=True)
class GenerationDrop:
    """One candidate that failed validation during synthetic generation."""

    candidate: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"candidate": self.candidate, "reason": self.reason}


@dataclass(slots=True)
class GenerationTracker:
    """Per-request drop accounting and response metadata for generators."""

    optimizer: str
    target_count: int
    floor_count: int
    source_context: str = ""
    dropped_examples: list[GenerationDrop] = field(default_factory=list)
    returned_count: int = 0
    surplus_exhausted: bool = False

    def record_drop(self, candidate: str, reason: Exception | str) -> None:
        self.dropped_examples.append(
            GenerationDrop(candidate=candidate, reason=str(reason))
        )

    def dropped_examples_summary(self) -> str:
        if not self.dropped_examples:
            return ""
        return (
            f"; dropped_examples={[drop.to_dict() for drop in self.dropped_examples]}"
        )

    def finalize(
        self,
        *,
        returned_count: int,
        source_context: str,
        surplus_exhausted: bool,
    ) -> None:
        self.returned_count = returned_count
        self.source_context = source_context
        self.surplus_exhausted = surplus_exhausted

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "requested_count": self.target_count,
            "returned_count": self.returned_count,
            "shortfall_count": max(self.target_count - self.returned_count, 0),
            "floor_count": self.floor_count,
            "surplus_exhausted": self.surplus_exhausted,
            "dropped_count": len(self.dropped_examples),
            "dropped_examples": [drop.to_dict() for drop in self.dropped_examples],
        }

    def log_summary(self) -> None:
        if not self.dropped_examples and not self.surplus_exhausted:
            return
        logger.warning(
            "%s generation returned %d/%d examples from %s; "
            "floor=%d; shortfall=%d; surplus_exhausted=%s; dropped_examples=%s",
            self.optimizer,
            self.returned_count,
            self.target_count,
            self.source_context or "synthetic candidates",
            self.floor_count,
            max(self.target_count - self.returned_count, 0),
            self.surplus_exhausted,
            [drop.to_dict() for drop in self.dropped_examples],
        )


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

    def __init__(self, agent_inferrer: Optional[Any] = None):
        """
        Initialize base generator

        Args:
            agent_inferrer: Utility for inferring correct agents (optional)
        """
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
        floor_count: int = DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
        generation_tracker: GenerationTracker | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Require enough grounded examples or fail once the floor is crossed."""
        if generation_tracker is not None:
            generation_tracker.finalize(
                returned_count=len(examples),
                source_context=source_context,
                surplus_exhausted=len(examples) < target_count,
            )
            generation_tracker.log_summary()
        dropped_examples_summary = (
            generation_tracker.dropped_examples_summary()
            if generation_tracker is not None
            else ""
        )
        if generation_tracker is None and len(examples) < target_count:
            logger.warning(
                "%s generation returned %d/%d examples from %s; floor=%d; "
                "shortfall=%d; surplus_exhausted=%s",
                self.__class__.__name__,
                len(examples),
                target_count,
                source_context,
                floor_count,
                target_count - len(examples),
                True,
            )

        if len(examples) > target_count:
            raise ValueError(
                f"{self.__class__.__name__} generated {len(examples)} unique "
                f"grounded examples but target_count={target_count}; "
                f"source_context={source_context}"
            )

        if not examples:
            message = (
                f"{self.__class__.__name__} generated 0 unique grounded examples "
                f"but target_count={target_count}; source_context={source_context}"
            )
            message += dropped_examples_summary
            if cause is not None:
                raise ValueError(message) from cause
            raise ValueError(message)

        if len(examples) < floor_count:
            message = (
                f"{self.__class__.__name__} generated {len(examples)} unique "
                f"grounded examples but target_count={target_count}; "
                f"floor_count={floor_count}; source_context={source_context}"
            )
            message += dropped_examples_summary
            if cause is not None:
                raise ValueError(message) from cause
            raise ValueError(message)

    @staticmethod
    def _generation_floor_count(raw_floor_count: Any) -> int:
        if (
            isinstance(raw_floor_count, bool)
            or not isinstance(raw_floor_count, int)
            or raw_floor_count <= 0
        ):
            raise ValueError(
                "synthetic_generation_floor_count must be a positive integer"
            )
        return raw_floor_count

    def get_generator_info(self) -> Dict[str, Any]:
        """
        Get information about this generator

        Returns:
            Dictionary with generator metadata
        """
        return {
            "name": self.__class__.__name__,
            "has_agent_inferrer": self.agent_inferrer is not None,
        }
