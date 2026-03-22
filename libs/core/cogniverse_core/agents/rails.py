"""
Content Rails for agent input/output enforcement.

Rails run before and after _process_impl() to enforce topic boundaries,
content safety, and output format constraints. Inspired by NeMo Guardrails
Colang patterns but implemented as lightweight Python checks integrated
directly into the AgentBase.process() pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RailBlockedError(Exception):
    """Raised when a rail blocks the request or response."""

    def __init__(self, rail_name: str, reason: str):
        self.rail_name = rail_name
        self.reason = reason
        super().__init__(f"Blocked by {rail_name}: {reason}")


class Rail(ABC):
    """Abstract base for a single rail check."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique rail identifier."""

    @abstractmethod
    def check(self, data: Dict[str, Any]) -> None:
        """
        Validate data against this rail.

        Args:
            data: Input or output dict to validate.

        Raises:
            RailBlockedError: If the data violates this rail.
        """


class TopicBoundaryRail(Rail):
    """
    Blocks queries outside allowed topic domains.

    Matches query text against a list of allowed topic keywords.
    If none match, the query is considered off-topic.
    """

    def __init__(self, allowed_topics: List[str]):
        self._allowed_topics = [t.lower() for t in allowed_topics]

    @property
    def name(self) -> str:
        return "topic_boundary"

    def check(self, data: Dict[str, Any]) -> None:
        query = data.get("query", "")
        if not query:
            return

        query_lower = query.lower()
        for topic in self._allowed_topics:
            if topic in query_lower:
                return

        raise RailBlockedError(
            self.name,
            f"Query is outside allowed topics: {self._allowed_topics}",
        )


class ContentSafetyRail(Rail):
    """
    Blocks queries or outputs containing disallowed patterns.

    Uses regex patterns to detect prompt injection, PII leakage,
    or other unsafe content.
    """

    def __init__(self, blocked_patterns: List[str]):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in blocked_patterns]
        self._raw_patterns = blocked_patterns

    @property
    def name(self) -> str:
        return "content_safety"

    def check(self, data: Dict[str, Any]) -> None:
        text_fields = ["query", "summary", "report", "text", "message"]
        for field in text_fields:
            value = data.get(field, "")
            if not isinstance(value, str):
                continue
            for i, pattern in enumerate(self._patterns):
                if pattern.search(value):
                    raise RailBlockedError(
                        self.name,
                        f"Content matched blocked pattern: {self._raw_patterns[i]}",
                    )


class OutputFormatRail(Rail):
    """
    Validates that output contains required fields with expected types.

    Enforces structural contracts on agent output.
    """

    def __init__(self, required_fields: Dict[str, str]):
        """
        Args:
            required_fields: Mapping of field_name → expected_type_name
                             e.g. {"results": "list", "confidence": "float"}
        """
        self._required_fields = required_fields

    @property
    def name(self) -> str:
        return "output_format"

    def check(self, data: Dict[str, Any]) -> None:
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "list": list,
            "dict": dict,
            "bool": bool,
        }

        for field_name, expected_type_name in self._required_fields.items():
            if field_name not in data:
                raise RailBlockedError(
                    self.name,
                    f"Missing required field: {field_name}",
                )
            expected = type_map.get(expected_type_name)
            if expected and not isinstance(data[field_name], expected):
                raise RailBlockedError(
                    self.name,
                    f"Field '{field_name}' must be {expected_type_name}, "
                    f"got {type(data[field_name]).__name__}",
                )


class RailChain:
    """
    Ordered chain of rails applied to input or output.

    Rails run sequentially. The first violation raises RailBlockedError.
    """

    def __init__(self, rails: Optional[List[Rail]] = None):
        self._rails: List[Rail] = rails or []

    def add(self, rail: Rail) -> None:
        self._rails.append(rail)

    def check(self, data: Dict[str, Any]) -> None:
        """Run all rails. Raises RailBlockedError on first failure."""
        for rail in self._rails:
            rail.check(data)

    def __len__(self) -> int:
        return len(self._rails)

    @property
    def rail_names(self) -> List[str]:
        return [r.name for r in self._rails]


class RailsConfig(BaseModel):
    """Configuration for content rails loaded from config.json."""

    enabled: bool = Field(True, description="Enable content rail enforcement")
    input_rails: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of input rail definitions",
    )
    output_rails: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of output rail definitions",
    )

    def build_input_chain(self) -> RailChain:
        """Build a RailChain from input rail definitions."""
        return self._build_chain(self.input_rails)

    def build_output_chain(self) -> RailChain:
        """Build a RailChain from output rail definitions."""
        return self._build_chain(self.output_rails)

    @staticmethod
    def _build_chain(rail_defs: List[Dict[str, Any]]) -> RailChain:
        chain = RailChain()
        for defn in rail_defs:
            rail_type = defn.get("type")
            params = defn.get("params", {})

            if rail_type == "topic_boundary":
                chain.add(TopicBoundaryRail(**params))
            elif rail_type == "content_safety":
                chain.add(ContentSafetyRail(**params))
            elif rail_type == "output_format":
                chain.add(OutputFormatRail(**params))
            else:
                raise ValueError(
                    f"Unknown rail type '{rail_type}'. "
                    "Supported types: topic_boundary, content_safety, output_format."
                )

        return chain
