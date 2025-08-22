# src/routing/base.py
"""
Base classes and interfaces for the comprehensive routing system.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SearchModality(Enum):
    """Enum for search modalities."""

    VIDEO = "video"
    TEXT = "text"
    BOTH = "both"
    NONE = "none"


class GenerationType(Enum):
    """Enum for generation types."""

    RAW_RESULTS = "raw_results"
    SUMMARY = "summary"
    DETAILED_REPORT = "detailed_report"


@dataclass
class RoutingDecision:
    """
    Represents a routing decision with full context and metadata.
    """

    search_modality: SearchModality
    generation_type: GenerationType
    confidence_score: float = 0.0
    routing_method: str = ""
    temporal_info: dict[str, Any] | None = None
    entities_detected: list[dict[str, Any]] | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "search_modality": self.search_modality.value,
            "generation_type": self.generation_type.value,
            "confidence_score": self.confidence_score,
            "routing_method": self.routing_method,
            "temporal_info": self.temporal_info,
            "entities_detected": self.entities_detected,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "needs_video_search": self.search_modality
            in [SearchModality.VIDEO, SearchModality.BOTH],
            "needs_text_search": self.search_modality
            in [SearchModality.TEXT, SearchModality.BOTH],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingDecision":
        """Create from dictionary format."""
        return cls(
            search_modality=SearchModality(data.get("search_modality", "both")),
            generation_type=GenerationType(data.get("generation_type", "raw_results")),
            confidence_score=data.get("confidence_score", 0.0),
            routing_method=data.get("routing_method", ""),
            temporal_info=data.get("temporal_info"),
            entities_detected=data.get("entities_detected"),
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoutingMetrics:
    """
    Metrics for routing performance tracking.
    """

    query: str
    decision: RoutingDecision
    execution_time: float
    success: bool
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "query": self.query,
            "decision": self.decision.to_dict(),
            "execution_time": self.execution_time,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class RoutingStrategy(ABC):
    """
    Abstract base class for routing strategies.
    Each strategy implements a different approach to query routing.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the routing strategy.

        Args:
            config: Strategy-specific configuration
        """
        self.config = config or {}
        self.metrics_history: list[RoutingMetrics] = []
        self.name = self.__class__.__name__

    @abstractmethod
    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """
        Route a query to determine search modality and generation type.

        Args:
            query: The user query to route
            context: Optional context information (conversation history, etc.)

        Returns:
            RoutingDecision with routing information
        """
        pass

    @abstractmethod
    def get_confidence(self, query: str, decision: RoutingDecision) -> float:
        """
        Calculate confidence score for a routing decision.

        Args:
            query: The user query
            decision: The routing decision made

        Returns:
            Confidence score between 0 and 1
        """
        pass

    def record_metrics(
        self,
        query: str,
        decision: RoutingDecision,
        execution_time: float,
        success: bool,
        error: str | None = None,
    ):
        """
        Record routing metrics for performance tracking.

        Args:
            query: The routed query
            decision: The routing decision made
            execution_time: Time taken to make the decision
            success: Whether routing was successful
            error: Error message if routing failed
        """
        metrics = RoutingMetrics(
            query=query,
            decision=decision,
            execution_time=execution_time,
            success=success,
            error=error,
        )
        self.metrics_history.append(metrics)

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for this strategy.

        Returns:
            Dictionary with performance metrics
        """
        if not self.metrics_history:
            return {"total_requests": 0}

        total = len(self.metrics_history)
        successful = sum(1 for m in self.metrics_history if m.success)
        avg_time = sum(m.execution_time for m in self.metrics_history) / total

        modality_distribution = {}
        for m in self.metrics_history:
            if m.success:
                modality = m.decision.search_modality.value
                modality_distribution[modality] = (
                    modality_distribution.get(modality, 0) + 1
                )

        return {
            "strategy_name": self.name,
            "total_requests": total,
            "successful_requests": successful,
            "success_rate": successful / total,
            "average_execution_time": avg_time,
            "modality_distribution": modality_distribution,
        }

    def export_metrics(self, filepath: str):
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to save metrics
        """
        metrics_data = [m.to_dict() for m in self.metrics_history]
        with open(filepath, "w") as f:
            json.dump(metrics_data, f, indent=2)

    def load_metrics(self, filepath: str):
        """
        Load metrics from a JSON file.

        Args:
            filepath: Path to load metrics from
        """
        with open(filepath) as f:
            metrics_data = json.load(f)

        self.metrics_history = []
        for data in metrics_data:
            decision_data = data.pop("decision")
            decision = RoutingDecision.from_dict(decision_data)
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            metrics = RoutingMetrics(decision=decision, **data)
            self.metrics_history.append(metrics)


class TemporalExtractor:
    """
    Utility class for extracting temporal information from queries.
    """

    @staticmethod
    def extract_temporal_info(query: str) -> dict[str, Any] | None:
        """
        Extract temporal information from a query.

        Args:
            query: The query to analyze

        Returns:
            Dictionary with temporal information or None
        """
        import re
        from datetime import date, timedelta

        today = date.today()
        temporal_info = {}
        query_lower = query.lower()

        # Common temporal patterns
        patterns = {
            "yesterday": (today - timedelta(days=1), today),
            "last week": (today - timedelta(weeks=1), today),
            "last month": (today - timedelta(days=30), today),
            "this week": (today - timedelta(days=today.weekday()), today),
            "this month": (today.replace(day=1), today),
            "today": (today, today),
            "last year": (today - timedelta(days=365), today),
            "this year": (date(today.year, 1, 1), today),
        }

        for pattern, (start_date, end_date) in patterns.items():
            if pattern in query_lower:
                temporal_info = {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "detected_pattern": pattern,
                }
                break

        # Look for specific date patterns (YYYY-MM-DD)
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        dates = re.findall(date_pattern, query)
        if dates:
            temporal_info["specific_dates"] = dates
            if len(dates) == 1:
                temporal_info["start_date"] = dates[0]
                temporal_info["end_date"] = dates[0]
            elif len(dates) >= 2:
                temporal_info["start_date"] = dates[0]
                temporal_info["end_date"] = dates[1]

        return temporal_info if temporal_info else None
