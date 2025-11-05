"""
Multi-Modal Reranking

Reranks search results considering multiple modalities and cross-modal relevance.
Improves result quality by analyzing complementarity, temporal alignment, and
modality-query alignment.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_core.config.manager import ConfigManager


class QueryModality(Enum):
    """Query modality types"""

    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    TEXT = "text"
    MIXED = "mixed"


@dataclass
class SearchResult:
    """
    Generic search result from any modality

    Attributes:
        id: Unique result identifier
        title: Result title
        content: Result content/description
        modality: Source modality
        score: Original search score
        metadata: Additional result metadata
        timestamp: Optional temporal information
    """

    id: str
    title: str
    content: str
    modality: str
    score: float
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None


class MultiModalReranker:
    """
    Rerank results considering multiple modalities

    Features:
    - Cross-modal relevance scoring
    - Modality-query alignment
    - Temporal alignment for time-sensitive queries
    - Result complementarity analysis
    - Diversity-aware ranking
    """

    def __init__(
        self,
        cross_modal_weight: float = 0.3,
        temporal_weight: float = 0.2,
        complementarity_weight: float = 0.2,
        diversity_weight: float = 0.15,
        original_score_weight: float = 0.15,
    ):
        """
        Initialize reranker with scoring weights

        Args:
            cross_modal_weight: Weight for cross-modal relevance
            temporal_weight: Weight for temporal alignment
            complementarity_weight: Weight for result complementarity
            diversity_weight: Weight for diversity bonus
            original_score_weight: Weight for original search score
        """
        self.cross_modal_weight = cross_modal_weight
        self.temporal_weight = temporal_weight
        self.complementarity_weight = complementarity_weight
        self.diversity_weight = diversity_weight
        self.original_score_weight = original_score_weight

        # Ensure weights sum to 1.0
        total_weight = (
            cross_modal_weight
            + temporal_weight
            + complementarity_weight
            + diversity_weight
            + original_score_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    async def rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        modalities: List[QueryModality],
        context: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Rerank results based on multi-modal factors

        Args:
            results: List of search results from various modalities
            query: Original user query
            modalities: Detected query modalities
            context: Optional context (temporal, user preferences, etc.)

        Returns:
            Reranked list of search results
        """
        if not results:
            return []

        context = context or {}
        reranked = []

        # Calculate scores for each result
        for result in results:
            score_components = {
                "cross_modal": self._calculate_cross_modal_score(
                    result, query, modalities
                ),
                "temporal": self._calculate_temporal_score(result, context),
                "complementarity": self._calculate_complementarity_score(
                    result, results
                ),
                "diversity": self._calculate_diversity_score(result, reranked),
                "original": result.score,
            }

            # Calculate weighted final score
            final_score = (
                score_components["cross_modal"] * self.cross_modal_weight
                + score_components["temporal"] * self.temporal_weight
                + score_components["complementarity"] * self.complementarity_weight
                + score_components["diversity"] * self.diversity_weight
                + score_components["original"] * self.original_score_weight
            )

            reranked.append((final_score, result, score_components))

        # Sort by final score (descending)
        reranked.sort(reverse=True, key=lambda x: x[0])

        # Return results with updated scores
        final_results = []
        for final_score, result, components in reranked:
            # Update result metadata with scoring details
            result.metadata["reranking_score"] = final_score
            result.metadata["score_components"] = components
            final_results.append(result)

        return final_results

    def _calculate_cross_modal_score(
        self,
        result: SearchResult,
        query: str,
        modalities: List[QueryModality],
    ) -> float:
        """
        Calculate cross-modal relevance score

        Measures how well the result's modality aligns with query modalities
        """
        if not modalities:
            return 0.5  # Neutral score

        # Convert result modality to QueryModality
        try:
            result_modality = QueryModality(result.modality.lower())
        except ValueError:
            result_modality = QueryModality.TEXT

        # Direct match bonus
        if result_modality in modalities:
            return 1.0

        # Partial match for related modalities
        modality_compatibility = {
            QueryModality.VIDEO: [QueryModality.IMAGE, QueryModality.AUDIO],
            QueryModality.IMAGE: [QueryModality.VIDEO],
            QueryModality.AUDIO: [QueryModality.VIDEO],
            QueryModality.DOCUMENT: [QueryModality.TEXT],
            QueryModality.TEXT: [QueryModality.DOCUMENT],
        }

        compatible_modalities = modality_compatibility.get(result_modality, [])
        if any(m in modalities for m in compatible_modalities):
            return 0.7

        # Mixed query accepts all modalities
        if QueryModality.MIXED in modalities:
            return 0.8

        return 0.3  # Low score for unrelated modalities

    def _calculate_temporal_score(
        self,
        result: SearchResult,
        context: Dict,
    ) -> float:
        """
        Calculate temporal alignment score

        Measures how well result timestamp aligns with query temporal context
        """
        if not result.timestamp or "temporal" not in context:
            return 0.5  # Neutral score if no temporal info

        temporal_context = context["temporal"]
        time_range = temporal_context.get("time_range")

        if not time_range:
            return 0.5

        start_time, end_time = time_range

        # Check if result falls within time range
        if start_time <= result.timestamp <= end_time:
            # Calculate how centered the result is in the range
            range_duration = (end_time - start_time).total_seconds()
            if range_duration > 0:
                offset_from_start = (result.timestamp - start_time).total_seconds()
                centrality = 1.0 - abs(0.5 - (offset_from_start / range_duration)) * 2
                return 0.7 + (centrality * 0.3)  # 0.7-1.0 range
            return 1.0

        # Result outside range - penalize based on distance
        if result.timestamp < start_time:
            days_before = (start_time - result.timestamp).days
        else:
            days_before = (result.timestamp - end_time).days

        # Exponential decay
        if days_before < 30:
            return 0.5
        elif days_before < 90:
            return 0.3
        elif days_before < 365:
            return 0.2
        else:
            return 0.1

    def _calculate_complementarity_score(
        self,
        result: SearchResult,
        all_results: List[SearchResult],
    ) -> float:
        """
        Calculate how well this result complements others

        Measures unique information contribution
        """
        if len(all_results) <= 1:
            return 1.0

        # Simple keyword-based complementarity
        result_keywords = set(
            result.content.lower().split() + result.title.lower().split()
        )

        # Calculate average keyword overlap with other results
        overlaps = []
        for other in all_results:
            if other.id == result.id:
                continue

            other_keywords = set(
                other.content.lower().split() + other.title.lower().split()
            )

            if len(result_keywords) > 0 and len(other_keywords) > 0:
                overlap = len(result_keywords & other_keywords)
                union = len(result_keywords | other_keywords)
                overlaps.append(overlap / union if union > 0 else 0)

        if not overlaps:
            return 1.0

        avg_overlap = sum(overlaps) / len(overlaps)

        # Low overlap means high complementarity
        return 1.0 - avg_overlap

    def _calculate_diversity_score(
        self,
        result: SearchResult,
        selected_results: List[tuple],
    ) -> float:
        """
        Calculate diversity bonus

        Rewards results from underrepresented modalities
        """
        if not selected_results:
            return 1.0  # First result gets max diversity

        # Count modalities in already selected results
        modality_counts = {}
        for _, selected_result, _ in selected_results:
            modality = selected_result.modality
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        # Give bonus to underrepresented modalities
        result_modality = result.modality
        count = modality_counts.get(result_modality, 0)

        # Diversity score decreases with count
        if count == 0:
            return 1.0
        elif count == 1:
            return 0.8
        elif count == 2:
            return 0.6
        elif count == 3:
            return 0.4
        else:
            return 0.2

    def get_modality_distribution(
        self, results: List[SearchResult]
    ) -> Dict[str, int]:
        """
        Get distribution of modalities in results

        Args:
            results: List of search results

        Returns:
            Dictionary mapping modality to count
        """
        distribution = {}
        for result in results:
            modality = result.modality
            distribution[modality] = distribution.get(modality, 0) + 1
        return distribution

    def analyze_ranking_quality(
        self, results: List[SearchResult]
    ) -> Dict[str, Any]:
        """
        Analyze quality metrics of ranked results

        Args:
            results: Ranked search results

        Returns:
            Dictionary with quality metrics
        """
        if not results:
            return {
                "diversity": 0.0,
                "average_score": 0.0,
                "modality_distribution": {},
                "temporal_coverage": 0.0,
            }

        # Calculate diversity (normalized entropy)
        distribution = self.get_modality_distribution(results)
        total = len(results)
        entropy = 0.0
        for count in distribution.values():
            p = count / total
            if p > 0:
                entropy += -p * (p if p == 0 else __import__("math").log2(p))

        max_entropy = __import__("math").log2(len(distribution)) if distribution else 1
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        # Calculate average reranking score
        avg_score = sum(
            r.metadata.get("reranking_score", 0.0) for r in results
        ) / len(results)

        # Calculate temporal coverage
        timestamps = [r.timestamp for r in results if r.timestamp]
        if len(timestamps) > 1:
            time_range = (max(timestamps) - min(timestamps)).days
            temporal_coverage = min(1.0, time_range / 365)  # Normalize to year
        else:
            temporal_coverage = 0.0

        return {
            "diversity": diversity,
            "average_score": avg_score,
            "modality_distribution": distribution,
            "temporal_coverage": temporal_coverage,
            "total_results": len(results),
        }


class ConfigurableMultiModalReranker:
    """
    Multi-modal reranker with configurable backend

    Supports three modes based on config.json settings:
    - Pure heuristic (default MultiModalReranker)
    - Pure learned (LiteLLM-based neural reranker)
    - Hybrid (combination of both)

    Configuration loaded from config.json under "reranking" section.
    """

    def __init__(self, tenant_id: str = "default", config_manager: "ConfigManager" = None):
        """
        Initialize configurable reranker from config.json

        Args:
            tenant_id: Tenant identifier for config scoping
            config_manager: ConfigManager instance (required for dependency injection)

        Raises:
            ValueError: If config_manager is not provided
        """
        import logging

        from cogniverse_core.config.utils import get_config_value

        from cogniverse_agents.search.hybrid_reranker import HybridReranker
        from cogniverse_agents.search.learned_reranker import LearnedReranker

        if config_manager is None:
            raise ValueError(
                "config_manager is required for ConfigurableMultiModalReranker. "
                "Pass ConfigManager() explicitly."
            )

        logger = logging.getLogger(__name__)

        # Store for later use
        self.tenant_id = tenant_id
        self.config_manager = config_manager

        # Load config
        rerank_config = get_config_value("reranking", {}, tenant_id=tenant_id, config_manager=config_manager)
        self.enabled = rerank_config.get("enabled", False)
        model_key = rerank_config.get("model", "heuristic")
        use_hybrid = rerank_config.get("use_hybrid", False)

        # Initialize heuristic reranker (always available)
        self.heuristic_reranker = MultiModalReranker()

        # Initialize learned reranker if enabled and not heuristic
        self.learned_reranker: Optional[LearnedReranker] = None
        if self.enabled and model_key != "heuristic":
            try:
                self.learned_reranker = LearnedReranker(tenant_id=tenant_id, config_manager=config_manager)
                logger.info("Initialized learned reranker")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize learned reranker: {e}. "
                    "Falling back to heuristic reranking."
                )
                self.learned_reranker = None

        # Initialize hybrid reranker if configured
        self.hybrid_reranker: Optional[HybridReranker] = None
        if self.enabled and use_hybrid and self.learned_reranker:
            try:
                self.hybrid_reranker = HybridReranker(
                    heuristic_reranker=self.heuristic_reranker,
                    learned_reranker=self.learned_reranker,
                    tenant_id=tenant_id,
                    config_manager=config_manager,
                )
                logger.info("Initialized hybrid reranker")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid reranker: {e}")
                self.hybrid_reranker = None

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        modalities: List[QueryModality],
        context: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Rerank using configured strategy

        Args:
            query: User query
            results: Search results to rerank
            modalities: Detected query modalities
            context: Optional context

        Returns:
            Reranked results
        """
        if not results:
            return []

        # If reranking disabled, return original results
        if not self.enabled:
            return results

        # Route to appropriate reranker
        if self.hybrid_reranker:
            # Hybrid: combine heuristic and learned
            return await self.hybrid_reranker.rerank_hybrid(
                query, results, modalities, context
            )
        elif self.learned_reranker:
            # Pure learned: use LiteLLM neural reranker
            return await self.learned_reranker.rerank(query, results)
        else:
            # Pure heuristic: use multi-modal logic
            return await self.heuristic_reranker.rerank_results(
                results, query, modalities, context
            )

    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about active reranker configuration

        Returns:
            Dictionary with reranker details
        """
        from cogniverse_core.config.utils import get_config_value

        rerank_config = get_config_value("reranking", {}, tenant_id=self.tenant_id, config_manager=self.config_manager)

        return {
            "enabled": self.enabled,
            "model": rerank_config.get("model", "heuristic"),
            "use_hybrid": rerank_config.get("use_hybrid", False),
            "hybrid_strategy": rerank_config.get("hybrid_strategy"),
            "learned_available": self.learned_reranker is not None,
            "hybrid_available": self.hybrid_reranker is not None,
            "max_results_to_rerank": rerank_config.get("max_results_to_rerank", 100),
        }
