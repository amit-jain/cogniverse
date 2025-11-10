"""
Hybrid Reranking System

Combines heuristic multi-modal reranking with learned neural reranking.

Strategies:
- weighted_ensemble: Weighted combination of heuristic and learned scores
- cascade: Heuristic filtering followed by learned reranking
- consensus: Agreement-based ranking using both methods
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager

from cogniverse_foundation.config.utils import get_config_value

from cogniverse_agents.search.learned_reranker import LearnedReranker
from cogniverse_agents.search.multi_modal_reranker import (
    MultiModalReranker,
    QueryModality,
    SearchResult,
)

logger = logging.getLogger(__name__)


class HybridReranker:
    """
    Hybrid reranker combining heuristic and learned approaches

    Configuration loaded from config.json under "reranking" section.
    """

    def __init__(
        self,
        heuristic_reranker: Optional[MultiModalReranker] = None,
        learned_reranker: Optional[LearnedReranker] = None,
        strategy: Optional[str] = None,
        learned_weight: Optional[float] = None,
        heuristic_weight: Optional[float] = None,
        tenant_id: str = "default",
        config_manager: "ConfigManager" = None,
    ):
        """
        Initialize hybrid reranker

        Args:
            heuristic_reranker: Heuristic multi-modal reranker (creates if None)
            learned_reranker: Learned reranker (creates if None)
            strategy: Fusion strategy (loads from config if None)
            learned_weight: Weight for learned scores (loads from config if None)
            heuristic_weight: Weight for heuristic scores (loads from config if None)
            tenant_id: Tenant identifier for config scoping
            config_manager: ConfigManager instance (required for dependency injection)

        Raises:
            ValueError: If config_manager is not provided
        """
        if config_manager is None:
            raise ValueError(
                "config_manager is required for HybridReranker. "
                "Pass create_default_config_manager() explicitly."
            )

        # Load config
        rerank_config = get_config_value("reranking", {}, tenant_id=tenant_id, config_manager=config_manager)

        # Initialize rerankers
        self.heuristic_reranker = heuristic_reranker or MultiModalReranker()
        self.learned_reranker = learned_reranker or LearnedReranker(tenant_id=tenant_id, config_manager=config_manager)

        # Load fusion settings from config
        self.strategy = strategy or rerank_config.get(
            "hybrid_strategy", "weighted_ensemble"
        )
        self.learned_weight = (
            learned_weight
            if learned_weight is not None
            else rerank_config.get("learned_weight", 0.7)
        )
        self.heuristic_weight = (
            heuristic_weight
            if heuristic_weight is not None
            else rerank_config.get("heuristic_weight", 0.3)
        )

        # Validate strategy
        valid_strategies = ["weighted_ensemble", "cascade", "consensus"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate weights
        total_weight = self.learned_weight + self.heuristic_weight
        if not (0.99 <= total_weight <= 1.01):
            logger.warning(
                f"Weights don't sum to 1.0 (got {total_weight}). "
                "Normalizing weights."
            )
            self.learned_weight /= total_weight
            self.heuristic_weight /= total_weight

        logger.info(
            f"Initialized HybridReranker with strategy={self.strategy}, "
            f"learned_weight={self.learned_weight}, "
            f"heuristic_weight={self.heuristic_weight}"
        )

    async def rerank_hybrid(
        self,
        query: str,
        results: List[SearchResult],
        modalities: List[QueryModality],
        context: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Hybrid reranking combining heuristic and learned approaches

        Args:
            query: Search query
            results: Results to rerank
            modalities: Query modalities
            context: Additional context

        Returns:
            Reranked results with combined scores
        """
        if not results:
            return []

        if self.strategy == "weighted_ensemble":
            return await self._weighted_ensemble(query, results, modalities, context)
        elif self.strategy == "cascade":
            return await self._cascade(query, results, modalities, context)
        elif self.strategy == "consensus":
            return await self._consensus(query, results, modalities, context)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _weighted_ensemble(
        self,
        query: str,
        results: List[SearchResult],
        modalities: List[QueryModality],
        context: Optional[Dict],
    ) -> List[SearchResult]:
        """
        Combine scores from both rerankers with weights

        Both rerankers run in parallel, then scores are combined.
        """
        # Get heuristic scores
        heuristic_results = await self.heuristic_reranker.rerank_results(
            results, query, modalities, context
        )
        heuristic_scores = {
            r.id: r.metadata.get("reranking_score", 0.0) for r in heuristic_results
        }

        # Get learned scores
        learned_results = await self.learned_reranker.rerank(query, results)
        learned_scores = {
            r.id: r.metadata.get("reranking_score", 0.0) for r in learned_results
        }

        # Combine scores
        final_results = []
        for result in results:
            h_score = heuristic_scores.get(result.id, 0.0)
            l_score = learned_scores.get(result.id, 0.0)

            final_score = (
                h_score * self.heuristic_weight + l_score * self.learned_weight
            )

            result.metadata["reranking_score"] = final_score
            result.metadata["heuristic_score"] = h_score
            result.metadata["learned_score"] = l_score
            result.metadata["fusion_strategy"] = "weighted_ensemble"
            final_results.append(result)

        # Sort by final score
        final_results.sort(
            key=lambda x: x.metadata["reranking_score"], reverse=True
        )

        return final_results

    async def _cascade(
        self,
        query: str,
        results: List[SearchResult],
        modalities: List[QueryModality],
        context: Optional[Dict],
    ) -> List[SearchResult]:
        """
        Use heuristic for filtering, learned for final ranking

        More efficient as it reduces expensive learned model calls.
        """
        # Step 1: Heuristic filtering (keep top 50% or min 10)
        heuristic_results = await self.heuristic_reranker.rerank_results(
            results, query, modalities, context
        )

        top_k = max(10, len(results) // 2)
        filtered = heuristic_results[:top_k]

        # Step 2: Learned reranking on filtered set
        final_results = await self.learned_reranker.rerank(query, filtered)

        # Add fusion metadata
        for result in final_results:
            result.metadata["fusion_strategy"] = "cascade"

        return final_results

    async def _consensus(
        self,
        query: str,
        results: List[SearchResult],
        modalities: List[QueryModality],
        context: Optional[Dict],
    ) -> List[SearchResult]:
        """
        Require agreement between both methods

        Results must rank highly in BOTH heuristic and learned to be top-ranked.
        Uses Borda count for consensus scoring.
        """
        # Get both rankings
        heuristic_results = await self.heuristic_reranker.rerank_results(
            results, query, modalities, context
        )
        learned_results = await self.learned_reranker.rerank(query, results)

        # Create rank maps
        heuristic_ranks = {r.id: idx for idx, r in enumerate(heuristic_results)}
        learned_ranks = {r.id: idx for idx, r in enumerate(learned_results)}

        # Compute consensus scores using Borda count
        # Lower rank number = better position
        consensus_scores = {}
        for result in results:
            h_rank = heuristic_ranks.get(result.id, len(results))
            l_rank = learned_ranks.get(result.id, len(results))

            # Convert ranks to scores (higher is better)
            h_score = len(results) - h_rank
            l_score = len(results) - l_rank

            # Geometric mean emphasizes agreement
            consensus_score = (h_score * l_score) ** 0.5

            consensus_scores[result.id] = consensus_score

        # Sort by consensus score
        for result in results:
            result.metadata["reranking_score"] = consensus_scores[result.id]
            result.metadata["heuristic_rank"] = heuristic_ranks.get(
                result.id, len(results)
            )
            result.metadata["learned_rank"] = learned_ranks.get(
                result.id, len(results)
            )
            result.metadata["fusion_strategy"] = "consensus"

        results.sort(key=lambda x: x.metadata["reranking_score"], reverse=True)

        return results
