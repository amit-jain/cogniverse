"""
Lazy Modality Execution

Execute expensive modalities only when needed based on cost/benefit analysis.
Part of Phase 12: Production Readiness.
"""

import logging
from typing import Any, Callable, Dict, List

from cogniverse_agents.search.rerankers.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class LazyModalityExecutor:
    """
    Lazily evaluate expensive modalities only when needed

    Strategy:
    1. Execute modalities in order of cost (fast → slow)
    2. Evaluate if expensive modalities needed after each step
    3. Only execute expensive ones if required

    Features:
    - Cost-based ordering
    - Early stopping when sufficient results
    - Quality threshold evaluation
    - Execution tracking

    Example:
        executor = LazyModalityExecutor()

        results = await executor.execute_with_lazy_evaluation(
            query="machine learning tutorials",
            modalities=[QueryModality.VIDEO, QueryModality.TEXT],
            context={"quality_threshold": 0.8},
            modality_executor=my_executor_func
        )
    """

    # Cost ranking (fast → slow)
    # Lower cost = faster execution
    MODALITY_COST = {
        QueryModality.TEXT: 1,
        QueryModality.DOCUMENT: 2,
        QueryModality.IMAGE: 5,
        QueryModality.VIDEO: 8,
        QueryModality.AUDIO: 10,
        QueryModality.MIXED: 15,  # Requires multiple modalities
    }

    def __init__(self, default_quality_threshold: float = 0.8):
        """
        Initialize lazy executor

        Args:
            default_quality_threshold: Default quality threshold for early stopping
        """
        self.default_quality_threshold = default_quality_threshold
        self.execution_stats = {
            "total_executions": 0,
            "early_stops": 0,
            "full_executions": 0,
            "modalities_skipped": 0,
        }

        logger.info(
            f"⚡ Initialized LazyModalityExecutor "
            f"(quality_threshold: {default_quality_threshold})"
        )

    async def execute_with_lazy_evaluation(
        self,
        query: str,
        modalities: List[QueryModality],
        context: Dict[str, Any],
        modality_executor: Callable,
    ) -> Dict[str, Any]:
        """
        Execute modalities with lazy evaluation

        Args:
            query: Query string
            modalities: List of modalities to execute
            context: Execution context
            modality_executor: Async function to execute modality
                               signature: (query, modality, context) -> result

        Returns:
            {
                "results": {modality: result},
                "executed_modalities": [modality1, modality2],
                "skipped_modalities": [modality3],
                "early_stopped": bool,
                "total_cost": int,
            }
        """
        self.execution_stats["total_executions"] += 1

        quality_threshold = context.get(
            "quality_threshold", self.default_quality_threshold
        )

        # Sort modalities by cost (cheap first)
        sorted_modalities = sorted(
            modalities, key=lambda m: self.MODALITY_COST.get(m, 100)
        )

        logger.info(
            f"🚀 Lazy execution order: " f"{[m.value for m in sorted_modalities]}"
        )

        results = {}
        executed_modalities = []
        skipped_modalities = []
        total_cost = 0
        early_stopped = False

        for modality in sorted_modalities:
            modality_cost = self.MODALITY_COST.get(modality, 100)

            # Execute modality
            logger.info(f"▶️ Executing {modality.value} (cost: {modality_cost})")

            result = await modality_executor(query, modality, context)
            results[modality] = result
            executed_modalities.append(modality)
            total_cost += modality_cost

            # Check if we have sufficient results
            if self._is_sufficient(results, quality_threshold, context):
                # Calculate remaining modalities
                remaining = sorted_modalities[len(executed_modalities) :]
                skipped_modalities = remaining

                if skipped_modalities:
                    early_stopped = True
                    self.execution_stats["early_stops"] += 1
                    self.execution_stats["modalities_skipped"] += len(
                        skipped_modalities
                    )

                    logger.info(
                        f"✋ Early stop: Sufficient results after {modality.value}, "
                        f"skipping {[m.value for m in skipped_modalities]}"
                    )
                    break
        else:
            # Executed all modalities
            self.execution_stats["full_executions"] += 1

        logger.info(
            f"✅ Lazy execution complete: "
            f"{len(executed_modalities)} executed, "
            f"{len(skipped_modalities)} skipped "
            f"(total_cost: {total_cost})"
        )

        return {
            "results": results,
            "executed_modalities": [m.value for m in executed_modalities],
            "skipped_modalities": [m.value for m in skipped_modalities],
            "early_stopped": early_stopped,
            "total_cost": total_cost,
        }

    def _is_sufficient(
        self,
        results: Dict[QueryModality, Any],
        quality_threshold: float,
        context: Dict[str, Any],
    ) -> bool:
        """
        Determine if results are sufficient

        Args:
            results: Current results
            quality_threshold: Quality threshold
            context: Execution context

        Returns:
            True if results are sufficient
        """
        if not results:
            return False

        # Strategy 1: Check if we have minimum number of results
        min_results = context.get("min_results_required", 5)
        total_result_count = sum(
            self._count_results(result) for result in results.values()
        )

        if total_result_count < min_results:
            logger.debug(f"Insufficient count: {total_result_count} < {min_results}")
            return False

        # Strategy 2: Check if we have high confidence results
        avg_confidence = self._calculate_avg_confidence(results)
        if avg_confidence < quality_threshold:
            logger.debug(
                f"Insufficient quality: {avg_confidence:.2f} < {quality_threshold}"
            )
            return False

        # Strategy 3: Check if we already executed expensive modality
        # If we've executed VIDEO or AUDIO, we probably have good results
        expensive_modalities = {QueryModality.VIDEO, QueryModality.AUDIO}
        if any(m in results for m in expensive_modalities):
            logger.debug("Executed expensive modality - considering sufficient")
            return True

        logger.debug(
            f"Results sufficient: count={total_result_count}, quality={avg_confidence:.2f}"
        )
        return True

    def _count_results(self, result: Any) -> int:
        """
        Count number of results in a modality result

        Args:
            result: Modality result

        Returns:
            Number of results
        """
        if isinstance(result, list):
            return len(result)
        elif isinstance(result, dict):
            # Check common result keys
            if "results" in result:
                return len(result["results"])
            if "items" in result:
                return len(result["items"])
            # Assume dict with results
            return len(result)
        else:
            return 1 if result else 0

    def _calculate_avg_confidence(self, results: Dict[QueryModality, Any]) -> float:
        """
        Calculate average confidence across results

        Args:
            results: Modality results

        Returns:
            Average confidence score (0-1)
        """
        confidences = []

        for modality_result in results.values():
            if isinstance(modality_result, dict):
                # Check for confidence field
                if "confidence" in modality_result:
                    confidences.append(modality_result["confidence"])
                elif "results" in modality_result:
                    # Extract confidence from result items
                    for item in modality_result["results"]:
                        if isinstance(item, dict) and "confidence" in item:
                            confidences.append(item["confidence"])
                        elif isinstance(item, dict) and "score" in item:
                            confidences.append(item["score"])

            elif isinstance(modality_result, list):
                # List of result items
                for item in modality_result:
                    if isinstance(item, dict):
                        if "confidence" in item:
                            confidences.append(item["confidence"])
                        elif "score" in item:
                            confidences.append(item["score"])

        if not confidences:
            # No confidence found, assume moderate confidence
            return 0.7

        return sum(confidences) / len(confidences)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics

        Returns:
            Execution statistics
        """
        stats = self.execution_stats.copy()

        if stats["total_executions"] > 0:
            stats["early_stop_rate"] = stats["early_stops"] / stats["total_executions"]
            stats["avg_modalities_skipped"] = (
                stats["modalities_skipped"] / stats["total_executions"]
            )
        else:
            stats["early_stop_rate"] = 0.0
            stats["avg_modalities_skipped"] = 0.0

        return stats

    def reset_stats(self):
        """Reset statistics"""
        self.execution_stats = {
            "total_executions": 0,
            "early_stops": 0,
            "full_executions": 0,
            "modalities_skipped": 0,
        }
        logger.info("📊 Reset lazy executor statistics")
