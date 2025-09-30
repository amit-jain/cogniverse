# src/routing/router.py
"""
Comprehensive Router implementation with tiered architecture.
Implements the hybrid approach described in COMPREHENSIVE_ROUTING.md.
"""

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base import (
    GenerationType,
    RoutingDecision,
    RoutingMetrics,
    RoutingStrategy,
    SearchModality,
)
from .config_validator import RouterConfigValidator
from .strategies import (
    GLiNERRoutingStrategy,
    KeywordRoutingStrategy,
    LangExtractRoutingStrategy,
    LLMRoutingStrategy,
)

logger = logging.getLogger(__name__)


class RoutingTier(Enum):
    """Routing tier levels for the hybrid architecture."""

    FAST_PATH = "fast_path"  # GLiNER2 for common patterns
    SLOW_PATH = "slow_path"  # SmolLM3 + DSPy for complex queries
    LANGEXTRACT = "langextract"  # LangExtract for structured extraction
    FALLBACK = "fallback"  # Keyword-based as ultimate fallback


@dataclass
class RouterConfig:
    """Configuration for the comprehensive router."""

    # Tier configuration
    enable_fast_path: bool = True
    enable_slow_path: bool = True
    enable_langextract: bool = True
    enable_fallback: bool = True

    # Confidence thresholds for tier escalation
    fast_path_confidence_threshold: float = 0.7
    slow_path_confidence_threshold: float = 0.6
    langextract_confidence_threshold: float = 0.5

    # Performance settings
    max_routing_time_ms: int = 1000  # Maximum time for routing decision
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

    # Metrics and monitoring
    enable_metrics: bool = True
    metrics_batch_size: int = 100

    # Strategy configurations
    gliner_config: dict[str, Any] | None = None
    llm_config: dict[str, Any] | None = None
    langextract_config: dict[str, Any] | None = None
    keyword_config: dict[str, Any] | None = None

    # Auto-optimization settings
    enable_auto_optimization: bool = True
    optimization_threshold: int = 1000  # Number of queries before optimization
    min_accuracy_threshold: float = 0.8  # Minimum acceptable accuracy


class ComprehensiveRouter:
    """
    Implements a tiered, hybrid routing architecture as described in COMPREHENSIVE_ROUTING.md.

    Architecture:
    - Tier 1 (Fast Path): GLiNER2 for high-speed, low-cost extraction
    - Tier 2 (Slow Path): SmolLM3 + DSPy for complex reasoning
    - Tier 3 (Fallback): Keyword-based for ultimate reliability
    """

    def __init__(self, config: RouterConfig | None = None):
        """
        Initialize the comprehensive router.

        Args:
            config: Router configuration
        """
        self.config = config or RouterConfig()

        # Validate ensemble configuration if present
        ensemble_config = self._get_ensemble_config()
        RouterConfigValidator.validate_ensemble_config(ensemble_config)

        self.strategies: dict[RoutingTier, RoutingStrategy] = {}
        self._initialize_strategies()
        self.cache: dict[str, tuple[RoutingDecision, float]] = {}
        self.metrics_buffer: list[RoutingMetrics] = []
        self.query_count = 0
        self.optimizer = None  # Placeholder for testing

    def _initialize_strategies(self):
        """Initialize routing strategies for each tier."""
        # Handle both RouterConfig and RoutingConfig
        if isinstance(self.config, dict):
            tier_config = self.config.get("tier_config", {})
            gliner_cfg = self.config.get("gliner_config", {})
            llm_cfg = self.config.get("llm_config", {})
            langextract_cfg = self.config.get("langextract_config", {})
            keyword_cfg = self.config.get("keyword_config", {})
        elif hasattr(self.config, "tier_config"):
            tier_config = self.config.tier_config
            gliner_cfg = self.config.gliner_config
            llm_cfg = self.config.llm_config
            langextract_cfg = self.config.langextract_config
            keyword_cfg = self.config.keyword_config
        else:
            tier_config = {
                "enable_fast_path": getattr(self.config, "enable_fast_path", True),
                "enable_slow_path": getattr(self.config, "enable_slow_path", True),
                "enable_langextract": getattr(self.config, "enable_langextract", True),
                "enable_fallback": getattr(self.config, "enable_fallback", True),
            }
            gliner_cfg = getattr(self.config, "gliner_config", {})
            llm_cfg = getattr(self.config, "llm_config", {})
            langextract_cfg = getattr(self.config, "langextract_config", {})
            keyword_cfg = getattr(self.config, "keyword_config", {})

        # Fast Path: GLiNER2
        if tier_config.get("enable_fast_path", True):
            self.strategies[RoutingTier.FAST_PATH] = GLiNERRoutingStrategy(gliner_cfg)
            logger.info("Initialized Fast Path (GLiNER) routing strategy")

        # Slow Path: LLM (SmolLM3 or other)
        if tier_config.get("enable_slow_path", True):
            self.strategies[RoutingTier.SLOW_PATH] = LLMRoutingStrategy(llm_cfg)
            logger.info("Initialized Slow Path (LLM) routing strategy")

        # LangExtract Path: Structured extraction
        if tier_config.get("enable_langextract", True):
            self.strategies[RoutingTier.LANGEXTRACT] = LangExtractRoutingStrategy(
                langextract_cfg
            )
            logger.info("Initialized LangExtract routing strategy")

        # Fallback: Keyword-based
        if tier_config.get("enable_fallback", True):
            self.strategies[RoutingTier.FALLBACK] = KeywordRoutingStrategy(keyword_cfg)
            logger.info("Initialized Fallback (Keyword) routing strategy")

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """
        Route a query through either ensemble or tiered architecture.

        Args:
            query: The user query to route
            context: Optional context information

        Returns:
            RoutingDecision with routing information
        """
        start_time = time.time()
        self.query_count += 1

        # Check cache first
        if hasattr(self.config, "cache_config"):
            enable_caching = self.config.cache_config.get("enable_caching", True)
        else:
            enable_caching = True  # Default to enabled

        if enable_caching:
            cached_decision = self._check_cache(query)
            if cached_decision:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_decision

        # Check if ensemble mode is enabled
        ensemble_config = self._get_ensemble_config()
        if ensemble_config and ensemble_config.get("enabled", False):
            logger.debug(f"Using ensemble routing for: {query[:50]}...")
            return await self._ensemble_route(query, context, start_time)

        # Fall back to tiered routing
        return await self._tiered_route(query, context, start_time)

    def _get_ensemble_config(self) -> dict[str, Any] | None:
        """Get ensemble configuration."""
        if isinstance(self.config, dict):
            return self.config.get("ensemble_config")
        else:
            return getattr(self.config, "ensemble_config", None)

    async def _ensemble_route(
        self, query: str, context: dict[str, Any] | None, start_time: float
    ) -> RoutingDecision:
        """
        Route using ensemble method - run multiple strategies in parallel and vote.

        Args:
            query: The user query
            context: Optional context
            start_time: Start time for metrics

        Returns:
            RoutingDecision from ensemble voting
        """
        ensemble_config = self._get_ensemble_config()
        enabled_strategies = ensemble_config.get("enabled_strategies", [])
        voting_method = ensemble_config.get("voting_method", "weighted")
        min_agreement = ensemble_config.get("min_agreement", 0.5)
        strategy_weights = ensemble_config.get("strategy_weights", {})

        logger.debug(
            f"Ensemble routing with {len(enabled_strategies)} strategies: {enabled_strategies}"
        )

        # Map strategy names to tiers
        strategy_tier_map = {
            "gliner": RoutingTier.FAST_PATH,
            "llm": RoutingTier.SLOW_PATH,
            "langextract": RoutingTier.LANGEXTRACT,
            "keyword": RoutingTier.FALLBACK,
        }

        # Create async tasks for enabled strategies
        tasks = []
        for strategy_name in enabled_strategies:
            tier = strategy_tier_map.get(strategy_name)
            if tier and tier in self.strategies:
                task = asyncio.create_task(
                    self._safe_strategy_call(
                        self.strategies[tier], query, context, strategy_name
                    )
                )
                tasks.append((strategy_name, task))

        if not tasks:
            logger.warning(
                f"No valid strategies found for ensemble routing. Enabled: {enabled_strategies}"
            )
            return await self._tiered_route(query, context, start_time)

        # Wait for all strategies to complete (with timeout)
        timeout = ensemble_config.get("timeout_seconds", 10.0)
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Ensemble routing timeout after {timeout}s")
            return await self._tiered_route(query, context, start_time)

        # Collect successful decisions
        decisions = []
        for i, (strategy_name, _) in enumerate(tasks):
            result = completed_tasks[i]
            if isinstance(result, RoutingDecision):
                decisions.append((strategy_name, result))
            elif isinstance(result, Exception):
                logger.warning(f"Strategy {strategy_name} failed: {result}")

        if not decisions:
            logger.warning("All ensemble strategies failed, falling back to tiered")
            return await self._tiered_route(query, context, start_time)

        # Vote on decisions
        final_decision = self._vote_on_decisions(
            decisions, voting_method, strategy_weights, min_agreement
        )

        # Update cache and metrics
        self._update_cache(query, final_decision)
        self._record_routing_metrics(
            query, final_decision, time.time() - start_time, None
        )

        # Add ensemble metadata
        final_decision.metadata.update(
            {
                "routing_method": "ensemble",
                "voting_method": voting_method,
                "strategies_used": [name for name, _ in decisions],
                "num_strategies": len(decisions),
            }
        )

        return final_decision

    async def _safe_strategy_call(
        self, strategy, query: str, context: dict[str, Any] | None, strategy_name: str
    ) -> RoutingDecision:
        """Safely call a strategy with error handling."""
        try:
            decision = await strategy.route(query, context)
            decision.metadata["strategy_name"] = strategy_name
            return decision
        except Exception as e:
            logger.warning(f"Strategy {strategy_name} failed: {e}")
            raise e

    def _vote_on_decisions(
        self,
        decisions: list[tuple[str, RoutingDecision]],
        voting_method: str,
        strategy_weights: dict[str, float],
        min_agreement: float,
    ) -> RoutingDecision:
        """
        Vote on routing decisions from multiple strategies.

        Args:
            decisions: List of (strategy_name, decision) tuples
            voting_method: "majority", "weighted", or "confidence_weighted"
            strategy_weights: Manual weights per strategy
            min_agreement: Minimum agreement threshold

        Returns:
            Final RoutingDecision
        """
        if not decisions:
            raise ValueError("No decisions to vote on")

        if len(decisions) == 1:
            # Even with one decision, mark it as ensemble
            single_decision = decisions[0][1]
            original_method = single_decision.routing_method
            single_decision.routing_method = "ensemble"
            single_decision.metadata.update(
                {
                    "original_routing_method": original_method,
                    "ensemble_size": 1,
                }
            )
            return single_decision

        # Extract decision components
        search_modalities = []
        generation_types = []
        confidence_scores = []
        reasonings = []

        for strategy_name, decision in decisions:
            weight = strategy_weights.get(strategy_name, 1.0)

            # Weight the votes
            for _ in range(int(weight * 10)):  # Scale weights to integers
                search_modalities.append(decision.search_modality)
                generation_types.append(decision.generation_type)

            confidence_scores.append(decision.confidence_score * weight)
            reasonings.append(f"{strategy_name}: {decision.reasoning}")

        # Vote on search modality
        modality_votes = Counter(search_modalities)
        final_modality = modality_votes.most_common(1)[0][0]

        # Check agreement threshold
        modality_agreement = modality_votes.most_common(1)[0][1] / len(
            search_modalities
        )
        if modality_agreement < min_agreement:
            logger.warning(
                f"Low agreement on search modality: {modality_agreement:.2f}"
            )
            final_modality = SearchModality.BOTH  # Default to both when uncertain

        # Vote on generation type
        generation_votes = Counter(generation_types)
        final_generation = generation_votes.most_common(1)[0][0]

        # Calculate confidence score
        if voting_method == "confidence_weighted":
            # Weight by individual confidence scores
            total_weight = sum(d.confidence_score for _, d in decisions)
            final_confidence = (
                sum(d.confidence_score**2 for _, d in decisions) / total_weight
                if total_weight > 0
                else 0.0
            )
        elif voting_method == "weighted":
            # Use manual weights
            total_weight = sum(strategy_weights.get(name, 1.0) for name, _ in decisions)
            final_confidence = (
                sum(confidence_scores) / total_weight if total_weight > 0 else 0.0
            )
        else:  # majority
            final_confidence = sum(d.confidence_score for _, d in decisions) / len(
                decisions
            )

        # Combine reasoning
        final_reasoning = f"Ensemble ({voting_method}): " + "; ".join(reasonings)

        return RoutingDecision(
            search_modality=final_modality,
            generation_type=final_generation,
            confidence_score=min(final_confidence, 1.0),  # Cap at 1.0
            routing_method="ensemble",
            reasoning=final_reasoning,
            metadata={
                "agreement_score": modality_agreement,
                "strategies_count": len(decisions),
            },
        )

    async def _tiered_route(
        self, query: str, context: dict[str, Any] | None, start_time: float
    ) -> RoutingDecision:
        """Original tiered routing logic."""
        # Get thresholds
        if isinstance(self.config, dict):
            fast_threshold = self.config.get("tier_config", {}).get(
                "fast_path_confidence_threshold", 0.7
            )
            slow_threshold = self.config.get("tier_config", {}).get(
                "slow_path_confidence_threshold", 0.6
            )
            langextract_threshold = self.config.get("tier_config", {}).get(
                "langextract_confidence_threshold", 0.5
            )
        elif hasattr(self.config, "tier_config"):
            fast_threshold = self.config.tier_config.get(
                "fast_path_confidence_threshold", 0.7
            )
            slow_threshold = self.config.tier_config.get(
                "slow_path_confidence_threshold", 0.6
            )
            langextract_threshold = self.config.tier_config.get(
                "langextract_confidence_threshold", 0.5
            )
        else:
            fast_threshold = getattr(self.config, "fast_path_confidence_threshold", 0.7)
            slow_threshold = getattr(self.config, "slow_path_confidence_threshold", 0.6)
            langextract_threshold = getattr(
                self.config, "langextract_confidence_threshold", 0.5
            )

        # Try Fast Path (Tier 1)
        fast_decision = None
        if RoutingTier.FAST_PATH in self.strategies:
            fast_decision = await self._try_fast_path(query, context)
            if fast_decision and fast_decision.confidence_score >= fast_threshold:
                self._update_cache(query, fast_decision)
                self._record_routing_metrics(
                    query,
                    fast_decision,
                    time.time() - start_time,
                    RoutingTier.FAST_PATH,
                )
                return fast_decision

        # Escalate to Slow Path (Tier 2)
        if RoutingTier.SLOW_PATH in self.strategies:
            slow_decision = await self._try_slow_path(query, context, fast_decision)
            if slow_decision and slow_decision.confidence_score >= slow_threshold:
                self._update_cache(query, slow_decision)
                self._record_routing_metrics(
                    query,
                    slow_decision,
                    time.time() - start_time,
                    RoutingTier.SLOW_PATH,
                )
                return slow_decision

        # Try LangExtract (Tier 3)
        if RoutingTier.LANGEXTRACT in self.strategies:
            langextract_decision = await self._try_langextract(query, context)
            if (
                langextract_decision
                and langextract_decision.confidence_score >= langextract_threshold
            ):
                self._update_cache(query, langextract_decision)
                self._record_routing_metrics(
                    query,
                    langextract_decision,
                    time.time() - start_time,
                    RoutingTier.LANGEXTRACT,
                )
                return langextract_decision

        # Ultimate Fallback (Tier 4)
        if RoutingTier.FALLBACK in self.strategies:
            fallback_decision = await self._try_fallback(query, context)
            self._update_cache(query, fallback_decision)
            self._record_routing_metrics(
                query, fallback_decision, time.time() - start_time, RoutingTier.FALLBACK
            )
            return fallback_decision

        # No strategies available, return default
        default_decision = RoutingDecision(
            search_modality=SearchModality.BOTH,
            generation_type=GenerationType.RAW_RESULTS,
            confidence_score=0.0,
            routing_method="no_strategy",
            reasoning="No routing strategies available",
        )

        self._record_routing_metrics(
            query, default_decision, time.time() - start_time, None
        )
        return default_decision

    async def _try_fast_path(
        self, query: str, context: dict[str, Any] | None
    ) -> RoutingDecision | None:
        """
        Try the fast path (GLiNER) routing.

        Args:
            query: The user query
            context: Optional context

        Returns:
            RoutingDecision or None if failed
        """
        try:
            logger.debug(f"Attempting Fast Path routing for: {query[:50]}...")
            decision = await self.strategies[RoutingTier.FAST_PATH].route(
                query, context
            )
            decision.metadata["tier"] = RoutingTier.FAST_PATH.value
            return decision
        except Exception as e:
            logger.warning(f"Fast Path routing failed: {e}")
            return None

    async def _try_slow_path(
        self,
        query: str,
        context: dict[str, Any] | None,
        fast_decision: RoutingDecision | None,
    ) -> RoutingDecision | None:
        """
        Try the slow path (LLM) routing.

        Args:
            query: The user query
            context: Optional context
            fast_decision: The decision from fast path (if any)

        Returns:
            RoutingDecision or None if failed
        """
        try:
            logger.debug(f"Escalating to Slow Path for: {query[:50]}...")

            # Enhance context with fast path results if available
            enhanced_context = context or {}
            if fast_decision:
                enhanced_context["fast_path_decision"] = fast_decision.to_dict()

            decision = await self.strategies[RoutingTier.SLOW_PATH].route(
                query, enhanced_context
            )
            decision.metadata["tier"] = RoutingTier.SLOW_PATH.value

            # If fast path had partial results, merge them
            if fast_decision and fast_decision.entities_detected:
                decision.entities_detected = (
                    decision.entities_detected or []
                ) + fast_decision.entities_detected

            return decision
        except Exception as e:
            logger.warning(f"Slow Path routing failed: {e}")
            return None

    async def _try_langextract(
        self, query: str, context: dict[str, Any] | None
    ) -> RoutingDecision | None:
        """
        Try LangExtract routing for structured extraction.

        Args:
            query: The user query
            context: Optional context

        Returns:
            RoutingDecision or None if failed
        """
        try:
            logger.debug(f"Trying LangExtract routing for: {query[:50]}...")
            decision = await self.strategies[RoutingTier.LANGEXTRACT].route(
                query, context
            )
            decision.metadata["tier"] = RoutingTier.LANGEXTRACT.value
            return decision
        except Exception as e:
            logger.warning(f"LangExtract routing failed: {e}")
            return None

    async def _try_fallback(
        self, query: str, context: dict[str, Any] | None
    ) -> RoutingDecision:
        """
        Try the fallback (keyword) routing.

        Args:
            query: The user query
            context: Optional context

        Returns:
            RoutingDecision (always succeeds)
        """
        logger.debug(f"Using Fallback routing for: {query[:50]}...")
        decision = await self.strategies[RoutingTier.FALLBACK].route(query, context)
        decision.metadata["tier"] = RoutingTier.FALLBACK.value
        return decision

    def _check_cache(self, query: str) -> RoutingDecision | None:
        """
        Check if a routing decision is cached.

        Args:
            query: The query to check

        Returns:
            Cached RoutingDecision or None
        """
        if query in self.cache:
            decision, timestamp = self.cache[query]
            # Get cache TTL
            if isinstance(self.config, dict):
                cache_ttl = self.config.get("cache_config", {}).get(
                    "cache_ttl_seconds", 300
                )
            else:
                cache_ttl = getattr(self.config, "cache_ttl_seconds", 300)

            if time.time() - timestamp < cache_ttl:
                return decision
            else:
                # Cache expired
                del self.cache[query]
        return None

    def _update_cache(self, query: str, decision: RoutingDecision):
        """
        Update the routing cache.

        Args:
            query: The query
            decision: The routing decision
        """
        # Check if caching is enabled
        if isinstance(self.config, dict):
            enable_caching = self.config.get("cache_config", {}).get(
                "enable_caching", True
            )
        else:
            enable_caching = getattr(self.config, "enable_caching", True)

        if enable_caching:
            self.cache[query] = (decision, time.time())

            # Limit cache size
            if len(self.cache) > 1000:
                # Remove oldest entries
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:100]:
                    del self.cache[key]

    def _record_routing_metrics(
        self,
        query: str,
        decision: RoutingDecision,
        execution_time: float,
        tier: RoutingTier | None,
    ):
        """
        Record routing metrics for analysis.

        Args:
            query: The routed query
            decision: The routing decision
            execution_time: Time taken
            tier: The tier that handled the query
        """
        # Check if metrics are enabled
        if hasattr(self.config, "monitoring_config"):
            enable_metrics = self.config.monitoring_config.get("enable_metrics", True)
            metrics_batch_size = self.config.monitoring_config.get(
                "metrics_batch_size", 100
            )
        else:
            enable_metrics = True
            metrics_batch_size = 100

        if enable_metrics:
            metrics = RoutingMetrics(
                query=query,
                decision=decision,
                execution_time=execution_time,
                success=decision.confidence_score > 0,
            )
            if tier:
                metrics.decision.metadata["tier"] = tier.value

            self.metrics_buffer.append(metrics)

            # Flush metrics buffer if it's full
            if len(self.metrics_buffer) >= metrics_batch_size:
                self._flush_metrics()

    def _flush_metrics(self):
        """Flush metrics buffer to storage or monitoring system."""
        if self.metrics_buffer:
            # Here you would send metrics to a monitoring system
            # For now, just log summary
            avg_time = sum(m.execution_time for m in self.metrics_buffer) / len(
                self.metrics_buffer
            )
            success_rate = sum(1 for m in self.metrics_buffer if m.success) / len(
                self.metrics_buffer
            )

            tier_distribution = {}
            for m in self.metrics_buffer:
                tier = m.decision.metadata.get("tier", "unknown")
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

            logger.info(
                f"Metrics Summary - Avg Time: {avg_time:.3f}s, Success Rate: {success_rate:.2%}, "
                f"Tier Distribution: {tier_distribution}"
            )

            self.metrics_buffer.clear()

    def get_performance_report(self) -> dict[str, Any]:
        """
        Get a comprehensive performance report.

        Returns:
            Dictionary with performance metrics for all tiers
        """
        report = {
            "total_queries": self.query_count,
            "cache_size": len(self.cache),
            "cache_hit_rate": 0.0,  # Would need to track this
            "tier_performance": {},
        }

        for tier, strategy in self.strategies.items():
            report["tier_performance"][tier.value] = strategy.get_performance_stats()

        return report

    async def optimize_routing(
        self, training_data: list[tuple[str, RoutingDecision]] | None = None
    ):
        """
        Optimize routing strategies based on collected metrics.

        Args:
            training_data: Optional training data for optimization
        """
        # Check if auto-optimization is enabled
        if hasattr(self.config, "optimization_config"):
            enable_auto_opt = self.config.optimization_config.get(
                "enable_auto_optimization", False
            )
        else:
            enable_auto_opt = False

        if not enable_auto_opt:
            logger.info("Auto-optimization is disabled")
            return

        logger.info("Starting routing optimization...")

        # Collect performance data from all strategies
        performance_data = self.get_performance_report()

        # Identify underperforming tiers
        for tier_name, tier_stats in performance_data["tier_performance"].items():
            # Get minimum accuracy threshold
            if hasattr(self.config, "optimization_config"):
                min_accuracy = self.config.optimization_config.get("min_accuracy", 0.8)
            else:
                min_accuracy = 0.8

            if tier_stats.get("success_rate", 1.0) < min_accuracy:
                logger.warning(
                    f"Tier {tier_name} is underperforming with success rate: "
                    f"{tier_stats.get('success_rate', 0):.2%}"
                )

                # Here you would trigger re-optimization of the specific tier
                # For example, re-compile DSPy programs or retrain GLiNER

        logger.info("Routing optimization completed")

    def export_metrics(self, filepath: str):
        """
        Export all collected metrics.

        Args:
            filepath: Path to save metrics
        """
        all_metrics = []

        # Collect metrics from all strategies
        for tier, strategy in self.strategies.items():
            for metric in strategy.metrics_history:
                metric_dict = metric.to_dict()
                metric_dict["tier"] = tier.value
                all_metrics.append(metric_dict)

        # Add buffered metrics
        for metric in self.metrics_buffer:
            all_metrics.append(metric.to_dict())

        with open(filepath, "w") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info(f"Exported {len(all_metrics)} metrics to {filepath}")

    async def _run_ensemble(self, query: str) -> RoutingDecision:
        """
        Run ensemble voting across multiple strategies (placeholder for future implementation).

        Args:
            query: The query to route

        Returns:
            RoutingDecision from ensemble voting
        """
        # For now, just use the standard routing logic
        return await self.route(query)


class TieredRouter(ComprehensiveRouter):
    """
    Specialized implementation focusing on the tiered architecture.
    Provides additional methods for tier management and optimization.
    """

    def __init__(self, config: RouterConfig | None = None):
        super().__init__(config)
        self.tier_usage_stats = dict.fromkeys(RoutingTier, 0)
        self.tier_success_stats = dict.fromkeys(RoutingTier, 0)

    async def route(
        self, query: str, context: dict[str, Any] | None = None
    ) -> RoutingDecision:
        """Enhanced routing with tier tracking."""
        decision = await super().route(query, context)

        # Track tier usage
        tier_used = decision.metadata.get("tier")
        if tier_used:
            for tier in RoutingTier:
                if tier.value == tier_used:
                    self.tier_usage_stats[tier] += 1
                    if decision.confidence_score > 0.5:
                        self.tier_success_stats[tier] += 1
                    break

        # Check if rebalancing is needed
        if self.query_count % 100 == 0:
            self._check_tier_balance()

        return decision

    def _check_tier_balance(self):
        """Check and log tier usage balance."""
        total_usage = sum(self.tier_usage_stats.values())
        if total_usage == 0:
            return

        tier_percentages = {
            tier: (count / total_usage * 100)
            for tier, count in self.tier_usage_stats.items()
        }

        # Log if slow path is being used too often (>20%)
        if tier_percentages.get(RoutingTier.SLOW_PATH, 0) > 20:
            logger.warning(
                f"High Slow Path usage: {tier_percentages[RoutingTier.SLOW_PATH]:.1f}%. "
                "Consider optimizing Fast Path strategy."
            )

        # Log if fallback is being used too often (>10%)
        if tier_percentages.get(RoutingTier.FALLBACK, 0) > 10:
            logger.warning(
                f"High Fallback usage: {tier_percentages[RoutingTier.FALLBACK]:.1f}%. "
                "Consider improving primary strategies."
            )

    def get_tier_statistics(self) -> dict[str, Any]:
        """Get detailed statistics for each tier."""
        stats = {}

        for tier in RoutingTier:
            usage = self.tier_usage_stats[tier]
            success = self.tier_success_stats[tier]

            stats[tier.value] = {
                "usage_count": usage,
                "success_count": success,
                "success_rate": (success / usage * 100) if usage > 0 else 0,
                "usage_percentage": (
                    (usage / self.query_count * 100) if self.query_count > 0 else 0
                ),
            }

        return stats
