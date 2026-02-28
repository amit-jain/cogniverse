# src/routing/optimizer.py
"""
Auto-tuning optimization layer for the comprehensive routing system.
Implements continuous learning and adaptation based on the COMPREHENSIVE_ROUTING.md architecture.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

from .base import (
    RoutingDecision,
    RoutingStrategy,
    SearchModality,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Metrics for tracking optimization performance."""

    timestamp: datetime
    strategy_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_latency: float
    confidence_correlation: float  # Correlation between confidence and accuracy
    error_rate: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class OptimizationConfig:
    """Configuration for the routing optimizer."""

    # Optimization triggers
    min_samples_for_optimization: int = 100
    optimization_interval_seconds: int = 3600  # 1 hour
    performance_degradation_threshold: float = 0.1  # 10% drop triggers optimization

    # Performance thresholds
    min_accuracy: float = 0.8
    min_precision: float = 0.75
    min_recall: float = 0.75
    max_acceptable_latency_ms: float = 100

    # Learning parameters
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.01

    # DSPy optimization settings (for LLM strategies)
    dspy_enabled: bool = True
    dspy_max_bootstrapped_demos: int = 10
    dspy_max_labeled_demos: int = 50
    dspy_metric: str = "f1"  # accuracy, precision, recall, f1

    # GLiNER optimization settings
    gliner_threshold_optimization: bool = True
    gliner_label_optimization: bool = True
    gliner_threshold_step: float = 0.05

    # Data management
    max_history_size: int = 10000


class RoutingOptimizer:
    """
    Base optimizer for routing strategies.
    Tracks performance and triggers optimization when needed.
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        tenant_id: str,
        config: OptimizationConfig | None = None,
    ):
        """
        Initialize the routing optimizer.

        Args:
            telemetry_provider: Telemetry provider for artifact persistence.
            tenant_id: Tenant identifier for multi-tenant isolation.
            config: Optimization configuration
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for RoutingOptimizer")
        self._artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

        # Handle both OptimizationConfig and dict
        if isinstance(config, dict):
            from dataclasses import fields

            valid_fields = {f.name for f in fields(OptimizationConfig)}
            filtered_config = {
                k: v for k, v in config.items() if k in valid_fields
            }
            self.config = (
                OptimizationConfig(**filtered_config)
                if filtered_config
                else OptimizationConfig()
            )
        else:
            self.config = config or OptimizationConfig()

        self.performance_history: deque = deque(maxlen=self.config.max_history_size)
        self.optimization_history: list[OptimizationMetrics] = []
        self.last_optimization_time = datetime.now()
        self.baseline_metrics: OptimizationMetrics | None = None

    def track_performance(
        self,
        query: str,
        predicted: RoutingDecision,
        actual: RoutingDecision | None = None,
        user_feedback: dict[str, Any] | None = None,
    ):
        """
        Track routing performance for a single query.

        Args:
            query: The routed query
            predicted: The predicted routing decision
            actual: The actual/correct routing decision (if known)
            user_feedback: Optional user feedback on the routing
        """
        record = {
            "timestamp": datetime.now(),
            "query": query,
            "predicted": predicted.to_dict(),
            "actual": actual.to_dict() if actual else None,
            "user_feedback": user_feedback,
            "latency": predicted.metadata.get("latency_ms", 0),
        }

        self.performance_history.append(record)

        # Check if optimization is needed
        if self._should_optimize():
            asyncio.create_task(self.optimize())

    def _should_optimize(self) -> bool:
        """
        Determine if optimization should be triggered.

        Returns:
            True if optimization should run
        """
        # Check time since last optimization
        time_since_last = datetime.now() - self.last_optimization_time
        if time_since_last.total_seconds() < self.config.optimization_interval_seconds:
            return False

        # Check if we have enough samples
        if len(self.performance_history) < self.config.min_samples_for_optimization:
            return False

        # Check for performance degradation
        current_metrics = self._calculate_current_metrics()
        if self.baseline_metrics:
            degradation = self.baseline_metrics.accuracy - current_metrics.accuracy
            if degradation > self.config.performance_degradation_threshold:
                logger.warning(f"Performance degradation detected: {degradation:.2%}")
                return True

        # Regular optimization interval reached
        return True

    def _calculate_current_metrics(self) -> OptimizationMetrics:
        """
        Calculate current performance metrics from history.

        Returns:
            Current optimization metrics
        """
        # Filter records with ground truth
        records_with_truth = [
            r for r in self.performance_history if r.get("actual") is not None
        ]

        if not records_with_truth:
            # Return default metrics if no ground truth available
            return OptimizationMetrics(
                timestamp=datetime.now(),
                strategy_name="unknown",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                avg_latency=0.0,
                confidence_correlation=0.0,
                error_rate=1.0,
                sample_count=0,
            )

        # Calculate metrics
        correct_predictions = 0
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        latencies = []
        confidences = []
        accuracies = []

        for record in records_with_truth:
            predicted = record["predicted"]
            actual = record["actual"]

            # Check if prediction matches actual
            pred_modality = predicted["search_modality"]
            actual_modality = actual["search_modality"]

            is_correct = pred_modality == actual_modality
            if is_correct:
                correct_predictions += 1
                true_positives[pred_modality] += 1
            else:
                false_positives[pred_modality] += 1
                false_negatives[actual_modality] += 1

            # Track latency and confidence
            latencies.append(record.get("latency", 0))
            confidences.append(predicted.get("confidence_score", 0))
            accuracies.append(1 if is_correct else 0)

        # Calculate aggregate metrics
        accuracy = correct_predictions / len(records_with_truth)

        # Calculate precision and recall (macro-averaged)
        precisions = []
        recalls = []
        for modality in ["video", "text", "both"]:
            tp = true_positives[modality]
            fp = false_positives[modality]
            fn = false_negatives[modality]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        # Calculate confidence correlation
        if confidences and accuracies:
            confidence_correlation = np.corrcoef(confidences, accuracies)[0, 1]
            if np.isnan(confidence_correlation):
                confidence_correlation = 0.0
        else:
            confidence_correlation = 0.0

        return OptimizationMetrics(
            timestamp=datetime.now(),
            strategy_name=self.__class__.__name__,
            accuracy=accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=f1_score,
            avg_latency=np.mean(latencies) if latencies else 0,
            confidence_correlation=confidence_correlation,
            error_rate=1 - accuracy,
            sample_count=len(records_with_truth),
        )

    async def optimize(self):
        """
        Run optimization process.
        To be overridden by specific optimizers.
        """
        logger.info("Starting routing optimization...")
        self.last_optimization_time = datetime.now()

        # Calculate current metrics
        current_metrics = self._calculate_current_metrics()
        self.optimization_history.append(current_metrics)

        # Update baseline if first run or improved
        if (
            not self.baseline_metrics
            or current_metrics.accuracy > self.baseline_metrics.accuracy
        ):
            self.baseline_metrics = current_metrics

        # Export metrics
        self._export_metrics(current_metrics)

        logger.info(
            f"Optimization complete. Current accuracy: {current_metrics.accuracy:.2%}"
        )

    def _export_metrics(self, metrics: OptimizationMetrics):
        """Export metrics to telemetry experiment store."""
        asyncio.ensure_future(
            self._artifact_manager.log_optimization_run(
                "routing_optimizer", metrics.to_dict()
            )
        )


class AutoTuningOptimizer(RoutingOptimizer):
    """
    Advanced optimizer with auto-tuning capabilities.
    Implements the continuous learning approach from COMPREHENSIVE_ROUTING.md.
    """

    def __init__(
        self,
        strategy: RoutingStrategy,
        telemetry_provider: TelemetryProvider,
        tenant_id: str,
        config: OptimizationConfig | None = None,
    ):
        """
        Initialize the auto-tuning optimizer.

        Args:
            strategy: The routing strategy to optimize
            telemetry_provider: Telemetry provider for artifact persistence.
            tenant_id: Tenant identifier for multi-tenant isolation.
            config: Optimization configuration
        """
        super().__init__(telemetry_provider, tenant_id, config)
        self.strategy = strategy
        self.optimization_attempts = 0
        self.best_params: dict[str, Any] | None = None
        self.best_performance = 0.0
        self.training_examples = []
        self.label_performance = {}
        self.threshold_performance = {}
        self.temperature_performance = {}

    async def record_performance(
        self,
        query: str,
        decision: RoutingDecision,
        latency_ms: float,
        actual_modality: SearchModality | None = None,
    ):
        """
        Record performance for a routing decision.

        Args:
            query: The query that was routed
            decision: The routing decision made
            latency_ms: Latency in milliseconds
            actual_modality: The actual/correct modality (if known)
        """
        # Convert to format expected by track_performance
        actual_decision = None
        if actual_modality:
            actual_decision = RoutingDecision(
                search_modality=actual_modality,
                generation_type=decision.generation_type,
                confidence_score=1.0,
                routing_method="ground_truth",
            )

        # Update metadata with latency
        decision.metadata["latency_ms"] = latency_ms

        # Track the performance
        self.track_performance(query, decision, actual_decision)

    def should_optimize(self) -> bool:
        """Check if optimization should be triggered."""
        return self._should_optimize()

    async def optimize(self):
        """
        Run auto-tuning optimization for the strategy.
        """
        await super().optimize()

        self.optimization_attempts += 1
        logger.info(
            f"Running auto-tuning optimization attempt #{self.optimization_attempts}"
        )

        # Get strategy type and run appropriate optimization
        strategy_name = self.strategy.__class__.__name__

        if "GLiNER" in strategy_name:
            await self._optimize_gliner()
        elif "LLM" in strategy_name:
            await self._optimize_llm()
        elif "Keyword" in strategy_name:
            await self._optimize_keyword()
        elif "Hybrid" in strategy_name or "Ensemble" in strategy_name:
            await self._optimize_composite()
        else:
            logger.warning(f"No optimization available for strategy: {strategy_name}")

    async def _optimize_gliner(self):
        """Optimize GLiNER strategy parameters."""
        logger.info("Optimizing GLiNER strategy...")

        if not self.config.gliner_threshold_optimization:
            return

        # Get current threshold
        current_threshold = self.strategy.config.get("gliner_threshold", 0.3)

        # Try different thresholds
        best_threshold = current_threshold
        best_score = 0.0

        for threshold in np.arange(0.1, 0.9, self.config.gliner_threshold_step):
            # Update strategy threshold
            self.strategy.config["gliner_threshold"] = threshold

            # Evaluate on recent data
            score = await self._evaluate_strategy()

            if score > best_score:
                best_score = score
                best_threshold = threshold

        # Apply best threshold
        self.strategy.config["gliner_threshold"] = best_threshold
        logger.info(
            f"Optimized GLiNER threshold: {best_threshold:.2f} (score: {best_score:.3f})"
        )

        # Optimize labels if enabled
        if self.config.gliner_label_optimization:
            await self._optimize_gliner_labels()

    async def _optimize_gliner_labels(self):
        """Optimize GLiNER label selection."""
        # Analyze which labels are most effective
        label_effectiveness = defaultdict(float)

        for record in list(self.performance_history)[-100:]:  # Last 100 records
            if record.get("predicted", {}).get("entities_detected"):
                entities = record["predicted"]["entities_detected"]
                is_correct = (
                    record.get("actual")
                    and record["predicted"]["search_modality"]
                    == record["actual"]["search_modality"]
                )

                for entity in entities:
                    label = entity.get("label", "")
                    if is_correct:
                        label_effectiveness[label] += 1
                    else:
                        label_effectiveness[label] -= 0.5

        # Select top performing labels
        if label_effectiveness:
            sorted_labels = sorted(
                label_effectiveness.items(), key=lambda x: x[1], reverse=True
            )
            top_labels = [label for label, score in sorted_labels[:15] if score > 0]

            if top_labels:
                self.strategy.config["gliner_labels"] = top_labels
                logger.info(f"Updated GLiNER labels: {top_labels}")

    async def _optimize_llm(self):
        """Optimize LLM strategy using DSPy if available."""
        logger.info("Optimizing LLM strategy...")

        if not self.config.dspy_enabled:
            # Basic optimization without DSPy
            await self._optimize_llm_temperature()
            return

        try:
            import importlib.util

            if importlib.util.find_spec("dspy") is None:
                raise ImportError("DSPy not available")

            import dspy  # noqa: F401
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch

            # Prepare training data from history
            training_data = self._prepare_training_data()

            if len(training_data) < 10:
                logger.warning("Insufficient training data for DSPy optimization")
                return

            # Create DSPy optimizer
            _ = BootstrapFewShotWithRandomSearch(
                metric=self._create_dspy_metric(),
                max_bootstrapped_demos=self.config.dspy_max_bootstrapped_demos,
                max_labeled_demos=self.config.dspy_max_labeled_demos,
            )

            # Run optimization (would need actual DSPy module implementation)
            logger.info("DSPy optimization would run here with prepared training data")

        except ImportError:
            logger.warning("DSPy not available, using basic optimization")
            await self._optimize_llm_temperature()

    async def _optimize_llm_temperature(self):
        """Optimize LLM temperature parameter."""
        current_temp = self.strategy.config.get("temperature", 0.1)

        best_temp = current_temp
        best_score = 0.0

        for temp in [0.0, 0.1, 0.3, 0.5, 0.7]:
            self.strategy.config["temperature"] = temp
            score = await self._evaluate_strategy()

            if score > best_score:
                best_score = score
                best_temp = temp

        self.strategy.config["temperature"] = best_temp
        logger.info(f"Optimized LLM temperature: {best_temp} (score: {best_score:.3f})")

    async def _optimize_keyword(self):
        """Optimize keyword lists based on performance."""
        logger.info("Optimizing keyword strategy...")

        # Analyze which keywords are most predictive
        keyword_effectiveness = defaultdict(float)

        for record in list(self.performance_history)[-200:]:
            query = record.get("query", "").lower()
            predicted = record.get("predicted", {})
            actual = record.get("actual")

            if actual:
                is_correct = predicted["search_modality"] == actual["search_modality"]

                # Check which keywords appeared
                for keyword_list_name in ["video_keywords", "text_keywords"]:
                    keywords = self.strategy.config.get(keyword_list_name, [])
                    for keyword in keywords:
                        if keyword in query:
                            if is_correct:
                                keyword_effectiveness[keyword] += 1
                            else:
                                keyword_effectiveness[keyword] -= 0.5

        # Update keyword lists with most effective keywords
        if keyword_effectiveness:
            sorted_keywords = sorted(
                keyword_effectiveness.items(), key=lambda x: x[1], reverse=True
            )

            # Separate into video and text keywords based on historical performance
            # This would need more sophisticated analysis in production
            effective_keywords = [kw for kw, score in sorted_keywords if score > 0]

            logger.info(f"Updated effective keywords: {effective_keywords[:10]}")

    async def _optimize_composite(self):
        """Optimize composite strategies (Hybrid/Ensemble)."""
        logger.info("Optimizing composite strategy...")

        # For ensemble, optimize weights
        if hasattr(self.strategy, "weights"):
            await self._optimize_ensemble_weights()

        # For hybrid, optimize confidence thresholds
        if hasattr(self.strategy, "confidence_threshold"):
            await self._optimize_confidence_thresholds()

    async def _optimize_ensemble_weights(self):
        """Optimize ensemble strategy weights."""
        # Analyze per-strategy performance
        strategy_scores = defaultdict(list)

        for record in list(self.performance_history)[-100:]:
            predicted = record.get("predicted", {})
            actual = record.get("actual")

            if actual and "routing_method" in predicted:
                method = predicted["routing_method"]
                is_correct = predicted["search_modality"] == actual["search_modality"]
                strategy_scores[method].append(1 if is_correct else 0)

        # Update weights based on performance
        new_weights = {}
        for strategy_name, scores in strategy_scores.items():
            if scores:
                avg_score = np.mean(scores)
                new_weights[strategy_name] = max(
                    0.1, avg_score
                )  # Minimum weight of 0.1

        if new_weights:
            self.strategy.weights = new_weights
            logger.info(f"Updated ensemble weights: {new_weights}")

    async def _optimize_confidence_thresholds(self):
        """Optimize confidence thresholds for tier escalation."""
        best_threshold = self.strategy.confidence_threshold
        best_score = 0.0

        for threshold in [0.5, 0.6, 0.7, 0.8]:
            self.strategy.confidence_threshold = threshold
            score = await self._evaluate_strategy()

            if score > best_score:
                best_score = score
                best_threshold = threshold

        self.strategy.confidence_threshold = best_threshold
        logger.info(
            f"Optimized confidence threshold: {best_threshold} (score: {best_score:.3f})"
        )

    async def _evaluate_strategy(self) -> float:
        """
        Evaluate strategy performance on recent data.

        Returns:
            Performance score (0-1)
        """
        # Simple evaluation using recent performance history
        # In production, this would use a holdout validation set

        correct = 0
        total = 0

        for record in list(self.performance_history)[-50:]:
            if record.get("actual"):
                total += 1
                predicted = record["predicted"]
                actual = record["actual"]

                if predicted["search_modality"] == actual["search_modality"]:
                    correct += 1

        if total == 0:
            return 0.0

        accuracy = correct / total

        # Consider other metrics based on configuration
        if self.config.dspy_metric == "f1":
            # Would calculate F1 score here
            return accuracy  # Simplified

        return accuracy

    def _prepare_training_data(self) -> list[tuple[str, dict[str, Any]]]:
        """
        Prepare training data from performance history.

        Returns:
            List of (query, ground_truth) tuples
        """
        training_data = []

        for record in self.performance_history:
            if record.get("actual"):
                query = record["query"]
                ground_truth = record["actual"]
                training_data.append((query, ground_truth))

        return training_data

    def _create_dspy_metric(self) -> Callable:
        """
        Create a metric function for DSPy optimization.

        Returns:
            Metric function
        """

        def metric(predicted, ground_truth):
            # Simple accuracy metric
            if predicted.search_modality == ground_truth.search_modality:
                return 1.0
            return 0.0

        return metric

    async def save_checkpoint(self):
        """Save optimization checkpoint to telemetry."""
        checkpoint = {
            "strategy_name": self.strategy.__class__.__name__,
            "strategy_config": self.strategy.config,
            "optimization_attempts": self.optimization_attempts,
            "best_params": self.best_params,
            "best_performance": self.best_performance,
            "baseline_metrics": (
                self.baseline_metrics.to_dict() if self.baseline_metrics else None
            ),
            "timestamp": datetime.now().isoformat(),
        }

        await self._artifact_manager.save_blob(
            "checkpoint", "routing_optimizer", json.dumps(checkpoint)
        )
        logger.info("Saved routing optimizer checkpoint to telemetry")

    async def load_checkpoint(self):
        """Load optimization checkpoint from telemetry."""
        content = await self._artifact_manager.load_blob(
            "checkpoint", "routing_optimizer"
        )
        if not content:
            logger.info("No checkpoint found in telemetry")
            return

        checkpoint = json.loads(content)

        # Restore configuration
        self.strategy.config.update(checkpoint.get("strategy_config", {}))
        self.optimization_attempts = checkpoint.get("optimization_attempts", 0)
        self.best_params = checkpoint.get("best_params")
        self.best_performance = checkpoint.get("best_performance", 0.0)

        logger.info("Loaded routing optimizer checkpoint from telemetry")

    def get_performance_report(self) -> dict[str, Any]:
        """
        Get a performance report for the optimizer.

        Returns:
            Dictionary containing performance metrics
        """
        metrics = self._calculate_current_metrics()

        report = {
            "total_samples": len(self.performance_history),
            "average_accuracy": metrics.accuracy,
            "average_precision": metrics.precision,
            "average_recall": metrics.recall,
            "average_f1_score": metrics.f1_score,
            "average_latency_ms": metrics.avg_latency,
            "confidence_correlation": metrics.confidence_correlation,
            "error_rate": metrics.error_rate,
            "optimization_attempts": self.optimization_attempts,
            "best_performance": self.best_performance,
            "strategy_name": self.strategy.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }

        return report
