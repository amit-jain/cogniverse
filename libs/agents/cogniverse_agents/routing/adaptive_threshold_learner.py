"""
Adaptive Threshold Learning for Dynamic System Optimization

This module implements adaptive threshold learning that automatically adjusts
various decision boundaries and confidence thresholds based on system performance
and user feedback. It learns optimal thresholds for routing confidence, similarity
matching, pattern acceptance, and other critical system parameters.

Key Features:
- Multi-parameter threshold optimization using reinforcement learning
- Performance-based threshold adjustment with feedback loops
- Statistical significance testing for threshold changes
- Automatic rollback for performance degradation
- Configurable learning rates and adaptation strategies
- Integration with GRPO and SIMBA for system-wide optimization
"""

import json
import logging
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from cogniverse_core.common.tenant_utils import get_tenant_storage_path

logger = logging.getLogger(__name__)


class ThresholdParameter(Enum):
    """Types of threshold parameters that can be adapted"""

    ROUTING_CONFIDENCE = "routing_confidence"
    SIMILARITY_THRESHOLD = "similarity_threshold"
    PATTERN_CONFIDENCE = "pattern_confidence"
    ENTITY_CONFIDENCE = "entity_confidence"
    RELATIONSHIP_CONFIDENCE = "relationship_confidence"
    SEARCH_QUALITY_THRESHOLD = "search_quality_threshold"
    USER_SATISFACTION_THRESHOLD = "user_satisfaction_threshold"
    ENHANCEMENT_QUALITY_THRESHOLD = "enhancement_quality_threshold"
    ORCHESTRATION_THRESHOLD = "orchestration_threshold"


class AdaptationStrategy(Enum):
    """Adaptation strategies for threshold learning"""

    GRADIENT_BASED = "gradient_based"  # Gradient ascent/descent
    EVOLUTIONARY = "evolutionary"  # Genetic algorithm approach
    BANDIT = "bandit"  # Multi-armed bandit
    BAYESIAN = "bayesian"  # Bayesian optimization
    STATISTICAL = "statistical"  # Statistical hypothesis testing


@dataclass
class ThresholdConfig:
    """Configuration for a single threshold parameter"""

    parameter: ThresholdParameter
    initial_value: float
    min_value: float
    max_value: float
    learning_rate: float = 0.01
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.GRADIENT_BASED

    # Performance metrics to track
    primary_metric: str = "success_rate"  # Main metric to optimize
    secondary_metrics: List[str] = field(default_factory=list)

    # Learning constraints
    min_samples_for_update: int = 50
    significance_threshold: float = 0.05
    max_change_per_update: float = 0.1
    rollback_threshold: float = -0.05  # Rollback if performance drops by this amount

    # Exploration parameters
    exploration_rate: float = 0.1
    exploration_decay: float = 0.99


@dataclass
class PerformanceMetrics:
    """Performance metrics for threshold evaluation"""

    success_rate: float = 0.0
    average_confidence: float = 0.0
    response_time: float = 0.0
    user_satisfaction: float = 0.0
    search_quality: float = 0.0
    enhancement_quality: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    # Computed metrics
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Sample metadata
    sample_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ThresholdState:
    """Current state of a threshold parameter"""

    parameter: ThresholdParameter
    current_value: float
    best_value: float
    best_performance: float

    # Learning history
    value_history: List[Tuple[float, float]] = field(
        default_factory=list
    )  # (value, performance)
    performance_samples: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Statistics
    total_updates: int = 0
    successful_updates: int = 0
    rollbacks: int = 0
    last_update: datetime = field(default_factory=datetime.now)

    # Exploration state
    exploration_count: int = 0
    exploitation_count: int = 0


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold learning"""

    # Threshold configurations
    threshold_configs: Dict[ThresholdParameter, ThresholdConfig] = field(
        default_factory=dict
    )

    # Global learning parameters
    global_learning_rate: float = 0.01
    performance_window_size: int = 100
    update_frequency: int = 50  # Update every N samples

    # Statistical testing
    statistical_test: str = "t_test"  # t_test, mannwhitney, wilcoxon
    min_effect_size: float = 0.1
    confidence_level: float = 0.95

    # Rollback and safety
    enable_automatic_rollback: bool = True
    rollback_window_size: int = 20
    performance_degradation_threshold: float = 0.05

    # Multi-parameter optimization
    enable_joint_optimization: bool = False
    correlation_threshold: float = 0.3

    # Storage
    state_file: str = "adaptive_thresholds_state.pkl"
    metrics_file: str = "threshold_metrics.json"
    history_file: str = "threshold_history.pkl"

    # Experiment tracking
    enable_ab_testing: bool = False
    ab_test_split: float = 0.5
    ab_test_duration_hours: int = 24


class AdaptiveThresholdLearner:
    """
    Adaptive threshold learning system for dynamic system optimization

    This system automatically adjusts various threshold parameters based on
    system performance metrics and user feedback to optimize overall performance.

    Each tenant gets isolated storage for threshold states and metrics.
    """

    def __init__(
        self,
        tenant_id: str,
        config: Optional[AdaptiveThresholdConfig] = None,
        base_storage_dir: str = "data/adaptive_learning",
    ):
        """
        Initialize adaptive threshold learner

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            config: Learner configuration
            base_storage_dir: Base directory for storage

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id
        self.config = config or self._create_default_config()

        # Tenant-specific storage directory with org/tenant structure
        self.storage_dir = get_tenant_storage_path(base_storage_dir, tenant_id)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Threshold states
        self.threshold_states: Dict[ThresholdParameter, ThresholdState] = {}

        # Performance tracking
        self.current_metrics = PerformanceMetrics()
        self.metrics_history = deque(maxlen=self.config.performance_window_size * 10)

        # Learning state
        self.sample_count = 0
        self.last_update = datetime.now()
        self.active_experiments = {}

        # A/B testing state
        self.ab_test_groups = {}
        self.ab_test_results = defaultdict(list)

        # Load existing state
        self._load_stored_state()

        # Initialize default thresholds if not loaded
        self._initialize_default_thresholds()

        logger.info(
            f"Adaptive threshold learner initialized with {len(self.threshold_states)} parameters"
        )

    def _create_default_config(self) -> AdaptiveThresholdConfig:
        """Create default adaptive threshold configuration"""
        config = AdaptiveThresholdConfig()

        # Define default threshold configurations
        default_thresholds = {
            ThresholdParameter.ROUTING_CONFIDENCE: ThresholdConfig(
                parameter=ThresholdParameter.ROUTING_CONFIDENCE,
                initial_value=0.7,
                min_value=0.3,
                max_value=0.95,
                learning_rate=0.02,
                primary_metric="success_rate",
                secondary_metrics=["response_time", "user_satisfaction"],
            ),
            ThresholdParameter.SIMILARITY_THRESHOLD: ThresholdConfig(
                parameter=ThresholdParameter.SIMILARITY_THRESHOLD,
                initial_value=0.7,
                min_value=0.4,
                max_value=0.95,
                learning_rate=0.01,
                primary_metric="search_quality",
                secondary_metrics=["precision", "recall"],
            ),
            ThresholdParameter.PATTERN_CONFIDENCE: ThresholdConfig(
                parameter=ThresholdParameter.PATTERN_CONFIDENCE,
                initial_value=0.5,
                min_value=0.2,
                max_value=0.9,
                learning_rate=0.015,
                primary_metric="enhancement_quality",
                secondary_metrics=["success_rate"],
            ),
            ThresholdParameter.ENTITY_CONFIDENCE: ThresholdConfig(
                parameter=ThresholdParameter.ENTITY_CONFIDENCE,
                initial_value=0.5,
                min_value=0.2,
                max_value=0.9,
                learning_rate=0.01,
                primary_metric="precision",
                secondary_metrics=["f1_score"],
            ),
            ThresholdParameter.RELATIONSHIP_CONFIDENCE: ThresholdConfig(
                parameter=ThresholdParameter.RELATIONSHIP_CONFIDENCE,
                initial_value=0.6,
                min_value=0.3,
                max_value=0.95,
                learning_rate=0.01,
                primary_metric="precision",
                secondary_metrics=["f1_score"],
            ),
        }

        config.threshold_configs = default_thresholds
        return config

    def _initialize_default_thresholds(self):
        """Initialize default threshold states"""
        for param, config in self.config.threshold_configs.items():
            if param not in self.threshold_states:
                self.threshold_states[param] = ThresholdState(
                    parameter=param,
                    current_value=config.initial_value,
                    best_value=config.initial_value,
                    best_performance=0.0,
                )

    async def record_performance_sample(
        self,
        routing_success: bool,
        routing_confidence: float,
        search_quality: float,
        response_time: float,
        user_satisfaction: Optional[float] = None,
        enhancement_applied: bool = False,
        enhancement_quality: float = 0.0,
        threshold_decisions: Optional[Dict[ThresholdParameter, bool]] = None,
    ) -> None:
        """
        Record a performance sample for threshold learning

        Args:
            routing_success: Whether routing was successful
            routing_confidence: Confidence of routing decision
            search_quality: Quality of search results (0-1)
            response_time: System response time in seconds
            user_satisfaction: Optional user satisfaction score (0-1)
            enhancement_applied: Whether query enhancement was applied
            enhancement_quality: Quality of enhancement (0-1)
            threshold_decisions: Decisions made by various thresholds
        """
        try:
            self.sample_count += 1

            # Create performance metrics
            metrics = PerformanceMetrics(
                success_rate=1.0 if routing_success else 0.0,
                average_confidence=routing_confidence,
                response_time=response_time,
                user_satisfaction=user_satisfaction or 0.5,
                search_quality=search_quality,
                enhancement_quality=enhancement_quality if enhancement_applied else 0.0,
                sample_count=1,
            )

            # Calculate derived metrics
            if threshold_decisions:
                # Calculate precision/recall/F1 based on threshold decisions
                self._update_classification_metrics(
                    metrics, threshold_decisions, routing_success
                )

            # Update current metrics (moving average)
            self._update_current_metrics(metrics)

            # Add to performance samples for each threshold
            for param, state in self.threshold_states.items():
                performance_score = self._calculate_performance_score(metrics, param)
                state.performance_samples.append(
                    (state.current_value, performance_score, datetime.now())
                )

            # Store metrics history
            self.metrics_history.append(metrics)

            # Trigger threshold updates if conditions met
            if self._should_trigger_update():
                await self._update_thresholds()

            # Periodic persistence
            if self.sample_count % 100 == 0:
                await self._persist_state()

        except Exception as e:
            logger.error(f"Failed to record performance sample: {e}")

    def _update_classification_metrics(
        self,
        metrics: PerformanceMetrics,
        threshold_decisions: Dict[ThresholdParameter, bool],
        actual_success: bool,
    ):
        """Update precision, recall, F1 score based on threshold decisions"""
        # This is simplified - in practice, you'd need to track true/false positives/negatives
        # for each threshold parameter separately

        decisions = list(threshold_decisions.values())
        if decisions:
            predicted_positive = any(decisions)  # Any threshold triggered

            if predicted_positive and actual_success:
                tp = 1
                fp = 0
            elif predicted_positive and not actual_success:
                tp = 0
                fp = 1
            elif not predicted_positive and actual_success:
                tp = 0
                fn = 1
            else:
                tp = 0
                fn = 0

            # Update metrics (these would be running averages in practice)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics.precision = precision
            metrics.recall = recall
            metrics.f1_score = f1

    def _update_current_metrics(self, new_metrics: PerformanceMetrics):
        """Update current metrics using moving average"""
        alpha = 0.1  # Learning rate for moving average

        self.current_metrics.success_rate = (
            1 - alpha
        ) * self.current_metrics.success_rate + alpha * new_metrics.success_rate
        self.current_metrics.average_confidence = (
            (1 - alpha) * self.current_metrics.average_confidence
            + alpha * new_metrics.average_confidence
        )
        self.current_metrics.response_time = (
            1 - alpha
        ) * self.current_metrics.response_time + alpha * new_metrics.response_time
        self.current_metrics.user_satisfaction = (
            (1 - alpha) * self.current_metrics.user_satisfaction
            + alpha * new_metrics.user_satisfaction
        )
        self.current_metrics.search_quality = (
            1 - alpha
        ) * self.current_metrics.search_quality + alpha * new_metrics.search_quality
        self.current_metrics.enhancement_quality = (
            (1 - alpha) * self.current_metrics.enhancement_quality
            + alpha * new_metrics.enhancement_quality
        )
        self.current_metrics.precision = (
            1 - alpha
        ) * self.current_metrics.precision + alpha * new_metrics.precision
        self.current_metrics.recall = (
            1 - alpha
        ) * self.current_metrics.recall + alpha * new_metrics.recall
        self.current_metrics.f1_score = (
            1 - alpha
        ) * self.current_metrics.f1_score + alpha * new_metrics.f1_score

        self.current_metrics.sample_count += 1
        self.current_metrics.timestamp = datetime.now()

    def _calculate_performance_score(
        self, metrics: PerformanceMetrics, parameter: ThresholdParameter
    ) -> float:
        """Calculate performance score for a specific parameter"""
        config = self.config.threshold_configs.get(parameter)
        if not config:
            return 0.0

        # Get primary metric value
        primary_value = getattr(metrics, config.primary_metric, 0.0)

        # Get secondary metrics
        secondary_values = [
            getattr(metrics, metric, 0.0) for metric in config.secondary_metrics
        ]

        # Weighted combination (80% primary, 20% secondary)
        if secondary_values:
            secondary_avg = np.mean(secondary_values)
            performance_score = 0.8 * primary_value + 0.2 * secondary_avg
        else:
            performance_score = primary_value

        return performance_score

    def _should_trigger_update(self) -> bool:
        """Determine if thresholds should be updated"""
        # Update based on sample count
        if self.sample_count % self.config.update_frequency == 0:
            return True

        # Update if performance is consistently declining
        if len(self.metrics_history) >= 20:
            recent_performance = [
                m.success_rate for m in list(self.metrics_history)[-10:]
            ]
            older_performance = [
                m.success_rate for m in list(self.metrics_history)[-20:-10]
            ]

            if np.mean(recent_performance) < np.mean(older_performance) - 0.05:
                return True

        return False

    async def _update_thresholds(self):
        """Update all threshold parameters based on collected performance data"""
        logger.info("Updating adaptive thresholds...")

        updates_made = 0

        for parameter, state in self.threshold_states.items():
            config = self.config.threshold_configs.get(parameter)
            if (
                not config
                or len(state.performance_samples) < config.min_samples_for_update
            ):
                continue

            try:
                # Calculate new threshold value
                new_value = await self._calculate_optimal_threshold(
                    parameter, state, config
                )

                # Validate new value
                if new_value is not None and self._validate_threshold_change(
                    state, new_value, config
                ):
                    old_value = state.current_value
                    state.current_value = new_value
                    state.total_updates += 1
                    state.last_update = datetime.now()

                    logger.info(
                        f"Updated {parameter.value}: {old_value:.3f} -> {new_value:.3f}"
                    )
                    updates_made += 1

                    # Record in history
                    current_performance = self._calculate_current_performance(parameter)
                    state.value_history.append((new_value, current_performance))

                    # Update best value if performance improved
                    if current_performance > state.best_performance:
                        state.best_value = new_value
                        state.best_performance = current_performance
                        state.successful_updates += 1

            except Exception as e:
                logger.error(f"Failed to update threshold {parameter.value}: {e}")

        logger.info(f"Threshold update complete: {updates_made} parameters updated")

        # Check for rollbacks
        if self.config.enable_automatic_rollback:
            await self._check_and_perform_rollbacks()

    async def _calculate_optimal_threshold(
        self,
        parameter: ThresholdParameter,
        state: ThresholdState,
        config: ThresholdConfig,
    ) -> Optional[float]:
        """Calculate optimal threshold value using specified adaptation strategy"""

        if config.adaptation_strategy == AdaptationStrategy.GRADIENT_BASED:
            return await self._gradient_based_update(parameter, state, config)
        elif config.adaptation_strategy == AdaptationStrategy.EVOLUTIONARY:
            return await self._evolutionary_update(parameter, state, config)
        elif config.adaptation_strategy == AdaptationStrategy.BANDIT:
            return await self._bandit_update(parameter, state, config)
        elif config.adaptation_strategy == AdaptationStrategy.STATISTICAL:
            return await self._statistical_update(parameter, state, config)
        else:
            logger.warning(f"Unknown adaptation strategy: {config.adaptation_strategy}")
            return None

    async def _gradient_based_update(
        self,
        parameter: ThresholdParameter,
        state: ThresholdState,
        config: ThresholdConfig,
    ) -> Optional[float]:
        """Update threshold using gradient-based optimization"""
        try:
            # Get recent performance samples
            recent_samples = list(state.performance_samples)[
                -config.min_samples_for_update :
            ]

            if len(recent_samples) < config.min_samples_for_update:
                return None

            # Calculate gradient
            values = [sample[0] for sample in recent_samples]

            # Simple gradient calculation using finite differences
            if len(values) >= 2:
                # Group samples by similar threshold values
                value_groups = defaultdict(list)
                for val, perf, _ in recent_samples:
                    rounded_val = round(val, 2)
                    value_groups[rounded_val].append(perf)

                # Calculate average performance for each value
                value_performance = {}
                for val, perfs in value_groups.items():
                    if len(perfs) >= 3:  # Minimum samples for reliable estimate
                        value_performance[val] = np.mean(perfs)

                if len(value_performance) >= 2:
                    # Calculate gradient
                    sorted_items = sorted(value_performance.items())
                    gradient = 0.0
                    count = 0

                    for i in range(len(sorted_items) - 1):
                        val1, perf1 = sorted_items[i]
                        val2, perf2 = sorted_items[i + 1]

                        if abs(val2 - val1) > 0.01:  # Avoid division by zero
                            grad = (perf2 - perf1) / (val2 - val1)
                            gradient += grad
                            count += 1

                    if count > 0:
                        gradient = gradient / count

                        # Update threshold in direction of gradient
                        new_value = (
                            state.current_value + config.learning_rate * gradient
                        )

                        # Apply constraints
                        new_value = np.clip(
                            new_value, config.min_value, config.max_value
                        )

                        # Limit change per update
                        max_change = config.max_change_per_update
                        change = new_value - state.current_value
                        if abs(change) > max_change:
                            new_value = (
                                state.current_value + np.sign(change) * max_change
                            )

                        return new_value

            return None

        except Exception as e:
            logger.error(f"Gradient-based update failed for {parameter.value}: {e}")
            return None

    async def _statistical_update(
        self,
        parameter: ThresholdParameter,
        state: ThresholdState,
        config: ThresholdConfig,
    ) -> Optional[float]:
        """Update threshold using statistical hypothesis testing"""
        try:
            # Get performance samples around current threshold
            current_samples = [
                perf
                for val, perf, _ in state.performance_samples
                if abs(val - state.current_value) < 0.05
            ]

            if len(current_samples) < 10:
                return None

            # Test different threshold values
            best_value = state.current_value
            best_mean = np.mean(current_samples)

            # Test values in both directions
            test_values = [
                state.current_value - 0.05,
                state.current_value - 0.02,
                state.current_value + 0.02,
                state.current_value + 0.05,
            ]

            for test_value in test_values:
                if test_value < config.min_value or test_value > config.max_value:
                    continue

                # Get samples for test value
                test_samples = [
                    perf
                    for val, perf, _ in state.performance_samples
                    if abs(val - test_value) < 0.02
                ]

                if len(test_samples) >= 5:
                    test_mean = np.mean(test_samples)

                    # Perform statistical test
                    if len(current_samples) >= 5 and len(test_samples) >= 5:
                        try:
                            # Use Mann-Whitney U test for non-parametric comparison
                            statistic, p_value = stats.mannwhitneyu(
                                test_samples, current_samples, alternative="greater"
                            )

                            # Check if improvement is statistically significant
                            if (
                                p_value < config.significance_threshold
                                and test_mean > best_mean
                            ):
                                # Also check effect size
                                effect_size = abs(test_mean - best_mean) / np.std(
                                    current_samples + test_samples
                                )

                                if effect_size >= self.config.min_effect_size:
                                    best_value = test_value
                                    best_mean = test_mean

                        except Exception as e:
                            logger.warning(f"Statistical test failed: {e}")
                            continue

            # Return best value if it's different from current
            if abs(best_value - state.current_value) > 0.01:
                return best_value
            else:
                return None

        except Exception as e:
            logger.error(f"Statistical update failed for {parameter.value}: {e}")
            return None

    async def _evolutionary_update(
        self,
        parameter: ThresholdParameter,
        state: ThresholdState,
        config: ThresholdConfig,
    ) -> Optional[float]:
        """Update threshold using evolutionary approach"""
        # Simplified evolutionary approach - could be expanded
        try:
            # Generate candidate values (mutations)
            candidates = []
            for _ in range(5):
                # Random mutation around current value
                mutation = np.random.normal(0, config.learning_rate)
                candidate = state.current_value + mutation
                candidate = np.clip(candidate, config.min_value, config.max_value)
                candidates.append(candidate)

            # Add current value
            candidates.append(state.current_value)

            # Select best candidate (this is simplified - would need actual evaluation)
            best_candidate = max(
                candidates, key=lambda x: self._estimate_performance(parameter, x)
            )

            if best_candidate != state.current_value:
                return best_candidate
            else:
                return None

        except Exception as e:
            logger.error(f"Evolutionary update failed for {parameter.value}: {e}")
            return None

    async def _bandit_update(
        self,
        parameter: ThresholdParameter,
        state: ThresholdState,
        config: ThresholdConfig,
    ) -> Optional[float]:
        """Update threshold using multi-armed bandit approach"""
        # Simplified bandit approach - could use UCB, Thompson sampling, etc.
        try:
            # Exploration vs exploitation
            if np.random.random() < config.exploration_rate:
                # Explore: random value within bounds
                new_value = np.random.uniform(config.min_value, config.max_value)
                state.exploration_count += 1
                return new_value
            else:
                # Exploit: move toward best known value
                if state.best_value != state.current_value:
                    direction = np.sign(state.best_value - state.current_value)
                    step = config.learning_rate * direction
                    new_value = state.current_value + step
                    new_value = np.clip(new_value, config.min_value, config.max_value)
                    state.exploitation_count += 1
                    return new_value
                else:
                    return None

        except Exception as e:
            logger.error(f"Bandit update failed for {parameter.value}: {e}")
            return None

    def _validate_threshold_change(
        self, state: ThresholdState, new_value: float, config: ThresholdConfig
    ) -> bool:
        """Validate that a threshold change is reasonable"""
        # Check bounds
        if new_value < config.min_value or new_value > config.max_value:
            return False

        # Check maximum change per update
        change = abs(new_value - state.current_value)
        if change > config.max_change_per_update:
            return False

        # Check if we're oscillating
        if len(state.value_history) >= 3:
            recent_values = [vh[0] for vh in state.value_history[-3:]]
            if len(set(recent_values)) <= 1:  # All same values (oscillating)
                return False

        return True

    def _calculate_current_performance(self, parameter: ThresholdParameter) -> float:
        """Calculate current performance for a parameter"""
        config = self.config.threshold_configs.get(parameter)
        if not config:
            return 0.0

        return self._calculate_performance_score(self.current_metrics, parameter)

    def _estimate_performance(
        self, parameter: ThresholdParameter, threshold_value: float
    ) -> float:
        """Estimate performance for a given threshold value"""
        # This is simplified - could use more sophisticated models
        state = self.threshold_states[parameter]

        # Find similar threshold values in history
        similar_samples = [
            perf
            for val, perf, _ in state.performance_samples
            if abs(val - threshold_value) < 0.1
        ]

        if similar_samples:
            return np.mean(similar_samples)
        else:
            # Fallback to current performance
            return self._calculate_current_performance(parameter)

    async def _check_and_perform_rollbacks(self):
        """Check for performance degradation and rollback if needed"""
        try:
            if not self.config.enable_automatic_rollback:
                return

            # Check recent performance vs historical
            if len(self.metrics_history) < self.config.rollback_window_size * 2:
                return

            recent_metrics = list(self.metrics_history)[
                -self.config.rollback_window_size :
            ]
            older_metrics = list(self.metrics_history)[
                -self.config.rollback_window_size
                * 2 : -self.config.rollback_window_size
            ]

            recent_performance = np.mean([m.success_rate for m in recent_metrics])
            older_performance = np.mean([m.success_rate for m in older_metrics])

            performance_drop = older_performance - recent_performance

            if performance_drop > self.config.performance_degradation_threshold:
                logger.warning(
                    f"Performance degradation detected: {performance_drop:.3f}"
                )

                # Rollback thresholds that were recently updated
                rollback_count = 0
                for parameter, state in self.threshold_states.items():
                    if (
                        datetime.now() - state.last_update
                    ).total_seconds() < 3600:  # Updated in last hour
                        if state.best_value != state.current_value:
                            logger.info(
                                f"Rolling back {parameter.value}: {state.current_value:.3f} -> {state.best_value:.3f}"
                            )
                            state.current_value = state.best_value
                            state.rollbacks += 1
                            rollback_count += 1

                if rollback_count > 0:
                    logger.info(f"Performed {rollback_count} threshold rollbacks")

        except Exception as e:
            logger.error(f"Rollback check failed: {e}")

    def get_current_thresholds(self) -> Dict[ThresholdParameter, float]:
        """Get current threshold values"""
        return {
            param: state.current_value for param, state in self.threshold_states.items()
        }

    def get_threshold_value(self, parameter: ThresholdParameter) -> float:
        """Get current value for a specific threshold parameter"""
        state = self.threshold_states.get(parameter)
        return state.current_value if state else 0.5  # Default fallback

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and metrics"""
        threshold_status = {}
        for param, state in self.threshold_states.items():
            threshold_status[param.value] = {
                "current_value": round(state.current_value, 3),
                "best_value": round(state.best_value, 3),
                "best_performance": round(state.best_performance, 3),
                "total_updates": state.total_updates,
                "successful_updates": state.successful_updates,
                "rollbacks": state.rollbacks,
                "update_success_rate": round(
                    state.successful_updates / max(1, state.total_updates), 3
                ),
                "last_update": state.last_update.isoformat(),
                "samples_collected": len(state.performance_samples),
                "exploration_count": state.exploration_count,
                "exploitation_count": state.exploitation_count,
            }

        return {
            "adaptive_learning_enabled": True,
            "total_samples": self.sample_count,
            "current_performance": {
                "success_rate": round(self.current_metrics.success_rate, 3),
                "average_confidence": round(self.current_metrics.average_confidence, 3),
                "response_time": round(self.current_metrics.response_time, 3),
                "user_satisfaction": round(self.current_metrics.user_satisfaction, 3),
                "search_quality": round(self.current_metrics.search_quality, 3),
                "enhancement_quality": round(
                    self.current_metrics.enhancement_quality, 3
                ),
                "f1_score": round(self.current_metrics.f1_score, 3),
            },
            "threshold_status": threshold_status,
            "config": {
                "update_frequency": self.config.update_frequency,
                "performance_window_size": self.config.performance_window_size,
                "enable_automatic_rollback": self.config.enable_automatic_rollback,
                "statistical_test": self.config.statistical_test,
            },
        }

    async def _persist_state(self):
        """Persist adaptive learning state"""
        try:
            # Save threshold states
            state_file = self.storage_dir / self.config.state_file
            with open(state_file, "wb") as f:
                # Convert deque to list for pickling
                pickle_states = {}
                for param, state in self.threshold_states.items():
                    pickle_state = state.__dict__.copy()
                    pickle_state["performance_samples"] = list(
                        pickle_state["performance_samples"]
                    )
                    pickle_states[param] = pickle_state

                pickle.dump(pickle_states, f)

            # Save current metrics
            metrics_file = self.storage_dir / self.config.metrics_file
            metrics_dict = {
                "sample_count": self.sample_count,
                "last_update": self.last_update.isoformat(),
                "current_metrics": {
                    "success_rate": self.current_metrics.success_rate,
                    "average_confidence": self.current_metrics.average_confidence,
                    "response_time": self.current_metrics.response_time,
                    "user_satisfaction": self.current_metrics.user_satisfaction,
                    "search_quality": self.current_metrics.search_quality,
                    "enhancement_quality": self.current_metrics.enhancement_quality,
                    "precision": self.current_metrics.precision,
                    "recall": self.current_metrics.recall,
                    "f1_score": self.current_metrics.f1_score,
                    "sample_count": self.current_metrics.sample_count,
                    "timestamp": self.current_metrics.timestamp.isoformat(),
                },
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics_dict, f, indent=2)

            logger.debug("Adaptive threshold state persisted")

        except Exception as e:
            logger.error(f"Failed to persist adaptive threshold state: {e}")

    def _load_stored_state(self):
        """Load previously stored adaptive learning state"""
        try:
            # Load threshold states
            state_file = self.storage_dir / self.config.state_file
            if state_file.exists():
                with open(state_file, "rb") as f:
                    pickle_states = pickle.load(f)

                for param, pickle_state in pickle_states.items():
                    # Convert list back to deque
                    pickle_state["performance_samples"] = deque(
                        pickle_state["performance_samples"], maxlen=1000
                    )

                    # Create ThresholdState object
                    state = ThresholdState(
                        parameter=param,
                        current_value=pickle_state["current_value"],
                        best_value=pickle_state["best_value"],
                        best_performance=pickle_state["best_performance"],
                        value_history=pickle_state.get("value_history", []),
                        performance_samples=pickle_state["performance_samples"],
                        total_updates=pickle_state.get("total_updates", 0),
                        successful_updates=pickle_state.get("successful_updates", 0),
                        rollbacks=pickle_state.get("rollbacks", 0),
                        last_update=pickle_state.get("last_update", datetime.now()),
                        exploration_count=pickle_state.get("exploration_count", 0),
                        exploitation_count=pickle_state.get("exploitation_count", 0),
                    )

                    self.threshold_states[param] = state

                logger.info(f"Loaded {len(self.threshold_states)} threshold states")

            # Load metrics
            metrics_file = self.storage_dir / self.config.metrics_file
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_dict = json.load(f)

                self.sample_count = metrics_dict.get("sample_count", 0)

                if "last_update" in metrics_dict:
                    self.last_update = datetime.fromisoformat(
                        metrics_dict["last_update"]
                    )

                current_metrics_dict = metrics_dict.get("current_metrics", {})
                if current_metrics_dict:
                    self.current_metrics = PerformanceMetrics(
                        success_rate=current_metrics_dict.get("success_rate", 0.0),
                        average_confidence=current_metrics_dict.get(
                            "average_confidence", 0.0
                        ),
                        response_time=current_metrics_dict.get("response_time", 0.0),
                        user_satisfaction=current_metrics_dict.get(
                            "user_satisfaction", 0.0
                        ),
                        search_quality=current_metrics_dict.get("search_quality", 0.0),
                        enhancement_quality=current_metrics_dict.get(
                            "enhancement_quality", 0.0
                        ),
                        precision=current_metrics_dict.get("precision", 0.0),
                        recall=current_metrics_dict.get("recall", 0.0),
                        f1_score=current_metrics_dict.get("f1_score", 0.0),
                        sample_count=current_metrics_dict.get("sample_count", 0),
                    )

                    if "timestamp" in current_metrics_dict:
                        self.current_metrics.timestamp = datetime.fromisoformat(
                            current_metrics_dict["timestamp"]
                        )

                logger.info("Loaded adaptive threshold metrics")

        except Exception as e:
            logger.error(f"Failed to load stored adaptive threshold state: {e}")

    async def reset_learning_state(self):
        """Reset all learning state (useful for testing or fresh start)"""
        logger.warning("Resetting adaptive threshold learning state...")

        self.threshold_states.clear()
        self.metrics_history.clear()
        self.sample_count = 0
        self.last_update = datetime.now()
        self.active_experiments.clear()
        self.ab_test_groups.clear()
        self.ab_test_results.clear()

        self.current_metrics = PerformanceMetrics()

        # Clear stored files
        try:
            for filename in [
                self.config.state_file,
                self.config.metrics_file,
                self.config.history_file,
            ]:
                file_path = self.storage_dir / filename
                if file_path.exists():
                    file_path.unlink()

        except Exception as e:
            logger.error(f"Failed to clear stored files: {e}")

        # Re-initialize default thresholds
        self._initialize_default_thresholds()

        logger.info("Adaptive threshold learning state reset complete")


# Factory function
def create_adaptive_threshold_learner(
    tenant_id: str,
    config: Optional[AdaptiveThresholdConfig] = None,
    base_storage_dir: str = "data/adaptive_learning",
) -> AdaptiveThresholdLearner:
    """Create adaptive threshold learner instance"""
    return AdaptiveThresholdLearner(
        tenant_id=tenant_id, config=config, base_storage_dir=base_storage_dir
    )
