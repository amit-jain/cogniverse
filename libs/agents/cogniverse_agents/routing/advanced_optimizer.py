"""
Advanced Multi-Stage Optimizer for Routing Optimization

This module implements a sophisticated routing optimizer using DSPy's advanced
optimization techniques including GEPA, MIPROv2, SIMBA, and BootstrapFewShot.
It learns to improve routing choices over time by analyzing the effectiveness
of different agents for various query types.

Key Features:
- Advanced DSPy optimization techniques (GEPA, MIPROv2, SIMBA)
- Configurable optimizer selection based on dataset size and strategy
- Reward signal generation from user feedback and agent performance
- Experience replay for stable learning
- Confidence calibration and threshold adaptation
- Multi-stage optimization pipeline
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
from dspy.teleprompt import GEPA, SIMBA, BootstrapFewShot, MIPROv2

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class RoutingExperience:
    """Single routing experience for learning"""

    query: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    enhanced_query: str
    chosen_agent: str
    routing_confidence: float

    # Outcome metrics
    search_quality: float  # Quality of search results (0-1)
    agent_success: bool  # Did the agent complete successfully
    user_satisfaction: Optional[float] = None  # Explicit user feedback (0-1)
    processing_time: float = 0.0

    # Computed reward
    reward: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional metadata for orchestration integration
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyOptimizationResult:
    """Result of policy optimization step"""

    optimization_performed: bool
    optimizer_used: str
    training_examples_count: int
    improvement_score: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationMetrics:
    """Metrics for tracking optimization performance"""

    total_experiences: int
    avg_reward: float
    successful_routes: int
    failed_routes: int
    confidence_accuracy: float  # How well confidence predicts success
    agent_preferences: Dict[str, float]  # Learned preferences per agent
    query_type_accuracy: Dict[str, float]  # Accuracy per query type
    improvement_rate: float  # Rate of improvement over time
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AdvancedOptimizerConfig:
    """Configuration for advanced routing optimization"""

    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    experience_replay_size: int = 1000
    update_frequency: int = 10  # Update every N experiences

    # Advanced optimizer selection
    optimizer_strategy: str = (
        "adaptive"  # "adaptive", "gepa", "mipro", "simba", "bootstrap"
    )
    force_optimizer: Optional[str] = (
        None  # Force specific optimizer regardless of data size
    )
    enable_multi_stage: bool = True  # Enable multi-stage optimization pipeline

    # Optimizer thresholds
    bootstrap_threshold: int = 20
    simba_threshold: int = 50
    mipro_threshold: int = 100
    gepa_threshold: int = 200

    # Reward computation
    search_quality_weight: float = 0.4
    agent_success_weight: float = 0.3
    user_satisfaction_weight: float = 0.3
    processing_time_penalty: float = 0.1

    # Exploration vs exploitation
    exploration_epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Confidence calibration
    confidence_learning_rate: float = 0.01
    confidence_smoothing: float = 0.1

    # Storage
    experience_file: str = "routing_experiences.pkl"
    model_file: str = "grpo_routing_model.pkl"
    metrics_file: str = "optimization_metrics.json"
    enable_persistence: bool = True  # Set to False in tests for isolation

    # Minimum experiences before starting optimization
    min_experiences_for_training: int = 50


class AdvancedRoutingOptimizer:
    """
    Advanced multi-stage optimizer for routing decisions

    This class learns from routing experiences to improve future routing decisions.
    It uses reward signals derived from search quality, agent performance, and
    user satisfaction to optimize the routing policy using advanced DSPy techniques
    including GEPA, MIPROv2, SIMBA, and BootstrapFewShot.

    Each tenant gets isolated storage for experiences, models, and metrics.
    """

    def __init__(
        self,
        tenant_id: str,
        llm_config: LLMEndpointConfig,
        telemetry_provider: TelemetryProvider,
        config: Optional[AdvancedOptimizerConfig] = None,
    ):
        """
        Initialize advanced routing optimizer

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            llm_config: LLM endpoint configuration (REQUIRED)
            telemetry_provider: Telemetry provider for artifact persistence.
            config: Optimizer configuration

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id
        self.llm_config = llm_config
        self.config = config or AdvancedOptimizerConfig()
        self._artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

        self.experiences: List[RoutingExperience] = []
        self.experience_replay: List[RoutingExperience] = []
        self.metrics = OptimizationMetrics(
            total_experiences=0,
            avg_reward=0.0,
            successful_routes=0,
            failed_routes=0,
            confidence_accuracy=0.0,
            agent_preferences={},
            query_type_accuracy={},
            improvement_rate=0.0,
        )
        self.advanced_optimizer = None
        self.routing_policy = None
        self.baseline_policy = None
        self.confidence_calibrator = None
        self.current_epsilon = self.config.exploration_epsilon
        self.training_step = 0
        self.last_update = datetime.now()

        self._load_stored_data()
        self._initialize_advanced_components()

        logger.info(
            "Advanced routing optimizer initialized with %d experiences",
            len(self.experiences),
        )

    def _initialize_advanced_components(self):
        """Initialize advanced optimization components"""

        class RoutingPolicySignature(dspy.Signature):
            """Optimized routing decision based on learned policy"""

            query = dspy.InputField(desc="User query to route")
            entities = dspy.InputField(desc="Extracted entities from query")
            relationships = dspy.InputField(desc="Extracted relationships from query")
            enhanced_query = dspy.InputField(
                desc="Enhanced query with relationship context"
            )

            recommended_agent = dspy.OutputField(desc="Best agent for this query")
            confidence = dspy.OutputField(desc="Confidence in routing decision (0-1)")
            reasoning = dspy.OutputField(desc="Reasoning for routing choice")

        class OptimizedRoutingPolicy(dspy.Module):
            def __init__(self):
                super().__init__()
                self.route = dspy.ChainOfThought(RoutingPolicySignature)

            def forward(
                self, query, entities=None, relationships=None, enhanced_query=None
            ):
                entities_str = json.dumps(entities or [], default=str)
                relationships_str = json.dumps(relationships or [], default=str)

                return self.route(
                    query=query,
                    entities=entities_str,
                    relationships=relationships_str,
                    enhanced_query=enhanced_query or query,
                )

        self.routing_policy = OptimizedRoutingPolicy()

        if len(self.experiences) >= self.config.min_experiences_for_training:
            self.advanced_optimizer = self._create_advanced_optimizer()
            logger.info("Advanced optimizer initialized with sufficient experience data")
        else:
            logger.info(
                "Need %d more experiences to start advanced optimization training",
                self.config.min_experiences_for_training - len(self.experiences),
            )

        self._initialize_confidence_calibrator()

    def _create_advanced_optimizer(self):
        """
        Create an advanced multi-stage optimizer using DSPy's GEPA, MIPROv2, and SIMBA

        This implements a sophisticated optimization pipeline:
        1. GEPA for reflective prompt evolution
        2. MIPROv2 for metric-aware instruction optimization
        3. SIMBA for similarity-based memory augmentation
        4. Multi-stage optimization with experience-based learning
        """

        def routing_accuracy_metric(
            gold, pred, trace=None, pred_name=None, pred_trace=None
        ):
            """
            Metric for GEPA optimizer to evaluate routing quality.
            Returns a score between 0 and 1.
            """
            try:
                pred_agent = getattr(pred, "agent_type", None) or getattr(
                    pred, "prediction", None
                )
                gold_agent = (
                    getattr(gold, "agent_type", None) or gold.get("agent_type")
                    if isinstance(gold, dict)
                    else None
                )

                if pred_agent == gold_agent:
                    return 1.0
                else:
                    return 0.0
            except Exception as e:
                logger.warning("Error in routing_accuracy_metric: %s", e)
                return 0.0

        class AdvancedMultiStageOptimizer:
            def __init__(
                self,
                config: AdvancedOptimizerConfig,
                llm_config: LLMEndpointConfig,
            ):
                self.config = config
                self._lm = create_dspy_lm(llm_config)
                self.gepa_optimizer = GEPA(
                    metric=routing_accuracy_metric,
                    auto="light",
                    reflection_lm=self._lm,  # Required by GEPA
                )
                self.mipro_optimizer = MIPROv2(
                    metric=routing_accuracy_metric  # Required by MIPRO
                )
                self.simba_optimizer = SIMBA(
                    metric=routing_accuracy_metric  # Required by SIMBA
                )
                self.bootstrap_optimizer = BootstrapFewShot(
                    metric=routing_accuracy_metric
                )

                self.optimization_stages = [
                    ("bootstrap", self.bootstrap_optimizer, config.bootstrap_threshold),
                    ("simba", self.simba_optimizer, config.simba_threshold),
                    ("mipro", self.mipro_optimizer, config.mipro_threshold),
                    ("gepa", self.gepa_optimizer, config.gepa_threshold),
                ]

                self.optimizers = {
                    "bootstrap": self.bootstrap_optimizer,
                    "simba": self.simba_optimizer,
                    "mipro": self.mipro_optimizer,
                    "gepa": self.gepa_optimizer,
                }

            def compile(self, module, trainset, **kwargs):
                """Advanced multi-stage optimization with configurable strategy"""
                with dspy.context(lm=self._lm):
                    try:
                        dataset_size = len(trainset)
                        logger.info(
                            "Starting optimization with %d examples, strategy: %s",
                            dataset_size,
                            self.config.optimizer_strategy,
                        )

                        selected_optimizer, optimizer_name = self._select_optimizer(
                            dataset_size
                        )

                        optimized_module = self._apply_optimizer(
                            selected_optimizer,
                            optimizer_name,
                            module,
                            trainset,
                            **kwargs,
                        )

                        logger.info("Optimization complete using %s", optimizer_name)
                        return optimized_module

                    except Exception as e:
                        raise RuntimeError(
                            f"Optimization failed with strategy "
                            f"'{self.config.optimizer_strategy}': {e}"
                        ) from e

            def _select_optimizer(self, dataset_size):
                """Select optimizer based on config strategy and dataset size"""
                if self.config.force_optimizer:
                    if self.config.force_optimizer in self.optimizers:
                        optimizer = self.optimizers[self.config.force_optimizer]
                        return optimizer, self.config.force_optimizer
                    raise ValueError(
                        f"Unknown forced optimizer '{self.config.force_optimizer}'. "
                        f"Valid options: {list(self.optimizers)}"
                    )

                if self.config.optimizer_strategy == "adaptive":
                    applicable_optimizers = [
                        (name, optimizer)
                        for name, optimizer, min_size in self.optimization_stages
                        if dataset_size >= min_size
                    ]

                    if applicable_optimizers:
                        name, optimizer = applicable_optimizers[-1]
                        return optimizer, name
                    return self.bootstrap_optimizer, "bootstrap"

                elif self.config.optimizer_strategy in self.optimizers:
                    optimizer = self.optimizers[self.config.optimizer_strategy]
                    return optimizer, self.config.optimizer_strategy

                raise ValueError(
                    f"Unknown optimizer strategy '{self.config.optimizer_strategy}'. "
                    f"Valid options: adaptive, {list(self.optimizers)}"
                )

            def _apply_optimizer(
                self, optimizer, optimizer_name, module, trainset, **kwargs
            ):
                """Apply specific optimizer with appropriate parameters based on API"""

                if optimizer_name == "gepa":
                    # GEPA: Reflective prompt evolution
                    # API: compile(student, trainset, teacher=None, valset=None)
                    return optimizer.compile(
                        module,
                        trainset=trainset,
                        valset=kwargs.get("valset", None),
                    )

                elif optimizer_name == "mipro":
                    # MIPROv2: Metric-aware instruction optimization
                    # API accepts max_bootstrapped_demos and max_labeled_demos
                    return optimizer.compile(
                        module,
                        trainset=trainset,
                        max_bootstrapped_demos=kwargs.get("max_bootstrapped_demos", 4),
                        max_labeled_demos=kwargs.get("max_labeled_demos", 8),
                        valset=kwargs.get("valset", None),
                    )

                elif optimizer_name == "simba":
                    # SIMBA: Similarity-based memory augmentation
                    # API: compile(student, trainset, seed=0)
                    return optimizer.compile(
                        module,
                        trainset=trainset,
                        seed=kwargs.get("seed", 0),
                    )

                else:
                    # Bootstrap or fallback
                    # API: compile(student, teacher=None, trainset)
                    return optimizer.compile(
                        module,
                        trainset=trainset,
                        teacher=kwargs.get("teacher", None),
                    )

            def get_optimization_info(self, dataset_size):
                """Get information about which optimizers will be used"""
                applicable = [
                    (name, min_size)
                    for name, _, min_size in self.optimization_stages
                    if dataset_size >= min_size
                ]
                return {
                    "dataset_size": dataset_size,
                    "applicable_optimizers": applicable,
                    "primary_optimizer": (
                        applicable[-1][0] if applicable else "bootstrap"
                    ),
                    "optimization_stages": len(applicable),
                }

        return AdvancedMultiStageOptimizer(self.config, self.llm_config)

    def _initialize_confidence_calibrator(self):
        """Initialize confidence calibration component"""
        try:

            class ConfidenceCalibratorSignature(dspy.Signature):
                """Calibrate confidence scores based on historical accuracy"""

                raw_confidence = dspy.InputField(
                    desc="Raw confidence from routing model"
                )
                query_complexity = dspy.InputField(
                    desc="Complexity indicators of the query"
                )
                historical_accuracy = dspy.InputField(
                    desc="Historical accuracy for similar queries"
                )

                calibrated_confidence = dspy.OutputField(
                    desc="Calibrated confidence score (0-1)"
                )

            class ConfidenceCalibrator(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.calibrate = dspy.ChainOfThought(ConfidenceCalibratorSignature)

                def forward(
                    self, raw_confidence, query_complexity=0.5, historical_accuracy=0.7
                ):
                    return self.calibrate(
                        raw_confidence=str(raw_confidence),
                        query_complexity=str(query_complexity),
                        historical_accuracy=str(historical_accuracy),
                    )

            self.confidence_calibrator = ConfidenceCalibrator()
            logger.info("Confidence calibrator initialized")

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize confidence calibrator: {e}"
            ) from e

    async def record_routing_experience(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhanced_query: str,
        chosen_agent: str,
        routing_confidence: float,
        search_quality: float,
        agent_success: bool,
        processing_time: float = 0.0,
        user_satisfaction: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Record a routing experience and compute reward

        Args:
            query: Original user query
            entities: Extracted entities
            relationships: Extracted relationships
            enhanced_query: Enhanced query with relationships
            chosen_agent: Agent that was selected
            routing_confidence: Confidence in routing decision
            search_quality: Quality of search results (0-1)
            agent_success: Whether agent completed successfully
            processing_time: Time taken for processing
            user_satisfaction: Optional explicit user feedback (0-1)
            metadata: Optional additional metadata (e.g., from orchestration)

        Returns:
            Computed reward for this experience
        """
        reward = self._compute_reward(
            search_quality=search_quality,
            agent_success=agent_success,
            processing_time=processing_time,
            user_satisfaction=user_satisfaction,
        )

        experience = RoutingExperience(
            query=query,
            entities=entities,
            relationships=relationships,
            enhanced_query=enhanced_query,
            chosen_agent=chosen_agent,
            routing_confidence=routing_confidence,
            search_quality=search_quality,
            agent_success=agent_success,
            user_satisfaction=user_satisfaction,
            processing_time=processing_time,
            reward=reward,
            metadata=metadata or {},
        )

        await self._store_experience(experience)

        if self._should_trigger_optimization():
            await self._run_optimization_step()

        logger.debug(
            f"Recorded routing experience: reward={reward:.3f}, agent={chosen_agent}"
        )
        return reward

    def _compute_reward(
        self,
        search_quality: float,
        agent_success: bool,
        processing_time: float,
        user_satisfaction: Optional[float] = None,
    ) -> float:
        """Compute reward signal from routing outcome"""

        reward = (
            search_quality * self.config.search_quality_weight
            + (1.0 if agent_success else 0.0) * self.config.agent_success_weight
        )

        if user_satisfaction is not None:
            reward += user_satisfaction * self.config.user_satisfaction_weight
        else:
            total_weight = (
                self.config.search_quality_weight + self.config.agent_success_weight
            )
            reward = reward / total_weight

        if processing_time > 0:
            # Penalty increases with processing time (sigmoid-like)
            time_penalty = self.config.processing_time_penalty * (
                1.0 - 1.0 / (1.0 + processing_time / 10.0)
            )
            reward = max(0.0, reward - time_penalty)

        return min(1.0, max(0.0, reward))  # Clamp to [0, 1]

    async def _store_experience(self, experience: RoutingExperience):
        """Store routing experience for learning"""
        self.experiences.append(experience)

        self.experience_replay.append(experience)
        if len(self.experience_replay) > self.config.experience_replay_size:
            self.experience_replay.pop(0)

        self._update_metrics(experience)

        if len(self.experiences) % 10 == 0:
            await self._persist_data()

    def _update_metrics(self, experience: RoutingExperience):
        """Update optimization metrics with new experience"""
        self.metrics.total_experiences += 1

        if experience.agent_success:
            self.metrics.successful_routes += 1
        else:
            self.metrics.failed_routes += 1

        if self.metrics.avg_reward == 0.0:
            self.metrics.avg_reward = experience.reward
        else:
            alpha = 0.1
            self.metrics.avg_reward = (
                1 - alpha
            ) * self.metrics.avg_reward + alpha * experience.reward

        agent = experience.chosen_agent
        if agent not in self.metrics.agent_preferences:
            self.metrics.agent_preferences[agent] = experience.reward
        else:
            self.metrics.agent_preferences[agent] = (
                0.9 * self.metrics.agent_preferences[agent] + 0.1 * experience.reward
            )

        if len(self.experiences) > 1:
            confidence_predictions = [
                exp.routing_confidence for exp in self.experiences[-100:]
            ]
            actual_outcomes = [
                1.0 if exp.agent_success else 0.0 for exp in self.experiences[-100:]
            ]

            if confidence_predictions and actual_outcomes:
                try:
                    corr_matrix = np.corrcoef(confidence_predictions, actual_outcomes)
                    correlation = (
                        corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    )
                    self.metrics.confidence_accuracy = max(0.0, correlation)
                except Exception as e:
                    raise RuntimeError(
                        "Failed to compute confidence correlation: "
                        f"conf_values={confidence_predictions[:3]}..., "
                        f"outcome_values={actual_outcomes[:3]}...: {e}"
                    ) from e

        self.metrics.last_updated = datetime.now()

    def _should_trigger_optimization(self) -> bool:
        """Determine if optimization should be triggered"""
        if len(self.experiences) < self.config.min_experiences_for_training:
            return False

        if len(self.experiences) % self.config.update_frequency == 0:
            return True

        recent_rewards = [exp.reward for exp in self.experiences[-10:]]
        if len(recent_rewards) >= 10:
            recent_avg = np.mean(recent_rewards)
            if recent_avg < self.metrics.avg_reward - 0.1:
                return True

        return False

    async def _run_optimization_step(self):
        """Run one step of GRPO optimization"""
        if (
            self.advanced_optimizer is None
            and len(self.experiences) >= self.config.min_experiences_for_training
        ):
            logger.info("Lazy initializing advanced optimizer after reaching threshold")
            self.advanced_optimizer = self._create_advanced_optimizer()
            logger.info("Advanced optimizer initialized")

        if (
            not self.advanced_optimizer
            or len(self.experience_replay) < self.config.batch_size
        ):
            return

        try:
            logger.info("Running GRPO optimization step...")

            batch_experiences = np.random.choice(
                self.experience_replay,
                size=min(self.config.batch_size, len(self.experience_replay)),
                replace=False,
            ).tolist()

            training_examples = []
            for exp in batch_experiences:
                example = dspy.Example(
                    query=exp.query,
                    entities=json.dumps(exp.entities, default=str),
                    relationships=json.dumps(exp.relationships, default=str),
                    enhanced_query=exp.enhanced_query,
                    recommended_agent=exp.chosen_agent,
                    confidence=str(exp.routing_confidence),
                    reasoning=f"Selected based on reward={exp.reward:.3f}",
                ).with_inputs("query", "entities", "relationships", "enhanced_query")

                training_examples.append(example)

            if self.routing_policy and training_examples:
                optimized_policy = self.advanced_optimizer.compile(
                    self.routing_policy,
                    trainset=training_examples,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=8,
                )

                self.routing_policy = optimized_policy
                self.training_step += 1

                self.current_epsilon = max(
                    self.config.min_epsilon,
                    self.current_epsilon * self.config.epsilon_decay,
                )

                logger.info(
                    "GRPO optimization step %d complete. Epsilon: %.3f",
                    self.training_step,
                    self.current_epsilon,
                )

                if len(self.experiences) > 100:
                    old_rewards = [exp.reward for exp in self.experiences[-200:-100]]
                    new_rewards = [exp.reward for exp in self.experiences[-100:]]

                    if old_rewards and new_rewards:
                        self.metrics.improvement_rate = np.mean(new_rewards) - np.mean(
                            old_rewards
                        )

        except Exception as e:
            raise RuntimeError(
                f"GRPO optimization step {self.training_step} failed "
                f"for tenant '{self.tenant_id}': {e}"
            ) from e

    async def get_routing_recommendations(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get routing recommendations using the optimized policy

        Args:
            query: User query
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            Routing recommendations with confidence and reasoning
        """
        try:
            if (
                not self.routing_policy
                or len(self.experiences) < self.config.min_experiences_for_training
            ):
                return self._get_baseline_recommendations(
                    query, entities, relationships
                )

            enhanced_query = query
            prediction = self.routing_policy(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
            )

            recommended_agent = prediction.recommended_agent
            raw_confidence = float(prediction.confidence)
            reasoning = prediction.reasoning

            calibrated_confidence = await self._calibrate_confidence(
                raw_confidence, query, entities, relationships
            )

            return {
                "recommended_agent": recommended_agent,
                "confidence": calibrated_confidence,
                "reasoning": f"Optimized policy: {reasoning}",
                "optimization_ready": True,
                "experiences_count": len(self.experiences),
                "training_step": self.training_step,
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to get routing recommendations for query '{query[:80]}': {e}"
            ) from e

    def _get_baseline_recommendations(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Provide baseline recommendations when optimization not ready"""

        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["video", "visual", "watch", "clip", "footage"]
        ):
            agent = "search_agent"
            confidence = 0.8
            reasoning = "Query contains video-related keywords"
        elif any(
            word in query_lower
            for word in ["summary", "summarize", "overview", "brief"]
        ):
            agent = "summarizer_agent"
            confidence = 0.7
            reasoning = "Query requests summary/overview"
        elif any(
            word in query_lower for word in ["detail", "detailed", "analysis", "report"]
        ):
            agent = "detailed_report_agent"
            confidence = 0.7
            reasoning = "Query requests detailed analysis"
        else:
            # Default to video search for general queries
            agent = "search_agent"
            confidence = 0.6
            reasoning = "Default routing to video search agent"

        if agent in self.metrics.agent_preferences:
            agent_performance = self.metrics.agent_preferences[agent]
            confidence = min(1.0, confidence * (0.8 + 0.4 * agent_performance))

        return {
            "recommended_agent": agent,
            "confidence": confidence,
            "reasoning": reasoning,
            "optimization_ready": False,
            "experiences_count": len(self.experiences),
            "training_step": self.training_step,
        }

    async def optimize_routing_decision(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhanced_query: str,
        baseline_prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply GRPO optimization to improve routing decision

        Args:
            query: User query
            entities: Extracted entities
            relationships: Extracted relationships
            enhanced_query: Enhanced query
            baseline_prediction: Baseline routing prediction

        Returns:
            Optimized routing decision
        """
        try:
            if not self.routing_policy or not self.advanced_optimizer:
                return self._apply_baseline_improvements(baseline_prediction)

            if np.random.random() < self.current_epsilon:
                return self._add_exploration_noise(baseline_prediction)

            optimized_prediction = self.routing_policy(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
            )

            # Extract optimized values
            optimized_agent = optimized_prediction.recommended_agent
            raw_confidence = float(optimized_prediction.confidence)
            reasoning = optimized_prediction.reasoning

            calibrated_confidence = await self._calibrate_confidence(
                raw_confidence, query, entities, relationships
            )

            return {
                "recommended_agent": optimized_agent,
                "confidence": calibrated_confidence,
                "reasoning": f"GRPO-optimized: {reasoning}",
                "optimization_applied": True,
                "exploration_epsilon": self.current_epsilon,
                "training_step": self.training_step,
            }

        except Exception as e:
            raise RuntimeError(
                f"GRPO routing optimization failed for query '{query[:80]}': {e}"
            ) from e

    async def _calibrate_confidence(
        self,
        raw_confidence: float,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> float:
        """Calibrate confidence score based on historical accuracy"""

        if not self.confidence_calibrator:
            return raw_confidence

        try:
            query_complexity = (
                min(1.0, len(query.split()) / 20.0)  # Word count complexity
                + min(1.0, len(entities) / 10.0)  # Entity complexity
                + min(1.0, len(relationships) / 5.0)  # Relationship complexity
            ) / 3.0

            historical_accuracy = self._get_historical_accuracy_for_query_type(query)
            calibrated_result = self.confidence_calibrator(
                raw_confidence=raw_confidence,
                query_complexity=query_complexity,
                historical_accuracy=historical_accuracy,
            )

            calibrated_confidence = float(calibrated_result.calibrated_confidence)
            return max(0.0, min(1.0, calibrated_confidence))

        except Exception as e:
            raise RuntimeError(
                f"Confidence calibration failed for raw_confidence={raw_confidence}: {e}"
            ) from e

    def _get_historical_accuracy_for_query_type(self, query: str) -> float:
        """Get historical accuracy for similar query types"""
        if not self.experiences:
            return 0.7

        query_words = set(query.lower().split())

        similar_experiences = []
        for exp in self.experiences[-200:]:
            exp_words = set(exp.query.lower().split())
            similarity = len(query_words.intersection(exp_words)) / max(
                len(query_words), len(exp_words), 1
            )

            if similarity > 0.3:
                similar_experiences.append(exp)

        if not similar_experiences:
            return self.metrics.confidence_accuracy

        success_rate = sum(1 for exp in similar_experiences if exp.agent_success) / len(
            similar_experiences
        )
        return success_rate

    def _apply_baseline_improvements(
        self, baseline_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply basic improvements to baseline prediction"""
        improved = baseline_prediction.copy()

        agent = baseline_prediction.get("recommended_agent", "search_agent")
        if agent in self.metrics.agent_preferences:
            agent_performance = self.metrics.agent_preferences[agent]
            confidence_boost = (agent_performance - 0.5) * 0.1  # Small adjustment

            original_confidence = baseline_prediction.get("confidence", 0.7)
            improved["confidence"] = max(
                0.0, min(1.0, original_confidence + confidence_boost)
            )

        improved["optimization_applied"] = False
        improved["baseline_improvements"] = True

        return improved

    def _add_exploration_noise(
        self, baseline_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add exploration noise to baseline prediction"""
        exploration = baseline_prediction.copy()

        confidence_noise = np.random.normal(0, 0.05)
        original_confidence = baseline_prediction.get("confidence", 0.7)
        exploration["confidence"] = max(
            0.0, min(1.0, original_confidence + confidence_noise)
        )

        if np.random.random() < 0.1:
            agents = ["search_agent", "summarizer_agent", "detailed_report_agent"]
            current_agent = baseline_prediction.get("recommended_agent", "search_agent")
            other_agents = [a for a in agents if a != current_agent]
            if other_agents:
                exploration["recommended_agent"] = np.random.choice(other_agents)
                exploration["reasoning"] = (
                    f"Exploration: trying {exploration['recommended_agent']}"
                )

        exploration["exploration"] = True

        return exploration

    async def _persist_data(self):
        """Persist experiences and metrics via ArtifactManager."""
        try:
            if self.experiences:
                demos = []
                for exp in self.experiences:
                    exp_dict = asdict(exp)
                    exp_dict["timestamp"] = exp_dict["timestamp"].isoformat()
                    demos.append(
                        {
                            "input": json.dumps(
                                {
                                    "query": exp_dict["query"],
                                    "entities": exp_dict["entities"],
                                    "relationships": exp_dict["relationships"],
                                    "enhanced_query": exp_dict["enhanced_query"],
                                },
                                default=str,
                            ),
                            "output": json.dumps(
                                {
                                    "chosen_agent": exp_dict["chosen_agent"],
                                    "routing_confidence": exp_dict[
                                        "routing_confidence"
                                    ],
                                    "reward": exp_dict["reward"],
                                },
                                default=str,
                            ),
                            "metadata": json.dumps(
                                {
                                    "search_quality": exp_dict["search_quality"],
                                    "agent_success": exp_dict["agent_success"],
                                    "timestamp": exp_dict["timestamp"],
                                },
                                default=str,
                            ),
                        }
                    )
                await self._artifact_manager.save_demonstrations(
                    "routing_optimizer", demos
                )

            metrics_dict = {
                "total_experiences": self.metrics.total_experiences,
                "avg_reward": float(self.metrics.avg_reward),
                "successful_routes": self.metrics.successful_routes,
                "failed_routes": self.metrics.failed_routes,
                "confidence_accuracy": float(self.metrics.confidence_accuracy),
                "agent_preferences": self.metrics.agent_preferences,
                "query_type_accuracy": self.metrics.query_type_accuracy,
                "improvement_rate": float(self.metrics.improvement_rate),
                "training_step": self.training_step,
                "current_epsilon": float(self.current_epsilon),
            }
            await self._artifact_manager.log_optimization_run(
                "routing_optimizer", metrics_dict
            )

            logger.debug("Persisted %d experiences and metrics", len(self.experiences))

        except Exception as e:
            raise RuntimeError(
                f"Failed to persist optimization data for tenant '{self.tenant_id}': {e}"
            ) from e

    def _load_stored_data(self):
        """Load previously stored experiences and metrics from telemetry."""
        if not self.config.enable_persistence:
            logger.info("Persistence disabled, skipping data loading")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            task = asyncio.ensure_future(
                self._artifact_manager.load_demonstrations("routing_optimizer")
            )
            task.add_done_callback(self._on_demos_loaded)
        else:
            demos = asyncio.run(
                self._artifact_manager.load_demonstrations("routing_optimizer")
            )
            self._apply_loaded_demos(demos)

    def _on_demos_loaded(self, future: asyncio.Future):
        """Callback for async demo loading."""
        demos = future.result()
        self._apply_loaded_demos(demos)

    @staticmethod
    def _parse_demo_field(value: Any) -> dict:
        """Parse a demo field that may be a JSON string or already a dict."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value) if value else {}
        return {}

    def _apply_loaded_demos(self, demos):
        """Apply loaded demos to experience buffer."""
        if not demos:
            return
        for demo in demos:
            inp = self._parse_demo_field(demo.get("input", "{}"))
            out = self._parse_demo_field(demo.get("output", "{}"))
            meta = self._parse_demo_field(demo.get("metadata", "{}"))
            exp = RoutingExperience(
                query=inp.get("query", ""),
                entities=inp.get("entities", []),
                relationships=inp.get("relationships", []),
                enhanced_query=inp.get("enhanced_query", ""),
                chosen_agent=out.get("chosen_agent", ""),
                routing_confidence=float(
                    out.get("routing_confidence", 0.0)
                ),
                search_quality=float(meta.get("search_quality", 0.0)),
                agent_success=bool(meta.get("agent_success", False)),
                reward=float(out.get("reward", 0.0))
                if out.get("reward") is not None
                else None,
                timestamp=datetime.fromisoformat(meta["timestamp"])
                if "timestamp" in meta
                else datetime.now(),
            )
            self.experiences.append(exp)

        self.experience_replay = self.experiences[
            -self.config.experience_replay_size :
        ]
        logger.info("Loaded %d routing experiences", len(self.experiences))

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and metrics"""
        return {
            "optimizer_ready": self.advanced_optimizer is not None,
            "total_experiences": len(self.experiences),
            "experience_replay_size": len(self.experience_replay),
            "training_step": self.training_step,
            "current_epsilon": self.current_epsilon,
            "metrics": {
                "total_experiences": self.metrics.total_experiences,
                "avg_reward": round(self.metrics.avg_reward, 3),
                "successful_routes": self.metrics.successful_routes,
                "failed_routes": self.metrics.failed_routes,
                "success_rate": round(
                    self.metrics.successful_routes
                    / max(1, self.metrics.total_experiences),
                    3,
                ),
                "confidence_accuracy": round(self.metrics.confidence_accuracy, 3),
                "agent_preferences": {
                    k: round(v, 3) for k, v in self.metrics.agent_preferences.items()
                },
                "improvement_rate": round(self.metrics.improvement_rate, 3),
                "last_updated": self.metrics.last_updated.isoformat(),
            },
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "experience_replay_size": self.config.experience_replay_size,
                "min_experiences_for_training": self.config.min_experiences_for_training,
            },
        }

    async def optimize_routing_policy(self) -> Dict[str, Any]:
        """
        Trigger routing policy optimization

        This is called by UnifiedOptimizer to run routing optimization
        as part of the unified optimization cycle.

        Returns:
            Optimization results
        """
        try:
            # Run optimization step
            await self._run_optimization_step()

            return {
                "status": "success",
                "total_experiences": len(self.experiences),
                "training_step": self.training_step,
                "metrics": {
                    "avg_reward": self.metrics.avg_reward,
                    "successful_routes": self.metrics.successful_routes,
                    "failed_routes": self.metrics.failed_routes,
                },
            }
        except Exception as e:
            raise RuntimeError(
                f"Routing policy optimization failed for tenant '{self.tenant_id}': {e}"
            ) from e

    async def generate_synthetic_training_data(
        self,
        count: int = 100,
        backend: Optional[Any] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Any] = None,
    ) -> int:
        """
        Generate synthetic training data using libs/synthetic system

        Args:
            count: Number of synthetic examples to generate
            backend: Optional Backend instance for content sampling
            backend_config: Backend configuration with profiles
            generator_config: Optional SyntheticGeneratorConfig for DSPy modules

        Returns:
            Number of examples added to experiences
        """
        from cogniverse_synthetic import (
            SyntheticDataRequest,
            SyntheticDataService,
        )

        logger.info("Generating %d synthetic routing examples...", count)

        # Generate synthetic data directly
        service = SyntheticDataService(
            backend=backend,
            backend_config=backend_config,
            generator_config=generator_config,
        )
        request = SyntheticDataRequest(optimizer="routing", count=count)
        response = await service.generate(request)

        initial_count = len(self.experiences)
        for example_data in response.data:
            experience = RoutingExperience(
                query=example_data["query"],
                entities=example_data["entities"],
                relationships=example_data["relationships"],
                enhanced_query=example_data["enhanced_query"],
                chosen_agent=example_data["chosen_agent"],
                routing_confidence=example_data["routing_confidence"],
                search_quality=example_data["search_quality"],
                agent_success=example_data["agent_success"],
                user_satisfaction=example_data.get("user_satisfaction"),
                processing_time=example_data.get("processing_time", 0.0),
                reward=example_data.get("reward"),
                metadata=example_data.get("metadata", {}),
            )
            self.experiences.append(experience)
            self.experience_replay.append(experience)

        added_count = len(self.experiences) - initial_count

        logger.info(
            "Added %d synthetic examples to routing experiences (total: %d)",
            added_count,
            len(self.experiences),
        )

        return added_count

    async def reset_optimization(self):
        """Reset optimization state (useful for testing or fresh start)"""
        logger.warning("Resetting advanced optimization state...")

        self.experiences.clear()
        self.experience_replay.clear()
        self.training_step = 0
        self.current_epsilon = self.config.exploration_epsilon

        self.metrics = OptimizationMetrics(
            total_experiences=0,
            avg_reward=0.0,
            successful_routes=0,
            failed_routes=0,
            confidence_accuracy=0.0,
            agent_preferences={},
            query_type_accuracy={},
            improvement_rate=0.0,
        )

        self._initialize_advanced_components()

        logger.info("Advanced optimization reset complete")
