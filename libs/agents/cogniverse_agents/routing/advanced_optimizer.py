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

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# DSPy 3.0 imports
import dspy
import numpy as np
from dspy.teleprompt import GEPA, SIMBA, BootstrapFewShot, MIPROv2

from cogniverse_core.common.tenant_utils import get_tenant_storage_path
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

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
        config: Optional[AdvancedOptimizerConfig] = None,
        base_storage_dir: str = "data/optimization",
    ):
        """
        Initialize advanced routing optimizer

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            llm_config: LLM endpoint configuration (REQUIRED)
            config: Optimizer configuration
            base_storage_dir: Base directory for storage

        Raises:
            ValueError: If tenant_id is empty or None
        """
        if not tenant_id:
            raise ValueError("tenant_id is required - no default tenant")

        self.tenant_id = tenant_id
        self.llm_config = llm_config
        self.config = config or AdvancedOptimizerConfig()

        # Tenant-specific storage directory with org/tenant structure
        self.storage_dir = get_tenant_storage_path(base_storage_dir, tenant_id)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Experience storage
        self.experiences: List[RoutingExperience] = []
        self.experience_replay: List[RoutingExperience] = []

        # Metrics tracking
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

        # GRPO components
        self.advanced_optimizer = None
        self.routing_policy = None
        self.baseline_policy = None

        # Confidence calibration
        self.confidence_calibrator = None

        # State
        self.current_epsilon = self.config.exploration_epsilon
        self.training_step = 0
        self.last_update = datetime.now()

        # Load existing data
        self._load_stored_data()

        # Initialize advanced optimization components
        self._initialize_advanced_components()

        logger.info(
            f"Advanced routing optimizer initialized with {len(self.experiences)} experiences"
        )

    def _initialize_advanced_components(self):
        """Initialize advanced optimization components"""

        # Create routing policy signatures
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

        # Create policy module
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

        # Initialize advanced optimizer
        if len(self.experiences) >= self.config.min_experiences_for_training:
            self.advanced_optimizer = self._create_advanced_optimizer()
            logger.info(
                "Advanced optimizer initialized with sufficient experience data"
            )
        else:
            logger.info(
                f"Need {self.config.min_experiences_for_training - len(self.experiences)} more experiences to start advanced optimization training"
            )

        # Initialize confidence calibrator
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
                # Extract predicted and gold agent types
                pred_agent = getattr(pred, "agent_type", None) or getattr(
                    pred, "prediction", None
                )
                gold_agent = (
                    getattr(gold, "agent_type", None) or gold.get("agent_type")
                    if isinstance(gold, dict)
                    else None
                )

                # Basic accuracy: 1 if correct, 0 if incorrect
                if pred_agent == gold_agent:
                    return 1.0
                else:
                    return 0.0
            except Exception as e:
                logger.warning(f"Error in routing_accuracy_metric: {e}")
                return 0.0

        class AdvancedMultiStageOptimizer:
            def __init__(
                self,
                config: AdvancedOptimizerConfig,
                llm_config: LLMEndpointConfig,
            ):
                self.config = config

                # Create LM via centralized factory (stored for dspy.context scoping)
                self._lm = create_dspy_lm(llm_config)

                # Initialize all advanced optimizers with required parameters
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
                    metric=routing_accuracy_metric  # Optional but good for consistency
                )

                # Optimization strategy based on config and data size
                self.optimization_stages = [
                    ("bootstrap", self.bootstrap_optimizer, config.bootstrap_threshold),
                    ("simba", self.simba_optimizer, config.simba_threshold),
                    ("mipro", self.mipro_optimizer, config.mipro_threshold),
                    ("gepa", self.gepa_optimizer, config.gepa_threshold),
                ]

                # Optimizer mapping for direct selection
                self.optimizers = {
                    "bootstrap": self.bootstrap_optimizer,
                    "simba": self.simba_optimizer,
                    "mipro": self.mipro_optimizer,
                    "gepa": self.gepa_optimizer,
                }

            def compile(self, module, trainset, **kwargs):
                """Advanced multi-stage optimization with configurable strategy"""
                # Scope the LM to this optimizer's compile call
                with dspy.context(lm=self._lm):
                    try:
                        dataset_size = len(trainset)
                        logger.info(
                            f"Starting optimization with {dataset_size} examples, strategy: {self.config.optimizer_strategy}"
                        )

                        # Select optimizer based on configuration
                        selected_optimizer, optimizer_name = self._select_optimizer(
                            dataset_size
                        )

                        # Apply the selected optimization
                        optimized_module = self._apply_optimizer(
                            selected_optimizer,
                            optimizer_name,
                            module,
                            trainset,
                            **kwargs,
                        )

                        logger.info(f"Optimization complete using {optimizer_name}")
                        return optimized_module

                    except Exception as e:
                        logger.error(
                            f"Optimization failed: {e}, falling back to bootstrap"
                        )
                        return self.bootstrap_optimizer.compile(
                            module, trainset=trainset, **kwargs
                        )

            def _select_optimizer(self, dataset_size):
                """Select optimizer based on config strategy and dataset size"""
                # Force specific optimizer if configured
                if self.config.force_optimizer:
                    if self.config.force_optimizer in self.optimizers:
                        optimizer = self.optimizers[self.config.force_optimizer]
                        return optimizer, self.config.force_optimizer
                    else:
                        logger.warning(
                            f"Unknown forced optimizer: {self.config.force_optimizer}"
                        )

                # Strategy-based selection
                if self.config.optimizer_strategy == "adaptive":
                    # Adaptive: select best optimizer based on dataset size
                    applicable_optimizers = [
                        (name, optimizer)
                        for name, optimizer, min_size in self.optimization_stages
                        if dataset_size >= min_size
                    ]

                    if applicable_optimizers:
                        # Use the most advanced applicable optimizer
                        name, optimizer = applicable_optimizers[-1]
                        return optimizer, name
                    else:
                        # Fallback to bootstrap
                        return self.bootstrap_optimizer, "bootstrap"

                elif self.config.optimizer_strategy in self.optimizers:
                    # Direct strategy selection
                    optimizer = self.optimizers[self.config.optimizer_strategy]
                    return optimizer, self.config.optimizer_strategy

                else:
                    # Unknown strategy, use bootstrap
                    logger.warning(
                        f"Unknown optimizer strategy: {self.config.optimizer_strategy}"
                    )
                    return self.bootstrap_optimizer, "bootstrap"

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
            logger.error(f"Failed to initialize confidence calibrator: {e}")
            self.confidence_calibrator = None

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
        # Compute reward
        reward = self._compute_reward(
            search_quality=search_quality,
            agent_success=agent_success,
            processing_time=processing_time,
            user_satisfaction=user_satisfaction,
        )

        # Create experience
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

        # Store experience
        await self._store_experience(experience)

        # Trigger optimization if conditions met
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

        # Base reward from search quality and agent success
        reward = (
            search_quality * self.config.search_quality_weight
            + (1.0 if agent_success else 0.0) * self.config.agent_success_weight
        )

        # Add user satisfaction if available
        if user_satisfaction is not None:
            reward += user_satisfaction * self.config.user_satisfaction_weight
        else:
            # If no user satisfaction, normalize the weights
            total_weight = (
                self.config.search_quality_weight + self.config.agent_success_weight
            )
            reward = reward / total_weight

        # Apply processing time penalty (longer processing = lower reward)
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

        # Add to experience replay buffer
        self.experience_replay.append(experience)
        if len(self.experience_replay) > self.config.experience_replay_size:
            self.experience_replay.pop(0)

        # Update metrics
        self._update_metrics(experience)

        # Persist data periodically
        if len(self.experiences) % 10 == 0:
            await self._persist_data()

    def _update_metrics(self, experience: RoutingExperience):
        """Update optimization metrics with new experience"""
        self.metrics.total_experiences += 1

        if experience.agent_success:
            self.metrics.successful_routes += 1
        else:
            self.metrics.failed_routes += 1

        # Update average reward (moving average)
        if self.metrics.avg_reward == 0.0:
            self.metrics.avg_reward = experience.reward
        else:
            alpha = 0.1  # Learning rate for moving average
            self.metrics.avg_reward = (
                1 - alpha
            ) * self.metrics.avg_reward + alpha * experience.reward

        # Update agent preferences
        agent = experience.chosen_agent
        if agent not in self.metrics.agent_preferences:
            self.metrics.agent_preferences[agent] = experience.reward
        else:
            # Moving average
            self.metrics.agent_preferences[agent] = (
                0.9 * self.metrics.agent_preferences[agent] + 0.1 * experience.reward
            )

        # Update confidence accuracy (how well confidence predicts success)
        if len(self.experiences) > 1:
            confidence_predictions = [
                exp.routing_confidence for exp in self.experiences[-100:]
            ]
            actual_outcomes = [
                1.0 if exp.agent_success else 0.0 for exp in self.experiences[-100:]
            ]

            if confidence_predictions and actual_outcomes:
                # Compute simple correlation between confidence and success
                try:
                    # Use numpy for correlation calculation (more reliable than scipy dependency)
                    corr_matrix = np.corrcoef(confidence_predictions, actual_outcomes)
                    correlation = (
                        corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    )
                    self.metrics.confidence_accuracy = max(0.0, correlation)
                except Exception:
                    # Fallback calculation - measure alignment between confidence and success
                    avg_conf = np.mean(confidence_predictions)
                    avg_success = np.mean(actual_outcomes)
                    self.metrics.confidence_accuracy = 1.0 - abs(avg_conf - avg_success)

        self.metrics.last_updated = datetime.now()

    def _should_trigger_optimization(self) -> bool:
        """Determine if optimization should be triggered"""
        if len(self.experiences) < self.config.min_experiences_for_training:
            return False

        # Trigger every N experiences
        if len(self.experiences) % self.config.update_frequency == 0:
            return True

        # Trigger if performance is declining
        recent_rewards = [exp.reward for exp in self.experiences[-10:]]
        if len(recent_rewards) >= 10:
            recent_avg = np.mean(recent_rewards)
            if recent_avg < self.metrics.avg_reward - 0.1:  # Significant decline
                return True

        return False

    async def _run_optimization_step(self):
        """Run one step of GRPO optimization"""
        # Lazy initialize advanced optimizer if we now have enough experiences
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

            # Sample batch from experience replay
            batch_experiences = np.random.choice(
                self.experience_replay,
                size=min(self.config.batch_size, len(self.experience_replay)),
                replace=False,
            ).tolist()

            # Prepare training data
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

            # Run bootstrap optimization
            if self.routing_policy and training_examples:
                optimized_policy = self.advanced_optimizer.compile(
                    self.routing_policy,
                    trainset=training_examples,
                    max_bootstrapped_demos=4,
                    max_labeled_demos=8,
                )

                self.routing_policy = optimized_policy
                self.training_step += 1

                # Update exploration rate
                self.current_epsilon = max(
                    self.config.min_epsilon,
                    self.current_epsilon * self.config.epsilon_decay,
                )

                logger.info(
                    f"GRPO optimization step {self.training_step} complete. Epsilon: {self.current_epsilon:.3f}"
                )

                # Update improvement rate metric
                if len(self.experiences) > 100:
                    old_rewards = [exp.reward for exp in self.experiences[-200:-100]]
                    new_rewards = [exp.reward for exp in self.experiences[-100:]]

                    if old_rewards and new_rewards:
                        self.metrics.improvement_rate = np.mean(new_rewards) - np.mean(
                            old_rewards
                        )

        except Exception as e:
            logger.error(f"Advanced optimization step failed: {e}")

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
            # If optimization not ready, provide basic recommendations
            if (
                not self.routing_policy
                or len(self.experiences) < self.config.min_experiences_for_training
            ):
                return self._get_baseline_recommendations(
                    query, entities, relationships
                )

            # Use optimized policy for recommendations
            enhanced_query = query  # Could enhance with relationships here

            prediction = self.routing_policy(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
            )

            # Extract and calibrate results
            recommended_agent = prediction.recommended_agent
            raw_confidence = float(prediction.confidence)
            reasoning = prediction.reasoning

            # Apply confidence calibration
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
            logger.error(f"Failed to get routing recommendations: {e}")
            return self._get_baseline_recommendations(query, entities, relationships)

    def _get_baseline_recommendations(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Provide baseline recommendations when optimization not ready"""

        # Simple rule-based recommendations
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

        # Adjust confidence based on historical performance if available
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
            # If optimization not ready, return baseline
            if not self.routing_policy or not self.advanced_optimizer:
                return self._apply_baseline_improvements(baseline_prediction)

            # Apply exploration vs exploitation
            if np.random.random() < self.current_epsilon:
                # Exploration: use baseline with some randomization
                return self._add_exploration_noise(baseline_prediction)

            # Exploitation: use optimized policy
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

            # Apply confidence calibration
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
            logger.error(f"GRPO optimization failed, using baseline: {e}")
            return self._apply_baseline_improvements(baseline_prediction)

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
            # Compute query complexity indicators
            query_complexity = (
                min(1.0, len(query.split()) / 20.0)  # Word count complexity
                + min(1.0, len(entities) / 10.0)  # Entity complexity
                + min(1.0, len(relationships) / 5.0)  # Relationship complexity
            ) / 3.0

            # Get historical accuracy for similar queries
            historical_accuracy = self._get_historical_accuracy_for_query_type(query)

            # Apply calibration
            calibrated_result = self.confidence_calibrator(
                raw_confidence=raw_confidence,
                query_complexity=query_complexity,
                historical_accuracy=historical_accuracy,
            )

            calibrated_confidence = float(calibrated_result.calibrated_confidence)
            return max(0.0, min(1.0, calibrated_confidence))

        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return raw_confidence

    def _get_historical_accuracy_for_query_type(self, query: str) -> float:
        """Get historical accuracy for similar query types"""
        if not self.experiences:
            return 0.7  # Default

        # Simple similarity: count common words
        query_words = set(query.lower().split())

        similar_experiences = []
        for exp in self.experiences[-200:]:  # Last 200 experiences
            exp_words = set(exp.query.lower().split())
            similarity = len(query_words.intersection(exp_words)) / max(
                len(query_words), len(exp_words), 1
            )

            if similarity > 0.3:  # Minimum similarity threshold
                similar_experiences.append(exp)

        if not similar_experiences:
            return self.metrics.confidence_accuracy

        # Return success rate for similar queries
        success_rate = sum(1 for exp in similar_experiences if exp.agent_success) / len(
            similar_experiences
        )
        return success_rate

    def _apply_baseline_improvements(
        self, baseline_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply basic improvements to baseline prediction"""
        improved = baseline_prediction.copy()

        # Apply agent preference learning
        agent = baseline_prediction.get("recommended_agent", "search_agent")
        if agent in self.metrics.agent_preferences:
            # Boost confidence if this agent has performed well historically
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

        # Randomly adjust confidence
        confidence_noise = np.random.normal(0, 0.05)  # Small noise
        original_confidence = baseline_prediction.get("confidence", 0.7)
        exploration["confidence"] = max(
            0.0, min(1.0, original_confidence + confidence_noise)
        )

        # Occasionally suggest different agent (small probability)
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
        """Persist experiences and metrics to disk"""
        try:
            # Save experiences
            experience_file = self.storage_dir / self.config.experience_file
            with open(experience_file, "wb") as f:
                pickle.dump(self.experiences, f)

            # Save metrics
            metrics_file = self.storage_dir / self.config.metrics_file
            metrics_dict = {
                "total_experiences": self.metrics.total_experiences,
                "avg_reward": self.metrics.avg_reward,
                "successful_routes": self.metrics.successful_routes,
                "failed_routes": self.metrics.failed_routes,
                "confidence_accuracy": self.metrics.confidence_accuracy,
                "agent_preferences": self.metrics.agent_preferences,
                "query_type_accuracy": self.metrics.query_type_accuracy,
                "improvement_rate": self.metrics.improvement_rate,
                "last_updated": self.metrics.last_updated.isoformat(),
                "training_step": self.training_step,
                "current_epsilon": self.current_epsilon,
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics_dict, f, indent=2)

            logger.debug(f"Persisted {len(self.experiences)} experiences and metrics")

        except Exception as e:
            logger.error(f"Failed to persist optimization data: {e}")

    def _load_stored_data(self):
        """Load previously stored experiences and metrics if persistence enabled"""
        if not self.config.enable_persistence:
            logger.info("Persistence disabled, skipping data loading")
            return

        try:
            # Load experiences
            experience_file = self.storage_dir / self.config.experience_file
            if experience_file.exists():
                with open(experience_file, "rb") as f:
                    self.experiences = pickle.load(f)
                    self.experience_replay = self.experiences[
                        -self.config.experience_replay_size :
                    ]
                logger.info(f"Loaded {len(self.experiences)} routing experiences")

            # Load metrics
            metrics_file = self.storage_dir / self.config.metrics_file
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_dict = json.load(f)

                self.metrics.total_experiences = metrics_dict.get(
                    "total_experiences", 0
                )
                self.metrics.avg_reward = metrics_dict.get("avg_reward", 0.0)
                self.metrics.successful_routes = metrics_dict.get(
                    "successful_routes", 0
                )
                self.metrics.failed_routes = metrics_dict.get("failed_routes", 0)
                self.metrics.confidence_accuracy = metrics_dict.get(
                    "confidence_accuracy", 0.0
                )
                self.metrics.agent_preferences = metrics_dict.get(
                    "agent_preferences", {}
                )
                self.metrics.query_type_accuracy = metrics_dict.get(
                    "query_type_accuracy", {}
                )
                self.metrics.improvement_rate = metrics_dict.get(
                    "improvement_rate", 0.0
                )

                if "last_updated" in metrics_dict:
                    self.metrics.last_updated = datetime.fromisoformat(
                        metrics_dict["last_updated"]
                    )

                self.training_step = metrics_dict.get("training_step", 0)
                self.current_epsilon = metrics_dict.get(
                    "current_epsilon", self.config.exploration_epsilon
                )

                logger.info("Loaded optimization metrics")

        except Exception as e:
            logger.error(f"Failed to load stored optimization data: {e}")

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
            logger.error(f"Routing policy optimization failed: {e}")
            return {"status": "error", "error": str(e)}

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

        logger.info(f"🔄 Generating {count} synthetic routing examples...")

        # Generate synthetic data directly
        service = SyntheticDataService(
            backend=backend,
            backend_config=backend_config,
            generator_config=generator_config,
        )
        request = SyntheticDataRequest(optimizer="routing", count=count)
        response = await service.generate(request)

        # Convert to RoutingExperience objects
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
            f"✅ Added {added_count} synthetic examples to routing experiences "
            f"(total: {len(self.experiences)})"
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

        # Clear stored files
        try:
            experience_file = self.storage_dir / self.config.experience_file
            metrics_file = self.storage_dir / self.config.metrics_file

            if experience_file.exists():
                experience_file.unlink()
            if metrics_file.exists():
                metrics_file.unlink()

        except Exception as e:
            logger.error(f"Failed to clear stored files: {e}")

        # Re-initialize components
        self._initialize_advanced_components()

        logger.info("Advanced optimization reset complete")
