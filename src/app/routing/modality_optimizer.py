"""
Modality Optimizer

Per-modality routing optimization using XGBoost meta-learning for automatic decisions.
Part of Phase 11: Multi-Modal Optimization.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.app.routing.modality_evaluator import ModalityEvaluator
from src.app.routing.modality_span_collector import ModalitySpanCollector
from src.app.routing.synthetic_data_generator import (
    ModalityExample,
    SyntheticDataGenerator,
)
from src.app.routing.xgboost_meta_models import (
    ModelingContext,
    TrainingDecisionModel,
    TrainingStrategy,
    TrainingStrategyModel,
)
from src.app.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class ModalityOptimizer:
    """
    Per-modality routing optimizer with XGBoost meta-learning

    Key Features:
    - Separate optimization per modality (VIDEO, DOCUMENT, IMAGE, AUDIO)
    - XGBoost meta-models for automatic decision-making
    - Synthetic data generation for cold start
    - Progressive training strategies (synthetic → hybrid → pure real)
    - Tracks training history and performance metrics

    Usage:
        optimizer = ModalityOptimizer(tenant_id="default")

        # Evaluate and potentially train all modalities
        results = await optimizer.optimize_all_modalities()

        # Or optimize specific modality
        result = await optimizer.optimize_modality(QueryModality.VIDEO)
    """

    def __init__(
        self,
        tenant_id: str = "default",
        model_dir: Optional[Path] = None,
        vespa_client=None,
    ):
        """
        Initialize modality optimizer

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            model_dir: Directory for saving models (defaults to outputs/models/modality)
            vespa_client: Optional Vespa client for synthetic data generation
        """
        self.tenant_id = tenant_id
        self.model_dir = model_dir or Path("outputs/models/modality")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.span_collector = ModalitySpanCollector(tenant_id)
        self.evaluator = ModalityEvaluator(self.span_collector, tenant_id)
        self.synthetic_generator = SyntheticDataGenerator(vespa_client)

        # Initialize XGBoost meta-models
        self.training_decision_model = TrainingDecisionModel(model_dir=self.model_dir)
        self.training_strategy_model = TrainingStrategyModel(model_dir=self.model_dir)

        # Load existing meta-models if available
        self.training_decision_model.load()
        self.training_strategy_model.load()

        # Training history per modality
        self.training_history: Dict[QueryModality, List[Dict[str, Any]]] = {}

        logger.info(
            f"🔧 Initialized ModalityOptimizer for tenant '{tenant_id}' "
            f"(model_dir: {self.model_dir})"
        )

    async def optimize_all_modalities(
        self,
        lookback_hours: int = 24,
        min_confidence: float = 0.7,
    ) -> Dict[QueryModality, Dict[str, Any]]:
        """
        Evaluate and optimize all modalities

        Args:
            lookback_hours: How far back to look for spans
            min_confidence: Minimum confidence threshold for span collection

        Returns:
            Dictionary mapping each modality to optimization results
        """
        logger.info(
            f"🚀 Starting optimization for all modalities "
            f"(lookback: {lookback_hours}h, min_confidence: {min_confidence})"
        )

        results = {}

        # Get span statistics to see which modalities have data
        stats = await self.span_collector.get_modality_statistics(lookback_hours)
        available_modalities = stats.get("modality_distribution", {})

        if not available_modalities:
            logger.warning("⚠️ No modality data available - skipping optimization")
            return {}

        # Optimize each modality that has data
        for modality_str in available_modalities:
            try:
                modality = QueryModality(modality_str)
                result = await self.optimize_modality(
                    modality,
                    lookback_hours=lookback_hours,
                    min_confidence=min_confidence,
                )
                results[modality] = result
            except Exception as e:
                logger.error(f"❌ Error optimizing {modality_str}: {e}")
                results[QueryModality(modality_str)] = {
                    "status": "error",
                    "error": str(e),
                }

        # Log summary
        trained_count = sum(1 for r in results.values() if r.get("trained", False))
        logger.info(
            f"✅ Optimization complete: {trained_count}/{len(results)} modalities trained"
        )

        return results

    async def optimize_modality(
        self,
        modality: QueryModality,
        lookback_hours: int = 24,
        min_confidence: float = 0.7,
        force_training: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate and potentially train a specific modality

        Args:
            modality: Modality to optimize
            lookback_hours: How far back to look for spans
            min_confidence: Minimum confidence threshold
            force_training: Force training regardless of decision model

        Returns:
            Optimization results including decision, strategy, and metrics
        """
        logger.info(f"🎯 Optimizing {modality.value} modality")

        # Step 1: Collect training examples
        training_examples = await self.evaluator.create_training_examples(
            lookback_hours=lookback_hours,
            min_confidence=min_confidence,
            augment_with_synthetic=False,  # We'll handle synthetic separately
        )

        modality_examples = training_examples.get(modality, [])

        logger.info(
            f"📊 Collected {len(modality_examples)} real examples for {modality.value}"
        )

        # Step 2: Build modeling context
        context = self._build_modeling_context(modality, modality_examples)

        # Step 3: Decide whether to train
        if not force_training:
            should_train, expected_improvement = (
                self.training_decision_model.should_train(context)
            )

            if not should_train:
                logger.info(
                    f"⏭️ Skipping {modality.value} training "
                    f"(expected improvement: {expected_improvement:.3f})"
                )
                return {
                    "modality": modality.value,
                    "trained": False,
                    "reason": "insufficient_benefit",
                    "expected_improvement": expected_improvement,
                    "context": self._context_to_dict(context),
                }
        else:
            expected_improvement = 0.0
            logger.info(f"🔨 Force training {modality.value}")

        # Step 4: Select training strategy
        strategy = self.training_strategy_model.select_strategy(context)

        logger.info(f"📋 Selected strategy for {modality.value}: {strategy.value}")

        # Step 5: Prepare training data based on strategy
        training_data = await self._prepare_training_data(
            modality, modality_examples, strategy
        )

        if not training_data:
            logger.warning(f"⚠️ No training data available for {modality.value}")
            return {
                "modality": modality.value,
                "trained": False,
                "reason": "no_training_data",
                "strategy": strategy.value,
                "context": self._context_to_dict(context),
            }

        logger.info(
            f"📚 Prepared {len(training_data)} training examples "
            f"(strategy: {strategy.value})"
        )

        # Step 6: Train modality-specific model (placeholder for now)
        training_result = self._train_modality_model(modality, training_data, strategy)

        # Step 7: Record training history
        self._record_training(modality, context, strategy, training_result)

        return {
            "modality": modality.value,
            "trained": True,
            "strategy": strategy.value,
            "examples_count": len(training_data),
            "expected_improvement": expected_improvement,
            "training_result": training_result,
            "context": self._context_to_dict(context),
        }

    def _build_modeling_context(
        self, modality: QueryModality, examples: List[ModalityExample]
    ) -> ModelingContext:
        """
        Build ModelingContext from current state

        Args:
            modality: Query modality
            examples: Real training examples

        Returns:
            ModelingContext for decision-making
        """
        # Count real vs synthetic examples
        real_examples = [ex for ex in examples if not ex.is_synthetic]
        synthetic_examples = [ex for ex in examples if ex.is_synthetic]

        # Calculate metrics
        success_count = sum(1 for ex in real_examples if ex.success)
        success_rate = success_count / len(real_examples) if real_examples else 0.0

        # Calculate average confidence
        confidences = [
            ex.modality_features.get("routing_confidence", 0.0)
            for ex in real_examples
            if ex.modality_features
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Get days since last training
        days_since_last = self._get_days_since_last_training(modality)

        # Estimate current performance (based on success rate and confidence)
        current_performance = (success_rate * 0.7) + (avg_confidence * 0.3)

        # Estimate data quality (based on feature diversity and confidence variance)
        data_quality = self._estimate_data_quality(real_examples)

        # Estimate feature diversity
        feature_diversity = self._estimate_feature_diversity(real_examples)

        return ModelingContext(
            modality=modality,
            real_sample_count=len(real_examples),
            synthetic_sample_count=len(synthetic_examples),
            success_rate=success_rate,
            avg_confidence=avg_confidence,
            days_since_last_training=days_since_last,
            current_performance_score=current_performance,
            data_quality_score=data_quality,
            feature_diversity=feature_diversity,
        )

    def _get_days_since_last_training(self, modality: QueryModality) -> int:
        """Get days since modality was last trained"""
        history = self.training_history.get(modality, [])
        if not history:
            return 999  # Never trained

        last_training = history[-1]
        last_timestamp = last_training.get("timestamp")
        if not last_timestamp:
            return 999

        days_diff = (datetime.now() - last_timestamp).days
        return days_diff

    def _estimate_data_quality(self, examples: List[ModalityExample]) -> float:
        """
        Estimate data quality based on feature completeness and variance

        Args:
            examples: Training examples

        Returns:
            Quality score (0-1)
        """
        if not examples:
            return 0.0

        # Check feature completeness
        complete_features = sum(
            1 for ex in examples if ex.modality_features and len(ex.modality_features) > 5
        )
        completeness_score = complete_features / len(examples)

        # Check confidence variance (more variance = more diverse scenarios)
        confidences = [
            ex.modality_features.get("routing_confidence", 0.0)
            for ex in examples
            if ex.modality_features
        ]

        if len(confidences) > 1:
            variance = sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences) / len(confidences)
            variance_score = min(variance * 2, 1.0)  # Scale to 0-1
        else:
            variance_score = 0.0

        # Combine scores
        quality = (completeness_score * 0.7) + (variance_score * 0.3)
        return quality

    def _estimate_feature_diversity(self, examples: List[ModalityExample]) -> float:
        """
        Estimate feature diversity based on unique patterns in queries

        Args:
            examples: Training examples

        Returns:
            Diversity score (0-1)
        """
        if not examples:
            return 0.0

        # Count unique query patterns
        unique_queries = len(set(ex.query.lower().strip() for ex in examples))
        query_diversity = min(unique_queries / len(examples), 1.0)

        # Count unique agents
        unique_agents = len(set(ex.correct_agent for ex in examples))
        agent_diversity = min(unique_agents / 4, 1.0)  # Assume max 4 agents

        # Combine
        diversity = (query_diversity * 0.6) + (agent_diversity * 0.4)
        return diversity

    async def _prepare_training_data(
        self,
        modality: QueryModality,
        real_examples: List[ModalityExample],
        strategy: TrainingStrategy,
    ) -> List[ModalityExample]:
        """
        Prepare training data based on selected strategy

        Args:
            modality: Query modality
            real_examples: Real training examples
            strategy: Selected training strategy

        Returns:
            Combined training data
        """
        if strategy == TrainingStrategy.SKIP:
            return []

        if strategy == TrainingStrategy.PURE_REAL:
            return real_examples

        if strategy == TrainingStrategy.SYNTHETIC:
            # Generate synthetic data
            synthetic_count = max(50, len(real_examples) * 2)
            synthetic_examples = (
                await self.synthetic_generator.generate_from_ingested_data(
                    modality, target_count=synthetic_count
                )
            )
            return synthetic_examples

        if strategy == TrainingStrategy.HYBRID:
            # Mix real and synthetic
            synthetic_count = len(real_examples)  # 1:1 ratio
            synthetic_examples = (
                await self.synthetic_generator.generate_from_ingested_data(
                    modality, target_count=synthetic_count
                )
            )
            return real_examples + synthetic_examples

        return real_examples

    def _train_modality_model(
        self,
        modality: QueryModality,
        training_data: List[ModalityExample],
        strategy: TrainingStrategy,
    ) -> Dict[str, Any]:
        """
        Train modality-specific routing model

        This is a placeholder for the actual model training logic.
        In full implementation, this would train a classifier to predict
        the correct agent given query features.

        Args:
            modality: Query modality
            training_data: Training examples
            strategy: Training strategy used

        Returns:
            Training results/metrics
        """
        logger.info(
            f"🎓 Training {modality.value} model with {len(training_data)} examples "
            f"(strategy: {strategy.value})"
        )

        # Placeholder: In real implementation, would train classifier here
        # For now, just return success metrics

        # Simulate training metrics
        accuracy = 0.85 + (len(training_data) / 1000) * 0.1  # Better with more data
        accuracy = min(accuracy, 0.95)

        return {
            "status": "success",
            "training_samples": len(training_data),
            "strategy": strategy.value,
            "estimated_accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
        }

    def _record_training(
        self,
        modality: QueryModality,
        context: ModelingContext,
        strategy: TrainingStrategy,
        result: Dict[str, Any],
    ):
        """Record training in history"""
        if modality not in self.training_history:
            self.training_history[modality] = []

        self.training_history[modality].append({
            "timestamp": datetime.now(),
            "context": self._context_to_dict(context),
            "strategy": strategy.value,
            "result": result,
        })

        logger.info(
            f"📝 Recorded training for {modality.value} "
            f"(total history: {len(self.training_history[modality])})"
        )

    def _context_to_dict(self, context: ModelingContext) -> Dict[str, Any]:
        """Convert ModelingContext to dictionary"""
        return {
            "modality": context.modality.value,
            "real_sample_count": context.real_sample_count,
            "synthetic_sample_count": context.synthetic_sample_count,
            "success_rate": context.success_rate,
            "avg_confidence": context.avg_confidence,
            "days_since_last_training": context.days_since_last_training,
            "current_performance_score": context.current_performance_score,
            "data_quality_score": context.data_quality_score,
            "feature_diversity": context.feature_diversity,
        }

    def get_training_history(
        self, modality: Optional[QueryModality] = None
    ) -> Dict[QueryModality, List[Dict[str, Any]]]:
        """
        Get training history

        Args:
            modality: Optional specific modality, or None for all

        Returns:
            Training history
        """
        if modality:
            return {modality: self.training_history.get(modality, [])}
        return self.training_history

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of all modality optimizations

        Returns:
            Summary statistics
        """
        summary = {
            "total_modalities": len(self.training_history),
            "modalities": {},
            "meta_models": {
                "training_decision_model": self.training_decision_model.is_trained,
                "training_strategy_model": self.training_strategy_model.is_trained,
            },
        }

        for modality, history in self.training_history.items():
            if history:
                last_training = history[-1]
                summary["modalities"][modality.value] = {
                    "training_count": len(history),
                    "last_training": last_training["timestamp"].isoformat(),
                    "last_strategy": last_training["strategy"],
                    "last_result": last_training["result"],
                }

        return summary
