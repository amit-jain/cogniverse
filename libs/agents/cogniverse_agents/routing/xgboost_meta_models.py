"""
XGBoost Meta-Models for Multi-Modal Optimization

These meta-models use XGBoost to make automatic decisions about training strategies,
eliminating the need for hardcoded thresholds.

Part of Phase 11: Multi-Modal Optimization.
"""

import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb

from cogniverse_agents.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """Training strategy options"""

    PURE_REAL = "pure_real"  # Train on real data only
    HYBRID = "hybrid"  # Train on mix of real + synthetic
    SYNTHETIC = "synthetic"  # Train on synthetic only (cold start)
    SKIP = "skip"  # Skip training (insufficient data/benefit)


@dataclass
class ModelingContext:
    """
    Context information for meta-model decision making

    Attributes:
        modality: Query modality
        real_sample_count: Number of real examples available
        synthetic_sample_count: Number of synthetic examples available
        success_rate: Current success rate (0-1)
        avg_confidence: Average routing confidence (0-1)
        days_since_last_training: Days since model was last trained
        current_performance_score: Current model performance score
        data_quality_score: Quality score of available data (0-1)
        feature_diversity: Diversity of features in data (0-1)
    """

    modality: QueryModality
    real_sample_count: int
    synthetic_sample_count: int
    success_rate: float
    avg_confidence: float
    days_since_last_training: int
    current_performance_score: float
    data_quality_score: float = 0.8
    feature_diversity: float = 0.7


class TrainingDecisionModel:
    """
    XGBoost model to predict if training will be beneficial

    Input Features:
    - real_sample_count
    - synthetic_sample_count
    - success_rate
    - avg_confidence
    - days_since_last_training
    - current_performance_score
    - data_quality_score
    - feature_diversity

    Output: Binary decision (train/skip) + expected improvement estimate
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize training decision model

        Args:
            model_dir: Directory to save/load models (defaults to outputs/models/xgboost)
        """
        self.model_dir = model_dir or Path("outputs/models/xgboost")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False

        logger.info(
            f"🧠 Initialized TrainingDecisionModel (model_dir: {self.model_dir})"
        )

    def train(
        self,
        training_contexts: List[ModelingContext],
        training_outcomes: List[float],
        **xgb_params,
    ) -> Dict[str, Any]:
        """
        Train model on historical training contexts and outcomes

        Args:
            training_contexts: List of historical contexts
            training_outcomes: List of performance improvements achieved (0-1)
            xgb_params: Optional XGBoost hyperparameters

        Returns:
            Training metrics
        """
        if len(training_contexts) < 10:
            logger.warning(
                f"⚠️ Only {len(training_contexts)} samples - need at least 10 for training"
            )
            return {"status": "insufficient_data", "samples": len(training_contexts)}

        # Convert contexts to feature matrix
        X = self._contexts_to_features(training_contexts)
        # Binary labels: improvement > threshold means training was beneficial
        y = (np.array(training_outcomes) > 0.02).astype(int)  # 2% improvement threshold

        # Default XGBoost params for binary classification
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "random_state": 42,
        }
        default_params.update(xgb_params)

        # Train model
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y)
        self.is_trained = True

        # Compute training metrics
        predictions = self.model.predict(X)
        accuracy = (predictions == y).mean()

        logger.info(
            f"✅ Trained TrainingDecisionModel on {len(training_contexts)} samples "
            f"(accuracy: {accuracy:.3f})"
        )

        return {
            "status": "success",
            "samples": len(training_contexts),
            "accuracy": float(accuracy),
            "positive_rate": float(y.mean()),
        }

    def should_train(self, context: ModelingContext) -> Tuple[bool, float]:
        """
        Predict if training will be beneficial

        Args:
            context: Current modeling context

        Returns:
            Tuple of (should_train, expected_improvement)
        """
        if not self.is_trained:
            # Fallback heuristic when model not trained
            return self._fallback_decision(context)

        # Get model prediction
        X = self._contexts_to_features([context])
        prob = self.model.predict_proba(X)[0, 1]  # Probability of beneficial training

        should_train = bool(prob > 0.5)
        expected_improvement = float(
            prob * 0.1
        )  # Scale to reasonable improvement estimate

        logger.info(
            f"🤔 Training decision for {context.modality.value}: "
            f"should_train={should_train}, expected_improvement={expected_improvement:.3f}"
        )

        return should_train, expected_improvement

    def _fallback_decision(self, context: ModelingContext) -> Tuple[bool, float]:
        """Fallback heuristic when model not trained"""
        # Simple rules:
        # - Need enough data (50+ samples or good synthetic data)
        # - Success rate below 0.85 (room for improvement)
        # - At least 7 days since last training

        has_enough_data = context.real_sample_count >= 50 or (
            context.synthetic_sample_count >= 100 and context.data_quality_score > 0.7
        )

        has_room_for_improvement = context.success_rate < 0.85
        stale_model = context.days_since_last_training >= 7

        should_train = has_enough_data and (has_room_for_improvement or stale_model)

        # Estimate improvement based on current performance gap
        expected_improvement = (
            max(0, 0.85 - context.success_rate) * 0.5 if should_train else 0.0
        )

        return should_train, expected_improvement

    def _contexts_to_features(self, contexts: List[ModelingContext]) -> np.ndarray:
        """Convert contexts to feature matrix"""
        features = []
        for ctx in contexts:
            features.append(
                [
                    ctx.real_sample_count,
                    ctx.synthetic_sample_count,
                    ctx.success_rate,
                    ctx.avg_confidence,
                    ctx.days_since_last_training,
                    ctx.current_performance_score,
                    ctx.data_quality_score,
                    ctx.feature_diversity,
                ]
            )
        return np.array(features)

    def save(self, filename: str = "training_decision_model.pkl"):
        """Save model to disk"""
        if not self.is_trained:
            logger.warning("⚠️ Model not trained - nothing to save")
            return

        filepath = self.model_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"💾 Saved TrainingDecisionModel to {filepath}")

    def load(self, filename: str = "training_decision_model.pkl"):
        """Load model from disk"""
        filepath = self.model_dir / filename

        if not filepath.exists():
            logger.warning(f"⚠️ Model file not found: {filepath}")
            return False

        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info(f"📂 Loaded TrainingDecisionModel from {filepath}")
        return True


class TrainingStrategyModel:
    """
    XGBoost model to select optimal training strategy

    Input Features:
    - Same as TrainingDecisionModel
    - Plus: is_cold_start (boolean)

    Output: Multi-class prediction (PURE_REAL, HYBRID, SYNTHETIC, SKIP)
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize training strategy model"""
        self.model_dir = model_dir or Path("outputs/models/xgboost")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False

        logger.info(
            f"🧠 Initialized TrainingStrategyModel (model_dir: {self.model_dir})"
        )

    def train(
        self,
        training_contexts: List[ModelingContext],
        best_strategies: List[TrainingStrategy],
        **xgb_params,
    ) -> Dict[str, Any]:
        """
        Train model on historical contexts and their best strategies

        Args:
            training_contexts: List of historical contexts
            best_strategies: List of strategies that worked best
            xgb_params: Optional XGBoost hyperparameters

        Returns:
            Training metrics
        """
        if len(training_contexts) < 20:
            logger.warning(
                f"⚠️ Only {len(training_contexts)} samples - need at least 20 for training"
            )
            return {"status": "insufficient_data", "samples": len(training_contexts)}

        # Convert to features and labels
        X = self._contexts_to_features(training_contexts)
        y = np.array([self._strategy_to_label(s) for s in best_strategies])

        # Default XGBoost params for multi-class
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "objective": "multi:softmax",
            "num_class": 4,  # PURE_REAL, HYBRID, SYNTHETIC, SKIP
            "random_state": 42,
        }
        default_params.update(xgb_params)

        # Train model
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y)
        self.is_trained = True

        # Compute metrics
        predictions = self.model.predict(X)
        accuracy = (predictions == y).mean()

        logger.info(
            f"✅ Trained TrainingStrategyModel on {len(training_contexts)} samples "
            f"(accuracy: {accuracy:.3f})"
        )

        return {
            "status": "success",
            "samples": len(training_contexts),
            "accuracy": float(accuracy),
        }

    def select_strategy(self, context: ModelingContext) -> TrainingStrategy:
        """
        Select optimal training strategy

        Args:
            context: Current modeling context

        Returns:
            Recommended training strategy
        """
        if not self.is_trained:
            # Fallback heuristic
            return self._fallback_strategy(context)

        # Get model prediction
        X = self._contexts_to_features([context])
        prediction = self.model.predict(X)[0]
        strategy = self._label_to_strategy(int(prediction))

        logger.info(
            f"📋 Strategy for {context.modality.value}: {strategy.value} "
            f"(real: {context.real_sample_count}, synthetic: {context.synthetic_sample_count})"
        )

        return strategy

    def _fallback_strategy(self, context: ModelingContext) -> TrainingStrategy:
        """Fallback heuristic when model not trained"""
        # Cold start: use synthetic if available
        if context.real_sample_count < 10:
            if context.synthetic_sample_count >= 50:
                return TrainingStrategy.SYNTHETIC
            else:
                return TrainingStrategy.SKIP

        # Enough real data: use pure real
        if context.real_sample_count >= 100:
            return TrainingStrategy.PURE_REAL

        # Moderate real data: use hybrid if synthetic available
        if context.real_sample_count >= 30:
            if context.synthetic_sample_count >= 50:
                return TrainingStrategy.HYBRID
            else:
                return TrainingStrategy.PURE_REAL

        # Not enough data yet
        return TrainingStrategy.SKIP

    def _contexts_to_features(self, contexts: List[ModelingContext]) -> np.ndarray:
        """Convert contexts to feature matrix"""
        features = []
        for ctx in contexts:
            is_cold_start = 1 if ctx.real_sample_count < 10 else 0
            features.append(
                [
                    ctx.real_sample_count,
                    ctx.synthetic_sample_count,
                    ctx.success_rate,
                    ctx.avg_confidence,
                    ctx.days_since_last_training,
                    ctx.current_performance_score,
                    ctx.data_quality_score,
                    ctx.feature_diversity,
                    is_cold_start,
                ]
            )
        return np.array(features)

    def _strategy_to_label(self, strategy: TrainingStrategy) -> int:
        """Convert strategy to numeric label"""
        mapping = {
            TrainingStrategy.PURE_REAL: 0,
            TrainingStrategy.HYBRID: 1,
            TrainingStrategy.SYNTHETIC: 2,
            TrainingStrategy.SKIP: 3,
        }
        return mapping[strategy]

    def _label_to_strategy(self, label: int) -> TrainingStrategy:
        """Convert numeric label to strategy"""
        mapping = {
            0: TrainingStrategy.PURE_REAL,
            1: TrainingStrategy.HYBRID,
            2: TrainingStrategy.SYNTHETIC,
            3: TrainingStrategy.SKIP,
        }
        return mapping.get(label, TrainingStrategy.SKIP)

    def save(self, filename: str = "training_strategy_model.pkl"):
        """Save model to disk"""
        if not self.is_trained:
            logger.warning("⚠️ Model not trained - nothing to save")
            return

        filepath = self.model_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"💾 Saved TrainingStrategyModel to {filepath}")

    def load(self, filename: str = "training_strategy_model.pkl"):
        """Load model from disk"""
        filepath = self.model_dir / filename

        if not filepath.exists():
            logger.warning(f"⚠️ Model file not found: {filepath}")
            return False

        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info(f"📂 Loaded TrainingStrategyModel from {filepath}")
        return True


class FusionBenefitModel:
    """
    XGBoost model to predict benefit of multi-modal fusion

    Input Features:
    - primary_modality_confidence
    - secondary_modality_confidence
    - modality_agreement (do they agree on agent?)
    - query_ambiguity_score
    - historical_fusion_success_rate

    Output: Regression prediction of expected benefit (0-1)
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """Initialize fusion benefit model"""
        self.model_dir = model_dir or Path("outputs/models/xgboost")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[xgb.XGBRegressor] = None
        self.is_trained = False

        logger.info(f"🧠 Initialized FusionBenefitModel (model_dir: {self.model_dir})")

    def train(
        self,
        fusion_contexts: List[Dict[str, float]],
        fusion_benefits: List[float],
        **xgb_params,
    ) -> Dict[str, Any]:
        """
        Train model on historical fusion contexts and benefits

        Args:
            fusion_contexts: List of fusion context dicts with features
            fusion_benefits: List of actual benefits achieved (0-1)
            xgb_params: Optional XGBoost hyperparameters

        Returns:
            Training metrics
        """
        if len(fusion_contexts) < 10:
            logger.warning(
                f"⚠️ Only {len(fusion_contexts)} samples - need at least 10 for training"
            )
            return {"status": "insufficient_data", "samples": len(fusion_contexts)}

        # Convert to features and labels
        X = self._contexts_to_features(fusion_contexts)
        y = np.array(fusion_benefits)

        # Default XGBoost params for regression
        default_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        default_params.update(xgb_params)

        # Train model
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(X, y)
        self.is_trained = True

        # Compute metrics
        predictions = self.model.predict(X)
        mae = np.abs(predictions - y).mean()
        rmse = np.sqrt(((predictions - y) ** 2).mean())

        logger.info(
            f"✅ Trained FusionBenefitModel on {len(fusion_contexts)} samples "
            f"(MAE: {mae:.3f}, RMSE: {rmse:.3f})"
        )

        return {
            "status": "success",
            "samples": len(fusion_contexts),
            "mae": float(mae),
            "rmse": float(rmse),
        }

    def predict_benefit(self, fusion_context: Dict[str, float]) -> float:
        """
        Predict benefit of multi-modal fusion

        Args:
            fusion_context: Dict with fusion features

        Returns:
            Expected benefit (0-1)
        """
        if not self.is_trained:
            # Fallback heuristic
            return self._fallback_benefit(fusion_context)

        # Get model prediction
        X = self._contexts_to_features([fusion_context])
        benefit = float(self.model.predict(X)[0])

        # Clip to [0, 1]
        benefit = np.clip(benefit, 0.0, 1.0)

        logger.debug(f"🔮 Predicted fusion benefit: {benefit:.3f}")

        return benefit

    def _fallback_benefit(self, fusion_context: Dict[str, float]) -> float:
        """Fallback heuristic when model not trained"""
        # Simple heuristic: fusion beneficial when:
        # - Both modalities have moderate confidence
        # - They disagree on agent (need to resolve ambiguity)
        # - Query is ambiguous

        secondary_conf = fusion_context.get("secondary_modality_confidence", 0.3)
        agreement = fusion_context.get("modality_agreement", 0.0)
        ambiguity = fusion_context.get("query_ambiguity_score", 0.5)

        # Benefit higher when:
        # - Secondary modality has decent confidence
        # - Modalities disagree (1 - agreement)
        # - Query is ambiguous
        benefit = (secondary_conf * 0.4) + ((1 - agreement) * 0.3) + (ambiguity * 0.3)

        return np.clip(benefit, 0.0, 1.0)

    def _contexts_to_features(self, contexts: List[Dict[str, float]]) -> np.ndarray:
        """Convert fusion contexts to feature matrix"""
        features = []
        for ctx in contexts:
            features.append(
                [
                    ctx.get("primary_modality_confidence", 0.5),
                    ctx.get("secondary_modality_confidence", 0.3),
                    ctx.get("modality_agreement", 0.0),
                    ctx.get("query_ambiguity_score", 0.5),
                    ctx.get("historical_fusion_success_rate", 0.7),
                ]
            )
        return np.array(features)

    def save(self, filename: str = "fusion_benefit_model.pkl"):
        """Save model to disk"""
        if not self.is_trained:
            logger.warning("⚠️ Model not trained - nothing to save")
            return

        filepath = self.model_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"💾 Saved FusionBenefitModel to {filepath}")

    def load(self, filename: str = "fusion_benefit_model.pkl"):
        """Load model from disk"""
        filepath = self.model_dir / filename

        if not filepath.exists():
            logger.warning(f"⚠️ Model file not found: {filepath}")
            return False

        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info(f"📂 Loaded FusionBenefitModel from {filepath}")
        return True
