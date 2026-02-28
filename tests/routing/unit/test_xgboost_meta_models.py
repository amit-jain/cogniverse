"""
Unit tests for XGBoost Meta-Models
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_agents.routing.xgboost_meta_models import (
    FusionBenefitModel,
    ModelingContext,
    TrainingDecisionModel,
    TrainingStrategy,
    TrainingStrategyModel,
)
from cogniverse_agents.search.multi_modal_reranker import QueryModality


def _make_mock_telemetry_provider():
    """Create a mock TelemetryProvider with in-memory blob stores."""
    provider = MagicMock()
    blobs: dict[str, bytes] = {}

    async def save_blob(key, data, metadata=None):
        blobs[key] = data
        return key

    async def load_blob(key):
        return blobs.get(key)

    provider.save_blob = AsyncMock(side_effect=save_blob)
    provider.load_blob = AsyncMock(side_effect=load_blob)
    provider.datasets = MagicMock()
    provider.datasets.create_dataset = AsyncMock(return_value="ds-test")
    provider.datasets.get_dataset = AsyncMock(return_value=None)
    provider.experiments = MagicMock()
    provider.experiments.create_experiment = AsyncMock(return_value="exp-test")
    provider.experiments.log_run = AsyncMock(return_value="run-test")
    return provider


class TestModelingContext:
    """Test ModelingContext dataclass"""

    def test_context_creation(self):
        """Test creating modeling context"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            synthetic_sample_count=50,
            success_rate=0.85,
            avg_confidence=0.9,
            days_since_last_training=7,
            current_performance_score=0.82,
        )

        assert context.modality == QueryModality.VIDEO
        assert context.real_sample_count == 100
        assert context.synthetic_sample_count == 50
        assert context.success_rate == 0.85
        assert context.data_quality_score == 0.8  # Default
        assert context.feature_diversity == 0.7  # Default


class TestTrainingDecisionModel:
    """Test TrainingDecisionModel functionality"""

    @pytest.fixture
    def model(self):
        """Create model instance with telemetry-backed storage."""
        return TrainingDecisionModel(
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test-tenant",
        )

    def test_initialization(self, model):
        """Test model initialization"""
        assert model.is_trained is False
        assert model.model is None

    def test_fallback_decision_sufficient_real_data(self, model):
        """Test fallback decision with sufficient real data"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            synthetic_sample_count=0,
            success_rate=0.75,  # Room for improvement
            avg_confidence=0.8,
            days_since_last_training=10,
            current_performance_score=0.75,
        )

        should_train, improvement = model._fallback_decision(context)

        assert should_train is True
        assert improvement > 0

    def test_fallback_decision_insufficient_data(self, model):
        """Test fallback decision with insufficient data"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=20,
            synthetic_sample_count=10,
            success_rate=0.85,
            avg_confidence=0.9,
            days_since_last_training=2,
            current_performance_score=0.85,
        )

        should_train, improvement = model._fallback_decision(context)

        assert should_train is False
        assert improvement == 0.0

    def test_fallback_decision_good_synthetic_data(self, model):
        """Test fallback decision with good synthetic data"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=30,
            synthetic_sample_count=150,
            success_rate=0.70,
            avg_confidence=0.75,
            days_since_last_training=10,
            current_performance_score=0.70,
            data_quality_score=0.85,
        )

        should_train, improvement = model._fallback_decision(context)

        assert should_train is True
        assert improvement > 0

    def test_fallback_decision_high_success_rate(self, model):
        """Test fallback decision when already performing well"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            synthetic_sample_count=50,
            success_rate=0.95,  # Already excellent
            avg_confidence=0.95,
            days_since_last_training=3,
            current_performance_score=0.95,
        )

        should_train, improvement = model._fallback_decision(context)

        # Should still train if model is stale, but not based on performance gap
        assert improvement >= 0

    def test_contexts_to_features(self, model):
        """Test converting contexts to feature matrix"""
        contexts = [
            ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=100,
                synthetic_sample_count=50,
                success_rate=0.85,
                avg_confidence=0.9,
                days_since_last_training=7,
                current_performance_score=0.82,
            ),
            ModelingContext(
                modality=QueryModality.DOCUMENT,
                real_sample_count=200,
                synthetic_sample_count=100,
                success_rate=0.90,
                avg_confidence=0.92,
                days_since_last_training=14,
                current_performance_score=0.88,
            ),
        ]

        features = model._contexts_to_features(contexts)

        assert features.shape == (2, 8)  # 2 contexts, 8 features each
        assert features[0, 0] == 100  # real_sample_count
        assert features[0, 2] == 0.85  # success_rate
        assert features[1, 0] == 200  # real_sample_count

    def test_train_insufficient_data(self, model):
        """Test training with insufficient data"""
        contexts = [
            ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=100,
                synthetic_sample_count=50,
                success_rate=0.85,
                avg_confidence=0.9,
                days_since_last_training=7,
                current_performance_score=0.82,
            )
        ]
        outcomes = [0.05]  # 5% improvement

        result = model.train(contexts, outcomes)

        assert result["status"] == "insufficient_data"
        assert result["samples"] == 1
        assert model.is_trained is False

    def test_train_success(self, model):
        """Test successful training"""
        # Create synthetic training data
        contexts = []
        outcomes = []

        for i in range(50):
            # Create contexts with varying characteristics
            real_count = 50 + i * 10
            success_rate = 0.7 + (i % 5) * 0.05

            context = ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=real_count,
                synthetic_sample_count=100,
                success_rate=success_rate,
                avg_confidence=0.8 + (i % 3) * 0.05,
                days_since_last_training=i % 14,
                current_performance_score=success_rate,
            )
            contexts.append(context)

            # Outcome: training beneficial if success_rate < 0.85
            improvement = max(0, 0.85 - success_rate) * 0.5
            outcomes.append(improvement)

        result = model.train(contexts, outcomes)

        assert result["status"] == "success"
        assert result["samples"] == 50
        assert "accuracy" in result
        assert model.is_trained is True
        assert model.model is not None

    def test_should_train_with_trained_model(self, model):
        """Test should_train with trained model"""
        # Train model first
        contexts = []
        outcomes = []

        for i in range(20):
            context = ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=100 + i * 10,
                synthetic_sample_count=100,
                success_rate=0.75 + (i % 5) * 0.03,
                avg_confidence=0.85,
                days_since_last_training=7,
                current_performance_score=0.75 + (i % 5) * 0.03,
            )
            contexts.append(context)
            # Vary outcomes: some beneficial (>2%), some not
            outcomes.append(0.05 if i % 2 == 0 else 0.01)

        model.train(contexts, outcomes)

        # Now test prediction
        test_context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=150,
            synthetic_sample_count=100,
            success_rate=0.80,
            avg_confidence=0.85,
            days_since_last_training=10,
            current_performance_score=0.80,
        )

        should_train, improvement = model.should_train(test_context)

        assert isinstance(should_train, bool)
        assert isinstance(improvement, float)
        assert 0 <= improvement <= 1

    def test_should_train_without_trained_model(self, model):
        """Test should_train falls back to heuristic when model not trained"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            synthetic_sample_count=50,
            success_rate=0.75,
            avg_confidence=0.85,
            days_since_last_training=10,
            current_performance_score=0.75,
        )

        should_train, improvement = model.should_train(context)

        assert isinstance(should_train, bool)
        assert isinstance(improvement, float)


class TestTrainingStrategyModel:
    """Test TrainingStrategyModel functionality"""

    @pytest.fixture
    def model(self):
        """Create model instance with telemetry-backed storage."""
        return TrainingStrategyModel(
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test-tenant",
        )

    def test_initialization(self, model):
        """Test model initialization"""
        assert model.is_trained is False

    def test_fallback_strategy_cold_start(self, model):
        """Test fallback strategy during cold start"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=5,
            synthetic_sample_count=100,
            success_rate=0.0,
            avg_confidence=0.0,
            days_since_last_training=0,
            current_performance_score=0.0,
        )

        strategy = model._fallback_strategy(context)
        assert strategy == TrainingStrategy.SYNTHETIC

    def test_fallback_strategy_insufficient_data(self, model):
        """Test fallback strategy with insufficient data"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=5,
            synthetic_sample_count=10,
            success_rate=0.5,
            avg_confidence=0.6,
            days_since_last_training=1,
            current_performance_score=0.5,
        )

        strategy = model._fallback_strategy(context)
        assert strategy == TrainingStrategy.SKIP

    def test_fallback_strategy_pure_real(self, model):
        """Test fallback strategy with abundant real data"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=150,
            synthetic_sample_count=50,
            success_rate=0.85,
            avg_confidence=0.9,
            days_since_last_training=7,
            current_performance_score=0.85,
        )

        strategy = model._fallback_strategy(context)
        assert strategy == TrainingStrategy.PURE_REAL

    def test_fallback_strategy_hybrid(self, model):
        """Test fallback strategy for hybrid approach"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=50,
            synthetic_sample_count=80,
            success_rate=0.80,
            avg_confidence=0.85,
            days_since_last_training=5,
            current_performance_score=0.80,
        )

        strategy = model._fallback_strategy(context)
        assert strategy == TrainingStrategy.HYBRID

    def test_strategy_label_conversion(self, model):
        """Test converting between strategies and labels"""
        for strategy in TrainingStrategy:
            label = model._strategy_to_label(strategy)
            assert isinstance(label, int)
            assert 0 <= label <= 3

            converted_back = model._label_to_strategy(label)
            assert converted_back == strategy

    def test_train_insufficient_data(self, model):
        """Test training with insufficient data"""
        contexts = [
            ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=100,
                synthetic_sample_count=50,
                success_rate=0.85,
                avg_confidence=0.9,
                days_since_last_training=7,
                current_performance_score=0.85,
            )
        ]
        strategies = [TrainingStrategy.PURE_REAL]

        result = model.train(contexts, strategies)

        assert result["status"] == "insufficient_data"
        assert model.is_trained is False

    def test_train_success(self, model):
        """Test successful training"""
        contexts = []
        strategies = []

        # Create diverse training data
        for i in range(50):
            real_count = i * 10
            synthetic_count = 100 - i

            context = ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=real_count,
                synthetic_sample_count=synthetic_count,
                success_rate=0.8,
                avg_confidence=0.85,
                days_since_last_training=i % 14,
                current_performance_score=0.8,
            )
            contexts.append(context)

            # Assign strategy based on data availability
            if real_count < 10:
                strategies.append(
                    TrainingStrategy.SYNTHETIC
                    if synthetic_count > 50
                    else TrainingStrategy.SKIP
                )
            elif real_count >= 100:
                strategies.append(TrainingStrategy.PURE_REAL)
            elif real_count >= 30:
                strategies.append(
                    TrainingStrategy.HYBRID
                    if synthetic_count > 50
                    else TrainingStrategy.PURE_REAL
                )
            else:
                strategies.append(TrainingStrategy.SKIP)

        result = model.train(contexts, strategies)

        assert result["status"] == "success"
        assert result["samples"] == 50
        assert model.is_trained is True

    def test_select_strategy_with_trained_model(self, model):
        """Test strategy selection with trained model"""
        # Train model
        contexts = []
        strategies = []

        for i in range(30):
            context = ModelingContext(
                modality=QueryModality.VIDEO,
                real_sample_count=100 + i * 5,
                synthetic_sample_count=50,
                success_rate=0.85,
                avg_confidence=0.9,
                days_since_last_training=7,
                current_performance_score=0.85,
            )
            contexts.append(context)
            strategies.append(TrainingStrategy.PURE_REAL)

        model.train(contexts, strategies)

        # Test prediction
        test_context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=120,
            synthetic_sample_count=50,
            success_rate=0.85,
            avg_confidence=0.9,
            days_since_last_training=7,
            current_performance_score=0.85,
        )

        strategy = model.select_strategy(test_context)
        assert isinstance(strategy, TrainingStrategy)


class TestFusionBenefitModel:
    """Test FusionBenefitModel functionality"""

    @pytest.fixture
    def model(self):
        """Create model instance with telemetry-backed storage."""
        return FusionBenefitModel(
            telemetry_provider=_make_mock_telemetry_provider(),
            tenant_id="test-tenant",
        )

    def test_initialization(self, model):
        """Test model initialization"""
        assert model.is_trained is False

    def test_fallback_benefit_high_disagreement(self, model):
        """Test fallback benefit with high modality disagreement"""
        context = {
            "primary_modality_confidence": 0.7,
            "secondary_modality_confidence": 0.6,
            "modality_agreement": 0.2,  # Low agreement
            "query_ambiguity_score": 0.8,
            "historical_fusion_success_rate": 0.75,
        }

        benefit = model._fallback_benefit(context)

        assert isinstance(benefit, float)
        assert 0 <= benefit <= 1
        # High disagreement and ambiguity should increase benefit
        assert benefit > 0.4

    def test_fallback_benefit_high_agreement(self, model):
        """Test fallback benefit with high modality agreement"""
        context = {
            "primary_modality_confidence": 0.9,
            "secondary_modality_confidence": 0.3,
            "modality_agreement": 0.9,  # High agreement
            "query_ambiguity_score": 0.2,
            "historical_fusion_success_rate": 0.75,
        }

        benefit = model._fallback_benefit(context)

        assert isinstance(benefit, float)
        assert 0 <= benefit <= 1
        # High agreement and low ambiguity should decrease benefit
        assert benefit < 0.5

    def test_contexts_to_features(self, model):
        """Test converting fusion contexts to features"""
        contexts = [
            {
                "primary_modality_confidence": 0.8,
                "secondary_modality_confidence": 0.6,
                "modality_agreement": 0.5,
                "query_ambiguity_score": 0.7,
                "historical_fusion_success_rate": 0.75,
            },
            {
                "primary_modality_confidence": 0.9,
                "secondary_modality_confidence": 0.4,
                "modality_agreement": 0.8,
                "query_ambiguity_score": 0.3,
                "historical_fusion_success_rate": 0.70,
            },
        ]

        features = model._contexts_to_features(contexts)

        assert features.shape == (2, 5)  # 2 contexts, 5 features
        assert features[0, 0] == 0.8  # primary_modality_confidence
        assert features[1, 2] == 0.8  # modality_agreement

    def test_train_insufficient_data(self, model):
        """Test training with insufficient data"""
        contexts = [
            {
                "primary_modality_confidence": 0.8,
                "secondary_modality_confidence": 0.6,
                "modality_agreement": 0.5,
                "query_ambiguity_score": 0.7,
                "historical_fusion_success_rate": 0.75,
            }
        ]
        benefits = [0.3]

        result = model.train(contexts, benefits)

        assert result["status"] == "insufficient_data"
        assert model.is_trained is False

    def test_train_success(self, model):
        """Test successful training"""
        contexts = []
        benefits = []

        for i in range(30):
            context = {
                "primary_modality_confidence": 0.7 + (i % 3) * 0.1,
                "secondary_modality_confidence": 0.4 + (i % 5) * 0.1,
                "modality_agreement": (i % 10) * 0.1,
                "query_ambiguity_score": 0.5 + (i % 4) * 0.1,
                "historical_fusion_success_rate": 0.7,
            }
            contexts.append(context)

            # Benefit correlates with disagreement and ambiguity
            benefit = (1 - context["modality_agreement"]) * 0.5 + context[
                "query_ambiguity_score"
            ] * 0.3
            benefits.append(min(benefit, 1.0))

        result = model.train(contexts, benefits)

        assert result["status"] == "success"
        assert result["samples"] == 30
        assert "mae" in result
        assert "rmse" in result
        assert model.is_trained is True

    def test_predict_benefit_with_trained_model(self, model):
        """Test benefit prediction with trained model"""
        # Train model
        contexts = []
        benefits = []

        for i in range(20):
            context = {
                "primary_modality_confidence": 0.8,
                "secondary_modality_confidence": 0.5,
                "modality_agreement": 0.5,
                "query_ambiguity_score": 0.6,
                "historical_fusion_success_rate": 0.75,
            }
            contexts.append(context)
            benefits.append(0.4)

        model.train(contexts, benefits)

        # Test prediction
        test_context = {
            "primary_modality_confidence": 0.8,
            "secondary_modality_confidence": 0.5,
            "modality_agreement": 0.5,
            "query_ambiguity_score": 0.6,
            "historical_fusion_success_rate": 0.75,
        }

        benefit = model.predict_benefit(test_context)

        assert isinstance(benefit, float)
        assert 0 <= benefit <= 1

    def test_predict_benefit_clipping(self, model):
        """Test that benefit predictions are clipped to [0, 1]"""
        # Test without trained model (fallback)
        context = {
            "primary_modality_confidence": 1.0,
            "secondary_modality_confidence": 1.0,
            "modality_agreement": 0.0,
            "query_ambiguity_score": 1.0,
            "historical_fusion_success_rate": 1.0,
        }

        benefit = model.predict_benefit(context)

        assert 0 <= benefit <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
