"""
Unit tests for ModalityOptimizer
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer
from cogniverse_agents.routing.synthetic_data_generator import ModalityExample
from cogniverse_agents.routing.xgboost_meta_models import (
    ModelingContext,
    TrainingStrategy,
)
from cogniverse_agents.search.multi_modal_reranker import QueryModality


class TestModalityOptimizer:
    """Test ModalityOptimizer functionality"""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for models"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_components(self):
        """Create mocked components"""
        with (
            patch(
                "src.app.routing.modality_optimizer.ModalitySpanCollector"
            ) as mock_collector,
            patch(
                "src.app.routing.modality_optimizer.ModalityEvaluator"
            ) as mock_evaluator,
            patch(
                "src.app.routing.modality_optimizer.SyntheticDataGenerator"
            ) as mock_generator,
            patch(
                "src.app.routing.modality_optimizer.TrainingDecisionModel"
            ) as mock_decision,
            patch(
                "src.app.routing.modality_optimizer.TrainingStrategyModel"
            ) as mock_strategy,
        ):

            collector = MagicMock()
            evaluator = MagicMock()
            generator = MagicMock()
            decision_model = MagicMock()
            strategy_model = MagicMock()

            mock_collector.return_value = collector
            mock_evaluator.return_value = evaluator
            mock_generator.return_value = generator
            mock_decision.return_value = decision_model
            mock_strategy.return_value = strategy_model

            yield {
                "collector": collector,
                "evaluator": evaluator,
                "generator": generator,
                "decision_model": decision_model,
                "strategy_model": strategy_model,
            }

    @pytest.fixture
    def optimizer(self, temp_model_dir, mock_components):
        """Create optimizer instance with mocked components"""
        return ModalityOptimizer(
            tenant_id="test-tenant", model_dir=temp_model_dir, vespa_client=None
        )

    def test_initialization(self, optimizer, temp_model_dir):
        """Test optimizer initialization"""
        assert optimizer.tenant_id == "test-tenant"
        assert optimizer.model_dir == temp_model_dir
        assert optimizer.span_collector is not None
        assert optimizer.evaluator is not None
        assert optimizer.training_history == {}

    def test_build_modeling_context(self, optimizer):
        """Test building ModelingContext from examples"""
        examples = [
            ModalityExample(
                query="show me videos",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                modality_features={"routing_confidence": 0.9},
                is_synthetic=False,
            ),
            ModalityExample(
                query="watch tutorial",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                modality_features={"routing_confidence": 0.85},
                is_synthetic=False,
            ),
            ModalityExample(
                query="find demonstrations",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=False,
                modality_features={"routing_confidence": 0.7},
                is_synthetic=False,
            ),
        ]

        context = optimizer._build_modeling_context(QueryModality.VIDEO, examples)

        assert context.modality == QueryModality.VIDEO
        assert context.real_sample_count == 3
        assert context.synthetic_sample_count == 0
        assert context.success_rate == 2 / 3  # 2 successes out of 3
        assert 0.8 < context.avg_confidence < 0.9  # Average of 0.9, 0.85, 0.7

    def test_build_modeling_context_with_synthetic(self, optimizer):
        """Test building context with synthetic examples"""
        examples = [
            ModalityExample(
                query="real query",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                is_synthetic=False,
            ),
            ModalityExample(
                query="synthetic query",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                is_synthetic=True,
            ),
        ]

        context = optimizer._build_modeling_context(QueryModality.VIDEO, examples)

        assert context.real_sample_count == 1
        assert context.synthetic_sample_count == 1

    def test_get_days_since_last_training_never_trained(self, optimizer):
        """Test getting days since last training when never trained"""
        days = optimizer._get_days_since_last_training(QueryModality.VIDEO)
        assert days == 999

    def test_get_days_since_last_training_with_history(self, optimizer):
        """Test getting days since last training with history"""
        optimizer.training_history[QueryModality.VIDEO] = [
            {"timestamp": datetime.now(), "result": {}}
        ]

        days = optimizer._get_days_since_last_training(QueryModality.VIDEO)
        assert days == 0

    def test_estimate_data_quality_empty(self, optimizer):
        """Test data quality estimation with no examples"""
        quality = optimizer._estimate_data_quality([])
        assert quality == 0.0

    def test_estimate_data_quality_complete_features(self, optimizer):
        """Test data quality estimation with complete features"""
        examples = [
            ModalityExample(
                query="test query",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                modality_features={
                    "routing_confidence": 0.9,
                    "feature1": "value1",
                    "feature2": "value2",
                    "feature3": "value3",
                    "feature4": "value4",
                    "feature5": "value5",
                },
            )
        ]

        quality = optimizer._estimate_data_quality(examples)
        assert 0 < quality <= 1.0

    def test_estimate_feature_diversity_empty(self, optimizer):
        """Test feature diversity estimation with no examples"""
        diversity = optimizer._estimate_feature_diversity([])
        assert diversity == 0.0

    def test_estimate_feature_diversity_varied_queries(self, optimizer):
        """Test feature diversity with varied queries and agents"""
        examples = [
            ModalityExample(
                query="query 1",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
            ),
            ModalityExample(
                query="query 2",
                modality=QueryModality.VIDEO,
                correct_agent="document_agent",
                success=True,
            ),
            ModalityExample(
                query="query 3",
                modality=QueryModality.VIDEO,
                correct_agent="image_search_agent",
                success=True,
            ),
        ]

        diversity = optimizer._estimate_feature_diversity(examples)
        assert 0 < diversity <= 1.0

    @pytest.mark.asyncio
    async def test_prepare_training_data_skip(self, optimizer):
        """Test preparing training data with SKIP strategy"""
        examples = [
            ModalityExample(
                query="test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
            )
        ]

        data = await optimizer._prepare_training_data(
            QueryModality.VIDEO, examples, TrainingStrategy.SKIP
        )

        assert data == []

    @pytest.mark.asyncio
    async def test_prepare_training_data_pure_real(self, optimizer):
        """Test preparing training data with PURE_REAL strategy"""
        examples = [
            ModalityExample(
                query="test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
            )
        ]

        data = await optimizer._prepare_training_data(
            QueryModality.VIDEO, examples, TrainingStrategy.PURE_REAL
        )

        assert data == examples

    @pytest.mark.asyncio
    async def test_prepare_training_data_synthetic(self, optimizer, mock_components):
        """Test preparing training data with SYNTHETIC strategy"""
        examples = [
            ModalityExample(
                query="test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
            )
        ]

        # Mock synthetic generation
        synthetic_examples = [
            ModalityExample(
                query="synthetic test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                is_synthetic=True,
            )
        ]
        mock_components["generator"].generate_from_ingested_data = AsyncMock(
            return_value=synthetic_examples
        )

        data = await optimizer._prepare_training_data(
            QueryModality.VIDEO, examples, TrainingStrategy.SYNTHETIC
        )

        assert len(data) == 1
        assert data[0].is_synthetic is True

    @pytest.mark.asyncio
    async def test_prepare_training_data_hybrid(self, optimizer, mock_components):
        """Test preparing training data with HYBRID strategy"""
        examples = [
            ModalityExample(
                query="real test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                is_synthetic=False,
            )
        ]

        # Mock synthetic generation
        synthetic_examples = [
            ModalityExample(
                query="synthetic test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                is_synthetic=True,
            )
        ]
        mock_components["generator"].generate_from_ingested_data = AsyncMock(
            return_value=synthetic_examples
        )

        data = await optimizer._prepare_training_data(
            QueryModality.VIDEO, examples, TrainingStrategy.HYBRID
        )

        assert len(data) == 2  # 1 real + 1 synthetic
        assert any(ex.is_synthetic for ex in data)
        assert any(not ex.is_synthetic for ex in data)

    def test_train_modality_model(self, optimizer):
        """Test training modality model (placeholder)"""
        training_data = [
            ModalityExample(
                query="test",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
            )
        ]

        result = optimizer._train_modality_model(
            QueryModality.VIDEO, training_data, TrainingStrategy.PURE_REAL
        )

        assert result["status"] == "success"
        assert result["training_samples"] == 1
        assert result["strategy"] == TrainingStrategy.PURE_REAL.value
        assert "validation_accuracy" in result
        assert (
            "optimizer" in result
        )  # DSPy optimizer used (MIPROv2 or BootstrapFewShot)
        assert "model_path" in result

    def test_record_training(self, optimizer):
        """Test recording training in history"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=10,
            synthetic_sample_count=5,
            success_rate=0.8,
            avg_confidence=0.85,
            days_since_last_training=7,
            current_performance_score=0.82,
        )

        result = {"status": "success", "accuracy": 0.9}

        optimizer._record_training(
            QueryModality.VIDEO, context, TrainingStrategy.PURE_REAL, result
        )

        assert QueryModality.VIDEO in optimizer.training_history
        assert len(optimizer.training_history[QueryModality.VIDEO]) == 1

        recorded = optimizer.training_history[QueryModality.VIDEO][0]
        assert recorded["strategy"] == TrainingStrategy.PURE_REAL.value
        assert recorded["result"] == result

    def test_context_to_dict(self, optimizer):
        """Test converting ModelingContext to dict"""
        context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=10,
            synthetic_sample_count=5,
            success_rate=0.8,
            avg_confidence=0.85,
            days_since_last_training=7,
            current_performance_score=0.82,
        )

        context_dict = optimizer._context_to_dict(context)

        assert context_dict["modality"] == "video"
        assert context_dict["real_sample_count"] == 10
        assert context_dict["synthetic_sample_count"] == 5
        assert context_dict["success_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_optimize_modality_skip_training(self, optimizer, mock_components):
        """Test optimize_modality when training is skipped"""
        # Mock evaluator to return examples
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={"routing_confidence": 0.9},
                )
            ]
        }
        mock_components["evaluator"].create_training_examples = AsyncMock(
            return_value=examples
        )

        # Mock decision model to skip training
        mock_components["decision_model"].should_train = MagicMock(
            return_value=(False, 0.01)
        )

        result = await optimizer.optimize_modality(QueryModality.VIDEO)

        assert result["trained"] is False
        assert result["reason"] == "insufficient_benefit"

    @pytest.mark.asyncio
    async def test_optimize_modality_with_training(self, optimizer, mock_components):
        """Test optimize_modality with training"""
        # Mock evaluator to return examples
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={"routing_confidence": 0.9},
                )
            ]
        }
        mock_components["evaluator"].create_training_examples = AsyncMock(
            return_value=examples
        )

        # Mock decision model to train
        mock_components["decision_model"].should_train = MagicMock(
            return_value=(True, 0.05)
        )

        # Mock strategy model
        mock_components["strategy_model"].select_strategy = MagicMock(
            return_value=TrainingStrategy.PURE_REAL
        )

        result = await optimizer.optimize_modality(QueryModality.VIDEO)

        assert result["trained"] is True
        assert result["strategy"] == TrainingStrategy.PURE_REAL.value
        assert result["examples_count"] == 1

    @pytest.mark.asyncio
    async def test_optimize_modality_force_training(self, optimizer, mock_components):
        """Test optimize_modality with force_training=True"""
        # Mock evaluator to return examples
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={"routing_confidence": 0.9},
                )
            ]
        }
        mock_components["evaluator"].create_training_examples = AsyncMock(
            return_value=examples
        )

        # Mock strategy model
        mock_components["strategy_model"].select_strategy = MagicMock(
            return_value=TrainingStrategy.PURE_REAL
        )

        result = await optimizer.optimize_modality(
            QueryModality.VIDEO, force_training=True
        )

        assert result["trained"] is True
        # should_train should not be called when force_training=True
        mock_components["decision_model"].should_train.assert_not_called()

    @pytest.mark.asyncio
    async def test_optimize_all_modalities(self, optimizer, mock_components):
        """Test optimizing all modalities"""
        # Mock span collector stats
        mock_components["collector"].get_modality_statistics = AsyncMock(
            return_value={
                "modality_distribution": {
                    "video": {"count": 10},
                    "document": {"count": 5},
                }
            }
        )

        # Mock evaluator
        mock_components["evaluator"].create_training_examples = AsyncMock(
            return_value={
                QueryModality.VIDEO: [
                    ModalityExample(
                        query="test",
                        modality=QueryModality.VIDEO,
                        correct_agent="video_search_agent",
                        success=True,
                        modality_features={"routing_confidence": 0.9},
                    )
                ],
                QueryModality.DOCUMENT: [
                    ModalityExample(
                        query="test doc",
                        modality=QueryModality.DOCUMENT,
                        correct_agent="document_agent",
                        success=True,
                        modality_features={"routing_confidence": 0.85},
                    )
                ],
            }
        )

        # Mock decision and strategy models
        mock_components["decision_model"].should_train = MagicMock(
            return_value=(True, 0.05)
        )
        mock_components["strategy_model"].select_strategy = MagicMock(
            return_value=TrainingStrategy.PURE_REAL
        )

        results = await optimizer.optimize_all_modalities()

        assert len(results) == 2
        assert QueryModality.VIDEO in results
        assert QueryModality.DOCUMENT in results

    def test_get_training_history_specific_modality(self, optimizer):
        """Test getting training history for specific modality"""
        optimizer.training_history[QueryModality.VIDEO] = [
            {"timestamp": datetime.now(), "result": {}}
        ]

        history = optimizer.get_training_history(QueryModality.VIDEO)

        assert len(history) == 1
        assert QueryModality.VIDEO in history

    def test_get_training_history_all(self, optimizer):
        """Test getting training history for all modalities"""
        optimizer.training_history[QueryModality.VIDEO] = [
            {"timestamp": datetime.now(), "result": {}}
        ]
        optimizer.training_history[QueryModality.DOCUMENT] = [
            {"timestamp": datetime.now(), "result": {}}
        ]

        history = optimizer.get_training_history()

        assert len(history) == 2
        assert QueryModality.VIDEO in history
        assert QueryModality.DOCUMENT in history

    def test_get_optimization_summary(self, optimizer):
        """Test getting optimization summary"""
        optimizer.training_history[QueryModality.VIDEO] = [
            {
                "timestamp": datetime.now(),
                "strategy": TrainingStrategy.PURE_REAL.value,
                "result": {"status": "success"},
            }
        ]

        summary = optimizer.get_optimization_summary()

        assert summary["total_modalities"] == 1
        assert "video" in summary["modalities"]
        assert summary["modalities"]["video"]["training_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
