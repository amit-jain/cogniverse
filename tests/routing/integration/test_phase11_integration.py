"""
Integration tests for Phase 11: Multi-Modal Optimization

Tests the complete workflow from span collection through optimization.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer
from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer
from cogniverse_agents.routing.synthetic_data_generator import (
    ModalityExample,
    SyntheticDataGenerator,
)
from cogniverse_agents.routing.xgboost_meta_models import TrainingStrategy
from cogniverse_agents.search.multi_modal_reranker import QueryModality


class TestPhase11Integration:
    """Integration tests for Phase 11 components"""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for models"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self, temp_model_dir):
        """Test complete optimization workflow from spans to training"""
        # Mock Phoenix client to avoid external dependencies
        with patch("src.app.routing.modality_span_collector.px.Client"):
            # Step 1: Initialize components
            optimizer = ModalityOptimizer(
                tenant_id="test-tenant",
                model_dir=temp_model_dir,
                vespa_client=None,
            )

            # Step 2: Mock span collector to return test spans
            test_examples = {
                QueryModality.VIDEO: [
                    ModalityExample(
                        query="show me machine learning videos",
                        modality=QueryModality.VIDEO,
                        correct_agent="video_search_agent",
                        success=True,
                        modality_features={"routing_confidence": 0.9},
                        is_synthetic=False,
                    ),
                    ModalityExample(
                        query="watch neural networks tutorial",
                        modality=QueryModality.VIDEO,
                        correct_agent="video_search_agent",
                        success=True,
                        modality_features={"routing_confidence": 0.85},
                        is_synthetic=False,
                    ),
                ]
            }

            optimizer.evaluator.create_training_examples = AsyncMock(
                return_value=test_examples
            )

            # Step 3: Mock decision and strategy models to test workflow
            optimizer.training_decision_model.should_train = MagicMock(
                return_value=(True, 0.05)
            )
            optimizer.training_strategy_model.select_strategy = MagicMock(
                return_value=TrainingStrategy.PURE_REAL
            )

            # Step 4: Run optimization
            result = await optimizer.optimize_modality(QueryModality.VIDEO)

            # Verify workflow completed
            assert result["trained"] is True
            assert result["modality"] == "video"
            assert result["strategy"] == TrainingStrategy.PURE_REAL.value
            assert result["examples_count"] == 2

            # Verify training was recorded
            assert QueryModality.VIDEO in optimizer.training_history
            assert len(optimizer.training_history[QueryModality.VIDEO]) == 1

    @pytest.mark.asyncio
    async def test_synthetic_data_augmentation(self, temp_model_dir):
        """Test synthetic data generation and augmentation"""
        # Initialize generator
        generator = SyntheticDataGenerator(vespa_client=None)

        # Generate synthetic examples
        synthetic_examples = await generator.generate_from_ingested_data(
            QueryModality.VIDEO, target_count=10
        )

        # Verify synthetic data
        assert len(synthetic_examples) == 10
        assert all(ex.is_synthetic for ex in synthetic_examples)
        assert all(ex.modality == QueryModality.VIDEO for ex in synthetic_examples)
        assert all(
            ex.correct_agent == "video_search_agent" for ex in synthetic_examples
        )

        # Verify variety in generated queries
        unique_queries = set(ex.query for ex in synthetic_examples)
        assert len(unique_queries) > 3  # Should have some variety

    @pytest.mark.asyncio
    async def test_cross_modal_fusion_workflow(self, temp_model_dir):
        """Test cross-modal fusion recommendation workflow"""
        # Initialize optimizer
        fusion_optimizer = CrossModalOptimizer(model_dir=temp_model_dir)

        # Test fusion recommendation with high benefit
        recommendations = fusion_optimizer.get_fusion_recommendations(
            query_text="show me videos and documents about machine learning",
            detected_modalities=[
                (QueryModality.VIDEO, 0.7),
                (QueryModality.DOCUMENT, 0.65),
            ],
            fusion_threshold=0.5,
        )

        # Should recommend fusion for ambiguous multi-modal query
        assert "should_fuse" in recommendations
        assert "fusion_benefit" in recommendations
        assert "primary_modality" in recommendations

        # Record fusion result
        if recommendations["should_fuse"]:
            fusion_context = {
                "primary_modality_confidence": 0.7,
                "secondary_modality_confidence": 0.65,
                "modality_agreement": 0.8,
                "query_ambiguity_score": 0.6,
                "historical_fusion_success_rate": 0.7,
            }
            fusion_optimizer.record_fusion_result(
                primary_modality=QueryModality.VIDEO,
                secondary_modality=QueryModality.DOCUMENT,
                fusion_context=fusion_context,
                success=True,
                improvement=0.15,
            )

            # Verify fusion was recorded
            assert len(fusion_optimizer.fusion_history) == 1

            # Verify success rate was updated
            pair = (QueryModality.VIDEO, QueryModality.DOCUMENT)
            assert pair in fusion_optimizer.fusion_success_rates

    @pytest.mark.asyncio
    async def test_modality_optimizer_with_synthetic_strategy(self, temp_model_dir):
        """Test optimizer with synthetic training strategy"""
        with patch("src.app.routing.modality_span_collector.px.Client"):
            optimizer = ModalityOptimizer(
                tenant_id="test-tenant",
                model_dir=temp_model_dir,
                vespa_client=None,
            )

            # Mock to return no real examples (cold start scenario)
            optimizer.evaluator.create_training_examples = AsyncMock(
                return_value={QueryModality.VIDEO: []}
            )

            # Mock decision to train
            optimizer.training_decision_model.should_train = MagicMock(
                return_value=(True, 0.08)
            )

            # Mock strategy to use synthetic data
            optimizer.training_strategy_model.select_strategy = MagicMock(
                return_value=TrainingStrategy.SYNTHETIC
            )

            # Mock synthetic generation
            optimizer.synthetic_generator.generate_from_ingested_data = AsyncMock(
                return_value=[
                    ModalityExample(
                        query="synthetic query",
                        modality=QueryModality.VIDEO,
                        correct_agent="video_search_agent",
                        success=True,
                        is_synthetic=True,
                    )
                ]
            )

            # Run optimization
            result = await optimizer.optimize_modality(QueryModality.VIDEO)

            # Verify synthetic strategy was used
            assert result["trained"] is True
            assert result["strategy"] == TrainingStrategy.SYNTHETIC.value
            assert result["examples_count"] > 0

    @pytest.mark.asyncio
    async def test_optimize_all_modalities(self, temp_model_dir):
        """Test optimizing multiple modalities"""
        with patch("src.app.routing.modality_span_collector.px.Client"):
            optimizer = ModalityOptimizer(
                tenant_id="test-tenant",
                model_dir=temp_model_dir,
                vespa_client=None,
            )

            # Mock span collector stats
            optimizer.span_collector.get_modality_statistics = AsyncMock(
                return_value={
                    "modality_distribution": {
                        "video": {"count": 10},
                        "document": {"count": 5},
                    }
                }
            )

            # Mock evaluator
            optimizer.evaluator.create_training_examples = AsyncMock(
                return_value={
                    QueryModality.VIDEO: [
                        ModalityExample(
                            query="test video",
                            modality=QueryModality.VIDEO,
                            correct_agent="video_search_agent",
                            success=True,
                            modality_features={"routing_confidence": 0.9},
                        )
                    ],
                    QueryModality.DOCUMENT: [
                        ModalityExample(
                            query="test document",
                            modality=QueryModality.DOCUMENT,
                            correct_agent="document_agent",
                            success=True,
                            modality_features={"routing_confidence": 0.85},
                        )
                    ],
                }
            )

            # Mock decision models
            optimizer.training_decision_model.should_train = MagicMock(
                return_value=(True, 0.05)
            )
            optimizer.training_strategy_model.select_strategy = MagicMock(
                return_value=TrainingStrategy.PURE_REAL
            )

            # Run batch optimization
            results = await optimizer.optimize_all_modalities()

            # Verify both modalities were optimized
            assert len(results) == 2
            assert QueryModality.VIDEO in results
            assert QueryModality.DOCUMENT in results

    def test_fusion_statistics_and_export(self, temp_model_dir):
        """Test fusion statistics collection and export"""
        fusion_optimizer = CrossModalOptimizer(model_dir=temp_model_dir)

        # Record multiple fusion results
        for i in range(5):
            fusion_context = {
                "primary_modality_confidence": 0.8,
                "secondary_modality_confidence": 0.6,
                "modality_agreement": 0.7,
                "query_ambiguity_score": 0.5,
                "historical_fusion_success_rate": 0.7,
            }
            fusion_optimizer.record_fusion_result(
                primary_modality=QueryModality.VIDEO,
                secondary_modality=QueryModality.DOCUMENT,
                fusion_context=fusion_context,
                success=(i % 2 == 0),  # Alternate success/failure
                improvement=0.1 if i % 2 == 0 else 0.0,
            )

        # Get statistics
        stats = fusion_optimizer.get_fusion_statistics()

        assert stats["total_fusions"] == 5
        assert "video+document" in stats["modality_pairs"]
        assert stats["modality_pairs"]["video+document"]["count"] == 5

        # Export data
        export_path = temp_model_dir / "fusion_export.json"
        success = fusion_optimizer.export_fusion_data(export_path)

        assert success is True
        assert export_path.exists()

    @pytest.mark.asyncio
    async def test_optimization_summary(self, temp_model_dir):
        """Test getting optimization summary"""
        with patch("src.app.routing.modality_span_collector.px.Client"):
            optimizer = ModalityOptimizer(
                tenant_id="test-tenant",
                model_dir=temp_model_dir,
                vespa_client=None,
            )

            # Add some training history
            optimizer.training_history[QueryModality.VIDEO] = [
                {
                    "timestamp": datetime.now(),
                    "strategy": TrainingStrategy.PURE_REAL.value,
                    "result": {"status": "success", "accuracy": 0.9},
                }
            ]

            # Get summary
            summary = optimizer.get_optimization_summary()

            assert summary["total_modalities"] == 1
            assert "video" in summary["modalities"]
            assert summary["modalities"]["video"]["training_count"] == 1

    def test_modality_context_building(self, temp_model_dir):
        """Test building modeling context from examples"""
        with patch("src.app.routing.modality_span_collector.px.Client"):
            optimizer = ModalityOptimizer(
                tenant_id="test-tenant",
                model_dir=temp_model_dir,
                vespa_client=None,
            )

            examples = [
                ModalityExample(
                    query="query 1",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={"routing_confidence": 0.9},
                ),
                ModalityExample(
                    query="query 2",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=False,
                    modality_features={"routing_confidence": 0.6},
                ),
            ]

            context = optimizer._build_modeling_context(QueryModality.VIDEO, examples)

            assert context.modality == QueryModality.VIDEO
            assert context.real_sample_count == 2
            assert context.success_rate == 0.5  # 1/2
            assert 0.7 < context.avg_confidence < 0.8  # Average of 0.9 and 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
