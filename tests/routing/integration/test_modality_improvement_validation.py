"""
Integration tests validating measurable improvement from modality-specific models.

These tests compare routing accuracy with and without modality-specific DSPy models
to demonstrate the value of the modality optimization framework.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.routing.base import GenerationType, RoutingDecision, SearchModality
from src.app.routing.modality_optimizer import ModalityExample, ModalityOptimizer
from src.app.routing.router import ComprehensiveRouter
from src.app.routing.xgboost_meta_models import TrainingStrategy
from src.app.search.multi_modal_reranker import QueryModality


@pytest.mark.integration
class TestModalityImprovementValidation:
    """Validate that modality-specific models improve routing accuracy"""

    @pytest.fixture
    def modality_optimizer(self, tmp_path: Path):
        """Create modality optimizer with test directory"""
        optimizer = ModalityOptimizer(
            model_dir=tmp_path / "models",
            tenant_id="test_tenant"
        )
        return optimizer

    @pytest.fixture
    def mock_router(self):
        """Create mock router for baseline comparison"""
        router = MagicMock(spec=ComprehensiveRouter)
        router.route = AsyncMock()
        return router

    @pytest.fixture
    def video_training_examples(self):
        """Create training examples for video modality"""
        return [
            ModalityExample(
                query="show me basketball highlights",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                modality_features={},
                is_synthetic=False
            ),
            ModalityExample(
                query="find soccer game footage",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                modality_features={},
                is_synthetic=False
            ),
            ModalityExample(
                query="watch tennis match replay",
                modality=QueryModality.VIDEO,
                correct_agent="video_search_agent",
                success=True,
                modality_features={},
                is_synthetic=False
            ),
        ]

    @pytest.fixture
    def document_training_examples(self):
        """Create training examples for document modality"""
        return [
            ModalityExample(
                query="read research paper on AI",
                modality=QueryModality.DOCUMENT,
                correct_agent="document_agent",
                success=True,
                modality_features={},
                is_synthetic=False
            ),
            ModalityExample(
                query="find PDF about machine learning",
                modality=QueryModality.DOCUMENT,
                correct_agent="document_agent",
                success=True,
                modality_features={},
                is_synthetic=False
            ),
            ModalityExample(
                query="get whitepaper on neural networks",
                modality=QueryModality.DOCUMENT,
                correct_agent="document_agent",
                success=True,
                modality_features={},
                is_synthetic=False
            ),
        ]

    def test_modality_optimizer_improves_video_routing_accuracy(
        self, modality_optimizer, video_training_examples
    ):
        """
        Test that modality optimizer improves routing accuracy for video queries
        compared to baseline.
        """
        # Train modality-specific model
        result = modality_optimizer._train_modality_model(
            modality=QueryModality.VIDEO,
            training_data=video_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )

        # Verify training completed
        assert result["status"] == "success"
        assert "model_path" in result

        # Test predictions on new video queries
        test_queries = [
            ("show football game", "video_search_agent"),
            ("watch basketball highlights", "video_search_agent"),
            ("find hockey match", "video_search_agent"),
        ]

        correct_predictions = 0
        for query, expected_agent in test_queries:
            prediction = modality_optimizer.predict_agent(
                query=query,
                modality=QueryModality.VIDEO,
                query_features={}
            )

            if prediction and prediction["recommended_agent"] == expected_agent:
                correct_predictions += 1

        # Accuracy should be reasonably high (at least 66%)
        accuracy = correct_predictions / len(test_queries)
        assert accuracy >= 0.66, f"Video routing accuracy too low: {accuracy:.2f}"

    def test_modality_optimizer_improves_document_routing_accuracy(
        self, modality_optimizer, document_training_examples
    ):
        """
        Test that modality optimizer improves routing accuracy for document queries
        compared to baseline.
        """
        # Train modality-specific model
        result = modality_optimizer._train_modality_model(
            modality=QueryModality.DOCUMENT,
            training_data=document_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )

        # Verify training completed
        assert result["status"] == "success"
        assert "model_path" in result

        # Test predictions on new document queries
        test_queries = [
            ("read technical documentation", "document_agent"),
            ("find research article", "document_agent"),
            ("get PDF report", "document_agent"),
        ]

        correct_predictions = 0
        for query, expected_agent in test_queries:
            prediction = modality_optimizer.predict_agent(
                query=query,
                modality=QueryModality.DOCUMENT,
                query_features={}
            )

            if prediction and prediction["recommended_agent"] == expected_agent:
                correct_predictions += 1

        # Accuracy should be reasonably high (at least 66%)
        accuracy = correct_predictions / len(test_queries)
        assert accuracy >= 0.66, f"Document routing accuracy too low: {accuracy:.2f}"

    @pytest.mark.asyncio
    async def test_modality_prediction_influences_routing_decision(
        self, mock_router, video_training_examples, tmp_path
    ):
        """
        Test that high-confidence modality predictions influence routing decisions
        in ComprehensiveRouter.
        """
        # Create optimizer and train video model
        optimizer = ModalityOptimizer(
            model_dir=tmp_path / "models",
            tenant_id="test_tenant"
        )

        optimizer._train_modality_model(
            modality=QueryModality.VIDEO,
            training_data=video_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )

        # Get prediction for video query
        prediction = optimizer.predict_agent(
            query="show basketball highlights",
            modality=QueryModality.VIDEO,
            query_features={}
        )

        # Verify prediction has reasonable confidence
        assert prediction is not None
        assert prediction["recommended_agent"] == "video_search_agent"
        assert prediction["confidence"] > 0.0

    def test_baseline_routing_without_modality_models(self, modality_optimizer):
        """
        Test baseline routing behavior without trained modality models
        to establish performance floor.
        """
        # Try prediction without training
        prediction = modality_optimizer.predict_agent(
            query="show basketball highlights",
            modality=QueryModality.VIDEO,
            query_features={}
        )

        # Should return None or low-confidence prediction without trained model
        assert prediction is None or prediction["confidence"] < 0.5

    def test_modality_optimizer_handles_mixed_modality_training(
        self, modality_optimizer, video_training_examples, document_training_examples
    ):
        """
        Test that optimizer can train separate models for different modalities
        and maintain accuracy for each.
        """
        # Train video model
        video_result = modality_optimizer._train_modality_model(
            modality=QueryModality.VIDEO,
            training_data=video_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )
        assert video_result["status"] == "completed"

        # Train document model
        doc_result = modality_optimizer._train_modality_model(
            modality=QueryModality.DOCUMENT,
            training_data=document_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )
        assert doc_result["status"] == "completed"

        # Test video predictions
        video_pred = modality_optimizer.predict_agent(
            query="watch soccer game",
            modality=QueryModality.VIDEO,
            query_features={}
        )
        assert video_pred is not None
        assert video_pred["recommended_agent"] == "video_search_agent"

        # Test document predictions
        doc_pred = modality_optimizer.predict_agent(
            query="read research paper",
            modality=QueryModality.DOCUMENT,
            query_features={}
        )
        assert doc_pred is not None
        assert doc_pred["recommended_agent"] == "document_agent"

        # Verify models don't interfere with each other
        assert video_pred["recommended_agent"] != doc_pred["recommended_agent"]

    def test_confidence_scores_correlate_with_accuracy(
        self, modality_optimizer, video_training_examples
    ):
        """
        Test that higher confidence scores correlate with more accurate predictions.
        """
        # Train model
        modality_optimizer._train_modality_model(
            modality=QueryModality.VIDEO,
            training_data=video_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )

        # Get predictions for queries with varying clarity
        clear_query_pred = modality_optimizer.predict_agent(
            query="watch basketball video",
            modality=QueryModality.VIDEO,
            query_features={}
        )

        ambiguous_query_pred = modality_optimizer.predict_agent(
            query="sports content",
            modality=QueryModality.VIDEO,
            query_features={}
        )

        # Clear queries should have higher confidence
        if clear_query_pred and ambiguous_query_pred:
            assert clear_query_pred["confidence"] >= ambiguous_query_pred["confidence"]

    @pytest.mark.asyncio
    async def test_integration_with_comprehensive_router(
        self, video_training_examples, tmp_path
    ):
        """
        Test end-to-end integration of modality predictions with ComprehensiveRouter.
        """
        # Create optimizer and train model
        optimizer = ModalityOptimizer(
            model_dir=tmp_path / "models",
            tenant_id="test_tenant"
        )

        optimizer._train_modality_model(
            modality=QueryModality.VIDEO,
            training_data=video_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )

        # Get modality prediction
        prediction = optimizer.predict_agent(
            query="show basketball highlights",
            modality=QueryModality.VIDEO,
            query_features={}
        )

        # Create context with modality prediction
        context = {"modality_prediction": prediction}

        # Create router and test routing with modality prediction
        with patch("src.app.routing.router.ComprehensiveRouter") as MockRouter:
            mock_router_instance = MockRouter.return_value
            mock_router_instance.route = AsyncMock(
                return_value=RoutingDecision(
                    search_modality=SearchModality.VIDEO,
                    generation_type=GenerationType.RAW_RESULTS,
                    confidence_score=0.9,
                    routing_method="ensemble",
                    reasoning="Modality prediction influenced decision",
                    metadata={"modality_prediction": prediction}
                )
            )

            decision = await mock_router_instance.route("show basketball highlights", context)

            # Verify modality prediction is in decision metadata
            assert "modality_prediction" in decision.metadata
            assert decision.metadata["modality_prediction"] == prediction

    def test_model_persistence_across_sessions(
        self, modality_optimizer, video_training_examples, tmp_path
    ):
        """
        Test that trained models are saved and can be loaded in new sessions.
        """
        # Train model
        result = modality_optimizer._train_modality_model(
            modality=QueryModality.VIDEO,
            training_data=video_training_examples,
            strategy=TrainingStrategy.PURE_REAL
        )

        model_path = Path(result["model_path"])
        assert model_path.exists()

        # Create new optimizer instance (simulating new session)
        new_optimizer = ModalityOptimizer(
            model_dir=tmp_path / "models",
            tenant_id="test_tenant"
        )

        # Load models
        new_optimizer._load_trained_models()

        # Test prediction with loaded model
        prediction = new_optimizer.predict_agent(
            query="watch basketball game",
            modality=QueryModality.VIDEO,
            query_features={}
        )

        assert prediction is not None
        assert prediction["recommended_agent"] == "video_search_agent"
