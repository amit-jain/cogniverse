"""
Unit tests for CrossModalOptimizer
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer
from cogniverse_agents.search.multi_modal_reranker import QueryModality


class TestCrossModalOptimizer:
    """Test CrossModalOptimizer functionality"""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for models"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_fusion_model(self):
        """Create mocked FusionBenefitModel"""
        with patch(
            "cogniverse_agents.routing.cross_modal_optimizer.FusionBenefitModel"
        ) as mock:
            model = MagicMock()
            model.is_trained = False
            model.predict_benefit = MagicMock(return_value=0.6)
            mock.return_value = model
            yield model

    @pytest.fixture
    def mock_span_collector(self):
        """Create mocked ModalitySpanCollector"""
        with patch(
            "cogniverse_agents.routing.cross_modal_optimizer.ModalitySpanCollector"
        ) as mock:
            collector = MagicMock()
            mock.return_value = collector
            yield collector

    @pytest.fixture
    def optimizer(self, temp_model_dir, mock_fusion_model, mock_span_collector):
        """Create optimizer instance"""
        return CrossModalOptimizer(tenant_id="test-tenant", model_dir=temp_model_dir)

    def test_initialization(self, optimizer, temp_model_dir):
        """Test optimizer initialization"""
        assert optimizer.model_dir == temp_model_dir
        assert optimizer.fusion_model is not None
        assert optimizer.fusion_history == []
        assert optimizer.fusion_success_rates == {}

    def test_calculate_modality_agreement_same_modality(self, optimizer):
        """Test agreement calculation when modalities are the same"""
        agreement = optimizer._calculate_modality_agreement(
            QueryModality.VIDEO, QueryModality.VIDEO, 0.8, 0.7
        )
        assert agreement == 1.0

    def test_calculate_modality_agreement_different_similar_confidence(self, optimizer):
        """Test agreement with different modalities but similar confidence"""
        agreement = optimizer._calculate_modality_agreement(
            QueryModality.VIDEO, QueryModality.DOCUMENT, 0.8, 0.75
        )
        # Small confidence diff -> high agreement
        assert 0.9 < agreement <= 1.0

    def test_calculate_modality_agreement_different_large_confidence_diff(
        self, optimizer
    ):
        """Test agreement with different modalities and large confidence difference"""
        agreement = optimizer._calculate_modality_agreement(
            QueryModality.VIDEO, QueryModality.DOCUMENT, 0.9, 0.3
        )
        # Large confidence diff -> low agreement
        assert 0.0 <= agreement < 0.8

    def test_calculate_modality_agreement_no_secondary(self, optimizer):
        """Test agreement when no secondary modality"""
        agreement = optimizer._calculate_modality_agreement(
            QueryModality.VIDEO, None, 0.8, 0.0
        )
        assert agreement == 1.0

    def test_calculate_query_ambiguity_low_confidence(self, optimizer):
        """Test ambiguity calculation with low primary confidence"""
        ambiguity = optimizer._calculate_query_ambiguity("test query", 0.5, 0.0)
        # Low confidence -> high ambiguity
        assert ambiguity > 0.4

    def test_calculate_query_ambiguity_high_confidence(self, optimizer):
        """Test ambiguity calculation with high primary confidence"""
        ambiguity = optimizer._calculate_query_ambiguity("test query", 0.95, 0.0)
        # High confidence -> low ambiguity
        assert ambiguity < 0.5

    def test_calculate_query_ambiguity_similar_confidences(self, optimizer):
        """Test ambiguity with similar primary and secondary confidence"""
        ambiguity = optimizer._calculate_query_ambiguity("test query", 0.6, 0.65)
        # Similar confidences -> high ambiguity
        assert ambiguity > 0.4

    def test_calculate_query_ambiguity_ambiguous_keywords(self, optimizer):
        """Test ambiguity detection with ambiguous keywords"""
        ambiguity = optimizer._calculate_query_ambiguity(
            "show me videos or documents", 0.8, 0.0
        )
        # "or" keyword -> higher ambiguity than without
        # Ambiguity = (0.2 * 0.7) + (0.8 * 0.3) = 0.14 + 0.24 = 0.38
        assert ambiguity > 0.35

    def test_calculate_query_ambiguity_short_query(self, optimizer):
        """Test ambiguity with short query"""
        ambiguity = optimizer._calculate_query_ambiguity("test it", 0.8, 0.0)
        # 2 words -> ambiguous (0.6 * 0.3 = 0.18) + (0.2 * 0.7 = 0.14) = 0.32
        assert ambiguity > 0.3

    def test_get_historical_fusion_success_no_history(self, optimizer):
        """Test getting historical success with no history"""
        success_rate = optimizer._get_historical_fusion_success(
            QueryModality.VIDEO, QueryModality.DOCUMENT
        )
        assert success_rate == 0.7  # Default

    def test_get_historical_fusion_success_with_history(self, optimizer):
        """Test getting historical success with recorded history"""
        pair = (QueryModality.VIDEO, QueryModality.DOCUMENT)
        optimizer.fusion_success_rates[pair] = 0.85

        success_rate = optimizer._get_historical_fusion_success(
            QueryModality.VIDEO, QueryModality.DOCUMENT
        )
        assert success_rate == 0.85

    def test_build_fusion_context(self, optimizer):
        """Test building fusion context"""
        context = optimizer._build_fusion_context(
            primary_modality=QueryModality.VIDEO,
            primary_confidence=0.8,
            secondary_modality=QueryModality.DOCUMENT,
            secondary_confidence=0.6,
            query_text="show me videos and documents",
        )

        assert "primary_modality_confidence" in context
        assert "secondary_modality_confidence" in context
        assert "modality_agreement" in context
        assert "query_ambiguity_score" in context
        assert "historical_fusion_success_rate" in context

        assert context["primary_modality_confidence"] == 0.8
        assert context["secondary_modality_confidence"] == 0.6

    def test_predict_fusion_benefit(self, optimizer, mock_fusion_model):
        """Test predicting fusion benefit"""
        mock_fusion_model.predict_benefit = MagicMock(return_value=0.7)

        benefit = optimizer.predict_fusion_benefit(
            primary_modality=QueryModality.VIDEO,
            primary_confidence=0.8,
            secondary_modality=QueryModality.DOCUMENT,
            secondary_confidence=0.6,
            query_text="test query",
        )

        assert benefit == 0.7
        assert mock_fusion_model.predict_benefit.called

    def test_record_fusion_result(self, optimizer):
        """Test recording fusion result"""
        fusion_context = {
            "primary_modality_confidence": 0.8,
            "secondary_modality_confidence": 0.6,
            "modality_agreement": 0.7,
            "query_ambiguity_score": 0.5,
            "historical_fusion_success_rate": 0.7,
        }

        optimizer.record_fusion_result(
            primary_modality=QueryModality.VIDEO,
            secondary_modality=QueryModality.DOCUMENT,
            fusion_context=fusion_context,
            success=True,
            improvement=0.15,
        )

        assert len(optimizer.fusion_history) == 1
        recorded = optimizer.fusion_history[0]
        assert recorded["primary_modality"] == "video"
        assert recorded["secondary_modality"] == "document"
        assert recorded["success"] is True
        assert recorded["improvement"] == 0.15

        # Check success rate updated
        pair = (QueryModality.VIDEO, QueryModality.DOCUMENT)
        assert pair in optimizer.fusion_success_rates

    def test_record_fusion_result_updates_success_rate(self, optimizer):
        """Test that recording results updates success rate with EMA"""
        fusion_context = {
            "primary_modality_confidence": 0.8,
            "secondary_modality_confidence": 0.6,
            "modality_agreement": 0.7,
            "query_ambiguity_score": 0.5,
            "historical_fusion_success_rate": 0.7,
        }

        # Record successful fusion
        optimizer.record_fusion_result(
            primary_modality=QueryModality.VIDEO,
            secondary_modality=QueryModality.DOCUMENT,
            fusion_context=fusion_context,
            success=True,
            improvement=0.15,
        )

        pair = (QueryModality.VIDEO, QueryModality.DOCUMENT)
        first_rate = optimizer.fusion_success_rates[pair]

        # Record another successful fusion
        optimizer.record_fusion_result(
            primary_modality=QueryModality.VIDEO,
            secondary_modality=QueryModality.DOCUMENT,
            fusion_context=fusion_context,
            success=True,
            improvement=0.10,
        )

        second_rate = optimizer.fusion_success_rates[pair]

        # Rate should change due to EMA
        assert second_rate != first_rate

    def test_train_fusion_model_insufficient_data(self, optimizer, mock_fusion_model):
        """Test training with insufficient data"""
        # Add only 5 records (need 10)
        for i in range(5):
            optimizer.fusion_history.append(
                {
                    "fusion_context": {},
                    "success": True,
                    "improvement": 0.1,
                }
            )

        result = optimizer.train_fusion_model()

        assert result["status"] == "insufficient_data"
        assert result["samples"] == 5

    def test_train_fusion_model_success(self, optimizer, mock_fusion_model):
        """Test successful training"""
        # Add sufficient records
        for i in range(15):
            optimizer.fusion_history.append(
                {
                    "fusion_context": {
                        "primary_modality_confidence": 0.8,
                        "secondary_modality_confidence": 0.6,
                        "modality_agreement": 0.7,
                        "query_ambiguity_score": 0.5,
                        "historical_fusion_success_rate": 0.7,
                    },
                    "success": i % 2 == 0,  # Alternate success/failure
                    "improvement": 0.1 if i % 2 == 0 else 0.0,
                }
            )

        mock_fusion_model.train = MagicMock(
            return_value={"status": "success", "mae": 0.05, "rmse": 0.08}
        )

        result = optimizer.train_fusion_model()

        assert result["status"] == "success"
        assert mock_fusion_model.train.called
        assert mock_fusion_model.save.called

    def test_get_fusion_recommendations_single_modality(self, optimizer):
        """Test fusion recommendations with single modality"""
        recommendations = optimizer.get_fusion_recommendations(
            query_text="test query",
            detected_modalities=[(QueryModality.VIDEO, 0.9)],
            fusion_threshold=0.5,
        )

        assert recommendations["should_fuse"] is False
        assert recommendations["reason"] == "only_one_modality"

    def test_get_fusion_recommendations_high_benefit(
        self, optimizer, mock_fusion_model
    ):
        """Test fusion recommendations with high predicted benefit"""
        mock_fusion_model.predict_benefit = MagicMock(return_value=0.8)

        recommendations = optimizer.get_fusion_recommendations(
            query_text="test query",
            detected_modalities=[
                (QueryModality.VIDEO, 0.7),
                (QueryModality.DOCUMENT, 0.6),
            ],
            fusion_threshold=0.5,
        )

        assert recommendations["should_fuse"] is True
        assert recommendations["fusion_benefit"] == 0.8
        assert recommendations["primary_modality"] == "video"
        assert recommendations["secondary_modality"] == "document"

    def test_get_fusion_recommendations_low_benefit(self, optimizer, mock_fusion_model):
        """Test fusion recommendations with low predicted benefit"""
        mock_fusion_model.predict_benefit = MagicMock(return_value=0.3)

        recommendations = optimizer.get_fusion_recommendations(
            query_text="test query",
            detected_modalities=[
                (QueryModality.VIDEO, 0.9),
                (QueryModality.DOCUMENT, 0.4),
            ],
            fusion_threshold=0.5,
        )

        assert recommendations["should_fuse"] is False
        assert recommendations["reason"] == "insufficient_benefit"
        assert recommendations["fusion_benefit"] == 0.3

    def test_get_fusion_statistics_empty(self, optimizer):
        """Test getting statistics with no history"""
        stats = optimizer.get_fusion_statistics()

        assert stats["total_fusions"] == 0
        assert stats["overall_success_rate"] == 0.0
        assert stats["modality_pairs"] == {}

    def test_get_fusion_statistics_with_history(self, optimizer):
        """Test getting statistics with fusion history"""
        # Add some fusion records
        optimizer.fusion_history = [
            {
                "primary_modality": "video",
                "secondary_modality": "document",
                "success": True,
            },
            {
                "primary_modality": "video",
                "secondary_modality": "document",
                "success": True,
            },
            {
                "primary_modality": "video",
                "secondary_modality": "document",
                "success": False,
            },
            {
                "primary_modality": "image",
                "secondary_modality": "text",
                "success": True,
            },
        ]

        stats = optimizer.get_fusion_statistics()

        assert stats["total_fusions"] == 4
        assert stats["overall_success_rate"] == 0.75  # 3/4 success

        # Check video+document pair
        assert "video+document" in stats["modality_pairs"]
        video_doc = stats["modality_pairs"]["video+document"]
        assert video_doc["count"] == 3
        assert video_doc["success_count"] == 2
        assert video_doc["success_rate"] == 2 / 3

        # Check image+text pair
        assert "image+text" in stats["modality_pairs"]
        image_text = stats["modality_pairs"]["image+text"]
        assert image_text["count"] == 1
        assert image_text["success_rate"] == 1.0

    def test_get_top_fusion_pairs(self, optimizer):
        """Test getting top fusion pairs"""
        # Add fusion records
        optimizer.fusion_history = [
            {
                "primary_modality": "video",
                "secondary_modality": "document",
                "success": True,
            },
            {
                "primary_modality": "video",
                "secondary_modality": "document",
                "success": True,
            },
            {
                "primary_modality": "video",
                "secondary_modality": "document",
                "success": True,
            },
            {
                "primary_modality": "image",
                "secondary_modality": "text",
                "success": True,
            },
            {
                "primary_modality": "image",
                "secondary_modality": "text",
                "success": False,
            },
        ]

        top_pairs = optimizer.get_top_fusion_pairs(top_k=2)

        assert len(top_pairs) == 2
        # video+document should be first (100% success rate, 3 count)
        assert top_pairs[0]["pair"] == "video+document"
        assert top_pairs[0]["success_rate"] == 1.0
        assert top_pairs[0]["count"] == 3

    def test_clear_history(self, optimizer):
        """Test clearing fusion history"""
        # Add some data
        optimizer.fusion_history = [{"test": "data"}]
        optimizer.fusion_success_rates = {
            (QueryModality.VIDEO, QueryModality.DOCUMENT): 0.8
        }

        optimizer.clear_history()

        assert optimizer.fusion_history == []
        assert optimizer.fusion_success_rates == {}

    def test_export_fusion_data(self, optimizer, temp_model_dir):
        """Test exporting fusion data"""
        # Add some fusion history
        optimizer.fusion_history = [
            {
                "timestamp": (
                    optimizer.fusion_history[0]["timestamp"]
                    if optimizer.fusion_history
                    else pytest.importorskip("datetime").datetime.now()
                ),
                "primary_modality": "video",
                "secondary_modality": "document",
                "fusion_context": {},
                "success": True,
                "improvement": 0.1,
            }
        ]
        optimizer.fusion_success_rates = {
            (QueryModality.VIDEO, QueryModality.DOCUMENT): 0.85
        }

        export_path = temp_model_dir / "fusion_data.json"
        success = optimizer.export_fusion_data(export_path)

        assert success is True
        assert export_path.exists()

        # Verify exported data
        with open(export_path) as f:
            data = json.load(f)

        assert "fusion_history" in data
        assert "fusion_success_rates" in data
        assert "export_timestamp" in data
        assert "video+document" in data["fusion_success_rates"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
