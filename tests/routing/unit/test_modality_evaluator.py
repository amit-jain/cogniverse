"""
Unit tests for ModalityEvaluator
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cogniverse_agents.routing.modality_evaluator import ModalityEvaluator
from cogniverse_agents.routing.modality_example import ModalityExample
from cogniverse_agents.search.multi_modal_reranker import QueryModality


class TestModalityEvaluator:
    """Test ModalityEvaluator functionality"""

    @pytest.fixture
    def mock_span_collector(self):
        """Create mocked span collector"""
        with patch("cogniverse_agents.routing.modality_evaluator.ModalitySpanCollector") as mock:
            collector = MagicMock()
            mock.return_value = collector
            yield collector

    @pytest.fixture
    def evaluator(self, mock_span_collector):
        """Create evaluator instance with mocked collector"""
        return ModalityEvaluator(
            span_collector=mock_span_collector, tenant_id="test-tenant"
        )

    def test_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.tenant_id == "test-tenant"
        assert evaluator.span_collector is not None

    def test_feature_definitions_exist(self, evaluator):
        """Test that feature definitions exist for all modalities"""
        assert len(evaluator.VIDEO_FEATURES) > 0
        assert len(evaluator.DOCUMENT_FEATURES) > 0
        assert len(evaluator.IMAGE_FEATURES) > 0
        assert len(evaluator.AUDIO_FEATURES) > 0

        # Check structure
        assert "keywords" in evaluator.VIDEO_FEATURES
        assert "keywords" in evaluator.DOCUMENT_FEATURES

    def test_extract_query_nested_format(self, evaluator):
        """Test query extraction from nested attributes"""
        attributes = {"query": {"text": "show me machine learning videos"}}

        query = evaluator._extract_query(attributes)
        assert query == "show me machine learning videos"

    def test_extract_query_dot_notation(self, evaluator):
        """Test query extraction from dot notation"""
        attributes = {"query.text": "research papers on neural networks"}

        query = evaluator._extract_query(attributes)
        assert query == "research papers on neural networks"

    def test_extract_query_input_format(self, evaluator):
        """Test query extraction from input attributes"""
        attributes = {"input": {"value": "diagram of CNN architecture"}}

        query = evaluator._extract_query(attributes)
        assert query == "diagram of CNN architecture"

    def test_extract_query_missing(self, evaluator):
        """Test query extraction when query is missing"""
        attributes = {}
        query = evaluator._extract_query(attributes)
        assert query is None

    def test_extract_agent_nested_format(self, evaluator):
        """Test agent extraction from nested attributes"""
        attributes = {"routing": {"selected_agent": "video_search_agent"}}

        agent = evaluator._extract_agent(attributes)
        assert agent == "video_search_agent"

    def test_extract_agent_dot_notation(self, evaluator):
        """Test agent extraction from dot notation"""
        attributes = {"routing.selected_agent": "document_agent"}

        agent = evaluator._extract_agent(attributes)
        assert agent == "document_agent"

    def test_extract_agent_output_format(self, evaluator):
        """Test agent extraction from output attributes"""
        attributes = {"output": {"agent": "image_search_agent"}}

        agent = evaluator._extract_agent(attributes)
        assert agent == "image_search_agent"

    def test_extract_modality_features_video(self, evaluator):
        """Test feature extraction for video queries"""
        query = "show me a tutorial on machine learning"
        attributes = {"routing": {"confidence": 0.9, "detected_modalities": ["video"]}}

        features = evaluator._extract_modality_features(
            query, QueryModality.VIDEO, attributes
        )

        assert "keywords_keywords" in features
        assert "has_keywords" in features
        assert features["has_keywords"] is True  # Contains "show" and "tutorial"
        assert features["routing_confidence"] == 0.9
        assert features["detected_modalities"] == ["video"]
        assert features["query_length"] == 7

    def test_extract_modality_features_document(self, evaluator):
        """Test feature extraction for document queries"""
        query = "read research papers on TensorFlow"
        attributes = {"routing": {"confidence": 0.85}}

        features = evaluator._extract_modality_features(
            query, QueryModality.DOCUMENT, attributes
        )

        assert features["has_keywords"] is True  # Contains "read" and "research"
        assert features["has_research"] is True  # Contains "research"
        assert features["has_specific_entity"] is True  # "TensorFlow" is capitalized
        assert features["query_length"] == 5

    def test_extract_modality_features_image(self, evaluator):
        """Test feature extraction for image queries"""
        query = "diagram of neural network architecture?"
        attributes = {}

        features = evaluator._extract_modality_features(
            query, QueryModality.IMAGE, attributes
        )

        assert features["has_keywords"] is True  # Contains "diagram"
        assert features["has_visual"] is True  # Contains "architecture"
        assert features["has_question"] is True  # Ends with "?"

    def test_extract_modality_features_audio(self, evaluator):
        """Test feature extraction for audio queries"""
        query = "listen to podcast about AI"
        attributes = {}

        features = evaluator._extract_modality_features(
            query, QueryModality.AUDIO, attributes
        )

        assert features["has_keywords"] is True  # Contains "listen" and "podcast"

    def test_has_capitalized_entity(self, evaluator):
        """Test capitalized entity detection"""
        # Has entity
        assert evaluator._has_capitalized_entity("learn about TensorFlow") is True
        assert evaluator._has_capitalized_entity("tutorial on PyTorch") is True

        # No entity (first word doesn't count)
        assert (
            evaluator._has_capitalized_entity("tutorial on machine learning") is False
        )
        assert evaluator._has_capitalized_entity("show me videos") is False
        # First word is capitalized but doesn't count as entity
        assert evaluator._has_capitalized_entity("PyTorch tutorial") is False

    def test_extract_result_count_from_output(self, evaluator):
        """Test result count extraction from output attributes"""
        attributes = {"output": {"result_count": 42}}

        count = evaluator._extract_result_count(attributes)
        assert count == 42

    def test_extract_result_count_from_results_array(self, evaluator):
        """Test result count extraction from results array"""
        attributes = {"results": [{"id": 1}, {"id": 2}, {"id": 3}]}

        count = evaluator._extract_result_count(attributes)
        assert count == 3

    def test_extract_result_count_missing(self, evaluator):
        """Test result count extraction when missing"""
        attributes = {}
        count = evaluator._extract_result_count(attributes)
        assert count == 0

    def test_span_to_example_success(self, evaluator):
        """Test converting span to example successfully"""
        span_data = {
            "span_id": "span-123",
            "status_code": "OK",
            "attributes": {
                "query": {"text": "show me machine learning videos"},
                "routing": {
                    "selected_agent": "video_search_agent",
                    "confidence": 0.9,
                    "detected_modalities": ["video"],
                },
                "output": {"result_count": 10},
            },
        }

        example = evaluator._span_to_example(span_data, QueryModality.VIDEO)

        assert example is not None
        assert example.query == "show me machine learning videos"
        assert example.modality == QueryModality.VIDEO
        assert example.correct_agent == "video_search_agent"
        assert example.success is True
        assert example.is_synthetic is False
        assert example.modality_features is not None
        assert example.modality_features["routing_confidence"] == 0.9
        assert example.modality_features["result_count"] == 10

    def test_span_to_example_failure(self, evaluator):
        """Test converting span with error status"""
        span_data = {
            "span_id": "span-456",
            "status_code": "ERROR",
            "attributes": {
                "query": {"text": "test query"},
                "routing": {"selected_agent": "video_search_agent"},
            },
        }

        example = evaluator._span_to_example(span_data, QueryModality.VIDEO)

        assert example is not None
        assert example.success is False

    def test_span_to_example_missing_query(self, evaluator):
        """Test converting span without query returns None"""
        span_data = {
            "span_id": "span-789",
            "status_code": "OK",
            "attributes": {"routing": {"selected_agent": "video_search_agent"}},
        }

        example = evaluator._span_to_example(span_data, QueryModality.VIDEO)
        assert example is None

    def test_span_to_example_missing_agent(self, evaluator):
        """Test converting span without agent returns None"""
        span_data = {
            "span_id": "span-789",
            "status_code": "OK",
            "attributes": {"query": {"text": "test query"}},
        }

        example = evaluator._span_to_example(span_data, QueryModality.VIDEO)
        assert example is None

    @pytest.mark.asyncio
    async def test_create_training_examples_basic(self, evaluator, mock_span_collector):
        """Test creating training examples from spans"""
        # Mock span collector response
        mock_spans = {
            QueryModality.VIDEO: [
                {
                    "span_id": "span-1",
                    "status_code": "OK",
                    "attributes": {
                        "query": {"text": "show me machine learning tutorial"},
                        "routing": {
                            "selected_agent": "video_search_agent",
                            "confidence": 0.9,
                        },
                    },
                },
                {
                    "span_id": "span-2",
                    "status_code": "OK",
                    "attributes": {
                        "query": {"text": "watch neural networks explained"},
                        "routing": {
                            "selected_agent": "video_search_agent",
                            "confidence": 0.85,
                        },
                    },
                },
            ],
            QueryModality.DOCUMENT: [
                {
                    "span_id": "span-3",
                    "status_code": "OK",
                    "attributes": {
                        "query": {"text": "research papers on deep learning"},
                        "routing": {
                            "selected_agent": "document_agent",
                            "confidence": 0.88,
                        },
                    },
                }
            ],
        }

        mock_span_collector.collect_spans_by_modality = AsyncMock(
            return_value=mock_spans
        )

        examples = await evaluator.create_training_examples(
            lookback_hours=24, min_confidence=0.7, augment_with_synthetic=False
        )

        assert QueryModality.VIDEO in examples
        assert QueryModality.DOCUMENT in examples
        assert len(examples[QueryModality.VIDEO]) == 2
        assert len(examples[QueryModality.DOCUMENT]) == 1

        # Check example properties
        video_example = examples[QueryModality.VIDEO][0]
        assert isinstance(video_example, ModalityExample)
        assert video_example.modality == QueryModality.VIDEO
        assert video_example.correct_agent == "video_search_agent"
        assert video_example.is_synthetic is False

    @pytest.mark.asyncio
    async def test_create_training_examples_empty(self, evaluator, mock_span_collector):
        """Test creating training examples when no spans found"""
        mock_span_collector.collect_spans_by_modality = AsyncMock(return_value={})

        examples = await evaluator.create_training_examples()

        assert examples == {}

    @pytest.mark.asyncio
    async def test_create_training_examples_with_synthetic(
        self, evaluator, mock_span_collector
    ):
        """Test creating training examples with synthetic augmentation"""
        mock_spans = {
            QueryModality.VIDEO: [
                {
                    "span_id": "span-1",
                    "status_code": "OK",
                    "attributes": {
                        "query": {"text": "show me tutorial"},
                        "routing": {
                            "selected_agent": "video_search_agent",
                            "confidence": 0.9,
                        },
                    },
                }
            ]
        }

        mock_span_collector.collect_spans_by_modality = AsyncMock(
            return_value=mock_spans
        )

        examples = await evaluator.create_training_examples(
            augment_with_synthetic=True, synthetic_ratio=0.5  # Add 50% synthetic
        )

        assert QueryModality.VIDEO in examples
        # Should have 1 real + 0 synthetic (0.5 * 1 = 0 when rounded)
        # Let's check it doesn't break
        assert len(examples[QueryModality.VIDEO]) >= 1

    def test_filter_by_quality_query_length(self, evaluator):
        """Test filtering by query length"""
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="short",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                ),
                ModalityExample(
                    query="this is a longer query",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                ),
            ]
        }

        filtered = evaluator.filter_by_quality(examples, min_query_length=3)

        assert len(filtered[QueryModality.VIDEO]) == 1
        assert filtered[QueryModality.VIDEO][0].query == "this is a longer query"

    def test_filter_by_quality_confidence(self, evaluator):
        """Test filtering by confidence"""
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="high confidence query",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={"routing_confidence": 0.9},
                ),
                ModalityExample(
                    query="low confidence query",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={"routing_confidence": 0.5},
                ),
            ]
        }

        filtered = evaluator.filter_by_quality(examples, min_confidence=0.7)

        assert len(filtered[QueryModality.VIDEO]) == 1
        assert filtered[QueryModality.VIDEO][0].query == "high confidence query"

    def test_filter_by_quality_success(self, evaluator):
        """Test filtering by success status"""
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="successful query test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                ),
                ModalityExample(
                    query="failed query test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=False,
                ),
            ]
        }

        filtered = evaluator.filter_by_quality(
            examples, min_query_length=3, require_success=True
        )

        assert len(filtered[QueryModality.VIDEO]) == 1
        assert filtered[QueryModality.VIDEO][0].query == "successful query test"

    def test_filter_by_quality_no_requirement(self, evaluator):
        """Test filtering with no success requirement"""
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="successful query test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                ),
                ModalityExample(
                    query="failed query test",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=False,
                ),
            ]
        }

        filtered = evaluator.filter_by_quality(
            examples, min_query_length=3, require_success=False
        )

        assert len(filtered[QueryModality.VIDEO]) == 2

    def test_get_feature_statistics(self, evaluator):
        """Test computing feature statistics"""
        examples = {
            QueryModality.VIDEO: [
                ModalityExample(
                    query="show me tutorial",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=True,
                    modality_features={
                        "query_length": 3,
                        "has_keywords": True,
                        "has_temporal": False,
                        "routing_confidence": 0.9,
                    },
                ),
                ModalityExample(
                    query="watch video",
                    modality=QueryModality.VIDEO,
                    correct_agent="video_search_agent",
                    success=False,
                    modality_features={
                        "query_length": 2,
                        "has_keywords": True,
                        "has_temporal": True,
                        "routing_confidence": 0.7,
                    },
                ),
            ]
        }

        stats = evaluator.get_feature_statistics(examples)

        assert "modality_counts" in stats
        assert stats["modality_counts"]["video"] == 2

        assert "success_rates" in stats
        assert stats["success_rates"]["video"] == 0.5  # 1 success / 2 total

        assert "feature_coverage" in stats
        assert "has_keywords" in stats["feature_coverage"]["video"]

        assert "query_characteristics" in stats
        assert stats["query_characteristics"]["video"]["avg_length"] == 2.5  # (3+2)/2
        assert stats["query_characteristics"]["video"]["min_length"] == 2
        assert stats["query_characteristics"]["video"]["max_length"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
