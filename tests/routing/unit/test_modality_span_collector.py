"""
Unit tests for ModalitySpanCollector
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cogniverse_agents.routing.modality_span_collector import ModalitySpanCollector
from cogniverse_agents.search.multi_modal_reranker import QueryModality


class TestModalitySpanCollector:
    """Test ModalitySpanCollector functionality"""

    @pytest.fixture
    def collector(self):
        """Create collector instance with mocked Phoenix client"""
        with patch("src.app.routing.modality_span_collector.px.Client"):
            collector = ModalitySpanCollector(tenant_id="test-tenant")
            return collector

    def test_initialization(self, collector):
        """Test collector initialization"""
        assert collector.tenant_id == "test-tenant"
        assert collector.project_name is not None

    def test_extract_modality_intent_nested_format(self, collector):
        """Test extracting modality intent from nested attributes"""
        span_row = pd.Series(
            {"attributes": {"query": {"modality_intent": ["video", "document"]}}}
        )

        intent = collector._extract_modality_intent(span_row)
        assert intent == ["video", "document"]

    def test_extract_modality_intent_dot_notation(self, collector):
        """Test extracting modality intent from dot notation"""
        span_row = pd.Series({"attributes": {"query.modality_intent": ["image"]}})

        intent = collector._extract_modality_intent(span_row)
        assert intent == ["image"]

    def test_extract_modality_intent_fallback(self, collector):
        """Test fallback to routing.detected_modalities"""
        span_row = pd.Series(
            {"attributes": {"routing": {"detected_modalities": ["audio"]}}}
        )

        intent = collector._extract_modality_intent(span_row)
        assert intent == ["audio"]

    def test_determine_primary_modality_video_priority(self, collector):
        """Test that video has highest priority"""
        modality_intent = ["text", "video", "document"]
        primary = collector._determine_primary_modality(modality_intent)
        assert primary == QueryModality.VIDEO

    def test_determine_primary_modality_image_second(self, collector):
        """Test that image has second priority"""
        modality_intent = ["text", "image", "document"]
        primary = collector._determine_primary_modality(modality_intent)
        assert primary == QueryModality.IMAGE

    def test_determine_primary_modality_single(self, collector):
        """Test single modality detection"""
        modality_intent = ["document"]
        primary = collector._determine_primary_modality(modality_intent)
        assert primary == QueryModality.DOCUMENT

    def test_determine_primary_modality_empty(self, collector):
        """Test empty modality list"""
        modality_intent = []
        primary = collector._determine_primary_modality(modality_intent)
        assert primary is None

    def test_extract_confidence(self, collector):
        """Test extracting routing confidence"""
        span_row = pd.Series({"attributes": {"routing": {"confidence": 0.85}}})

        confidence = collector._extract_confidence(span_row)
        assert confidence == 0.85

    def test_extract_confidence_missing(self, collector):
        """Test extracting confidence when missing"""
        span_row = pd.Series({"attributes": {}})
        confidence = collector._extract_confidence(span_row)
        assert confidence == 0.0

    def test_extract_success_ok_status(self, collector):
        """Test success extraction with OK status"""
        span_row = pd.Series({"status_code": "OK", "attributes": {}})

        success = collector._extract_success(span_row)
        assert success is True

    def test_extract_success_error_status(self, collector):
        """Test success extraction with error status"""
        span_row = pd.Series({"status_code": "ERROR", "attributes": {}})

        success = collector._extract_success(span_row)
        assert success is False

    def test_extract_success_with_error_attribute(self, collector):
        """Test success extraction when error in attributes"""
        span_row = pd.Series(
            {
                "status_code": "OK",
                "attributes": {"routing": {"error": "Something went wrong"}},
            }
        )

        success = collector._extract_success(span_row)
        assert success is False

    def test_span_row_to_dict(self, collector):
        """Test converting span row to dictionary"""
        span_row = pd.Series(
            {
                "context.span_id": "span-123",
                "start_time": datetime(2025, 10, 1, 12, 0, 0),
                "end_time": datetime(2025, 10, 1, 12, 0, 1),
                "status_code": "OK",
                "attributes": {"test": "data"},
                "name": "cogniverse.routing",
            }
        )

        span_dict = collector._span_row_to_dict(span_row)

        assert span_dict["span_id"] == "span-123"
        assert span_dict["status_code"] == "OK"
        assert span_dict["attributes"] == {"test": "data"}
        assert span_dict["name"] == "cogniverse.routing"

    @pytest.mark.asyncio
    async def test_collect_spans_by_modality_empty(self, collector):
        """Test collecting spans when none found"""
        # Mock Phoenix client to return empty dataframe
        collector.phoenix_client.get_spans_dataframe = MagicMock(
            return_value=pd.DataFrame()
        )

        result = await collector.collect_spans_by_modality(lookback_hours=1)

        assert result == {}

    @pytest.mark.asyncio
    async def test_collect_spans_by_modality_groups_correctly(self, collector):
        """Test that spans are grouped by modality correctly"""
        # Create mock spans
        spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-1",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 0),
                    "attributes": {
                        "query": {"modality_intent": ["video"]},
                        "routing": {"confidence": 0.9},
                    },
                },
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-2",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 1),
                    "attributes": {
                        "query": {"modality_intent": ["video"]},
                        "routing": {"confidence": 0.85},
                    },
                },
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-3",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 2),
                    "attributes": {
                        "query": {"modality_intent": ["document"]},
                        "routing": {"confidence": 0.88},
                    },
                },
            ]
        )

        collector.phoenix_client.get_spans_dataframe = MagicMock(return_value=spans_df)

        result = await collector.collect_spans_by_modality(lookback_hours=1)

        assert QueryModality.VIDEO in result
        assert QueryModality.DOCUMENT in result
        assert len(result[QueryModality.VIDEO]) == 2
        assert len(result[QueryModality.DOCUMENT]) == 1

    @pytest.mark.asyncio
    async def test_collect_spans_applies_confidence_filter(self, collector):
        """Test that min_confidence filter works"""
        spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-1",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 0),
                    "attributes": {
                        "query": {"modality_intent": ["video"]},
                        "routing": {"confidence": 0.9},  # Above threshold
                    },
                },
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-2",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 1),
                    "attributes": {
                        "query": {"modality_intent": ["video"]},
                        "routing": {"confidence": 0.5},  # Below threshold
                    },
                },
            ]
        )

        collector.phoenix_client.get_spans_dataframe = MagicMock(return_value=spans_df)

        result = await collector.collect_spans_by_modality(
            lookback_hours=1, min_confidence=0.7
        )

        assert QueryModality.VIDEO in result
        assert len(result[QueryModality.VIDEO]) == 1  # Only high confidence span

    @pytest.mark.asyncio
    async def test_collect_spans_handles_multi_modality(self, collector):
        """Test handling of queries with multiple modalities"""
        spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-1",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 0),
                    "attributes": {
                        "query": {
                            "modality_intent": ["video", "document"]
                        },  # Multi-modal
                        "routing": {"confidence": 0.9},
                    },
                },
            ]
        )

        collector.phoenix_client.get_spans_dataframe = MagicMock(return_value=spans_df)

        result = await collector.collect_spans_by_modality(lookback_hours=1)

        # Should be grouped under VIDEO (higher priority)
        assert QueryModality.VIDEO in result
        assert len(result[QueryModality.VIDEO]) == 1

    @pytest.mark.asyncio
    async def test_get_modality_statistics(self, collector):
        """Test getting modality statistics"""
        spans_df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-1",
                    "status_code": "OK",
                    "start_time": datetime(2025, 10, 1, 12, 0, 0),
                    "attributes": {
                        "query": {"modality_intent": ["video"]},
                        "routing": {"confidence": 0.9},
                    },
                },
                {
                    "name": "cogniverse.routing",
                    "context.span_id": "span-2",
                    "status_code": "ERROR",
                    "start_time": datetime(2025, 10, 1, 12, 0, 1),
                    "attributes": {
                        "query": {"modality_intent": ["video"]},
                        "routing": {"confidence": 0.5},
                    },
                },
            ]
        )

        collector.phoenix_client.get_spans_dataframe = MagicMock(return_value=spans_df)

        stats = await collector.get_modality_statistics(lookback_hours=1)

        assert stats["total_spans"] == 2
        assert "video" in stats["modality_distribution"]
        assert stats["modality_distribution"]["video"]["count"] == 2
        assert stats["modality_distribution"]["video"]["success_count"] == 1
        assert stats["modality_distribution"]["video"]["success_rate"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
