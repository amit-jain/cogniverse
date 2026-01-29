"""
Unit tests for training method selector.

Tests property access and method recommendation logic.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from cogniverse_finetuning.dataset.method_selector import (
    TrainingMethodSelector,
)


@pytest.mark.unit
class TestPropertyAccess:
    """Test that selector uses public properties instead of private attributes"""

    @pytest.fixture
    def mock_provider(self):
        """Mock provider with public properties"""
        provider = Mock()
        # Public properties (should be used)
        provider.traces = Mock()
        provider.annotations = Mock()
        # Private attributes (should NOT be used)
        provider._trace_store = Mock()
        provider._annotation_store = Mock()
        return provider

    @pytest.fixture
    def selector(self):
        """Create selector without services"""
        return TrainingMethodSelector()

    @pytest.mark.asyncio
    async def test_uses_public_traces_property(self, selector, mock_provider):
        """Test that selector accesses .traces (not ._trace_store)"""
        # Setup mock responses
        mock_provider.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "routing_agent",
                        "start_time": datetime.utcnow(),
                    }
                ]
            )
        )

        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "span_id": "span1",
                        "result.label": "approved",
                        "result.score": 1.0,
                    },
                ]
            )
        )

        try:
            await selector.analyze_data(
                provider=mock_provider,
                project="test-project",
                agent_type="routing",
            )
        except Exception:
            # May fail due to incomplete mocking, but we check the calls
            pass

        # Verify public properties were called
        mock_provider.traces.get_spans.assert_called()
        mock_provider.annotations.get_annotations.assert_called()

        # Verify private attributes were NOT called
        assert (
            not mock_provider._trace_store.get_spans.called
            if hasattr(mock_provider._trace_store, "get_spans")
            else True
        )
        assert (
            not mock_provider._annotation_store.get_annotations.called
            if hasattr(mock_provider._annotation_store, "get_annotations")
            else True
        )


@pytest.mark.unit
class TestMethodRecommendation:
    """Test method recommendation logic"""

    @pytest.fixture
    def selector(self):
        """Create selector"""
        return TrainingMethodSelector()

    def test_recommend_dpo_with_sufficient_pairs(self, selector):
        """Test DPO recommended when sufficient preference pairs"""
        method, confidence = selector._recommend_method(
            approved_count=100,
            preference_pairs=25,  # >= min_dpo_pairs (20)
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert method == "dpo"
        assert confidence > 0.0

    def test_recommend_sft_with_sufficient_approved(self, selector):
        """Test SFT recommended when sufficient approved examples"""
        method, confidence = selector._recommend_method(
            approved_count=60,  # >= min_sft_examples (50)
            preference_pairs=5,  # < min_dpo_pairs (20)
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert method == "sft"
        assert confidence > 0.0

    def test_recommend_insufficient_with_not_enough_data(self, selector):
        """Test insufficient recommended when not enough data"""
        method, confidence = selector._recommend_method(
            approved_count=30,  # < min_sft_examples (50)
            preference_pairs=5,  # < min_dpo_pairs (20)
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert method == "insufficient"
        assert confidence == 1.0

    def test_dpo_preferred_over_sft(self, selector):
        """Test DPO is preferred when both thresholds met"""
        method, confidence = selector._recommend_method(
            approved_count=100,  # >= min_sft_examples
            preference_pairs=25,  # >= min_dpo_pairs
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        # DPO should be preferred (more sample-efficient)
        assert method == "dpo"

    def test_confidence_increases_with_more_data(self, selector):
        """Test that confidence increases with more data"""
        # With just enough pairs
        _, conf_low = selector._recommend_method(
            approved_count=0,
            preference_pairs=20,  # exactly min_dpo_pairs
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        # With 2x the pairs
        _, conf_high = selector._recommend_method(
            approved_count=0,
            preference_pairs=40,  # 2x min_dpo_pairs
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert conf_high > conf_low

    def test_confidence_caps_at_1_0(self, selector):
        """Test that confidence caps at 1.0"""
        _, confidence = selector._recommend_method(
            approved_count=0,
            preference_pairs=1000,  # way more than needed
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert confidence == 1.0


@pytest.mark.unit
class TestDataAnalysis:
    """Test data analysis flow"""

    @pytest.fixture
    def selector(self):
        """Create selector"""
        return TrainingMethodSelector()

    @pytest.fixture
    def mock_provider(self):
        """Mock provider"""
        provider = Mock()
        provider.traces = Mock()
        provider.annotations = Mock()
        return provider

    @pytest.mark.asyncio
    async def test_analyze_with_no_spans(self, selector, mock_provider):
        """Test analysis when no spans found"""
        mock_provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame())

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
        )

        assert analysis.total_spans == 0
        assert analysis.approved_count == 0
        assert analysis.rejected_count == 0
        assert analysis.preference_pairs == 0
        assert analysis.needs_synthetic is True
        assert analysis.recommended_method == "insufficient"

    @pytest.mark.asyncio
    async def test_analyze_with_spans_no_annotations(self, selector, mock_provider):
        """Test analysis when spans exist but no annotations"""
        mock_provider.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "routing_agent",
                        "start_time": datetime.utcnow(),
                    }
                ]
            )
        )
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame()
        )

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
        )

        assert analysis.total_spans == 1
        assert analysis.approved_count == 0
        assert analysis.rejected_count == 0
        assert analysis.preference_pairs == 0
        assert analysis.needs_synthetic is True

    @pytest.mark.asyncio
    async def test_analyze_with_sufficient_preference_pairs(
        self, selector, mock_provider
    ):
        """Test analysis with sufficient preference pairs for DPO"""
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": f"span{i}",
                    "name": "routing_agent",
                    "start_time": datetime.utcnow(),
                }
                for i in range(25)
            ]
        )

        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": f"span{i}",
                    "result.label": "approved",
                    "result.score": 1.0,
                }
                for i in range(25)
            ]
            + [
                {
                    "span_id": f"span{i}",
                    "result.label": "rejected",
                    "result.score": 0.0,
                }
                for i in range(25)
            ]
        )

        mock_provider.traces.get_spans = AsyncMock(return_value=spans_df)
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=annotations_df
        )

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
            min_dpo_pairs=20,
        )

        assert analysis.total_spans == 25
        assert analysis.approved_count == 25
        assert analysis.rejected_count == 25
        assert analysis.preference_pairs == 25  # All spans have both
        assert analysis.needs_synthetic is False
        assert analysis.recommended_method == "dpo"


@pytest.mark.unit
class TestAgentFiltering:
    """Test filtering spans by agent type"""

    @pytest.fixture
    def selector(self):
        """Create selector"""
        return TrainingMethodSelector()

    def test_filter_routing_agent_spans(self, selector):
        """Test filtering for routing agent"""
        spans_df = pd.DataFrame(
            [
                {"name": "routing_agent", "context.span_id": "span1"},
                {"name": "video_search_agent", "context.span_id": "span2"},
                {"name": "router_decision", "context.span_id": "span3"},
            ]
        )

        filtered = selector._filter_agent_spans(spans_df, "routing")

        # Should match "routing" or "route" spans
        assert len(filtered) == 2
        assert "span1" in filtered["context.span_id"].values
        assert "span3" in filtered["context.span_id"].values

    def test_filter_profile_selection_spans(self, selector):
        """Test filtering for profile selection agent"""
        spans_df = pd.DataFrame(
            [
                {"name": "profile_selector", "context.span_id": "span1"},
                {"name": "routing_agent", "context.span_id": "span2"},
                {"name": "selection_decision", "context.span_id": "span3"},
            ]
        )

        filtered = selector._filter_agent_spans(spans_df, "profile_selection")

        # Should match "profile" or "selection" spans
        assert len(filtered) == 2
        assert "span1" in filtered["context.span_id"].values
        assert "span3" in filtered["context.span_id"].values

    def test_filter_entity_extraction_spans(self, selector):
        """Test filtering for entity extraction agent"""
        spans_df = pd.DataFrame(
            [
                {"name": "entity_extractor", "context.span_id": "span1"},
                {"name": "routing_agent", "context.span_id": "span2"},
                {"name": "extraction_task", "context.span_id": "span3"},
            ]
        )

        filtered = selector._filter_agent_spans(spans_df, "entity_extraction")

        # Should match "entity" or "extraction" spans
        assert len(filtered) == 2
        assert "span1" in filtered["context.span_id"].values
        assert "span3" in filtered["context.span_id"].values
