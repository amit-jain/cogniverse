"""
Unit tests for preference pair extraction.

Tests deduplication logic and proper property access.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from cogniverse_finetuning.dataset.preference_extractor import (
    PreferenceDataset,
    PreferencePair,
    PreferencePairExtractor,
)


@pytest.mark.unit
class TestPreferencePairDeduplication:
    """Test preference pair deduplication logic"""

    @pytest.fixture
    def mock_provider(self):
        """Mock telemetry provider with proper public properties"""
        provider = Mock()
        provider.traces = Mock()
        provider.annotations = Mock()
        return provider

    @pytest.fixture
    def extractor(self, mock_provider):
        """Create extractor with mocked provider"""
        return PreferencePairExtractor(provider=mock_provider)

    def test_skip_identical_chosen_rejected(self, extractor):
        """Test that pairs with identical chosen/rejected are skipped"""
        # Mock spans DataFrame
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span1",
                    "name": "routing_agent",
                    "attributes.input.query": "test query",
                    "attributes.output.response": "same response",
                    "start_time": datetime.utcnow(),
                }
            ]
        )

        # Mock annotations with identical responses
        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": "span1",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata.response": "same response",  # Same!
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": "same response",  # Same!
                },
            ]
        )

        # Create pairs
        pairs = extractor._create_preference_pairs(spans_df, annotations_df, "routing")

        # Should be empty because chosen == rejected
        assert len(pairs) == 0

    def test_keep_different_chosen_rejected(self, extractor):
        """Test that pairs with different chosen/rejected are kept"""
        # Mock spans DataFrame
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span1",
                    "name": "routing_agent",
                    "attributes.input.query": "test query",
                    "attributes.output.response": "default response",
                    "start_time": datetime.utcnow(),
                }
            ]
        )

        # Mock annotations with different responses
        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": "span1",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata.response": "good response",  # Different
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": "bad response",  # Different
                },
            ]
        )

        # Create pairs
        pairs = extractor._create_preference_pairs(spans_df, annotations_df, "routing")

        # Should have 1 pair
        assert len(pairs) == 1
        assert pairs[0].chosen == "good response"
        assert pairs[0].rejected == "bad response"

    def test_multiple_pairs_some_identical(self, extractor):
        """Test filtering when some pairs are identical"""
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span1",
                    "name": "routing_agent",
                    "attributes.input.query": "query1",
                    "attributes.output.response": "default1",
                    "start_time": datetime.utcnow(),
                },
                {
                    "context.span_id": "span2",
                    "name": "routing_agent",
                    "attributes.input.query": "query2",
                    "attributes.output.response": "default2",
                    "start_time": datetime.utcnow(),
                },
            ]
        )

        annotations_df = pd.DataFrame(
            [
                # span1: identical (should be filtered)
                {
                    "span_id": "span1",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata.response": "same",
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": "same",
                },
                # span2: different (should be kept)
                {
                    "span_id": "span2",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata.response": "good",
                },
                {
                    "span_id": "span2",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": "bad",
                },
            ]
        )

        pairs = extractor._create_preference_pairs(spans_df, annotations_df, "routing")

        # Should have only 1 pair (span2)
        assert len(pairs) == 1
        assert pairs[0].prompt == "query2"


@pytest.mark.unit
class TestPropertyAccess:
    """Test that extractor uses public properties instead of private attributes"""

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

    @pytest.mark.asyncio
    async def test_uses_public_traces_property(self, mock_provider):
        """Test that extractor accesses .traces (not ._trace_store)"""
        # Setup mock responses
        mock_provider.traces.get_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "routing_agent",
                        "attributes.input.query": "test",
                        "attributes.output.response": "response",
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
                    {
                        "span_id": "span1",
                        "result.label": "rejected",
                        "result.score": 0.0,
                    },
                ]
            )
        )

        extractor = PreferencePairExtractor(provider=mock_provider)

        try:
            await extractor.extract(
                project="test-project",
                agent_type="routing",
                min_pairs=1,
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
class TestPreferenceDataset:
    """Test PreferenceDataset data structure"""

    def test_to_dataframe(self):
        """Test conversion to DataFrame"""
        pairs = [
            PreferencePair(
                prompt="Q1",
                chosen="Good",
                rejected="Bad",
                metadata={"span_id": "span1"},
            ),
            PreferencePair(
                prompt="Q2",
                chosen="Good2",
                rejected="Bad2",
                metadata={"span_id": "span2"},
            ),
        ]

        dataset = PreferenceDataset(pairs=pairs, metadata={"project": "test"})

        df = dataset.to_dataframe()

        assert len(df) == 2
        assert "prompt" in df.columns
        assert "chosen" in df.columns
        assert "rejected" in df.columns
        assert "span_id" in df.columns
        assert df.iloc[0]["prompt"] == "Q1"
        assert df.iloc[1]["chosen"] == "Good2"

    def test_save_jsonl(self, tmp_path):
        """Test saving dataset as JSONL"""
        pairs = [
            PreferencePair(prompt="Q1", chosen="Good", rejected="Bad", metadata={}),
        ]

        dataset = PreferenceDataset(pairs=pairs, metadata={})
        output_path = tmp_path / "test.jsonl"

        dataset.save(str(output_path), format="jsonl")

        assert output_path.exists()

    def test_save_parquet(self, tmp_path):
        """Test saving dataset as Parquet"""
        pairs = [
            PreferencePair(prompt="Q1", chosen="Good", rejected="Bad", metadata={}),
        ]

        dataset = PreferenceDataset(pairs=pairs, metadata={})
        output_path = tmp_path / "test.parquet"

        dataset.save(str(output_path), format="parquet")

        assert output_path.exists()

    def test_save_invalid_format(self, tmp_path):
        """Test that invalid format raises error"""
        pairs = [
            PreferencePair(prompt="Q1", chosen="Good", rejected="Bad", metadata={}),
        ]

        dataset = PreferenceDataset(pairs=pairs, metadata={})
        output_path = tmp_path / "test.invalid"

        with pytest.raises(ValueError, match="Unsupported format"):
            dataset.save(str(output_path), format="invalid")


@pytest.mark.unit
class TestExtractPromptAndResponse:
    """Test extraction of prompts and responses from span attributes"""

    @pytest.fixture
    def extractor(self):
        """Create extractor with dummy provider"""
        provider = Mock()
        provider.traces = Mock()
        provider.annotations = Mock()
        return PreferencePairExtractor(provider=provider)

    def test_extract_prompt_from_query(self, extractor):
        """Test extracting prompt from attributes.input.query"""
        span_row = pd.Series(
            {
                "attributes.input.query": "test query",
                "context.span_id": "span1",
            }
        )

        prompt = extractor._extract_prompt(span_row, "routing")

        assert prompt == "test query"

    def test_extract_prompt_from_text(self, extractor):
        """Test extracting prompt from attributes.input.text"""
        span_row = pd.Series(
            {
                "attributes.input.text": "test text",
                "context.span_id": "span1",
            }
        )

        prompt = extractor._extract_prompt(span_row, "routing")

        assert prompt == "test text"

    def test_extract_prompt_empty_when_missing(self, extractor):
        """Test that empty string is returned when no prompt found"""
        span_row = pd.Series(
            {
                "context.span_id": "span1",
            }
        )

        prompt = extractor._extract_prompt(span_row, "routing")

        assert prompt == ""

    def test_extract_response_from_annotation_metadata(self, extractor):
        """Test extracting response from annotation metadata"""
        annotation_row = pd.Series(
            {
                "span_id": "span1",
                "metadata.response": "test response",
            }
        )
        span_row = pd.Series(
            {
                "context.span_id": "span1",
            }
        )

        response = extractor._extract_response_from_annotation(annotation_row, span_row)

        assert response == "test response"

    def test_extract_response_from_span_output(self, extractor):
        """Test extracting response from span output when annotation has no response"""
        annotation_row = pd.Series(
            {
                "span_id": "span1",
            }
        )
        span_row = pd.Series(
            {
                "context.span_id": "span1",
                "attributes.output.response": "span response",
            }
        )

        response = extractor._extract_response_from_annotation(annotation_row, span_row)

        assert response == "span response"
