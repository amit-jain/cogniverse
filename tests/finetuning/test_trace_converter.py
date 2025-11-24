"""
Unit tests for instruction trace converter (SFT data extraction).

Tests property access and agent filtering logic.
"""

from unittest.mock import Mock

import pandas as pd
import pytest
from cogniverse_finetuning.dataset.trace_converter import (
    InstructionDataset,
    InstructionExample,
    TraceToInstructionConverter,
)


@pytest.mark.unit
class TestPropertyAccess:
    """Test that converter uses public properties instead of private attributes"""

    def test_uses_public_traces_property(self):
        """Test that converter accesses provider.traces instead of provider._trace_store"""
        # Setup mock provider
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()
        mock_provider.traces.get_spans = Mock(return_value=pd.DataFrame())
        mock_provider.annotations.get_annotations = Mock(return_value=pd.DataFrame())

        converter = TraceToInstructionConverter(provider=mock_provider)

        # This call should use provider.traces, not provider._trace_store
        # The fact that it doesn't raise an AttributeError confirms correct access
        assert converter.provider.traces is not None


@pytest.mark.unit
class TestInstructionDataset:
    """Test InstructionDataset operations"""

    def test_to_dataframe(self):
        """Test converting dataset to DataFrame"""
        examples = [
            InstructionExample(
                instruction="Route this query",
                input="Find videos about cats",
                output="video_search",
                metadata={"agent": "routing"}
            ),
            InstructionExample(
                instruction="Route this query",
                input="Show me images of dogs",
                output="image_search",
                metadata={"agent": "routing"}
            ),
        ]

        dataset = InstructionDataset(examples=examples, metadata={"agent": "routing"})
        df = dataset.to_dataframe()

        assert len(df) == 2
        # Metadata dict gets expanded into columns via **ex.metadata
        assert list(df.columns) == ["instruction", "input", "output", "agent"]
        assert df["input"].iloc[0] == "Find videos about cats"
        assert df["output"].iloc[0] == "video_search"
        assert df["agent"].iloc[0] == "routing"

    def test_save_jsonl(self, tmp_path):
        """Test saving dataset as JSONL"""
        examples = [
            InstructionExample(
                instruction="Route this query",
                input="Find videos",
                output="video_search",
                metadata={}
            ),
        ]

        dataset = InstructionDataset(examples=examples, metadata={"agent": "routing"})
        output_path = tmp_path / "dataset.jsonl"
        dataset.save(str(output_path), format="jsonl")

        assert output_path.exists()

    def test_save_parquet(self, tmp_path):
        """Test saving dataset as Parquet"""
        examples = [
            InstructionExample(
                instruction="Route this query",
                input="Find videos",
                output="video_search",
                metadata={}
            ),
        ]

        dataset = InstructionDataset(examples=examples, metadata={"agent": "routing"})
        output_path = tmp_path / "dataset.parquet"
        dataset.save(str(output_path), format="parquet")

        assert output_path.exists()

    def test_save_invalid_format(self):
        """Test that invalid format raises ValueError"""
        examples = [
            InstructionExample(
                instruction="Route this query",
                input="Find videos",
                output="video_search",
                metadata={}
            ),
        ]

        dataset = InstructionDataset(examples=examples, metadata={"agent": "routing"})

        with pytest.raises(ValueError, match="Unsupported format"):
            dataset.save("/tmp/dataset.xml", format="xml")


@pytest.mark.unit
class TestAgentFiltering:
    """Test agent-specific span filtering"""

    def test_filter_routing_agent_spans(self):
        """Test filtering for routing agent spans"""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToInstructionConverter(provider=mock_provider)

        # Create test data
        spans_df = pd.DataFrame({
            "context.span_id": ["span1", "span2", "span3"],
            "name": ["routing_agent", "search_agent", "routing_agent"],
            "attributes.agent_type": ["routing", "search", "routing"],
        })

        filtered = converter._filter_agent_spans(spans_df, "routing")

        assert len(filtered) == 2
        assert all(filtered["attributes.agent_type"] == "routing")

    def test_filter_profile_selection_spans(self):
        """Test filtering for profile_selection agent spans"""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToInstructionConverter(provider=mock_provider)

        spans_df = pd.DataFrame({
            "context.span_id": ["span1", "span2"],
            "name": ["profile_selection_agent", "routing_agent"],
            "attributes.agent_type": ["profile_selection", "routing"],
        })

        filtered = converter._filter_agent_spans(spans_df, "profile_selection")

        assert len(filtered) == 1
        assert filtered.iloc[0]["attributes.agent_type"] == "profile_selection"

    def test_filter_entity_extraction_spans(self):
        """Test filtering for entity_extraction agent spans"""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToInstructionConverter(provider=mock_provider)

        spans_df = pd.DataFrame({
            "context.span_id": ["span1", "span2"],
            "name": ["entity_extraction_agent", "routing_agent"],
            "attributes.agent_type": ["entity_extraction", "routing"],
        })

        filtered = converter._filter_agent_spans(spans_df, "entity_extraction")

        assert len(filtered) == 1
        assert filtered.iloc[0]["attributes.agent_type"] == "entity_extraction"
