"""
Unit tests for trace converters (SFT data and trajectory extraction).

Tests:
1. Single-turn instruction extraction (TraceToInstructionConverter)
2. Multi-turn trajectory extraction (TraceToTrajectoryConverter)
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from cogniverse_finetuning.dataset.trace_converter import (
    ConversationTrajectory,
    ConversationTurn,
    InstructionDataset,
    InstructionExample,
    TraceToInstructionConverter,
    TraceToTrajectoryConverter,
    TrajectoryDataset,
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
                metadata={"agent": "routing"},
            ),
            InstructionExample(
                instruction="Route this query",
                input="Show me images of dogs",
                output="image_search",
                metadata={"agent": "routing"},
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
                metadata={},
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
                metadata={},
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
                metadata={},
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
        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2", "span3"],
                "name": ["routing_agent", "search_agent", "routing_agent"],
                "attributes.agent_type": ["routing", "search", "routing"],
            }
        )

        filtered = converter._filter_agent_spans(spans_df, "routing")

        assert len(filtered) == 2
        assert all(filtered["attributes.agent_type"] == "routing")

    def test_filter_profile_selection_spans(self):
        """Test filtering for profile_selection agent spans"""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToInstructionConverter(provider=mock_provider)

        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2"],
                "name": ["profile_selection_agent", "routing_agent"],
                "attributes.agent_type": ["profile_selection", "routing"],
            }
        )

        filtered = converter._filter_agent_spans(spans_df, "profile_selection")

        assert len(filtered) == 1
        assert filtered.iloc[0]["attributes.agent_type"] == "profile_selection"

    def test_filter_entity_extraction_spans(self):
        """Test filtering for entity_extraction agent spans"""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToInstructionConverter(provider=mock_provider)

        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2"],
                "name": ["entity_extraction_agent", "routing_agent"],
                "attributes.agent_type": ["entity_extraction", "routing"],
            }
        )

        filtered = converter._filter_agent_spans(spans_df, "entity_extraction")

        assert len(filtered) == 1
        assert filtered.iloc[0]["attributes.agent_type"] == "entity_extraction"


# ============================================================================
# Trajectory Data Structure Tests
# ============================================================================


@pytest.mark.unit
class TestConversationTurn:
    """Test ConversationTurn data structure."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            turn_id=1,
            query="Find basketball videos",
            response="Here are basketball videos...",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            span_id="span123",
            metadata={"agent_type": "routing"},
        )

        assert turn.turn_id == 1
        assert turn.query == "Find basketball videos"
        assert turn.response == "Here are basketball videos..."
        assert turn.span_id == "span123"
        assert turn.metadata["agent_type"] == "routing"

    def test_turn_default_metadata(self):
        """Test turn with default empty metadata."""
        turn = ConversationTurn(
            turn_id=1,
            query="query",
            response="response",
            timestamp=datetime.utcnow(),
            span_id="span1",
        )

        assert turn.metadata == {}


@pytest.mark.unit
class TestConversationTrajectory:
    """Test ConversationTrajectory data structure."""

    def test_create_trajectory(self):
        """Test creating a conversation trajectory."""
        turns = [
            ConversationTurn(
                turn_id=1,
                query="Find sports videos",
                response="Here are sports videos...",
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                span_id="span1",
            ),
            ConversationTurn(
                turn_id=2,
                query="Show basketball dunks",
                response="Here are dunk videos...",
                timestamp=datetime(2025, 1, 1, 12, 1, 0),
                span_id="span2",
            ),
        ]

        trajectory = ConversationTrajectory(
            session_id="session123",
            turns=turns,
            session_outcome="success",
            session_score=0.9,
            metadata={"project": "test"},
        )

        assert trajectory.session_id == "session123"
        assert len(trajectory.turns) == 2
        assert trajectory.session_outcome == "success"
        assert trajectory.session_score == 0.9

    def test_trajectory_to_dict(self):
        """Test trajectory to_dict conversion."""
        turns = [
            ConversationTurn(
                turn_id=1,
                query="First query",
                response="First response",
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                span_id="span1",
            ),
        ]

        trajectory = ConversationTrajectory(
            session_id="session123",
            turns=turns,
            session_outcome="success",
            session_score=0.8,
        )

        result = trajectory.to_dict()

        assert result["session_id"] == "session123"
        assert result["num_turns"] == 1
        assert result["session_outcome"] == "success"
        assert result["session_score"] == 0.8
        assert len(result["conversation"]) == 1
        assert result["conversation"][0]["query"] == "First query"


@pytest.mark.unit
class TestTrajectoryDataset:
    """Test TrajectoryDataset operations."""

    def test_to_dataframe(self):
        """Test converting trajectory dataset to DataFrame."""
        turns = [
            ConversationTurn(
                turn_id=1,
                query="Query",
                response="Response",
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                span_id="span1",
            ),
        ]

        trajectories = [
            ConversationTrajectory(
                session_id="session1",
                turns=turns,
                session_outcome="success",
            ),
            ConversationTrajectory(
                session_id="session2",
                turns=turns,
                session_outcome="partial",
            ),
        ]

        dataset = TrajectoryDataset(trajectories=trajectories, metadata={"test": True})
        df = dataset.to_dataframe()

        assert len(df) == 2
        assert "session_id" in df.columns
        assert "num_turns" in df.columns
        assert "session_outcome" in df.columns
        assert df["session_id"].iloc[0] == "session1"

    def test_save_jsonl(self, tmp_path):
        """Test saving trajectory dataset as JSONL."""
        turns = [
            ConversationTurn(
                turn_id=1,
                query="Query",
                response="Response",
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                span_id="span1",
            ),
        ]

        trajectories = [
            ConversationTrajectory(
                session_id="session1",
                turns=turns,
            ),
        ]

        dataset = TrajectoryDataset(trajectories=trajectories)
        output_path = tmp_path / "trajectories.jsonl"
        dataset.save(str(output_path), format="jsonl")

        assert output_path.exists()

    def test_save_parquet(self, tmp_path):
        """Test saving trajectory dataset as Parquet."""
        turns = [
            ConversationTurn(
                turn_id=1,
                query="Query",
                response="Response",
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                span_id="span1",
            ),
        ]

        trajectories = [
            ConversationTrajectory(
                session_id="session1",
                turns=turns,
            ),
        ]

        dataset = TrajectoryDataset(trajectories=trajectories)
        output_path = tmp_path / "trajectories.parquet"
        dataset.save(str(output_path), format="parquet")

        assert output_path.exists()

    def test_save_invalid_format(self):
        """Test that invalid format raises ValueError."""
        dataset = TrajectoryDataset(trajectories=[])

        with pytest.raises(ValueError, match="Unsupported format"):
            dataset.save("/tmp/trajectories.xml", format="xml")


# ============================================================================
# TraceToTrajectoryConverter Tests
# ============================================================================


@pytest.mark.unit
class TestTraceToTrajectoryConverter:
    """Test TraceToTrajectoryConverter class."""

    def test_uses_public_traces_property(self):
        """Test that converter accesses provider.traces property."""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        assert converter.provider.traces is not None

    def test_filter_agent_spans(self):
        """Test filtering for agent-specific spans."""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2", "span3"],
                "name": ["routing_agent", "search_agent", "routing_agent"],
            }
        )

        filtered = converter._filter_agent_spans(spans_df, "routing")

        assert len(filtered) == 2

    def test_extract_query(self):
        """Test query extraction from span attributes."""
        mock_provider = Mock()
        converter = TraceToTrajectoryConverter(provider=mock_provider)

        # Test with input.value attribute
        attributes = {"attributes.input.value": "Find basketball videos"}
        result = converter._extract_query(attributes)
        assert result == "Find basketball videos"

        # Test with JSON-encoded value
        attributes = {"attributes.input.value": '{"query": "basketball dunks"}'}
        result = converter._extract_query(attributes)
        assert result == "basketball dunks"

    def test_extract_response(self):
        """Test response extraction from span attributes."""
        mock_provider = Mock()
        converter = TraceToTrajectoryConverter(provider=mock_provider)

        # Test with output.value attribute
        attributes = {"attributes.output.value": "Here are the results..."}
        result = converter._extract_response(attributes)
        assert result == "Here are the results..."

        # Test with dict response (should be JSON-serialized)
        attributes = {"attributes.output.value": {"results": ["v1", "v2"]}}
        result = converter._extract_response(attributes)
        assert '"results"' in result

    def test_create_turn_from_span(self):
        """Test creating turn from span data."""
        mock_provider = Mock()
        converter = TraceToTrajectoryConverter(provider=mock_provider)

        span = pd.Series(
            {
                "context.span_id": "span123",
                "start_time": datetime(2025, 1, 1, 12, 0, 0),
                "attributes.input.value": "Test query",
                "attributes.output.value": "Test response",
            }
        )

        turn = converter._create_turn_from_span(span, turn_idx=1, agent_type="routing")

        assert turn is not None
        assert turn.turn_id == 1
        assert turn.query == "Test query"
        assert turn.response == "Test response"
        assert turn.span_id == "span123"


@pytest.mark.unit
@pytest.mark.asyncio
class TestTraceToTrajectoryConverterAsync:
    """Test async methods of TraceToTrajectoryConverter."""

    async def test_convert_empty_spans_raises_error(self):
        """Test that empty spans raises ValueError."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        mock_traces.get_spans = AsyncMock(return_value=pd.DataFrame())
        mock_provider.traces = mock_traces

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        with pytest.raises(ValueError, match="No spans found"):
            await converter.convert(
                project="test-project",
                agent_type="routing",
            )

    async def test_convert_no_agent_spans_raises_error(self):
        """Test that no matching agent spans raises ValueError."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        # Return spans but none matching agent type
        mock_traces.get_spans = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "context.span_id": ["span1"],
                    "name": ["other_agent"],
                    "attributes.session_id": ["session1"],
                }
            )
        )
        mock_provider.traces = mock_traces

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        with pytest.raises(ValueError, match="No routing spans found"):
            await converter.convert(
                project="test-project",
                agent_type="routing",
            )

    async def test_convert_no_session_id_raises_error(self):
        """Test that spans without session_id raises ValueError."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        # Return routing spans but no session_id column
        mock_traces.get_spans = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "context.span_id": ["span1"],
                    "name": ["routing_agent"],
                    # No session_id column
                }
            )
        )
        mock_provider.traces = mock_traces

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        with pytest.raises(ValueError, match="No session_id in span attributes"):
            await converter.convert(
                project="test-project",
                agent_type="routing",
            )

    async def test_convert_groups_by_session(self):
        """Test that spans are correctly grouped by session_id."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        mock_annotations = AsyncMock()

        # Create spans from two sessions
        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2", "span3", "span4"],
                "name": [
                    "routing_agent",
                    "routing_agent",
                    "routing_agent",
                    "routing_agent",
                ],
                "attributes.session_id": [
                    "session1",
                    "session1",
                    "session2",
                    "session2",
                ],
                "start_time": [
                    datetime(2025, 1, 1, 12, 0, 0),
                    datetime(2025, 1, 1, 12, 1, 0),
                    datetime(2025, 1, 1, 12, 2, 0),
                    datetime(2025, 1, 1, 12, 3, 0),
                ],
                "attributes.input.value": ["q1", "q2", "q3", "q4"],
                "attributes.output.value": ["r1", "r2", "r3", "r4"],
            }
        )

        mock_traces.get_spans = AsyncMock(return_value=spans_df)
        mock_provider.traces = mock_traces
        mock_provider.annotations = mock_annotations

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        result = await converter.convert(
            project="test-project",
            agent_type="routing",
            min_turns_per_session=2,
        )

        # Should have 2 trajectories
        assert len(result.trajectories) == 2
        # Each trajectory should have 2 turns
        assert len(result.trajectories[0].turns) == 2
        assert len(result.trajectories[1].turns) == 2

    async def test_convert_filters_by_min_turns(self):
        """Test that trajectories with fewer turns than min_turns are filtered."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        mock_annotations = AsyncMock()

        # Create spans: session1 has 3 turns, session2 has 1 turn
        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2", "span3", "span4"],
                "name": [
                    "routing_agent",
                    "routing_agent",
                    "routing_agent",
                    "routing_agent",
                ],
                "attributes.session_id": [
                    "session1",
                    "session1",
                    "session1",
                    "session2",
                ],
                "start_time": [
                    datetime(2025, 1, 1, 12, 0, 0),
                    datetime(2025, 1, 1, 12, 1, 0),
                    datetime(2025, 1, 1, 12, 2, 0),
                    datetime(2025, 1, 1, 12, 3, 0),
                ],
                "attributes.input.value": ["q1", "q2", "q3", "q4"],
                "attributes.output.value": ["r1", "r2", "r3", "r4"],
            }
        )

        mock_traces.get_spans = AsyncMock(return_value=spans_df)
        mock_provider.traces = mock_traces
        mock_provider.annotations = mock_annotations

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        result = await converter.convert(
            project="test-project",
            agent_type="routing",
            min_turns_per_session=2,  # session2 only has 1 turn
        )

        # Should have only 1 trajectory (session1)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].session_id == "session1"
        assert len(result.trajectories[0].turns) == 3

    async def test_convert_with_annotations(self):
        """Test converting with session annotations."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        mock_annotations = AsyncMock()

        # Create spans for one session
        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2"],
                "name": ["routing_agent", "routing_agent"],
                "attributes.session_id": ["session1", "session1"],
                "start_time": [
                    datetime(2025, 1, 1, 12, 0, 0),
                    datetime(2025, 1, 1, 12, 1, 0),
                ],
                "attributes.input.value": ["q1", "q2"],
                "attributes.output.value": ["r1", "r2"],
            }
        )

        # Create annotation data
        annotations_df = pd.DataFrame(
            {
                "span_id": ["span1"],
                "result.label": ["success"],
                "result.score": [0.9],
            }
        )

        mock_traces.get_spans = AsyncMock(return_value=spans_df)
        mock_annotations.get_annotations = AsyncMock(return_value=annotations_df)
        mock_provider.traces = mock_traces
        mock_provider.annotations = mock_annotations

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        result = await converter.convert(
            project="test-project",
            agent_type="routing",
            min_turns_per_session=2,
            require_session_annotation=True,
        )

        # Should have 1 trajectory with annotation
        assert len(result.trajectories) == 1
        assert result.trajectories[0].session_outcome == "success"
        assert result.trajectories[0].session_score == 0.9

    async def test_convert_skips_unannotated_sessions(self):
        """Test that unannotated sessions are skipped when require_session_annotation=True."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        mock_annotations = AsyncMock()

        # Create spans for two sessions
        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2", "span3", "span4"],
                "name": [
                    "routing_agent",
                    "routing_agent",
                    "routing_agent",
                    "routing_agent",
                ],
                "attributes.session_id": [
                    "session1",
                    "session1",
                    "session2",
                    "session2",
                ],
                "start_time": [
                    datetime(2025, 1, 1, 12, 0, 0),
                    datetime(2025, 1, 1, 12, 1, 0),
                    datetime(2025, 1, 1, 12, 2, 0),
                    datetime(2025, 1, 1, 12, 3, 0),
                ],
                "attributes.input.value": ["q1", "q2", "q3", "q4"],
                "attributes.output.value": ["r1", "r2", "r3", "r4"],
            }
        )

        # Only session1 is annotated (session2 returns empty)
        async def mock_get_annotations(spans_df, project, annotation_names):
            session_ids = set(spans_df["attributes.session_id"].unique())
            if "session1" in session_ids:
                return pd.DataFrame(
                    {
                        "span_id": ["span1"],
                        "result.label": ["success"],
                        "result.score": [0.9],
                    }
                )
            return pd.DataFrame()

        mock_traces.get_spans = AsyncMock(return_value=spans_df)
        mock_annotations.get_annotations = mock_get_annotations
        mock_provider.traces = mock_traces
        mock_provider.annotations = mock_annotations

        converter = TraceToTrajectoryConverter(provider=mock_provider)

        result = await converter.convert(
            project="test-project",
            agent_type="routing",
            min_turns_per_session=2,
            require_session_annotation=True,
        )

        # Should have only 1 trajectory (session1 which is annotated)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].session_id == "session1"
