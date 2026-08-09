"""
Tests for trace converters (SFT data and trajectory extraction).

Tests:
1. Single-turn instruction extraction (TraceToInstructionConverter)
2. Multi-turn trajectory extraction (TraceToTrajectoryConverter)
"""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pandas as pd
import pytest

from cogniverse_finetuning.dataset.formatters import InstructionFormatter
from cogniverse_finetuning.dataset.preference_extractor import PreferencePairExtractor
from cogniverse_finetuning.dataset.trace_converter import (
    ConversationTrajectory,
    ConversationTurn,
    InstructionDataset,
    InstructionExample,
    TraceToInstructionConverter,
    TraceToTrajectoryConverter,
    TrajectoryDataset,
)
from cogniverse_finetuning.orchestrator import validate_sft_dataset
from cogniverse_telemetry_phoenix.provider import PhoenixProvider


@pytest.mark.unit
class TestPropertyAccess:
    """Test that converter uses public properties instead of private attributes"""

    @pytest.mark.asyncio
    async def test_convert_queries_via_public_traces_property(self):
        """convert() must fetch spans through provider.traces.get_all_spans (the
        public property), not a private _trace_store."""
        mock_provider = Mock()
        mock_provider.traces.get_all_spans = AsyncMock(return_value=pd.DataFrame())
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame()
        )

        converter = TraceToInstructionConverter(provider=mock_provider)

        # Empty spans → "insufficient annotated traces"; the point is that the
        # public traces property was queried on the way there.
        with pytest.raises(Exception):
            await converter.convert(
                project="cogniverse-t", agent_type="routing", min_annotations=1
            )

        mock_provider.traces.get_all_spans.assert_awaited_once()


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

    def test_filter_gateway_agent_spans(self):
        """Test filtering for routing agent spans"""
        mock_provider = Mock()
        mock_provider.traces = Mock()
        mock_provider.annotations = Mock()

        converter = TraceToInstructionConverter(provider=mock_provider)

        # Create test data
        spans_df = pd.DataFrame(
            {
                "context.span_id": ["span1", "span2", "span3"],
                "name": ["gateway_agent", "search_agent", "gateway_agent"],
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
                "name": ["profile_selection_agent", "gateway_agent"],
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
                "name": ["entity_extraction_agent", "gateway_agent"],
                "attributes.agent_type": ["entity_extraction", "routing"],
            }
        )

        filtered = converter._filter_agent_spans(spans_df, "entity_extraction")

        assert len(filtered) == 1
        assert filtered.iloc[0]["attributes.agent_type"] == "entity_extraction"

    def test_filter_approved_rejects_conflicting_label_and_score(self):
        converter = TraceToInstructionConverter(provider=Mock())
        annotations = pd.DataFrame(
            [
                {
                    "span_id": "rejected-high",
                    "result.label": "rejected",
                    "result.score": 1.0,
                },
                {
                    "span_id": "approved-low",
                    "result.label": "approved",
                    "result.score": 0.1,
                },
                {
                    "span_id": "approved-consistent",
                    "result.label": "approved",
                    "result.score": 1.0,
                },
                {
                    "span_id": "score-only",
                    "result.label": None,
                    "result.score": 0.95,
                },
                {
                    "span_id": "rejected-consistent",
                    "result.label": "rejected",
                    "result.score": 0.0,
                },
            ]
        )

        approved = converter._filter_approved(annotations)

        assert approved["span_id"].tolist() == [
            "approved-consistent",
            "score-only",
        ]


@pytest.mark.unit
class TestCanonicalInstructionOutput:
    def test_malformed_approved_span_raises_with_span_context(self):
        converter = TraceToInstructionConverter(provider=Mock())
        spans = pd.DataFrame(
            [
                {
                    "context.span_id": "span-malformed-routing",
                    "name": "cogniverse.routing",
                    "attributes.input.value": "find sunset videos",
                    "attributes.output.value": {"confidence": 0.98},
                }
            ]
        )
        annotations = pd.DataFrame(
            [{"span_id": "span-malformed-routing", "result.label": "approved"}]
        )

        with pytest.raises(
            ValueError,
            match=(
                "approved routing span span-malformed-routing requires exactly "
                "the recommended_agent field"
            ),
        ):
            converter._create_instruction_examples(spans, annotations, "routing")


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
            timestamp=datetime.now(timezone.utc),
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
                "name": ["gateway_agent", "search_agent", "gateway_agent"],
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

    def test_extract_response_projects_operational_output(self):
        """Operational response fields do not leak into model-facing JSON."""
        mock_provider = Mock()
        converter = TraceToTrajectoryConverter(provider=mock_provider)

        attributes = {
            "attributes.output.value": {
                "recommended_agent": "video_search",
                "confidence": 0.99,
                "reasoning": "The request asks for videos.",
            }
        }

        result = converter._extract_response(
            attributes,
            "routing",
            context="routing trajectory span span123 turn 1",
        )

        assert result == '{"recommended_agent":"video_search"}'

    def test_create_turn_from_span(self):
        """Test creating turn from span data."""
        mock_provider = Mock()
        converter = TraceToTrajectoryConverter(provider=mock_provider)

        span = pd.Series(
            {
                "context.span_id": "span123",
                "start_time": datetime(2025, 1, 1, 12, 0, 0),
                "attributes.input.value": "Test query",
                "attributes.output.value": {
                    "recommended_agent": "video_search",
                    "confidence": 0.99,
                },
            }
        )

        turn = converter._create_turn_from_span(span, turn_idx=1, agent_type="routing")

        assert turn is not None
        assert turn.turn_id == 1
        assert turn.query == "Test query"
        assert turn.response == '{"recommended_agent":"video_search"}'
        assert turn.span_id == "span123"

    def test_create_turn_rejects_malformed_persisted_timestamp(self):
        converter = TraceToTrajectoryConverter(provider=Mock())
        span = pd.Series(
            {
                "context.span_id": "span-invalid-time",
                "start_time": "not-a-timestamp",
                "attributes.input.value": "Test query",
                "attributes.output.value": {"recommended_agent": "video_search"},
            }
        )

        with pytest.raises(
            ValueError,
            match=(
                "routing trajectory span span-invalid-time turn 3 has invalid "
                "start_time 'not-a-timestamp'"
            ),
        ) as captured:
            converter._create_turn_from_span(
                span,
                turn_idx=3,
                agent_type="routing",
            )

        assert isinstance(captured.value.__cause__, ValueError)

        span["start_time"] = "2025-01-01T12:00:00"
        with pytest.raises(
            ValueError,
            match=(
                "routing trajectory span span-invalid-time turn 3 requires "
                "timezone-aware start_time"
            ),
        ):
            converter._create_turn_from_span(
                span,
                turn_idx=3,
                agent_type="routing",
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestTraceToTrajectoryConverterAsync:
    """Test async methods of TraceToTrajectoryConverter."""

    async def test_convert_empty_spans_raises_error(self):
        """Test that empty spans raises ValueError."""
        mock_provider = Mock()
        mock_traces = AsyncMock()
        mock_traces.get_all_spans = AsyncMock(return_value=pd.DataFrame())
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
        mock_traces.get_all_spans = AsyncMock(
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
        mock_traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                {
                    "context.span_id": ["span1"],
                    "name": ["gateway_agent"],
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

    async def test_convert_rejects_malformed_turn_instead_of_dropping_it(self):
        mock_provider = Mock()
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span-valid",
                        "name": "gateway_agent",
                        "attributes.session_id": "session1",
                        "start_time": datetime(2025, 1, 1, 12, 0, 0),
                        "attributes.input.value": "find sunset videos",
                        "attributes.output.value": {
                            "recommended_agent": "video_search"
                        },
                    },
                    {
                        "context.span_id": "span-malformed",
                        "name": "gateway_agent",
                        "attributes.session_id": "session1",
                        "start_time": datetime(2025, 1, 1, 12, 1, 0),
                        "attributes.input.value": "find launch documents",
                        "attributes.output.value": {"confidence": 0.99},
                    },
                ]
            )
        )
        mock_provider.annotations = Mock()
        converter = TraceToTrajectoryConverter(provider=mock_provider)

        with pytest.raises(
            ValueError,
            match=(
                "routing trajectory span span-malformed turn 2 requires exactly "
                "the recommended_agent field"
            ),
        ):
            await converter.convert(
                project="test-project",
                agent_type="routing",
                min_turns_per_session=2,
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
                    "gateway_agent",
                    "gateway_agent",
                    "gateway_agent",
                    "gateway_agent",
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
                "attributes.output.value": [
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                ],
            }
        )

        mock_traces.get_all_spans = AsyncMock(return_value=spans_df)
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
                    "gateway_agent",
                    "gateway_agent",
                    "gateway_agent",
                    "gateway_agent",
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
                "attributes.output.value": [
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                ],
            }
        )

        mock_traces.get_all_spans = AsyncMock(return_value=spans_df)
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
                "name": ["gateway_agent", "gateway_agent"],
                "attributes.session_id": ["session1", "session1"],
                "start_time": [
                    datetime(2025, 1, 1, 12, 0, 0),
                    datetime(2025, 1, 1, 12, 1, 0),
                ],
                "attributes.input.value": ["q1", "q2"],
                "attributes.output.value": [
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                ],
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

        mock_traces.get_all_spans = AsyncMock(return_value=spans_df)
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
                    "gateway_agent",
                    "gateway_agent",
                    "gateway_agent",
                    "gateway_agent",
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
                "attributes.output.value": [
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                    {"recommended_agent": "video_search"},
                    {"recommended_agent": "document_agent"},
                ],
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

        mock_traces.get_all_spans = AsyncMock(return_value=spans_df)
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


# ============================================================================
# Trajectory ChatML Formatter Tests
# ============================================================================


def _make_trajectory(
    session_id: str, turns_data: list[tuple[str, str]]
) -> ConversationTrajectory:
    """Helper to create a trajectory from (query, response) pairs."""
    turns = [
        ConversationTurn(
            turn_id=i + 1,
            query=q,
            response=r,
            timestamp=datetime(2025, 1, 1, 12, i, 0),
            span_id=f"span_{session_id}_{i}",
        )
        for i, (q, r) in enumerate(turns_data)
    ]
    return ConversationTrajectory(session_id=session_id, turns=turns)


@pytest.mark.unit
class TestTrajectoryFormatter:
    """Test ChatML formatting for multi-turn trajectories."""

    def test_single_trajectory_formatting(self):
        """Test formatting a single trajectory with two turns."""
        traj = _make_trajectory(
            "s1",
            [("What is RAG?", "RAG is..."), ("How does it work?", "It works by...")],
        )
        result = InstructionFormatter.format_trajectory_chatml([traj])

        assert len(result) == 1
        text = result[0]["text"]
        # System prompt present
        assert "<|im_start|>system\nYou are a helpful assistant.<|im_end|>" in text
        # Both turns present
        assert "<|im_start|>user\nWhat is RAG?<|im_end|>" in text
        assert "<|im_start|>assistant\nRAG is...<|im_end|>" in text
        assert "<|im_start|>user\nHow does it work?<|im_end|>" in text
        assert "<|im_start|>assistant\nIt works by...<|im_end|>" in text

    def test_custom_system_prompt(self):
        """Test that custom system prompt is used."""
        traj = _make_trajectory("s1", [("query", "response")])
        result = InstructionFormatter.format_trajectory_chatml(
            [traj], system_prompt="You are a video search assistant."
        )

        text = result[0]["text"]
        assert "You are a video search assistant." in text
        assert "You are a helpful assistant." not in text

    def test_multiple_trajectories(self):
        """Test formatting multiple trajectories produces one example each."""
        trajs = [
            _make_trajectory("s1", [("q1", "r1"), ("q2", "r2")]),
            _make_trajectory("s2", [("q3", "r3")]),
        ]
        result = InstructionFormatter.format_trajectory_chatml(trajs)

        assert len(result) == 2
        assert "q1" in result[0]["text"]
        assert "q3" in result[1]["text"]

    def test_empty_turns_raises_value_error(self):
        """Test that a trajectory with no turns raises ValueError."""
        traj = ConversationTrajectory(session_id="empty", turns=[])
        with pytest.raises(ValueError, match="has no turns"):
            InstructionFormatter.format_trajectory_chatml([traj])

    def test_empty_query_raises_value_error(self):
        """Test that a turn with empty query raises ValueError."""
        traj = _make_trajectory("s1", [("", "response")])
        with pytest.raises(ValueError, match="empty query"):
            InstructionFormatter.format_trajectory_chatml([traj])

    def test_empty_response_raises_value_error(self):
        """Test that a turn with empty response raises ValueError."""
        traj = _make_trajectory("s1", [("query", "")])
        with pytest.raises(ValueError, match="empty response"):
            InstructionFormatter.format_trajectory_chatml([traj])

    def test_output_passes_sft_validation(self):
        """Test that formatted output passes validate_sft_dataset."""
        traj = _make_trajectory("s1", [("q1", "r1"), ("q2", "r2")])
        result = InstructionFormatter.format_trajectory_chatml([traj])

        # Should not raise — all items have "text" field
        validate_sft_dataset(result)


def _phoenix_provider(http_endpoint: str, grpc_endpoint: str) -> PhoenixProvider:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": "finetuning-history-tests",
            "http_endpoint": http_endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return provider


def _routing_span(
    *,
    name: str,
    query: str,
    recommended_agent: str,
    start_time: datetime,
) -> tuple[dict, str]:
    span_id = uuid4().hex[:16]
    return (
        {
            "name": name,
            "context": {"trace_id": uuid4().hex, "span_id": span_id},
            "span_kind": "CHAIN",
            "start_time": start_time.isoformat(),
            "end_time": (start_time + timedelta(microseconds=1)).isoformat(),
            "status_code": "OK",
            "attributes": {
                "input.value": query,
                "output.value": json.dumps(
                    {"recommended_agent": recommended_agent}, separators=(",", ":")
                ),
            },
        },
        span_id,
    )


async def _log_spans(http_endpoint: str, project: str, spans: list[dict]) -> None:
    from phoenix.client import AsyncClient as PhoenixAsyncClient

    async with httpx.AsyncClient(base_url=http_endpoint, timeout=120) as http_client:
        client = PhoenixAsyncClient(
            base_url=http_endpoint,
            http_client=http_client,
        )
        result = await client.spans.log_spans(
            project_identifier=project,
            spans=spans,
            timeout=120,
        )
    assert result == {"total_received": len(spans), "total_queued": len(spans)}


async def _wait_for_span_count(
    provider: PhoenixProvider,
    project: str,
    expected_count: int,
) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        frame = await provider.traces.get_all_spans(project=project)
        if len(frame) == expected_count:
            return
        await asyncio.sleep(0.25)
    pytest.fail(f"Phoenix did not expose exactly {expected_count} spans in {project}")


async def _add_routing_reviews(
    provider: PhoenixProvider,
    project: str,
    span_id: str,
    *,
    chosen: str,
    rejected: str,
) -> None:
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="history_chosen",
        label="approved",
        score=1.0,
        metadata={
            "response": json.dumps({"recommended_agent": chosen}, separators=(",", ":"))
        },
        project=project,
    )
    await provider.annotations.add_annotation(
        span_id=span_id,
        name="history_rejected",
        label="rejected",
        score=0.0,
        metadata={
            "response": json.dumps(
                {"recommended_agent": rejected}, separators=(",", ":")
            )
        },
        project=project,
    )


async def _wait_for_training_records(
    provider: PhoenixProvider,
    project: str,
):
    deadline = time.monotonic() + 60
    last_error = None
    while time.monotonic() < deadline:
        try:
            instruction = await TraceToInstructionConverter(provider).convert(
                project=project,
                agent_type="routing",
                min_annotations=1,
            )
            preference = await PreferencePairExtractor(provider).extract(
                project=project,
                agent_type="routing",
                min_pairs=1,
            )
            return instruction, preference
        except ValueError as error:
            last_error = error
            await asyncio.sleep(0.25)
    pytest.fail(f"training records did not become visible in {project}: {last_error}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oldest_reviewed_span_survives_real_phoenix_pagination(
    phoenix_container,
):
    provider = _phoenix_provider(
        phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
    )
    project = f"finetuning-history-{uuid4().hex}"
    target_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    target, target_span_id = _routing_span(
        name="routing_agent.oldest_reviewed",
        query="find the oldest reviewed aurora video",
        recommended_agent="video_search",
        start_time=target_time,
    )
    noise = [
        _routing_span(
            name="search_noise",
            query=f"irrelevant newer request {index}",
            recommended_agent="document_agent",
            start_time=target_time + timedelta(seconds=1, microseconds=index),
        )[0]
        for index in range(10_001)
    ]
    await _log_spans(phoenix_container["http_endpoint"], project, [target, *noise])
    await _wait_for_span_count(provider, project, 10_002)
    await _add_routing_reviews(
        provider,
        project,
        target_span_id,
        chosen="video_search",
        rejected="document_agent",
    )

    instruction, preference = await _wait_for_training_records(provider, project)

    assert instruction.metadata["total_spans"] == 10_002
    assert [
        (example.input, example.output, example.metadata["span_id"])
        for example in instruction.examples
    ] == [
        (
            "find the oldest reviewed aurora video",
            '{"recommended_agent":"video_search"}',
            target_span_id,
        )
    ]
    assert preference.metadata["total_spans"] == 10_002
    assert [
        (pair.prompt, pair.chosen, pair.rejected, pair.metadata["span_id"])
        for pair in preference.pairs
    ] == [
        (
            "find the oldest reviewed aurora video",
            '{"recommended_agent":"video_search"}',
            '{"recommended_agent":"document_agent"}',
            target_span_id,
        )
    ]


async def _seed_reviewed_project(
    provider: PhoenixProvider,
    http_endpoint: str,
    project: str,
    *,
    query: str,
    chosen: str,
    rejected: str,
) -> str:
    target, span_id = _routing_span(
        name="routing_agent.tenant_reviewed",
        query=query,
        recommended_agent=chosen,
        start_time=datetime.now(timezone.utc),
    )
    await _log_spans(http_endpoint, project, [target])
    await _wait_for_span_count(provider, project, 1)
    await _add_routing_reviews(
        provider,
        project,
        span_id,
        chosen=chosen,
        rejected=rejected,
    )
    return span_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_real_phoenix_reads_keep_tenant_history_isolated(
    phoenix_container,
):
    provider = _phoenix_provider(
        phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
    )
    alpha_project = f"finetuning-alpha-{uuid4().hex}"
    beta_project = f"finetuning-beta-{uuid4().hex}"
    alpha_span_id, beta_span_id = await asyncio.gather(
        _seed_reviewed_project(
            provider,
            phoenix_container["http_endpoint"],
            alpha_project,
            query="alpha tenant exact video query",
            chosen="video_search",
            rejected="document_agent",
        ),
        _seed_reviewed_project(
            provider,
            phoenix_container["http_endpoint"],
            beta_project,
            query="beta tenant exact document query",
            chosen="document_agent",
            rejected="video_search",
        ),
    )

    (
        (alpha_instruction, alpha_preference),
        (
            beta_instruction,
            beta_preference,
        ),
    ) = await asyncio.gather(
        _wait_for_training_records(provider, alpha_project),
        _wait_for_training_records(provider, beta_project),
    )

    assert [
        (example.input, example.output, example.metadata["span_id"])
        for example in alpha_instruction.examples
    ] == [
        (
            "alpha tenant exact video query",
            '{"recommended_agent":"video_search"}',
            alpha_span_id,
        )
    ]
    assert [
        (pair.prompt, pair.chosen, pair.rejected, pair.metadata["span_id"])
        for pair in alpha_preference.pairs
    ] == [
        (
            "alpha tenant exact video query",
            '{"recommended_agent":"video_search"}',
            '{"recommended_agent":"document_agent"}',
            alpha_span_id,
        )
    ]
    assert [
        (example.input, example.output, example.metadata["span_id"])
        for example in beta_instruction.examples
    ] == [
        (
            "beta tenant exact document query",
            '{"recommended_agent":"document_agent"}',
            beta_span_id,
        )
    ]
    assert [
        (pair.prompt, pair.chosen, pair.rejected, pair.metadata["span_id"])
        for pair in beta_preference.pairs
    ] == [
        (
            "beta tenant exact document query",
            '{"recommended_agent":"document_agent"}',
            '{"recommended_agent":"video_search"}',
            beta_span_id,
        )
    ]


class _GatewayTimeoutHandler(BaseHTTPRequestHandler):
    def _send_timeout(self) -> None:
        body = b'{"detail":"upstream Phoenix timed out"}'
        self.send_response(504)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


setattr(_GatewayTimeoutHandler, "do_GET", _GatewayTimeoutHandler._send_timeout)
setattr(_GatewayTimeoutHandler, "do_POST", _GatewayTimeoutHandler._send_timeout)


@contextmanager
def _gateway_timeout_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayTimeoutHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gateway_timeout_is_not_reported_as_empty_training_history():
    project = "finetuning-timeout-tenant"
    with _gateway_timeout_endpoint() as endpoint:
        provider = _phoenix_provider(endpoint, "http://127.0.0.1:1")
        consumers = (
            TraceToInstructionConverter(provider).convert(
                project=project,
                agent_type="routing",
                min_annotations=1,
            ),
            PreferencePairExtractor(provider).extract(
                project=project,
                agent_type="routing",
                min_pairs=1,
            ),
        )
        results = await asyncio.gather(*consumers, return_exceptions=True)

    assert [type(result) for result in results] == [RuntimeError, RuntimeError]
    assert [str(result) for result in results] == [
        f"Failed to query every span from Phoenix project {project}",
        f"Failed to query every span from Phoenix project {project}",
    ]
    for result in results:
        cause = result.__cause__
        assert isinstance(cause, httpx.HTTPStatusError)
        assert cause.response.status_code == 504
        assert cause.response.json() == {"detail": "upstream Phoenix timed out"}
