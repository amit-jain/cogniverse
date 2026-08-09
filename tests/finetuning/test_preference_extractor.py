"""
Unit tests for preference pair extraction.

Tests deduplication logic and proper property access.
"""

from datetime import datetime, timezone
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
                    "name": "gateway_agent",
                    "attributes.input.query": "test query",
                    "attributes.output.response": "same response",
                    "start_time": datetime.now(timezone.utc),
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
                    "metadata.response": '{"recommended_agent":"video_search"}',
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": '{"recommended_agent":"video_search"}',
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
                    "name": "gateway_agent",
                    "attributes.input.query": "test query",
                    "attributes.output.response": "default response",
                    "start_time": datetime.now(timezone.utc),
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
                    "metadata.response": '{"recommended_agent":"video_search"}',
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": '{"recommended_agent":"document_agent"}',
                },
            ]
        )

        # Create pairs
        pairs = extractor._create_preference_pairs(spans_df, annotations_df, "routing")

        # Should have 1 pair
        assert len(pairs) == 1
        assert pairs[0].chosen == '{"recommended_agent":"video_search"}'
        assert pairs[0].rejected == '{"recommended_agent":"document_agent"}'

    def test_multiple_pairs_some_identical(self, extractor):
        """Test filtering when some pairs are identical"""
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span1",
                    "name": "gateway_agent",
                    "attributes.input.query": "query1",
                    "attributes.output.response": "default1",
                    "start_time": datetime.now(timezone.utc),
                },
                {
                    "context.span_id": "span2",
                    "name": "gateway_agent",
                    "attributes.input.query": "query2",
                    "attributes.output.response": "default2",
                    "start_time": datetime.now(timezone.utc),
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
                    "metadata.response": '{"recommended_agent":"video_search"}',
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": '{"recommended_agent":"video_search"}',
                },
                # span2: different (should be kept)
                {
                    "span_id": "span2",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata.response": '{"recommended_agent":"video_search"}',
                },
                {
                    "span_id": "span2",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata.response": '{"recommended_agent":"document_agent"}',
                },
            ]
        )

        pairs = extractor._create_preference_pairs(spans_df, annotations_df, "routing")

        # Should have only 1 pair (span2)
        assert len(pairs) == 1
        assert pairs[0].prompt == "query2"

    def test_span_output_cannot_replace_a_missing_reviewed_response(self, extractor):
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span1",
                    "name": "gateway_agent",
                    "attributes.input.query": "find sunset videos",
                    "attributes.output.response": "default route",
                }
            ]
        )
        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": "span1",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata": {},
                },
                {
                    "span_id": "span1",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata": {"response": "document_agent"},
                },
            ]
        )

        with pytest.raises(
            ValueError,
            match=(
                "routing preference span span1 chosen response must be present "
                "in annotation metadata"
            ),
        ):
            extractor._create_preference_pairs(
                spans_df,
                annotations_df,
                "routing",
            )

    def test_operational_responses_are_projected_to_exact_training_json(
        self, extractor
    ):
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span-projection",
                    "name": "gateway_agent",
                    "attributes.input.value": "find sunset videos",
                }
            ]
        )
        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": "span-projection",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata": {
                        "response": {
                            "recommended_agent": "video_search",
                            "confidence": 0.99,
                            "reasoning": "The request asks for videos.",
                        }
                    },
                },
                {
                    "span_id": "span-projection",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata": {
                        "response": {
                            "recommended_agent": "document_agent",
                            "confidence": 0.31,
                            "reasoning": "This route ignores the requested medium.",
                        }
                    },
                },
            ]
        )

        pairs = extractor._create_preference_pairs(spans_df, annotations_df, "routing")

        assert len(pairs) == 1
        assert pairs[0].chosen == '{"recommended_agent":"video_search"}'
        assert pairs[0].rejected == '{"recommended_agent":"document_agent"}'

    @pytest.mark.parametrize(
        ("malformed_role", "approved_response", "rejected_response"),
        [
            (
                "chosen",
                {"confidence": 0.99},
                {"recommended_agent": "document_agent"},
            ),
            (
                "rejected",
                {"recommended_agent": "video_search"},
                {"confidence": 0.31},
            ),
        ],
    )
    def test_malformed_response_raises_with_span_and_role_context(
        self,
        extractor,
        malformed_role,
        approved_response,
        rejected_response,
    ):
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span-malformed",
                    "name": "gateway_agent",
                    "attributes.input.query": "find sunset videos",
                }
            ]
        )
        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": "span-malformed",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata": {"response": approved_response},
                },
                {
                    "span_id": "span-malformed",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata": {"response": rejected_response},
                },
            ]
        )

        with pytest.raises(
            ValueError,
            match=(
                rf"routing preference span span-malformed {malformed_role} response "
                r"requires exactly the recommended_agent field"
            ),
        ):
            extractor._create_preference_pairs(spans_df, annotations_df, "routing")


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
    async def test_extract_pairs_via_public_stores(self, mock_provider):
        """extract() drives the full span→filter→annotation→pair path through
        the provider's public .traces/.annotations stores and returns a
        PreferenceDataset whose pairs and metadata reflect the source data.
        """
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "routing_agent",
                        "attributes.input.query": "find sunset videos",
                        "attributes.output.response": "default route",
                        "start_time": datetime.now(timezone.utc),
                    }
                ]
            )
        )

        # Real Phoenix annotation frames are INDEXED by span_id (no span_id
        # column) with metadata as a single dict column — model that shape so
        # the extractor is exercised against the boundary's actual contract.
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "result.label": "approved",
                        "result.score": 1.0,
                        "metadata": {"response": {"recommended_agent": "video_search"}},
                    },
                    {
                        "result.label": "rejected",
                        "result.score": 0.0,
                        "metadata": {
                            "response": {"recommended_agent": "document_agent"}
                        },
                    },
                ],
                index=pd.Index(["span1", "span1"], name="span_id"),
            )
        )

        extractor = PreferencePairExtractor(provider=mock_provider)

        dataset = await extractor.extract(
            project="test-project",
            agent_type="routing",
            min_pairs=1,
        )

        # The public stores carried the call, with the documented query kwargs.
        mock_provider.traces.get_all_spans.assert_called_once_with(
            project="test-project",
            start_time=None,
            end_time=None,
        )
        assert (
            mock_provider.annotations.get_annotations.call_args.kwargs["project"]
            == "test-project"
        )
        mock_provider._trace_store.get_all_spans.assert_not_called()
        mock_provider._annotation_store.get_annotations.assert_not_called()

        # The returned dataset reflects the source span + its two annotations.
        assert len(dataset.pairs) == 1
        pair = dataset.pairs[0]
        assert pair.prompt == "find sunset videos"
        assert pair.chosen == '{"recommended_agent":"video_search"}'
        assert pair.rejected == '{"recommended_agent":"document_agent"}'
        assert pair.metadata["span_id"] == "span1"
        assert pair.metadata["agent_type"] == "routing"
        assert pair.metadata["chosen_score"] == 1.0
        assert pair.metadata["rejected_score"] == 0.0

        assert dataset.metadata["project"] == "test-project"
        assert dataset.metadata["agent_type"] == "routing"
        assert dataset.metadata["total_spans"] == 1
        assert dataset.metadata["agent_spans"] == 1
        assert dataset.metadata["total_annotations"] == 2
        assert dataset.metadata["preference_pairs"] == 1

    @pytest.mark.asyncio
    async def test_extract_raises_below_min_pairs(self, mock_provider):
        """When fewer pairs than min_pairs are found, extract() raises rather
        than returning a short dataset."""
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "routing_agent",
                        "attributes.input.query": "find sunset videos",
                        "attributes.output.response": "default route",
                        "start_time": datetime.now(timezone.utc),
                    }
                ]
            )
        )
        # Real Phoenix annotation frames are INDEXED by span_id (no span_id
        # column) with metadata as a single dict column — model that shape so
        # the extractor is exercised against the boundary's actual contract.
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "result.label": "approved",
                        "result.score": 1.0,
                        "metadata": {"response": {"recommended_agent": "video_search"}},
                    },
                    {
                        "result.label": "rejected",
                        "result.score": 0.0,
                        "metadata": {
                            "response": {"recommended_agent": "document_agent"}
                        },
                    },
                ],
                index=pd.Index(["span1", "span1"], name="span_id"),
            )
        )

        extractor = PreferencePairExtractor(provider=mock_provider)

        with pytest.raises(ValueError, match="Insufficient preference pairs"):
            await extractor.extract(
                project="test-project",
                agent_type="routing",
                min_pairs=5,
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

    def test_extract_prompt_from_nested_input_dict(self, extractor):
        """Phoenix groups input.query into a nested attributes.input dict."""
        span_row = pd.Series(
            {
                "attributes.input": {"query": "find sunset videos"},
                "context.span_id": "span1",
            }
        )

        assert extractor._extract_prompt(span_row, "routing") == "find sunset videos"

    def test_extract_response_from_nested_metadata_dict(self, extractor):
        """Phoenix returns annotation metadata as a nested dict column."""
        annotation_row = pd.Series(
            {
                "result.label": "approved",
                "metadata": {"response": "good route"},
            }
        )
        span_row = pd.Series({"context.span_id": "span1"})

        response = extractor._extract_response_from_annotation(annotation_row, span_row)

        assert response == "good route"

    def test_extract_response_from_nested_span_output_dict(self, extractor):
        """Span output falls back to the nested attributes.output dict."""
        annotation_row = pd.Series({"result.label": "approved"})
        span_row = pd.Series(
            {
                "context.span_id": "span1",
                "attributes.output": {"response": "span route"},
            }
        )

        response = extractor._extract_response_from_annotation(annotation_row, span_row)

        assert response == "span route"
