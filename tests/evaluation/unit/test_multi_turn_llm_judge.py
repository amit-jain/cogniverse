"""
Unit tests for multi-turn LLM judge evaluation.

Tests validate:
1. Single-turn evaluation (backward compatibility)
2. Multi-turn evaluation with conversation history
3. Empty conversation handling
4. Conversation history formatting
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cogniverse_evaluation.evaluators.llm_judge import (
    SyncLLMReferenceFreeEvaluator,
    SyncLLMReferenceBasedEvaluator,
    SyncLLMHybridEvaluator,
)


class TestSingleTurnBackwardCompatibility:
    """Test that single-turn evaluation still works (backward compatibility)."""

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_single_turn_query_dict(self, mock_call_llm):
        """Test single-turn evaluation with query dict format."""
        mock_call_llm.return_value = "Score: 8/10\nGood visual match for the query."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        # Single-turn format
        input_data = {"query": "basketball dunk"}
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        result = evaluator.evaluate(input=input_data, output=output_data)

        assert result.score == 0.8
        assert "relevant" in result.label

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_single_turn_query_string(self, mock_call_llm):
        """Test single-turn evaluation with query string format."""
        mock_call_llm.return_value = "Score: 7/10\nPartially relevant."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        # String format
        result = evaluator.evaluate(
            input="basketball dunk",
            output={"results": [{"video_id": "v1", "score": 0.9}]},
        )

        assert result.score == 0.7


class TestMultiTurnEvaluation:
    """Test multi-turn evaluation with conversation history."""

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_multi_turn_conversation_format(self, mock_call_llm):
        """Test multi-turn evaluation with conversation list."""
        mock_call_llm.return_value = "Score: 9/10\nExcellent match considering context."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        # Multi-turn format
        input_data = {
            "conversation": [
                {"query": "show me basketball videos", "response": "Here are basketball videos..."},
                {"query": "any dunks?", "response": ""},  # Current turn
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.95}]}

        result = evaluator.evaluate(input=input_data, output=output_data)

        assert result.score == 0.9
        # Verify history was included in prompt
        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]
        assert "Previous Conversation" in prompt
        assert "show me basketball videos" in prompt
        assert "any dunks?" in prompt

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_multi_turn_three_turns(self, mock_call_llm):
        """Test multi-turn evaluation with three turns."""
        mock_call_llm.return_value = "Score: 8/10\nGood contextual match."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        input_data = {
            "conversation": [
                {"query": "find sports videos", "response": "Here are sports videos..."},
                {"query": "basketball specifically", "response": "Here are basketball videos..."},
                {"query": "show dunks from those", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        result = evaluator.evaluate(input=input_data, output=output_data)

        # Verify all history turns are in prompt
        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]
        assert "Turn 1:" in prompt
        assert "Turn 2:" in prompt
        assert "find sports videos" in prompt
        assert "basketball specifically" in prompt

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_multi_turn_single_turn_in_conversation(self, mock_call_llm):
        """Test conversation format with only one turn (no history)."""
        mock_call_llm.return_value = "Score: 7/10\nRelevant results."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        input_data = {
            "conversation": [
                {"query": "basketball dunk", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        result = evaluator.evaluate(input=input_data, output=output_data)

        # Should not have "Previous Conversation" since only one turn
        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]
        assert "Previous Conversation" not in prompt
        assert "basketball dunk" in prompt


class TestReferenceBasedMultiTurn:
    """Test reference-based evaluator with multi-turn."""

    @patch.object(SyncLLMReferenceBasedEvaluator, "_call_llm")
    @patch.object(SyncLLMReferenceBasedEvaluator, "_fetch_video_metadata")
    def test_reference_based_multi_turn(self, mock_fetch_metadata, mock_call_llm):
        """Test reference-based evaluation with conversation history."""
        mock_call_llm.return_value = "Score: 8/10\nGood precision and recall."
        mock_fetch_metadata.return_value = MagicMock(
            video_id="v1", title="Basketball Highlights", description=None, tags=None
        )

        evaluator = SyncLLMReferenceBasedEvaluator(model_name="test-model")
        evaluator.fetch_metadata = False  # Skip metadata fetching for test

        input_data = {
            "conversation": [
                {"query": "show me sports", "response": "Here are sports videos..."},
                {"query": "basketball dunks", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}
        expected = ["v1", "v2"]

        result = evaluator.evaluate(
            input=input_data, output=output_data, expected=expected
        )

        # Verify history in prompt
        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]
        assert "Previous Conversation" in prompt
        assert "show me sports" in prompt


class TestHybridMultiTurn:
    """Test hybrid evaluator with multi-turn."""

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    @patch.object(SyncLLMReferenceBasedEvaluator, "_call_llm")
    def test_hybrid_multi_turn(self, mock_ref_llm, mock_free_llm):
        """Test hybrid evaluation with conversation history."""
        mock_free_llm.return_value = "Score: 8/10\nVisually relevant."
        mock_ref_llm.return_value = "Score: 7/10\nGood recall."

        evaluator = SyncLLMHybridEvaluator(model_name="test-model")
        evaluator.reference_based.fetch_metadata = False

        input_data = {
            "conversation": [
                {"query": "sports highlights", "response": "Here are highlights..."},
                {"query": "dunks only", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}
        expected = ["v1"]

        result = evaluator.evaluate(
            input=input_data, output=output_data, expected=expected
        )

        # Both evaluators should have been called with conversation format
        assert mock_free_llm.called
        assert mock_ref_llm.called


class TestEdgeCases:
    """Test edge cases for multi-turn evaluation."""

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_empty_conversation(self, mock_call_llm):
        """Test with empty conversation list."""
        mock_call_llm.return_value = "Score: 5/10\nNo results."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        input_data = {"conversation": []}
        output_data = {"results": []}

        result = evaluator.evaluate(input=input_data, output=output_data)

        # Should handle gracefully
        assert result.label == "no_results"

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_conversation_with_empty_queries(self, mock_call_llm):
        """Test conversation with empty query strings."""
        mock_call_llm.return_value = "Score: 6/10\nPartially relevant."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        input_data = {
            "conversation": [
                {"query": "", "response": "Some response"},
                {"query": "actual query", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        result = evaluator.evaluate(input=input_data, output=output_data)

        # Should handle gracefully
        assert result.score is not None

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_mixed_format_fallback(self, mock_call_llm):
        """Test that query key takes precedence if both formats present."""
        mock_call_llm.return_value = "Score: 7/10\nRelevant."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        # Both query and conversation present - conversation should take precedence
        input_data = {
            "query": "ignored query",
            "conversation": [
                {"query": "used query", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        result = evaluator.evaluate(input=input_data, output=output_data)

        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]
        assert "used query" in prompt


class TestConversationHistoryFormatting:
    """Test the formatting of conversation history in prompts."""

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_history_format_structure(self, mock_call_llm):
        """Test that history is formatted with turn numbers."""
        mock_call_llm.return_value = "Score: 8/10\nGood match."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        input_data = {
            "conversation": [
                {"query": "first query", "response": "first response"},
                {"query": "second query", "response": "second response"},
                {"query": "current query", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        evaluator.evaluate(input=input_data, output=output_data)

        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]

        # Verify structure
        assert "### Previous Conversation:" in prompt
        assert "Turn 1:" in prompt
        assert "Turn 2:" in prompt
        assert "User: first query" in prompt
        assert "Assistant: first response" in prompt
        assert "User: second query" in prompt
        assert "Assistant: second response" in prompt
        assert 'Current Query: "current query"' in prompt

    @patch.object(SyncLLMReferenceFreeEvaluator, "_call_llm")
    def test_context_coherence_question_added(self, mock_call_llm):
        """Test that conversation coherence question is added for multi-turn."""
        mock_call_llm.return_value = "Score: 8/10\nGood match."

        evaluator = SyncLLMReferenceFreeEvaluator(model_name="test-model")

        # Multi-turn with history
        input_data = {
            "conversation": [
                {"query": "first query", "response": "first response"},
                {"query": "current query", "response": ""},
            ]
        }
        output_data = {"results": [{"video_id": "v1", "score": 0.9}]}

        evaluator.evaluate(input=input_data, output=output_data)

        call_args = mock_call_llm.call_args
        prompt = call_args[0][0]

        # Should include conversation history question
        assert "conversation history" in prompt.lower()
