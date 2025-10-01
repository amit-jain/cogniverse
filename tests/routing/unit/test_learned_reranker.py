"""
Unit tests for LiteLLM-based learned reranker:
- Configuration loading
- LiteLLM integration
- Error handling
- SearchResult processing
"""

from unittest.mock import Mock, patch

import pytest

from src.app.search.learned_reranker import LearnedReranker
from src.app.search.multi_modal_reranker import SearchResult


class TestLearnedReranker:
    """Test LiteLLM-based learned reranker"""

    @pytest.fixture
    def sample_results(self):
        """Create sample search results"""
        return [
            SearchResult(
                id="doc-1",
                title="Machine Learning",
                content="Introduction to ML",
                modality="text",
                score=0.8,
                metadata={},
            ),
            SearchResult(
                id="doc-2",
                title="Deep Learning",
                content="Neural networks guide",
                modality="text",
                score=0.7,
                metadata={},
            ),
        ]

    def test_reranker_initialization_from_config(self):
        """Test reranker loads model from config"""
        with patch("src.app.search.learned_reranker.get_config_value") as mock_config:
            mock_config.return_value = {
                "model": "cohere",
                "supported_models": {"cohere": "cohere/rerank-english-v3.0"},
                "top_n": 10,
                "max_results_to_rerank": 100,
            }

            reranker = LearnedReranker()
            assert reranker.model == "cohere/rerank-english-v3.0"
            assert reranker.default_top_n == 10
            assert reranker.max_results_to_rerank == 100

    def test_reranker_initialization_with_explicit_model(self):
        """Test reranker with explicit model string"""
        reranker = LearnedReranker(model="jina/jina-reranker-v2-base")
        assert reranker.model == "jina/jina-reranker-v2-base"

    def test_reranker_raises_on_heuristic_model(self):
        """Test reranker raises error if model is heuristic"""
        with patch("src.app.search.learned_reranker.get_config_value") as mock_config:
            mock_config.return_value = {
                "model": "heuristic",
                "supported_models": {},
            }

            with pytest.raises(ValueError, match="requires a learned model"):
                LearnedReranker()

    def test_reranker_raises_on_unknown_model(self):
        """Test reranker raises error for unknown model key"""
        with patch("src.app.search.learned_reranker.get_config_value") as mock_config:
            mock_config.return_value = {
                "model": "unknown_model",
                "supported_models": {"cohere": "cohere/rerank-english-v3.0"},
            }

            with pytest.raises(ValueError, match="not found in supported_models"):
                LearnedReranker()

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, sample_results):
        """Test reranking with empty results"""
        reranker = LearnedReranker(model="cohere/rerank-english-v3.0")
        results = await reranker.rerank("test query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_calls_litellm(self, sample_results):
        """Test rerank calls LiteLLM arerank"""
        with patch("src.app.search.learned_reranker.arerank") as mock_arerank:
            # Mock LiteLLM response
            mock_response = Mock()
            mock_result_item = Mock()
            mock_result_item.index = 1  # Second doc ranked first
            mock_result_item.relevance_score = 0.95
            mock_response.results = [mock_result_item]
            mock_arerank.return_value = mock_response

            reranker = LearnedReranker(model="cohere/rerank-english-v3.0")
            results = await reranker.rerank("test query", sample_results)

            # Verify LiteLLM was called
            mock_arerank.assert_called_once()
            call_args = mock_arerank.call_args
            assert call_args.kwargs["model"] == "cohere/rerank-english-v3.0"
            assert call_args.kwargs["query"] == "test query"
            assert len(call_args.kwargs["documents"]) == 2

            # Verify results updated
            assert len(results) == 1
            assert results[0].id == "doc-2"  # Index 1
            assert results[0].metadata["reranking_score"] == 0.95
            assert results[0].metadata["reranker_model"] == "cohere/rerank-english-v3.0"

    @pytest.mark.asyncio
    async def test_rerank_limits_max_results(self, sample_results):
        """Test rerank limits results to max_results_to_rerank"""
        with patch("src.app.search.learned_reranker.get_config_value") as mock_config:
            mock_config.return_value = {
                "model": "cohere",
                "supported_models": {"cohere": "cohere/rerank-english-v3.0"},
                "max_results_to_rerank": 1,  # Limit to 1
            }

            with patch("src.app.search.learned_reranker.arerank") as mock_arerank:
                mock_response = Mock()
                mock_response.results = []
                mock_arerank.return_value = mock_response

                reranker = LearnedReranker()
                await reranker.rerank("test", sample_results)

                # Should only send 1 document to LiteLLM
                call_args = mock_arerank.call_args
                assert len(call_args.kwargs["documents"]) == 1

    @pytest.mark.asyncio
    async def test_rerank_handles_litellm_error(self, sample_results):
        """Test rerank returns original results on LiteLLM error"""
        with patch("src.app.search.learned_reranker.arerank") as mock_arerank:
            mock_arerank.side_effect = Exception("LiteLLM API error")

            reranker = LearnedReranker(model="cohere/rerank-english-v3.0")
            results = await reranker.rerank("test query", sample_results)

            # Should return original results on error
            assert len(results) == 2
            assert results == sample_results

    def test_rerank_sync_version(self, sample_results):
        """Test synchronous rerank_sync method"""
        with patch("src.app.search.learned_reranker.rerank") as mock_rerank:
            mock_response = Mock()
            mock_result_item = Mock()
            mock_result_item.index = 0
            mock_result_item.relevance_score = 0.9
            mock_response.results = [mock_result_item]
            mock_rerank.return_value = mock_response

            reranker = LearnedReranker(model="cohere/rerank-english-v3.0")
            results = reranker.rerank_sync("test query", sample_results)

            # Verify sync version was called
            mock_rerank.assert_called_once()
            assert len(results) == 1

    def test_get_model_info(self):
        """Test get_model_info returns correct information"""
        with patch("src.app.search.learned_reranker.get_config_value") as mock_config:
            mock_config.return_value = {
                "model": "cohere",
                "supported_models": {"cohere": "cohere/rerank-english-v3.0"},
                "top_n": 5,
                "max_results_to_rerank": 50,
            }

            reranker = LearnedReranker()
            info = reranker.get_model_info()

            assert info["model"] == "cohere/rerank-english-v3.0"
            assert info["default_top_n"] == 5
            assert info["max_results_to_rerank"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
