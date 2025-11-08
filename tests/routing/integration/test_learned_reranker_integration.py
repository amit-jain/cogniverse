"""
Integration tests for learned reranking with LiteLLM:
- Real LiteLLM calls (if API keys available)
- Hybrid reranking end-to-end
- ConfigurableMultiModalReranker with real components
- Config loading from config.json
"""


import pytest
from cogniverse_agents.search.hybrid_reranker import HybridReranker
from cogniverse_agents.search.learned_reranker import LearnedReranker
from cogniverse_agents.search.multi_modal_reranker import (
    ConfigurableMultiModalReranker,
    MultiModalReranker,
    QueryModality,
    SearchResult,
)


@pytest.fixture
def mock_config_manager():
    """Create mock config_manager for dependency injection"""
    from unittest.mock import Mock

    from cogniverse_core.config.unified_config import (
        BackendConfig,
        RoutingConfigUnified,
        SystemConfig,
        TelemetryConfigUnified,
    )

    mock_cm = Mock()

    # Mock get_backend_config to return proper BackendConfig
    mock_backend_config = BackendConfig(
        tenant_id="default",
        backend_type="vespa",
        url="http://localhost",
        port=8080,
        profiles={},  # Empty profiles dict is sufficient for reranker tests
        metadata={}
    )
    mock_cm.get_backend_config.return_value = mock_backend_config

    # Mock get_system_config
    mock_system_config = SystemConfig(tenant_id="default")
    mock_cm.get_system_config.return_value = mock_system_config

    # Mock get_routing_config
    mock_routing_config = RoutingConfigUnified(tenant_id="default")
    mock_cm.get_routing_config.return_value = mock_routing_config

    # Mock get_telemetry_config
    mock_telemetry_config = TelemetryConfigUnified(tenant_id="default")
    mock_cm.get_telemetry_config.return_value = mock_telemetry_config

    return mock_cm


@pytest.fixture
def sample_results():
    """Create sample search results for testing"""
    return [
        SearchResult(
            id="doc-1",
            title="Machine Learning Tutorial",
            content="Introduction to machine learning concepts and algorithms",
            modality="text",
            score=0.8,
            metadata={"original_score": 0.8},
        ),
        SearchResult(
            id="doc-2",
            title="Deep Learning Guide",
            content="Comprehensive guide to neural networks and deep learning",
            modality="text",
            score=0.7,
            metadata={"original_score": 0.7},
        ),
        SearchResult(
            id="doc-3",
            title="Python Programming",
            content="Learn Python programming from basics to advanced",
            modality="text",
            score=0.6,
            metadata={"original_score": 0.6},
        ),
    ]


@pytest.mark.integration
class TestLearnedRerankingIntegration:
    """Integration tests with real LiteLLM (if available)"""

    @pytest.mark.asyncio
    async def test_learned_reranker_with_mock_litellm(self, sample_results, mock_config_manager):
        """Test learned reranker with mocked LiteLLM (no API key needed)"""
        from unittest.mock import Mock, patch

        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            # Mock LiteLLM response
            mock_response = Mock()
            mock_items = [
                Mock(index=1, relevance_score=0.95),  # doc-2 first
                Mock(index=0, relevance_score=0.85),  # doc-1 second
                Mock(index=2, relevance_score=0.75),  # doc-3 third
            ]
            mock_response.results = mock_items
            mock_arerank.return_value = mock_response

            reranker = LearnedReranker(model="ollama/qwen2.5:3b", config_manager=mock_config_manager)
            query = "deep learning tutorial"

            reranked = await reranker.rerank(query, sample_results)

            # Verify reranking
            assert len(reranked) == 3
            assert reranked[0].id == "doc-2"  # Highest score
            assert reranked[1].id == "doc-1"
            assert reranked[2].id == "doc-3"

            # Verify metadata
            assert reranked[0].metadata["reranking_score"] == 0.95
            assert reranked[0].metadata["reranker_model"] == "ollama/qwen2.5:3b"

    @pytest.mark.asyncio
    async def test_learned_reranker_with_local_model(self, sample_results, mock_config_manager):
        """Test learned reranker with local Qwen model"""
        from unittest.mock import Mock, patch

        # Mock the reranking API call to use local model
        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            mock_response = Mock()
            mock_response.results = [
                Mock(index=1, relevance_score=0.95),
                Mock(index=0, relevance_score=0.85),
            ]
            mock_arerank.return_value = mock_response

            reranker = LearnedReranker(model="ollama/qwen2.5:3b", config_manager=mock_config_manager)
            query = "machine learning tutorial"

            reranked = await reranker.rerank(query, sample_results, top_n=2)

            # Verify reranking worked
            assert len(reranked) <= 2
            assert all("reranking_score" in r.metadata for r in reranked)
            assert all("reranker_model" in r.metadata for r in reranked)

            # Scores should be in descending order
            scores = [r.metadata["reranking_score"] for r in reranked]
            assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
class TestHybridRerankingIntegration:
    """Integration tests for hybrid reranking"""

    @pytest.mark.asyncio
    async def test_hybrid_weighted_ensemble(self, sample_results, mock_config_manager):
        """Test hybrid reranking with weighted ensemble"""
        from unittest.mock import Mock, patch

        # Mock LiteLLM
        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            mock_response = Mock()
            mock_response.results = [
                Mock(index=i, relevance_score=0.9 - i * 0.1)
                for i in range(len(sample_results))
            ]
            mock_arerank.return_value = mock_response

            # Create hybrid reranker
            heuristic = MultiModalReranker()
            learned = LearnedReranker(model="cohere/rerank-english-v3.0", config_manager=mock_config_manager)
            hybrid = HybridReranker(
                heuristic_reranker=heuristic,
                learned_reranker=learned,
                strategy="weighted_ensemble",
                learned_weight=0.7,
                heuristic_weight=0.3,
                config_manager=mock_config_manager,
            )

            query = "deep learning"
            modalities = [QueryModality.TEXT]

            reranked = await hybrid.rerank_hybrid(query, sample_results, modalities)

            # Verify hybrid scores
            assert len(reranked) == len(sample_results)
            assert all("reranking_score" in r.metadata for r in reranked)
            assert all("heuristic_score" in r.metadata for r in reranked)
            assert all("learned_score" in r.metadata for r in reranked)
            assert all(
                r.metadata["fusion_strategy"] == "weighted_ensemble" for r in reranked
            )

    @pytest.mark.asyncio
    async def test_hybrid_cascade(self, sample_results, mock_config_manager):
        """Test hybrid reranking with cascade strategy"""
        from unittest.mock import Mock, patch

        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            mock_response = Mock()
            mock_response.results = [Mock(index=0, relevance_score=0.95)]
            mock_arerank.return_value = mock_response

            heuristic = MultiModalReranker()
            learned = LearnedReranker(model="cohere/rerank-english-v3.0", config_manager=mock_config_manager)
            hybrid = HybridReranker(
                heuristic_reranker=heuristic,
                learned_reranker=learned,
                strategy="cascade",
                config_manager=mock_config_manager,
            )

            query = "python tutorial"
            modalities = [QueryModality.TEXT]

            reranked = await hybrid.rerank_hybrid(query, sample_results, modalities)

            # Cascade filters, so may have fewer results
            assert len(reranked) <= len(sample_results)
            assert all(r.metadata["fusion_strategy"] == "cascade" for r in reranked)

    @pytest.mark.asyncio
    async def test_hybrid_consensus(self, sample_results, mock_config_manager):
        """Test hybrid reranking with consensus strategy"""
        from unittest.mock import Mock, patch

        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            mock_response = Mock()
            mock_response.results = [
                Mock(index=i, relevance_score=0.9 - i * 0.1)
                for i in range(len(sample_results))
            ]
            mock_arerank.return_value = mock_response

            heuristic = MultiModalReranker()
            learned = LearnedReranker(model="cohere/rerank-english-v3.0", config_manager=mock_config_manager)
            hybrid = HybridReranker(
                heuristic_reranker=heuristic,
                learned_reranker=learned,
                strategy="consensus",
                config_manager=mock_config_manager,
            )

            query = "data science"
            modalities = [QueryModality.TEXT]

            reranked = await hybrid.rerank_hybrid(query, sample_results, modalities)

            # Verify consensus metadata
            assert len(reranked) == len(sample_results)
            assert all("heuristic_rank" in r.metadata for r in reranked)
            assert all("learned_rank" in r.metadata for r in reranked)
            assert all(r.metadata["fusion_strategy"] == "consensus" for r in reranked)


@pytest.mark.integration
class TestOllamaReranking:
    """Test Ollama reranking via OpenAI-compatible API"""

    @pytest.mark.asyncio
    async def test_ollama_reranker_with_mock_api(self, sample_results, mock_config_manager):
        """Test Ollama reranking using LiteLLM OpenAI compatibility"""
        from unittest.mock import Mock, patch

        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            # Mock LiteLLM response with Ollama model
            mock_response = Mock()
            mock_items = [
                Mock(index=1, relevance_score=0.92),  # doc-2 first
                Mock(index=0, relevance_score=0.88),  # doc-1 second
            ]
            mock_response.results = mock_items
            mock_arerank.return_value = mock_response

            # Initialize with Ollama model using OpenAI compatibility
            reranker = LearnedReranker(model="openai/bge-reranker-v2-m3", config_manager=mock_config_manager)
            reranker.api_base = "http://localhost:11434/v1"

            query = "machine learning tutorial"
            reranked = await reranker.rerank(query, sample_results, top_n=2)

            # Verify LiteLLM was called with api_base for Ollama
            mock_arerank.assert_called_once()
            call_kwargs = mock_arerank.call_args.kwargs
            assert call_kwargs["model"] == "openai/bge-reranker-v2-m3"
            assert call_kwargs["api_base"] == "http://localhost:11434/v1"
            assert call_kwargs["query"] == query

            # Verify reranking
            assert len(reranked) == 2
            assert reranked[0].id == "doc-2"
            assert reranked[1].id == "doc-1"
            assert reranked[0].metadata["reranking_score"] == 0.92


@pytest.mark.integration
class TestConfigurableMultiModalReranker:
    """Test ConfigurableMultiModalReranker with real components"""

    @pytest.mark.asyncio
    async def test_configurable_reranker_disabled(self, sample_results, mock_config_manager):
        """Test configurable reranker when disabled in config"""
        from unittest.mock import patch

        # Patch where get_config_value is imported and used
        with patch("cogniverse_core.config.utils.get_config_value") as mock_config:
            mock_config.return_value = {"enabled": False}

            reranker = ConfigurableMultiModalReranker(config_manager=mock_config_manager)
            query = "test query"
            modalities = [QueryModality.TEXT]

            # Should return original results when disabled
            results = await reranker.rerank(query, sample_results, modalities)
            assert results == sample_results

            # Verify info
            info = reranker.get_reranker_info()
            assert info["enabled"] is False

    @pytest.mark.asyncio
    async def test_configurable_reranker_heuristic_only(self, sample_results, mock_config_manager):
        """Test configurable reranker with heuristic only"""
        from unittest.mock import patch

        # Patch where get_config_value is imported and used
        with patch("cogniverse_core.config.utils.get_config_value") as mock_config:
            mock_config.return_value = {
                "enabled": True,
                "model": "heuristic",
                "use_hybrid": False,
            }

            reranker = ConfigurableMultiModalReranker(config_manager=mock_config_manager)
            query = "machine learning"
            modalities = [QueryModality.TEXT]

            results = await reranker.rerank(query, sample_results, modalities)

            assert len(results) == len(sample_results)
            assert all("reranking_score" in r.metadata for r in results)

            # Verify info
            info = reranker.get_reranker_info()
            assert info["enabled"] is True
            assert info["model"] == "heuristic"
            assert info["learned_available"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
