"""
Integration tests for learned reranking with LiteLLM:
- Real LiteLLM calls (if API keys available)
- Hybrid reranking end-to-end
- Config loading from config.json
"""

import pytest

from cogniverse_agents.search.hybrid_reranker import HybridReranker
from cogniverse_agents.search.learned_reranker import LearnedReranker
from cogniverse_agents.search.multi_modal_reranker import MultiModalReranker
from cogniverse_agents.search.types import QueryModality, RerankerSearchResult


@pytest.fixture
def mock_config_manager():
    """Create mock config_manager for dependency injection"""
    from unittest.mock import Mock

    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        RoutingConfigUnified,
        SystemConfig,
    )
    from cogniverse_foundation.telemetry.config import TelemetryConfig

    mock_cm = Mock()

    # Mock get_backend_config to return proper BackendConfig
    mock_backend_config = BackendConfig(
        tenant_id="test:unit",
        backend_type="vespa",
        url="http://localhost",
        port=8080,
        profiles={},  # Empty profiles dict is sufficient for reranker tests
        metadata={},
    )
    mock_cm.get_backend_config.return_value = mock_backend_config

    # Mock get_system_config
    mock_system_config = SystemConfig()
    mock_cm.get_system_config.return_value = mock_system_config

    # Mock get_routing_config
    mock_routing_config = RoutingConfigUnified(tenant_id="test:unit")
    mock_cm.get_routing_config.return_value = mock_routing_config

    # Mock get_telemetry_config
    mock_telemetry_config = TelemetryConfig()
    mock_cm.get_telemetry_config.return_value = mock_telemetry_config

    return mock_cm


@pytest.fixture
def sample_results():
    """Create sample search results for testing"""
    return [
        RerankerSearchResult(
            id="doc-1",
            title="Machine Learning Tutorial",
            content="Introduction to machine learning concepts and algorithms",
            modality="text",
            score=0.8,
            metadata={"original_score": 0.8},
        ),
        RerankerSearchResult(
            id="doc-2",
            title="Deep Learning Guide",
            content="Comprehensive guide to neural networks and deep learning",
            modality="text",
            score=0.7,
            metadata={"original_score": 0.7},
        ),
        RerankerSearchResult(
            id="doc-3",
            title="Python Programming",
            content="Learn Python programming from basics to advanced",
            modality="text",
            score=0.6,
            metadata={"original_score": 0.6},
        ),
    ]


@pytest.mark.unit
class TestLearnedRerankingIntegration:
    """Integration tests with real LiteLLM (if available)"""

    @pytest.mark.asyncio
    async def test_learned_reranker_with_mock_litellm(
        self, sample_results, mock_config_manager
    ):
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

            reranker = LearnedReranker(
                model="openai/test-reranker",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            query = "deep learning tutorial"

            reranked = await reranker.rerank(query, sample_results)

            # Verify reranking
            assert len(reranked) == 3
            assert reranked[0].id == "doc-2"  # Highest score
            assert reranked[1].id == "doc-1"
            assert reranked[2].id == "doc-3"

            # Verify metadata
            assert reranked[0].metadata["reranking_score"] == 0.95
            assert reranked[0].metadata["reranker_model"] == "openai/test-reranker"

    @pytest.mark.asyncio
    async def test_learned_reranker_with_local_model(
        self, sample_results, mock_config_manager
    ):
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

            reranker = LearnedReranker(
                model="openai/test-reranker",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            query = "machine learning tutorial"

            reranked = await reranker.rerank(query, sample_results, top_n=2)

            # Verify reranking worked
            assert len(reranked) <= 2
            assert all("reranking_score" in r.metadata for r in reranked)
            assert all("reranker_model" in r.metadata for r in reranked)

            # Scores should be in descending order
            scores = [r.metadata["reranking_score"] for r in reranked]
            assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
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
            learned = LearnedReranker(
                model="cohere/rerank-english-v3.0",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            hybrid = HybridReranker(
                heuristic_reranker=heuristic,
                learned_reranker=learned,
                strategy="weighted_ensemble",
                learned_weight=0.7,
                heuristic_weight=0.3,
                config_manager=mock_config_manager,
                tenant_id="test:unit",
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
            learned = LearnedReranker(
                model="cohere/rerank-english-v3.0",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            hybrid = HybridReranker(
                heuristic_reranker=heuristic,
                learned_reranker=learned,
                strategy="cascade",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
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
            learned = LearnedReranker(
                model="cohere/rerank-english-v3.0",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            hybrid = HybridReranker(
                heuristic_reranker=heuristic,
                learned_reranker=learned,
                strategy="consensus",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )

            query = "data science"
            modalities = [QueryModality.TEXT]

            reranked = await hybrid.rerank_hybrid(query, sample_results, modalities)

            # Verify consensus metadata
            assert len(reranked) == len(sample_results)
            assert all("heuristic_rank" in r.metadata for r in reranked)
            assert all("learned_rank" in r.metadata for r in reranked)
            assert all(r.metadata["fusion_strategy"] == "consensus" for r in reranked)


@pytest.mark.unit
class TestRerankingOAICompat:
    """Test reranking via OpenAI-compatible local LM API"""

    @pytest.mark.asyncio
    async def test_reranker_with_mock_oai_api(
        self, sample_results, mock_config_manager
    ):
        """Test reranking using LiteLLM OpenAI-compat with a custom api_base"""
        from unittest.mock import Mock, patch

        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            # Mock LiteLLM response with the configured model
            mock_response = Mock()
            mock_items = [
                Mock(index=1, relevance_score=0.92),  # doc-2 first
                Mock(index=0, relevance_score=0.88),  # doc-1 second
            ]
            mock_response.results = mock_items
            mock_arerank.return_value = mock_response

            # Initialize with the configured model using OpenAI compatibility
            reranker = LearnedReranker(
                model="openai/bge-reranker-v2-m3",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            reranker.api_base = "http://localhost:11434/v1"

            query = "machine learning tutorial"
            reranked = await reranker.rerank(query, sample_results, top_n=2)

            # Verify LiteLLM was called with api_base for the local LM
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


@pytest.mark.unit
class TestRerankingCorrectsRetrievalOrder:
    """A complex query where lexical retrieval ranks the wrong doc first and the
    reranker must correct it. The cross-encoder is mocked (no reranker model is
    deployed in this env), but the relevance values mirror what a query-aware
    cross-encoder produces: the misleading high-retrieval doc is penalised and
    the true answer is promoted. Assertions are on the *resulting order*, not
    just the id set."""

    @pytest.fixture
    def thaw_results(self):
        # Retrieval order (by score, descending) is deliberately WRONG for the
        # query: the microwave doc wins on keyword overlap ("frozen"/"defrost")
        # but the query explicitly excludes microwaves; the cold-water doc is
        # the real answer yet retrieves lowest.
        return [
            RerankerSearchResult(
                id="doc-microwave",
                title="Quick microwave defrosting for any frozen meat",
                content="Defrost frozen meat fast in the microwave on low power.",
                modality="text",
                score=0.92,  # highest retrieval score
                metadata={},
            ),
            RerankerSearchResult(
                id="doc-recipe",
                title="Grilled salmon dinner recipes",
                content="Twelve salmon recipes for weeknight dinners.",
                modality="text",
                score=0.70,
                metadata={},
            ),
            RerankerSearchResult(
                id="doc-coldwater",
                title="Thawing fish safely in cold water, step by step",
                content="Submerge sealed frozen fish in cold water to thaw safely without a microwave.",
                modality="text",
                score=0.55,  # lowest retrieval score, but the real answer
                metadata={},
            ),
        ]

    @pytest.mark.asyncio
    async def test_reranker_promotes_true_answer_over_misleading_top_hit(
        self, thaw_results, mock_config_manager
    ):
        from unittest.mock import Mock, patch

        query = "how to safely thaw frozen salmon without using a microwave"

        with patch("cogniverse_agents.search.learned_reranker.arerank") as mock_arerank:
            # Query-aware relevance (what a cross-encoder yields): cold-water
            # answer top, recipe tangential, microwave doc penalised despite its
            # keyword overlap. response.results is relevance-sorted; .index maps
            # back to the input list [microwave=0, recipe=1, coldwater=2].
            mock_response = Mock()
            mock_response.results = [
                Mock(index=2, relevance_score=0.94),  # doc-coldwater
                Mock(index=1, relevance_score=0.38),  # doc-recipe
                Mock(index=0, relevance_score=0.05),  # doc-microwave
            ]
            mock_arerank.return_value = mock_response

            reranker = LearnedReranker(
                model="openai/bge-reranker-v2-m3",
                config_manager=mock_config_manager,
                tenant_id="test:unit",
            )
            reranked = await reranker.rerank(query, thaw_results)

            # The reranker passed the right query + docs to the cross-encoder.
            call_kwargs = mock_arerank.call_args.kwargs
            assert call_kwargs["query"] == query
            assert call_kwargs["documents"][0].startswith("Quick microwave defrosting")

            # Exact corrected order — the true answer rose from retrieval-rank 3
            # to #1, the misleading top retrieval hit sank to last.
            assert [r.id for r in reranked] == [
                "doc-coldwater",
                "doc-recipe",
                "doc-microwave",
            ]
            # Reranking actually changed the top result (not a pass-through).
            assert reranked[0].id != thaw_results[0].id
            assert reranked[0].id == "doc-coldwater"
            assert reranked[-1].id == "doc-microwave"
            assert reranked[0].metadata["reranking_score"] == 0.94


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
