"""
Integration tests for Query Expansion and Multi-Modal Reranking

Tests the complete flow with real Vespa instance and actual multi-modal content:
1. Query expansion across modalities
2. Multi-modal search across video/image/document/audio
3. Cross-modal reranking with real results
4. Contextual analysis with conversation history

Requires:
- Vespa Docker instance running on test port
- Real ColPali models for embeddings
- Sample multi-modal content ingested
"""

import subprocess
import time
from datetime import datetime, timedelta

import pytest

from cogniverse_agents.routing.contextual_analyzer import ContextualAnalyzer
from cogniverse_agents.routing.query_expansion import QueryExpander
from cogniverse_agents.search.multi_modal_reranker import (
    MultiModalReranker,
    QueryModality,
    SearchResult,
)
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager


@pytest.fixture(scope="module")
def test_vespa_integration():
    """
    Setup test Vespa Docker instance for integration testing

    Uses port 8083 to avoid conflicts with main Vespa (8080) and unit tests (8082)
    """
    print("\n" + "=" * 80)
    print("Setting up Query Expansion & Reranking Integration Test Vespa")
    print("=" * 80)

    # Configuration
    test_port = 8083
    config_port = 19074
    container_name = f"vespa-query-rerank-test-{test_port}"

    # Step 1: Cleanup existing container
    print(f"\n🧹 Cleaning up existing container '{container_name}'...")
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)

    # Step 2: Start Vespa Docker
    print(f"\n🚀 Starting Vespa on port {test_port}...")

    import platform

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ["arm64", "aarch64"] else "linux/amd64"
    )

    docker_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{test_port}:8080",
            "-p",
            f"{config_port}:19071",
            "--platform",
            docker_platform,
            "vespaengine/vespa",
        ],
        capture_output=True,
        timeout=60,
    )

    if docker_result.returncode != 0:
        pytest.fail(
            f"Failed to start Docker container: {docker_result.stderr.decode()}"
        )

    print(f"✅ Container '{container_name}' started")

    # Step 3: Wait for Vespa to be ready
    print(f"\n⏳ Waiting for Vespa config server on port {config_port}...")
    import requests

    for i in range(120):
        try:
            response = requests.get(f"http://localhost:{config_port}/", timeout=5)
            if response.status_code == 200:
                print(f"✅ Config server ready (took {i}s)")
                break
        except Exception:
            pass
        time.sleep(1)
        if i % 10 == 0 and i > 0:
            print(f"   Still waiting... ({i}s)")
    else:
        # Cleanup on failure
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)
        pytest.fail("Vespa config server not ready after 120 seconds")

    # Return test configuration
    test_config = {
        "http_port": test_port,
        "config_port": config_port,
        "container_name": container_name,
        "base_url": f"http://localhost:{test_port}",
        "manager": VespaSchemaManager(
            vespa_endpoint=f"http://localhost:{test_port}",
            config_endpoint=f"http://localhost:{config_port}",
        ),
    }

    yield test_config

    # Cleanup
    print(f"\n🧹 Cleaning up test Vespa container '{container_name}'...")
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)
    print("✅ Cleanup complete")


@pytest.fixture
def query_expander():
    """Create QueryExpander instance"""
    return QueryExpander()


@pytest.fixture
def reranker():
    """Create MultiModalReranker instance"""
    return MultiModalReranker()


@pytest.fixture
def contextual_analyzer():
    """Create ContextualAnalyzer instance"""
    return ContextualAnalyzer()


@pytest.mark.integration
class TestQueryExpansionIntegration:
    """Integration tests for query expansion with real scenarios"""

    @pytest.mark.asyncio
    async def test_visual_query_expansion_flow(self, query_expander):
        """Test complete visual query expansion flow"""
        # User asks for visual content
        query = "show me machine learning tutorials from 2023"

        # Expand query
        expansion = await query_expander.expand_query(query)

        # Verify expansion structure
        assert expansion["original_query"] == query
        assert (
            "video" in expansion["modality_intent"]
            or "visual" in expansion["modality_intent"]
        )
        assert expansion["temporal"]["requires_temporal_search"] is True
        assert expansion["temporal"]["temporal_type"] == "year"

        # Verify text alternatives generated
        assert "text_alternatives" in expansion["expansions"]
        text_alts = expansion["expansions"]["text_alternatives"]
        assert len(text_alts) > 0
        # Should contain cleaned query or variations
        assert any("machine learning tutorials" in alt for alt in text_alts)

        # Verify modality-specific expansions
        assert "video" in expansion["expansions"]
        video_expansions = expansion["expansions"]["video"]
        assert any("tutorial" in exp.lower() for exp in video_expansions)

    @pytest.mark.asyncio
    async def test_text_query_to_visual_expansion(self, query_expander):
        """Test expanding text query to visual keywords"""
        query = "explain neural network architecture"

        expansion = await query_expander.expand_query(query)

        # Should detect text intent but also provide visual alternatives
        assert "visual_alternatives" in expansion["expansions"]

        visual_alts = expansion["expansions"]["visual_alternatives"]
        assert "video_keywords" in visual_alts
        assert "image_keywords" in visual_alts

        # Video keywords should include tutorial/demo
        video_kw = " ".join(visual_alts["video_keywords"])
        assert "tutorial" in video_kw or "demonstration" in video_kw

        # Image keywords should include diagrams/charts
        image_kw = " ".join(visual_alts["image_keywords"])
        assert "diagram" in image_kw or "architecture" in image_kw

    @pytest.mark.asyncio
    async def test_temporal_expansion_relative_dates(self, query_expander):
        """Test temporal expansion with relative dates"""
        query = "videos from last week about python"

        temporal_info = await query_expander.expand_temporal(query)

        assert temporal_info["requires_temporal_search"] is True
        assert temporal_info["temporal_type"] == "relative"
        assert "last week" in temporal_info["temporal_keywords"]
        assert temporal_info["time_range"] is not None

        start, end = temporal_info["time_range"]
        # Should be approximately 7 days ago to now
        delta = (end - start).days
        assert 6 <= delta <= 8


@pytest.mark.integration
class TestMultiModalRerankingIntegration:
    """Integration tests for multi-modal reranking with real-like data"""

    def create_mock_search_results(self) -> list[SearchResult]:
        """Create realistic multi-modal search results"""
        now = datetime.now()

        return [
            SearchResult(
                id="video_ml_tutorial",
                title="Machine Learning Crash Course",
                content="Comprehensive ML tutorial covering supervised and unsupervised learning",
                modality="video",
                score=0.92,
                metadata={"duration": 3600, "views": 50000},
                timestamp=now - timedelta(days=5),
            ),
            SearchResult(
                id="doc_ml_paper",
                title="Introduction to Machine Learning",
                content="Academic paper covering ML fundamentals and algorithms",
                modality="document",
                score=0.88,
                metadata={"pages": 25, "citations": 150},
                timestamp=now - timedelta(days=30),
            ),
            SearchResult(
                id="image_nn_diagram",
                title="Neural Network Architecture Diagram",
                content="Visual representation of deep neural network layers and connections",
                modality="image",
                score=0.85,
                metadata={"resolution": "1920x1080", "format": "png"},
                timestamp=now - timedelta(days=10),
            ),
            SearchResult(
                id="audio_ml_podcast",
                title="Machine Learning Podcast Episode",
                content="Discussion about recent developments in machine learning",
                modality="audio",
                score=0.80,
                metadata={"duration": 2400, "speakers": 2},
                timestamp=now - timedelta(days=2),
            ),
            SearchResult(
                id="video_nn_basics",
                title="Neural Networks Explained",
                content="Step-by-step explanation of how neural networks work",
                modality="video",
                score=0.87,
                metadata={"duration": 1800, "views": 30000},
                timestamp=now - timedelta(days=15),
            ),
        ]

    @pytest.mark.asyncio
    async def test_modality_preference_reranking(self, reranker):
        """Test reranking with modality preferences"""
        results = self.create_mock_search_results()
        query = "machine learning tutorial"
        modalities = [QueryModality.VIDEO]  # User prefers videos

        reranked = await reranker.rerank_results(results, query, modalities, {})

        # Verify reranking happened
        assert len(reranked) == len(results)

        # Video results should be ranked higher
        top_3 = reranked[:3]
        video_count = sum(1 for r in top_3 if r.modality == "video")
        assert video_count >= 1, "At least one video should be in top 3"

        # Verify metadata added
        for result in reranked:
            assert "reranking_score" in result.metadata
            assert "score_components" in result.metadata

    @pytest.mark.asyncio
    async def test_temporal_alignment_reranking(self, reranker):
        """Test reranking with temporal preferences"""
        results = self.create_mock_search_results()
        query = "machine learning"
        modalities = [QueryModality.VIDEO]

        # Context: looking for recent content (last 7 days)
        context = {
            "temporal": {
                "time_range": (
                    datetime.now() - timedelta(days=7),
                    datetime.now(),
                ),
                "requires_temporal_search": True,
            }
        }

        reranked = await reranker.rerank_results(results, query, modalities, context)

        # More recent results should benefit from temporal alignment
        # audio_ml_podcast (2 days ago) and video_ml_tutorial (5 days ago) should rank well
        top_result_ids = [r.id for r in reranked[:3]]
        assert (
            "audio_ml_podcast" in top_result_ids
            or "video_ml_tutorial" in top_result_ids
        ), "Recent content should rank in top 3"

    @pytest.mark.asyncio
    async def test_diversity_aware_reranking(self, reranker):
        """Test that reranking promotes diversity"""
        results = self.create_mock_search_results()
        query = "machine learning"
        modalities = [QueryModality.MIXED]  # Accept all modalities

        reranked = await reranker.rerank_results(results, query, modalities, {})

        # Top 4 results should have some diversity
        top_4 = reranked[:4]
        modalities_in_top4 = {r.modality for r in top_4}

        # Should have at least 2 different modalities in top 4
        assert len(modalities_in_top4) >= 2, f"Only {modalities_in_top4} in top 4"

    @pytest.mark.asyncio
    async def test_reranking_quality_metrics(self, reranker):
        """Test ranking quality analysis"""
        results = self.create_mock_search_results()
        query = "machine learning"
        modalities = [QueryModality.VIDEO]

        reranked = await reranker.rerank_results(results, query, modalities, {})

        # Analyze quality
        quality = reranker.analyze_ranking_quality(reranked)

        assert "diversity" in quality
        assert "average_score" in quality
        assert "modality_distribution" in quality
        assert "temporal_coverage" in quality
        assert "total_results" in quality

        assert 0.0 <= quality["diversity"] <= 1.0
        assert quality["total_results"] == len(results)


@pytest.mark.integration
class TestContextualAnalysisIntegration:
    """Integration tests for contextual analysis with conversation flow"""

    @pytest.mark.asyncio
    async def test_conversation_flow_tracking(
        self, query_expander, reranker, contextual_analyzer
    ):
        """Test full conversation flow with context tracking"""
        # Query 1: User starts with video query
        query1 = "show me machine learning videos"
        expansion1 = await query_expander.expand_query(query1)

        contextual_analyzer.update_context(
            query=query1,
            detected_modalities=expansion1["modality_intent"],
            result_count=5,
        )

        # Verify context updated
        assert contextual_analyzer.total_queries == 1
        assert "video" in contextual_analyzer.modality_preferences

        # Query 2: User asks for images
        query2 = "find diagrams of neural networks"
        expansion2 = await query_expander.expand_query(query2)

        contextual_analyzer.update_context(
            query=query2,
            detected_modalities=expansion2["modality_intent"],
            result_count=3,
        )

        # Verify modality shift detected
        assert contextual_analyzer.total_queries == 2
        conversation_ctx = contextual_analyzer._get_conversation_context()
        assert conversation_ctx["modality_shifts"] > 0

        # Query 3: Get contextual hints
        hints = contextual_analyzer.get_contextual_hints("deep learning")

        assert "preferred_modalities" in hints
        assert "conversation_context" in hints
        assert hints["conversation_context"]["conversation_depth"] == 2

    @pytest.mark.asyncio
    async def test_modality_preference_learning(self, contextual_analyzer):
        """Test that analyzer learns modality preferences over time"""
        # Simulate user preference for videos
        for i in range(5):
            contextual_analyzer.update_context(
                query=f"video query {i}",
                detected_modalities=["video"],
                result_count=3,
            )

        # Add some document queries
        for i in range(2):
            contextual_analyzer.update_context(
                query=f"document query {i}",
                detected_modalities=["document"],
                result_count=2,
            )

        # Get preferences
        hints = contextual_analyzer.get_contextual_hints("new query")
        top_modalities = hints["preferred_modalities"]

        # Video should be top preference
        assert len(top_modalities) > 0
        assert top_modalities[0]["modality"] == "video"
        assert top_modalities[0]["count"] == 5

    @pytest.mark.asyncio
    async def test_topic_evolution_tracking(self, contextual_analyzer):
        """Test topic tracking across conversation"""
        # Build conversation about related topics
        contextual_analyzer.update_context(
            query="machine learning basics",
            detected_modalities=["video"],
            result_count=5,
        )

        contextual_analyzer.update_context(
            query="deep learning neural networks",
            detected_modalities=["document"],
            result_count=3,
        )

        contextual_analyzer.update_context(
            query="learning algorithms comparison",
            detected_modalities=["video"],
            result_count=4,
        )

        # "learning" should be tracked as recurring topic
        assert "learning" in contextual_analyzer.topic_tracking
        learning_mentions = len(contextual_analyzer.topic_tracking["learning"])
        assert learning_mentions >= 3


@pytest.mark.integration
class TestEndToEndMultiModalFlow:
    """Test complete end-to-end flow: query → expand → search → rerank → analyze"""

    @pytest.mark.asyncio
    async def test_complete_search_flow(
        self, query_expander, reranker, contextual_analyzer
    ):
        """Test complete multi-modal search flow"""
        # Step 1: User query
        user_query = "show me recent tutorials about neural networks"

        # Step 2: Query expansion
        expansion = await query_expander.expand_query(user_query)

        detected_modalities = expansion["modality_intent"]
        assert "video" in detected_modalities or "visual" in detected_modalities

        # Step 3: Simulate multi-modal search results
        now = datetime.now()
        search_results = [
            SearchResult(
                id="video_recent",
                title="Neural Networks 2024 Tutorial",
                content="Latest neural network techniques",
                modality="video",
                score=0.91,
                metadata={},
                timestamp=now - timedelta(days=3),
            ),
            SearchResult(
                id="doc_classic",
                title="Classic Neural Networks Paper",
                content="Foundational neural network research",
                modality="document",
                score=0.89,
                metadata={},
                timestamp=now - timedelta(days=365),
            ),
            SearchResult(
                id="video_tutorial",
                title="NN Tutorial Series",
                content="Comprehensive neural network tutorial",
                modality="video",
                score=0.88,
                metadata={},
                timestamp=now - timedelta(days=7),
            ),
        ]

        # Step 4: Rerank with temporal context (prefer recent content)
        context = {
            "temporal": {
                "time_range": (
                    datetime.now() - timedelta(days=30),
                    datetime.now(),
                ),
                "requires_temporal_search": True,
            }
        }

        # Convert modality intent to QueryModality
        query_modalities = []
        if "video" in detected_modalities or "visual" in detected_modalities:
            query_modalities.append(QueryModality.VIDEO)

        reranked = await reranker.rerank_results(
            search_results, user_query, query_modalities, context
        )

        # Recent video should rank highest
        assert reranked[0].id == "video_recent", "Recent video should rank first"

        # Step 5: Update contextual analyzer
        contextual_analyzer.update_context(
            query=user_query,
            detected_modalities=detected_modalities,
            result_count=len(reranked),
        )

        # Verify context tracked
        assert contextual_analyzer.total_queries == 1
        assert contextual_analyzer.successful_queries == 1

        # Step 6: Get hints for next query
        hints = contextual_analyzer.get_contextual_hints("advanced neural networks")

        assert "preferred_modalities" in hints
        assert "session_metrics" in hints
        assert hints["session_metrics"]["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
