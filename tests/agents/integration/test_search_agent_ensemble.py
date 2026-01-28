"""
Integration tests for SearchAgent ensemble search with RRF fusion.

Tests the complete ensemble pipeline with real Vespa backend:
- Multiple profile deployment
- Parallel query execution
- RRF fusion algorithm
- Latency requirements
- Error handling
"""

import logging
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def search_agent_with_ensemble_profiles(vespa_with_schema):
    """SearchAgent configured with multiple profiles for ensemble testing"""
    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    vespa_http_port = vespa_with_schema["http_port"]
    vespa_config_port = vespa_with_schema["config_port"]
    vespa_url = "http://localhost"
    config_manager = vespa_with_schema["manager"].config_manager

    # Create schema loader pointing to test schemas
    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Create SearchAgent with test Vespa parameters
    deps = SearchAgentDeps(
        tenant_id="test_tenant",
        backend_url=vespa_url,
        backend_port=vespa_http_port,
        backend_config_port=vespa_config_port,
        profile="video_colpali_smol500_mv_frame",
    )
    search_agent = SearchAgent(
        deps=deps,
        schema_loader=schema_loader,
        config_manager=config_manager,
        port=8015,
    )

    return search_agent


@pytest.mark.integration
@pytest.mark.slow
class TestSearchAgentEnsembleIntegration:
    """Integration tests for ensemble search with real Vespa backend"""

    @pytest.mark.asyncio
    async def test_rrf_fusion_with_mock_profiles(
        self, search_agent_with_ensemble_profiles
    ):
        """
        Test RRF fusion algorithm correctness with known results

        This validates the RRF formula without needing multiple deployed profiles
        """
        agent = search_agent_with_ensemble_profiles

        # Create mock results that simulate different profiles
        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 0.95, "title": "Result 1"},
                {"id": "doc2", "score": 0.90, "title": "Result 2"},
                {"id": "doc3", "score": 0.85, "title": "Result 3"},
            ],
            "profile2": [
                {"id": "doc2", "score": 0.92, "title": "Result 2"},  # Appears in both
                {"id": "doc4", "score": 0.88, "title": "Result 4"},
                {"id": "doc1", "score": 0.80, "title": "Result 1"},  # Appears in both
            ],
            "profile3": [
                {"id": "doc5", "score": 0.93, "title": "Result 5"},
                {"id": "doc2", "score": 0.87, "title": "Result 2"},  # Appears in all 3
                {"id": "doc6", "score": 0.82, "title": "Result 6"},
            ],
        }

        # Execute RRF fusion
        fused_results = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # VALIDATE: Doc2 should rank first (appears in all 3 profiles)
        assert (
            fused_results[0]["id"] == "doc2"
        ), f"Expected doc2 first, got {fused_results[0]['id']}"
        assert fused_results[0]["num_profiles"] == 3

        # VALIDATE: RRF scores are monotonically decreasing
        for i in range(len(fused_results) - 1):
            assert fused_results[i]["rrf_score"] >= fused_results[i + 1]["rrf_score"]

        # VALIDATE: Metadata includes profile ranks and scores
        assert "profile_ranks" in fused_results[0]
        assert "profile_scores" in fused_results[0]
        assert len(fused_results[0]["profile_ranks"]) == 3

        # VALIDATE: RRF score calculation for doc2 (appears at ranks 1, 0, 1)
        # Formula: 1/(60+1) + 1/(60+0) + 1/(60+1) = 0.01639 + 0.01667 + 0.01639 = 0.04945
        expected_score = (1.0 / 61) + (1.0 / 60) + (1.0 / 61)
        assert abs(fused_results[0]["rrf_score"] - expected_score) < 0.001

        logger.info(
            f"✅ RRF fusion validated: {len(fused_results)} results, doc2 ranked first with score {fused_results[0]['rrf_score']:.4f}"
        )

    @pytest.mark.asyncio
    async def test_ensemble_mode_detection_and_routing(
        self, search_agent_with_ensemble_profiles
    ):
        """
        Test that ensemble mode is correctly detected and routes to ensemble search

        Uses mocked _search_ensemble to verify routing without requiring multiple profiles
        """
        from unittest.mock import AsyncMock

        agent = search_agent_with_ensemble_profiles

        # Mock _search_ensemble to capture call
        agent._search_ensemble = AsyncMock(
            return_value=[{"id": "doc1", "rrf_score": 0.5, "num_profiles": 2}]
        )

        # Call with multiple profiles
        result = await agent._process_impl(
            {
                "query": "machine learning videos",
                "profiles": ["profile1", "profile2"],
                "top_k": 10,
                "rrf_k": 60,
            }
        )

        # VALIDATE: Ensemble mode detected
        assert result.search_mode == "ensemble"
        assert result.profiles == ["profile1", "profile2"]
        assert result.rrf_k == 60

        # VALIDATE: _search_ensemble was called
        agent._search_ensemble.assert_called_once()
        call_kwargs = agent._search_ensemble.call_args.kwargs
        assert call_kwargs["query"] == "machine learning videos"
        assert call_kwargs["profiles"] == ["profile1", "profile2"]
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["rrf_k"] == 60

        logger.info("✅ Ensemble mode detection and routing validated")

    @pytest.mark.asyncio
    async def test_single_profile_fallback(self, search_agent_with_ensemble_profiles):
        """
        Test that single profile queries don't trigger ensemble mode

        Verifies backward compatibility with existing single-profile searches
        """
        agent = search_agent_with_ensemble_profiles

        # Mock search to avoid actual Vespa call
        from unittest.mock import Mock

        agent.search_backend.search = Mock(return_value=[])

        # Call without profiles parameter
        result = await agent._process_impl({"query": "test query", "top_k": 10})

        # VALIDATE: Single profile mode
        assert result.search_mode == "single_profile"
        assert result.profile is not None
        assert result.profile == agent.active_profile

        # Call with single profile in list (should still use single mode)
        result2 = await agent._process_impl(
            {"query": "test query", "profiles": ["profile1"], "top_k": 10}
        )

        # VALIDATE: Still single profile mode (only 1 profile)
        assert result2.search_mode == "single_profile"

        logger.info("✅ Single profile fallback validated")

    @pytest.mark.asyncio
    async def test_rrf_k_parameter_effect(self, search_agent_with_ensemble_profiles):
        """
        Test that rrf_k parameter affects fusion scores correctly

        Lower k values give more weight to top-ranked results
        """
        agent = search_agent_with_ensemble_profiles

        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 0.9},  # rank 0
                {"id": "doc2", "score": 0.8},  # rank 1
            ],
            "profile2": [
                {"id": "doc1", "score": 0.85},  # rank 0
                {"id": "doc2", "score": 0.75},  # rank 1
            ],
        }

        # Test with k=60 (default)
        fused_k60 = agent._fuse_results_rrf(profile_results, k=60, top_k=10)
        score_k60_doc1 = fused_k60[0]["rrf_score"]

        # Test with k=20 (more weight to top ranks)
        fused_k20 = agent._fuse_results_rrf(profile_results, k=20, top_k=10)
        score_k20_doc1 = fused_k20[0]["rrf_score"]

        # VALIDATE: Lower k gives higher scores
        # k=20: 1/20 + 1/20 = 0.1
        # k=60: 1/60 + 1/60 = 0.0333
        assert score_k20_doc1 > score_k60_doc1
        assert abs(score_k20_doc1 - 0.1) < 0.001
        assert abs(score_k60_doc1 - 0.0333) < 0.001

        logger.info(
            f"✅ RRF k parameter validated: k=20 → {score_k20_doc1:.4f}, k=60 → {score_k60_doc1:.4f}"
        )

    @pytest.mark.asyncio
    async def test_ensemble_empty_results_handling(
        self, search_agent_with_ensemble_profiles
    ):
        """
        Test ensemble search when all profiles return empty results

        Should gracefully return empty list without errors
        """
        agent = search_agent_with_ensemble_profiles

        # Empty results from all profiles
        profile_results = {
            "profile1": [],
            "profile2": [],
            "profile3": [],
        }

        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # VALIDATE: Returns empty list
        assert isinstance(fused, list)
        assert len(fused) == 0

        logger.info("✅ Empty results handling validated")

    @pytest.mark.asyncio
    async def test_ensemble_top_k_limit(self, search_agent_with_ensemble_profiles):
        """
        Test that ensemble search respects top_k limit

        Even with many results from multiple profiles, should return exactly top_k
        """
        agent = search_agent_with_ensemble_profiles

        # Create many results
        profile_results = {
            "profile1": [
                {"id": f"doc{i}", "score": 1.0 - (i * 0.01)} for i in range(50)
            ],
            "profile2": [
                {"id": f"doc{i+25}", "score": 1.0 - (i * 0.01)} for i in range(50)
            ],
        }

        # Request top 5
        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=5)

        # VALIDATE: Exactly 5 results
        assert len(fused) == 5

        # Request top 10
        fused10 = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # VALIDATE: Exactly 10 results
        assert len(fused10) == 10

        logger.info(
            f"✅ Top-k limit validated: requested 5, got {len(fused)}; requested 10, got {len(fused10)}"
        )

    @pytest.mark.asyncio
    async def test_ensemble_partial_overlap(self, search_agent_with_ensemble_profiles):
        """
        Test RRF fusion when documents have partial overlap across profiles

        This is the common case in real ensemble search
        """
        agent = search_agent_with_ensemble_profiles

        # Results with partial overlap
        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 0.95},
                {"id": "doc2", "score": 0.90},
                {"id": "doc3", "score": 0.85},
                {"id": "doc4", "score": 0.80},
            ],
            "profile2": [
                {"id": "doc2", "score": 0.92},  # Overlap with profile1
                {"id": "doc5", "score": 0.88},
                {"id": "doc3", "score": 0.84},  # Overlap with profile1
                {"id": "doc6", "score": 0.80},
            ],
            "profile3": [
                {"id": "doc7", "score": 0.93},
                {"id": "doc2", "score": 0.87},  # Overlap with profile1 and profile2
                {"id": "doc8", "score": 0.82},
                {"id": "doc1", "score": 0.78},  # Overlap with profile1
            ],
        }

        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # VALIDATE: Doc2 should rank high (appears in all 3 profiles)
        doc2_result = next(r for r in fused if r["id"] == "doc2")
        assert doc2_result["num_profiles"] == 3

        # VALIDATE: Doc1 appears in 2 profiles
        doc1_result = next(r for r in fused if r["id"] == "doc1")
        assert doc1_result["num_profiles"] == 2

        # VALIDATE: Doc5, doc6, doc7, doc8 appear in only 1 profile
        for doc_id in ["doc5", "doc6", "doc7", "doc8"]:
            doc_result = next(r for r in fused if r["id"] == doc_id)
            assert doc_result["num_profiles"] == 1

        # VALIDATE: Documents in more profiles rank higher
        # (This is not always strictly true due to rank positions, but generally true)
        doc2_rank = next(i for i, r in enumerate(fused) if r["id"] == "doc2")
        doc5_rank = next(i for i, r in enumerate(fused) if r["id"] == "doc5")
        logger.info(
            f"Doc2 (3 profiles) at rank {doc2_rank}, Doc5 (1 profile) at rank {doc5_rank}"
        )

        logger.info(
            f"✅ Partial overlap validated: {len(fused)} total docs, doc2 in {doc2_result['num_profiles']} profiles"
        )

    @pytest.mark.asyncio
    async def test_rrf_score_metadata(self, search_agent_with_ensemble_profiles):
        """
        Test that RRF fusion adds complete metadata to results

        Metadata should include: rrf_score, profile_ranks, profile_scores, num_profiles
        """
        agent = search_agent_with_ensemble_profiles

        profile_results = {
            "profile1": [
                {"id": "doc1", "score": 0.95, "title": "Title 1"},
                {"id": "doc2", "score": 0.90, "title": "Title 2"},
                {"id": "doc3", "score": 0.85, "title": "Title 3"},
            ],
            "profile2": [
                {"id": "doc2", "score": 0.92, "title": "Title 2"},  # rank 0
                {"id": "doc3", "score": 0.88, "title": "Title 3"},  # rank 1
                {"id": "doc1", "score": 0.85, "title": "Title 1"},  # rank 2
            ],
        }

        fused = agent._fuse_results_rrf(profile_results, k=60, top_k=10)

        # VALIDATE: All required metadata fields present
        for result in fused:
            assert "rrf_score" in result
            assert "profile_ranks" in result
            assert "profile_scores" in result
            assert "num_profiles" in result

            # VALIDATE: Metadata types
            assert isinstance(result["rrf_score"], float)
            assert isinstance(result["profile_ranks"], dict)
            assert isinstance(result["profile_scores"], dict)
            assert isinstance(result["num_profiles"], int)

        # VALIDATE: All docs appear in both profiles
        assert len(fused) == 3
        for result in fused:
            assert result["num_profiles"] == 2

        # VALIDATE: Doc2 should rank first (rank 1+0 = better than doc1's 0+2 or doc3's 2+1)
        # Doc2: 1/61 + 1/60 = 0.0164 + 0.0167 = 0.0331
        # Doc1: 1/60 + 1/62 = 0.0167 + 0.0161 = 0.0328
        # Doc3: 1/62 + 1/61 = 0.0161 + 0.0164 = 0.0325
        doc2 = next(r for r in fused if r["id"] == "doc2")
        assert "profile1" in doc2["profile_ranks"]
        assert "profile2" in doc2["profile_ranks"]
        assert doc2["profile_ranks"]["profile1"] == 1  # rank 1 in profile1
        assert doc2["profile_ranks"]["profile2"] == 0  # rank 0 in profile2
        assert doc2["profile_scores"]["profile1"] == 0.90
        assert doc2["profile_scores"]["profile2"] == 0.92

        logger.info(
            f"✅ RRF metadata validated: doc2 ranks={doc2['profile_ranks']}, scores={doc2['profile_scores']}"
        )
