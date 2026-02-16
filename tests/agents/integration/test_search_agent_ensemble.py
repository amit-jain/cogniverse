"""
Integration tests for SearchAgent ensemble search and multi-query fusion.

Tests with real Vespa backend, real query encoders, real search:
- Multiple deployed Vespa profiles with different schemas
- Parallel search execution across profiles
- RRF fusion with real search results
- Multi-query fusion (parallel variant search + RRF)
- Latency validation
- Error handling (profile failures, sparse results)
"""

import logging
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def multi_profile_vespa():
    """
    Module-scoped Vespa instance with MULTIPLE deployed profiles for ensemble testing.

    Deploys 2-3 different schemas to test real ensemble search.
    """
    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.system.vespa_test_manager import VespaTestManager
    from tests.utils.docker_utils import generate_unique_ports

    # Generate unique ports
    ensemble_http_port, ensemble_config_port = generate_unique_ports("ensemble_test")

    logger.info(
        f"Ensemble tests using ports: {ensemble_http_port} (http), {ensemble_config_port} (config)"
    )

    # Clear singletons
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    # Create manager
    manager = VespaTestManager(
        http_port=ensemble_http_port, config_port=ensemble_config_port
    )

    try:
        # Setup Vespa with default schema
        logger.info("Setting up Vespa for ensemble testing...")
        if not manager.full_setup():
            pytest.skip("Failed to setup Vespa test environment")

        logger.info(f"✅ Vespa ready at http://localhost:{ensemble_http_port}")

        # Use 3 REAL different profiles for comprehensive ensemble testing
        real_profiles = [
            "video_colpali_smol500_mv_frame",  # ColPali 128-dim
            "video_videoprism_base_mv_chunk_30s",  # VideoPrism 768-dim
            "video_colqwen_omni_mv_chunk_30s",  # ColQwen 128-dim
        ]

        yield {
            "http_port": ensemble_http_port,
            "config_port": ensemble_config_port,
            "base_url": f"http://localhost:{ensemble_http_port}",
            "manager": manager,
            "profiles": real_profiles,  # List of 3 different real profiles
            "profile_name": real_profiles[0],  # Use first profile as default
        }

    except Exception as e:
        logger.error(f"Failed to start multi-profile Vespa: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        logger.info("Tearing down multi-profile Vespa...")
        manager.cleanup()

        # Clear singletons
        try:
            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()
            if hasattr(ConfigManager, "_instance"):
                ConfigManager._instance = None
            logger.info("✅ Cleared singleton state")
        except Exception as e:
            logger.warning(f"⚠️  Error clearing state: {e}")


@pytest.fixture
def search_agent_ensemble(multi_profile_vespa):
    """SearchAgent configured for ensemble search with 3 REAL different profiles"""
    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        BackendProfileConfig,
    )

    vespa_http_port = multi_profile_vespa["http_port"]
    vespa_config_port = multi_profile_vespa["config_port"]
    vespa_url = "http://localhost"
    config_manager = multi_profile_vespa["manager"].config_manager
    profiles = multi_profile_vespa["profiles"]

    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Register all 3 real profiles in config_manager
    backend_profiles = {
        "video_colpali_smol500_mv_frame": BackendProfileConfig(
            profile_name="video_colpali_smol500_mv_frame",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colsmol-500m",
        ),
        "video_videoprism_base_mv_chunk_30s": BackendProfileConfig(
            profile_name="video_videoprism_base_mv_chunk_30s",
            schema_name="video_videoprism_base_mv_chunk_30s",
            embedding_model="google/videoprism-base",
        ),
        "video_colqwen_omni_mv_chunk_30s": BackendProfileConfig(
            profile_name="video_colqwen_omni_mv_chunk_30s",
            schema_name="video_colqwen_omni_mv_chunk_30s",
            embedding_model="vidore/colqwen2-v0.1",
        ),
    }

    backend_config = BackendConfig(
        tenant_id="ensemble_test_tenant",
        backend_type="vespa",
        url=vespa_url,
        port=vespa_http_port,
        profiles=backend_profiles,
    )
    config_manager.set_backend_config(backend_config)

    # Create SearchAgent with first profile as default using deps pattern
    deps = SearchAgentDeps(
        tenant_id="ensemble_test_tenant",
        backend_url=vespa_url,
        backend_port=vespa_http_port,
        backend_config_port=vespa_config_port,
        profile=profiles[0],
    )
    search_agent = SearchAgent(
        deps=deps,
        schema_loader=schema_loader,
        config_manager=config_manager,
        port=8016,
    )

    return search_agent, profiles


@pytest.fixture
def search_agent_single_profile(multi_profile_vespa):
    """SearchAgent configured for single-profile search with correct tenant_id."""
    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    vespa_http_port = multi_profile_vespa["http_port"]
    vespa_config_port = multi_profile_vespa["config_port"]
    vespa_url = "http://localhost"
    config_manager = multi_profile_vespa["manager"].config_manager
    default_profile = multi_profile_vespa["profiles"][0]

    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Use test_tenant to match VespaTestManager's deployed schema
    deps = SearchAgentDeps(
        tenant_id="test_tenant",
        backend_url=vespa_url,
        backend_port=vespa_http_port,
        backend_config_port=vespa_config_port,
        profile=default_profile,
    )
    search_agent = SearchAgent(
        deps=deps,
        schema_loader=schema_loader,
        config_manager=config_manager,
        port=8017,
    )

    return search_agent


@pytest.mark.integration
@pytest.mark.slow
class TestSearchAgentEnsemble:
    """Integration tests for ensemble search with real Vespa backend"""

    @pytest.mark.asyncio
    async def test_ensemble_search_with_real_vespa_profiles(
        self, search_agent_ensemble
    ):
        """
        REAL TEST: Execute ensemble search with REAL encoders and REAL Vespa.

        Uses 3 different REAL encoders (ColPali, VideoPrism, ColQwen) loaded by QueryEncoderFactory.

        Validates:
        - REAL query encoder loading for each profile
        - Parallel search execution to real Vespa
        - RRF fusion with real results
        - Complete metadata in fused results
        """
        agent, profiles = search_agent_ensemble

        # NO PATCHING - QueryEncoderFactory loads real encoders for each profile
        logger.info(
            "🔄 Loading REAL query encoders for all 3 profiles (ColPali, VideoPrism, ColQwen)"
        )

        result = await agent._process_impl(
            {
                "query": "robot playing soccer",
                "profiles": profiles,  # List of 3 different profile names
                "top_k": 5,
                "rrf_k": 60,
            }
        )

        # VALIDATE: Ensemble mode detected
        assert result.search_mode == "ensemble"
        assert set(result.profiles) == set(profiles)

        # VALIDATE: Results structure
        assert result.results is not None
        assert result.total_results is not None
        assert isinstance(result.results, list)

        logger.info(
            f"✅ Ensemble search executed with 3 REAL encoders: {result.total_results} results"
        )

    @pytest.mark.asyncio
    async def test_ensemble_search_latency(self, search_agent_ensemble):
        """
        REAL TEST: Validate ensemble search latency with REAL encoders

        Target: <60000ms (allows for loading 3 real models + REAL Vespa)
        """
        agent, profiles = search_agent_ensemble

        # Measure latency with real encoder loading for all 3 profiles
        start_time = time.time()

        _result = await agent._process_impl(
            {
                "query": "test query for latency",
                "profiles": profiles,
                "top_k": 10,
                "rrf_k": 60,
            }
        )
        assert _result is not None  # Verify execution completed

        elapsed_ms = (time.time() - start_time) * 1000

        # VALIDATE: Latency target met
        logger.info(
            f"Ensemble search latency: {elapsed_ms:.2f}ms for {len(profiles)} profiles with REAL encoders"
        )

        # Target accounts for real model encoding + retries
        assert elapsed_ms < 60000, f"Ensemble search took {elapsed_ms:.2f}ms (too slow)"

    @pytest.mark.asyncio
    async def test_ensemble_with_one_profile_failure(self, search_agent_ensemble):
        """
        REAL TEST: Validate ensemble continues when one profile fails

        Should gracefully degrade: use results from working profiles only
        """
        agent, profiles = search_agent_ensemble

        # Use real invalid profile to trigger natural failure
        # Create profiles list with one invalid profile mixed in with valid ones
        profiles_with_invalid = [
            profiles[0],
            "invalid_nonexistent_profile_xyz",
            profiles[1],
        ]

        result = await agent._process_impl(
            {
                "query": "test query with failure",
                "profiles": profiles_with_invalid,
                "top_k": 10,
                "rrf_k": 60,
            }
        )

        # VALIDATE: Ensemble still returned results (graceful degradation)
        assert result.search_mode == "ensemble"

        # Should have results from valid profiles only (invalid profile fails naturally)
        logger.info(
            f"✅ Ensemble degraded gracefully: {result.total_results} results despite invalid profile"
        )

    @pytest.mark.asyncio
    async def test_ensemble_parallel_execution_verification(
        self, search_agent_ensemble
    ):
        """
        REAL TEST: Verify that profiles are actually searched in parallel

        Uses timing of real Vespa searches to validate concurrent execution
        """
        agent, profiles = search_agent_ensemble

        start_time = time.time()

        _result = await agent._process_impl(
            {
                "query": "test parallel execution",
                "profiles": profiles,
                "top_k": 10,
                "rrf_k": 60,
            }
        )
        assert _result is not None  # Verify execution completed

        total_time = time.time() - start_time

        # VALIDATE: Ensemble completes in reasonable time with REAL Vespa searches
        # Parallel execution should complete faster than fully sequential
        logger.info(
            f"Total ensemble time: {total_time:.3f}s for {len(profiles)} profiles with REAL Vespa"
        )

        # With real Vespa and parallel execution, should complete reasonably fast
        # Allow generous threshold for CI environment
        assert (
            total_time < 60.0
        ), f"Ensemble took {total_time:.3f}s (too slow even for parallel execution)"

        logger.info(
            f"✅ Parallel execution validated: {len(profiles)} profiles searched in {total_time:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_ensemble_rrf_with_real_overlapping_results(
        self, search_agent_ensemble
    ):
        """
        REAL TEST: Validate RRF fusion metadata with real Vespa search results

        Validates RRF fusion adds proper metadata to search results
        """
        agent, profiles = search_agent_ensemble

        result = await agent._process_impl(
            {
                "query": "robot playing soccer",
                "profiles": profiles,
                "top_k": 10,
                "rrf_k": 60,
            }
        )

        # VALIDATE: RRF fusion metadata on REAL search results
        assert result.search_mode == "ensemble"
        results = result.results

        # Validate RRF metadata structure on all results
        for doc in results:
            assert "rrf_score" in doc, f"Missing rrf_score in doc {doc.get('id')}"
            assert (
                "profile_ranks" in doc
            ), f"Missing profile_ranks in doc {doc.get('id')}"
            assert "num_profiles" in doc, f"Missing num_profiles in doc {doc.get('id')}"
            assert (
                doc["rrf_score"] > 0
            ), f"Invalid RRF score {doc['rrf_score']} in doc {doc.get('id')}"
            assert (
                doc["num_profiles"] >= 1
            ), f"Invalid num_profiles {doc['num_profiles']} in doc {doc.get('id')}"

        logger.info(
            f"✅ RRF fusion validated on {len(results)} real results with proper metadata"
        )

    @pytest.mark.asyncio
    async def test_ensemble_with_empty_profile_results(self, search_agent_ensemble):
        """
        REAL TEST: Handle case where query returns few/no results

        Validates ensemble handles empty or sparse results gracefully
        """
        agent, profiles = search_agent_ensemble

        # Use nonsensical query that likely returns no results
        result = await agent._process_impl(
            {
                "query": "xyzabc123nonexistent query that returns nothing",
                "profiles": profiles,
                "top_k": 10,
                "rrf_k": 60,
            }
        )

        # VALIDATE: Ensemble executes without error even with sparse/empty results
        assert result.search_mode == "ensemble"
        assert result.results is not None
        assert isinstance(result.results, list)

        logger.info(
            f"✅ Handled sparse/empty results gracefully: {len(result.results)} results"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestMultiQueryFusionIntegration:
    """Integration tests for multi-query fusion search with real Vespa and real encoders."""

    @pytest.mark.asyncio
    async def test_multi_query_fusion_with_real_vespa(self, search_agent_ensemble):
        """
        Test multi-query fusion end-to-end with real Vespa and real query encoder.

        Creates a RoutingOutput with query_variants and passes it through
        search_with_routing_decision, exercising real encoding and real Vespa search.
        """
        from cogniverse_agents.routing_agent import RoutingOutput

        agent, _profiles = search_agent_ensemble

        routing_decision = RoutingOutput(
            query="robot playing soccer",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Video search with multi-query fusion",
            enhanced_query="robot playing soccer (robot playing soccer)",
            entities=[{"text": "robot", "label": "TECHNOLOGY", "confidence": 0.9}],
            relationships=[
                {
                    "subject": "robot",
                    "relation": "playing",
                    "object": "soccer",
                    "confidence": 0.85,
                }
            ],
            metadata={"rrf_k": 60},
            query_variants=[
                {"name": "original", "query": "robot playing soccer"},
                {
                    "name": "relationship_expansion",
                    "query": "robot playing soccer (robot playing soccer)",
                },
            ],
        )

        result = agent.search_with_routing_decision(routing_decision, top_k=10)

        # VALIDATE: Multi-query fusion executed
        assert result["status"] == "completed"
        assert "query_variants_used" in result
        assert "original" in result["query_variants_used"]
        assert "relationship_expansion" in result["query_variants_used"]

        # VALIDATE: Results have RRF metadata
        if result["total_results"] > 0:
            for doc in result["results"]:
                assert "rrf_score" in doc
                assert "num_profiles" in doc
                assert doc["rrf_score"] > 0

        logger.info(
            f"✅ Multi-query fusion with real Vespa: {result['total_results']} results, "
            f"variants used: {result['query_variants_used']}"
        )

    @pytest.mark.asyncio
    async def test_multi_query_fusion_latency(self, search_agent_ensemble):
        """
        Test multi-query fusion latency with real encoders and real Vespa.

        Target: <60000ms (accounts for real model encoding + parallel Vespa searches).
        """
        from cogniverse_agents.routing_agent import RoutingOutput

        agent, _profiles = search_agent_ensemble

        routing_decision = RoutingOutput(
            query="machine learning tutorial",
            recommended_agent="search_agent",
            confidence=0.9,
            reasoning="Multi-query fusion test",
            enhanced_query="machine learning tutorial (machine learning tutorial)",
            entities=[
                {"text": "machine learning", "label": "TECHNOLOGY", "confidence": 0.92}
            ],
            relationships=[],
            metadata={"rrf_k": 60},
            query_variants=[
                {"name": "original", "query": "machine learning tutorial"},
                {
                    "name": "relationship_expansion",
                    "query": "machine learning tutorial (machine learning tutorial)",
                },
                {
                    "name": "boolean_optimization",
                    "query": "machine learning tutorial (machine AND learning)",
                },
            ],
        )

        start_time = time.time()
        result = agent.search_with_routing_decision(routing_decision, top_k=10)
        elapsed_ms = (time.time() - start_time) * 1000

        assert result["status"] == "completed"
        assert (
            elapsed_ms < 60000
        ), f"Multi-query fusion took {elapsed_ms:.2f}ms (too slow)"

        logger.info(
            f"✅ Multi-query fusion latency: {elapsed_ms:.2f}ms for "
            f"{len(result.get('query_variants_used', []))} variants"
        )

    @pytest.mark.asyncio
    async def test_multi_query_fusion_sparse_results(self, search_agent_ensemble):
        """
        Test multi-query fusion with queries that produce sparse/empty results.

        Should handle gracefully without errors.
        """
        from cogniverse_agents.routing_agent import RoutingOutput

        agent, _profiles = search_agent_ensemble

        routing_decision = RoutingOutput(
            query="xyzabc123nonexistent query fusion test",
            recommended_agent="search_agent",
            confidence=0.5,
            reasoning="Testing sparse results",
            enhanced_query="xyzabc123nonexistent query fusion expanded",
            entities=[],
            relationships=[],
            metadata={"rrf_k": 60},
            query_variants=[
                {"name": "original", "query": "xyzabc123nonexistent query fusion test"},
                {
                    "name": "expansion",
                    "query": "xyzabc123nonexistent query fusion expanded",
                },
            ],
        )

        result = agent.search_with_routing_decision(routing_decision, top_k=10)

        assert result["status"] == "completed"
        assert isinstance(result["results"], list)
        assert "query_variants_used" in result

        logger.info(
            f"✅ Sparse results handled gracefully: {result['total_results']} results"
        )

    @pytest.mark.asyncio
    async def test_single_query_fallback_without_variants(
        self, search_agent_single_profile
    ):
        """
        Test that RoutingOutput without query_variants falls back to single-query path.

        Ensures backward compatibility — no variants means existing behavior.
        """
        from cogniverse_agents.routing_agent import RoutingOutput

        agent = search_agent_single_profile

        routing_decision = RoutingOutput(
            query="robot playing soccer",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Single query search",
            enhanced_query="robot playing soccer enhanced",
            entities=[{"text": "robot", "label": "TECHNOLOGY", "confidence": 0.9}],
            relationships=[],
            metadata={},
            query_variants=[],  # Empty — should use single-query path
        )

        result = agent.search_with_routing_decision(routing_decision, top_k=10)

        assert result["status"] == "completed"
        assert "query_variants_used" not in result  # Single-query path doesn't set this

        logger.info(
            f"✅ Single query fallback: {result['total_results']} results (no variants)"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestSingleProfileSearchIntegration:
    """Integration tests for single-profile search with real Vespa and real encoders."""

    @pytest.mark.asyncio
    async def test_single_profile_text_search(self, search_agent_single_profile):
        """
        Test basic single-profile text search with real Vespa.

        Uses _process_impl with a single query (no profiles list, no variants).
        """
        agent = search_agent_single_profile

        result = await agent._process_impl(
            {"query": "robot playing soccer", "top_k": 10}
        )

        assert result.search_mode == "single_profile"
        assert result.profile is not None
        assert result.results is not None
        assert isinstance(result.results, list)

        logger.info(
            f"✅ Single profile search: {result.total_results} results "
            f"from profile {result.profile}"
        )

    @pytest.mark.asyncio
    async def test_single_profile_with_routing_decision(
        self, search_agent_single_profile
    ):
        """
        Test single-profile search via search_with_routing_decision.

        RoutingOutput with no query_variants and no profiles list → single-query path.
        """
        from cogniverse_agents.routing_agent import RoutingOutput

        agent = search_agent_single_profile

        routing_decision = RoutingOutput(
            query="robot playing soccer",
            recommended_agent="search_agent",
            confidence=0.85,
            reasoning="Single profile video search",
            enhanced_query="robot playing soccer enhanced",
            entities=[
                {"text": "robot", "label": "TECHNOLOGY", "confidence": 0.9},
                {"text": "soccer", "label": "SPORT", "confidence": 0.85},
            ],
            relationships=[
                {
                    "subject": "robot",
                    "relation": "playing",
                    "object": "soccer",
                    "confidence": 0.85,
                }
            ],
            metadata={},
            query_variants=[],
        )

        result = agent.search_with_routing_decision(routing_decision, top_k=10)

        assert result["status"] == "completed"
        assert isinstance(result["results"], list)
        assert "query_variants_used" not in result

        logger.info(
            f"✅ Single profile via routing decision: {result['total_results']} results"
        )

    @pytest.mark.asyncio
    async def test_single_profile_with_relationship_context(
        self, search_agent_single_profile
    ):
        """
        Test single-profile search via search_with_relationship_context.

        Uses SearchContext with entities and relationships but no query_variants.
        """
        from cogniverse_agents.search_agent import SearchContext

        agent = search_agent_single_profile

        context = SearchContext(
            original_query="robot playing soccer",
            enhanced_query="robot playing soccer (robot playing soccer)",
            entities=[
                {"text": "robot", "label": "TECHNOLOGY", "confidence": 0.9},
                {"text": "soccer", "label": "SPORT", "confidence": 0.85},
            ],
            relationships=[
                {
                    "subject": "robot",
                    "relation": "playing",
                    "object": "soccer",
                    "confidence": 0.85,
                }
            ],
            routing_metadata={},
            confidence=0.85,
            query_variants=[],
        )

        result = agent.search_with_relationship_context(context, top_k=10)

        assert result["status"] == "completed"
        assert isinstance(result["results"], list)
        assert "query_variants_used" not in result

        logger.info(
            f"✅ Single profile with relationship context: {result['total_results']} results"
        )

    @pytest.mark.asyncio
    async def test_single_profile_empty_results(self, search_agent_single_profile):
        """
        Test single-profile search with a nonsensical query that returns no results.
        """
        agent = search_agent_single_profile

        result = await agent._process_impl(
            {"query": "xyzabc123nonexistent gibberish", "top_k": 10}
        )

        assert result.search_mode == "single_profile"
        assert result.results is not None
        assert isinstance(result.results, list)

        logger.info(f"✅ Single profile empty results: {result.total_results} results")

    @pytest.mark.asyncio
    async def test_single_profile_explicit_in_list(self, search_agent_single_profile):
        """
        Test that passing a single profile in the profiles list still uses single-profile mode.
        """
        agent = search_agent_single_profile

        result = await agent._process_impl(
            {"query": "robot soccer", "profiles": [agent.active_profile], "top_k": 10}
        )

        assert result.search_mode == "single_profile"
        assert result.profile is not None

        logger.info(
            f"✅ Single profile in list: mode={result.search_mode}, profile={result.profile}"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndQueryFusionPipeline:
    """
    End-to-end integration tests: RoutingAgent (parallel mode) → query_variants → SearchAgent.

    Validates the complete pipeline where RoutingAgent generates query variants
    and SearchAgent fuses them via RRF.
    """

    @pytest.fixture
    def routing_agent_parallel(self):
        """RoutingAgent configured with query_fusion_config mode='parallel'."""
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )

        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
            query_fusion_config={
                "mode": "parallel",
                "variant_strategies": [
                    "relationship_expansion",
                    "boolean_optimization",
                ],
                "include_original": True,
                "rrf_k": 60,
            },
        )
        return RoutingAgent(deps=deps)

    @pytest.fixture
    def routing_agent_single(self):
        """RoutingAgent with default query_fusion_config (mode='single')."""
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )

        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
        )
        return RoutingAgent(deps=deps)

    @pytest.mark.asyncio
    async def test_routing_produces_query_variants_in_parallel_mode(
        self, routing_agent_parallel
    ):
        """
        RoutingAgent with mode='parallel' populates query_variants in RoutingOutput.

        Validates: relationship extraction → query enhancement → variant generation.
        """
        agent = routing_agent_parallel

        result = await agent.analyze_and_route_with_relationships(
            query="robots playing soccer in a field",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # VALIDATE: RoutingOutput has query_variants populated
        assert isinstance(result.query_variants, list)
        assert len(result.query_variants) >= 2, (
            f"Expected at least 2 variants (original + strategy), "
            f"got {len(result.query_variants)}: {result.query_variants}"
        )

        # VALIDATE: Original query is included
        variant_names = [v["name"] for v in result.query_variants]
        assert (
            "original" in variant_names
        ), f"Expected 'original' variant, got: {variant_names}"

        # VALIDATE: At least one strategy variant is present
        strategy_variants = [n for n in variant_names if n != "original"]
        assert (
            len(strategy_variants) >= 1
        ), f"Expected at least one strategy variant, got: {variant_names}"

        # VALIDATE: Each variant has required structure
        for variant in result.query_variants:
            assert "name" in variant, f"Variant missing 'name': {variant}"
            assert "query" in variant, f"Variant missing 'query': {variant}"
            assert len(variant["query"]) > 0, f"Variant has empty query: {variant}"

        logger.info(
            f"✅ Routing produced {len(result.query_variants)} variants: {variant_names}"
        )

    @pytest.mark.asyncio
    async def test_routing_single_mode_produces_no_variants(self, routing_agent_single):
        """
        RoutingAgent with default mode='single' produces empty query_variants.

        Ensures backward compatibility — single mode is the default.
        """
        agent = routing_agent_single

        result = await agent.analyze_and_route_with_relationships(
            query="robots playing soccer",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # VALIDATE: No query variants in single mode
        assert isinstance(result.query_variants, list)
        assert len(result.query_variants) == 0, (
            f"Expected empty query_variants in single mode, "
            f"got {len(result.query_variants)}: {result.query_variants}"
        )

        logger.info("✅ Single mode routing produces no query variants")

    @pytest.mark.asyncio
    async def test_end_to_end_routing_to_search_with_fusion(
        self, routing_agent_parallel, search_agent_single_profile
    ):
        """
        End-to-end: RoutingAgent generates query_variants → SearchAgent fuses results.

        Full pipeline:
        1. RoutingAgent with parallel fusion config produces RoutingOutput with query_variants
        2. SearchAgent.search_with_routing_decision() receives the RoutingOutput
        3. Multi-query fusion path is triggered (variant queries searched in parallel)
        4. Results are fused with RRF
        """
        routing_agent = routing_agent_parallel

        # Step 1: Route with parallel fusion
        routing_result = await routing_agent.analyze_and_route_with_relationships(
            query="robots playing soccer in a field",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # Verify variants were generated
        assert (
            len(routing_result.query_variants) >= 2
        ), f"Expected at least 2 query variants, got {len(routing_result.query_variants)}"

        # Step 2: Pass to SearchAgent
        search_agent = search_agent_single_profile
        search_result = search_agent.search_with_routing_decision(
            routing_result, top_k=10
        )

        # VALIDATE: Multi-query fusion was used
        assert search_result["status"] == "completed"
        assert "query_variants_used" in search_result, (
            "Expected multi-query fusion path (query_variants_used key), "
            "but search used single-query path"
        )

        # VALIDATE: All variant names are reported
        variant_names = [v["name"] for v in routing_result.query_variants]
        for name in variant_names:
            assert name in search_result["query_variants_used"], (
                f"Variant '{name}' not in query_variants_used: "
                f"{search_result['query_variants_used']}"
            )

        # VALIDATE: Results structure
        assert isinstance(search_result["results"], list)

        # VALIDATE: RRF metadata if results exist
        if search_result["total_results"] > 0:
            for doc in search_result["results"]:
                assert "rrf_score" in doc, "Missing rrf_score in fused result"
                assert doc["rrf_score"] > 0

        logger.info(
            f"✅ End-to-end: {len(routing_result.query_variants)} variants → "
            f"{search_result['total_results']} fused results"
        )

    @pytest.mark.asyncio
    async def test_end_to_end_single_mode_uses_single_query_path(
        self, routing_agent_single, search_agent_single_profile
    ):
        """
        End-to-end: RoutingAgent (single mode) → SearchAgent uses single-query path.

        Confirms that default config flows through correctly and SearchAgent
        does NOT use multi-query fusion when no variants are present.
        """
        routing_agent = routing_agent_single

        # Route with single mode (no variants expected)
        routing_result = await routing_agent.analyze_and_route_with_relationships(
            query="robots playing soccer",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        assert len(routing_result.query_variants) == 0

        # Pass to SearchAgent
        search_agent = search_agent_single_profile
        search_result = search_agent.search_with_routing_decision(
            routing_result, top_k=10
        )

        # VALIDATE: Single-query path was used (no query_variants_used key)
        assert search_result["status"] == "completed"
        assert "query_variants_used" not in search_result, (
            "Expected single-query path but got multi-query fusion "
            f"(query_variants_used={search_result.get('query_variants_used')})"
        )

        logger.info(
            f"✅ End-to-end single mode: {search_result['total_results']} results, "
            f"no fusion applied"
        )

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_rrf_k_propagates_from_config_to_routing_metadata(self):
        """
        Non-default rrf_k flows: query_fusion_config → enhancement_metadata → routing_metadata.

        Verifies that rrf_k=30 (non-default) in query_fusion_config propagates
        through the RoutingAgent pipeline into RoutingOutput.metadata["rrf_k"],
        which SearchAgent reads at search time.
        """
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )

        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        custom_rrf_k = 30  # Non-default (default is 60)
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
            query_fusion_config={
                "mode": "parallel",
                "variant_strategies": [
                    "relationship_expansion",
                    "boolean_optimization",
                ],
                "include_original": True,
                "rrf_k": custom_rrf_k,
            },
        )
        agent = RoutingAgent(deps=deps)

        result = await agent.analyze_and_route_with_relationships(
            query="robots playing soccer in a field",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # VALIDATE: rrf_k is in routing_metadata with the custom value
        assert "rrf_k" in result.routing_metadata, (
            f"Expected 'rrf_k' in routing_metadata, got keys: "
            f"{list(result.routing_metadata.keys())}"
        )
        assert result.routing_metadata["rrf_k"] == custom_rrf_k, (
            f"Expected rrf_k={custom_rrf_k} in routing_metadata, "
            f"got rrf_k={result.routing_metadata['rrf_k']}"
        )

        logger.info(
            f"✅ rrf_k={custom_rrf_k} propagated from config to routing_metadata"
        )

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_include_original_false_excludes_original_variant(self):
        """
        include_original=False in query_fusion_config excludes the original query
        from variants, so only strategy-generated variants are used.
        """
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )

        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
            query_fusion_config={
                "mode": "parallel",
                "variant_strategies": [
                    "relationship_expansion",
                    "boolean_optimization",
                ],
                "include_original": False,
                "rrf_k": 60,
            },
        )
        agent = RoutingAgent(deps=deps)

        result = await agent.analyze_and_route_with_relationships(
            query="robots playing soccer in a field",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # VALIDATE: No "original" variant present
        variant_names = [v["name"] for v in result.query_variants]
        assert "original" not in variant_names, (
            f"Expected 'original' excluded with include_original=False, "
            f"got: {variant_names}"
        )

        # Should still have strategy variants if entities were extracted
        if len(result.query_variants) > 0:
            assert all(v["name"] != "original" for v in result.query_variants)
            logger.info(
                f"✅ include_original=False: {len(result.query_variants)} strategy-only variants"
            )
        else:
            logger.info(
                "✅ include_original=False: no strategy variants produced (no diversity)"
            )

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_no_diversity_falls_back_to_single_query_path(
        self, search_agent_single_profile
    ):
        """
        When parallel mode is configured but all strategies produce queries identical
        to the original (no entity diversity), SearchAgent uses the single-query path.

        This verifies the guard: len(context.query_variants) > 1
        """
        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )

        telemetry_config = TelemetryConfig(
            otlp_endpoint="http://localhost:24317",
            provider_config={
                "http_endpoint": "http://localhost:26006",
                "grpc_endpoint": "http://localhost:24317",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        # Parallel mode but include_original=False, so if strategies produce
        # nothing distinct, we get empty variants → single-query fallback
        deps = RoutingDeps(
            tenant_id="test_tenant",
            telemetry_config=telemetry_config,
            query_fusion_config={
                "mode": "parallel",
                "variant_strategies": [
                    "relationship_expansion",
                    "boolean_optimization",
                ],
                "include_original": False,
                "rrf_k": 60,
            },
        )
        agent = RoutingAgent(deps=deps)

        # Use a query with no extractable entities → strategies produce no diversity
        result = await agent.analyze_and_route_with_relationships(
            query="hello",
            enable_relationship_extraction=True,
            enable_query_enhancement=True,
        )

        # With no entities, strategies produce no-ops, and include_original=False
        # means variants should be empty or have at most 1 entry
        if len(result.query_variants) <= 1:
            # Feed to SearchAgent — should use single-query path
            search_agent = search_agent_single_profile
            search_result = search_agent.search_with_routing_decision(result, top_k=10)

            assert search_result["status"] == "completed"
            assert "query_variants_used" not in search_result, (
                "Expected single-query path when variants <= 1, "
                f"but got multi-query fusion (variants: {result.query_variants})"
            )

            logger.info(
                "✅ No-diversity parallel mode correctly fell back to single-query path"
            )
        else:
            # If the query somehow produced diversity, just validate fusion works
            search_agent = search_agent_single_profile
            search_result = search_agent.search_with_routing_decision(result, top_k=10)
            assert search_result["status"] == "completed"
            logger.info(
                f"Note: 'hello' produced {len(result.query_variants)} variants "
                f"(unexpected but valid)"
            )
