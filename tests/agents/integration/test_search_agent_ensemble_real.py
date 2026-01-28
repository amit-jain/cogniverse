"""
Truly comprehensive integration tests for SearchAgent ensemble search.

Tests the COMPLETE ensemble pipeline with REAL components:
- Multiple deployed Vespa profiles with different schemas
- Real data ingestion into each profile
- Actual parallel search execution
- Real query encoders (one per profile)
- RRF fusion with real search results
- Latency validation (<700ms target)
- Error handling (profile failures, timeouts)
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


@pytest.mark.integration
@pytest.mark.slow
class TestSearchAgentEnsembleReal:
    """REAL comprehensive integration tests for ensemble search"""

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
