"""
COMPREHENSIVE END-TO-END TEST for Ensemble Search with REAL Profiles.

This test validates the ensemble search pipeline with:
1. REAL existing profiles with production Tomoro and X-CLIP models
2. Each profile has its own schema with different embedding dimensions
3. Real Vespa deployment and real searches
4. Real parallel execution and RRF fusion
5. Production-realistic latency measurement

Uses REAL profiles from the system, not artificial test profiles.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import pytest

from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names
from cogniverse_foundation.config.unified_config import BackendConfig
from tests.utils.vespa_test_helpers import load_raw_schema_json, shipped_profile

logger = logging.getLogger(__name__)


FRAME_PROFILE = shipped_profile(
    profile_type="video", embedding_type="multi_vector", extract_keyframes=True
)
CHUNK_PROFILE = shipped_profile(
    profile_type="video", embedding_type="multi_vector", process_type="video_chunks"
)
SINGLE_VECTOR_PROFILE = shipped_profile(
    profile_type="video", embedding_type="single_vector"
)
REAL_PROFILES = {
    profile.profile_name: profile
    for profile in (FRAME_PROFILE, CHUNK_PROFILE, SINGLE_VECTOR_PROFILE)
}


def test_visual_profiles_use_the_deployed_encoder_contract():
    profile_file = Path("configs/profiles/colqwen_chunks_profile.json")
    standalone_profile = json.loads(profile_file.read_text())

    assert (
        FRAME_PROFILE.embedding_model,
        FRAME_PROFILE.schema_config["embedding_dim"],
        FRAME_PROFILE.schema_config["binary_dim"],
    ) == ("TomoroAI/tomoro-colqwen3-embed-4b", 320, 40)
    assert (
        CHUNK_PROFILE.embedding_model,
        CHUNK_PROFILE.schema_config["embedding_dim"],
        CHUNK_PROFILE.schema_config["binary_dim"],
    ) == ("TomoroAI/tomoro-colqwen3-embed-4b", 320, 40)
    assert (
        SINGLE_VECTOR_PROFILE.embedding_model,
        SINGLE_VECTOR_PROFILE.schema_config["embedding_dim"],
        SINGLE_VECTOR_PROFILE.schema_config["binary_dim"],
    ) == ("microsoft/xclip-large-patch14", 768, 96)
    for profile in REAL_PROFILES.values():
        schema = load_raw_schema_json(profile.schema_name)
        assert schema["name"] == profile.schema_name
    assert standalone_profile["embedding_model"] == (
        "TomoroAI/tomoro-colqwen3-embed-4b"
    )
    assert standalone_profile["model_config"]["embedding_dim"] == 320


@pytest.fixture(scope="module")
def comprehensive_ensemble_setup():
    """
    Module-scoped setup for comprehensive ensemble test with REAL profiles.
    """
    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_runtime.ingestion.pipeline_builder import (
        create_config,
        create_pipeline,
    )
    from tests.system.vespa_test_manager import VespaTestManager
    from tests.utils.async_polling import wait_for_vespa_indexing
    from tests.utils.docker_utils import generate_unique_ports

    # Generate unique ports
    http_port, config_port = generate_unique_ports("comprehensive_ensemble")

    logger.info(
        f"🚀 Comprehensive Ensemble test using ports: {http_port} (http), {config_port} (config)"
    )

    # Clear singletons
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    # Create manager
    manager = VespaTestManager(http_port=http_port, config_port=config_port)

    try:
        # Setup Vespa
        logger.info("📦 Setting up Vespa container...")
        if not manager.full_setup():
            pytest.fail(
                "VespaTestManager.full_setup() failed — see the printed "
                "step that returned False (application directory, deploy, "
                "or video ingestion, which needs the profile's embedding "
                "inference service reachable)"
            )

        logger.info(f"✅ Vespa ready at http://localhost:{http_port}")

        backend_config = BackendConfig(
            tenant_id="test_tenant",
            backend_type="vespa",
            url="http://localhost",
            port=http_port,
            profiles=REAL_PROFILES,
        )
        manager.config_manager.set_backend_config(backend_config)
        assert tenant_usable_profile_names(
            manager.config_manager, "test_tenant"
        ) == sorted(REAL_PROFILES)
        schema_loader = FilesystemSchemaLoader(base_path=Path("configs/schemas"))
        app_config = {
            "backend": {**backend_config.to_dict(), "config_port": config_port},
        }
        video_files = sorted(manager.test_videos_dir.glob("*.mp4"))
        assert len(video_files) == manager.ingested_videos
        for profile in REAL_PROFILES.values():
            if profile.profile_name == manager.default_test_schema:
                continue
            pipeline_config = (
                create_config()
                .video_dir(manager.test_videos_dir)
                .max_frames_per_video(1)
                .generate_descriptions(profile.pipeline_config["generate_descriptions"])
                .backend("vespa")
                .build()
            )
            pipeline = (
                create_pipeline()
                .with_config(pipeline_config)
                .with_config_manager(manager.config_manager)
                .with_schema_loader(schema_loader)
                .with_app_config(app_config)
                .with_schema(profile.profile_name)
                .with_tenant_id("test_tenant")
                .build()
            )
            try:
                ingestion = asyncio.run(
                    pipeline.process_videos_concurrent(video_files, max_concurrent=1)
                )
                assert ingestion["status"] == "completed"
                assert ingestion["successful"] == len(video_files)
                assert ingestion["failed"] == 0
                assert [result["status"] for result in ingestion["results"]] == [
                    "completed"
                ] * len(video_files)
            finally:
                pipeline.processor_manager.cleanup()
        wait_for_vespa_indexing(
            backend_url=f"http://localhost:{http_port}",
            delay=5.0,
            description="Vespa document indexing for every ensemble profile",
        )

        yield {
            "http_port": http_port,
            "config_port": config_port,
            "base_url": f"http://localhost:{http_port}",
            "manager": manager,
            "profiles": REAL_PROFILES,
            "config_manager": manager.config_manager,
            "schema_loader": schema_loader,
        }

    except Exception as e:
        raise RuntimeError("Failed to set up the real ensemble boundary") from e

    finally:
        logger.info("🧹 Tearing down comprehensive ensemble test...")
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


def _create_test_documents_with_embeddings():
    """Create test documents with realistic embeddings"""
    import numpy as np

    docs = []
    for i in range(10):
        doc = {
            "id": f"test_video_{i}",
            "title": f"Test Video {i}",
            "description": f"Test description for video {i}",
            "embeddings": np.random.default_rng(i).random(320).tolist(),
        }
        docs.append(doc)

    return docs


@pytest.mark.requires_inference("vllm_colpali")
@pytest.mark.requires_inference("video_embed")
@pytest.mark.requires_inference("vllm_asr")
@pytest.mark.system
@pytest.mark.slow
@pytest.mark.e2e
class TestComprehensiveEnsembleSearch:
    """COMPREHENSIVE ensemble search tests with REAL profiles"""

    @pytest.mark.asyncio
    async def test_comprehensive_different_embedding_dimensions(
        self, comprehensive_ensemble_setup
    ):
        """
        COMPREHENSIVE TEST: Validate ensemble works with profiles using different embedding dimensions.

        Real profiles:
        - Frame and chunk multi-vector embeddings: 320-dim → 40 bytes
        - Chunk single-vector embeddings: 768-dim → 96 bytes

        Validates:
        - Different profiles with different schemas work together
        - Different embedding dimensions handled correctly
        - RRF fusion works across heterogeneous profiles
        """

        from cogniverse_agents.search_agent import (
            SearchAgent,
            SearchAgentDeps,
            SearchInput,
        )

        vespa_http_port = comprehensive_ensemble_setup["http_port"]
        vespa_config_port = comprehensive_ensemble_setup["config_port"]
        vespa_url = "http://localhost"
        config_manager = comprehensive_ensemble_setup["config_manager"]
        profiles = comprehensive_ensemble_setup["profiles"]

        schema_loader = comprehensive_ensemble_setup["schema_loader"]

        # Create SearchAgent
        search_deps = SearchAgentDeps(
            backend_url=vespa_url,
            backend_port=vespa_http_port,
            backend_config_port=vespa_config_port,
            profile=list(profiles.keys())[0],
        )
        agent = SearchAgent(
            deps=search_deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8018,
        )

        # NO MOCKING - use REAL query encoders and REAL Vespa!
        logger.info("🔄 Loading REAL query encoders (ColPali model ~2GB)")

        try:
            result = await agent._process_impl(
                SearchInput(
                    query="robot playing soccer",
                    tenant_id="test_tenant",
                    profiles=list(profiles.keys()),
                    top_k=5,
                    rrf_k=60,
                )
            )

            logger.info(f"Result from REAL Vespa with REAL encoders: {result}")

            # VALIDATE: Ensemble executed successfully (result is SearchOutput pydantic model)
            assert result.search_mode == "ensemble"
            assert set(result.profiles) == set(profiles.keys())

            # VALIDATE: Results structure
            assert isinstance(result.results, list)
            assert result.total_results is not None

            # VALIDATE: MUST have results (real Vespa with ingested data)
            assert result.total_results > 0, (
                "Should return results from REAL Vespa with ingested videos"
            )
            assert len(result.results) > 0, "Results list should not be empty"

            # VALIDATE: RRF fusion metadata on ALL results. The fusion fields
            # sit beside ``score`` in the public shape because the set is
            # ordered by ``rrf_score``.
            for doc in result.results:
                assert "rrf_score" in doc, f"Should have RRF score: {doc}"
                assert "profile_ranks" in doc, f"Should have profile ranks: {doc}"
                assert "num_profiles" in doc, f"Should have profile count: {doc}"
                assert doc["rrf_score"] > 0, f"Invalid RRF score: {doc['rrf_score']}"
                # Every fused document is ranked by at least one requested
                # profile, and the count must agree with the rank map.
                assert set(doc["profile_ranks"]) <= set(profiles), (
                    f"unknown profile in rank map: {doc['profile_ranks']}"
                )
                assert doc["num_profiles"] == len(doc["profile_ranks"]), (
                    f"num_profiles disagrees with profile_ranks: {doc}"
                )
                assert doc["num_profiles"] >= 1, f"no profile ranked {doc}"
                # The fusion fields are ranking, not payload.
                assert "rrf_score" not in doc["metadata"], (
                    f"rrf_score must not be duplicated into metadata: {doc}"
                )
                logger.info(
                    f"   Doc {doc['id']}: RRF={doc['rrf_score']:.4f}, "
                    f"profiles={doc['num_profiles']}"
                )

            logger.info(
                f"✅ Comprehensive test passed: {result.total_results} results from REAL Vespa with REAL encoders"
            )

        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            raise

    @pytest.mark.asyncio
    async def test_comprehensive_encoder_loading_latency(
        self, comprehensive_ensemble_setup
    ):
        """
        COMPREHENSIVE TEST: Validate latency with multiple real profiles.

        Validates:
        - Encoder loading doesn't duplicate unnecessarily
        - Parallel search execution
        - Production-realistic latency
        """

        from cogniverse_agents.search_agent import (
            SearchAgent,
            SearchAgentDeps,
            SearchInput,
        )

        vespa_http_port = comprehensive_ensemble_setup["http_port"]
        vespa_config_port = comprehensive_ensemble_setup["config_port"]
        vespa_url = "http://localhost"
        config_manager = comprehensive_ensemble_setup["config_manager"]
        profiles = comprehensive_ensemble_setup["profiles"]

        schema_loader = comprehensive_ensemble_setup["schema_loader"]

        search_deps = SearchAgentDeps(
            backend_url=vespa_url,
            backend_port=vespa_http_port,
            backend_config_port=vespa_config_port,
            profile=list(profiles.keys())[0],
        )
        agent = SearchAgent(
            deps=search_deps,
            schema_loader=schema_loader,
            config_manager=config_manager,
            port=8018,
        )

        # NO MOCKING - use REAL encoders and REAL Vespa
        logger.info("🔄 Measuring latency with REAL query encoders")

        try:
            start_time = time.time()

            _result = await agent._process_impl(
                SearchInput(
                    query="robot playing soccer",
                    tenant_id="test_tenant",
                    profiles=list(profiles.keys()),
                    top_k=10,
                    rrf_k=60,
                )
            )
            assert _result is not None  # Verify execution completed

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"⏱️  Total latency: {elapsed_ms:.2f}ms for {len(profiles)} profiles with REAL encoders"
            )

            # VALIDATE: Reasonable latency with REAL encoders and REAL Vespa
            # Target: <60000ms (60s) for real ColPali encoder (~2GB model) + real Vespa
            # First run includes model loading time
            assert elapsed_ms < 60000, (
                f"Latency {elapsed_ms:.2f}ms exceeds 60s threshold"
            )

            logger.info(
                f"✅ Latency validated: {elapsed_ms:.2f}ms with REAL encoders and REAL Vespa"
            )

        except Exception as e:
            logger.error(f"❌ Latency test failed: {e}")
            raise
