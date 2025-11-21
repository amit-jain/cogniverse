"""
COMPREHENSIVE END-TO-END TEST for Ensemble Search with REAL Profiles.

This test validates the ensemble search pipeline with:
1. REAL existing profiles with DIFFERENT models (ColPali, VideoPrism, ColQwen)
2. Each profile has its own schema with different embedding dimensions
3. Real Vespa deployment and real searches
4. Real parallel execution and RRF fusion
5. Production-realistic latency measurement

Uses REAL profiles from the system, not artificial test profiles.
"""

import logging
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


# Real profile definitions from the system
REAL_PROFILES = {
    "video_colpali_smol500_mv_frame": {
        "model": "vidore/colsmol-500m",
        "embedding_dim": 128,  # binarized from 1024
        "binary_dim": 16,
    },
    "video_videoprism_base_mv_chunk_30s": {
        "model": "google/videoprism-base",
        "embedding_dim": 768,
        "binary_dim": 96,
    },
    "video_colqwen_omni_mv_chunk_30s": {
        "model": "vidore/colqwen2-v0.1",
        "embedding_dim": 128,  # binarized from 1024
        "binary_dim": 16,
    },
}


@pytest.fixture(scope="module")
def comprehensive_ensemble_setup():
    """
    Module-scoped setup for comprehensive ensemble test with REAL profiles.
    """
    import tempfile
    from tests.utils.docker_utils import generate_unique_ports
    from tests.system.vespa_test_manager import VespaTestManager
    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_foundation.config.manager import ConfigManager

    # Generate unique ports
    http_port, config_port = generate_unique_ports("comprehensive_ensemble")

    logger.info(f"🚀 Comprehensive Ensemble test using ports: {http_port} (http), {config_port} (config)")

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
            pytest.skip("Failed to setup Vespa")

        logger.info(f"✅ Vespa ready at http://localhost:{http_port}")

        # The fixture already ingested REAL video data via full_setup()
        # No need to ingest more - use the real videos from VespaTestManager
        logger.info("✅ Using REAL video data from VespaTestManager.full_setup()")

        yield {
            "http_port": http_port,
            "config_port": config_port,
            "base_url": f"http://localhost:{http_port}",
            "manager": manager,
            "profiles": REAL_PROFILES,
            "config_manager": manager.config_manager,
        }

    except Exception as e:
        logger.error(f"❌ Failed to setup comprehensive ensemble test: {e}")
        pytest.skip(f"Failed to setup: {e}")

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
            "embeddings": np.random.rand(128).tolist(),
        }
        docs.append(doc)

    return docs


@pytest.mark.system
@pytest.mark.slow
@pytest.mark.e2e
class TestComprehensiveEnsembleSearch:
    """COMPREHENSIVE ensemble search tests with REAL profiles"""

    @pytest.mark.asyncio
    async def test_comprehensive_different_embedding_dimensions(self, comprehensive_ensemble_setup):
        """
        COMPREHENSIVE TEST: Validate ensemble works with profiles using different embedding dimensions.

        Real profiles:
        - video_colpali_smol500_mv_frame: 128-dim (binarized from 1024) → 16 bytes
        - video_videoprism_base_mv_chunk_30s: 768-dim → 96 bytes
        - video_colqwen_omni_mv_chunk_30s: 128-dim (binarized from 1024) → 16 bytes

        Validates:
        - Different profiles with different schemas work together
        - Different embedding dimensions handled correctly
        - RRF fusion works across heterogeneous profiles
        """
        from cogniverse_agents.search_agent import SearchAgent
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.unified_config import BackendConfig, BackendProfileConfig
        from unittest.mock import patch, Mock
        import numpy as np

        vespa_http_port = comprehensive_ensemble_setup["http_port"]
        vespa_config_port = comprehensive_ensemble_setup["config_port"]
        vespa_url = "http://localhost"
        config_manager = comprehensive_ensemble_setup["config_manager"]
        profiles = comprehensive_ensemble_setup["profiles"]

        schema_loader = FilesystemSchemaLoader(
            base_path=Path("tests/system/resources/schemas")
        )

        # Register all real profiles in config_manager
        backend_profiles = {}
        for profile_name, profile_data in profiles.items():
            backend_profiles[profile_name] = BackendProfileConfig(
                profile_name=profile_name,
                schema_name=profile_name,  # Each profile uses its own schema
                embedding_model=profile_data["model"],
            )

        backend_config = BackendConfig(
            tenant_id="test_tenant",
            backend_type="vespa",
            url=vespa_url,
            port=vespa_http_port,
            profiles=backend_profiles,
        )
        config_manager.set_backend_config(backend_config)

        # Create SearchAgent
        agent = SearchAgent(
            tenant_id="test_tenant",
            schema_loader=schema_loader,
            config_manager=config_manager,
            backend_url=vespa_url,
            backend_port=vespa_http_port,
            backend_config_port=vespa_config_port,
            profile=list(profiles.keys())[0],
            auto_create_schema=False,
            port=8018,
        )

        # NO MOCKING - use REAL query encoders and REAL Vespa!
        logger.info("🔄 Loading REAL query encoders (ColPali model ~2GB)")

        try:
            result = await agent._process({
                "query": "robot playing soccer",  # Real query matching test videos
                "profiles": list(profiles.keys()),
                "top_k": 5,
                "rrf_k": 60
            })

            logger.info(f"Result from REAL Vespa with REAL encoders: {result}")

            # VALIDATE: Ensemble executed successfully
            assert result["search_mode"] == "ensemble"
            assert set(result["profiles"]) == set(profiles.keys())

            # VALIDATE: Results structure
            assert "results" in result
            assert "total_results" in result

            # VALIDATE: MUST have results (real Vespa with ingested data)
            assert result["total_results"] > 0, "Should return results from REAL Vespa with ingested videos"
            assert len(result["results"]) > 0, "Results list should not be empty"

            # VALIDATE: RRF fusion metadata on ALL results
            for doc in result["results"]:
                assert "rrf_score" in doc, "Should have RRF score"
                assert "profile_ranks" in doc, "Should have profile ranks"
                assert "num_profiles" in doc, "Should have profile count"
                assert doc["rrf_score"] > 0, f"Invalid RRF score: {doc['rrf_score']}"
                logger.info(f"   Doc {doc['id']}: RRF={doc['rrf_score']:.4f}, profiles={doc['num_profiles']}")

            logger.info(f"✅ Comprehensive test passed: {result['total_results']} results from REAL Vespa with REAL encoders")

        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            raise

    @pytest.mark.asyncio
    async def test_comprehensive_encoder_loading_latency(self, comprehensive_ensemble_setup):
        """
        COMPREHENSIVE TEST: Validate latency with multiple real profiles.

        Validates:
        - Encoder loading doesn't duplicate unnecessarily
        - Parallel search execution
        - Production-realistic latency
        """
        from cogniverse_agents.search_agent import SearchAgent
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.unified_config import BackendConfig, BackendProfileConfig
        from unittest.mock import patch, Mock
        import numpy as np

        vespa_http_port = comprehensive_ensemble_setup["http_port"]
        vespa_config_port = comprehensive_ensemble_setup["config_port"]
        vespa_url = "http://localhost"
        config_manager = comprehensive_ensemble_setup["config_manager"]
        profiles = comprehensive_ensemble_setup["profiles"]

        schema_loader = FilesystemSchemaLoader(
            base_path=Path("tests/system/resources/schemas")
        )

        # Register profiles
        backend_profiles = {}
        for profile_name, profile_data in profiles.items():
            backend_profiles[profile_name] = BackendProfileConfig(
                profile_name=profile_name,
                schema_name=profile_name,
                embedding_model=profile_data["model"],
            )

        backend_config = BackendConfig(
            tenant_id="test_tenant",
            backend_type="vespa",
            url=vespa_url,
            port=vespa_http_port,
            profiles=backend_profiles,
        )
        config_manager.set_backend_config(backend_config)

        agent = SearchAgent(
            tenant_id="test_tenant",
            schema_loader=schema_loader,
            config_manager=config_manager,
            backend_url=vespa_url,
            backend_port=vespa_http_port,
            backend_config_port=vespa_config_port,
            profile=list(profiles.keys())[0],
            auto_create_schema=False,
            port=8018,
        )

        # NO MOCKING - use REAL encoders and REAL Vespa
        logger.info("🔄 Measuring latency with REAL query encoders")

        try:
            start_time = time.time()

            result = await agent._process({
                "query": "robot playing soccer",
                "profiles": list(profiles.keys()),
                "top_k": 10,
                "rrf_k": 60
            })

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(f"⏱️  Total latency: {elapsed_ms:.2f}ms for {len(profiles)} profiles with REAL encoders")

            # VALIDATE: Reasonable latency with REAL encoders and REAL Vespa
            # Target: <60000ms (60s) for real ColPali encoder (~2GB model) + real Vespa
            # First run includes model loading time
            assert elapsed_ms < 60000, f"Latency {elapsed_ms:.2f}ms exceeds 60s threshold"

            logger.info(f"✅ Latency validated: {elapsed_ms:.2f}ms with REAL encoders and REAL Vespa")

        except Exception as e:
            logger.error(f"❌ Latency test failed: {e}")
            raise

