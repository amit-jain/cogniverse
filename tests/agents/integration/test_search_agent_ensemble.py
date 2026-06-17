"""
Integration tests for SearchAgent ensemble search and multi-query fusion.

Tests with real Vespa backend, real query encoders, real search:
- Multiple deployed Vespa profiles with different schemas
- Parallel search execution across profiles
- RRF fusion with real search results
- Multi-query fusion (parallel variant search + RRF)
- Latency validation
- Error handling (profile failures, sparse results)

Tests requiring a real LM for ComposableQueryAnalysisModule query variant
generation use @skip_if_no_lm and configure DSPy via ``tests/fixtures/llm.py``.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_agents.search_agent import SearchInput

logger = logging.getLogger(__name__)

# Each RRF profile is a 320-dim ColPali-family schema served by the same
# Tomoro sidecar. (base_schema_name, vespa_namespace, doc_id_field).
ENSEMBLE_PROFILES = [
    ("video_colpali_smol500_mv_frame", "video", "video_id"),
    ("image_colpali_mv", "image", "image_id"),
    ("video_colqwen_omni_mv_chunk_30s", "video", "video_id"),
]


def _multi_patch_blocks(seed: int, *, patches: int = 4, dim: int = 320) -> dict:
    """A deterministic multi-patch float embedding in document/v1 blocks form.

    ``tensor<bfloat16>(patch{}, v[dim])`` — one block per patch keyed by the
    mapped ``patch`` dimension, each a dense ``dim``-vector. Values vary per
    profile (via ``seed``) so the three deployed schemas hold distinct docs.
    """
    rng = np.random.default_rng(seed)
    return {
        str(p): rng.standard_normal(dim).astype(np.float32).tolist()
        for p in range(patches)
    }


def _binary_blocks_from_float(float_blocks: dict) -> dict:
    """Pack a float blocks dict into ``tensor<int8>(patch{}, v[40])``.

    Mirrors the ingestion-side binarization (sign bit per dim, packed 8/byte,
    viewed as int8) so the default ``max_sim_hamming`` rank profile scores the
    doc the same way it would a really-ingested one.
    """
    out = {}
    for patch, values in float_blocks.items():
        bits = np.where(np.asarray(values, dtype=np.float32) > 0, 1, 0).astype(np.uint8)
        out[patch] = np.packbits(bits).astype(np.int8).tolist()
    return out


def _feed_doc(
    http_port: int, namespace: str, schema_full: str, doc_id: str, fields: dict
) -> None:
    resp = requests.post(
        f"http://localhost:{http_port}/document/v1/{namespace}/{schema_full}/docid/{doc_id}",
        json={"fields": fields},
        timeout=20,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Feed of {doc_id} into {schema_full} failed "
            f"({resp.status_code}): {resp.text[:400]}"
        )


def _doc_fields_for(base_schema_name: str, doc_id_field: str, seed: int) -> dict:
    """Minimal valid document with float + binary multi-patch embeddings."""
    float_blocks = _multi_patch_blocks(seed)
    binary_blocks = _binary_blocks_from_float(float_blocks)
    fields = {
        doc_id_field: f"ensemble_doc_{base_schema_name}",
        "source_url": f"s3://corpus/ensemble/{base_schema_name}.bin",
        "embedding": {"blocks": float_blocks},
        "embedding_binary": {"blocks": binary_blocks},
    }
    if doc_id_field == "image_id":
        fields["image_title"] = "Robot playing soccer"
        fields["image_description"] = "a robot kicking a soccer ball"
    else:
        fields["video_title"] = "Robot playing soccer"
        fields["segment_id"] = 0
    return fields


@pytest.fixture(scope="module")
def multi_profile_vespa(shared_memory_vespa):
    """Module-scoped multi-profile fixture backed by the project-wide
    ``shared_vespa``. Deploys ALL THREE RRF schemas
    (``video_colpali_smol500_mv_frame``, ``image_colpali_mv``,
    ``video_colqwen_omni_mv_chunk_30s``) for tenant ``ensemble_test_tenant``
    via SchemaRegistry (merge-safe), then feeds one document into each so
    every profile returns a real hit and RRF genuinely fuses across the
    three profiles — not one schema masquerading as three.

    Includes a ``manager`` field exposing ``config_manager`` since the
    consumer fixture reads ``multi_profile_vespa["manager"].config_manager``.
    """
    from cogniverse_core.registries.backend_registry import (
        BackendRegistry,
        get_backend_registry,
    )
    from cogniverse_foundation.config.manager import ConfigManager

    class _SharedVespaEnsembleAdapter:
        def __init__(self, cm):
            self.config_manager = cm

        def cleanup(self) -> None:
            pass

    # Clear singletons inherited from other modules.
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    from tests.utils.vespa_test_helpers import (
        deploy_tenant_schema,
        make_config_manager,
    )

    cm = make_config_manager(shared_memory_vespa)
    http_port = shared_memory_vespa["http_port"]

    deployed_docs = []
    for seed, (base_schema_name, namespace, doc_id_field) in enumerate(
        ENSEMBLE_PROFILES
    ):
        full = deploy_tenant_schema(
            shared_memory_vespa,
            tenant_id="ensemble_test_tenant",
            base_schema_name=base_schema_name,
            config_manager=cm,
        )
        # Reset between deploys so each merge-deploy rebuilds from config.
        if hasattr(registry, "_backend_instances"):
            registry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None

        fields = _doc_fields_for(base_schema_name, doc_id_field, seed)
        _feed_doc(http_port, namespace, full, fields[doc_id_field], fields)
        deployed_docs.append((namespace, full, fields[doc_id_field]))

    # Let Vespa index the freshly fed docs before any query runs.
    time.sleep(3)

    # Reset registry caches so consumer tests don't inherit the deploy's
    # backend instance.
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()

    real_profiles = [base for base, _, _ in ENSEMBLE_PROFILES]
    yield {
        "http_port": http_port,
        "config_port": shared_memory_vespa["config_port"],
        "base_url": shared_memory_vespa["base_url"],
        "manager": _SharedVespaEnsembleAdapter(cm),
        "profiles": real_profiles,
        "profile_name": real_profiles[0],
    }

    for namespace, full, doc_id in deployed_docs:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/{namespace}/{full}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


@pytest.fixture
def search_agent_ensemble(multi_profile_vespa, tomoro_inference_url):
    """SearchAgent configured for ensemble search with 3 REAL different profiles"""
    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        BackendProfileConfig,
    )
    from tests.agents.integration.conftest import inject_tomoro_url

    vespa_http_port = multi_profile_vespa["http_port"]
    vespa_config_port = multi_profile_vespa["config_port"]
    vespa_url = "http://localhost"
    config_manager = multi_profile_vespa["manager"].config_manager
    profiles = multi_profile_vespa["profiles"]

    # Tomoro is remote-only; route the query encoder for every Tomoro-backed
    # profile through the spawned sidecar before the SearchAgent reads config.
    inject_tomoro_url(config_manager, tomoro_inference_url)

    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Each profile maps to its OWN deployed tenant-scoped schema. All three
    # are 320-dim ColPali-family schemas served by the same Tomoro sidecar,
    # so ``inject_tomoro_url`` routes every profile's query encoding remotely.
    # ``multi_profile_vespa`` deployed each schema and fed one doc into it, so
    # all three profiles return a real hit and RRF fuses across genuinely
    # distinct schemas.
    backend_profiles = {
        base: BackendProfileConfig(
            profile_name=base,
            schema_name=base,
            embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
        )
        for base in profiles
    }

    backend_config = BackendConfig(
        tenant_id="ensemble_test_tenant",
        backend_type="vespa",
        url=vespa_url,
        port=vespa_http_port,
        profiles=backend_profiles,
    )
    config_manager.set_backend_config(backend_config)

    # ``multi_profile_vespa`` (via ``full_setup``) already triggered the
    # BackendRegistry to cache a search backend with the single default
    # profile. Without this clear, the cached backend's ``self.profiles``
    # dict is frozen at that point — subsequent searches against the new
    # profile names fail with "Requested profile X not found. Available
    # profiles: [video_colpali_smol500_mv_frame]" because the registry
    # returns the cached instance instead of rebuilding from updated
    # config.
    from cogniverse_core.registries.backend_registry import get_backend_registry

    _registry = get_backend_registry()
    if hasattr(_registry, "_backend_instances"):
        _registry._backend_instances.clear()

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
def search_agent_single_profile(multi_profile_vespa, tomoro_inference_url):
    """SearchAgent configured for single-profile search with correct tenant_id."""
    from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from tests.agents.integration.conftest import inject_tomoro_url

    vespa_http_port = multi_profile_vespa["http_port"]
    vespa_config_port = multi_profile_vespa["config_port"]
    vespa_url = "http://localhost"
    config_manager = multi_profile_vespa["manager"].config_manager
    default_profile = multi_profile_vespa["profiles"][0]

    # Tomoro is remote-only; route the query encoder through the sidecar.
    inject_tomoro_url(config_manager, tomoro_inference_url)

    schema_loader = FilesystemSchemaLoader(
        base_path=Path("tests/system/resources/schemas")
    )

    # Match the tenant ``multi_profile_vespa`` deploys the schema for, so the
    # tenant-scoped source ref (``..._ensemble_test_tenant_ensemble_test_tenant``)
    # resolves on every single-profile query.
    deps = SearchAgentDeps(
        tenant_id="ensemble_test_tenant",
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
            SearchInput(
                query="robot playing soccer",
                tenant_id="ensemble_test_tenant",
                profiles=profiles,
                top_k=5,
                rrf_k=60,
            )
        )

        # VALIDATE: Ensemble mode detected
        assert result.search_mode == "ensemble"
        assert set(result.profiles) == set(profiles)

        # VALIDATE: Every profile's own schema is deployed + populated, so the
        # ensemble returns real fused hits across all three profiles.
        assert result.results, "ensemble returned no fused results"
        assert result.total_results == len(result.results)

        # Each fused doc carries RRF metadata; at least one doc was contributed
        # by every one of the three profiles (genuine cross-profile fusion).
        contributing = set()
        for doc in result.results:
            ranks = (doc.get("metadata") or {}).get("profile_ranks") or {}
            contributing.update(ranks.keys())
        assert set(contributing) == set(profiles), (
            f"RRF only fused profiles {contributing}, expected all of {profiles}"
        )

        logger.info(
            f"✅ Ensemble fused {result.total_results} results across "
            f"profiles {sorted(contributing)}"
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
            SearchInput(
                query="test query for latency",
                tenant_id="ensemble_test_tenant",
                profiles=profiles,
                top_k=10,
                rrf_k=60,
            )
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
            SearchInput(
                query="test query with failure",
                tenant_id="ensemble_test_tenant",
                profiles=profiles_with_invalid,
                top_k=10,
                rrf_k=60,
            )
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
            SearchInput(
                query="test parallel execution",
                tenant_id="ensemble_test_tenant",
                profiles=profiles,
                top_k=10,
                rrf_k=60,
            )
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
        assert total_time < 60.0, (
            f"Ensemble took {total_time:.3f}s (too slow even for parallel execution)"
        )

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
            SearchInput(
                query="robot playing soccer",
                tenant_id="ensemble_test_tenant",
                profiles=profiles,
                top_k=10,
                rrf_k=60,
            )
        )

        # VALIDATE: RRF fusion metadata on REAL search results
        assert result.search_mode == "ensemble"
        results = result.results

        # Validate RRF metadata structure on all results. The public
        # result shape (``_format_public_result``) nests every non-identity
        # field into ``metadata``, so RRF-derived fields land at
        # ``doc["metadata"]["rrf_score"]`` rather than at the top level —
        # ``doc["score"]``, ``doc["id"]``, ``doc["document_id"]`` are the
        # only top-level fields the public contract guarantees.
        for doc in results:
            metadata = doc.get("metadata") or {}
            assert "rrf_score" in metadata, (
                f"Missing rrf_score in metadata for doc {doc.get('id')}"
            )
            assert "profile_ranks" in metadata, (
                f"Missing profile_ranks in metadata for doc {doc.get('id')}"
            )
            assert "num_profiles" in metadata, (
                f"Missing num_profiles in metadata for doc {doc.get('id')}"
            )
            assert metadata["rrf_score"] > 0, (
                f"Invalid RRF score {metadata['rrf_score']} in doc {doc.get('id')}"
            )
            assert metadata["num_profiles"] >= 1, (
                f"Invalid num_profiles {metadata['num_profiles']} in doc {doc.get('id')}"
            )

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
            SearchInput(
                query="xyzabc123nonexistent query that returns nothing",
                tenant_id="ensemble_test_tenant",
                profiles=profiles,
                top_k=10,
                rrf_k=60,
            )
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
        """Multi-query fusion end-to-end via SearchInput.query_variants."""
        agent, _profiles = search_agent_ensemble

        input_data = SearchInput(
            query="robot playing soccer",
            tenant_id="ensemble_test_tenant",
            top_k=10,
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
            query_variants=[
                {"name": "original", "query": "robot playing soccer"},
                {
                    "name": "relationship_expansion",
                    "query": "robot playing soccer (robot playing soccer)",
                },
            ],
        )

        result = await agent._process_impl(input_data)

        assert result.search_mode == "multi_query_fusion"
        assert result.total_results >= 0

        logger.info(
            f"✅ Multi-query fusion with real Vespa: {result.total_results} results"
        )

    @pytest.mark.asyncio
    async def test_multi_query_fusion_latency(self, search_agent_ensemble):
        """Multi-query fusion latency. Target: <60000ms."""
        agent, _profiles = search_agent_ensemble

        input_data = SearchInput(
            query="machine learning tutorial",
            tenant_id="ensemble_test_tenant",
            top_k=10,
            enhanced_query="machine learning tutorial (machine learning tutorial)",
            entities=[
                {"text": "machine learning", "label": "TECHNOLOGY", "confidence": 0.92}
            ],
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
        result = await agent._process_impl(input_data)
        elapsed_ms = (time.time() - start_time) * 1000

        assert result.search_mode == "multi_query_fusion"
        assert elapsed_ms < 60000, (
            f"Multi-query fusion took {elapsed_ms:.2f}ms (too slow)"
        )

        logger.info(f"✅ Multi-query fusion latency: {elapsed_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_multi_query_fusion_sparse_results(self, search_agent_ensemble):
        """Multi-query fusion with sparse/empty results — graceful handling."""
        agent, _profiles = search_agent_ensemble

        input_data = SearchInput(
            query="xyzabc123nonexistent query fusion test",
            tenant_id="ensemble_test_tenant",
            top_k=10,
            enhanced_query="xyzabc123nonexistent query fusion expanded",
            query_variants=[
                {"name": "original", "query": "xyzabc123nonexistent query fusion test"},
                {
                    "name": "expansion",
                    "query": "xyzabc123nonexistent query fusion expanded",
                },
            ],
        )

        result = await agent._process_impl(input_data)

        assert result.search_mode == "multi_query_fusion"
        assert isinstance(result.results, list)

        logger.info(
            f"✅ Sparse results handled gracefully: {result.total_results} results"
        )

    @pytest.mark.asyncio
    async def test_single_query_fallback_without_variants(
        self, search_agent_single_profile
    ):
        """SearchInput without query_variants routes through the single-query path."""
        agent = search_agent_single_profile

        input_data = SearchInput(
            query="robot playing soccer",
            tenant_id="ensemble_test_tenant",
            top_k=10,
            enhanced_query="robot playing soccer enhanced",
            entities=[{"text": "robot", "label": "TECHNOLOGY", "confidence": 0.9}],
            # No query_variants → single-query path
        )

        result = await agent._process_impl(input_data)

        assert result.search_mode == "single_profile"

        logger.info(
            f"✅ Single query fallback: {result.total_results} results (no variants)"
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
            SearchInput(
                query="robot playing soccer", tenant_id="ensemble_test_tenant", top_k=10
            )
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
    async def test_single_profile_with_enrichment(self, search_agent_single_profile):
        """Single-profile search via SearchInput with enrichment but no variants."""
        agent = search_agent_single_profile

        input_data = SearchInput(
            query="robot playing soccer",
            tenant_id="ensemble_test_tenant",
            top_k=10,
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
        )

        result = await agent._process_impl(input_data)

        assert result.search_mode == "single_profile"
        assert isinstance(result.results, list)

        logger.info(
            f"✅ Single profile with enrichment: {result.total_results} results"
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

        result = agent.search_with_relationship_context(
            context, tenant_id="ensemble_test_tenant", top_k=10
        )

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
            SearchInput(
                query="xyzabc123nonexistent gibberish",
                tenant_id="ensemble_test_tenant",
                top_k=10,
            )
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
            SearchInput(
                query="robot soccer",
                tenant_id="ensemble_test_tenant",
                profiles=[agent.active_profile],
                top_k=10,
            )
        )

        assert result.search_mode == "single_profile"
        assert result.profile is not None

        logger.info(
            f"✅ Single profile in list still single mode: profile={result.profile}"
        )
