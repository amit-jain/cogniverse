"""Real-Vespa, real-ColPali integration coverage for every ranking strategy.

Replaces the dormant ``tests/test_search_client.py`` and
``tests/test_colpali_search.py`` ad-hoc scripts with a parametrized
test that:

- Spins up a real Vespa container via the existing ``vespa_instance``
  fixture (deploys ``video_colpali_smol500_mv_frame_test_unit``).
- Spins up a real vLLM ColPali sidecar via ``vllm_sidecar`` and binds
  ``RemoteColPaliLoader`` against it.
- Seeds three documents with real per-token ColPali embeddings.
- Drives every rank profile in the schema through the production
  ``VespaSearchBackend.search`` against the real backend and asserts the
  returned results are non-empty, descending-ranked, and shape-correct.

The strategy names below are the schema's rank-profile names — the same
values the backend validates each query's ``strategy`` against. This is
the proper integration test the dormant scripts intended to be.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
import requests
from PIL import Image

from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader
from cogniverse_foundation.config.utils import get_config
from cogniverse_vespa.search_backend import VespaSearchBackend

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "vidore/colpali-v1.3-hf"
TENANT_ID = "test:unit"
TENANT_SCHEMA_NAME = "video_colpali_smol500_mv_frame_test_unit"
PROFILE_NAME = "test_colpali"

TEXT_ONLY_STRATEGIES = [
    "bm25_only",
    "bm25_no_description",
]
VISUAL_STRATEGIES = [
    "float_float",
    "binary_binary",
    "float_binary",
    "phased",
]
HYBRID_STRATEGIES = [
    "hybrid_float_bm25",
    "hybrid_binary_bm25",
    "hybrid_bm25_binary",
    "hybrid_bm25_float",
    "hybrid_float_bm25_no_description",
    "hybrid_binary_bm25_no_description",
    "hybrid_bm25_binary_no_description",
    "hybrid_bm25_float_no_description",
]


def _embeddings_to_vespa_tensors(embeddings: np.ndarray):
    """Convert (num_patches, 128) float32 embeddings to Vespa tensor format."""
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}
    return float_dict, binary_dict


@pytest.fixture(scope="module")
def vllm_colpali_url(vllm_sidecar):
    return vllm_sidecar.spawn(
        model=COLPALI_MODEL_NAME,
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "4096",
        ],
    )


@pytest.fixture(scope="module")
def colpali_client(vllm_colpali_url):
    loader = RemoteColPaliLoader(
        model_name=COLPALI_MODEL_NAME,
        config={"remote_inference_url": vllm_colpali_url},
        logger=logger,
    )
    client, _ = loader.load_model()
    return client


@pytest.fixture(scope="module")
def seeded_ranking_corpus(vespa_instance, colpali_client):
    """Feed three real-ColPali-embedded docs into Vespa for ranking tests."""
    test_docs = [
        {
            "color": (255, 0, 0),
            "title": "Sunset landscape ocean horizon",
            "video_id": "ranking_sunset_vid",
            "transcript": "the sun sets over the ocean horizon at golden hour",
        },
        {
            "color": (0, 0, 255),
            "title": "Ocean waves coastal scene",
            "video_id": "ranking_ocean_vid",
            "transcript": "ocean waves crash against the rocky coast under cloudy sky",
        },
        {
            "color": (0, 128, 0),
            "title": "Forest trail nature walk",
            "video_id": "ranking_forest_vid",
            "transcript": "person walking through dense green forest along trail",
        },
    ]

    http_port = vespa_instance["http_port"]

    for i, doc_info in enumerate(test_docs):
        img = Image.new("RGB", (224, 224), color=doc_info["color"])
        result = colpali_client.process_images([img], model_name=COLPALI_MODEL_NAME)
        embeddings_np = np.asarray(result["embeddings"]).astype(np.float32)
        float_dict, binary_dict = _embeddings_to_vespa_tensors(embeddings_np)

        doc_id = f"ranking_strat_doc_{i}"
        vespa_doc = {
            "fields": {
                "video_id": doc_info["video_id"],
                "video_title": doc_info["title"],
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "segment_description": doc_info["title"],
                "audio_transcript": doc_info["transcript"],
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }
        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
            json=vespa_doc,
            timeout=10,
        )
        assert resp.status_code in (200, 201), (
            f"Failed to feed doc {doc_id}: {resp.status_code}: {resp.text[:200]}"
        )

    time.sleep(5)
    yield test_docs

    for i in range(len(test_docs)):
        doc_id = f"ranking_strat_doc_{i}"
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
                timeout=5,
            )
        except Exception:
            pass


@pytest.fixture(scope="module")
def search_backend(vespa_instance, config_manager, schema_loader):
    """Production VespaSearchBackend wired to the real test Vespa + config.

    Profiles come from the same ConfigManager the search router uses, so
    ``profile=test_colpali`` resolves to the seeded
    ``video_colpali_smol500_mv_frame_test_unit`` schema and each query's
    ``strategy`` is validated against that schema's rank profiles.
    """
    cfg = get_config(tenant_id=TENANT_ID, config_manager=config_manager)
    backend_section = cfg.get("backend", {})
    config = {
        "url": "http://localhost",
        "port": vespa_instance["http_port"],
        "profiles": backend_section.get("profiles", {}),
        "default_profiles": backend_section.get("default_profiles", {}),
    }
    return VespaSearchBackend(
        config=config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


def _assert_results_well_formed(results, strategy):
    """Common shape/order assertions on a strategy's search results."""
    assert isinstance(results, list), (
        f"{strategy} must return list, got {type(results)}"
    )
    assert len(results) > 0, f"{strategy} returned 0 results from seeded ranking corpus"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"{strategy} results not in descending relevance order: {scores}"
    )
    for r in results:
        video_id = r.document.metadata.get("source_id", "")
        assert video_id.startswith("ranking_"), (
            f"{strategy} returned doc outside seeded corpus: {video_id}"
        )


@pytest.mark.integration
@pytest.mark.requires_vespa
@pytest.mark.requires_colpali
class TestRankingStrategiesReal:
    """Every rank profile in the schema exercised against real Vespa + ColPali."""

    @pytest.mark.parametrize("strategy", TEXT_ONLY_STRATEGIES)
    def test_text_only_strategy(self, strategy, search_backend, seeded_ranking_corpus):
        """Text-only strategies don't need query embeddings."""
        results = search_backend.search(
            {
                "query": "ocean waves",
                "type": "video",
                "profile": PROFILE_NAME,
                "strategy": strategy,
                "top_k": 10,
                "tenant_id": TENANT_ID,
            }
        )
        _assert_results_well_formed(results, strategy)

    @pytest.mark.parametrize("strategy", VISUAL_STRATEGIES)
    def test_visual_strategy(
        self,
        strategy,
        search_backend,
        colpali_client,
        seeded_ranking_corpus,
    ):
        """Visual strategies require pre-computed query embeddings."""
        result = colpali_client.process_queries(
            ["ocean waves coastal"], model_name=COLPALI_MODEL_NAME
        )
        embeddings = np.asarray(result["embeddings"]).astype(np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(0)

        results = search_backend.search(
            {
                "query": "",
                "type": "video",
                "profile": PROFILE_NAME,
                "strategy": strategy,
                "top_k": 10,
                "tenant_id": TENANT_ID,
                "query_embeddings": embeddings,
            }
        )
        _assert_results_well_formed(results, strategy)

    @pytest.mark.parametrize("strategy", HYBRID_STRATEGIES)
    def test_hybrid_strategy(
        self,
        strategy,
        search_backend,
        colpali_client,
        seeded_ranking_corpus,
    ):
        """Hybrid strategies use both text query and visual embeddings."""
        result = colpali_client.process_queries(
            ["ocean waves coastal"], model_name=COLPALI_MODEL_NAME
        )
        embeddings = np.asarray(result["embeddings"]).astype(np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(0)

        results = search_backend.search(
            {
                "query": "ocean coastal",
                "type": "video",
                "profile": PROFILE_NAME,
                "strategy": strategy,
                "top_k": 10,
                "tenant_id": TENANT_ID,
                "query_embeddings": embeddings,
            }
        )
        _assert_results_well_formed(results, strategy)


@pytest.mark.integration
@pytest.mark.requires_vespa
@pytest.mark.requires_colpali
class TestAutoSelectDefaultRanking:
    """Omitting ``strategy`` must auto-resolve the schema's default rank profile
    and return results — the contract SearchAgent now relies on after dropping
    its hardcoded ``binary_binary`` (which was invalid for audio). Before the
    backend fallback this raised "no default configured" on the 15-strategy
    video profile."""

    def test_search_without_strategy_auto_selects(
        self, search_backend, colpali_client, seeded_ranking_corpus
    ):
        result = colpali_client.process_queries(
            ["ocean waves coastal"], model_name=COLPALI_MODEL_NAME
        )
        embeddings = np.asarray(result["embeddings"]).astype(np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(0)

        results = search_backend.search(
            {
                "query": "ocean waves",
                "type": "video",
                "profile": PROFILE_NAME,
                "top_k": 10,
                "tenant_id": TENANT_ID,
                "query_embeddings": embeddings,
            }
        )
        assert isinstance(results, list)
        assert len(results) > 0, "auto-select returned no results"
        for r in results:
            assert r.document.metadata.get("source_id", "").startswith("ranking_")
