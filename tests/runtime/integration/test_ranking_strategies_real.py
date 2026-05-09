"""Real-Vespa, real-ColPali integration coverage for every RankingStrategy.

Replaces the dormant ``tests/test_search_client.py`` and
``tests/test_colpali_search.py`` ad-hoc scripts with a parametrized
test that:

- Spins up a real Vespa container via the existing ``vespa_instance``
  fixture (deploys ``video_colpali_smol500_mv_frame_test_unit``).
- Spins up a real vLLM ColPali sidecar via ``vllm_sidecar`` and binds
  ``RemoteColPaliLoader`` against it.
- Seeds three documents with real per-token ColPali embeddings.
- Drives each ``RankingStrategy`` enum variant through
  ``VespaVideoSearchClient.search`` against the real backend and
  asserts the returned results are non-empty, descending-ranked, and
  shape-correct.

This is the proper integration test the dormant scripts intended to be.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest
import requests
from PIL import Image

from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_vespa.vespa_search_client import (
    RankingStrategy,
    VespaVideoSearchClient,
)

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "vidore/colpali-v1.3-hf"
TENANT_SCHEMA_NAME = "video_colpali_smol500_mv_frame_test_unit"

TEXT_ONLY_STRATEGIES = [
    RankingStrategy.BM25_ONLY.value,
    RankingStrategy.BM25_NO_DESCRIPTION.value,
]
VISUAL_STRATEGIES = [
    RankingStrategy.FLOAT_FLOAT.value,
    RankingStrategy.BINARY_BINARY.value,
    RankingStrategy.FLOAT_BINARY.value,
    RankingStrategy.PHASED.value,
]
HYBRID_STRATEGIES = [
    RankingStrategy.HYBRID_FLOAT_BM25.value,
    RankingStrategy.HYBRID_BINARY_BM25.value,
    RankingStrategy.HYBRID_BM25_BINARY.value,
    RankingStrategy.HYBRID_BM25_FLOAT.value,
    RankingStrategy.HYBRID_FLOAT_BM25_NO_DESC.value,
    RankingStrategy.HYBRID_BINARY_BM25_NO_DESC.value,
    RankingStrategy.HYBRID_BM25_BINARY_NO_DESC.value,
    RankingStrategy.HYBRID_BM25_FLOAT_NO_DESC.value,
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
def vespa_search_client(vespa_instance):
    config_manager = create_default_config_manager()
    return VespaVideoSearchClient(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
        tenant_id="test:unit",
        config_manager=config_manager,
    )


def _assert_results_well_formed(results, strategy):
    """Common shape/order assertions on a strategy's search results."""
    assert isinstance(results, list), (
        f"{strategy} must return list, got {type(results)}"
    )
    assert len(results) > 0, f"{strategy} returned 0 results from seeded ranking corpus"
    scores = [r["relevance"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"{strategy} results not in descending relevance order: {scores}"
    )
    for r in results:
        assert "video_id" in r, f"{strategy} result missing video_id: {r}"
        assert r["video_id"].startswith("ranking_"), (
            f"{strategy} returned doc outside seeded corpus: {r['video_id']}"
        )


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestRankingStrategiesReal:
    """Every RankingStrategy enum variant exercised against real Vespa + ColPali."""

    @pytest.mark.parametrize("strategy", TEXT_ONLY_STRATEGIES)
    def test_text_only_strategy(
        self, strategy, vespa_search_client, seeded_ranking_corpus
    ):
        """Text-only strategies don't need query embeddings."""
        results = vespa_search_client.search(
            {
                "query": "ocean waves",
                "ranking": strategy,
                "top_k": 10,
                "schema": TENANT_SCHEMA_NAME,
            }
        )
        _assert_results_well_formed(results, strategy)

    @pytest.mark.parametrize("strategy", VISUAL_STRATEGIES)
    def test_visual_strategy(
        self,
        strategy,
        vespa_search_client,
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

        results = vespa_search_client.search(
            {
                "query": "",
                "ranking": strategy,
                "top_k": 10,
                "schema": TENANT_SCHEMA_NAME,
            },
            embeddings=embeddings,
        )
        _assert_results_well_formed(results, strategy)

    @pytest.mark.parametrize("strategy", HYBRID_STRATEGIES)
    def test_hybrid_strategy(
        self,
        strategy,
        vespa_search_client,
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

        results = vespa_search_client.search(
            {
                "query": "ocean coastal",
                "ranking": strategy,
                "top_k": 10,
                "schema": TENANT_SCHEMA_NAME,
            },
            embeddings=embeddings,
        )
        _assert_results_well_formed(results, strategy)
