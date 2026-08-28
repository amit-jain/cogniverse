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
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_foundation.config.utils import get_config
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.vespa_test_helpers import deploy_tenant_schema, schema_tensor_dim

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"
TENANT_ID = "test:unit"
TENANT_SCHEMA_NAME = "video_colpali_smol500_mv_frame_test_unit"
PROFILE_NAME = "test_colpali"
SOURCE_COLLAPSE_TENANT_ID = "source_collapse:unit"
SOURCE_COLLAPSE_PROFILE_NAME = "test_colpali_source_collapse"
SOURCE_COLLAPSE_SCHEMA_NAME = "video_colpali_smol500_mv_frame"

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
    """Convert (num_patches, 320) float32 embeddings to Vespa tensor format."""
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}
    return float_dict, binary_dict


def _single_patch_rank_tensors(float_score: float, first_binary_byte: int):
    float_vec = np.zeros((1, 320), dtype=np.float32)
    float_vec[0, 0] = float_score
    binary_vec = np.full((1, 40), -1, dtype=np.int8)
    binary_vec[0, 0] = np.int8(first_binary_byte)
    return {"0": float_vec[0].tolist()}, {"0": binary_vec[0].tolist()}


def _manual_embedding(scale_x: float, scale_y: float, dim: int) -> np.ndarray:
    """Build a one-patch embedding with a deterministic 2D signal."""
    embedding = np.zeros((1, dim), dtype=np.float32)
    embedding[0, 0] = scale_x
    embedding[0, 1] = scale_y
    return embedding


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
def phased_default_ranking_corpus(vespa_instance):
    """Three docs whose binary order differs from their float rerank order."""
    docs = [
        {
            "doc_id": "phased_default_doc_0",
            "source_id": "default_rank_low",
            "float_score": 1.0,
            "binary_first_byte": -1,
        },
        {
            "doc_id": "phased_default_doc_1",
            "source_id": "default_rank_mid",
            "float_score": 2.0,
            "binary_first_byte": -2,
        },
        {
            "doc_id": "phased_default_doc_2",
            "source_id": "default_rank_high",
            "float_score": 3.0,
            "binary_first_byte": -4,
        },
    ]

    http_port = vespa_instance["http_port"]

    for doc_info in docs:
        float_dict, binary_dict = _single_patch_rank_tensors(
            doc_info["float_score"], doc_info["binary_first_byte"]
        )

        vespa_doc = {
            "fields": {
                "video_id": doc_info["source_id"],
                "video_title": doc_info["source_id"],
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "segment_description": doc_info["source_id"],
                "audio_transcript": "",
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }
        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_info['doc_id']}",
            json=vespa_doc,
            timeout=10,
        )
        assert resp.status_code in (200, 201), (
            f"Failed to feed doc {doc_info['doc_id']}: {resp.status_code}: {resp.text[:200]}"
        )

    time.sleep(5)
    yield [
        doc["source_id"]
        for doc in sorted(docs, key=lambda d: d["float_score"], reverse=True)
    ]

    for doc_info in docs:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_info['doc_id']}",
                timeout=5,
            )
        except Exception:
            pass


@pytest.fixture(scope="module")
def source_collapse_schema(vespa_instance, config_manager):
    """Deploy a dedicated video schema for the collapse-by-source corpus."""
    config_manager.add_backend_profile(
        BackendProfileConfig(
            profile_name=SOURCE_COLLAPSE_PROFILE_NAME,
            type="video",
            schema_name=SOURCE_COLLAPSE_SCHEMA_NAME,
            embedding_model=COLPALI_MODEL_NAME,
            model_loader="colpali",
        ),
        tenant_id=SOURCE_COLLAPSE_TENANT_ID,
    )
    return deploy_tenant_schema(
        vespa_instance,
        tenant_id=SOURCE_COLLAPSE_TENANT_ID,
        base_schema_name=SOURCE_COLLAPSE_SCHEMA_NAME,
        config_manager=config_manager,
    )


@pytest.fixture(scope="module")
def seeded_source_collapse_corpus(vespa_instance, source_collapse_schema):
    """Feed a source-skewed corpus for collapse-by-source integration tests."""
    http_port = vespa_instance["http_port"]
    embedding_dim = schema_tensor_dim(SOURCE_COLLAPSE_SCHEMA_NAME, "embedding")
    schema_name = source_collapse_schema

    source_a_vectors = [(1.0, 1.0 - i * 0.01) for i in range(50)]
    source_b_vectors = [
        (0.90, 0.00),
        (0.80, 0.10),
        (0.70, 0.20),
        (0.60, 0.30),
        (0.50, 0.40),
        (0.40, 0.50),
        (0.30, 0.60),
        (0.20, 0.70),
        (0.10, 0.80),
    ]

    fed_docs = []

    for i, (x, y) in enumerate(source_a_vectors):
        doc_id = f"collapse_big_source_doc_{i:02d}"
        image = _manual_embedding(x, y, embedding_dim)
        float_dict, binary_dict = _embeddings_to_vespa_tensors(image)
        fed_docs.append(
            (
                doc_id,
                "collapse_big_source",
                f"big source frame {i}",
                float_dict,
                binary_dict,
            )
        )

    for i, (x, y) in enumerate(source_b_vectors):
        doc_id = f"collapse_other_source_doc_{i:02d}"
        source_id = f"collapse_other_source_{i:02d}"
        image = _manual_embedding(x, y, embedding_dim)
        float_dict, binary_dict = _embeddings_to_vespa_tensors(image)
        fed_docs.append(
            (doc_id, source_id, f"other source frame {i}", float_dict, binary_dict)
        )

    for doc_id, video_id, title, float_dict, binary_dict in fed_docs:
        vespa_doc = {
            "fields": {
                "video_id": video_id,
                "video_title": title,
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "segment_description": title,
                "audio_transcript": title,
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }
        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{schema_name}/docid/{doc_id}",
            json=vespa_doc,
            timeout=10,
        )
        assert resp.status_code in (200, 201), (
            f"Failed to feed doc {doc_id}: {resp.status_code}: {resp.text[:200]}"
        )

    time.sleep(5)
    yield {
        "source_a_doc_ids": [f"collapse_big_source_doc_{i:02d}" for i in range(50)],
        "source_b_doc_ids": [f"collapse_other_source_doc_{i:02d}" for i in range(9)],
        "source_a_id": "collapse_big_source",
    }

    for doc_id, *_ in fed_docs:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{schema_name}/docid/{doc_id}",
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


@pytest.fixture(scope="module")
def source_collapse_search_backend(
    vespa_instance, config_manager, schema_loader, source_collapse_schema
):
    """Backend wired to the isolated collapse-by-source tenant/profile."""
    cfg = get_config(tenant_id=SOURCE_COLLAPSE_TENANT_ID, config_manager=config_manager)
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
@pytest.mark.requires_inference("vllm_colpali")
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
@pytest.mark.requires_inference("vllm_colpali")
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


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestDefaultPhasedRanking:
    """The default rank profile must rerank on the float phase."""

    def test_default_returns_float_rerank_order(
        self,
        search_backend,
        phased_default_ranking_corpus,
    ):
        results = search_backend.search(
            {
                "query": "",
                "type": "video",
                "profile": PROFILE_NAME,
                "strategy": "default",
                "top_k": 3,
                "tenant_id": TENANT_ID,
                "query_embeddings": np.ones((1, 320), dtype=np.float32),
            }
        )

        assert [
            r.document.metadata["source_id"] for r in results
        ] == phased_default_ranking_corpus


@pytest.mark.integration
@pytest.mark.requires_vespa
class TestSourceGranularityCollapse:
    """Source granularity keeps only the best document per source."""

    def test_source_granularity_returns_distinct_sources(
        self, source_collapse_search_backend, seeded_source_collapse_corpus
    ):
        query_embeddings = _manual_embedding(
            1.0,
            1.0,
            schema_tensor_dim(SOURCE_COLLAPSE_SCHEMA_NAME, "embedding"),
        )

        results = source_collapse_search_backend.search(
            {
                "query": "",
                "type": "video",
                "profile": SOURCE_COLLAPSE_PROFILE_NAME,
                "strategy": "float_float",
                "top_k": 10,
                "tenant_id": SOURCE_COLLAPSE_TENANT_ID,
                "query_embeddings": query_embeddings,
                "result_granularity": "source",
            }
        )

        assert len(results) == 10, [r.document.id for r in results]
        assert len({r.document.metadata["source_id"] for r in results}) == 10, [
            r.document.metadata["source_id"] for r in results
        ]
        assert [r.score for r in results] == sorted(
            (r.score for r in results), reverse=True
        )
        assert [r.document.metadata["source_id"] for r in results] == [
            "collapse_big_source",
            "collapse_other_source_00",
            "collapse_other_source_02",
            "collapse_other_source_04",
            "collapse_other_source_05",
            "collapse_other_source_07",
            "collapse_other_source_01",
            "collapse_other_source_03",
            "collapse_other_source_06",
            "collapse_other_source_08",
        ]
        assert [r.document.id for r in results] == [
            "collapse_big_source_doc_00",
            "collapse_other_source_doc_00",
            "collapse_other_source_doc_02",
            "collapse_other_source_doc_04",
            "collapse_other_source_doc_05",
            "collapse_other_source_doc_07",
            "collapse_other_source_doc_01",
            "collapse_other_source_doc_03",
            "collapse_other_source_doc_06",
            "collapse_other_source_doc_08",
        ]
        assert results[0].segments_in_window == 50
        assert [segment["document_id"] for segment in results[0].matched_segments] == [
            f"collapse_big_source_doc_{i:02d}" for i in range(50)
        ]
        assert all(
            set(segment) == {"document_id", "score", "start_time", "end_time"}
            for segment in results[0].matched_segments
        )
        for result, expected_source_id, expected_doc_id in zip(
            results[1:],
            [
                "collapse_other_source_00",
                "collapse_other_source_02",
                "collapse_other_source_04",
                "collapse_other_source_05",
                "collapse_other_source_07",
                "collapse_other_source_01",
                "collapse_other_source_03",
                "collapse_other_source_06",
                "collapse_other_source_08",
            ],
            [
                "collapse_other_source_doc_00",
                "collapse_other_source_doc_02",
                "collapse_other_source_doc_04",
                "collapse_other_source_doc_05",
                "collapse_other_source_doc_07",
                "collapse_other_source_doc_01",
                "collapse_other_source_doc_03",
                "collapse_other_source_doc_06",
                "collapse_other_source_doc_08",
            ],
        ):
            assert result.document.metadata["source_id"] == expected_source_id
            assert result.document.id == expected_doc_id
            assert result.segments_in_window == 1
            assert result.matched_segments == [
                {
                    "document_id": result.document.id,
                    "score": result.score,
                    "start_time": 0.0,
                    "end_time": 5.0,
                }
            ]

    def test_segment_granularity_keeps_every_document(
        self, source_collapse_search_backend, seeded_source_collapse_corpus
    ):
        query_embeddings = _manual_embedding(
            1.0,
            1.0,
            schema_tensor_dim(SOURCE_COLLAPSE_SCHEMA_NAME, "embedding"),
        )

        results = source_collapse_search_backend.search(
            {
                "query": "",
                "type": "video",
                "profile": SOURCE_COLLAPSE_PROFILE_NAME,
                "strategy": "float_float",
                "top_k": 10,
                "tenant_id": SOURCE_COLLAPSE_TENANT_ID,
                "query_embeddings": query_embeddings,
                "result_granularity": "segment",
            }
        )

        assert len(results) == 10
        assert [r.document.id for r in results] == [
            f"collapse_big_source_doc_{i:02d}" for i in range(10)
        ]
        assert {r.document.metadata["source_id"] for r in results} == {
            "collapse_big_source"
        }
