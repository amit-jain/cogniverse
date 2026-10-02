"""Default and hybrid ranking on the ColQwen3 frame profile, against real Vespa.

ColQwen3 query encodings open with prompt-template tokens that reappear almost
bit-for-bit in every frame (Hamming distance 0-4 of 320 bits). The binary
MaxSim score each video rank profile ranks or pre-selects with must weight
every query token by its estimated cosine ``1 - 2h/320``, as the float MaxSim
it approximates does; otherwise bit noise on those template tokens decides
which 100 frames the float second phase may rerank.

The visual-first hybrids rank every frame by that MaxSim averaged over the
query tokens plus the frame's ``nativeRank`` for the query text, so a frame
without a text match keeps its visual score.

The corpus is the ten-video ActivityNet frame index of the
``flywheel_org:production`` tenant, recorded with ``RECORD_GOLDEN=1`` together
with the frames' text fields and the served ColQwen3 encodings of the pinned
golden queries. Every frame keeps the patches that are a pinned query token's
best float match or nearest binary match, so each frame's MaxSim scores for the
pinned queries equal its scores over all of its patches. Documents are fed through the production
ingestion path and searched through ``VespaSearchBackend`` with the shipped
profile config, the way ``POST /search`` serves the profile.

Re-record (live Vespa holding the tenant's frame index, served ColQwen3):

    RECORD_GOLDEN=1 LIVE_VESPA_URL=http://localhost:8080 \\
    COLPALI_INFERENCE_URL=http://localhost:29001 \\
    uv run pytest tests/backends/integration/test_colqwen_binary_maxsim_ranking.py
"""

from __future__ import annotations

import copy
import json
import os
import socket
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpr
import numpy as np
import pytest
import requests

from cogniverse_core.common.utils.retry import RetryConfig
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.schema_registry import DeployedSchemaNames
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus
from cogniverse_vespa.ingestion_client import document_namespace
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.vespa_test_helpers import make_config_manager, schema_tensor_dim

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
SCHEMAS_DIR = Path("configs/schemas")
SHIPPED_PROFILE = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
][BASE_SCHEMA]
GOLDEN_QUERIES = json.loads(
    Path("data/testset/evaluation/sample_videos_retrieval_queries.json").read_text()
)
CORPUS_PATH = Path(__file__).parent / "goldens" / "colqwen3_frame_corpus.npz"
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"

PINNED_QUERIES = ("people shoveling", "a fire starter", "catching")
RECORDED_FRAMES_PER_VIDEO = {
    "v_-uJnucdW6DY": 112,
    "v_-IMXSEIabMM": 65,
    "v_-vnSFKJNB94": 55,
    "v_-pkfcMUIEMo": 53,
    "v_-MbZ-W0AbN0": 39,
    "v_-HpCLXdtcas": 12,
    "v_-D1gdv_gQyw": 10,
    "v_-cAcA8dO7kA": 6,
    "v_-6dz6tBH77I": 5,
    "v_-nl4G-00PtA": 4,
}
RECORDED_FRAME_COUNT = 361
# Recorded array holding each text field the schema's default fieldset
# searches, other than the title (recorded as video_titles).
RECORDED_TEXT_ARRAYS = {
    "segment_description": "segment_descriptions",
    "audio_transcript": "audio_transcripts",
}


def _schema_json() -> dict:
    return json.loads((SCHEMAS_DIR / f"{BASE_SCHEMA}_schema.json").read_text())


def _searched_text_fields() -> set[str]:
    """The fields ``userInput`` searches: the schema's default fieldset."""
    (fieldset,) = [f for f in _schema_json()["fieldsets"] if f["name"] == "default"]
    return set(fieldset["fields"])


def _text_metadata_keys() -> dict[str, str]:
    """The Document metadata key the ingestion path maps to each text field."""
    mapping = _schema_json()["document_mapping"]
    renamed = {field: key for key, field in mapping["metadata_fields"].items()}
    return {field: renamed.get(field, field) for field in RECORDED_TEXT_ARRAYS}


def _dead_port() -> int:
    """A local TCP port with nothing listening on it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _video_key(title: str) -> str:
    """Golden ``expected_videos`` name a video by its title's file stem."""
    return title.rsplit(".", 1)[0]


def _expected_video(query: str) -> str:
    (expected,) = {
        video
        for entry in GOLDEN_QUERIES
        if entry["query"] == query
        for video in entry["expected_videos"]
    }
    return expected


def _binarize(vectors: np.ndarray) -> np.ndarray:
    return np.packbits((vectors > 0).astype(np.uint8), axis=1)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _min_hamming(query: np.ndarray, patches: np.ndarray) -> np.ndarray:
    """Per query token, the Hamming distance to its nearest patch."""
    xor = np.bitwise_xor(_binarize(query)[:, None, :], _binarize(patches)[None])
    return _POPCOUNT[xor].sum(-1).min(axis=1)


def _float_max_sim(query: np.ndarray, patches: np.ndarray) -> float:
    return float((query @ patches.T).max(axis=1).sum())


def _binary_max_sim(query: np.ndarray, patches: np.ndarray) -> float:
    dim = query.shape[1]
    return float((1.0 - 2.0 * _min_hamming(query, patches) / dim).sum())


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32) >> 16
    return bits.astype(np.uint16)


def _bf16_values(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _record_corpus() -> None:
    """Export the live frame index, encode the pinned queries, keep the
    patches that decide their MaxSim scores, and write the corpus file."""
    from cogniverse_core.query.encoders import ColPaliFamilyQueryEncoder

    endpoint = urlsplit(os.environ.get("LIVE_VESPA_URL", "http://localhost:8080"))
    tenant = os.environ.get("RANKING_CORPUS_TENANT", "flywheel_org:production")
    live_schema = f"{BASE_SCHEMA}_{tenant.replace(':', '_')}"
    exporter = VespaSearchBackend(
        backend_url=f"{endpoint.scheme}://{endpoint.hostname}",
        backend_port=endpoint.port,
        schema_name=live_schema,
        enable_metrics=False,
        enable_connection_pool=False,
        is_schema_deployed=lambda tenant_id, base: True,
    )
    exported = exporter.export_embeddings(schema=live_schema, include_embeddings=True)
    encoder = ColPaliFamilyQueryEncoder(
        SHIPPED_PROFILE["embedding_model"],
        inference_service_url=os.environ.get(
            "COLPALI_INFERENCE_URL", "http://localhost:29001"
        ),
    )
    queries = [encoder.encode(text) for text in PINNED_QUERIES]

    frames = []
    for doc in exported:
        blocks = doc["embedding"]["blocks"]
        patches = np.array([blocks[key] for key in sorted(blocks, key=int)], np.float32)
        binary = doc["embedding_binary"]["blocks"]
        stored_binary = np.array(
            [binary[key] for key in sorted(binary, key=int)], np.int8
        ).view(np.uint8)
        assert np.array_equal(stored_binary, _binarize(patches)), doc["id"]
        keep: set[int] = set()
        for query in queries:
            keep |= set((query @ patches.T).argmax(axis=1).tolist())
            xor = np.bitwise_xor(_binarize(query)[:, None, :], stored_binary[None])
            keep |= set(_POPCOUNT[xor].sum(-1).argmin(axis=1).tolist())
        kept = patches[sorted(keep)]
        for query in queries:
            assert _float_max_sim(query, kept) == pytest.approx(
                _float_max_sim(query, patches), abs=1e-5
            )
            assert _binary_max_sim(query, kept) == _binary_max_sim(query, patches)
        frames.append((doc, kept))

    frames.sort(key=lambda frame: (frame[0]["video_id"], frame[0]["segment_id"]))
    np.savez_compressed(
        CORPUS_PATH,
        doc_ids=np.array([doc["id"].split("::", 1)[1] for doc, _ in frames]),
        video_ids=np.array([doc["video_id"] for doc, _ in frames]),
        video_titles=np.array([doc["video_title"] for doc, _ in frames]),
        segment_ids=np.array([doc["segment_id"] for doc, _ in frames], np.int32),
        start_times=np.array([doc["start_time"] for doc, _ in frames], np.float64),
        end_times=np.array([doc["end_time"] for doc, _ in frames], np.float64),
        patch_offsets=np.cumsum([0] + [len(kept) for _, kept in frames]),
        patches_bf16=_bf16_bits(np.concatenate([kept for _, kept in frames])),
        query_texts=np.array(PINNED_QUERIES),
        query_offsets=np.cumsum([0] + [len(query) for query in queries]),
        query_vectors=np.concatenate(queries).astype(np.float32),
        **{
            array: np.array([doc[field] for doc, _ in frames])
            for field, array in RECORDED_TEXT_ARRAYS.items()
        },
    )


def _load_corpus() -> dict:
    with np.load(CORPUS_PATH) as data:
        corpus = {name: data[name] for name in data.files}
    corpus["patches"] = _bf16_values(corpus.pop("patches_bf16"))
    return corpus


def _frame_patches(corpus: dict, index: int) -> np.ndarray:
    offsets = corpus["patch_offsets"]
    return corpus["patches"][offsets[index] : offsets[index + 1]]


def _query_vectors(corpus: dict) -> dict[str, np.ndarray]:
    offsets = corpus["query_offsets"]
    return {
        str(text): corpus["query_vectors"][offsets[i] : offsets[i + 1]]
        for i, text in enumerate(corpus["query_texts"])
    }


def _shipped_rerank_count() -> int:
    schema = json.loads((SCHEMAS_DIR / f"{BASE_SCHEMA}_schema.json").read_text())
    (phased,) = [p for p in schema["rank_profiles"] if p["name"] == "phased"]
    return phased["second_phase"]["rerank_count"]


def _validate_corpus(corpus: dict) -> None:
    """The recording holds the pinned queries over the full frame index."""
    dim = schema_tensor_dim(BASE_SCHEMA, "embedding")
    assert len(corpus["doc_ids"]) == RECORDED_FRAME_COUNT, (
        f"recording holds {len(corpus['doc_ids'])} frames, "
        f"expected {RECORDED_FRAME_COUNT}"
    )
    assert tuple(str(q) for q in corpus["query_texts"]) == PINNED_QUERIES, (
        f"recorded queries {corpus['query_texts'].tolist()}"
    )
    titles, counts = np.unique(
        [_video_key(str(t)) for t in corpus["video_titles"]], return_counts=True
    )
    frames_per_video = dict(zip(titles.tolist(), counts.tolist()))
    for query in PINNED_QUERIES:
        assert _expected_video(query) in frames_per_video, (
            f"expected video of {query!r} is not in the recording"
        )
    rerank_count = _shipped_rerank_count()
    assert max(frames_per_video.values()) > rerank_count, (
        f"no video's frames fill the rerank window of {rerank_count}"
    )
    assert frames_per_video == RECORDED_FRAMES_PER_VIDEO, (
        f"frames per video {frames_per_video}"
    )
    assert corpus["patches"].shape[1] == dim, (
        f"patch width {corpus['patches'].shape[1]}, schema declares {dim}"
    )
    assert corpus["query_vectors"].shape[1] == dim, (
        f"query width {corpus['query_vectors'].shape[1]}, schema declares {dim}"
    )
    assert corpus["patch_offsets"][-1] == len(corpus["patches"]), (
        "patch offsets do not cover the recorded patches"
    )
    assert {"video_title", *RECORDED_TEXT_ARRAYS} == _searched_text_fields(), (
        f"recorded text fields {sorted(RECORDED_TEXT_ARRAYS)}, schema searches "
        f"{sorted(_searched_text_fields())}"
    )
    for field, array in RECORDED_TEXT_ARRAYS.items():
        assert array in corpus, f"recording holds no {field}"
        empty = sum(1 for text in corpus[array] if not str(text).strip())
        assert len(corpus[array]) == RECORDED_FRAME_COUNT and empty == 0, (
            f"{field}: {len(corpus[array])} frames recorded, {empty} without text"
        )


@pytest.fixture(scope="module")
def corpus() -> dict:
    if RECORD_GOLDEN:
        _record_corpus()
    recorded = _load_corpus()
    _validate_corpus(recorded)
    return recorded


@pytest.fixture(scope="module")
def ranked_frames(vespa_instance, corpus):
    """The recording fed into a tenant schema deployed from configs/schemas."""
    tenant = f"colqwenrank{uuid.uuid4().hex[:8]}:unit"
    config_manager = make_config_manager(vespa_instance)
    ingestion = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
    )
    ingestion.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name=BASE_SCHEMA
    )
    documents = []
    for i, doc_id in enumerate(corpus["doc_ids"]):
        document = Document(
            id=str(doc_id),
            content_type=ContentType.VIDEO,
            content_id=str(corpus["video_ids"][i]),
            status=ProcessingStatus.COMPLETED,
        )
        document.add_embedding(
            "embedding", _frame_patches(corpus, i), {"type": "float", "raw": True}
        )
        document.add_metadata("video_id", str(corpus["video_ids"][i]))
        document.add_metadata("video_title", str(corpus["video_titles"][i]))
        document.add_metadata("segment_index", int(corpus["segment_ids"][i]))
        document.add_metadata("start_time", float(corpus["start_times"][i]))
        document.add_metadata("end_time", float(corpus["end_times"][i]))
        for field, key in _text_metadata_keys().items():
            document.add_metadata(key, str(corpus[RECORDED_TEXT_ARRAYS[field]][i]))
        documents.append(document)
    for start in range(0, len(documents), 50):
        result = ingestion.ingest_documents(documents[start : start + 50], BASE_SCHEMA)
        assert result["failed_count"] == 0, result["failed_documents"]
    schema_name = ingestion.get_tenant_schema_name(tenant, BASE_SCHEMA)
    indexed = requests.post(
        f"http://localhost:{vespa_instance['http_port']}/search/",
        json={"yql": f"select * from {schema_name} where true", "hits": 0},
        timeout=30,
    ).json()["root"]["fields"]["totalCount"]
    assert indexed == RECORDED_FRAME_COUNT
    stored = requests.get(
        f"http://localhost:{vespa_instance['http_port']}/document/v1/"
        f"{document_namespace(schema_name)}/{schema_name}/docid/"
        f"{corpus['doc_ids'][0]}",
        timeout=30,
    ).json()["fields"]
    assert {field: stored[field] for field in RECORDED_TEXT_ARRAYS} == {
        field: str(corpus[array][0]) for field, array in RECORDED_TEXT_ARRAYS.items()
    }

    search = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": vespa_instance["http_port"],
            "profiles": {BASE_SCHEMA: SHIPPED_PROFILE},
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        is_schema_deployed=DeployedSchemaNames(config_manager),
        enable_connection_pool=False,
    )
    yield search, tenant, _query_vectors(corpus)
    search.close()


def _search(ranked_frames, query: str, strategy: str):
    search, tenant, vectors = ranked_frames
    return search.search(
        {
            "query": query,
            "type": "video",
            "profile": BASE_SCHEMA,
            "strategy": strategy,
            "top_k": 10,
            "tenant_id": tenant,
            "query_embeddings": vectors[query],
        }
    )


def _ranked_videos(results) -> list[str]:
    return [_video_key(r.document.metadata["video_title"]) for r in results]


# Video order the float MaxSim gives over the full frame index, which the
# default strategy must reproduce.
FLOAT_ORDER = {
    "people shoveling": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo", "v_-uJnucdW6DY"],
    "a fire starter": ["v_-D1gdv_gQyw", "v_-uJnucdW6DY", "v_-MbZ-W0AbN0"],
    "catching": ["v_-uJnucdW6DY", "v_-MbZ-W0AbN0", "v_-D1gdv_gQyw"],
}
BINARY_ORDER = {
    "people shoveling": ["v_-IMXSEIabMM", "v_-pkfcMUIEMo"],
    "a fire starter": ["v_-D1gdv_gQyw", "v_-MbZ-W0AbN0"],
    "catching": ["v_-uJnucdW6DY", "v_-MbZ-W0AbN0"],
}


class TestDefaultRankingOnColQwen3Frames:
    @pytest.mark.parametrize("query", PINNED_QUERIES)
    @pytest.mark.parametrize("strategy", ["default", "phased", "float_float"])
    def test_expected_video_ranks_first_in_float_order(
        self, ranked_frames, query, strategy
    ):
        ranked = _ranked_videos(_search(ranked_frames, query, strategy))

        assert ranked[0] == _expected_video(query)
        assert ranked[:3] == FLOAT_ORDER[query]
        assert len(ranked) == len(RECORDED_FRAMES_PER_VIDEO)

    @pytest.mark.parametrize("query", PINNED_QUERIES)
    def test_binary_strategy_ranks_by_content_tokens(self, ranked_frames, query):
        ranked = _ranked_videos(_search(ranked_frames, query, "binary_binary"))

        assert ranked[:2] == BINARY_ORDER[query]

    @pytest.mark.parametrize("query", PINNED_QUERIES)
    def test_scores_are_the_maxsim_of_the_best_frame(
        self, ranked_frames, corpus, query
    ):
        """Vespa's scores equal the MaxSim formulas over the recorded patches."""
        vectors = _query_vectors(corpus)[query]
        best = {}
        for i, title in enumerate(corpus["video_titles"]):
            patches = _frame_patches(corpus, i)
            key = _video_key(str(title))
            float_score = _float_max_sim(vectors, patches)
            binary_score = _binary_max_sim(vectors, patches)
            prior = best.get(key, (-np.inf, -np.inf))
            best[key] = (max(prior[0], float_score), max(prior[1], binary_score))

        default_top = _search(ranked_frames, query, "default")[0]
        binary_top = _search(ranked_frames, query, "binary_binary")[0]

        assert default_top.score == pytest.approx(
            best[FLOAT_ORDER[query][0]][0], abs=1e-4
        )
        assert binary_top.score == pytest.approx(
            best[BINARY_ORDER[query][0]][1], abs=1e-5
        )


HYBRID_STRATEGIES = (
    "hybrid_float_bm25",
    "hybrid_binary_bm25",
    "hybrid_float_bm25_no_description",
    "hybrid_binary_bm25_no_description",
)


def _native_ranks(ranked_frames, query: str, strategy: str) -> dict[str, float]:
    """Vespa's nativeRank of every frame for the strategy's own query."""
    search, tenant, vectors = ranked_frames
    schema_name = f"{BASE_SCHEMA}_{tenant.replace(':', '_')}"
    body = search._build_query(
        query,
        vectors[query],
        search._load_ranking_strategies()[BASE_SCHEMA][strategy],
        strategy,
        schema_name,
        RECORDED_FRAME_COUNT,
        {},
        "native-rank",
    )
    response = requests.post(
        f"{search.backend_url}:{search.backend_port}/search/",
        json={**body, "ranking.listFeatures": True},
        timeout=60,
    ).json()
    hits = response["root"]["children"]
    assert len(hits) == RECORDED_FRAME_COUNT
    return {
        h["fields"]["documentid"].split("::", 1)[1]: h["fields"]["rankfeatures"][
            "nativeRank"
        ]
        for h in hits
    }


def _fused_scores(corpus, ranked_frames, query: str, strategy: str) -> dict:
    """Each frame's visual MaxSim, normalized by query length, plus its
    nativeRank."""
    vectors = _query_vectors(corpus)[query]
    native = _native_ranks(ranked_frames, query, strategy)
    max_sim = _binary_max_sim if "binary" in strategy else _float_max_sim
    return {
        str(doc_id): max_sim(vectors, _frame_patches(corpus, i)) / len(vectors)
        + native[str(doc_id)]
        for i, doc_id in enumerate(corpus["doc_ids"])
    }


# Video order and best frame the fused hybrids give over the full frame index.
HYBRID_ORDER = {
    "hybrid_float_bm25": {
        "people shoveling": [
            "v_-pkfcMUIEMo",
            "v_-IMXSEIabMM",
            "v_-uJnucdW6DY",
            "v_-D1gdv_gQyw",
            "v_-MbZ-W0AbN0",
            "v_-nl4G-00PtA",
            "v_-vnSFKJNB94",
            "v_-cAcA8dO7kA",
            "v_-6dz6tBH77I",
            "v_-HpCLXdtcas",
        ],
        "a fire starter": [
            "v_-D1gdv_gQyw",
            "v_-uJnucdW6DY",
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-MbZ-W0AbN0",
            "v_-nl4G-00PtA",
            "v_-vnSFKJNB94",
            "v_-6dz6tBH77I",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
        ],
        "catching": [
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-D1gdv_gQyw",
            "v_-6dz6tBH77I",
            "v_-IMXSEIabMM",
            "v_-HpCLXdtcas",
            "v_-pkfcMUIEMo",
            "v_-cAcA8dO7kA",
            "v_-vnSFKJNB94",
            "v_-nl4G-00PtA",
        ],
    },
    "hybrid_binary_bm25": {
        "people shoveling": [
            "v_-pkfcMUIEMo",
            "v_-IMXSEIabMM",
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-nl4G-00PtA",
            "v_-D1gdv_gQyw",
            "v_-vnSFKJNB94",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
            "v_-6dz6tBH77I",
        ],
        "a fire starter": [
            "v_-D1gdv_gQyw",
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-vnSFKJNB94",
            "v_-nl4G-00PtA",
            "v_-6dz6tBH77I",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
        ],
        "catching": [
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-D1gdv_gQyw",
            "v_-vnSFKJNB94",
            "v_-IMXSEIabMM",
            "v_-HpCLXdtcas",
            "v_-pkfcMUIEMo",
            "v_-6dz6tBH77I",
            "v_-nl4G-00PtA",
            "v_-cAcA8dO7kA",
        ],
    },
    "hybrid_float_bm25_no_description": {
        "people shoveling": [
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-uJnucdW6DY",
            "v_-D1gdv_gQyw",
            "v_-MbZ-W0AbN0",
            "v_-vnSFKJNB94",
            "v_-cAcA8dO7kA",
            "v_-6dz6tBH77I",
            "v_-nl4G-00PtA",
            "v_-HpCLXdtcas",
        ],
        "a fire starter": [
            "v_-D1gdv_gQyw",
            "v_-uJnucdW6DY",
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-MbZ-W0AbN0",
            "v_-vnSFKJNB94",
            "v_-nl4G-00PtA",
            "v_-6dz6tBH77I",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
        ],
        "catching": [
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-D1gdv_gQyw",
            "v_-6dz6tBH77I",
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-cAcA8dO7kA",
            "v_-vnSFKJNB94",
            "v_-nl4G-00PtA",
            "v_-HpCLXdtcas",
        ],
    },
    "hybrid_binary_bm25_no_description": {
        "people shoveling": [
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-D1gdv_gQyw",
            "v_-nl4G-00PtA",
            "v_-vnSFKJNB94",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
            "v_-6dz6tBH77I",
        ],
        "a fire starter": [
            "v_-D1gdv_gQyw",
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-vnSFKJNB94",
            "v_-nl4G-00PtA",
            "v_-6dz6tBH77I",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
        ],
        "catching": [
            "v_-uJnucdW6DY",
            "v_-MbZ-W0AbN0",
            "v_-D1gdv_gQyw",
            "v_-vnSFKJNB94",
            "v_-IMXSEIabMM",
            "v_-pkfcMUIEMo",
            "v_-6dz6tBH77I",
            "v_-nl4G-00PtA",
            "v_-cAcA8dO7kA",
            "v_-HpCLXdtcas",
        ],
    },
}
_SHOVEL_FRAME = (
    "ad10d4d00bde8e6aaf479ec572da90c3ea359446e64749bbe75158eb4c22c1e4_seg_31"
)
_SHOVEL_FRAME_NO_DESCRIPTION = (
    "a1e071ec33a0937c3cb67ab4780f9c3214cf00c36fe4e0dbf9991fc679bf3b17_seg_1"
)
_FIRE_FRAME = "7a3f548576b6e9070d4604e883c6a98c78e22c2862a21af70d57e887457f047d_seg_8"
_CATCH_FRAME = "739064ebcf629a4ca93a7bb50177ff6cdab0d539a8008bba1dae766b957f8f2d_seg_8"
HYBRID_TOP_HIT = {
    "hybrid_float_bm25": {
        "people shoveling": (_SHOVEL_FRAME, 0.824828197385741),
        "a fire starter": (_FIRE_FRAME, 0.8255090974366946),
        "catching": (_CATCH_FRAME, 0.7297115070479256),
    },
    "hybrid_binary_bm25": {
        "people shoveling": (_SHOVEL_FRAME, 0.7580665447571603),
        "a fire starter": (_FIRE_FRAME, 0.7270843125975414),
        "catching": (_CATCH_FRAME, 0.65625),
    },
    "hybrid_float_bm25_no_description": {
        "people shoveling": (_SHOVEL_FRAME_NO_DESCRIPTION, 0.8119350313513485),
        "a fire starter": (_FIRE_FRAME, 0.7953657310467892),
        "catching": (_CATCH_FRAME, 0.7297115070479256),
    },
    "hybrid_binary_bm25_no_description": {
        "people shoveling": (_SHOVEL_FRAME_NO_DESCRIPTION, 0.7329315549170647),
        "a fire starter": (_FIRE_FRAME, 0.6969409462076359),
        "catching": (_CATCH_FRAME, 0.65625),
    },
}


class TestHybridFusionOnColQwen3Frames:
    @pytest.mark.parametrize("query", PINNED_QUERIES)
    @pytest.mark.parametrize("strategy", HYBRID_STRATEGIES)
    def test_hybrid_ranks_every_video_in_the_recorded_order(
        self, ranked_frames, query, strategy
    ):
        results = _search(ranked_frames, query, strategy)
        top_frame, top_score = HYBRID_TOP_HIT[strategy][query]

        assert _ranked_videos(results) == HYBRID_ORDER[strategy][query]
        assert results[0].document.id == top_frame
        assert results[0].score == pytest.approx(top_score, abs=1e-6)

    @pytest.mark.parametrize("query", PINNED_QUERIES)
    @pytest.mark.parametrize("strategy", HYBRID_STRATEGIES[:2])
    def test_every_video_scores_its_best_frame_fused_score(
        self, ranked_frames, corpus, query, strategy
    ):
        """Over all 361 frames: the visual MaxSim from the recorded patches,
        averaged over the query tokens, plus Vespa's nativeRank of the frame
        for the same query decides each video's score and best frame."""
        fused = _fused_scores(corpus, ranked_frames, query, strategy)
        best = {}
        for i, doc_id in enumerate(corpus["doc_ids"]):
            key = _video_key(str(corpus["video_titles"][i]))
            if fused[str(doc_id)] > best.get(key, (-np.inf, ""))[0]:
                best[key] = (fused[str(doc_id)], str(doc_id))

        results = _search(ranked_frames, query, strategy)

        assert _ranked_videos(results) == sorted(best, key=lambda v: -best[v][0])
        assert [r.document.id for r in results] == [
            best[_video_key(r.document.metadata["video_title"])][1] for r in results
        ]
        for result in results:
            video = _video_key(result.document.metadata["video_title"])
            assert result.score == pytest.approx(best[video][0], abs=1e-5)

    def test_concurrent_hybrid_searches_return_their_recorded_rankings(
        self, ranked_frames
    ):
        """Hybrid searches released together through one backend each get
        exactly the ranking their query and strategy get alone: the recorded
        video order and top frame."""
        cases = [(q, s) for q in PINNED_QUERIES for s in HYBRID_STRATEGIES] * 2
        barrier = threading.Barrier(len(cases))

        def released(case):
            barrier.wait(timeout=60)
            return _search(ranked_frames, *case)

        with ThreadPoolExecutor(max_workers=len(cases)) as pool:
            concurrent = list(pool.map(released, cases))

        assert [_ranked_videos(results) for results in concurrent] == [
            HYBRID_ORDER[strategy][query] for query, strategy in cases
        ]
        assert [results[0].document.id for results in concurrent] == [
            HYBRID_TOP_HIT[strategy][query][0] for query, strategy in cases
        ]
        assert [results[0].score for results in concurrent] == pytest.approx(
            [HYBRID_TOP_HIT[strategy][query][1] for query, strategy in cases],
            abs=1e-6,
        )

    def test_hybrid_search_raises_when_vespa_is_unreachable(
        self, ranked_frames, vespa_instance
    ):
        _, tenant, vectors = ranked_frames
        config_manager = make_config_manager(vespa_instance)
        port = _dead_port()
        unreachable = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": port,
                "profiles": {BASE_SCHEMA: SHIPPED_PROFILE},
            },
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
            is_schema_deployed=DeployedSchemaNames(config_manager),
            enable_connection_pool=False,
            retry_config=RetryConfig(max_attempts=1),
        )
        try:
            with pytest.raises(httpr.ConnectError) as excinfo:
                unreachable.search(
                    {
                        "query": PINNED_QUERIES[0],
                        "type": "video",
                        "profile": BASE_SCHEMA,
                        "strategy": "hybrid_float_bm25",
                        "top_k": 10,
                        "tenant_id": tenant,
                        "query_embeddings": vectors[PINNED_QUERIES[0]],
                    }
                )
        finally:
            unreachable.close()

        assert str(excinfo.value) == (
            f"error sending request for url (http://localhost:{port}/search/)"
        )


class TestRecordingPins:
    """Each pin on the recording fails when the property it guards changes."""

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda c: c.update(doc_ids=c["doc_ids"][:-1]),
                f"recording holds {RECORDED_FRAME_COUNT - 1} frames",
            ),
            (
                lambda c: c.update(
                    video_titles=np.where(
                        c["video_titles"] == "v_-HpCLXdtcas.mkv",
                        "v_-other.mp4",
                        c["video_titles"],
                    )
                ),
                "frames per video",
            ),
            (
                lambda c: c.update(patches=c["patches"][:, :128]),
                "patch width 128, schema declares 320",
            ),
            (
                lambda c: c.update(query_vectors=c["query_vectors"][:, :128]),
                "query width 128, schema declares 320",
            ),
            (
                lambda c: c.update(patch_offsets=c["patch_offsets"] + 1),
                "patch offsets do not cover the recorded patches",
            ),
            (
                lambda c: c.update(
                    query_texts=np.array(["a dog", *PINNED_QUERIES[1:]])
                ),
                "recorded queries",
            ),
            (
                lambda c: c.pop("audio_transcripts"),
                "recording holds no audio_transcript",
            ),
            (
                lambda c: c.update(
                    segment_descriptions=np.array(["", *c["segment_descriptions"][1:]])
                ),
                "segment_description: 361 frames recorded, 1 without text",
            ),
            (
                lambda c: c.update(audio_transcripts=c["audio_transcripts"][:-1]),
                "audio_transcript: 360 frames recorded, 0 without text",
            ),
        ],
    )
    def test_pin_fails_on_mutated_recording(self, corpus, mutate, message):
        mutated = copy.deepcopy(corpus)
        mutate(mutated)

        with pytest.raises(AssertionError, match=message):
            _validate_corpus(mutated)

    def test_expected_video_pin_fails_when_its_frames_are_relabelled(self, corpus):
        mutated = copy.deepcopy(corpus)
        titles = mutated["video_titles"]
        mutated["video_titles"] = np.where(
            titles == "v_-D1gdv_gQyw.mp4", "v_-other.mp4", titles
        )

        with pytest.raises(AssertionError, match="expected video of 'a fire starter'"):
            _validate_corpus(mutated)

    def test_text_field_pin_fails_when_the_schema_searches_another_field(
        self, corpus, monkeypatch
    ):
        searched = _searched_text_fields() | {"video_tags"}
        monkeypatch.setattr(
            sys.modules[__name__], "_searched_text_fields", lambda: searched
        )

        with pytest.raises(AssertionError, match="schema searches"):
            _validate_corpus(copy.deepcopy(corpus))

    def test_rerank_window_pin_fails_when_no_video_fills_it(self, corpus):
        mutated = copy.deepcopy(corpus)
        titles = mutated["video_titles"].copy()
        hub = np.flatnonzero(titles == "v_-uJnucdW6DY.mp4")
        titles[hub[::2]] = "v_-other.mp4"
        mutated["video_titles"] = titles

        with pytest.raises(
            AssertionError, match="no video's frames fill the rerank window of 100"
        ):
            _validate_corpus(mutated)


SHIPPED_PROFILES = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
]
COLQWEN3_PROFILES = [
    name
    for name, profile in SHIPPED_PROFILES.items()
    if profile.get("embedding_model") == SHIPPED_PROFILE["embedding_model"]
]


def _schema_fields(base_schema: str) -> tuple[str, str, str]:
    """The source identity, float patch and binary patch field names."""
    schema = json.loads((SCHEMAS_DIR / f"{base_schema}_schema.json").read_text())
    float_field = schema["document_mapping"]["embeddings"]
    (float_field,) = float_field.values()
    return schema["document_mapping"]["id"], float_field, f"{float_field}_binary"


class TestLinearBinaryMaxSimOnEveryColQwen3Schema:
    """``binary_binary`` scores ``sum_t (1 - 2 h_t / 320)`` on each schema the
    ColQwen3 encoder serves: one query token matches a patch bit-for-bit, the
    other sits 64 bits from its nearest patch, so the score is
    ``1 + (1 - 128/320) = 1.6``."""

    def test_every_colqwen3_profile_is_covered(self):
        assert sorted(COLQWEN3_PROFILES) == [
            "document_visual_colpali",
            "image_colpali_mv",
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]

    @pytest.mark.parametrize("profile_name", sorted(COLQWEN3_PROFILES))
    def test_binary_score_counts_each_token_linearly(
        self, vespa_instance, profile_name
    ):
        profile = SHIPPED_PROFILES[profile_name]
        base_schema = profile["schema_name"]
        id_field, float_field, binary_field = _schema_fields(base_schema)
        dim = schema_tensor_dim(base_schema, float_field)
        tenant = f"linbin{uuid.uuid4().hex[:8]}:unit"
        config_manager = make_config_manager(vespa_instance)
        ingestion = BackendRegistry.get_instance().get_ingestion_backend(
            name="vespa",
            tenant_id=tenant,
            config={
                "backend": {
                    "url": "http://localhost",
                    "config_port": vespa_instance["config_port"],
                    "port": vespa_instance["http_port"],
                }
            },
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        )
        schema_name = ingestion.schema_registry.deploy_schema(
            tenant_id=tenant, base_schema_name=base_schema
        )
        patches = np.stack([np.full(dim, 0.05), np.full(dim, -0.05)]).astype(np.float32)
        response = requests.post(
            f"http://localhost:{vespa_instance['http_port']}/document/v1/"
            f"content/{schema_name}/docid/linear_binary_doc",
            json={
                "fields": {
                    id_field: "linear_binary_doc",
                    float_field: {
                        "blocks": {str(i): p.tolist() for i, p in enumerate(patches)}
                    },
                    binary_field: {
                        "blocks": {
                            str(i): p.tolist()
                            for i, p in enumerate(_binarize(patches).view(np.int8))
                        }
                    },
                }
            },
            timeout=30,
        )
        assert response.status_code == 200, response.text
        query = np.full((2, dim), -0.05, dtype=np.float32)
        query[0] = 0.05
        query[1, :64] = 0.05
        assert _min_hamming(query, patches).tolist() == [0, 64]

        search = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": vespa_instance["http_port"],
                "profiles": {profile_name: profile},
            },
            config_manager=config_manager,
            schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
            is_schema_deployed=DeployedSchemaNames(config_manager),
            enable_connection_pool=False,
        )
        try:
            results = search.search(
                {
                    "query": "",
                    "type": profile["type"],
                    "profile": profile_name,
                    "strategy": "binary_binary",
                    "top_k": 10,
                    "tenant_id": tenant,
                    "query_embeddings": query,
                }
            )
        finally:
            search.close()

        assert [r.document.metadata[id_field] for r in results] == ["linear_binary_doc"]
        assert results[0].score == pytest.approx(1.0 + (1.0 - 128 / dim), abs=1e-6)
