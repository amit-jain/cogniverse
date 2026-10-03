"""Hybrid rank profiles, against real Vespa, on every schema.

A visual-first hybrid ranks every document by the sum of its normalized
embedding similarity and ``nativeRank`` over the profile's text fields, both
in its first phase:

- float MaxSim: the mean over query tokens of the best dot product with a
  stored token/patch vector;
- binary MaxSim: the mean over query tokens of ``1 - 2h/bits``, ``h`` the
  Hamming distance to the nearest stored vector;
- a dense float vector: ``closeness`` under the field's distance metric;
- a dense binary vector: the angular closeness ``1/(1 + θ)`` with the angle
  estimated from the Hamming distance as ``θ = πh/bits``.

A text-first hybrid (``hybrid_bm25_*``, ``"candidates": "text_matches"``)
ranks the documents matching the query text, and no others, by the same sum
of visual similarity and ``nativeRank``.

Each profile is searched through ``VespaSearchBackend``. A visual-first hybrid
gets two documents holding the same embedding, one of whose text fields carry
the query term: both come back, the text-less one scored by the visual
similarity alone and the other by that similarity plus its ``nativeRank``. A
text-first hybrid gets two documents holding the term with different
embeddings and one without it: the two text matches come back, each scored by
its own visual similarity plus its ``nativeRank``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
from cogniverse_vespa.ranking_strategy_extractor import RankingStrategyExtractor
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = [pytest.mark.integration]

SCHEMAS_DIR = Path("configs/schemas")
SHIPPED_PROFILES = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
]
QUERY_TERM = "kestrel"
TEXT_WITH_TERM = "kestrel harbour notes"
TEXT_WITHOUT_TERM = "harbour notes"
# Vespa's nativeRank of a document holding the one query term once in each
# of its text fields, beside a document holding it nowhere.
TERM_DOC_NATIVE_RANK = 0.3818623835995125


def _schema(base: str, schemas_dir: Path = SCHEMAS_DIR) -> dict:
    return json.loads((schemas_dir / f"{base}_schema.json").read_text())


def _profile(base: str, name: str, schemas_dir: Path = SCHEMAS_DIR) -> dict:
    (profile,) = [
        p for p in _schema(base, schemas_dir)["rank_profiles"] if p["name"] == name
    ]
    return profile


def _phase_terms(profile: dict, phase: str) -> list[str]:
    """The phase expression's summed terms, each profile function expanded."""
    functions = {f["name"]: f["expression"] for f in profile.get("functions", [])}
    expression = profile.get(phase)
    if expression is None:
        return []
    if isinstance(expression, dict):
        expression = expression["expression"]
    return [functions.get(term, term) for term in expression.split(" + ")]


def _scores_embedding(term: str) -> bool:
    return "attribute(" in term or "closeness(field," in term


def _is_fused(terms: list[str]) -> bool:
    return (
        len(terms) == 2
        and _scores_embedding(terms[0])
        and terms[1].startswith("nativeRank(")
    )


def _hybrids(schemas_dir: Path) -> list[tuple[str, dict, object]]:
    found = []
    for path in sorted(schemas_dir.glob("*_schema.json")):
        base = path.name.removesuffix("_schema.json")
        strategies = RankingStrategyExtractor().extract_from_schema(path)
        for name, info in strategies.items():
            if info.needs_text_query:
                found.append((base, _profile(base, name, schemas_dir), info))
    return found


def _fused_hybrids(schemas_dir: Path = SCHEMAS_DIR) -> list[tuple[str, str]]:
    """Every (schema, profile) that ranks every document by the sum of an
    embedding similarity and a nativeRank text score in its first phase."""
    return [
        (base, profile["name"])
        for base, profile, info in _hybrids(schemas_dir)
        if info.first_phase_embedding_field
        and not info.text_candidates_only
        and _is_fused(_phase_terms(profile, "first_phase"))
    ]


def _text_first_hybrids(schemas_dir: Path = SCHEMAS_DIR) -> list[tuple[str, str]]:
    """Every (schema, profile) that ranks the text matches alone by the sum of
    an embedding similarity and a nativeRank text score in its first phase."""
    return [
        (base, profile["name"])
        for base, profile, info in _hybrids(schemas_dir)
        if info.text_candidates_only
        and info.first_phase_embedding_field
        and _is_fused(_phase_terms(profile, "first_phase"))
    ]


FUSED_HYBRIDS = [
    ("audio_content", "hybrid_semantic_bm25"),
    ("audio_content", "hybrid_acoustic_bm25"),
    ("code_lateon_mv", "hybrid_float_bm25"),
    ("document_text", "hybrid_float_bm25"),
    ("document_text", "hybrid_binary_bm25"),
    ("document_visual", "hybrid_float_bm25"),
    ("document_visual", "hybrid_binary_bm25"),
    ("image_colpali_mv", "hybrid_float_bm25"),
    ("image_colpali_mv", "hybrid_binary_bm25"),
    ("knowledge_graph", "hybrid_binary_bm25"),
    ("knowledge_graph", "hybrid_float_bm25"),
    ("lateon_mv", "hybrid_binary_bm25"),
    ("lateon_mv", "hybrid_float_bm25"),
    ("video_colpali_smol500_mv_frame", "hybrid_float_bm25"),
    ("video_colpali_smol500_mv_frame", "hybrid_binary_bm25"),
    ("video_colpali_smol500_mv_frame", "hybrid_float_bm25_no_description"),
    ("video_colpali_smol500_mv_frame", "hybrid_binary_bm25_no_description"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_float_bm25"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_binary_bm25"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_float_bm25_no_description"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_binary_bm25_no_description"),
    ("video_xclip_sv_chunk_6s", "hybrid_float_bm25"),
    ("video_xclip_sv_chunk_6s", "hybrid_binary_bm25"),
]
TEXT_FIRST_HYBRIDS = [
    ("video_colpali_smol500_mv_frame", "hybrid_bm25_binary"),
    ("video_colpali_smol500_mv_frame", "hybrid_bm25_float"),
    ("video_colpali_smol500_mv_frame", "hybrid_bm25_binary_no_description"),
    ("video_colpali_smol500_mv_frame", "hybrid_bm25_float_no_description"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_bm25_binary"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_bm25_float"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_bm25_binary_no_description"),
    ("video_colqwen_omni_mv_chunk_30s", "hybrid_bm25_float_no_description"),
    ("video_xclip_sv_chunk_6s", "hybrid_bm25_binary"),
    ("video_xclip_sv_chunk_6s", "hybrid_bm25_float"),
]


def test_every_visual_first_hybrid_fuses_with_native_rank():
    assert _fused_hybrids() == FUSED_HYBRIDS


def test_every_text_first_hybrid_ranks_text_matches_with_native_rank():
    assert _text_first_hybrids() == TEXT_FIRST_HYBRIDS


def test_every_hybrid_with_an_embedding_and_bm25_in_its_name_is_fused():
    """No hybrid outside the two forms still sums BM25 with a similarity or
    lets one of them rank alone."""
    hybrids = {
        (base, profile["name"])
        for base, profile, info in _hybrids(SCHEMAS_DIR)
        if "bm25" in profile["name"]
        and (info.needs_float_embeddings or info.needs_binary_embeddings)
    }

    assert sorted(hybrids - set(_fused_hybrids()) - set(_text_first_hybrids())) == []


def _mutated_schemas(tmp_path: Path, base: str, name: str, **changes) -> Path:
    shutil.copytree(SCHEMAS_DIR, tmp_path, dirs_exist_ok=True)
    path = tmp_path / f"{base}_schema.json"
    schema = json.loads(path.read_text())
    (profile,) = [p for p in schema["rank_profiles"] if p["name"] == name]
    profile.update(changes)
    path.write_text(json.dumps(schema))
    return tmp_path


def test_a_hybrid_reranking_by_text_alone_is_not_fused(tmp_path):
    """The derivation drops a hybrid whose text score replaces the visual one
    in a second phase."""
    mutated = _mutated_schemas(
        tmp_path,
        "video_colpali_smol500_mv_frame",
        "hybrid_float_bm25",
        first_phase="visual_sim_float",
        second_phase={"expression": "text_sim", "rerank_count": 100},
    )

    assert _fused_hybrids(mutated) == [
        hybrid
        for hybrid in FUSED_HYBRIDS
        if hybrid != ("video_colpali_smol500_mv_frame", "hybrid_float_bm25")
    ]


def test_a_text_first_hybrid_ranking_by_visual_alone_is_not_fused(tmp_path):
    """The derivation drops a text-first hybrid that ranks its text matches by
    the visual similarity alone."""
    mutated = _mutated_schemas(
        tmp_path,
        "video_xclip_sv_chunk_6s",
        "hybrid_bm25_float",
        first_phase="visual_sim",
    )

    assert _text_first_hybrids(mutated) == [
        hybrid
        for hybrid in TEXT_FIRST_HYBRIDS
        if hybrid != ("video_xclip_sv_chunk_6s", "hybrid_bm25_float")
    ]


def test_a_text_first_hybrid_without_its_candidate_set_is_visual_first(tmp_path):
    """Without ``candidates: text_matches`` a fused profile ranks every
    document, so the derivation counts it visual-first."""
    shutil.copytree(SCHEMAS_DIR, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "video_xclip_sv_chunk_6s_schema.json"
    schema = json.loads(path.read_text())
    (profile,) = [
        p for p in schema["rank_profiles"] if p["name"] == "hybrid_bm25_float"
    ]
    del profile["candidates"]
    path.write_text(json.dumps(schema))

    assert ("video_xclip_sv_chunk_6s", "hybrid_bm25_float") in _fused_hybrids(tmp_path)
    assert ("video_xclip_sv_chunk_6s", "hybrid_bm25_float") not in (
        _text_first_hybrids(tmp_path)
    )


def _field_type(base: str, field: str) -> str:
    (declared,) = [
        f["type"] for f in _schema(base)["document"]["fields"] if f["name"] == field
    ]
    return declared


def _width(tensor_type: str) -> int:
    return int(re.search(r"v\[(\d+)\]", tensor_type).group(1))


def _text_fields(base: str) -> list[str]:
    return [
        f["name"]
        for f in _schema(base)["document"]["fields"]
        if f["type"] == "string" and "index" in f.get("indexing", [])
    ]


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def _bits(vectors: np.ndarray) -> np.ndarray:
    return np.packbits((vectors > 0).astype(np.uint8), axis=-1)


def _distance_metric(base: str, field: str) -> str:
    (declared,) = [f for f in _schema(base)["document"]["fields"] if f["name"] == field]
    for setting in declared.get("attribute", []):
        if setting.replace(" ", "").startswith("distance-metric:"):
            return setting.split(":", 1)[1].strip()
    return "euclidean"


def _dense_closeness(base: str, field: str, query, stored) -> float:
    metric = _distance_metric(base, field)
    if metric == "angular":
        cosine = np.dot(query, stored) / (
            np.linalg.norm(query) * np.linalg.norm(stored)
        )
        return float(1.0 / (1.0 + np.arccos(np.clip(cosine, -1.0, 1.0))))
    if metric == "euclidean":
        return float(1.0 / (1.0 + np.linalg.norm(query - stored)))
    raise AssertionError(f"{base}.{field}: no fixture for distance metric {metric}")


def _hamming(query_bits: np.ndarray, stored_bits: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(query_bits, stored_bits)].sum())


def _fixture(base: str, name: str):
    """The query vectors, and two stored embeddings with the visual score
    each must produce, the first scoring higher."""
    info = RankingStrategyExtractor().extract_from_schema(
        SCHEMAS_DIR / f"{base}_schema.json"
    )[name]
    field = info.embedding_field
    stored_type = _field_type(base, field)
    (query_type,) = info.inputs.values()
    binary = "int8" in query_type
    dim = _width(query_type) * (8 if binary else 1)
    if "{}" not in stored_type and binary:
        # Dense binary vector: angular closeness, angle estimated as pi*h/bits.
        query = np.ones(dim, np.float32)
        query[:64] = -1.0
        variants = []
        for sign in (1.0, -1.0):
            stored = _bits(np.full(dim, sign, np.float32))
            h = _hamming(_bits(query), stored)
            variants.append(
                (stored.view(np.int8).tolist(), 1.0 / (1.0 + math.pi * h / dim))
            )
        return field, query, variants
    if "{}" not in stored_type:
        query = np.zeros(dim, np.float32)
        query[:2] = [1.0, 1.0]
        variants = []
        for axis in (0, 2):
            stored = np.zeros(dim, np.float32)
            stored[axis] = 1.0
            variants.append(
                (stored.tolist(), _dense_closeness(base, field, query, stored))
            )
        return field, query, variants
    query = np.zeros((2, dim), np.float32)
    query[0, 0] = 1.0
    query[1, 1] = 0.5
    patches = np.zeros((2, dim), np.float32)
    patches[0, 0] = 0.75
    patches[1, :2] = [0.125, 0.25]
    if binary:
        query = np.where(query > 0, 1.0, -1.0).astype(np.float32)
        query[1, : dim // 4] = -query[1, : dim // 4]
        patches = np.where(patches > 0, 1.0, -1.0).astype(np.float32)
    variants = []
    for stored_patches in (patches, -patches):
        if binary:
            xor = np.bitwise_xor(_bits(query)[:, None, :], _bits(stored_patches)[None])
            nearest = _POPCOUNT[xor].sum(-1).min(axis=1)
            visual = float(np.mean(1.0 - 2.0 * nearest / dim))
            cells = _bits(stored_patches).view(np.int8)
        else:
            visual = float(np.mean((query @ stored_patches.T).max(axis=1)))
            cells = stored_patches
        stored = {"blocks": {str(i): c.tolist() for i, c in enumerate(cells)}}
        variants.append((stored, visual))
    return field, query, variants


def _fixture_vectors(base: str, name: str):
    """Query vectors, a stored embedding and the visual score it must produce."""
    field, query, ((stored, visual), _) = _fixture(base, name)
    return field, query, stored, visual


def _shipped_profile(base: str) -> tuple[str, dict]:
    for profile_name, profile in SHIPPED_PROFILES.items():
        if profile.get("schema_name") == base:
            return profile_name, profile
    return base, {"type": "document", "schema_name": base}


@pytest.fixture(scope="module")
def hybrid_corpus(vespa_instance):
    """A tenant with every schema that ships a hybrid deployed."""
    tenant = f"hybridfuse{uuid.uuid4().hex[:8]}:unit"
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
    deployed = {}
    for base in sorted({base for base, _ in FUSED_HYBRIDS + TEXT_FIRST_HYBRIDS}):
        deployed[base] = ingestion.schema_registry.deploy_schema(
            tenant_id=tenant, base_schema_name=base
        )
    yield tenant, config_manager, deployed


def _feed(vespa_instance, schema_name: str, doc_id: str, fields: dict):
    response = requests.post(
        f"http://localhost:{vespa_instance['http_port']}/document/v1/"
        f"{document_namespace(schema_name)}/{schema_name}/docid/{doc_id}",
        json={"fields": fields},
        timeout=30,
    )
    assert response.status_code == 200, response.text


def _remove(vespa_instance, schema_name: str, doc_id: str):
    response = requests.delete(
        f"http://localhost:{vespa_instance['http_port']}/document/v1/"
        f"{document_namespace(schema_name)}/{schema_name}/docid/{doc_id}",
        timeout=30,
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(("base", "strategy"), FUSED_HYBRIDS)
def test_hybrid_score_is_visual_similarity_plus_native_rank(
    vespa_instance, hybrid_corpus, base, strategy
):
    tenant, config_manager, deployed = hybrid_corpus
    schema_name = deployed[base]
    field, query, stored, visual = _fixture_vectors(base, strategy)
    id_field = _schema(base).get("document_mapping", {}).get("id")
    with_term, without_term = f"{strategy}_with_term", f"{strategy}_without_term"
    for doc_id, text in (
        (with_term, TEXT_WITH_TERM),
        (without_term, TEXT_WITHOUT_TERM),
    ):
        fields = {field: stored, **{f: text for f in _text_fields(base)}}
        if id_field:
            fields[id_field] = doc_id
        _feed(vespa_instance, schema_name, doc_id, fields)
    profile_name, profile = _shipped_profile(base)
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
                "query": QUERY_TERM,
                "type": profile["type"],
                "profile": profile_name,
                "strategy": strategy,
                "top_k": 10,
                "tenant_id": tenant,
                "query_embeddings": query,
            }
        )
    finally:
        search.close()
        for doc_id in (with_term, without_term):
            _remove(vespa_instance, schema_name, doc_id)

    assert [r.document.id for r in results] == [with_term, without_term]
    assert results[1].score == pytest.approx(visual, abs=1e-6)
    assert results[0].score == pytest.approx(visual + TERM_DOC_NATIVE_RANK, abs=1e-6)


def _backend_search(vespa_instance, config_manager, base, request, port=None):
    profile_name, profile = _shipped_profile(base)
    search = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": port or vespa_instance["http_port"],
            "profiles": {profile_name: profile},
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        is_schema_deployed=DeployedSchemaNames(config_manager),
        enable_connection_pool=False,
        retry_config=RetryConfig(max_attempts=1),
    )
    try:
        return search.search(
            {"type": profile["type"], "profile": profile_name, **request}
        )
    finally:
        search.close()


@pytest.mark.parametrize(("base", "strategy"), TEXT_FIRST_HYBRIDS)
def test_text_first_hybrid_ranks_text_matches_by_visual_plus_native_rank(
    vespa_instance, hybrid_corpus, base, strategy
):
    tenant, config_manager, deployed = hybrid_corpus
    schema_name = deployed[base]
    field, query, ((near, near_visual), (far, far_visual)) = _fixture(base, strategy)
    id_field = _schema(base).get("document_mapping", {}).get("id")
    docs = {
        f"{strategy}_near": (near, TEXT_WITH_TERM),
        f"{strategy}_far": (far, TEXT_WITH_TERM),
        f"{strategy}_textless": (near, TEXT_WITHOUT_TERM),
    }
    for doc_id, (stored, text) in docs.items():
        fields = {field: stored, **{f: text for f in _text_fields(base)}}
        if id_field:
            fields[id_field] = doc_id
        _feed(vespa_instance, schema_name, doc_id, fields)
    try:
        results = _backend_search(
            vespa_instance,
            config_manager,
            base,
            {
                "query": QUERY_TERM,
                "strategy": strategy,
                "top_k": 10,
                "tenant_id": tenant,
                "query_embeddings": query,
            },
        )
    finally:
        for doc_id in docs:
            _remove(vespa_instance, schema_name, doc_id)

    assert near_visual > far_visual
    assert [r.document.id for r in results] == [f"{strategy}_near", f"{strategy}_far"]
    assert [r.score for r in results] == pytest.approx(
        [near_visual + TERM_DOC_NATIVE_RANK, far_visual + TERM_DOC_NATIVE_RANK],
        abs=1e-6,
    )


# Vespa's default weakAnd target: the hits it keeps per content node before it
# skips documents that cannot beat its running threshold.
WEAK_AND_TARGET_HITS = 100
TWO_TERM_QUERY = "kestrel osprey"
TEXT_WITH_BOTH_TERMS = "kestrel osprey harbour notes"


def _nearest_neighbor_hybrid_search(vespa_instance, hybrid_corpus, **request):
    base, strategy = "video_xclip_sv_chunk_6s", "hybrid_float_bm25"
    tenant, config_manager, _ = hybrid_corpus
    _, query, _, _ = _fixture_vectors(base, strategy)
    profile_name, profile = _shipped_profile(base)
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
        return search.search(
            {
                "query": TWO_TERM_QUERY,
                "type": profile["type"],
                "profile": profile_name,
                "strategy": strategy,
                "tenant_id": tenant,
                "query_embeddings": query,
                "nearest_neighbor_approximate": False,
                **request,
            }
        )
    finally:
        search.close()


def test_nearest_neighbor_hybrid_scores_every_document_as_it_scores_alone(
    vespa_instance, hybrid_corpus
):
    """More documents hold the query terms than weakAnd keeps. Every document
    the hybrid ranks, including those holding one term after the documents
    holding both filled weakAnd's target, scores exactly what it scores as
    the query's only match: its closeness plus its full nativeRank."""
    base, strategy = "video_xclip_sv_chunk_6s", "hybrid_float_bm25"
    _, _, deployed = hybrid_corpus
    schema_name = deployed[base]
    field, _, stored, visual = _fixture_vectors(base, strategy)
    id_field = _schema(base)["document_mapping"]["id"]
    count = WEAK_AND_TARGET_HITS + WEAK_AND_TARGET_HITS // 2
    both = [f"nn_both_{i:03d}" for i in range(count)]
    one = [f"nn_one_{i:03d}" for i in range(count)]
    neither = "nn_neither"
    texts = {
        **dict.fromkeys(both, TEXT_WITH_BOTH_TERMS),
        **dict.fromkeys(one, TEXT_WITH_TERM),
        neither: TEXT_WITHOUT_TERM,
    }
    for doc_id, text in texts.items():
        fields = {field: stored, id_field: doc_id}
        fields.update({f: text for f in _text_fields(base)})
        _feed(vespa_instance, schema_name, doc_id, fields)
    try:
        results = _nearest_neighbor_hybrid_search(
            vespa_instance, hybrid_corpus, top_k=len(texts)
        )
        alone = {
            doc_id: _nearest_neighbor_hybrid_search(
                vespa_instance,
                hybrid_corpus,
                top_k=1,
                filters={id_field: doc_id},
            )
            for doc_id in (both[0], one[0], neither)
        }
    finally:
        for doc_id in texts:
            _remove(vespa_instance, schema_name, doc_id)

    assert {
        doc_id: [r.document.id for r in hits] for doc_id, hits in alone.items()
    } == {doc_id: [doc_id] for doc_id in alone}
    both_alone, one_alone, neither_alone = (
        alone[doc_id][0].score for doc_id in (both[0], one[0], neither)
    )
    assert neither_alone == pytest.approx(visual, abs=1e-6)
    # Equal scores order by source id under the profile's source grouping, so
    # this order also holds only if each kind outscores the next.
    assert [r.document.id for r in results] == [*both, *one, neither]
    assert [r.score for r in results] == pytest.approx(
        [both_alone] * count + [one_alone] * count + [neither_alone], abs=1e-6
    )


# A seeded single-vector corpus: four X-CLIP videos of three chunks each and
# four audio clips of two chunks each. A title holds a query term once, first,
# or not at all; transcripts never hold one, so each document's nativeRank for
# a single-term query is TERM_DOC_NATIVE_RANK or 0.
XCLIP = "video_xclip_sv_chunk_6s"
AUDIO = "audio_content"
XCLIP_TITLES = {
    "kestrel_cliffs": "kestrel cliffs walk",
    "harbour_nets": "harbour nets dawn",
    "kestrel_ridge": "kestrel ridge flight",
    "market_day": "market day stalls",
}
XCLIP_TRANSCRIPT = "gulls circle over the water"
XCLIP_QUERIES = ("kestrel", "harbour")
AUDIO_TITLES = {
    "ocean_waves": "ocean waves shore",
    "city_traffic": "city traffic noise",
    "ocean_storm": "ocean storm night",
    "forest_birds": "forest birds morning",
}
AUDIO_QUERY = "ocean"
XCLIP_STRATEGIES = (
    "hybrid_float_bm25",
    "hybrid_binary_bm25",
    "hybrid_bm25_float",
    "hybrid_bm25_binary",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


# Each chunk's pull toward each query vector: a chunk without the query term
# in its title (market_day_1, forest_birds_0) can outscore every chunk holding
# it on similarity alone.
XCLIP_PULLS = {
    "kestrel_cliffs": [(0.5, 0.1), (0.3, 0.0), (0.8, 0.2)],
    "harbour_nets": [(0.1, 0.6), (0.0, 0.3), (0.2, 0.9)],
    "kestrel_ridge": [(0.05, 0.0), (0.1, 0.1), (0.0, 0.05)],
    "market_day": [(0.2, 0.1), (6.0, 0.0), (0.1, 1.5)],
}
AUDIO_PULLS = {
    "ocean_waves": [0.1, 0.0],
    "city_traffic": [0.3, 0.2],
    "ocean_storm": [0.5, 0.2],
    "forest_birds": [6.0, 0.4],
}


def _single_vector_corpus() -> dict:
    """Query vectors and stored embeddings, generated from a fixed seed."""
    rng = np.random.default_rng(20261003)
    xclip_queries = {q: _unit(rng.standard_normal(768)) for q in XCLIP_QUERIES}
    xclip = {}
    for video, pulls in XCLIP_PULLS.items():
        for chunk, pull in enumerate(pulls):
            mixed = sum(w * xclip_queries[q] for w, q in zip(pull, XCLIP_QUERIES))
            xclip[f"{video}_{chunk}"] = _unit(mixed + _unit(rng.standard_normal(768)))
    audio_query = _unit(rng.standard_normal(512))
    audio = {}
    for clip, pulls in AUDIO_PULLS.items():
        for chunk, pull in enumerate(pulls):
            audio[f"{clip}_{chunk}"] = _unit(
                pull * audio_query + _unit(rng.standard_normal(512))
            )
    return {
        "xclip_queries": xclip_queries,
        "xclip": xclip,
        "audio_query": audio_query,
        "audio": audio,
    }


def _xclip_visual(strategy: str, query: np.ndarray, stored: np.ndarray) -> float:
    if "binary" in strategy:
        h = _hamming(_bits(query), _bits(stored))
        return 1.0 / (1.0 + math.pi * h / len(query))
    return _dense_closeness(XCLIP, "embedding", query, stored)


def _expected_ranking(scores: dict[str, float], source_of) -> list[tuple]:
    """(source, best document, its score), best source first."""
    best = {}
    for doc_id, score in scores.items():
        source = source_of(doc_id)
        if source not in best or score > best[source][1]:
            best[source] = (doc_id, score)
    return [
        (source, doc_id, score)
        for source, (doc_id, score) in sorted(best.items(), key=lambda kv: -kv[1][1])
    ]


def _expected_xclip(corpus: dict, strategy: str, query: str) -> list[tuple]:
    """Every chunk is a candidate of a visual-first hybrid; a text-first one
    keeps the chunks whose title holds the query term."""
    scores = {}
    for doc_id, stored in corpus["xclip"].items():
        holds_term = XCLIP_TITLES[doc_id.rsplit("_", 1)[0]].split()[0] == query
        if strategy.startswith("hybrid_bm25") and not holds_term:
            continue
        scores[doc_id] = _xclip_visual(
            strategy, corpus["xclip_queries"][query], stored
        ) + (TERM_DOC_NATIVE_RANK if holds_term else 0.0)
    return _expected_ranking(scores, lambda doc_id: doc_id.rsplit("_", 1)[0])


def _expected_audio(corpus: dict) -> list[tuple]:
    scores = {}
    for doc_id, stored in corpus["audio"].items():
        clip = doc_id.rsplit("_", 1)[0]
        holds_term = AUDIO_TITLES[clip].split()[0] == AUDIO_QUERY
        scores[doc_id] = _dense_closeness(
            AUDIO, "acoustic_embedding", corpus["audio_query"], stored
        ) + (TERM_DOC_NATIVE_RANK if holds_term else 0.0)
    return _expected_ranking(scores, lambda doc_id: doc_id.rsplit("_", 1)[0])


@pytest.fixture(scope="module")
def single_vector_corpus(vespa_instance):
    """The seeded corpus fed through the production ingestion path."""
    corpus = _single_vector_corpus()
    tenant = f"svhybrid{uuid.uuid4().hex[:8]}:unit"
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
    xclip_docs = []
    for doc_id, embedding in corpus["xclip"].items():
        video, chunk = doc_id.rsplit("_", 1)
        document = Document(
            id=doc_id,
            content_type=ContentType.VIDEO,
            content_id=video,
            status=ProcessingStatus.COMPLETED,
        )
        document.add_embedding("embedding", embedding, {"type": "float", "raw": True})
        document.add_metadata("video_id", video)
        document.add_metadata("video_title", XCLIP_TITLES[video])
        document.add_metadata("segment_index", int(chunk))
        document.add_metadata("start_time", 5.0 * int(chunk))
        document.add_metadata("end_time", 5.0 * int(chunk) + 6.0)
        document.add_metadata("audio_transcript", XCLIP_TRANSCRIPT)
        xclip_docs.append(document)
    audio_docs = []
    for doc_id, embedding in corpus["audio"].items():
        clip, chunk = doc_id.rsplit("_", 1)
        document = Document(
            id=doc_id,
            content_type=ContentType.AUDIO,
            content_id=clip,
            status=ProcessingStatus.COMPLETED,
        )
        document.add_metadata("audio_id", clip)
        document.add_metadata("audio_title", AUDIO_TITLES[clip])
        document.add_metadata("audio_transcript", AUDIO_TITLES[clip])
        document.add_metadata("chunk_index", int(chunk))
        document.add_metadata("acoustic_embedding", embedding.tolist())
        audio_docs.append(document)
    for base, documents in ((XCLIP, xclip_docs), (AUDIO, audio_docs)):
        ingestion.schema_registry.deploy_schema(tenant_id=tenant, base_schema_name=base)
        result = ingestion.ingest_documents(documents, base)
        assert result["failed_count"] == 0, result["failed_documents"]
        schema_name = ingestion.get_tenant_schema_name(tenant, base)
        indexed = requests.post(
            f"http://localhost:{vespa_instance['http_port']}/search/",
            json={"yql": f"select * from {schema_name} where true", "hits": 0},
            timeout=30,
        ).json()["root"]["fields"]["totalCount"]
        assert indexed == len(documents)
    yield corpus, tenant, config_manager


def _xclip_search(vespa_instance, single_vector_corpus, strategy, query, port=None):
    corpus, tenant, config_manager = single_vector_corpus
    return _backend_search(
        vespa_instance,
        config_manager,
        XCLIP,
        {
            "query": query,
            "strategy": strategy,
            "top_k": 10,
            "tenant_id": tenant,
            "query_embeddings": corpus["xclip_queries"][query],
        },
        port=port,
    )


def _audio_search(vespa_instance, single_vector_corpus):
    corpus, tenant, config_manager = single_vector_corpus
    return _backend_search(
        vespa_instance,
        config_manager,
        AUDIO,
        {
            "query": AUDIO_QUERY,
            "strategy": "hybrid_acoustic_bm25",
            "top_k": 10,
            "tenant_id": tenant,
            "query_embeddings": corpus["audio_query"],
        },
    )


def _ranked(results, source_field: str) -> list[tuple]:
    return [
        (r.document.metadata[source_field], r.document.id, r.score) for r in results
    ]


# Source order, best document and its score the X-CLIP hybrids give over the
# seeded corpus.
XCLIP_RANKING = {
    "hybrid_float_bm25": {
        "kestrel": [
            ("kestrel_cliffs", "kestrel_cliffs_2", 0.9095899256883291),
            ("market_day", "market_day_1", 0.858852175388682),
            ("kestrel_ridge", "kestrel_ridge_1", 0.7866987777248962),
            ("harbour_nets", "harbour_nets_2", 0.4142414141886994),
        ],
        "harbour": [
            ("harbour_nets", "harbour_nets_2", 0.9269060683571373),
            ("market_day", "market_day_2", 0.6247124882009493),
            ("kestrel_ridge", "kestrel_ridge_1", 0.4064570875601872),
            ("kestrel_cliffs", "kestrel_cliffs_2", 0.40534950653249746),
        ],
    },
    "hybrid_binary_bm25": {
        "kestrel": [
            ("kestrel_cliffs", "kestrel_cliffs_2", 0.9059393640888267),
            ("market_day", "market_day_1", 0.847466906800897),
            ("kestrel_ridge", "kestrel_ridge_1", 0.7889855580989471),
            ("harbour_nets", "harbour_nets_0", 0.41472053442390244),
        ],
        "harbour": [
            ("harbour_nets", "harbour_nets_2", 0.9331204320330677),
            ("market_day", "market_day_2", 0.6476466436058796),
            ("kestrel_ridge", "kestrel_ridge_1", 0.4078023221010049),
            ("kestrel_cliffs", "kestrel_cliffs_2", 0.4024317497580295),
        ],
    },
    "hybrid_bm25_float": {
        "kestrel": [
            ("kestrel_cliffs", "kestrel_cliffs_2", 0.9095899256883291),
            ("kestrel_ridge", "kestrel_ridge_1", 0.7866987777248962),
        ],
        "harbour": [
            ("harbour_nets", "harbour_nets_2", 0.926906044464576),
        ],
    },
    "hybrid_bm25_binary": {
        "kestrel": [
            ("kestrel_cliffs", "kestrel_cliffs_2", 0.9059393640888267),
            ("kestrel_ridge", "kestrel_ridge_1", 0.7889855580989471),
        ],
        "harbour": [
            ("harbour_nets", "harbour_nets_2", 0.9331204320330677),
        ],
    },
}
AUDIO_RANKING = [
    ("ocean_storm", "ocean_storm_0", 0.8801791397696893),
    ("forest_birds", "forest_birds_0", 0.8594281482454702),
    ("ocean_waves", "ocean_waves_0", 0.805164937952157),
    ("city_traffic", "city_traffic_0", 0.4545631024459095),
]


class TestSingleVectorHybridRanking:
    @pytest.mark.parametrize("query", XCLIP_QUERIES)
    @pytest.mark.parametrize("strategy", XCLIP_STRATEGIES)
    def test_xclip_hybrid_ranks_the_recorded_order(
        self, vespa_instance, single_vector_corpus, strategy, query
    ):
        ranked = _ranked(
            _xclip_search(vespa_instance, single_vector_corpus, strategy, query),
            "video_id",
        )

        expected = XCLIP_RANKING[strategy][query]
        assert [row[:2] for row in ranked] == [row[:2] for row in expected]
        assert [row[2] for row in ranked] == pytest.approx(
            [row[2] for row in expected], abs=1e-6
        )

    @pytest.mark.parametrize("query", XCLIP_QUERIES)
    @pytest.mark.parametrize("strategy", XCLIP_STRATEGIES)
    def test_xclip_scores_are_visual_similarity_plus_native_rank(
        self, vespa_instance, single_vector_corpus, strategy, query
    ):
        """Each video scores its best chunk's visual similarity, computed from
        the stored vectors, plus the chunk's nativeRank."""
        corpus = single_vector_corpus[0]
        ranked = _ranked(
            _xclip_search(vespa_instance, single_vector_corpus, strategy, query),
            "video_id",
        )

        expected = _expected_xclip(corpus, strategy, query)
        assert [row[:2] for row in ranked] == [row[:2] for row in expected]
        assert [row[2] for row in ranked] == pytest.approx(
            [row[2] for row in expected], abs=1e-6
        )

    def test_audio_hybrid_ranks_the_recorded_order(
        self, vespa_instance, single_vector_corpus
    ):
        ranked = _ranked(
            _audio_search(vespa_instance, single_vector_corpus), "audio_id"
        )

        assert [row[:2] for row in ranked] == [row[:2] for row in AUDIO_RANKING]
        assert [row[2] for row in ranked] == pytest.approx(
            [row[2] for row in AUDIO_RANKING], abs=1e-6
        )

    def test_audio_scores_are_acoustic_closeness_plus_native_rank(
        self, vespa_instance, single_vector_corpus
    ):
        ranked = _ranked(
            _audio_search(vespa_instance, single_vector_corpus), "audio_id"
        )

        expected = _expected_audio(single_vector_corpus[0])
        assert [row[:2] for row in ranked] == [row[:2] for row in expected]
        assert [row[2] for row in ranked] == pytest.approx(
            [row[2] for row in expected], abs=1e-6
        )

    def test_concurrent_hybrid_searches_return_their_recorded_rankings(
        self, vespa_instance, single_vector_corpus
    ):
        """Hybrid searches released together each get exactly the ranking
        their strategy and query get alone."""
        cases = [(s, q) for s in XCLIP_STRATEGIES for q in XCLIP_QUERIES] * 2
        barrier = threading.Barrier(len(cases) + 2)

        def released(case):
            barrier.wait(timeout=60)
            if case == "audio":
                return _ranked(
                    _audio_search(vespa_instance, single_vector_corpus), "audio_id"
                )
            return _ranked(
                _xclip_search(vespa_instance, single_vector_corpus, *case), "video_id"
            )

        with ThreadPoolExecutor(max_workers=len(cases) + 2) as pool:
            concurrent = list(pool.map(released, [*cases, "audio", "audio"]))

        expected = [XCLIP_RANKING[s][q] for s, q in cases] + [AUDIO_RANKING] * 2
        assert [[row[:2] for row in ranked] for ranked in concurrent] == [
            [row[:2] for row in rows] for rows in expected
        ]
        assert [row[2] for ranked in concurrent for row in ranked] == pytest.approx(
            [row[2] for rows in expected for row in rows], abs=1e-6
        )

    @pytest.mark.parametrize("strategy", XCLIP_STRATEGIES)
    def test_hybrid_search_raises_when_vespa_is_unreachable(
        self, vespa_instance, single_vector_corpus, strategy
    ):
        port = _dead_port()

        with pytest.raises(httpr.ConnectError) as excinfo:
            _xclip_search(
                vespa_instance,
                single_vector_corpus,
                strategy,
                XCLIP_QUERIES[0],
                port=port,
            )

        assert str(excinfo.value) == (
            f"error sending request for url (http://localhost:{port}/search/)"
        )


def _dead_port() -> int:
    """A local TCP port with nothing listening on it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port
