"""Visual-first hybrid rank profiles, against real Vespa, on every schema.

A visual-first hybrid ranks every document by the sum of its normalized
embedding similarity and ``nativeRank`` over the profile's text fields, both
in its first phase:

- float MaxSim: the mean over query tokens of the best dot product with a
  stored token/patch vector;
- binary MaxSim: the mean over query tokens of ``1 - 2h/bits``, ``h`` the
  Hamming distance to the nearest stored vector;
- dense ``closeness`` on single-vector schemas (nearestNeighbor retrieval).

Each profile is searched through ``VespaSearchBackend`` with two documents
holding the same embedding, one of whose text fields carry the query term:
both come back, the text-less one scored by the visual similarity alone and
the other by that similarity plus its ``nativeRank``.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.schema_registry import DeployedSchemaNames
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
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


def _expanded_first_phase(profile: dict) -> str:
    functions = {f["name"]: f["expression"] for f in profile.get("functions", [])}
    first = profile["first_phase"]
    expr = first["expression"] if isinstance(first, dict) else first
    return " + ".join(functions.get(term, term) for term in expr.split(" + "))


def _fused_hybrids(schemas_dir: Path = SCHEMAS_DIR) -> list[tuple[str, str]]:
    """Every (schema, profile) whose first phase sums an embedding similarity
    and a nativeRank text score."""
    found = []
    for path in sorted(schemas_dir.glob("*_schema.json")):
        base = path.name.removesuffix("_schema.json")
        strategies = RankingStrategyExtractor().extract_from_schema(path)
        for name, info in strategies.items():
            if not (info.needs_text_query and info.first_phase_embedding_field):
                continue
            profile = _profile(base, name, schemas_dir)
            terms = _expanded_first_phase(profile).split(" + ")
            if len(terms) == 2 and terms[1].startswith("nativeRank("):
                found.append((base, name))
    return found


FUSED_HYBRIDS = [
    ("audio_content", "hybrid_semantic_bm25"),
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
]


def test_every_visual_first_hybrid_fuses_with_native_rank():
    assert _fused_hybrids() == FUSED_HYBRIDS


def test_a_hybrid_reranking_by_text_alone_is_not_fused(tmp_path):
    """The derivation drops a hybrid whose text score replaces the visual one
    in a second phase."""
    shutil.copytree(SCHEMAS_DIR, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "video_colpali_smol500_mv_frame_schema.json"
    schema = json.loads(path.read_text())
    (profile,) = [
        p for p in schema["rank_profiles"] if p["name"] == "hybrid_float_bm25"
    ]
    profile["first_phase"] = "visual_sim_float"
    profile["second_phase"] = {"expression": "text_sim", "rerank_count": 100}
    path.write_text(json.dumps(schema))

    assert _fused_hybrids(tmp_path) == [
        hybrid
        for hybrid in FUSED_HYBRIDS
        if hybrid != ("video_colpali_smol500_mv_frame", "hybrid_float_bm25")
    ]


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


def _fixture_vectors(base: str, name: str):
    """Query vectors, stored vectors and the visual score they must produce."""
    info = RankingStrategyExtractor().extract_from_schema(
        SCHEMAS_DIR / f"{base}_schema.json"
    )[name]
    field = info.embedding_field
    stored_type = _field_type(base, field)
    (query_type,) = info.inputs.values()
    binary = "int8" in query_type
    dim = _width(query_type) * (8 if binary else 1)
    if "{}" not in stored_type:
        # Single-vector schema: angular closeness of one query vector.
        query = np.zeros(dim, np.float32)
        query[:2] = [1.0, 1.0]
        stored = np.zeros(dim, np.float32)
        stored[0] = 1.0
        angle = np.arccos(np.dot(query, stored) / np.linalg.norm(query))
        return field, query, stored.tolist(), float(1.0 / (1.0 + angle))
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
        xor = np.bitwise_xor(_bits(query)[:, None, :], _bits(patches)[None])
        nearest = _POPCOUNT[xor].sum(-1).min(axis=1)
        visual = float(np.mean(1.0 - 2.0 * nearest / dim))
        cells = _bits(patches).view(np.int8)
        stored = {"blocks": {str(i): c.tolist() for i, c in enumerate(cells)}}
        return field, query, stored, visual
    visual = float(np.mean((query @ patches.T).max(axis=1)))
    stored = {"blocks": {str(i): p.tolist() for i, p in enumerate(patches)}}
    return field, query, stored, visual


def _shipped_profile(base: str) -> tuple[str, dict]:
    for profile_name, profile in SHIPPED_PROFILES.items():
        if profile.get("schema_name") == base:
            return profile_name, profile
    return base, {"type": "document", "schema_name": base}


@pytest.fixture(scope="module")
def hybrid_corpus(vespa_instance):
    """Two documents per schema with the same embedding, one holding the
    query term in every searchable text field."""
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
    for base in sorted({base for base, _ in FUSED_HYBRIDS}):
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
