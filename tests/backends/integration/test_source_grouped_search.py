"""Source-granularity search groups segments by source inside real Vespa.

One grouped query per search returns the best ``top_k`` sources by their
best segment, whatever share of the matches one source holds. The ANN
candidate budget is explicit: when it leaves fewer than ``top_k`` sources
while more candidates may exist, the batch says so.
"""

import gzip
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests
from vespa.exceptions import VespaError

import cogniverse_vespa.search_backend as search_backend_module
from cogniverse_core.common.utils.retry import RetryConfig
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.schema_registry import DeployedSchemaNames
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    make_config_manager,
    schema_tensor_dim,
)

pytestmark = [pytest.mark.integration]

SCHEMAS_DIR = Path("configs/schemas")
SHIPPED_PROFILES = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
]
MV_PROFILE = "video_colpali_smol500_mv_frame"
ANN_PROFILE = "video_xclip_sv_chunk_6s"
ANN_WIDE_PROFILE = "xclip_wide_candidates"
TEXT_PROFILE = "document_text_semantic"
SUFFIX = uuid.uuid4().hex[:8]
MV_TENANT = f"srcgroup{SUFFIX}:mv"
ANN_TENANT = f"srcgroup{SUFFIX}:ann"
TEXT_TENANT = f"srcgroup{SUFFIX}:text"
FAST_RETRY = RetryConfig(
    max_attempts=2,
    initial_delay=0.01,
    exceptions=(VespaError, requests.RequestException, ConnectionError),
)

BIG_OTHERS = [f"bigother{j}" for j in range(9)]
DOMINANT_WINDOWS = {
    "sourdough_guide": [
        "Feed the sourdough starter with equal weights of flour and water every morning.",
        "A lively sourdough starter doubles in size a few hours after each feeding.",
        "Mix the fed starter into the bread dough once it smells pleasantly sour.",
        "Autolyse the flour and water before adding the sourdough starter and salt.",
        "Stretch and fold the sourdough dough every half hour during bulk fermentation.",
        "Shape the sourdough loaf tightly so it holds its height in the oven.",
        "Proof the shaped sourdough overnight in the refrigerator for a deeper flavor.",
        "Preheat a Dutch oven so the sourdough bread gets a crisp, blistered crust.",
        "Score the top of the sourdough loaf with a razor just before baking.",
        "Bake the sourdough bread covered for twenty minutes to trap the steam.",
        "Remove the lid and keep baking the sourdough until the crust turns deep brown.",
        "Let the baked sourdough bread cool completely before slicing into it.",
        "A sluggish sourdough starter usually needs warmer water or more frequent feeding.",
        "Discard part of the sourdough starter before feeding so it stays manageable.",
        "Whole wheat flour makes the sourdough starter ferment faster than white flour.",
        "An open crumb in sourdough bread comes from high hydration and gentle handling.",
        "Store a mature sourdough starter in the fridge between weekly bakes.",
        "The tang of sourdough bread comes from lactic and acetic acid in the starter.",
    ]
}
OTHER_WINDOWS = {
    "rye_loaf": [
        "A dense rye loaf leavened with sourdough starter bakes low and slow for hours."
    ],
    "pizza_dough": [
        "Neapolitan pizza dough ferments for a day before it is stretched and baked."
    ],
    "tomato_garden": [
        "Tomato seedlings need full sun and deep watering once they are transplanted."
    ],
    "rocket_launch": [
        "The rocket lifted off from the pad carrying a weather satellite into orbit."
    ],
    "chess_opening": [
        "The Sicilian Defence answers the king pawn opening with an early c5 advance."
    ],
}


def _feed(port: int, schema: str, doc_id: str, fields: dict) -> None:
    response = requests.post(
        f"http://localhost:{port}/document/v1/video/{schema}/docid/{doc_id}",
        json={"fields": fields},
        timeout=10,
    )
    assert response.status_code == 200, response.text


def _wait_for_count(port: int, schema: str, expected: int) -> None:
    deadline = time.monotonic() + 60
    count = None
    while time.monotonic() < deadline:
        body = requests.post(
            f"http://localhost:{port}/search/",
            json={"yql": f"select * from {schema} where true", "hits": 0},
            timeout=10,
        ).json()
        count = body["root"]["fields"]["totalCount"]
        if count == expected:
            return
        time.sleep(0.5)
    raise AssertionError(f"{schema} holds {count} documents, expected {expected}")


def _patch_tensors(x: float, y: float, dim: int):
    vector = np.zeros(dim, dtype=np.float32)
    vector[0], vector[1] = x, y
    binary = np.packbits((vector > 0).astype(np.uint8)).astype(np.int8)
    return {"0": vector.tolist()}, {"0": binary.tolist()}


def _dense_tensors(offset_axis: int, offset: float):
    vector = np.zeros(768, dtype=np.float32)
    vector[0] = 1.0
    vector[offset_axis] = offset
    binary = np.packbits((vector > 0).astype(np.uint8)).astype(np.int8)
    return {"values": vector.tolist()}, {"values": binary.tolist()}


def _video_fields(source_id: str, title: str, index: int, embedding, binary) -> dict:
    return {
        "video_id": source_id,
        "video_title": title,
        "segment_id": index,
        "start_time": float(index),
        "end_time": float(index + 1),
        "audio_transcript": title,
        "embedding": embedding,
        "embedding_binary": binary,
    }


@pytest.fixture(scope="module")
def config_manager(vespa_instance, pylate_server):
    return make_config_manager(
        vespa_instance, inference_service_urls={"colbert_pylate": pylate_server}
    )


@pytest.fixture(scope="module")
def mv_corpus(vespa_instance, config_manager):
    """Multi-vector corpus ranked without ANN: every match reaches grouping."""
    port = vespa_instance["http_port"]
    schema = deploy_tenant_schema(
        vespa_instance,
        tenant_id=MV_TENANT,
        base_schema_name=MV_PROFILE,
        config_manager=config_manager,
    )
    dim = schema_tensor_dim(MV_PROFILE, "embedding")
    fed = 0
    for i in range(401):
        tensors = _patch_tensors(1.0, 1.0 - 0.001 * i, dim)
        _feed(
            port,
            schema,
            f"bigdom_{i:03d}",
            _video_fields("bigdom", "bigcorpus harbour ferry", i, *tensors),
        )
        fed += 1
    for j, source_id in enumerate(BIG_OTHERS):
        tensors = _patch_tensors(0.5, 0.5 - 0.05 * j, dim)
        _feed(
            port,
            schema,
            f"{source_id}_000",
            _video_fields(source_id, "bigcorpus minority ferry", 0, *tensors),
        )
        fed += 1
    for i in range(18):
        tensors = _patch_tensors(1.0, 1.0 - 0.01 * i, dim)
        _feed(
            port,
            schema,
            f"smalldom_{i:02d}",
            _video_fields(
                "smalldom", f"smallcorpus harbour harbour d{i:02d}", i, *tensors
            ),
        )
        fed += 1
    _feed(
        port,
        schema,
        "smallother_00",
        _video_fields(
            "smallother",
            "smallcorpus harbour lighthouse beacon",
            0,
            *_patch_tensors(0.5, 0.5, dim),
        ),
    )
    fed += 1
    for source_id in ("tie_c", "tie_a", "tie_b"):
        for index in (1, 0):
            _feed(
                port,
                schema,
                f"{source_id}_{index}",
                _video_fields(
                    source_id,
                    "tiecorpus",
                    index,
                    *_patch_tensors(0.7, 0.7, dim),
                ),
            )
            fed += 1
    _wait_for_count(port, schema, fed)
    return schema


@pytest.fixture(scope="module")
def ann_corpus(vespa_instance, config_manager):
    """Single-vector corpus retrieved through nearestNeighbor."""
    port = vespa_instance["http_port"]
    schema = deploy_tenant_schema(
        vespa_instance,
        tenant_id=ANN_TENANT,
        base_schema_name=ANN_PROFILE,
        config_manager=config_manager,
    )
    fed = 0
    for i in range(401):
        _feed(
            port,
            schema,
            f"bigdom_{i:03d}",
            _video_fields(
                "bigdom", "bigcorpus harbour", i, *_dense_tensors(1, 0.001 * i)
            ),
        )
        fed += 1
    for j, source_id in enumerate(BIG_OTHERS):
        _feed(
            port,
            schema,
            f"{source_id}_000",
            _video_fields(
                source_id,
                "bigcorpus minority",
                0,
                *_dense_tensors(2, 0.6 + 0.05 * j),
            ),
        )
        fed += 1
    for i in range(18):
        _feed(
            port,
            schema,
            f"smalldom_{i:02d}",
            _video_fields(
                "smalldom",
                "smallcorpus harbour",
                i,
                *_dense_tensors(3, 0.0005 + 0.01 * i),
            ),
        )
        fed += 1
    _feed(
        port,
        schema,
        "smallother_00",
        _video_fields("smallother", "smallcorpus", 0, *_dense_tensors(4, 0.6)),
    )
    fed += 1
    _wait_for_count(port, schema, fed)
    return schema


@pytest.fixture(scope="module")
def text_corpus(vespa_instance, config_manager, pylate_server):
    """Natural passages embedded by the served LateOn model."""
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=TEXT_TENANT,
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
    model, _ = RemoteColBERTLoader(
        SHIPPED_PROFILES[TEXT_PROFILE]["embedding_model"],
        {"remote_inference_url": pylate_server},
        _resolved_headers={},
    ).load_model()
    try:
        documents = []
        for source_id, windows in {**DOMINANT_WINDOWS, **OTHER_WINDOWS}.items():
            embeddings = model.encode(windows, is_query=False)
            for index, (text, embedding) in enumerate(zip(windows, embeddings)):
                document = Document(
                    id=f"{source_id}_w{index:02d}",
                    content_type=ContentType.DOCUMENT,
                    content_id=source_id,
                    status=ProcessingStatus.COMPLETED,
                )
                document.add_embedding(
                    "embedding",
                    np.asarray(embedding, dtype=np.float32),
                    {"type": "float", "raw": True},
                )
                document.add_metadata("document_id", source_id)
                document.add_metadata("document_title", f"{source_id}.md")
                document.add_metadata("document_type", "markdown")
                document.add_metadata("document_path", f"/corpus/{source_id}.md")
                document.add_metadata("full_text", text)
                document.add_metadata("page_count", 1)
                document.add_metadata("chunk_index", index)
                document.add_metadata("chunk_count", len(windows))
                documents.append(document)
    finally:
        model._close()
    outcome = backend.ingest_documents(documents, "document_text")
    assert outcome == {
        "success_count": len(documents),
        "failed_count": 0,
        "failed_documents": [],
        "total_documents": len(documents),
    }
    schema = backend.get_tenant_schema_name(TEXT_TENANT, "document_text")
    _wait_for_count(vespa_instance["http_port"], schema, len(documents))
    return schema


def _backend(config_manager, port: int, **kwargs) -> VespaSearchBackend:
    profiles = {
        MV_PROFILE: SHIPPED_PROFILES[MV_PROFILE],
        ANN_PROFILE: SHIPPED_PROFILES[ANN_PROFILE],
        ANN_WIDE_PROFILE: {
            **SHIPPED_PROFILES[ANN_PROFILE],
            "source_collapse_oversample": 10,
        },
        TEXT_PROFILE: SHIPPED_PROFILES[TEXT_PROFILE],
    }
    return VespaSearchBackend(
        config={"url": "http://127.0.0.1", "port": port, "profiles": profiles},
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        is_schema_deployed=DeployedSchemaNames(config_manager),
        **kwargs,
    )


@pytest.fixture(scope="module")
def counted(vespa_instance, config_manager, mv_corpus, ann_corpus, text_corpus):
    """A backend whose every Vespa request passes a counting proxy."""
    with InterceptFaultProxy(f"http://localhost:{vespa_instance['http_port']}") as p:
        backend = _backend(config_manager, p.port)
        try:
            yield backend, p
        finally:
            backend.close()


def _search_requests(proxy) -> int:
    """Backend searches through the proxy, excluding connection health probes."""
    count = 0
    for _, path, body in proxy.requests:
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        if path.startswith("/search/") and b"model.restrict" in body:
            count += 1
    return count


def _query(profile: str, tenant: str, top_k: int, **extra) -> dict:
    query = {
        "query": "",
        "type": "video",
        "profile": profile,
        "tenant_id": tenant,
        "top_k": top_k,
    }
    query.update(extra)
    return query


def _mv_query(top_k: int, tag: str, strategy: str = "float_float", **extra) -> dict:
    dim = schema_tensor_dim(MV_PROFILE, "embedding")
    embedding = np.zeros((1, dim), dtype=np.float32)
    embedding[0, 0] = embedding[0, 1] = 1.0
    return _query(
        MV_PROFILE,
        MV_TENANT,
        top_k,
        strategy=strategy,
        query_embeddings=embedding,
        filters={"video_title": tag},
        **extra,
    )


def _ann_query(top_k: int, tag: str, profile: str = ANN_PROFILE, **extra) -> dict:
    embedding = np.zeros(768, dtype=np.float32)
    embedding[0] = 1.0
    return _query(
        profile,
        ANN_TENANT,
        top_k,
        strategy="float_float",
        query_embeddings=embedding,
        filters={"video_title": tag},
        **extra,
    )


def _window(top_k: int) -> int:
    budget = search_backend_module._source_collapse_fetch_limit(top_k, {})
    return max(1, budget // top_k)


def _assert_matches_reference(results, segments, top_k: int) -> None:
    """Each source carries its best segments, in the order a complete
    segment ranking gives; tied segments may appear in either order."""
    by_source: dict = {}
    for hit in segments:
        by_source.setdefault(hit.document.metadata["source_id"], []).append(
            (hit.score, hit.document.id)
        )
    ranked = sorted(by_source.items(), key=lambda item: (-max(item[1])[0], item[0]))[
        :top_k
    ]
    window = _window(top_k)
    assert [(hit.document.metadata["source_id"], hit.score) for hit in results] == [
        (source_id, max(hits)[0]) for source_id, hits in ranked
    ]
    for hit, (_, hits) in zip(results, ranked):
        scores = sorted((score for score, _ in hits), reverse=True)[:window]
        assert [row["score"] for row in hit.matched_segments] == scores
        assert hit.segments_in_window == len(scores)
        rows = [(row["score"], row["document_id"]) for row in hit.matched_segments]
        assert set(rows) <= set(hits)
        assert rows == sorted(rows, key=lambda row: (-row[0], row[1]))
        assert hit.document.id == rows[0][1]


def _complete_segments(backend, query: dict):
    reference = dict(query, result_granularity="segment", top_k=1000)
    segments = backend.search(reference)
    assert segments.total_count == len(segments)
    return segments


class TestSourceIdentityAttribute:
    def test_every_shipped_source_identity_is_a_groupable_attribute(self):
        identities = {}
        for path in sorted(SCHEMAS_DIR.glob("*_schema.json")):
            schema = json.loads(path.read_text())
            if (schema.get("document_mapping") or {}).get("id"):
                identities[schema["name"]] = (
                    search_backend_module._source_identity_attribute(
                        schema, schema_name=schema["name"]
                    )
                )
        assert identities == {
            "audio_content": "audio_id",
            "code_lateon_mv": "code_id",
            "document_text": "document_id",
            "document_visual": "document_id",
            "image_colpali_mv": "image_id",
            "lateon_mv": "text_id",
            "video_colpali_smol500_mv_frame": "video_id",
            "video_colqwen_omni_mv_chunk_30s": "video_id",
            "video_xclip_sv_chunk_6s": "video_id",
            "wiki_pages": "doc_id",
        }

    def test_a_source_identity_without_an_attribute_is_rejected(self):
        schema = {
            "name": "notes",
            "document_mapping": {"id": "note_id"},
            "document": {
                "fields": [
                    {
                        "name": "note_id",
                        "type": "string",
                        "indexing": ["summary", "index"],
                    }
                ]
            },
        }
        with pytest.raises(ValueError) as excinfo:
            search_backend_module._source_identity_attribute(
                schema, schema_name="notes"
            )
        assert str(excinfo.value) == (
            "Schema 'notes' source identity field 'note_id' is not an "
            "attribute; source granularity groups by it"
        )


class TestGroupedSourcesWithoutAnn:
    def test_a_dominant_source_leaves_room_for_every_other_source(self, counted):
        backend, proxy = counted
        query = _mv_query(10, "bigcorpus")
        before = _search_requests(proxy)

        results = backend.search(query)

        assert _search_requests(proxy) - before == 1
        assert [hit.document.metadata["source_id"] for hit in results] == [
            "bigdom",
            *BIG_OTHERS,
        ]
        assert [hit.document.id for hit in results] == [
            "bigdom_000",
            *(f"{source_id}_000" for source_id in BIG_OTHERS),
        ]
        assert [row["document_id"] for row in results[0].matched_segments] == [
            "bigdom_000",
            "bigdom_001",
            "bigdom_002",
            "bigdom_003",
        ]
        assert [set(row) for row in results[0].matched_segments] == [
            {"document_id", "score", "start_time", "end_time"}
        ] * 4
        assert [hit.segments_in_window for hit in results] == [4] + [1] * 9
        assert results.result_granularity == "source"
        assert results.total_count == 410
        assert results.num_collapsed_documents == 400
        assert results.source_search_incomplete is False
        _assert_matches_reference(results, _complete_segments(backend, query), 10)

    @pytest.mark.parametrize("strategy", ["float_float", "bm25_only", "default"])
    def test_eighteen_windows_of_one_source_do_not_hide_the_nineteenth(
        self, counted, strategy
    ):
        backend, proxy = counted
        query = _mv_query(2, "smallcorpus", strategy=strategy, query="harbour")
        before = _search_requests(proxy)

        results = backend.search(query)

        assert _search_requests(proxy) - before == 1
        assert [hit.document.metadata["source_id"] for hit in results] == [
            "smalldom",
            "smallother",
        ]
        assert results.total_count == 19
        assert results.source_search_incomplete is False
        _assert_matches_reference(results, _complete_segments(backend, query), 2)

    def test_filters_apply_before_grouping(self, counted):
        backend, _ = counted

        results = backend.search(_mv_query(10, "minority"))

        assert [hit.document.metadata["source_id"] for hit in results] == BIG_OTHERS
        assert results.total_count == 9
        assert results.source_search_incomplete is False

    def test_tied_sources_order_by_source_identity(self, counted):
        backend, _ = counted

        results = backend.search(_mv_query(3, "tiecorpus"))

        assert [
            (hit.document.metadata["source_id"], hit.document.id) for hit in results
        ] == [("tie_a", "tie_a_0"), ("tie_b", "tie_b_0"), ("tie_c", "tie_c_0")]
        assert len({hit.score for hit in results}) == 1
        assert [
            [row["document_id"] for row in hit.matched_segments] for hit in results
        ] == [["tie_a_0", "tie_a_1"], ["tie_b_0", "tie_b_1"], ["tie_c_0", "tie_c_1"]]

    def test_no_match_is_an_empty_complete_batch(self, counted):
        backend, proxy = counted
        before = _search_requests(proxy)

        results = backend.search(_mv_query(10, "absentcorpus"))

        assert _search_requests(proxy) - before == 1
        assert list(results) == []
        assert results.result_granularity == "source"
        assert results.total_count == 0
        assert results.num_collapsed_documents == 0
        assert results.source_search_incomplete is False


class TestGroupedSourcesWithAnn:
    def test_a_saturated_candidate_budget_reports_an_incomplete_search(self, counted):
        backend, proxy = counted
        before = _search_requests(proxy)

        results = backend.search(_ann_query(10, "bigcorpus"))

        assert _search_requests(proxy) - before == 1
        assert [hit.document.metadata["source_id"] for hit in results] == ["bigdom"]
        assert [row["document_id"] for row in results[0].matched_segments] == [
            "bigdom_000",
            "bigdom_001",
            "bigdom_002",
            "bigdom_003",
        ]
        assert results.total_count == 40
        assert results.source_search_incomplete is True

    def test_the_widest_budget_still_reports_what_it_could_not_reach(self, counted):
        backend, _ = counted

        results = backend.search(_ann_query(10, "bigcorpus", profile=ANN_WIDE_PROFILE))

        assert [hit.document.metadata["source_id"] for hit in results] == ["bigdom"]
        assert results.total_count == 100
        assert results.source_search_incomplete is True

    def test_a_budget_that_covers_the_matches_is_complete(self, counted):
        backend, _ = counted

        results = backend.search(_ann_query(10, "smallcorpus"))

        assert [hit.document.metadata["source_id"] for hit in results] == [
            "smalldom",
            "smallother",
        ]
        assert [hit.document.id for hit in results] == ["smalldom_00", "smallother_00"]
        assert results.total_count == 19
        assert results.source_search_incomplete is False

    def test_the_profile_oversample_sets_the_candidate_budget(self, counted):
        backend, _ = counted

        narrow = backend.search(_ann_query(2, "smallcorpus"))
        wide = backend.search(_ann_query(2, "smallcorpus", profile=ANN_WIDE_PROFILE))

        assert [hit.document.metadata["source_id"] for hit in narrow] == ["smalldom"]
        assert narrow.total_count == 8
        assert narrow.source_search_incomplete is True
        assert [hit.document.metadata["source_id"] for hit in wide] == [
            "smalldom",
            "smallother",
        ]
        assert wide.total_count == 19
        assert wide.source_search_incomplete is False

    def test_filters_apply_before_the_candidate_budget(self, counted):
        backend, _ = counted

        results = backend.search(_ann_query(10, "minority"))

        assert [hit.document.metadata["source_id"] for hit in results] == BIG_OTHERS
        assert results.total_count == 9
        assert results.source_search_incomplete is False


class TestNaturalPassagesWithTheDefaultRanking:
    def test_the_default_semantic_ranking_groups_the_served_embeddings(self, counted):
        backend, proxy = counted
        query = {
            "query": "feeding a sourdough starter and baking the bread",
            "type": "document",
            "profile": TEXT_PROFILE,
            "tenant_id": TEXT_TENANT,
            "top_k": 3,
        }
        before = _search_requests(proxy)

        results = backend.search(query)

        assert _search_requests(proxy) - before == 1
        assert [hit.document.metadata["source_id"] for hit in results] == [
            "sourdough_guide",
            "rye_loaf",
            "pizza_dough",
        ]
        assert results.total_count == 23
        assert results.source_search_incomplete is False
        _assert_matches_reference(results, _complete_segments(backend, query), 3)


class TestConcurrentProfiles:
    def test_concurrent_profiles_each_get_their_own_grouped_answer(self, counted):
        backend, proxy = counted
        queries = [
            _mv_query(10, "bigcorpus"),
            _ann_query(10, "bigcorpus"),
            _mv_query(10, "bigcorpus", result_granularity="segment"),
            _ann_query(2, "smallcorpus", profile=ANN_WIDE_PROFILE),
        ] * 3
        expected = [
            ("source", ["bigdom", *BIG_OTHERS], False),
            ("source", ["bigdom"], True),
            ("segment", ["bigdom"] * 10, False),
            ("source", ["smalldom", "smallother"], False),
        ] * 3
        barrier = threading.Barrier(len(queries))
        outcomes: dict = {}

        def run(slot: int, query: dict) -> None:
            barrier.wait()
            try:
                results = backend.search(query)
                outcomes[slot] = (
                    results.result_granularity,
                    [hit.document.metadata["source_id"] for hit in results],
                    results.source_search_incomplete,
                )
            except BaseException as exc:
                outcomes[slot] = exc

        before = _search_requests(proxy)
        threads = [
            threading.Thread(target=run, args=(slot, query))
            for slot, query in enumerate(queries)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)

        assert outcomes == dict(enumerate(expected))
        assert _search_requests(proxy) - before == len(queries)


def _fault_backend(config_manager, vespa_instance, intercept):
    proxy = InterceptFaultProxy(
        f"http://localhost:{vespa_instance['http_port']}", intercept
    )
    proxy.__enter__()
    return _backend(config_manager, proxy.port, retry_config=FAST_RETRY), proxy


def _grouped_body(total_count: int, root_children: list, **root) -> dict:
    return {
        "root": {
            "id": "toplevel",
            "relevance": 1.0,
            "fields": {"totalCount": total_count},
            "coverage": {"coverage": 100, "documents": 440, "full": True},
            "children": root_children,
            **root,
        }
    }


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            _grouped_body(
                3,
                [{"id": "group:root:0", "relevance": 1.0}],
                coverage={"coverage": 40, "degraded": {"timeout": True}},
            ),
            "Vespa query coverage degraded",
        ),
        (
            _grouped_body(
                3,
                [{"id": "group:root:0", "relevance": 1.0}],
                errors=[{"code": 12, "summary": "Timed out"}],
            ),
            "Vespa query returned errors",
        ),
        (
            _grouped_body(3, [{"id": "group:root:0", "relevance": 1.0}]),
            "Vespa grouped 3 matched segments into no source groups",
        ),
        (
            _grouped_body(3, []),
            "Vespa response has no grouping root",
        ),
        (
            {"root": {"id": "toplevel", "children": []}},
            "Vespa grouped response has no totalCount",
        ),
    ],
    ids=["degraded", "errors", "groups_lost", "root_lost", "count_lost"],
)
def test_a_degraded_grouped_response_raises_after_bounded_retries(
    vespa_instance, config_manager, mv_corpus, body, message
):
    def intercept(method, path, _body):
        return (200, body) if path.startswith("/search/") else None

    backend, proxy = _fault_backend(config_manager, vespa_instance, intercept)
    try:
        with pytest.raises(VespaError) as excinfo:
            backend.search(_mv_query(10, "bigcorpus"))
        assert message in str(excinfo.value), str(excinfo.value)
        assert _search_requests(proxy) == FAST_RETRY.max_attempts
    finally:
        backend.close()
        proxy.__exit__(None, None, None)


def test_a_failing_vespa_raises_instead_of_answering_empty(
    vespa_instance, config_manager, mv_corpus
):
    def intercept(method, path, _body):
        if path.startswith("/search/"):
            return 503, {"root": {"errors": [{"code": 8, "message": "unavailable"}]}}
        return None

    backend, proxy = _fault_backend(config_manager, vespa_instance, intercept)
    try:
        with pytest.raises(VespaError):
            backend.search(_mv_query(10, "bigcorpus"))
        assert _search_requests(proxy) == FAST_RETRY.max_attempts
    finally:
        backend.close()
        proxy.__exit__(None, None, None)
