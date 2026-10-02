"""Golden evaluation scores a content-hash tenant on its stored source titles.

Golden sets name each expected video by its original filename stem. The hash
tenant stores three videos under sha256 source ids, as multipart uploads are;
the filename tenant stores the same videos under their filename stems. Both are
served by the runtime search route over a real socket against real Vespa, and
the query embeddings come from a deterministic stand-in for the ColPali
inference sidecar, so every golden query's ranking is fixed and known. Every
golden consumer must score the two tenants identically.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import math
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.search.service import SearchService
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_evaluation.core.inspect_scorers import precision_scorer, recall_scorer
from cogniverse_evaluation.core.solvers import create_retrieval_solver
from cogniverse_evaluation.evaluators.golden_dataset import GoldenDatasetEvaluator
from cogniverse_evaluation.quality_monitor import QualityMonitor
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_foundation.config.utils import get_config
from cogniverse_foundation.telemetry.context import serialize_search_results
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.utils.vespa_test_helpers import (
    deploy_tenant_schema,
    make_config_manager,
    schema_tensor_dim,
    serve_search_route,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.requires_docker]

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
SHIPPED_PROFILE = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
][BASE_SCHEMA]
ENCODER_SERVICE = SHIPPED_PROFILE["inference_services"]["embedding"]
PROFILE = "golden_frames"
DIM = schema_tensor_dim(BASE_SCHEMA, "embedding")
BLOCK = 32
RUN = uuid.uuid4().hex[:8]
HASH_TENANT = f"goldhash{RUN}:prod"
FILENAME_TENANT = f"goldname{RUN}:prod"
UNTITLED_TENANT = f"goldnotitle{RUN}:prod"

TITLES = {
    "v_-uJnucdW6DY": "v_-uJnucdW6DY.mp4",
    "v_-HpCLXdtcas": "v_-HpCLXdtcas.mkv",
    "v_-IMXSEIabMM": "v_-IMXSEIabMM.mp4",
}
STEMS = list(TITLES)
BLOCK_OF = {stem: index for index, stem in enumerate(STEMS)}
HASH_ID = {
    stem: hashlib.sha256(title.encode()).hexdigest() for stem, title in TITLES.items()
}

# Each golden query ranks the three videos in a fixed order; its expected video
# sits at a different rank in each.
RANKINGS = {
    "golden query alpha": ["v_-uJnucdW6DY", "v_-HpCLXdtcas", "v_-IMXSEIabMM"],
    "golden query beta": ["v_-IMXSEIabMM", "v_-uJnucdW6DY", "v_-HpCLXdtcas"],
    "golden query gamma": ["v_-HpCLXdtcas", "v_-IMXSEIabMM", "v_-uJnucdW6DY"],
}
EXPECTED = {
    "golden query alpha": "v_-uJnucdW6DY",
    "golden query beta": "v_-HpCLXdtcas",
    "golden query gamma": "v_-IMXSEIabMM",
}
EXPECTED_MRR = {
    "golden query alpha": 1.0,
    "golden query beta": 1 / 3,
    "golden query gamma": 1 / 2,
}
EXPECTED_NDCG = {
    "golden query alpha": 1.0,
    "golden query beta": 1 / math.log2(4),
    "golden query gamma": 1 / math.log2(3),
}
GOLDEN_ROWS = [
    {
        "query": query,
        "expected_videos": [EXPECTED[query]],
        "ground_truth": query,
        "query_type": "question",
        "source": "test",
    }
    for query in RANKINGS
]


def _block(stem: str) -> slice:
    start = BLOCK_OF[stem] * BLOCK
    return slice(start, start + BLOCK)


def _query_vector(query: str) -> list[float]:
    """Positive on the first video's block and half the second's, so both the
    binary first phase and the float second phase rank first > second > third.
    """
    first, second, _ = RANKINGS[query]
    vector = np.zeros(DIM, dtype=np.float32)
    vector[_block(first)] = 3.0
    second_block = _block(second)
    vector[second_block.start : second_block.start + BLOCK // 2] = 2.0
    return vector.tolist()


def _document_tensors(stem: str) -> tuple[dict, dict]:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[_block(stem)] = 1.0
    binary = np.packbits((vector > 0).astype(np.uint8)).astype(np.int8)
    return {"0": vector.tolist()}, {"0": binary.tolist()}


class _PoolingStandIn(http.server.BaseHTTPRequestHandler):
    """Answers vLLM ``/pooling`` with the fixed vector of the query text."""

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["messages"][0]["content"][0]["text"]
        if self.path != "/pooling" or text not in RANKINGS:
            self.send_response(500)
            self.end_headers()
            return
        payload = json.dumps({"data": [{"data": [_query_vector(text)]}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


@pytest.fixture(autouse=True)
def _telemetry_disabled():
    """The search route opens spans; keep them off any collector."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if telemetry_manager_module._telemetry_manager is installed:
        telemetry_manager_module._telemetry_manager = None


@pytest.fixture(scope="module")
def encoder_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PoolingStandIn)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _feed(shared_vespa, schema: str, video_id: str, stem: str, title: str | None):
    embedding, binary = _document_tensors(stem)
    fields = {
        "video_id": video_id,
        "segment_id": 0,
        "embedding": embedding,
        "embedding_binary": binary,
        "source_url": f"s3://cogniverse-ingest/tenant/{video_id}.mp4",
    }
    if title is not None:
        fields["video_title"] = title
    response = requests.post(
        f"http://localhost:{shared_vespa['http_port']}/document/v1/"
        f"content/{schema}/docid/{video_id}_seg_0",
        json={"fields": fields},
        timeout=30,
    )
    assert response.status_code == 200, response.text


@pytest.fixture(scope="module")
def corpus(shared_vespa, encoder_url):
    """The three videos under hash ids, under filename ids, and a hash tenant
    whose second video was stored without a title."""
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()
    config_manager = make_config_manager(
        shared_vespa, inference_service_urls={ENCODER_SERVICE: encoder_url}
    )
    config_manager.get_system_config()
    for tenant in (HASH_TENANT, FILENAME_TENANT, UNTITLED_TENANT):
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(PROFILE, SHIPPED_PROFILE), tenant_id=tenant
        )
        schema = deploy_tenant_schema(
            shared_vespa,
            tenant_id=tenant,
            base_schema_name=BASE_SCHEMA,
            config_manager=config_manager,
        )
        for index, (stem, title) in enumerate(TITLES.items()):
            if tenant == FILENAME_TENANT:
                _feed(shared_vespa, schema, stem, stem, title)
            else:
                stored_title = (
                    None if tenant == UNTITLED_TENANT and index == 1 else title
                )
                _feed(shared_vespa, schema, HASH_ID[stem], stem, stored_title)
    with serve_search_route(
        config_manager, tenants=[HASH_TENANT, FILENAME_TENANT, UNTITLED_TENANT]
    ) as runtime_url:
        yield {"runtime_url": runtime_url, "config_manager": config_manager}
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()


async def _monitor(
    phoenix_container, runtime_url: str, tenant_id: str
) -> QualityMonitor:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        }
    )
    manager = ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)
    content = json.dumps(GOLDEN_ROWS)
    if await manager.load_blob("config", "golden_set_ground_truth") != content:
        _, version = await manager.save_blob_versioned(
            "config",
            "golden_set_ground_truth",
            content,
            consumed_example_ids=["golden_source_title_test:golden_rows"],
            decision="promote",
            scored=False,
            score=None,
            base_score=None,
            candidate_score=None,
        )
        await manager.activate_version("config", "golden_set_ground_truth", version)
    return QualityMonitor(
        tenant_id=tenant_id,
        search_profile=PROFILE,
        runtime_url=runtime_url,
        phoenix_http_endpoint=phoenix_container["http_endpoint"],
        llm_base_url="http://127.0.0.1:1",
        llm_model="unused",
        golden_dataset_path="unused-golden-rows-come-from-the-blob",
        telemetry_provider=provider,
    )


def _per_query(result, field: str) -> dict:
    return {entry["query"]: entry[field] for entry in result.per_query_scores}


class TestQualityMonitorGoldenSet:
    @pytest.mark.asyncio
    async def test_hash_and_filename_tenants_score_identically_and_concurrently(
        self, corpus, phoenix_container
    ):
        hash_monitor = await _monitor(
            phoenix_container, corpus["runtime_url"], HASH_TENANT
        )
        filename_monitor = await _monitor(
            phoenix_container, corpus["runtime_url"], FILENAME_TENANT
        )

        hash_result, filename_result = await asyncio.gather(
            hash_monitor.evaluate_golden_set(), filename_monitor.evaluate_golden_set()
        )

        for result in (hash_result, filename_result):
            assert (result.failed_query_count, result.failed_queries) == (0, [])
            assert result.query_count == 3
            assert _per_query(result, "retrieved_videos") == RANKINGS
            assert _per_query(result, "mrr") == pytest.approx(EXPECTED_MRR, abs=1e-12)
            assert _per_query(result, "ndcg") == pytest.approx(EXPECTED_NDCG, abs=1e-12)
            assert result.mean_mrr == pytest.approx(11 / 18, abs=1e-12)
            assert result.mean_ndcg == pytest.approx(
                sum(EXPECTED_NDCG.values()) / 3, abs=1e-12
            )
            assert [entry["query"] for entry in result.low_scoring_queries] == []
            assert [entry["query"] for entry in result.high_scoring_queries] == [
                "golden query alpha"
            ]

    @pytest.mark.asyncio
    async def test_hash_tenant_baseline_is_stored_with_its_real_scores(
        self, corpus, phoenix_container
    ):
        monitor = await _monitor(phoenix_container, corpus["runtime_url"], HASH_TENANT)

        await monitor.evaluate_golden_set()
        second = await monitor.evaluate_golden_set()

        frame = await monitor._get_dataset_store().get_dataset(
            f"quality-baseline-{monitor.tenant_id}"
        )
        payloads = [json.loads(row["input"]["payload"]) for _, row in frame.iterrows()]
        assert [
            (
                payload["mean_mrr"],
                payload["query_count"],
                payload["failed_query_count"],
            )
            for payload in payloads[-2:]
        ] == [(pytest.approx(11 / 18, abs=1e-12), 3, 0)] * 2
        assert second.baseline_mrr == pytest.approx(11 / 18, abs=1e-12)

    @pytest.mark.asyncio
    async def test_untitled_result_fails_its_queries_instead_of_scoring_zero(
        self, corpus, phoenix_container
    ):
        monitor = await _monitor(
            phoenix_container, corpus["runtime_url"], UNTITLED_TENANT
        )

        with pytest.raises(
            RuntimeError, match="No golden queries evaluated successfully"
        ):
            await monitor.evaluate_golden_set()

        with pytest.raises(DatasetNotFoundError):
            await monitor._get_dataset_store().get_dataset(
                f"quality-baseline-{monitor.tenant_id}"
            )


class _State:
    def __init__(self, query: str):
        self.input = {"query": query}
        self.metadata: dict = {}
        self.output = None


class _Target:
    def __init__(self, target: list[str]):
        self.target = target


class TestRetrievalSolverScoring:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tenant_id", [HASH_TENANT, FILENAME_TENANT])
    async def test_scorers_match_golden_ids_through_source_titles(
        self, corpus, tenant_id
    ):
        query = "golden query gamma"
        solver = create_retrieval_solver(
            profiles=[PROFILE],
            strategies=["default"],
            config={
                "runtime_url": corpus["runtime_url"],
                "tenant_id": tenant_id,
                "top_k": 10,
            },
        )

        state = await solver(_State(query), generate=None)

        rows = state.metadata["search_results"][f"{PROFILE}_default"]["results"]
        ids = HASH_ID if tenant_id == HASH_TENANT else {s: s for s in STEMS}
        assert [(row["video_id"], row["source_title"]) for row in rows] == [
            (ids[stem], TITLES[stem]) for stem in RANKINGS[query]
        ]
        target = _Target([EXPECTED[query]])
        precision = await precision_scorer()(state, target)
        recall = await recall_scorer()(state, target)
        assert (precision.value, recall.value) == (pytest.approx(1 / 3), 1.0)


def _service(config_manager, tenant_id: str) -> SearchService:
    return SearchService(
        config=get_config(tenant_id=tenant_id, config_manager=config_manager),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )


def _service_rows(config_manager, tenant_id: str, query: str) -> list[dict]:
    """The span result rows SearchService records for one real search."""
    results = _service(config_manager, tenant_id).search(
        query=query, profile=PROFILE, tenant_id=tenant_id
    )
    return json.loads(serialize_search_results(results))


class TestTraceRowScoring:
    @pytest.mark.asyncio
    async def test_span_rows_carry_titles_the_golden_evaluator_matches(self, corpus):
        query = "golden query beta"
        rows = _service_rows(corpus["config_manager"], HASH_TENANT, query)

        assert [(row["source_id"], row["source_title"]) for row in rows] == [
            (HASH_ID[stem], TITLES[stem]) for stem in RANKINGS[query]
        ]
        evaluator = GoldenDatasetEvaluator(
            {query: {"expected_videos": [EXPECTED[query]]}}
        )
        result = await evaluator.evaluate(
            input=query, output=rows, metadata={"is_test_query": True}
        )
        assert (result.score, result.label) == (pytest.approx(1 / 3), "poor")
        assert result.metadata["retrieved_videos"] == RANKINGS[query]

    @pytest.mark.asyncio
    async def test_untitled_span_row_is_not_evaluable(self, corpus):
        query = "golden query alpha"
        rows = _service_rows(corpus["config_manager"], UNTITLED_TENANT, query)
        untitled = HASH_ID["v_-HpCLXdtcas"]

        assert [(row["source_id"], row["source_title"]) for row in rows] == [
            (HASH_ID["v_-uJnucdW6DY"], TITLES["v_-uJnucdW6DY"]),
            (untitled, None),
            (HASH_ID["v_-IMXSEIabMM"], TITLES["v_-IMXSEIabMM"]),
        ]
        evaluator = GoldenDatasetEvaluator(
            {query: {"expected_videos": [EXPECTED[query]]}}
        )
        result = await evaluator.evaluate(
            input=query, output=rows, metadata={"is_test_query": True}
        )
        assert (result.score, result.label) == (-1.0, "not_evaluable")
        assert f"'{untitled}_seg_0' carries no source_title" in result.explanation

    def test_segment_granularity_results_carry_titles(self, corpus):
        query = "golden query alpha"

        results = _service(corpus["config_manager"], HASH_TENANT).search(
            query=query,
            profile=PROFILE,
            tenant_id=HASH_TENANT,
            result_granularity="segment",
        )

        assert [
            (result.document.id, result.to_dict()["source_title"]) for result in results
        ] == [(f"{HASH_ID[stem]}_seg_0", TITLES[stem]) for stem in RANKINGS[query]]
