"""
Integration tests for QualityMonitor with real Vespa search + real Phoenix.

Full round-trip: golden query → real /search → real Vespa with real data
→ real MRR/nDCG scoring → store baseline in real Phoenix → detect degradation.

Uses fixture-owned Vespa and Phoenix with a separate tenant per evaluation.
ColPali model generates real embeddings for test documents.
"""

import json
import logging
import socket
import time
import uuid
from datetime import datetime

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image
from vespa.application import Vespa

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.optimizer.golden_set_ground_truth import (
    GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
    GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
    GoldenSetGroundTruthStoreUnavailableError,
)
from cogniverse_core.common.models.model_loaders import RemoteColPaliLoader
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_evaluation.quality_monitor import (
    AgentEvalResult,
    AgentType,
    GoldenEvalResult,
    LiveEvalResult,
    OptimizationTrigger,
    QualityMonitor,
    Verdict,
)
from cogniverse_foundation.telemetry.providers.base import (
    DatasetStoreUnavailableError,
)
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "TomoroAI/tomoro-colqwen3-embed-4b"
SEARCH_PROFILE = "video_colpali_smol500_mv_frame"
GOLDEN_QUERIES = [
    {
        "query": "a solid red square",
        "expected_videos": ["v_red"],
        "ground_truth": "A solid red square",
        "query_type": "question",
        "source": "test",
    },
    {
        "query": "a solid blue square",
        "expected_videos": ["v_blue"],
        "ground_truth": "A solid blue square",
        "query_type": "question",
        "source": "test",
    },
    {
        "query": "cat sleeping on sofa",
        "expected_videos": ["v_nonexistent"],
        "ground_truth": "Cat sleeping",
        "query_type": "question",
        "source": "test",
    },
]


def _embeddings_to_vespa_tensors(embeddings: np.ndarray):
    """Convert embeddings to Vespa tensor format."""
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}
    return float_dict, binary_dict


@pytest.fixture(scope="module")
def vllm_colpali_url(vllm_sidecar, config_manager):
    """Spawn the ColPali vLLM sidecar and register its URL under the
    ``vllm_colpali`` service name. The /search/ route resolves the query
    encoder through QueryEncoderFactory → SystemConfig.inference_service_urls
    (the tenant's ``video_colpali_smol500_mv_frame`` profile declares
    ``inference_services.embedding='vllm_colpali'``), so without this
    registration every golden query 400s. Mirrors the ``tomoro_search_url``
    wiring in conftest.py; the module-scoped ``config_manager`` re-seeds
    SystemConfig per module, so the URL does not leak past this module.
    """
    url = vllm_sidecar.spawn(
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
    sys_cfg = config_manager.get_system_config()
    sys_cfg.inference_service_urls = dict(sys_cfg.inference_service_urls)
    sys_cfg.inference_service_urls["vllm_colpali"] = url
    config_manager.set_system_config(sys_cfg)
    # Drop any encoder cached before the URL existed (would be a local encoder).
    QueryEncoderFactory._encoder_cache.clear()
    yield url
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(scope="module")
def colpali_client(vllm_colpali_url):
    loader = RemoteColPaliLoader(
        model_name=COLPALI_MODEL_NAME,
        config={"remote_inference_url": vllm_colpali_url},
        logger=logger,
    )
    client, _ = loader.load_model()
    yield client
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture
def qm_tenant():
    return f"qm:{uuid.uuid4().hex}"


@pytest.fixture(scope="module")
def embedded_documents(colpali_client):
    """Embed the three color images once; each test feeds its own tenant."""
    documents = []
    for name, color in [
        ("red", (255, 0, 0)),
        ("blue", (0, 0, 255)),
        ("green", (0, 128, 0)),
    ]:
        result = colpali_client.process_images(
            [Image.new("RGB", (224, 224), color=color)],
            model_name=COLPALI_MODEL_NAME,
        )
        embeddings = np.asarray(
            result.get("embeddings") if isinstance(result, dict) else result
        ).astype(np.float32)
        float_dict, binary_dict = _embeddings_to_vespa_tensors(embeddings)
        documents.append(
            {
                "id": name,
                "fields": {
                    "video_id": f"v_{name}",
                    "video_title": f"A solid {name} square",
                    "segment_id": 0,
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "segment_description": f"A solid {name} square",
                    "audio_transcript": "",
                    "embedding": float_dict,
                    "embedding_binary": binary_dict,
                },
            }
        )
    return documents


@pytest.fixture
def seeded_vespa(
    vespa_instance, embedded_documents, config_manager, schema_loader, qm_tenant
):
    """Own the search profile, tenant metadata, schema and indexed documents."""
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        BackendProfileConfig,
    )
    from cogniverse_vespa.backend import VespaBackend
    from tests.utils.async_polling import wait_for_condition_sync

    config_manager.add_backend_profile(
        BackendProfileConfig(
            profile_name=SEARCH_PROFILE,
            type="video",
            schema_name=SEARCH_PROFILE,
            embedding_model=COLPALI_MODEL_NAME,
            model_loader="colpali",
            extra_config={"inference_services": {"embedding": "vllm_colpali"}},
        ),
        tenant_id=qm_tenant,
    )
    backend = VespaBackend(
        backend_config=BackendConfig(
            backend_type="vespa",
            url="http://localhost",
            port=vespa_instance["http_port"],
            tenant_id=qm_tenant,
        ),
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": qm_tenant})
    registry = SchemaRegistry(
        config_manager=config_manager, backend=backend, schema_loader=schema_loader
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    schema_name = registry.deploy_schema(
        tenant_id=qm_tenant, base_schema_name=SEARCH_PROFILE
    )
    tenant_name = qm_tenant.split(":")[1]
    backend.create_metadata_document(
        schema="tenant_metadata",
        doc_id=qm_tenant,
        fields={
            "tenant_full_id": qm_tenant,
            "org_id": "qm",
            "tenant_name": tenant_name,
            "created_at": 1700000000000,
            "created_by": "quality-monitor-test",
            "status": "active",
            "schemas_deployed": [SEARCH_PROFILE],
        },
    )

    statuses = {}
    app = Vespa(url=f"http://localhost:{vespa_instance['http_port']}")
    app.feed_iterable(
        iter=embedded_documents,
        schema=schema_name,
        namespace="video",
        callback=lambda response, doc_id: statuses.update(
            {doc_id: response.status_code}
        ),
    )
    assert statuses == {"red": 200, "blue": 200, "green": 200}

    def indexed_ids():
        response = app.query(
            yql=f"select video_id from {schema_name} where true",
            hits=10,
        )
        return {hit["fields"]["video_id"] for hit in response.hits}

    wait_for_condition_sync(
        lambda: indexed_ids() == {"v_red", "v_blue", "v_green"},
        timeout=60,
        description=f"indexed color documents for {qm_tenant}",
    )
    return backend


async def _seed_golden_rows(provider, tenant_id, rows):
    manager = ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)
    content = json.dumps(rows)
    if await manager.load_blob("config", "golden_set_ground_truth") == content:
        return
    _, version = await manager.save_blob_versioned(
        "config",
        "golden_set_ground_truth",
        content,
        consumed_example_ids=["quality_monitor_test:golden_rows"],
        decision="promote",
        scored=False,
        score=None,
        base_score=None,
        candidate_score=None,
    )
    await manager.activate_version("config", "golden_set_ground_truth", version)


@pytest.fixture
async def monitor_with_real_search(
    real_telemetry,
    phoenix_container,
    config_manager,
    schema_loader,
    seeded_vespa,
    qm_tenant,
):
    """Load tenant-owned Phoenix ground truth and call the real search router."""
    from cogniverse_runtime.admin import tenant_manager
    from cogniverse_runtime.routers import search

    provider = real_telemetry.get_provider(tenant_id=qm_tenant)
    await _seed_golden_rows(provider, qm_tenant, GOLDEN_QUERIES)
    await _seed_golden_rows(provider, qm_tenant, GOLDEN_QUERIES)
    manager = ArtifactManager(telemetry_provider=provider, tenant_id=qm_tenant)
    assert await manager.list_versions("config", "golden_set_ground_truth") == [
        {"version": 1, "name": f"dspy-config-{qm_tenant}-golden_set_ground_truth-v1"}
    ]
    frame = await provider.datasets.get_dataset(
        f"dspy-config-{qm_tenant}-golden_set_ground_truth"
    )
    assert frame.to_dict("records") == [
        {"input": {"content": json.dumps(GOLDEN_QUERIES)}, "output": {}, "metadata": {}}
    ]
    app = FastAPI()
    app.include_router(search.router, prefix="/search")
    app.dependency_overrides[search.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = lambda: (
        schema_loader
    )
    monitor = QualityMonitor(
        tenant_id=qm_tenant,
        runtime_url="http://testserver",
        phoenix_http_endpoint=phoenix_container["http_endpoint"],
        llm_base_url=get_llm_base_url(),
        llm_model=get_llm_model(),
        golden_dataset_path="",
        telemetry_provider=provider,
    )
    monitor._http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )
    tenant_manager.set_backend(seeded_vespa)
    try:
        yield monitor
    finally:
        tenant_manager.set_backend(None)
        await monitor.close()


async def _dataset_names(monitor, prefix):
    from phoenix.client import AsyncClient

    async with httpx.AsyncClient(base_url=monitor.phoenix_http_endpoint) as client:
        datasets = await AsyncClient(http_client=client).datasets.list()
    return [
        row["name"]
        for row in datasets
        if row["name"].startswith(f"{prefix}-{monitor.tenant_id}-")
    ]


def _assert_golden_result(result):
    assert result.query_count == 3
    assert result.failed_query_count == 0
    assert result.failed_queries == []
    assert result.mean_mrr == pytest.approx(2 / 3)
    assert result.mean_ndcg == pytest.approx(2 / 3)
    assert result.mean_precision_at_5 == pytest.approx(2 / 9)
    assert [entry["query"] for entry in result.per_query_scores] == [
        "a solid red square",
        "a solid blue square",
        "cat sleeping on sofa",
    ]
    assert [entry["expected_videos"] for entry in result.per_query_scores] == [
        ["v_red"],
        ["v_blue"],
        ["v_nonexistent"],
    ]
    assert [entry["mrr"] for entry in result.per_query_scores] == [1.0, 1.0, 0.0]
    assert [entry["ndcg"] for entry in result.per_query_scores] == [1.0, 1.0, 0.0]
    assert [entry["retrieved_videos"][0] for entry in result.per_query_scores[:2]] == [
        "v_red",
        "v_blue",
    ]
    for entry in result.per_query_scores:
        assert len(entry["retrieved_videos"]) == 3
        assert set(entry["retrieved_videos"]) == {"v_red", "v_blue", "v_green"}
    assert [entry["query"] for entry in result.low_scoring_queries] == [
        "cat sleeping on sofa"
    ]
    assert [entry["query"] for entry in result.high_scoring_queries] == [
        "a solid red square",
        "a solid blue square",
    ]


@pytest.mark.integration
class TestGoldenEvalRealVespa:
    """Golden eval against real Vespa with real ingested data."""

    @pytest.mark.asyncio
    async def test_golden_eval_with_real_data(self, monitor_with_real_search):
        monitor = monitor_with_real_search
        result = await monitor.evaluate_golden_set()

        assert await monitor._load_golden_queries_async() == GOLDEN_QUERIES
        assert result.tenant_id == monitor.tenant_id
        _assert_golden_result(result)

    @pytest.mark.asyncio
    async def test_golden_eval_stores_baseline_in_phoenix(
        self, monitor_with_real_search
    ):
        monitor = monitor_with_real_search
        result = await monitor.evaluate_golden_set()

        _assert_golden_result(result)
        assert result.baseline_mrr is None
        assert result.baseline_ndcg is None
        assert await monitor._read_baseline_metric("mean_mrr") == pytest.approx(2 / 3)
        frame = await monitor._get_dataset_store().get_dataset(
            f"quality-baseline-{monitor.tenant_id}"
        )
        assert len(frame) == 1
        assert json.loads(_roundtrip_value(frame, "payload")) == {
            "timestamp": result.timestamp.isoformat(),
            "mean_mrr": pytest.approx(2 / 3),
            "mean_ndcg": pytest.approx(2 / 3),
            "mean_precision_at_5": pytest.approx(2 / 9),
            "query_count": 3,
            "failed_query_count": 0,
        }

    @pytest.mark.asyncio
    async def test_degradation_detection_with_real_baseline(
        self, monitor_with_real_search, caplog
    ):
        monitor = monitor_with_real_search
        result1 = await monitor.evaluate_golden_set()
        _assert_golden_result(result1)
        assert result1.baseline_mrr is None
        await monitor._store_golden_eval_result(
            GoldenEvalResult(
                timestamp=datetime.utcnow(),
                tenant_id=monitor.tenant_id,
                mean_mrr=0.95,
                mean_ndcg=0.95,
                mean_precision_at_5=0.8,
                query_count=3,
            )
        )

        result2 = await monitor.evaluate_golden_set()
        _assert_golden_result(result2)
        assert result2.baseline_mrr == 0.95
        assert result2.baseline_ndcg == 0.95
        with caplog.at_level(
            logging.INFO, logger="cogniverse_evaluation.quality_monitor"
        ):
            verdicts = monitor.check_thresholds(result2, None)
        assert verdicts == {AgentType.SEARCH: Verdict.SKIP}
        assert [
            record.message
            for record in caplog.records
            if record.name == "cogniverse_evaluation.quality_monitor"
        ] == [
            "Golden MRR dropped 29.8% (0.950 → 0.667)",
            "Golden nDCG dropped 29.8% (0.950 → 0.667)",
            "XGBoost overrides search OPTIMIZE → SKIP (expected improvement 0.000 too low)",
        ]


@pytest.mark.integration
class TestForceOptimizationCycle:
    """Configured cycles persist golden results and trigger examples."""

    @pytest.mark.asyncio
    async def test_force_cycle_returns_status_dict(self, monitor_with_real_search):
        monitor = monitor_with_real_search
        result = await monitor.force_optimization_cycle()

        assert result == {
            "status": "ok",
            "agents_triggered": [
                "search",
                "summary",
                "report",
                "gateway",
                "routing",
                "query_enhancement",
                "entity_extraction",
                "profile_selection",
            ],
            "submitted_to_argo": False,
        }
        assert await monitor._load_golden_queries_async() == GOLDEN_QUERIES
        assert await monitor._read_baseline_metric("mean_mrr") == pytest.approx(2 / 3)
        live = await monitor.evaluate_live_traffic()
        assert live.agent_results == {}
        trigger_names = await _dataset_names(monitor, "optimization-trigger")
        assert len(trigger_names) == 1
        frame = await monitor._get_dataset_store().get_dataset(trigger_names[0])
        assert len(frame) == 3
        assert sorted(
            tuple(
                _roundtrip_value(frame.iloc[[i]], col)
                for col in ["agent", "category", "query"]
            )
            for i in range(len(frame))
        ) == [
            ("search", "high_scoring", "a solid blue square"),
            ("search", "high_scoring", "a solid red square"),
            ("search", "low_scoring", "cat sleeping on sofa"),
        ]

    @pytest.mark.requires_lm
    @pytest.mark.local_only
    @pytest.mark.asyncio
    async def test_force_cycle_with_live_spans_returns_ok(
        self, monitor_with_real_search, real_telemetry
    ):
        monitor = monitor_with_real_search
        query = "force cycle live seed"
        real_telemetry.register_project(
            tenant_id=monitor.tenant_id,
            project_name=None,
            otlp_endpoint=real_telemetry.config.provider_config["grpc_endpoint"],
            http_endpoint=real_telemetry.config.provider_config["http_endpoint"],
            use_sync_export=True,
        )
        _emit_agent_span(
            real_telemetry,
            monitor.tenant_id,
            "SearchAgent.process",
            query,
            json.dumps([{"video_id": "v_red", "score": 1.0}]),
        )
        real_telemetry.force_flush(timeout_millis=10000)
        spans = await _wait_for_span(
            monitor._make_span_evaluator(), "SearchAgent.process"
        )
        assert spans["attributes"].map(lambda attrs: attrs["query"]).tolist() == [query]

        result = await monitor.force_optimization_cycle()

        assert result == {
            "status": "ok",
            "agents_triggered": [
                "search",
                "summary",
                "report",
                "gateway",
                "routing",
                "query_enhancement",
                "entity_extraction",
                "profile_selection",
            ],
            "submitted_to_argo": False,
        }
        assert await monitor._load_golden_queries_async() == GOLDEN_QUERIES
        assert await monitor._read_baseline_metric("mean_mrr") == pytest.approx(2 / 3)
        live_names = await _dataset_names(monitor, "quality-live")
        assert len(live_names) == 1
        frame = await monitor._get_dataset_store().get_dataset(live_names[0])
        assert len(frame) == 1
        assert _roundtrip_value(frame, "agent") == "search"
        assert _roundtrip_value(frame, "sample_count") == "1"


def _live_result(tenant_id, sample_count):
    return LiveEvalResult(
        timestamp=datetime(2024, 1, 1),
        tenant_id=tenant_id,
        agent_results={
            AgentType.SEARCH: AgentEvalResult(
                agent=AgentType.SEARCH,
                score=0.75,
                baseline_score=0.9,
                degradation_pct=0.1,
                sample_count=sample_count,
            ),
        },
    )


@pytest.fixture
async def phoenix_monitor(phoenix_container, real_telemetry, qm_tenant):
    monitor = QualityMonitor(
        tenant_id=qm_tenant,
        runtime_url="http://testserver",
        phoenix_http_endpoint=phoenix_container["http_endpoint"],
        llm_base_url=get_llm_base_url(),
        llm_model=get_llm_model(),
        golden_dataset_path="",
        telemetry_provider=real_telemetry.get_provider(tenant_id=qm_tenant),
    )
    try:
        yield monitor
    finally:
        await monitor.close()


@pytest.mark.integration
class TestQualityMonitorTenantOwnership:
    @pytest.mark.asyncio
    async def test_concurrent_monitors_load_only_their_tenant_rows(
        self, phoenix_monitor, real_telemetry
    ):
        import asyncio

        first = phoenix_monitor
        await _seed_golden_rows(
            first._telemetry_provider, first.tenant_id, GOLDEN_QUERIES
        )
        second_tenant = f"qm:{uuid.uuid4().hex}"
        second_provider = real_telemetry.get_provider(tenant_id=second_tenant)
        second_rows = [{"query": "tenant two only", "expected_videos": ["v_second"]}]
        await _seed_golden_rows(second_provider, second_tenant, second_rows)
        second = QualityMonitor(
            tenant_id=second_tenant,
            runtime_url="http://unused",
            phoenix_http_endpoint=first.phoenix_http_endpoint,
            llm_base_url=first.llm_base_url,
            llm_model=first.llm_model,
            golden_dataset_path="",
            telemetry_provider=second_provider,
        )
        barrier = asyncio.Barrier(2)
        counts = {first.tenant_id: 2, second_tenant: 5}

        async def load(monitor):
            await barrier.wait()
            await monitor._store_live_eval_result(
                _live_result(monitor.tenant_id, counts[monitor.tenant_id])
            )
            return await monitor._load_golden_queries_async()

        try:
            rows = await asyncio.wait_for(
                asyncio.gather(load(first), load(second)), timeout=30
            )
            assert rows == [GOLDEN_QUERIES, second_rows]
            assert first._golden_queries == GOLDEN_QUERIES
            assert second._golden_queries == second_rows
            for monitor, count in [(first, 2), (second, 5)]:
                frame = await monitor._get_dataset_store().get_dataset(
                    f"quality-live-{monitor.tenant_id}-20240101_000000"
                )
                assert frame.to_dict("records") == [
                    {
                        "input": {"agent": "search"},
                        "output": {
                            "score": "0.75",
                            "baseline_score": "0.9",
                            "degradation_pct": "0.1",
                            "sample_count": str(count),
                        },
                        "metadata": {},
                    }
                ]
        finally:
            await second.close()

    @pytest.mark.asyncio
    async def test_golden_set_store_failure_raises_with_context(
        self, phoenix_monitor, phoenix_container
    ):
        from cogniverse_telemetry_phoenix.provider import PhoenixProvider

        healthy = phoenix_monitor
        await _seed_golden_rows(
            healthy._telemetry_provider, healthy.tenant_id, GOLDEN_QUERIES
        )
        assert await healthy._load_golden_queries_async() == GOLDEN_QUERIES
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            unavailable_url = f"http://127.0.0.1:{reserved.getsockname()[1]}"
            provider = PhoenixProvider()
            provider.initialize(
                {
                    "tenant_id": healthy.tenant_id,
                    "http_endpoint": unavailable_url,
                    "grpc_endpoint": phoenix_container["grpc_endpoint"],
                }
            )
            monitor = QualityMonitor(
                tenant_id=healthy.tenant_id,
                runtime_url=healthy.runtime_url,
                phoenix_http_endpoint=unavailable_url,
                llm_base_url=healthy.llm_base_url,
                llm_model=healthy.llm_model,
                golden_dataset_path="",
                telemetry_provider=provider,
            )
            try:
                with pytest.raises(GoldenSetGroundTruthStoreUnavailableError) as caught:
                    await monitor.force_optimization_cycle()
                store_error = caught.value.__cause__
                assert type(store_error) is DatasetStoreUnavailableError
                assert store_error.endpoint == unavailable_url
                assert (
                    store_error.dataset
                    == monitor._get_artifact_manager()._blob_dataset_name(
                        GOLDEN_SET_GROUND_TRUTH_BLOB_KIND,
                        GOLDEN_SET_GROUND_TRUTH_BLOB_KEY,
                    )
                )
                assert type(store_error.__cause__) is httpx.ConnectError
                assert str(store_error.__cause__) == "[Errno 111] Connection refused"
                assert caught.value.to_result() == {
                    "status": "golden_set_store_unavailable",
                    "retryable": True,
                    "error": "golden_set_ground_truth store unavailable",
                    "cause": {
                        "type": DatasetStoreUnavailableError.__name__,
                        "message": str(store_error),
                    },
                }
                with pytest.raises(httpx.ConnectError) as write_error:
                    await monitor._store_live_eval_result(
                        _live_result(monitor.tenant_id, 1)
                    )
                assert str(write_error.value) == "[Errno 111] Connection refused"
            finally:
                await monitor.close()


@pytest.mark.integration
class TestPhoenixReachabilityProbe:
    """_probe_phoenix_reachability() surfaces silent NoOpSpan
    fallbacks at startup. Before fix #11 Phoenix being unreachable was invisible."""

    def test_probe_passes_with_real_phoenix(self, real_telemetry):
        """With a live Phoenix, the probe must complete without raising."""
        import cogniverse_foundation.telemetry.manager as tmm

        original = tmm._telemetry_manager
        tmm._telemetry_manager = real_telemetry

        from cogniverse_runtime.main import _probe_phoenix_reachability

        try:
            _probe_phoenix_reachability()
        finally:
            tmm._telemetry_manager = original

    def test_probe_warns_without_raising_when_phoenix_down(self, caplog):
        """When Phoenix is down and TELEMETRY_REQUIRED is not set, the probe
        logs a WARNING but does NOT raise."""
        import logging
        import os

        import cogniverse_foundation.telemetry.manager as tmm
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )
        from cogniverse_foundation.telemetry.manager import TelemetryManager
        from cogniverse_runtime.main import _probe_phoenix_reachability

        config = TelemetryConfig(
            otlp_endpoint="localhost:19999",
            provider_config={
                "http_endpoint": "http://localhost:19999",
                "grpc_endpoint": "http://localhost:19998",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
        unreachable_manager = TelemetryManager(config=config)

        original = tmm._telemetry_manager
        original_env = os.environ.get("TELEMETRY_REQUIRED")
        os.environ.pop("TELEMETRY_REQUIRED", None)

        try:
            tmm._telemetry_manager = unreachable_manager
            with caplog.at_level(logging.WARNING):
                _probe_phoenix_reachability()  # must not raise
        finally:
            tmm._telemetry_manager = original
            if original_env is not None:
                os.environ["TELEMETRY_REQUIRED"] = original_env

    def test_probe_raises_when_required_and_phoenix_down(self):
        """When TELEMETRY_REQUIRED=true and Phoenix is down, the probe must
        raise RuntimeError so the sidecar fails fast at startup."""
        import os

        import cogniverse_foundation.telemetry.manager as tmm
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )
        from cogniverse_runtime.main import _probe_phoenix_reachability

        _broken_cfg = TelemetryConfig(
            otlp_endpoint="localhost:19999",
            provider_config={
                "http_endpoint": "http://localhost:19999",
                "grpc_endpoint": "http://localhost:19998",
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )

        class _BrokenManager:
            config = _broken_cfg  # assigned in enclosing scope above

            def span(self, *args, **kwargs):
                raise ConnectionRefusedError("Phoenix not reachable")

        original = tmm._telemetry_manager
        original_env = os.environ.get("TELEMETRY_REQUIRED")
        os.environ["TELEMETRY_REQUIRED"] = "true"

        try:
            tmm._telemetry_manager = _BrokenManager()
            with pytest.raises(RuntimeError, match="TELEMETRY_REQUIRED=true"):
                _probe_phoenix_reachability()
        finally:
            tmm._telemetry_manager = original
            if original_env is not None:
                os.environ["TELEMETRY_REQUIRED"] = original_env
            else:
                os.environ.pop("TELEMETRY_REQUIRED", None)


@pytest.mark.integration
class TestXGBoostGateViaPhoenixProvider:
    """QualityMonitor._apply_training_decision_model is dead
    code when telemetry_provider=None (which was always the case before fix #15).
    Now quality_monitor_cli._build_phoenix_provider() injects a real provider
    so the XGBoost gate is actually reachable."""

    def test_xgboost_gate_entered_when_provider_injected(self, real_telemetry):
        """When telemetry_provider is non-None, _apply_training_decision_model
        must enter the XGBoost branch (not the early-return branch)."""
        from unittest.mock import MagicMock, patch

        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]

        from cogniverse_telemetry_phoenix.provider import PhoenixProvider

        provider = PhoenixProvider()
        provider.initialize(
            {
                "tenant_id": "xgboost_gate_test",
                "http_endpoint": phoenix_url,
                "grpc_endpoint": real_telemetry.config.provider_config["grpc_endpoint"],
            }
        )

        from tests.utils.llm_config import get_llm_base_url, get_llm_model

        monitor = QualityMonitor(
            tenant_id="xgboost_gate_test",
            runtime_url="http://localhost:99999",
            phoenix_http_endpoint=phoenix_url,
            llm_base_url=get_llm_base_url(),
            llm_model=get_llm_model(),
            golden_dataset_path="/tmp/nonexistent_golden.csv",
            telemetry_provider=provider,
        )

        assert monitor._telemetry_provider is not None, (
            "telemetry_provider was not stored on QualityMonitor. "
            "Fix #15 constructor wiring has regressed."
        )

        verdicts = {AgentType.SEARCH: Verdict.OPTIMIZE}
        golden = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="xgboost_gate_test",
            mean_mrr=0.5,
            mean_ndcg=0.4,
            mean_precision_at_5=0.4,
            query_count=5,
        )

        with patch.object(monitor, "_get_training_decision_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.should_train.return_value = (True, 0.3)
            mock_get_model.return_value = mock_model

            monitor._apply_training_decision_model(verdicts, golden, None)

        mock_get_model.assert_called_once()

    def test_xgboost_gate_skipped_when_provider_none(self):
        """With telemetry_provider=None, verdicts are returned unchanged."""
        from tests.utils.llm_config import get_llm_base_url, get_llm_model

        monitor = QualityMonitor(
            tenant_id="xgboost_skip_test",
            runtime_url="http://localhost:99999",
            phoenix_http_endpoint="http://localhost:99999",
            llm_base_url=get_llm_base_url(),
            llm_model=get_llm_model(),
            golden_dataset_path="/tmp/nonexistent_golden.csv",
            telemetry_provider=None,
        )

        verdicts = {AgentType.SEARCH: Verdict.OPTIMIZE}
        result = monitor._apply_training_decision_model(verdicts, None, None)

        assert result == verdicts


def _roundtrip_value(df, col):
    """Pull a scalar from a Phoenix get_dataset() dataframe, tolerating both
    the flat-column layout and the nested input/output-dict layout
    to_dataframe() may produce depending on the input/output key split."""
    if col in df.columns:
        return df[col].iloc[-1]
    for c in df.columns:
        cell = df[c].iloc[-1]
        if isinstance(cell, dict) and col in cell:
            return cell[col]
    raise KeyError(f"{col!r} not in {list(df.columns)} nor any nested dict cell")


@pytest.mark.integration
class TestStoreOperationsRealPhoenix:
    """_store_* methods persist their payloads into a real Phoenix dataset.

    The unit-tier equivalents mocked _dataset_store and asserted only that
    create_dataset was called — they never proved the metrics, agent rows, or
    training examples actually reached Phoenix in the right shape. These store
    against real Phoenix and read the dataset back to assert the persisted
    values exactly.
    """

    @pytest.mark.asyncio
    async def test_store_golden_persists_metrics(self, phoenix_monitor):
        m = phoenix_monitor
        result = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id=m.tenant_id,
            mean_mrr=0.75,
            mean_ndcg=0.70,
            mean_precision_at_5=0.50,
            query_count=10,
        )
        await m._store_golden_eval_result(result)

        store = m._get_dataset_store()
        df = await store.get_dataset(f"quality-baseline-{m.tenant_id}")
        assert not df.empty
        assert json.loads(_roundtrip_value(df, "payload")) == {
            "timestamp": result.timestamp.isoformat(),
            "mean_mrr": 0.75,
            "mean_ndcg": 0.7,
            "mean_precision_at_5": 0.5,
            "query_count": 10,
            "failed_query_count": 0,
        }

    @pytest.mark.asyncio
    async def test_store_live_persists_agent_rows(self, phoenix_monitor):
        m = phoenix_monitor
        ts = datetime.utcnow()
        result = LiveEvalResult(
            timestamp=ts,
            tenant_id=m.tenant_id,
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.80,
                    baseline_score=0.85,
                    degradation_pct=0.06,
                    sample_count=20,
                ),
                AgentType.SUMMARY: AgentEvalResult(
                    agent=AgentType.SUMMARY,
                    score=0.30,
                    baseline_score=0.70,
                    degradation_pct=0.57,
                    sample_count=15,
                ),
            },
        )
        await m._store_live_eval_result(result)

        name = f"quality-live-{m.tenant_id}-{ts.strftime('%Y%m%d_%H%M%S')}"
        store = m._get_dataset_store()
        df = await store.get_dataset(name)

        agents = {str(_roundtrip_value(df.iloc[[i]], "agent")) for i in range(len(df))}
        assert agents == {"search", "summary"}

        by_agent = {}
        for i in range(len(df)):
            row = df.iloc[[i]]
            by_agent[str(_roundtrip_value(row, "agent"))] = row
        assert float(_roundtrip_value(by_agent["search"], "score")) == pytest.approx(
            0.80, abs=1e-6
        )
        assert float(
            _roundtrip_value(by_agent["summary"], "degradation_pct")
        ) == pytest.approx(0.57, abs=1e-6)
        assert sorted(df.to_dict("records"), key=lambda row: row["input"]["agent"]) == [
            {
                "input": {"agent": "search"},
                "output": {
                    "score": "0.8",
                    "baseline_score": "0.85",
                    "degradation_pct": "0.06",
                    "sample_count": "20",
                },
                "metadata": {},
            },
            {
                "input": {"agent": "summary"},
                "output": {
                    "score": "0.3",
                    "baseline_score": "0.7",
                    "degradation_pct": "0.57",
                    "sample_count": "15",
                },
                "metadata": {},
            },
        ]

    @pytest.mark.asyncio
    async def test_store_trigger_persists_examples(self, phoenix_monitor):
        m = phoenix_monitor
        ts = datetime.utcnow()
        trigger = OptimizationTrigger(
            timestamp=ts,
            tenant_id=m.tenant_id,
            agents_to_optimize=[AgentType.SEARCH],
            golden_eval=None,
            live_eval=None,
            low_scoring_examples={
                AgentType.SEARCH: [
                    {"query": "weak query", "score": 0.10, "output": {"hits": 0}}
                ],
            },
            high_scoring_examples={
                AgentType.SEARCH: [
                    {"query": "strong query", "score": 0.95, "output": {"hits": 5}}
                ],
            },
            misrouted_queries=[],
        )
        name = await m._store_trigger_dataset(trigger)

        assert name == (
            f"optimization-trigger-{m.tenant_id}-{ts.strftime('%Y%m%d_%H%M%S')}"
        )
        store = m._get_dataset_store()
        df = await store.get_dataset(name)
        assert len(df) == 2

        rows = {}
        for i in range(len(df)):
            row = df.iloc[[i]]
            rows[str(_roundtrip_value(row, "category"))] = row
        assert set(rows) == {"low_scoring", "high_scoring"}
        assert str(_roundtrip_value(rows["low_scoring"], "query")) == "weak query"
        assert float(_roundtrip_value(rows["low_scoring"], "score")) == pytest.approx(
            0.10, abs=1e-6
        )
        assert str(_roundtrip_value(rows["high_scoring"], "query")) == "strong query"


def _emit_agent_span(telemetry, tenant_id, span_name, query, output_value):
    """Emit one agent span the way production agents do: to the tenant-only
    user-ops project (no project_name suffix)."""
    with telemetry.span(
        name=span_name,
        tenant_id=tenant_id,
        attributes={
            "input.value": query,
            "output.value": output_value,
        },
    ):
        pass


async def _wait_for_span(span_evaluator, span_name, deadline_s=60):
    """Poll get_recent_spans until ``span_name`` is retrievable (shape-agnostic)."""
    import asyncio

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        df = await span_evaluator.get_recent_spans(
            hours=1,
            operation_name=span_name,
            limit=200,
            require_search_shape=False,
        )
        if df is not None and not df.empty:
            return df
        await asyncio.sleep(2)
    return None


@pytest.mark.integration
class TestLiveTrafficRealPhoenix:
    """evaluate_live_traffic must score SUMMARY/REPORT/GATEWAY agents whose
    outputs are strings / routing dicts — not just SEARCH. Regression guard for
    C4: get_recent_spans previously assumed search-result shape and dropped
    every non-search span, so 3 of 4 agent types scored zero live samples.
    """

    @pytest.mark.asyncio
    async def test_get_recent_spans_keeps_non_search_summary_span(self, real_telemetry):
        """A summary-string span is retrievable with require_search_shape=False
        (its text under outputs['value']) and dropped when the search shape is
        required — the exact C4 boundary, no LLM judge involved."""
        from cogniverse_evaluation.span_evaluator import SpanEvaluator

        tenant_id = "qmrt:live-shape"
        real_telemetry.register_project(
            tenant_id=tenant_id,
            project_name=None,
            otlp_endpoint=real_telemetry.config.provider_config["grpc_endpoint"],
            http_endpoint=real_telemetry.config.provider_config["http_endpoint"],
            use_sync_export=True,
        )

        _emit_agent_span(
            real_telemetry,
            tenant_id,
            "SummarizerAgent.process",
            "summarize the quarterly results",
            "The quarter showed strong growth across all product lines.",
        )
        real_telemetry.force_flush(timeout_millis=10000)

        evaluator = SpanEvaluator(
            tenant_id=tenant_id,
            project_name=f"cogniverse-{tenant_id}",
        )

        kept = await _wait_for_span(evaluator, "SummarizerAgent.process")
        assert kept is not None and not kept.empty, (
            "summary span not retrievable with require_search_shape=False"
        )
        row = kept.iloc[0]
        assert row["operation_name"] == "SummarizerAgent.process"
        assert (
            row["outputs"]["value"]
            == "The quarter showed strong growth across all product lines."
        )

        # With the search shape required, the same span is dropped.
        dropped = await evaluator.get_recent_spans(
            hours=1,
            operation_name="SummarizerAgent.process",
            limit=200,
            require_search_shape=True,
        )
        assert dropped.empty, (
            "summary span should be dropped when search-result shape is required"
        )

    @pytest.mark.requires_lm
    @pytest.mark.local_only
    @pytest.mark.asyncio
    async def test_evaluate_live_traffic_scores_summary_agent(self, real_telemetry):
        """Full path: a summary span is fetched, scored by the real LLM judge,
        and surfaced in evaluate_live_traffic().agent_results[SUMMARY] — not
        dropped as a non-search shape. The requires_lm marker provisions the
        exact judge LM this test consumes."""
        import asyncio

        tenant_id = "qmrt:live-score"
        real_telemetry.register_project(
            tenant_id=tenant_id,
            project_name=None,
            otlp_endpoint=real_telemetry.config.provider_config["grpc_endpoint"],
            http_endpoint=real_telemetry.config.provider_config["http_endpoint"],
            use_sync_export=True,
        )

        _emit_agent_span(
            real_telemetry,
            tenant_id,
            "SummarizerAgent.process",
            "summarize the onboarding guide",
            "The onboarding guide covers account setup, first project, and support.",
        )
        real_telemetry.force_flush(timeout_millis=10000)

        from cogniverse_evaluation.span_evaluator import SpanEvaluator

        probe = SpanEvaluator(
            tenant_id=tenant_id, project_name=f"cogniverse-{tenant_id}"
        )
        assert await _wait_for_span(probe, "SummarizerAgent.process") is not None, (
            "summary span never landed in Phoenix"
        )

        monitor = QualityMonitor(
            tenant_id=tenant_id,
            runtime_url="http://localhost:99999",
            phoenix_http_endpoint=real_telemetry.config.provider_config[
                "http_endpoint"
            ],
            llm_base_url=get_llm_base_url(),
            llm_model=get_llm_model(),
            golden_dataset_path="/tmp/nonexistent_golden.csv",
        )
        try:
            result = await asyncio.wait_for(
                monitor.evaluate_live_traffic(), timeout=180
            )
        finally:
            await monitor.close()

        assert AgentType.SUMMARY in result.agent_results, (
            f"SUMMARY agent not scored; got {list(result.agent_results)}"
        )
        summary_eval = result.agent_results[AgentType.SUMMARY]
        assert summary_eval.sample_count >= 1
        assert isinstance(summary_eval.score, float)
        assert summary_eval.score >= 0.0
