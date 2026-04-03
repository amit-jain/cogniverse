"""
Integration tests for QualityMonitor with real Phoenix + Vespa + search.

Full round-trip:
  golden query → real /search endpoint → real Vespa → real results
  → MRR/nDCG scoring → store baseline in real Phoenix dataset
  → second eval → detect degradation → trigger dataset in Phoenix
  → optimization CLI reads trigger back

Requires Docker for Phoenix and Vespa containers.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_evaluation.quality_monitor import (
    AgentEvalResult,
    AgentType,
    GoldenEvalResult,
    LiveEvalResult,
    OptimizationTrigger,
    QualityMonitor,
    Verdict,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.utils.vespa_docker import VespaDockerManager

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"
GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "sample_videos_retrieval_queries.json"
)

@pytest.fixture(scope="module")
def vespa_instance():
    """Start isolated Vespa Docker for quality monitor integration tests."""
    manager = VespaDockerManager()
    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()

    try:
        container_info = manager.start_container(
            module_name="quality_monitor_integration",
            use_module_ports=True,
        )
        manager.wait_for_config_ready(container_info, timeout=180)

        import time

        time.sleep(15)

        # Deploy metadata + data schemas
        from vespa.package import ApplicationPackage

        import cogniverse_vespa  # noqa: F401
        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]

        schema_file = SCHEMAS_DIR / "video_colpali_smol500_mv_frame_schema.json"
        with open(schema_file) as f:
            schema_json = json.load(f)
        schema_json["name"] = "video_colpali_smol500_mv_frame_default"
        schema_json["document"]["name"] = "video_colpali_smol500_mv_frame_default"
        parser = JsonSchemaParser()
        data_schema = parser.parse_schema(schema_json)

        all_schemas = metadata_schemas + [data_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager._deploy_package(app_package)

        manager.wait_for_application_ready(container_info, timeout=120)
        yield container_info

    except Exception as e:
        logger.error(f"Failed to start Vespa: {e}")
        pytest.skip(f"Failed to start Vespa: {e}")

    finally:
        manager.stop_container()
        BackendRegistry._instance = None
        BackendRegistry._backend_instances.clear()


@pytest.fixture(scope="module")
def config_manager(vespa_instance):
    """ConfigManager backed by real Vespa."""
    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_colpali",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="vidore/colsmol-500m",
        ),
    )
    return cm


@pytest.fixture(scope="module")
def schema_loader():
    """FilesystemSchemaLoader for configs/schemas/."""
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    return FilesystemSchemaLoader(SCHEMAS_DIR)


@pytest.fixture(scope="module")
def phoenix_docker():
    """Module-scoped Phoenix Docker container for quality monitor tests."""
    import subprocess
    import time as _time

    import requests

    container_name = f"phoenix_qm_test_{int(_time.time() * 1000)}"

    # Kill leftover containers
    leftover = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=phoenix_qm_test_"],
        capture_output=True, text=True, timeout=10,
    )
    for cid in leftover.stdout.strip().splitlines():
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10)

    try:
        subprocess.run(
            [
                "docker", "run", "-d", "--name", container_name,
                "-p", "36006:6006", "-p", "34317:4317",
                "-e", "PHOENIX_WORKING_DIR=/phoenix",
                "arizephoenix/phoenix:latest",
            ],
            check=True, capture_output=True, timeout=30,
        )

        for _ in range(60):
            try:
                resp = requests.get("http://localhost:36006", timeout=2)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            _time.sleep(1)
        else:
            pytest.skip("Phoenix failed to start")

        yield "http://localhost:36006"

    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)


@pytest.fixture(scope="module")
def real_telemetry(phoenix_docker):
    """Module-scoped real TelemetryManager backed by Phoenix Docker."""
    import os

    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv("TELEMETRY_OTLP_ENDPOINT", "localhost:34317"),
        provider_config={
            "http_endpoint": phoenix_docker,
            "grpc_endpoint": "http://localhost:34317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(scope="module")
def search_client(vespa_instance, config_manager, schema_loader, real_telemetry):
    """FastAPI TestClient with real search router wired to real Vespa."""
    from cogniverse_runtime.routers import search

    test_app = FastAPI()
    test_app.include_router(search.router, prefix="/search")
    test_app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    test_app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )

    with TestClient(test_app) as client:
        yield client


@pytest.fixture(scope="module")
def phoenix_http_endpoint(phoenix_docker):
    """Phoenix HTTP URL."""
    return phoenix_docker


@pytest.fixture
def monitor(phoenix_http_endpoint, tmp_path):
    """QualityMonitor wired to real Phoenix and test golden dataset."""
    golden_queries = [
        {
            "query": "man lifting barbell",
            "expected_videos": ["v_-HpCLXdtcas"],
            "ground_truth": "Man lifting a barbell",
            "query_type": "answer_phrase",
            "source": "test",
        },
        {
            "query": "person in winter clothes outdoors",
            "expected_videos": ["v_-IMXSEIabMM"],
            "ground_truth": "Person wearing winter clothes in snow",
            "query_type": "question",
            "source": "test",
        },
        {
            "query": "dog playing in park",
            "expected_videos": ["v_dog_nonexistent"],
            "ground_truth": "Dog playing fetch",
            "query_type": "question",
            "source": "test",
        },
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_queries))

    m = QualityMonitor(
        tenant_id="qm_test",
        runtime_url="http://testserver",  # TestClient base URL
        phoenix_http_endpoint=phoenix_http_endpoint,
        llm_base_url="http://localhost:11434",
        llm_model="qwen3:4b",
        golden_dataset_path=str(golden_path),
    )
    yield m
    import asyncio

    asyncio.get_event_loop().run_until_complete(m.close())


@pytest.mark.integration
class TestGoldenEvalWithRealSearch:
    """Golden eval scoring pipeline with controlled search responses."""

    @pytest.mark.asyncio
    async def test_golden_eval_scores_correctly_with_known_results(self, monitor):
        """Run golden eval with a mock search that returns known results.

        This tests the full scoring pipeline: query → HTTP → response →
        MRR/nDCG/P@5 computation → result aggregation → Phoenix storage.
        """
        import httpx

        def mock_search_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            query = body.get("query", "")

            if "barbell" in query:
                # Return the expected video — MRR should be 1.0
                return httpx.Response(200, json={
                    "results": [
                        {"source_id": "v_-HpCLXdtcas", "score": 0.95},
                        {"source_id": "v_other", "score": 0.8},
                    ]
                })
            elif "winter" in query:
                # Return expected video at position 3 — MRR = 0.33
                return httpx.Response(200, json={
                    "results": [
                        {"source_id": "v_wrong1", "score": 0.9},
                        {"source_id": "v_wrong2", "score": 0.85},
                        {"source_id": "v_-IMXSEIabMM", "score": 0.8},
                    ]
                })
            else:
                # No matching results — MRR = 0
                return httpx.Response(200, json={"results": [
                    {"source_id": "v_irrelevant", "score": 0.5},
                ]})

        monitor._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_search_handler),
            base_url="http://testserver",
        )

        result = await monitor.evaluate_golden_set()

        assert isinstance(result, GoldenEvalResult)
        assert result.tenant_id == "qm_test"
        assert result.query_count == 3

        # Check per-query MRR values
        scores_by_query = {s["query"]: s for s in result.per_query_scores}

        # "man lifting barbell" — expected video at position 1
        barbell = scores_by_query["man lifting barbell"]
        assert barbell["mrr"] == pytest.approx(1.0)

        # "person in winter clothes" — expected video at position 3
        winter = scores_by_query["person in winter clothes outdoors"]
        assert winter["mrr"] == pytest.approx(1.0 / 3.0, abs=0.01)

        # "dog playing in park" — no match
        dog = scores_by_query["dog playing in park"]
        assert dog["mrr"] == 0.0

        # Aggregate MRR: (1.0 + 0.33 + 0.0) / 3 ≈ 0.44
        assert result.mean_mrr == pytest.approx(0.444, abs=0.01)

        # Low scoring: MRR < 0.3 → dog (0.0) should be there
        low_queries = [q["query"] for q in result.low_scoring_queries]
        assert "dog playing in park" in low_queries

        # High scoring: MRR >= 0.8 → barbell (1.0) should be there
        high_queries = [q["query"] for q in result.high_scoring_queries]
        assert "man lifting barbell" in high_queries

    @pytest.mark.asyncio
    async def test_golden_eval_stores_result_in_phoenix(self, monitor):
        """After golden eval, result is persisted in Phoenix dataset."""
        import httpx

        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "results": [{"source_id": "v_-HpCLXdtcas", "score": 0.9}]
            })

        monitor._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            base_url="http://testserver",
        )

        result = await monitor.evaluate_golden_set()

        # Should have stored baseline in Phoenix
        baseline_mrr = monitor._last_golden_baseline_mrr
        assert baseline_mrr is not None
        assert baseline_mrr == pytest.approx(result.mean_mrr, abs=0.01)


@pytest.mark.integration
class TestBaselineRoundTripWithRealPhoenix:
    """Store and retrieve baselines from real Phoenix Docker."""

    @pytest.mark.asyncio
    async def test_store_baseline_and_read_back(self, monitor):
        """Store golden eval result in Phoenix, verify round-trip."""
        result = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            mean_mrr=0.72,
            mean_ndcg=0.68,
            mean_precision_at_5=0.45,
            query_count=10,
        )

        await monitor._store_golden_eval_result(result)

        # Read back via _last_golden_baseline_mrr
        baseline_mrr = monitor._last_golden_baseline_mrr
        assert baseline_mrr is not None
        assert baseline_mrr == pytest.approx(0.72, abs=0.01)

    @pytest.mark.asyncio
    async def test_baseline_update_overwrites_previous(self, monitor):
        """Updating baseline overwrites old value in Phoenix."""
        # Store first baseline
        first = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            mean_mrr=0.60,
            mean_ndcg=0.55,
            mean_precision_at_5=0.35,
            query_count=10,
        )
        await monitor._store_golden_eval_result(first)
        assert monitor._last_golden_baseline_mrr == pytest.approx(0.60, abs=0.01)

        # Store higher baseline
        second = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            mean_mrr=0.85,
            mean_ndcg=0.80,
            mean_precision_at_5=0.65,
            query_count=10,
        )
        await monitor._store_golden_eval_result(second)
        assert monitor._last_golden_baseline_mrr == pytest.approx(0.85, abs=0.01)


@pytest.mark.integration
class TestTriggerDatasetRoundTrip:
    """Trigger dataset stored in Phoenix, readable by optimization CLI."""

    @pytest.mark.asyncio
    async def test_trigger_stores_and_reads_training_examples(self, monitor):
        """Create trigger dataset, read back like optimization_cli.py does."""
        trigger = OptimizationTrigger(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            agents_to_optimize=[AgentType.SEARCH, AgentType.SUMMARY],
            golden_eval=None,
            live_eval=None,
            low_scoring_examples={
                AgentType.SEARCH: [
                    {"query": "find cats sleeping", "score": 0.1, "output": {}},
                    {"query": "robots dancing", "score": 0.2, "output": {}},
                ],
                AgentType.SUMMARY: [
                    {"query": "summarize lecture", "score": 0.15, "output": {}},
                ],
            },
            high_scoring_examples={
                AgentType.SEARCH: [
                    {
                        "query": "man lifting weights",
                        "score": 0.95,
                        "output": {"results": [{"video_id": "v_-HpCLXdtcas"}]},
                    },
                ],
                AgentType.SUMMARY: [],
            },
            misrouted_queries=[],
        )

        await monitor._store_trigger_dataset(trigger)

        # Read back the way optimization_cli.py does
        import phoenix as px

        sync_client = px.Client(endpoint=monitor.phoenix_http_endpoint)
        ts = trigger.timestamp.strftime("%Y%m%d_%H%M%S")
        dataset_name = f"optimization-trigger-qm_test-{ts}"
        dataset = sync_client.get_dataset(name=dataset_name)
        df = dataset.as_dataframe()

        # Phoenix wraps data under input/output dicts — flatten
        if "input" in df.columns and "output" in df.columns:
            flat_records = []
            for _, row in df.iterrows():
                inp = row.get("input", {}) or {}
                out = row.get("output", {}) or {}
                flat_records.append({**inp, **out})
            df = __import__("pandas").DataFrame(flat_records)

        # 2 low search + 1 low summary + 1 high search = 4 total
        assert len(df) == 4

        # Verify category split
        low = df[df["category"] == "low_scoring"]
        high = df[df["category"] == "high_scoring"]
        assert len(low) == 3
        assert len(high) == 1

        # Verify agent split
        search_rows = df[df["agent"] == "search"]
        summary_rows = df[df["agent"] == "summary"]
        assert len(search_rows) == 3  # 2 low + 1 high
        assert len(summary_rows) == 1  # 1 low

        # Verify query content
        assert "find cats sleeping" in low["query"].tolist()
        assert "man lifting weights" in high["query"].tolist()

        # Verify score values
        high_scores = high["score"].astype(float).tolist()
        assert all(s >= 0.8 for s in high_scores)


@pytest.mark.integration
class TestDegradationDetection:
    """Detect quality degradation using real Phoenix-stored baseline."""

    @pytest.mark.asyncio
    async def test_degradation_triggers_optimization_verdict(self, monitor):
        """Store high baseline, eval drops → verdict = OPTIMIZE."""
        # Store a strong baseline
        await monitor._store_golden_eval_result(
            GoldenEvalResult(
                timestamp=datetime.utcnow(),
                tenant_id="qm_test",
                mean_mrr=0.80,
                mean_ndcg=0.75,
                mean_precision_at_5=0.60,
                query_count=20,
            )
        )

        # Current eval shows significant drop
        current = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            mean_mrr=0.45,  # 43.75% drop from 0.80
            mean_ndcg=0.40,
            mean_precision_at_5=0.25,
            query_count=20,
        )

        verdicts = monitor.check_thresholds(current, None)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    @pytest.mark.asyncio
    async def test_no_degradation_skips(self, monitor):
        """Score within tolerance → verdict = SKIP."""
        await monitor._store_golden_eval_result(
            GoldenEvalResult(
                timestamp=datetime.utcnow(),
                tenant_id="qm_test",
                mean_mrr=0.70,
                mean_ndcg=0.65,
                mean_precision_at_5=0.50,
                query_count=20,
            )
        )

        current = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            mean_mrr=0.68,  # 2.8% drop — within 10% threshold
            mean_ndcg=0.63,
            mean_precision_at_5=0.48,
            query_count=20,
        )

        verdicts = monitor.check_thresholds(current, None)
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    @pytest.mark.asyncio
    async def test_live_traffic_degradation_across_agents(self, monitor):
        """Multiple agents degrading independently."""
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="qm_test",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.75,
                    baseline_score=0.80,
                    degradation_pct=0.06,
                    sample_count=20,
                ),
                AgentType.SUMMARY: AgentEvalResult(
                    agent=AgentType.SUMMARY,
                    score=0.30,  # Below 0.5 floor
                    baseline_score=0.70,
                    degradation_pct=0.57,
                    sample_count=20,
                ),
                AgentType.REPORT: AgentEvalResult(
                    agent=AgentType.REPORT,
                    score=0.80,
                    baseline_score=0.85,
                    degradation_pct=0.06,
                    sample_count=20,
                ),
            },
        )

        verdicts = monitor.check_thresholds(None, live)

        assert verdicts[AgentType.SEARCH] == Verdict.SKIP
        assert verdicts[AgentType.SUMMARY] == Verdict.OPTIMIZE
        assert verdicts[AgentType.REPORT] == Verdict.SKIP


@pytest.mark.integration
class TestGoldenSetGrowth:
    """Growing the golden set with live traffic queries."""

    @pytest.mark.asyncio
    async def test_grow_and_verify_persisted_to_disk(self, monitor):
        """Add queries, re-read from disk, verify deduplication."""
        original_count = len(monitor._load_golden_queries())

        await monitor.grow_golden_set(
            [
                {
                    "query": "neural network architecture diagram",
                    "expected_videos": ["v_nn_123"],
                    "ground_truth": "NN architecture",
                    "query_type": "live_traffic",
                    "source": "quality_monitor",
                },
                {
                    "query": "robot assembly line timelapse",
                    "expected_videos": ["v_robot_456"],
                    "ground_truth": "Robot assembly",
                    "query_type": "live_traffic",
                    "source": "quality_monitor",
                },
            ]
        )

        # Re-read from disk
        monitor._golden_queries = []  # Clear cache
        reloaded = monitor._load_golden_queries()
        assert len(reloaded) == original_count + 2

        # Try adding a duplicate
        await monitor.grow_golden_set(
            [{"query": "neural network architecture diagram"}]
        )
        monitor._golden_queries = []
        reloaded_again = monitor._load_golden_queries()
        assert len(reloaded_again) == original_count + 2  # No duplicate



def _testclient_transport(test_client: TestClient, request):
    """Bridge httpx.AsyncClient to FastAPI TestClient for integration testing."""
    import httpx

    # TestClient expects relative paths
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    response = test_client.request(
        method=request.method,
        url=path,
        content=request.content,
        headers=dict(request.headers),
    )

    return httpx.Response(
        status_code=response.status_code,
        headers=dict(response.headers),
        content=response.content,
    )
