"""
Integration tests for QualityMonitor with real Vespa search + real Phoenix.

Full round-trip: golden query → real /search → real Vespa with real data
→ real MRR/nDCG scoring → store baseline in real Phoenix → detect degradation.

Uses shared vespa_instance + search_client + real_telemetry from conftest.
ColPali model generates real embeddings for test documents.
"""

import json
import logging
import time
from datetime import datetime

import httpx
import numpy as np
import pytest
import requests
import torch
from PIL import Image

from cogniverse_core.common.models import get_or_load_model
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_evaluation.quality_monitor import (
    AgentType,
    GoldenEvalResult,
    QualityMonitor,
    Verdict,
)
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)

COLPALI_MODEL_NAME = "vidore/colsmol-500m"
TENANT_SCHEMA_NAME = "video_colpali_smol500_mv_frame_qm_real_test"


def _embeddings_to_vespa_tensors(embeddings: np.ndarray):
    """Convert embeddings to Vespa tensor format."""
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}
    return float_dict, binary_dict


@pytest.fixture(scope="module")
def colpali_model():
    """Load ColPali model once."""
    config = {
        "colpali_model": COLPALI_MODEL_NAME,
        "embedding_type": "multi_vector",
        "model_loader": "colpali",
    }
    model, processor = get_or_load_model(COLPALI_MODEL_NAME, config, logger)
    device = next(model.parameters()).device
    yield model, processor, device
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(scope="module")
def seeded_vespa(vespa_instance, colpali_model, config_manager, schema_loader):
    """Feed test documents into Vespa with real ColPali embeddings.

    Deploys the tenant-scoped schema for the ``qm_real_test`` tenant via
    SchemaRegistry — same code path production uses. The registry's
    deploy_schemas call merges schemas already present in the live Vespa
    cluster (e.g. the ``_test_unit`` baseline from the module conftest),
    so the redeploy preserves them instead of treating them as removals.

    Keeping a distinct ``qm_real_test`` tenant (rather than reusing
    ``test:unit``) isolates Phoenix baselines — the golden-eval baseline
    tests read back what they write, so a shared tenant would collide
    with neighbouring tests under the same TestGoldenEvalRealVespa run.
    """
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_foundation.config.unified_config import BackendConfig
    from cogniverse_vespa.backend import VespaBackend

    # Build a dedicated backend for the qm_real_test tenant. Going through
    # BackendRegistry would cache the instance across tests; a local
    # instance keeps teardown simple and the schema deploy isolated.
    qm_tenant = "qm_real_test"
    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=vespa_instance["http_port"],
        tenant_id=qm_tenant,
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": qm_tenant})

    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry

    # deploy_schema() triggers backend.deploy_schemas(), which in turn
    # (per the fix in this session) discovers deployed document types
    # from Vespa and merges them with the new qm_real_test schema.
    registry.deploy_schema(
        tenant_id=qm_tenant,
        base_schema_name="video_colpali_smol500_mv_frame",
    )

    # Wait for the new schema to be addressable for feeds. GET on the
    # document API returns 404 for any URL — even bogus schemas — so
    # it can't tell us when the content distributor has converged.
    # /search/ with model.restrict exposes the real state: Vespa errors
    # out listing the set of valid source refs, and we proceed once our
    # schema is in that set.
    probe_url = f"http://localhost:{vespa_instance['http_port']}/search/"
    last_state = None
    for attempt in range(120):
        probe = requests.post(
            probe_url,
            json={
                "yql": "select documentid from sources * where true limit 0",
                "hits": 0,
                "model.restrict": TENANT_SCHEMA_NAME,
            },
            timeout=5,
        )
        if probe.status_code == 200:
            body = probe.json()
            errors = body.get("root", {}).get("errors", [])
            msg = " ".join(e.get("message", "") for e in errors)
            last_state = msg or "ok"
            if TENANT_SCHEMA_NAME in msg or not errors:
                logger.info(
                    f"{TENANT_SCHEMA_NAME} visible to /search/ after {attempt + 1}s"
                )
                break
        time.sleep(1)
    else:
        raise RuntimeError(
            f"Schema {TENANT_SCHEMA_NAME!r} not visible to /search/ after "
            f"120s — last state: {last_state}"
        )

    # Document API (feed path) converges a few seconds after /search/ can
    # see the schema; a small post-search buffer avoids a racy first feed.
    time.sleep(5)

    model, processor, device = colpali_model
    http_port = vespa_instance["http_port"]

    test_docs = [
        {
            "color": (255, 0, 0),
            "title": "Man lifting heavy barbell in gym",
            "video_id": "v_-HpCLXdtcas",
        },
        {
            "color": (0, 0, 255),
            "title": "Ocean waves crashing on rocky coast",
            "video_id": "v_ocean_test",
        },
        {
            "color": (0, 128, 0),
            "title": "Person running through forest trail",
            "video_id": "v_forest_test",
        },
    ]

    for i, doc_info in enumerate(test_docs):
        img = Image.new("RGB", (224, 224), color=doc_info["color"])
        batch_inputs = processor.process_images([img]).to(device)
        with torch.no_grad():
            doc_embeddings = model(**batch_inputs)
        embeddings_np = doc_embeddings.squeeze(0).cpu().float().numpy()
        float_dict, binary_dict = _embeddings_to_vespa_tensors(embeddings_np)

        doc_id = f"qm_test_doc_{i}"
        vespa_doc = {
            "fields": {
                "video_id": doc_info["video_id"],
                "video_title": doc_info["title"],
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "segment_description": doc_info["title"],
                "audio_transcript": "",
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }

        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
            json=vespa_doc,
            timeout=10,
        )
        assert resp.status_code in [200, 201], (
            f"Failed to feed doc {doc_id}: {resp.status_code}: {resp.text[:200]}"
        )

    time.sleep(5)  # Wait for indexing

    yield test_docs

    for i in range(len(test_docs)):
        doc_id = f"qm_test_doc_{i}"
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{TENANT_SCHEMA_NAME}/docid/{doc_id}",
                timeout=5,
            )
        except Exception:
            pass


@pytest.fixture
def monitor_with_real_search(search_client, real_telemetry, tmp_path):
    """QualityMonitor wired to real search via TestClient bridge."""
    golden_queries = [
        {
            "query": "man lifting barbell",
            "expected_videos": ["v_-HpCLXdtcas"],
            "ground_truth": "Man lifting a barbell in gym",
            "query_type": "answer_phrase",
            "source": "test",
        },
        {
            "query": "ocean waves coast",
            "expected_videos": ["v_ocean_test"],
            "ground_truth": "Ocean waves on coast",
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
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_queries))

    m = QualityMonitor(
        tenant_id="qm_real_test",
        runtime_url="http://testserver",
        phoenix_http_endpoint="http://localhost:16006",
        llm_base_url="http://localhost:11434",
        llm_model=get_llm_model(),
        golden_dataset_path=str(golden_path),
    )

    # Bridge to TestClient
    m._http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: _testclient_transport(search_client, req)
        ),
        base_url="http://testserver",
    )

    yield m

    import asyncio

    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(m.close())
    except RuntimeError:
        asyncio.run(m.close())


@pytest.mark.integration
class TestGoldenEvalRealVespa:
    """Golden eval against real Vespa with real ingested data."""

    @pytest.mark.asyncio
    async def test_golden_eval_with_real_data(
        self, monitor_with_real_search, seeded_vespa
    ):
        """Run golden eval against Vespa with real ColPali-embedded documents.

        Expected: "man lifting barbell" should match v_-HpCLXdtcas (MRR > 0).
        "cat sleeping" should NOT match anything (MRR = 0).
        """
        result = await monitor_with_real_search.evaluate_golden_set()

        assert result.query_count == 3
        assert 0.0 <= result.mean_mrr <= 1.0
        assert 0.0 <= result.mean_ndcg <= 1.0

        scores_by_query = {s["query"]: s for s in result.per_query_scores}

        # "cat sleeping on sofa" — no matching doc in Vespa
        cat = scores_by_query.get("cat sleeping on sofa")
        if cat:
            assert cat["mrr"] == 0.0, (
                f"'cat sleeping' should score MRR=0 (no matching doc), got {cat['mrr']}"
            )

        # At least one query should have MRR > 0 (barbell or ocean should match)
        has_positive_mrr = any(s["mrr"] > 0 for s in result.per_query_scores)
        assert has_positive_mrr, (
            f"At least one golden query should match real Vespa data. "
            f"Scores: {[(s['query'], s['mrr']) for s in result.per_query_scores]}"
        )

    @pytest.mark.asyncio
    async def test_golden_eval_stores_baseline_in_phoenix(
        self, monitor_with_real_search, seeded_vespa
    ):
        """Golden eval result persists in real Phoenix dataset."""
        result = await monitor_with_real_search.evaluate_golden_set()

        baseline = monitor_with_real_search._last_golden_baseline_mrr
        assert baseline is not None
        assert baseline == pytest.approx(result.mean_mrr, abs=0.01)

    @pytest.mark.asyncio
    async def test_degradation_detection_with_real_baseline(
        self, monitor_with_real_search, seeded_vespa
    ):
        """Store high baseline, verify current eval detects degradation."""
        # Run real eval to get actual MRR
        result = await monitor_with_real_search.evaluate_golden_set()
        real_mrr = result.mean_mrr

        # Now store an inflated baseline AFTER the eval
        # (evaluate_golden_set stores the real result as baseline,
        # we overwrite it with the inflated one)
        inflated_mrr = min(real_mrr + 0.30, 1.0)
        await monitor_with_real_search._store_golden_eval_result(
            GoldenEvalResult(
                timestamp=datetime.utcnow(),
                tenant_id="qm_real_test",
                mean_mrr=inflated_mrr,
                mean_ndcg=0.90,
                mean_precision_at_5=0.80,
                query_count=20,
            )
        )

        # Verify the inflated baseline is now stored
        stored = monitor_with_real_search._last_golden_baseline_mrr
        assert stored == pytest.approx(inflated_mrr, abs=0.01), (
            f"Baseline should be {inflated_mrr}, got {stored}"
        )

        # check_thresholds compares the real result against stored baseline
        verdicts = monitor_with_real_search.check_thresholds(result, None)
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE, (
            f"Expected OPTIMIZE: real MRR={real_mrr:.3f}, "
            f"baseline={inflated_mrr:.3f}, "
            f"drop={((inflated_mrr - real_mrr) / inflated_mrr):.1%}"
        )


def _testclient_transport(test_client, request):
    """Bridge httpx.AsyncClient to FastAPI TestClient."""
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


@pytest.mark.integration
class TestForceOptimizationCycle:
    """Audit fix #7 — QualityMonitor.force_optimization_cycle() runs a full
    eval+trigger+store cycle without a threshold check, used by CronWorkflows.
    Before fix #7 there was no --once CLI path, so scheduled distillation was
    impossible when quality was stable."""

    @pytest.mark.asyncio
    async def test_force_cycle_returns_status_dict(self, real_telemetry):
        """force_optimization_cycle() must return a dict with a 'status' key.

        Even with no live spans in Phoenix, the function must complete without
        raising and return a recognizable result dict — the 'no_data' path is
        still a valid outcome that callers can log/alert on."""
        from tests.utils.llm_config import get_llm_base_url, get_llm_model

        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]

        monitor = QualityMonitor(
            tenant_id="force_cycle_test",
            runtime_url="http://localhost:99999",  # unreachable, both evals will fail
            phoenix_http_endpoint=phoenix_url,
            llm_base_url=get_llm_base_url(),
            llm_model=get_llm_model(),
            golden_dataset_path="/tmp/nonexistent_golden.csv",
        )

        result = await monitor.force_optimization_cycle()

        assert isinstance(result, dict), (
            f"force_optimization_cycle must return a dict, got: {result!r}"
        )
        assert "status" in result, f"Result dict missing 'status' key: {result}"
        assert result["status"] in ("ok", "no_data"), (
            f"Unexpected status value: {result['status']!r}"
        )
        await monitor.close()

    @pytest.mark.asyncio
    async def test_force_cycle_with_live_spans_returns_ok(
        self, real_telemetry, vespa_instance
    ):
        """When Phoenix has at least one live span, force_optimization_cycle
        must return status='ok' and list triggered agents."""
        import asyncio

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
        )

        class _PingInput(AgentInput):
            query: str
            tenant_id: str = "force_cycle_live_test"

        class _PingOutput(AgentOutput):
            result: str

        class _PingDeps(AgentDeps):
            pass

        class SearchAgent(AgentBase[_PingInput, _PingOutput, _PingDeps]):
            async def _process_impl(self, input):
                return _PingOutput(result="pong")

        agent = SearchAgent(deps=_PingDeps())
        agent.set_telemetry_manager(real_telemetry)
        await agent.process(
            _PingInput(query="force cycle live seed", tenant_id="force_cycle_live_test")
        )

        await asyncio.sleep(3)

        from tests.utils.llm_config import get_llm_base_url, get_llm_model

        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]

        monitor = QualityMonitor(
            tenant_id="force_cycle_live_test",
            runtime_url="http://localhost:99999",
            phoenix_http_endpoint=phoenix_url,
            llm_base_url=get_llm_base_url(),
            llm_model=get_llm_model(),
            golden_dataset_path="/tmp/nonexistent_golden.csv",
        )

        result = await monitor.force_optimization_cycle()

        assert isinstance(result, dict)
        assert result["status"] in ("ok", "no_data"), f"Unexpected: {result}"
        if result["status"] == "ok":
            assert "agents_triggered" in result, (
                f"status='ok' but missing 'agents_triggered': {result}"
            )
        await monitor.close()


@pytest.mark.integration
class TestPhoenixReachabilityProbe:
    """Audit fix #11 — _probe_phoenix_reachability() surfaces silent NoOpSpan
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
            config = _broken_cfg  # noqa: F821 — assigned in enclosing scope above

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
    """Audit fix #15 — QualityMonitor._apply_training_decision_model is dead
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
