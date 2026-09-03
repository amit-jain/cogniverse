"""Golden-MRR drop detection must compare against the PRIOR baseline.

Real Phoenix; the /search call is stubbed so the eval MRR is deterministic.
The drop is measured against the baseline captured before the run stored its
own result — reading the dataset back after the store would compare the run
against itself (drop always 0).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from cogniverse_evaluation.quality_monitor import (
    AgentType,
    GoldenEvalResult,
    QualityMonitor,
    Verdict,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


class _Resp:
    status_code = 200

    def __init__(self, results: list) -> None:
        self._results = results

    def json(self) -> dict:
        return {"results": self._results}


class _FakeSearchClient:
    """Returns fixed search hits so the golden eval MRR is deterministic."""

    def __init__(self, results: list) -> None:
        self._results = results

    async def post(self, url, json=None):
        return _Resp(self._results)


@pytest.fixture
def monitor(phoenix_container):
    tenant_id = f"qm_baseline_{uuid.uuid4().hex[:8]}"
    # The entrypoint injects the telemetry manager; built here explicitly so
    # the monitor's dataset access targets this test's Phoenix without a
    # config-store read (no Vespa in this stack).
    manager = TelemetryManager(
        TelemetryConfig(
            enabled=True,
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            service_name="qm-baseline-test",
            environment="test",
        )
    )
    m = QualityMonitor(
        tenant_id=tenant_id,
        runtime_url="http://unused",
        phoenix_http_endpoint=phoenix_container["http_endpoint"],
        llm_base_url="http://unused",
        llm_model="unused",
        golden_dataset_path="/unused.json",
        telemetry_provider=manager.get_provider(tenant_id=tenant_id),
    )
    # Pre-populate the loader's cache slot; evaluate_golden_set reads it first.
    m._golden_queries = [{"query": "q", "expected_videos": ["vidA"]}]
    # vidA at rank 2 -> reciprocal rank 0.5
    m._http_client = _FakeSearchClient([{"source_id": "other"}, {"source_id": "vidA"}])
    # This file pins the baseline-capture contract; the XGBoost timing gate
    # (its own policy, exercised by test_quality_monitor) would override the
    # threshold verdict off this stack's empty training history.
    m._apply_training_decision_model = lambda verdicts, golden, live: verdicts
    return m


@pytest.mark.asyncio
async def test_drop_measured_against_prior_baseline(monitor):
    # Seed a high prior baseline (0.9).
    await monitor._store_golden_eval_result(
        GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id=monitor.tenant_id,
            mean_mrr=0.9,
            mean_ndcg=0.9,
            mean_precision_at_5=0.8,
            query_count=10,
        )
    )

    # A fresh eval scores 0.5 and must capture 0.9 as its baseline — not read
    # back the 0.5 it stores during this same call.
    result = await monitor.evaluate_golden_set()
    assert result.mean_mrr == pytest.approx(0.5, abs=0.01)
    assert result.baseline_mrr == pytest.approx(0.9, abs=0.01)

    verdicts = monitor.check_thresholds(result, None)
    assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE


@pytest.mark.asyncio
async def test_no_prior_baseline_skips(monitor):
    # No baseline stored yet -> nothing to compare against -> SKIP.
    result = await monitor.evaluate_golden_set()
    assert result.baseline_mrr is None

    verdicts = monitor.check_thresholds(result, None)
    assert verdicts[AgentType.SEARCH] == Verdict.SKIP
