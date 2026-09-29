"""QualityMonitor baselines read back from a real Phoenix after repeated writes.

Golden and live baselines share one dataset that each write appends to, and
readers take the last matching row. A live write must not erase the golden
row, and a live score that returns to an earlier value must be the one read.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from cogniverse_evaluation.quality_monitor import (
    AgentType,
    GoldenEvalResult,
    QualityMonitor,
)
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

_UNREACHABLE = "http://127.0.0.1:1"


@pytest.fixture
def monitor(phoenix_container) -> QualityMonitor:
    tenant_id = f"acme:baseline-{uuid.uuid4().hex[:8]}"
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        }
    )
    return QualityMonitor(
        tenant_id=tenant_id,
        runtime_url=_UNREACHABLE,
        phoenix_http_endpoint=phoenix_container["http_endpoint"],
        llm_base_url=_UNREACHABLE,
        llm_model="unused",
        golden_dataset_path="unused-by-baseline-reads",
        telemetry_provider=provider,
    )


@pytest.mark.asyncio
async def test_live_baseline_writes_keep_the_golden_baseline(monitor):
    golden = GoldenEvalResult(
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        tenant_id=monitor.tenant_id,
        mean_mrr=0.75,
        mean_ndcg=0.7,
        mean_precision_at_5=0.5,
        query_count=10,
    )

    await monitor.update_baseline(
        golden_result=golden, live_results={AgentType.SEARCH: 0.8}
    )
    await monitor.update_baseline(live_results={AgentType.SUMMARY: 0.6})

    assert await monitor._read_baseline_metric("mean_mrr") == 0.75
    assert await monitor._read_baseline_metric("mean_ndcg") == 0.7
    assert await monitor._get_agent_baseline(AgentType.SEARCH) == 0.8
    assert await monitor._get_agent_baseline(AgentType.SUMMARY) == 0.6


@pytest.mark.asyncio
async def test_live_baseline_returning_to_an_earlier_score_is_read(monitor):
    for score in (0.8, 0.7, 0.8):
        await monitor.update_baseline(live_results={AgentType.SEARCH: score})

    assert await monitor._get_agent_baseline(AgentType.SEARCH) == 0.8
    frame = await monitor._get_dataset_store().get_dataset(
        f"quality-baseline-{monitor.tenant_id}"
    )
    assert [row["input"]["payload"] for _, row in frame.iterrows()] == [
        '{"agent": "search", "score": 0.8}',
        '{"agent": "search", "score": 0.7}',
        '{"agent": "search", "score": 0.8}',
    ]
