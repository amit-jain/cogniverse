"""Real-Phoenix round-trip for the routing tab's success computation.

Phoenix reports span outcome in a ``status_code`` column (OK / UNSET / ERROR),
never a ``status`` column. The old ``span_row.get('status') == 'OK'`` therefore
scored every healthy routing span (which comes back UNSET) as a failure, so the
success charts read 0%. ``_span_success`` treats anything that is not ERROR as
success.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from opentelemetry.trace import Status, StatusCode

from cogniverse_dashboard.tabs.routing_evaluation import _span_success
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingEvaluator

pytestmark = pytest.mark.integration


@pytest.fixture
def telemetry_manager(phoenix_container):
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    yield manager
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_routing_success_reads_status_code(phoenix_container, telemetry_manager):
    tenant = f"routing-success-{uuid4().hex[:8]}"
    project_name = "routing"
    full_project = f"cogniverse-{tenant}-{project_name}"

    telemetry_manager.register_project(
        tenant_id=tenant,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )

    # Healthy span: clean exit -> Phoenix records status_code UNSET.
    with telemetry_manager.span(
        name="cogniverse.routing",
        tenant_id=tenant,
        project_name=project_name,
        attributes={"routing.chosen_agent": "video_search", "routing.confidence": 0.83},
    ):
        pass
    # Failed span: explicit ERROR status.
    with telemetry_manager.span(
        name="cogniverse.routing",
        tenant_id=tenant,
        project_name=project_name,
        attributes={"routing.chosen_agent": "video_search", "routing.confidence": 0.4},
    ) as span:
        span.set_status(Status(StatusCode.ERROR, "boom"))
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant, project_name=project_name
    )
    evaluator = RoutingEvaluator(provider=provider, project_name=full_project)

    rows = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        rows = await evaluator.query_routing_spans(
            start_time=end - timedelta(hours=1), end_time=end, limit=50
        )
        if len(rows) >= 2:
            break
        await asyncio.sleep(2)

    assert len(rows) == 2

    by_conf = {}
    for r in rows:
        # No 'status' column exists — pins the exact bug the old code read.
        assert r.get("status") is None
        ra = r.get("attributes.routing")
        assert isinstance(ra, dict) and "confidence" in ra and "chosen_agent" in ra
        by_conf[round(float(ra["confidence"]), 2)] = r

    healthy = by_conf[0.83]
    failed = by_conf[0.4]
    assert str(healthy.get("status_code")).upper() == "UNSET"
    assert _span_success(healthy) is True
    assert str(failed.get("status_code")).upper() == "ERROR"
    assert _span_success(failed) is False
