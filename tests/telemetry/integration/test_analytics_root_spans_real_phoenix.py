"""PhoenixAnalytics.get_traces must bound ``limit`` by traces, not raw spans.

get_traces surfaces one TraceMetrics per root span. Without server-side
``root_spans_only``, ``limit`` bounds RAW spans, so a small limit returns the
newest spans — all children, since a trace's root is its oldest span — and the
client-side ``parent_id.isna()`` filter then finds zero roots. The Analytics
tab shows an empty window even though traces exist.

Emits one trace (1 root + 60 children) into a real Phoenix and asserts that
``get_traces(limit=10)`` still returns the single root.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.fixture(scope="module")
def analytics_telemetry(phoenix_container):
    """Real TelemetryManager (sync export) backed by the Phoenix container."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv(
            "TELEMETRY_OTLP_ENDPOINT", phoenix_container["otlp_endpoint"]
        ),
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.mark.asyncio
async def test_get_traces_returns_root_not_children(analytics_telemetry):
    from cogniverse_core.common.tenant_utils import canonical_tenant_id
    from cogniverse_telemetry_phoenix.evaluation.analytics import PhoenixAnalytics

    # Canonical (org:tenant) form so the emit project and the query project
    # derive identically — get_project_name passes tenant_id through verbatim.
    tenant = canonical_tenant_id("acme:analyticsroot")

    # One trace: a root plus 60 children, via the manager's span() context (the
    # exported path). The root is the outermost span, so it has the OLDEST
    # start_time and sorts last under Phoenix's default newest-first ordering.
    with analytics_telemetry.span("root_op", tenant_id=tenant):
        for i in range(60):
            with analytics_telemetry.span(f"child_{i}", tenant_id=tenant):
                pass

    # Push the OTLP exporter's buffer to Phoenix.
    analytics_telemetry.force_flush()

    project = analytics_telemetry.config.get_project_name(tenant)
    http = analytics_telemetry.config.provider_config["http_endpoint"]
    analytics = PhoenixAnalytics(telemetry_url=http)

    # limit=10 < 60 children: only the single root trace may surface. On the
    # pre-fix path this returns the 10 newest (child) spans, none of which is a
    # root, so metrics stays empty however long we wait.
    metrics = []
    for _ in range(20):
        metrics = analytics.get_traces(limit=10, project_name=project)
        if metrics:
            break
        await asyncio.sleep(1)

    assert len(metrics) == 1, [m.operation for m in metrics]
    assert metrics[0].operation == "root_op"
    assert metrics[0].timestamp.tzinfo is not None
