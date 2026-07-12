"""Real-Phoenix test for the Analytics tab's tenant-scoped trace fetch.

The tab called get_traces() with no project, so it queried Phoenix's ``default``
project and real tenant traffic showed "No traces found". fetch_tenant_traces
derives the tenant project so it can't be forgotten.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogniverse_dashboard.utils.traces import fetch_tenant_traces
from cogniverse_telemetry_phoenix.evaluation.analytics import (
    PhoenixAnalytics as Analytics,
)

pytestmark = pytest.mark.integration


def test_fetch_tenant_traces_scopes_to_tenant_project(
    phoenix_container, telemetry_manager_with_phoenix
):
    manager = telemetry_manager_with_phoenix
    tenant_id = f"antrace{uuid4().hex[:8]}"
    op_name = f"AnalyticsRoot_{uuid4().hex[:6]}"

    with manager.span(
        name=op_name, tenant_id=tenant_id, attributes={"input.query": "x"}
    ):
        pass
    manager.force_flush(timeout_millis=10000)

    analytics = Analytics(telemetry_url=phoenix_container["http_endpoint"])
    start = datetime.now(timezone.utc) - timedelta(hours=1)

    found = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        traces = fetch_tenant_traces(analytics, tenant_id, start, end, None)
        if any(t.operation == op_name for t in traces):
            found = traces
            break
        time.sleep(2)
    assert found is not None, "tenant trace not found via fetch_tenant_traces"

    # Without the tenant project, get_traces queries Phoenix's 'default' project
    # and misses tenant traffic — the exact bug the tab had.
    end = datetime.now(timezone.utc)
    default_traces = analytics.get_traces(start_time=start, end_time=end)
    assert not any(t.operation == op_name for t in default_traces)
