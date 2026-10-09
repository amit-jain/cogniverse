"""Real-Phoenix test for a tenant-scoped ``PhoenixAnalytics`` trace fetch.

The reader derives the tenant project from the same telemetry config and the
same canonical tenant form the producer used, and ``get_traces`` requires an
explicit project_name.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_telemetry_phoenix.evaluation.analytics import (
    PhoenixAnalytics as Analytics,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


def test_get_traces_scopes_to_tenant_project(
    phoenix_container, telemetry_manager_with_phoenix
):
    manager = telemetry_manager_with_phoenix
    tenant_id = f"antrace{uuid4().hex[:8]}:tenant"
    op_name = f"AnalyticsRoot_{uuid4().hex[:6]}"

    with manager.span(
        name=op_name, tenant_id=tenant_id, attributes={"input.query": "x"}
    ):
        pass
    manager.force_flush(timeout_millis=10000)

    analytics = Analytics(telemetry_url=phoenix_container["http_endpoint"])
    project_name = manager.config.get_project_name(canonical_tenant_id(tenant_id))
    start = datetime.now(timezone.utc) - timedelta(hours=1)

    found = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        traces = analytics.get_traces(
            start_time=start,
            end_time=end,
            operation_filter=None,
            limit=10000,
            project_name=project_name,
        )
        if any(t.operation == op_name for t in traces):
            found = traces
            break
        time.sleep(2)
    assert found is not None, "tenant trace not found in the tenant project"
    assert len(found) == 1
    assert [t.operation for t in found] == [op_name]

    # get_traces requires an explicit project_name.
    end = datetime.now(timezone.utc)
    with pytest.raises(TypeError, match="project_name"):
        analytics.get_traces(start_time=start, end_time=end)
