"""Real-Phoenix convergence of the batch/live trace readers on the writer project.

A span emitted through the telemetry writer lands in the project named by the
LOADED ``tenant_project_template``. ``TraceManager`` and the batch solver's
``_resolve_project`` must derive that SAME project, so a template override never
points the readers at a project nothing writes to. Emits a span under a
non-default template and reads it back through the reader from the identical
project.
"""

from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

_OVERRIDE_TEMPLATE = "acme-spans-{tenant_id}-v2"


@pytest.fixture
def override_template_manager(phoenix_container):
    """Real TelemetryManager (sync export) with a NON-default project template,
    installed as the global singleton the readers consult."""
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_evaluation.providers.registry import get_evaluation_registry
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    get_evaluation_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv(
            "TELEMETRY_OTLP_ENDPOINT", phoenix_container["otlp_endpoint"]
        ),
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
        tenant_project_template=_OVERRIDE_TEMPLATE,
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    get_evaluation_registry().clear_cache()


def test_readers_converge_on_writer_project(
    override_template_manager, phoenix_container
):
    from cogniverse_core.common.tenant_utils import canonical_tenant_id
    from cogniverse_evaluation.core.solvers import _resolve_project
    from cogniverse_evaluation.data.storage import ConnectionConfig, TelemetryStorage
    from cogniverse_evaluation.data.traces import TraceManager

    manager = override_template_manager
    tenant = canonical_tenant_id("acme:convergerd")
    op_name = f"ConvergeOp_{uuid4().hex[:8]}"

    # Writer path: emit a span; it lands in the override-template project.
    with manager.span(name=op_name, tenant_id=tenant, attributes={"input.query": "x"}):
        pass
    manager.force_flush(timeout_millis=10000)

    writer_project = manager.config.get_project_name(tenant)
    assert writer_project == "acme-spans-acme:convergerd-v2"

    # Reader derivation converges on the writer project (exact string), through
    # both the TraceManager constructor and the batch solver's resolution.
    storage = TelemetryStorage(
        ConnectionConfig(
            http_endpoint=phoenix_container["http_endpoint"],
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            enable_health_checks=False,
        )
    )
    trace_manager = TraceManager(tenant_id=tenant, storage=storage)
    assert trace_manager.project_name == writer_project
    assert _resolve_project({"tenant_id": tenant}) == writer_project

    # And the emitted span reads back from that SAME project via the reader —
    # exact operation-name identity, proving writer and reader share a namespace.
    names: set[str] = set()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        df = trace_manager.get_recent_traces(hours_back=1, limit=1000)
        if not df.empty and "name" in df.columns:
            names = set(df["name"].dropna().tolist())
            if op_name in names:
                break
        time.sleep(2)

    assert op_name in names, (
        f"span {op_name!r} not read back from {writer_project!r}; "
        f"reader saw {sorted(names)[:10]}"
    )
