"""Contract for the tenant-routing tracer used to instrument DSPy.

Every span an instrumented library (DSPy) emits must land in the project of
the tenant whose request is in flight -- resolved from a context variable,
never a shared cross-tenant project. These tests drive the real routing
tracer through real OTel SDK tracers (InMemory exporters, one per tenant) and
assert the exact exporter each span reaches, so a span cannot silently be
filed under the wrong tenant.
"""

from __future__ import annotations

from typing import Dict

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import NonRecordingSpan

from cogniverse_foundation.telemetry.tenant_context import (
    current_tenant_id,
    tenant_span_context,
)
from cogniverse_foundation.telemetry.tenant_routing import (
    TenantRoutingTracerProvider,
)


@pytest.fixture()
def tenant_tracers():
    """One real SDK tracer + InMemory exporter per tenant.

    Returns (resolve, exporters) where resolve(tenant_id) is the production-
    shaped callable the routing provider injects, and exporters[tenant_id]
    lets a test read exactly which spans that tenant received.
    """
    exporters: Dict[str, InMemorySpanExporter] = {}
    tracers: Dict[str, object] = {}

    def resolve(tenant_id: str):
        if tenant_id not in tracers:
            exporter = InMemorySpanExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            exporters[tenant_id] = exporter
            tracers[tenant_id] = provider.get_tracer(f"cogniverse-{tenant_id}")
        return tracers[tenant_id]

    return resolve, exporters


def _names(exporter: InMemorySpanExporter):
    return [s.name for s in exporter.get_finished_spans()]


def test_span_routes_to_the_tenant_in_context(tenant_tracers):
    resolve, exporters = tenant_tracers
    provider = TenantRoutingTracerProvider(resolve_tracer=resolve)
    tracer = provider.get_tracer("dspy")

    with tenant_span_context("acme:production"):
        with tracer.start_as_current_span("LM.__call__"):
            pass

    assert _names(exporters["acme:production"]) == ["LM.__call__"]
    assert set(exporters) == {"acme:production"}


def test_two_tenants_never_bleed_across_projects(tenant_tracers):
    resolve, exporters = tenant_tracers
    provider = TenantRoutingTracerProvider(resolve_tracer=resolve)
    tracer = provider.get_tracer("dspy")

    with tenant_span_context("acme:production"):
        with tracer.start_as_current_span("acme.LM"):
            pass
    with tenant_span_context("globex:production"):
        with tracer.start_as_current_span("globex.LM"):
            pass

    assert _names(exporters["acme:production"]) == ["acme.LM"]
    assert _names(exporters["globex:production"]) == ["globex.LM"]


def test_start_span_also_routes_by_context(tenant_tracers):
    resolve, exporters = tenant_tracers
    provider = TenantRoutingTracerProvider(resolve_tracer=resolve)
    tracer = provider.get_tracer("dspy")

    with tenant_span_context("acme:production"):
        span = tracer.start_span("Predict.forward")
        span.end()

    assert _names(exporters["acme:production"]) == ["Predict.forward"]


def test_no_tenant_in_context_yields_nonrecording_span_and_no_export(
    tenant_tracers, caplog
):
    resolve, exporters = tenant_tracers
    provider = TenantRoutingTracerProvider(resolve_tracer=resolve)
    tracer = provider.get_tracer("dspy")

    assert current_tenant_id() is None
    with caplog.at_level("WARNING"):
        with tracer.start_as_current_span("orphan.LM") as span:
            assert isinstance(span, NonRecordingSpan)

    # Nothing resolved, nothing exported -- a span with no tenant has nowhere
    # to go, and it is never filed under a shared fallback project.
    assert exporters == {}
    assert any("no tenant in context" in rec.message for rec in caplog.records), (
        f"expected a warning naming the missing tenant, got {[r.message for r in caplog.records]}"
    )


def test_context_is_reset_after_the_tenant_scope_exits():
    assert current_tenant_id() is None
    with tenant_span_context("acme:production"):
        assert current_tenant_id() == "acme:production"
        with tenant_span_context("globex:production"):
            assert current_tenant_id() == "globex:production"
        assert current_tenant_id() == "acme:production"
    assert current_tenant_id() is None
