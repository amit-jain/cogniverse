"""Tracer retirement preserves serving progress and real Phoenix span delivery."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import grpc
import httpx
import pytest
from fastapi import FastAPI
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2_grpc,
)
from phoenix.client import Client

from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


class _ExportGate(trace_service_pb2_grpc.TraceServiceServicer):
    """Pause real OTLP requests before forwarding them to the owned Phoenix."""

    def __init__(self, upstream: str):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.failed = threading.Event()
        self.fail = False
        self.accepted = set()
        self.attempts = 0
        self.channel = grpc.insecure_channel(upstream)
        self.client = trace_service_pb2_grpc.TraceServiceStub(self.channel)

    def Export(self, request, context):
        self.attempts += 1
        self.entered.set()
        if not self.release.wait(15):
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "export gate expired")
        if self.fail:
            self.failed.set()
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "collector rejects batch")
        result = self.client.Export(request, timeout=10)
        self.accepted.update(
            span.span_id.hex()
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
        return result


@pytest.fixture
def retirement(phoenix_container):
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    gate = _ExportGate(phoenix_container["otlp_endpoint"])
    executor = ThreadPoolExecutor(max_workers=8)
    server = grpc.server(executor)
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(gate, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    config = TelemetryConfig(
        provider="phoenix",
        otlp_endpoint=f"127.0.0.1:{port}",
        provider_config={"http_endpoint": phoenix_container["http_endpoint"]},
        tenant_cache_ttl_seconds=3600,
        max_cached_tenants=2,
        batch_config=BatchExportConfig(schedule_delay_millis=10),
    )
    manager = TelemetryManager(config)
    tenant = f"prodfixoptimization:t{uuid.uuid4().hex[:10]}"
    try:
        yield manager, gate, tenant, phoenix_container["http_endpoint"]
    finally:
        gate.release.set()
        TelemetryManager.reset()
        get_telemetry_registry().clear_cache()
        server.stop(0).wait()
        gate.channel.close()
        executor.shutdown(wait=True)


def _expire(manager, tenant, mode):
    if mode == "ttl":
        key = f"{tenant}:{manager.config.get_project_name(tenant)}"
        with manager._lock:
            manager._tracer_created_at[key] = time.monotonic() - 3601
    else:
        manager.config.max_cached_tenants = 1


def _request_app(manager):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/search/{tenant}")
    async def search(tenant: str):
        with manager.span(
            "api.search.request", tenant_id=tenant, component="search_service"
        ) as span:
            await asyncio.sleep(0)
            return {"span_id": f"{span.get_span_context().span_id:016x}"}

    return app


async def _assert_persisted(endpoint, expected):
    client = Client(base_url=endpoint)
    observed = {}
    for _ in range(100):
        for project in expected:
            frame = await asyncio.to_thread(
                client.spans.get_spans_dataframe, project_identifier=project
            )
            observed[project] = (
                set() if frame is None else set(frame["context.span_id"])
            )
        if observed == expected:
            break
        await asyncio.sleep(0.1)
    assert observed == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ttl", "lru"])
async def test_slow_export_retirement_keeps_requests_responsive(retirement, mode):
    manager, gate, tenant, endpoint = retirement
    with manager.span("seed", tenant_id=tenant) as span:
        seed_id = f"{span.get_span_context().span_id:016x}"
    assert await asyncio.to_thread(gate.entered.wait, 5) is True
    _expire(manager, tenant, mode)
    next_tenant = tenant if mode == "ttl" else f"{tenant}next"
    timer = threading.Timer(2, gate.release.set)
    timer.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_request_app(manager)),
            base_url="http://test",
        ) as client:
            request, sibling = await asyncio.gather(
                client.post(f"/search/{next_tenant}"), client.get("/health")
            )
        assert gate.release.is_set() is False
        assert sibling.json() == {"status": "healthy"}
        assert request.status_code == 200
        request_id = request.json()["span_id"]
    finally:
        gate.release.set()
        timer.cancel()
        timer.join()
    for _ in range(100):
        if manager.get_stats()["retirement_workers"] == 0:
            break
        await asyncio.sleep(0.01)
    assert manager.get_stats()["retired_providers"] == 0
    assert manager.get_stats()["retirement_workers"] == 0
    with manager.span("recovered", tenant_id=next_tenant) as span:
        recovery_id = f"{span.get_span_context().span_id:016x}"
    await asyncio.to_thread(manager.shutdown)
    expected = {manager.config.get_project_name(tenant): {seed_id}}
    expected.setdefault(manager.config.get_project_name(next_tenant), set()).update(
        {request_id, recovery_id}
    )
    assert gate.accepted == {seed_id, request_id, recovery_id}
    await _assert_persisted(endpoint, expected)
    assert manager.get_stats()["retired_providers"] == 0
    assert manager.get_stats()["retirement_workers"] == 0
    assert [
        thread.name
        for thread in threading.enumerate()
        if thread.name == "cogniverse-telemetry-retirement"
    ] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_retirement_waits_for_inflight_span_end(retirement, cancel):
    manager, gate, tenant, endpoint = retirement
    gate.release.set()
    entered = asyncio.Event()
    release = asyncio.Event()
    ids = {}

    async def held_request():
        with manager.span("held-request", tenant_id=tenant) as span:
            ids["held"] = f"{span.get_span_context().span_id:016x}"
            entered.set()
            await release.wait()

    task = asyncio.create_task(held_request())
    await entered.wait()
    _expire(manager, tenant, "ttl")
    with manager.span("replacement", tenant_id=tenant) as span:
        ids["replacement"] = f"{span.get_span_context().span_id:016x}"
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        release.set()
        await task
    await asyncio.to_thread(manager.shutdown)
    assert gate.accepted == set(ids.values())
    await _assert_persisted(
        endpoint, {manager.config.get_project_name(tenant): set(ids.values())}
    )
    assert manager.get_stats()["retired_providers"] == 0
    assert manager.get_stats()["retirement_workers"] == 0
    assert [
        thread.name
        for thread in threading.enumerate()
        if thread.name == "cogniverse-telemetry-retirement"
    ] == []


@pytest.mark.asyncio
@pytest.mark.expects_telemetry_loss_warning
async def test_failed_export_bounds_retirement_and_releases_workers(retirement, caplog):
    manager, gate, tenant, _ = retirement
    manager.config.max_cached_tenants = 1
    with manager.span("failed-seed", tenant_id=tenant):
        pass
    assert await asyncio.to_thread(gate.entered.wait, 5) is True
    gate.fail = True
    timer = threading.Timer(2, gate.release.set)
    timer.start()
    try:
        for index in range(8):
            with manager.span("churn", tenant_id=f"{tenant}{index}"):
                pass
        assert gate.release.is_set() is False
        assert manager.get_stats()["retired_providers"] == 1
        assert manager.get_stats()["retirement_workers"] == 1
        assert manager.get_stats()["cached_tenants"] == 1
    finally:
        gate.release.set()
        timer.cancel()
        timer.join()
    await asyncio.to_thread(manager.shutdown)
    assert gate.failed.is_set() is True
    assert gate.accepted == set()
    assert gate.attempts == 2
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "opentelemetry.exporter.otlp.proto.grpc.exporter"
    ] == [
        f"Failed to export traces to {manager.config.otlp_endpoint}, "
        "error code: StatusCode.INVALID_ARGUMENT"
    ]
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Telemetry provider retirement capacity")
    ] == [
        "Telemetry provider retirement capacity reached; "
        "new optional spans are not recorded until exporters drain"
    ] * 7
    # Spans refused while the exporters drain are the point of the bound: each
    # one says which tenant lost it rather than failing the request.
    assert [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("No tracer for span")
    ] == [
        f"No tracer for span churn: tenant={tenant}{index} "
        f"project={manager.config.get_project_name(f'{tenant}{index}')} "
        f"endpoint={manager.config.otlp_endpoint}; span not recorded"
        for index in range(1, 8)
    ]
    assert manager.get_stats()["retired_providers"] == 0
    assert manager.get_stats()["retirement_workers"] == 0
