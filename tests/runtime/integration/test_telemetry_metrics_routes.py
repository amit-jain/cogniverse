"""The telemetry metrics routes against real Phoenix.

Profile selections are recorded the way ``ProfileSelectionAgent`` records
them (through its own span writer, or with the same output slot and fixed
start and end times so the latencies are exact); A/B comparisons through the
``ab-compare`` job's own span writer. The runtime reads them through a
forwarding proxy in front of Phoenix's HTTP API, so a test can hold or fail
the reads.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_PROFILE_SELECTION,
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.optimization_cli import emit_ab_compare_span
from cogniverse_runtime.routers import telemetry_metrics
from tests.utils.approval_review import run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import ab_result, record_profile_selection

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    """The global telemetry manager: spans export to Phoenix, reads go
    through ``phoenix_proxy``."""
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": phoenix_proxy.url,
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture()
def app(phoenix_proxy):
    app = FastAPI()
    app.include_router(telemetry_metrics.router, prefix="/admin/tenant")
    yield app
    phoenix_proxy.intercept = None


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
        timeout=300,
    )


def _tenant(prefix="metrics"):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _record_agent_selection(telemetry, tenant_id, modality):
    """A profile selection recorded by ``ProfileSelectionAgent``'s writer."""
    run_in_own_loop(
        ProfileSelectionAgent._emit_profile_span(
            SimpleNamespace(telemetry_manager=telemetry),
            query="show me the keynote",
            tenant_id=tenant_id,
            available_profiles="audio_profile",
            selected_profile="audio_profile",
            intent="search",
            modality=modality,
            complexity="simple",
            confidence=0.8,
        )
    )


async def _until(client, path, predicate, timeout=90.0):
    deadline = time.monotonic() + timeout
    while True:
        response = await client.get(path)
        assert response.status_code == 200, response.text
        body = response.json()
        if predicate(body) or time.monotonic() > deadline:
            return body
        await asyncio.sleep(2)


def _profile_path(tenant):
    return f"/admin/tenant/{tenant}/telemetry/profile-selection?lookback_hours=1"


def _counted(expected):
    return lambda body: sum(m["count"] for m in body["modalities"]) >= expected


async def test_profile_selection_metrics_per_modality(telemetry, app):
    tenant = _tenant()
    for duration in (100, 200, 300, 400):
        record_profile_selection(telemetry, tenant, "video", duration)
    record_profile_selection(telemetry, tenant, "image", 50)
    record_profile_selection(telemetry, tenant, "image", 150, failed=True)
    _record_agent_selection(telemetry, tenant, "audio")
    # Neither another span name nor another tenant's selections count.
    with telemetry.span("cogniverse.routing", tenant_id=tenant):
        pass
    record_profile_selection(telemetry, _tenant("othertenant"), "video", 999)
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(client, _profile_path(tenant), _counted(7))

    video, image, audio = body["modalities"]
    assert [video, image] == [
        {
            "modality": "video",
            "count": 4,
            "p50_ms": pytest.approx(250.0, abs=1e-6),
            "p95_ms": pytest.approx(385.0, abs=1e-6),
            "p99_ms": pytest.approx(397.0, abs=1e-6),
            "success_rate": 1.0,
        },
        {
            "modality": "image",
            "count": 2,
            "p50_ms": pytest.approx(100.0, abs=1e-6),
            "p95_ms": pytest.approx(145.0, abs=1e-6),
            "p99_ms": pytest.approx(149.0, abs=1e-6),
            "success_rate": 0.5,
        },
    ]
    # The agent's own span finishes without setting a status: a success.
    assert (audio["modality"], audio["count"], audio["success_rate"]) == (
        "audio",
        1,
        1.0,
    )
    assert audio["p50_ms"] == audio["p95_ms"] == audio["p99_ms"]


async def test_every_selection_in_the_window_is_counted(telemetry, app):
    """More selections than one page of the backend's span reads."""
    tenant = _tenant("volume")
    for _ in range(1001):
        record_profile_selection(telemetry, tenant, "video", 10)
    telemetry.force_flush(timeout_millis=30000)

    async with _client(app) as client:
        body = await _until(client, _profile_path(tenant), _counted(1001), 180)

    assert body == {
        "modalities": [
            {
                "modality": "video",
                "count": 1001,
                "p50_ms": pytest.approx(10.0, abs=1e-6),
                "p95_ms": pytest.approx(10.0, abs=1e-6),
                "p99_ms": pytest.approx(10.0, abs=1e-6),
                "success_rate": 1.0,
            }
        ]
    }


async def test_a_tenant_with_no_selections_has_no_modalities(telemetry, app):
    async with _client(app) as client:
        response = await client.get(_profile_path(_tenant("empty")))
    assert (response.status_code, response.json()) == (200, {"modalities": []})


async def test_rlm_ab_comparisons_aggregate_per_dataset(telemetry, app):
    tenant = _tenant("rlmab")
    tracer = telemetry._get_tracer_for_project(tenant, None)
    rows = [
        ("ab-1", "first question", 200.0, 30, 0.25, False, "lectures"),
        ("ab-2", "second question", 400.0, 50, 0.25, True, "lectures"),
        ("ab-3", "third question", 600.0, 10, -0.5, False, "podcasts"),
    ]
    for ab_id, query, latency, tokens, judge, fallback, dataset in rows:
        emit_ab_compare_span(
            tracer,
            ab_result(ab_id, query, latency, tokens, judge, fallback),
            tenant,
            dataset,
        )
        time.sleep(0.01)
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(
            client,
            f"/admin/tenant/{tenant}/telemetry/rlm-ab?lookback_hours=1",
            lambda b: b["rows"] >= 3,
        )

    assert {key: body[key] for key in body if key != "comparisons"} == {
        "rows": 3,
        "avg_latency_delta_ms": 400.0,
        "avg_tokens_delta": 30.0,
        "avg_judge_delta": 0.0,
        "fallback_rate": pytest.approx(1 / 3),
        "per_dataset": [
            {
                "queries_dataset": "lectures",
                "rows": 2,
                "avg_latency_delta_ms": 300.0,
                "avg_tokens_delta": 40.0,
                "avg_judge_delta": 0.25,
            },
            {
                "queries_dataset": "podcasts",
                "rows": 1,
                "avg_latency_delta_ms": 600.0,
                "avg_tokens_delta": 10.0,
                "avg_judge_delta": -0.5,
            },
        ],
    }
    assert [
        {key: c[key] for key in c if key != "start_time"} for c in body["comparisons"]
    ] == [
        {
            "ab_id": ab_id,
            "query": query,
            "queries_dataset": dataset,
            "latency_delta_ms": latency,
            "tokens_delta": float(tokens),
            "judge_delta": judge,
            "with_rlm_was_fallback": fallback,
        }
        for ab_id, query, latency, tokens, judge, fallback, dataset in reversed(rows)
    ]


async def test_concurrent_reads_each_answer_their_own_tenant(
    telemetry, app, phoenix_proxy
):
    tenants = [_tenant("concurrent"), _tenant("concurrent")]
    for count, tenant in enumerate(tenants, start=1):
        for _ in range(count):
            record_profile_selection(telemetry, tenant, "video", 20)
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        for count, tenant in enumerate(tenants, start=1):
            await _until(client, _profile_path(tenant), _counted(count))
        reads = 6
        barrier = threading.Barrier(reads, timeout=60)
        held = []

        def hold_span_reads(method, path, body):
            # Every read reaches Phoenix before any is answered.
            if "spans" in path and "annotations" not in path and len(held) < reads:
                held.append(path)
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_span_reads
        responses = await asyncio.gather(
            *(client.get(_profile_path(tenants[i % 2])) for i in range(reads))
        )
        phoenix_proxy.intercept = None

    assert len(held) == reads
    assert [
        [(m["modality"], m["count"]) for m in r.json()["modalities"]] for r in responses
    ] == [[("video", 1)], [("video", 2)]] * 3


async def test_an_unreadable_telemetry_backend_answers_502(app, phoenix_proxy):
    tenant = _tenant("outage")
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    async with _client(app) as client:
        profile = await client.get(_profile_path(tenant))
        rlm = await client.get(f"/admin/tenant/{tenant}/telemetry/rlm-ab")
    assert [
        (
            response.status_code,
            {
                k: response.json()["detail"][k]
                for k in ("error", "message", "tenant_id")
            },
        )
        for response in (profile, rlm)
    ] == [
        (
            502,
            {
                "error": "telemetry_unavailable",
                "message": f"Could not read the {SPAN_NAME_PROFILE_SELECTION} "
                f"spans of tenant {tenant}.",
                "tenant_id": tenant,
            },
        ),
        (
            502,
            {
                "error": "telemetry_unavailable",
                "message": f"Could not read the rlm.ab_compare spans of tenant {tenant}.",
                "tenant_id": tenant,
            },
        ),
    ]
