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
import math
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pandas as pd
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
from cogniverse_foundation.telemetry.context import search_span
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.optimization_cli import emit_ab_compare_span
from cogniverse_runtime.routers import admin, telemetry_metrics
from tests.utils.approval_review import run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    SEARCH,
    ab_result,
    record_profile_selection,
    record_sample_traces,
    record_search,
    record_trace,
)

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
def app(phoenix_container, phoenix_proxy, monkeypatch):
    """The telemetry metrics routes, and the admin routes that upload a
    tenant's golden set straight to Phoenix."""
    monkeypatch.setattr(admin, "_phoenix_endpoints", {})
    admin.set_phoenix_endpoints(
        phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
    )
    app = FastAPI()
    app.include_router(telemetry_metrics.router, prefix="/admin/tenant")
    app.include_router(admin.router, prefix="/admin")
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
        "project": telemetry.config.get_project_name(tenant),
        "spans": 1001,
        "modalities": [
            {
                "modality": "video",
                "count": 1001,
                "p50_ms": pytest.approx(10.0, abs=1e-6),
                "p95_ms": pytest.approx(10.0, abs=1e-6),
                "p99_ms": pytest.approx(10.0, abs=1e-6),
                "success_rate": 1.0,
            }
        ],
    }


async def test_a_tenant_with_no_selections_has_no_modalities(telemetry, app):
    tenant = _tenant("empty")
    async with _client(app) as client:
        response = await client.get(_profile_path(tenant))
    assert (response.status_code, response.json()) == (
        200,
        {
            "project": telemetry.config.get_project_name(tenant),
            "spans": 0,
            "modalities": [],
        },
    )


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
    golden_tenants = [_tenant("concurrentgolden"), _tenant("concurrentgolden")]
    for count, tenant in enumerate(tenants, start=1):
        for age in range(count):
            record_profile_selection(telemetry, tenant, "video", 20)
            record_trace(telemetry, tenant, SEARCH, 20, minutes_ago=age + 1)
    for count, tenant in enumerate(golden_tenants, start=1):
        for query in (SUNSET, RED_CAR)[:count]:
            record_search(tenant, query, "video_colpali", "hybrid", ["sunset.mp4"])
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        for count, tenant in enumerate(tenants, start=1):
            await _until(client, _profile_path(tenant), _counted(count))
            await _until(client, _traces_path(tenant), _requests(2 * count))
        for count, tenant in enumerate(golden_tenants, start=1):
            await _upload_golden(client, tenant)
            await _until(client, _golden_path(tenant), _searched(count))
        reads = 10
        barrier = threading.Barrier(reads, timeout=60)
        held = []

        def hold_span_reads(method, path, body):
            # Every read reaches Phoenix before any is answered.
            if "spans" in path and "annotations" not in path and len(held) < reads:
                held.append(path)
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_span_reads
        paths = [
            (_profile_path if i < 4 else _traces_path)(tenants[i % 2]) for i in range(8)
        ] + [_golden_path(tenant) for tenant in golden_tenants]
        responses = await asyncio.gather(*(client.get(path) for path in paths))
        phoenix_proxy.intercept = None

    assert len(held) == reads
    assert [
        [(m["modality"], m["count"]) for m in r.json()["modalities"]]
        for r in responses[:4]
    ] == [[("video", 1)], [("video", 2)]] * 2
    assert [
        (
            r.json()["statistics"]["requests"],
            {t["operation"] for t in r.json()["traces"]},
        )
        for r in responses[4:8]
    ] == [
        # A profile selection is a root span too.
        (2, {SEARCH, SPAN_NAME_PROFILE_SELECTION}),
        (4, {SEARCH, SPAN_NAME_PROFILE_SELECTION}),
    ] * 2
    assert [
        [(q["query"], q["mrr"]) for q in r.json()["queries"]] for r in responses[8:]
    ] == [[(SUNSET, 1.0)], [(SUNSET, 1.0), (RED_CAR, 0.0)]]


async def test_an_unreadable_telemetry_backend_answers_502(app, phoenix_proxy):
    tenant = _tenant("outage")
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    async with _client(app) as client:
        profile = await client.get(_profile_path(tenant))
        rlm = await client.get(f"/admin/tenant/{tenant}/telemetry/rlm-ab")
        traces = await client.get(_traces_path(tenant))
        causes = await client.get(_root_causes_path(tenant))
    assert [
        (
            response.status_code,
            {
                k: response.json()["detail"][k]
                for k in ("error", "message", "tenant_id")
            },
        )
        for response in (profile, rlm, traces, causes)
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
        (
            502,
            {
                "error": "telemetry_unavailable",
                "message": f"Could not read the traces of tenant {tenant}.",
                "tenant_id": tenant,
            },
        ),
        (
            502,
            {
                "error": "telemetry_unavailable",
                "message": f"Could not read the traces of tenant {tenant}.",
                "tenant_id": tenant,
            },
        ),
    ]


def _traces_path(tenant, **params):
    query = "&".join(
        f"{key}={value}"
        for key, values in params.items()
        for value in (values if isinstance(values, list) else [values])
    )
    return f"/admin/tenant/{tenant}/telemetry/traces?lookback_hours=1&{query}"


def _requests(expected):
    return lambda body: body["statistics"]["requests"] >= expected


async def test_traces_are_the_root_spans_with_their_statistics(telemetry, app):
    tenant = _tenant("traces")
    expected = record_sample_traces(telemetry, tenant)
    record_trace(telemetry, _tenant("othertraces"), SEARCH, 5, minutes_ago=1)
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(client, _traces_path(tenant), _requests(6))

    assert body["traces"] == expected
    assert body["facets"] == {
        "operations": ["agent.dispatch", SEARCH],
        "profiles": ["audio", "video_colpali"],
        "strategies": ["bm25", "hybrid", "semantic"],
    }
    assert body["statistics"] == {
        "requests": 6,
        "succeeded": 5,
        "failed": 1,
        "success_rate": pytest.approx(5 / 6),
        "latency_ms": {
            "mean": pytest.approx(2050 / 6),
            "min": pytest.approx(50.0),
            "p50": pytest.approx(250.0),
            "p75": pytest.approx(375.0),
            "p90": pytest.approx(700.0),
            "p95": pytest.approx(850.0),
            "p99": pytest.approx(970.0),
            "max": pytest.approx(1000.0),
        },
        "outlier_bounds_ms": {
            "lower": pytest.approx(-250.0),
            "upper": pytest.approx(750.0),
        },
        "by_operation": [
            {
                "operation": SEARCH,
                "count": 5,
                "mean_ms": pytest.approx(400.0),
                "p95_ms": pytest.approx(880.0),
                "error_rate": pytest.approx(0.2),
            },
            {
                "operation": "agent.dispatch",
                "count": 1,
                "mean_ms": pytest.approx(50.0),
                "p95_ms": pytest.approx(50.0),
                "error_rate": 0.0,
            },
        ],
    }


def _root_causes_path(tenant, **params):
    return _traces_path(tenant, **params).replace("/traces?", "/root-causes?")


async def test_root_causes_name_the_failed_and_slow_traces(telemetry, app):
    tenant = _tenant("rootcauses")
    expected = record_sample_traces(telemetry, tenant)
    telemetry.force_flush(timeout_millis=10000)
    failed = next(row["trace_id"] for row in expected if row["error"])
    slowest = next(row["trace_id"] for row in expected if row["duration_ms"] == 400)

    async with _client(app) as client:
        await _until(client, _traces_path(tenant), _requests(6))
        body = (await client.get(_root_causes_path(tenant, slow_percentile=75))).json()
        searches = (
            await client.get(
                _root_causes_path(tenant, operation="agent", include_slow="false")
            )
        ).json()

    # Successful durations 50, 100, 200, 300, 400: P75 is 300, so the 400 ms
    # search is the one slow trace.
    assert (
        body["traces"],
        body["failed"],
        body["slow"],
        body["failure_rate"],
        body["slow_threshold_ms"],
    ) == (6, 1, 1, pytest.approx(1 / 6), pytest.approx(300.0))
    # One failure with an error matching no known issue and spread over no
    # pattern yields no failure hypothesis; the slow search yields two.
    assert (body["root_causes"], body["recommendations"]) == (
        [
            {
                "hypothesis": f"Operation '{SEARCH}' experiencing performance "
                "degradation",
                "confidence": 0.9,
                "category": "performance",
                "evidence": [
                    "1 slow traces for this operation",
                    "Average duration: 400.0ms",
                    "Duration range: 400.0-400.0ms",
                    "Slowdown factor: 2.5x",
                ],
                "affected_traces": [slowest],
                "suggested_action": f"Optimize '{SEARCH}' operation or increase "
                "resources",
            },
            {
                "hypothesis": "Profile 'video_colpali' has performance issues",
                "confidence": 0.85,
                "category": "configuration",
                "evidence": ["1 slow traces with this profile", "Mean latency: 400ms"],
                "affected_traces": [slowest],
                "suggested_action": "Review 'video_colpali' configuration and "
                "resource allocation",
            },
        ],
        [
            {
                "priority": "medium",
                "category": "performance",
                "recommendation": "Optimize slow operations",
                "details": [
                    "Profile slow operations",
                    "Add caching where appropriate",
                    "Consider asynchronous processing",
                ],
                "affected_components": [
                    f"Optimize '{SEARCH}' operation or increase resources"
                ],
            },
            {
                "priority": "medium",
                "category": "configuration",
                "recommendation": "Review configuration settings",
                "details": ["Check Profile 'video_colpali' has performance issues"],
                "affected_components": [
                    "Review 'video_colpali' configuration and resource allocation"
                ],
            },
        ],
    )
    assert failed not in {
        trace for cause in body["root_causes"] for trace in cause["affected_traces"]
    }

    assert searches == {
        "traces": 1,
        "failed": 0,
        "slow": 0,
        "failure_rate": 0.0,
        "slow_threshold_ms": None,
        "root_causes": [],
        "recommendations": [],
    }


async def test_traces_filter_by_operation_profile_and_strategy(telemetry, app):
    tenant = _tenant("tracefilters")
    expected = record_sample_traces(telemetry, tenant)
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        await _until(client, _traces_path(tenant), _requests(6))
        by_name = (await client.get(_traces_path(tenant, operation="SEARCH"))).json()
        by_values = (
            await client.get(
                _traces_path(
                    tenant, profile=["video_colpali", "audio"], strategy="hybrid"
                )
            )
        ).json()
        none = (await client.get(_traces_path(tenant, operation="ingest"))).json()

    assert [t["span_id"] for t in by_name["traces"]] == [
        row["span_id"] for row in expected[:5]
    ]
    assert [t["span_id"] for t in by_values["traces"]] == [
        row["span_id"] for row in expected[:4]
    ]
    assert by_values["statistics"]["by_operation"] == [
        {
            "operation": SEARCH,
            "count": 4,
            "mean_ms": pytest.approx(250.0),
            "p95_ms": pytest.approx(385.0),
            "error_rate": 0.0,
        }
    ]
    # Facets describe the window, whatever the filters keep.
    assert by_name["facets"] == by_values["facets"] == none["facets"]
    assert none["facets"]["operations"] == ["agent.dispatch", SEARCH]
    assert none["statistics"] == {
        "requests": 0,
        "succeeded": 0,
        "failed": 0,
        "success_rate": None,
        "latency_ms": dict.fromkeys(
            ("mean", "min", "p50", "p75", "p90", "p95", "p99", "max")
        ),
        "outlier_bounds_ms": None,
        "by_operation": [],
    }
    assert none["traces"] == []


async def test_a_search_traced_by_the_search_service_is_reported(telemetry, app):
    tenant = _tenant("searchtrace")
    with search_span(
        tenant, "keynote", top_k=5, ranking_strategy="hybrid", profile="video_colpali"
    ):
        pass
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        body = await _until(client, _traces_path(tenant), _requests(1))

    [trace] = body["traces"]
    assert {
        key: trace[key]
        for key in ("operation", "succeeded", "profile", "strategy", "error")
    } == {
        "operation": SEARCH,
        "succeeded": True,
        "profile": "video_colpali",
        "strategy": "hybrid",
        "error": None,
    }


SUNSET, RED_CAR, DOG = "sunset over the sea", "a red car", "dog on a beach"
GOLDEN = [
    {"query": SUNSET, "expected_videos": "sunset"},
    {"query": RED_CAR, "expected_videos": ["red_car", "garage"]},
    {"query": DOG, "expected_videos": ["dog"]},
]


def _golden_path(tenant):
    return f"/admin/tenant/{tenant}/evaluation/golden?lookback_hours=1"


async def _upload_golden(client, tenant):
    response = await client.put(
        f"/admin/tenants/{tenant}/golden_set_ground_truth", json=GOLDEN
    )
    assert (response.status_code, response.json()["row_count"]) == (200, 3)


def _searched(scored, failed=0):
    return lambda body: (
        (len(body["queries"]), body["failed_searches"])
        == (
            scored,
            failed,
        )
    )


async def test_golden_evaluation_scores_the_tenants_latest_searches(telemetry, app):
    tenant = _tenant("golden")
    record_search(tenant, SUNSET, "video_colpali", "hybrid", ["sunset.mp4"])
    sunset = record_search(
        tenant, SUNSET, "video_colpali", "hybrid", ["beach.mp4", "sunset.mp4"]
    )
    red_car = record_search(
        tenant,
        RED_CAR,
        "video_colpali",
        "hybrid",
        ["red_car.mp4", "red_car.mp4", "beach.mp4", "garage.mov"],
    )
    record_search(tenant, SUNSET, "video_colpali", "bm25", [], error="backend down")
    record_search(tenant, "not a golden query", "video_colpali", "bm25", ["dog.mp4"])
    other = _tenant("othergolden")
    record_search(other, DOG, "video_colpali", "bm25", ["dog.mp4"])
    telemetry.force_flush(timeout_millis=10000)

    async with _client(app) as client:
        await _upload_golden(client, tenant)
        await _upload_golden(client, other)
        # Phoenix shows the second sunset search only after the first.
        body = await _until(
            client,
            _golden_path(tenant),
            lambda body: (
                _searched(2, 1)(body) and body["queries"][0]["trace_id"] == sunset[0]
            ),
        )
        other_body = await _until(client, _golden_path(other), _searched(1))
        traces = (await client.get(_traces_path(tenant))).json()["traces"]

    # Phoenix keeps a span's start to the microsecond; the evaluation reports
    # the start the trace analytics report for the same trace.
    started = {trace["trace_id"]: trace["start_time"] for trace in traces}
    for trace_id, recorded in (sunset, red_car):
        stored = pd.Timestamp(started[trace_id])
        assert abs(stored - recorded) <= pd.Timedelta(1, "us")
    sunset_ndcg = 1 / math.log2(3)
    red_car_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert body == {
        "golden_queries": 3,
        "strategies": [
            {
                "profile": "video_colpali",
                "strategy": "hybrid",
                "queries": 2,
                "mrr": 0.75,
                "ndcg": pytest.approx((sunset_ndcg + red_car_ndcg) / 2),
                "recall_at_1": 0.25,
                "recall_at_5": 1.0,
                "precision_at_5": pytest.approx((1 / 2 + 2 / 3) / 2),
                "success_rate": 0.5,
            }
        ],
        "queries": [
            {
                "profile": "video_colpali",
                "strategy": "hybrid",
                "query": SUNSET,
                "expected": ["sunset"],
                "retrieved": ["beach", "sunset"],
                "searched_at": started[sunset[0]],
                "trace_id": sunset[0],
                "mrr": 0.5,
                "ndcg": pytest.approx(sunset_ndcg),
                "recall_at_1": 0.0,
                "recall_at_5": 1.0,
                "precision_at_5": 0.5,
            },
            {
                "profile": "video_colpali",
                "strategy": "hybrid",
                "query": RED_CAR,
                "expected": ["red_car", "garage"],
                "retrieved": ["red_car", "beach", "garage"],
                "searched_at": started[red_car[0]],
                "trace_id": red_car[0],
                "mrr": 1.0,
                "ndcg": pytest.approx(red_car_ndcg),
                "recall_at_1": 0.5,
                "recall_at_5": 1.0,
                "precision_at_5": pytest.approx(2 / 3),
            },
        ],
        "unsearched_queries": [DOG],
        "failed_searches": 1,
        "unscored_searches": 0,
    }
    assert [(q["query"], q["mrr"]) for q in other_body["queries"]] == [(DOG, 1.0)]
    assert other_body["unsearched_queries"] == [SUNSET, RED_CAR]


async def test_golden_evaluation_of_a_tenant_without_a_golden_set_answers_404(app):
    tenant = _tenant("nogolden")
    async with _client(app) as client:
        response = await client.get(_golden_path(tenant))
    assert response.status_code == 404
    assert {
        k: response.json()["detail"][k] for k in ("error", "message", "tenant_id")
    } == {
        "error": "golden_set_missing",
        "message": f"Tenant {tenant} has no golden set. Upload one with "
        f"PUT /admin/tenants/{tenant}/golden_set_ground_truth.",
        "tenant_id": tenant,
    }


async def test_golden_evaluation_reports_an_unreadable_backend_as_502(
    app, phoenix_proxy
):
    tenant = _tenant("goldenoutage")
    async with _client(app) as client:
        await _upload_golden(client, tenant)
        await _until(client, _golden_path(tenant), _searched(0))

        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        golden_down = await client.get(_golden_path(tenant))

        def fail_span_reads(method, path, body):
            return (503, {"detail": "down"}) if "/spans" in path else None

        phoenix_proxy.intercept = fail_span_reads
        spans_down = await client.get(_golden_path(tenant))

    assert [
        (
            response.status_code,
            {k: response.json()["detail"][k] for k in ("error", "message")},
        )
        for response in (golden_down, spans_down)
    ] == [
        (
            502,
            {
                "error": "golden_set_store_unavailable",
                "message": f"Could not read the golden set of tenant {tenant}.",
            },
        ),
        (
            502,
            {
                "error": "telemetry_unavailable",
                "message": f"Could not read the {SEARCH} spans of tenant {tenant}.",
            },
        ),
    ]
