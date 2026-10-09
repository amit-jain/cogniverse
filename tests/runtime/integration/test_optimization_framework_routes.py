"""The optimization framework routes against real Phoenix.

Searches are recorded by the search service's own span writers, routing
decisions with the gateway's output slot, and every read goes through a
forwarding proxy in front of Phoenix's HTTP API, so a test can hold or fail
it. Datasets are created in Phoenix and read back by the optimization CLI's
own loader; the recommender is trained, stored in the tenant's artifact store
and loaded again for each prediction.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.routing.profile_performance_optimizer import FEATURE_NAMES
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.optimization_cli import _dataset_profile_ground_truth_rows
from cogniverse_runtime.routers import optimization_framework
from cogniverse_sdk.document import source_title_key
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    record_routing,
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
def app(telemetry, phoenix_proxy):
    app = FastAPI()
    app.include_router(optimization_framework.router, prefix="/admin/tenant")
    yield app
    phoenix_proxy.intercept = None


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
        timeout=300,
    )


def _tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


async def _until(client, path, predicate, timeout=90.0):
    deadline = time.monotonic() + timeout
    while True:
        response = await client.get(path)
        assert response.status_code == 200, response.text
        body = response.json()
        if predicate(body) or time.monotonic() > deadline:
            return body
        await asyncio.sleep(2)


def _searches_path(tenant):
    return f"/admin/tenant/{tenant}/search-annotations?lookback_hours=1"


def _searches(count):
    return lambda body: len(body["searches"]) == count


def _annotated(count):
    return lambda body: sum(1 for s in body["searches"] if s["annotation"]) == count


SUNSET = "sunset over the sea"
HARBOUR = "boats in the harbour"
SUNSET_TITLES = ["Sunset Reel", "Beach Walk", "Night Sky", "Pier", "Waves", "Gulls"]
HARBOUR_TITLES = ["Harbour Tour", "Ferry"]


async def _seed_two_searches(telemetry, client, tenant):
    record_search(tenant, SUNSET, "video_colpali", "hybrid", SUNSET_TITLES)
    time.sleep(0.01)
    record_search(tenant, HARBOUR, "video_xclip", "bm25", HARBOUR_TITLES)
    telemetry.force_flush(timeout_millis=10000)
    body = await _until(client, _searches_path(tenant), _searches(2))
    return {search["query"]: search for search in body["searches"]}


async def test_a_reviewer_rates_searches_and_the_ratings_read_back(telemetry, app):
    tenant, other = _tenant("annotate"), _tenant("annotateother")
    record_search(other, "other tenant search", "video_colpali", "hybrid", ["X"])
    async with _client(app) as client:
        searches = await _seed_two_searches(telemetry, client, tenant)
        listed = (await client.get(_searches_path(tenant))).json()["searches"]
        assert [s["query"] for s in listed] == [HARBOUR, SUNSET]
        assert {
            query: (s["results"], s["profile"], s["strategy"], s["annotation"])
            for query, s in searches.items()
        } == {
            SUNSET: (SUNSET_TITLES[:5], "video_colpali", "hybrid", None),
            HARBOUR: (HARBOUR_TITLES, "video_xclip", "bm25", None),
        }

        sunset, harbour = searches[SUNSET]["span_id"], searches[HARBOUR]["span_id"]
        stars = await client.post(
            f"/admin/tenant/{tenant}/search-annotations/{sunset}",
            json={"kind": "stars", "value": 4, "notes": "  top hit is right  "},
        )
        thumbs = await client.post(
            f"/admin/tenant/{tenant}/search-annotations/{harbour}",
            json={"kind": "thumbs", "value": 0},
        )
        assert (stars.status_code, stars.json()) == (
            200,
            {
                "span_id": sunset,
                "label": "positive",
                "score": 0.8,
                "annotation_type": "stars",
            },
        )
        assert (thumbs.status_code, thumbs.json()) == (
            200,
            {
                "span_id": harbour,
                "label": "negative",
                "score": 0.0,
                "annotation_type": "thumbs",
            },
        )
        body = await _until(client, _searches_path(tenant), _annotated(2))
        assert {s["query"]: s["annotation"] for s in body["searches"]} == {
            SUNSET: {
                "label": "positive",
                "score": 0.8,
                "annotation_type": "stars",
                "notes": "top hit is right",
            },
            HARBOUR: {
                "label": "negative",
                "score": 0.0,
                "annotation_type": "thumbs",
                "notes": "User annotation",
            },
        }

        relevance = await client.post(
            f"/admin/tenant/{tenant}/search-annotations/{harbour}",
            json={"kind": "relevance", "value": 0.5, "notes": "partly relevant"},
        )
        assert relevance.json()["label"] == "neutral"
        body = await _until(
            client,
            _searches_path(tenant),
            lambda b: any(
                s["annotation"] and s["annotation"]["annotation_type"] == "relevance"
                for s in b["searches"]
            ),
        )
        assert {s["query"]: s["annotation"]["score"] for s in body["searches"]} == {
            SUNSET: 0.8,
            HARBOUR: 0.5,
        }
        count = await client.get(
            f"/admin/tenant/{tenant}/search-annotations/count?lookback_days=1"
        )
        assert count.json() == {"lookback_days": 1, "annotated_searches": 2}


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        ({"kind": "stars", "value": 6}, "A star rating is a whole number from 1 to 5."),
        (
            {"kind": "stars", "value": 2.5},
            "A star rating is a whole number from 1 to 5.",
        ),
        ({"kind": "thumbs", "value": 0.5}, "A thumbs rating is 1 (good) or 0 (bad)."),
        ({"kind": "relevance", "value": 1.2}, "A relevance score is between 0 and 1."),
    ],
)
async def test_a_rating_outside_its_scale_is_refused(app, body, detail):
    tenant = _tenant("badrating")
    async with _client(app) as client:
        response = await client.post(
            f"/admin/tenant/{tenant}/search-annotations/0123456789abcdef", json=body
        )
    assert (response.status_code, response.json()) == (400, {"detail": detail})


async def test_a_search_of_another_tenant_cannot_be_rated(telemetry, app):
    tenant, owner = _tenant("ratethief"), _tenant("rateowner")
    async with _client(app) as client:
        searches = await _seed_two_searches(telemetry, client, owner)
        span_id = searches[SUNSET]["span_id"]
        response = await client.post(
            f"/admin/tenant/{tenant}/search-annotations/{span_id}",
            json={"kind": "thumbs", "value": 1},
        )
        assert (response.status_code, response.json()) == (
            404,
            {"detail": f"Tenant {tenant} has no recorded search {span_id}."},
        )
        owned = (await client.get(_searches_path(owner))).json()["searches"]
    assert [s["annotation"] for s in owned] == [None, None]


async def test_concurrent_ratings_each_land_on_their_own_search(
    telemetry, app, phoenix_proxy
):
    tenant = _tenant("concurrentrate")
    queries = [f"concurrent query {index}" for index in range(6)]
    for query in queries:
        record_search(tenant, query, "video_colpali", "hybrid", [f"{query} title"])
    telemetry.force_flush(timeout_millis=10000)
    async with _client(app) as client:
        body = await _until(client, _searches_path(tenant), _searches(6))
        span_by_query = {s["query"]: s["span_id"] for s in body["searches"]}
        writes = 6
        barrier = threading.Barrier(writes, timeout=60)
        held = []

        def hold_annotation_writes(method, path, body):
            # Every rating reaches Phoenix before any is written.
            if method == "POST" and "span_annotations" in path and len(held) < writes:
                held.append(path)
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_annotation_writes
        responses = await asyncio.gather(
            *(
                client.post(
                    f"/admin/tenant/{tenant}/search-annotations/{span_by_query[query]}",
                    json={"kind": "relevance", "value": index / 10},
                )
                for index, query in enumerate(queries)
            )
        )
        phoenix_proxy.intercept = None
        assert [r.status_code for r in responses] == [200] * 6
        body = await _until(client, _searches_path(tenant), _annotated(6))
    assert len(held) == writes
    assert {s["query"]: s["annotation"]["score"] for s in body["searches"]} == {
        query: index / 10 for index, query in enumerate(queries)
    }


async def test_golden_dataset_keeps_the_well_rated_searches(telemetry, app):
    tenant = _tenant("golden")
    async with _client(app) as client:
        searches = await _seed_two_searches(telemetry, client, tenant)
        # An unrated search contributes nothing.
        record_search(tenant, "keynote talk", "video_colpali", "hybrid", ["Keynote"])
        telemetry.force_flush(timeout_millis=10000)
        await _until(client, _searches_path(tenant), _searches(3))
        for query, value in ((SUNSET, 1.0), (HARBOUR, 0.6)):
            response = await client.post(
                f"/admin/tenant/{tenant}/search-annotations/{searches[query]['span_id']}",
                json={"kind": "relevance", "value": value},
            )
            assert response.status_code == 200
        await _until(client, _searches_path(tenant), _annotated(2))
        built = await client.post(
            f"/admin/tenant/{tenant}/golden-dataset",
            json={"min_rating": 0.8, "lookback_days": 1},
        )
        everything = await client.post(
            f"/admin/tenant/{tenant}/golden-dataset",
            json={"min_rating": 0.5, "lookback_days": 1},
        )

    sunset_keys = [source_title_key(title) for title in SUNSET_TITLES[:5]]
    assert built.status_code == 200
    body = built.json()
    assert list(body["dataset"]) == [SUNSET]
    entry = body["dataset"][SUNSET]
    assert {key: entry[key] for key in entry if key != "timestamp"} == {
        "expected_videos": sunset_keys,
        "relevance_scores": {
            key: 1.0 / (rank + 1) for rank, key in enumerate(sunset_keys)
        },
        "avg_relevance": 1.0,
        "profile": "video_colpali",
    }
    assert entry["timestamp"][:10] == datetime.now(timezone.utc).date().isoformat()
    assert body["untitled_results"] == 0
    assert sorted(everything.json()["dataset"]) == sorted([SUNSET, HARBOUR])
    assert everything.json()["dataset"][HARBOUR]["expected_videos"] == [
        source_title_key(title) for title in HARBOUR_TITLES
    ]


async def test_a_dataset_uploaded_as_csv_is_the_tenants_and_trains_profiles(
    telemetry, app
):
    tenant, other = _tenant("dataset"), _tenant("datasetother")
    csv = "query,expected_videos,category\nsunset over the sea,sunset_reel,nature\nboats,harbour_tour,travel\n"
    async with _client(app) as client:
        created = await client.post(
            f"/admin/tenant/{tenant}/datasets",
            data={"name": "golden_eval"},
            files={"file": ("golden.csv", io.BytesIO(csv.encode()), "text/csv")},
        )
        await client.post(
            f"/admin/tenant/{other}/datasets",
            data={"name": "golden_eval"},
            files={"file": ("golden.csv", io.BytesIO(csv.encode()), "text/csv")},
        )
        listed = await client.get(f"/admin/tenant/{tenant}/datasets")
        malformed = await client.post(
            f"/admin/tenant/{tenant}/datasets",
            data={"name": "broken"},
            files={
                "file": ("broken.csv", io.BytesIO(b"title\nno query\n"), "text/csv")
            },
        )

    name = f"golden_eval-{tenant}"
    assert created.status_code == 200
    assert {k: v for k, v in created.json().items() if k != "dataset_id"} == {
        "name": name,
        "examples": 2,
    }
    assert [(d["name"], d["examples"]) for d in listed.json()["datasets"]] == [
        (name, 2)
    ]
    assert (malformed.status_code, malformed.json()) == (
        400,
        {"detail": "The CSV cannot become a dataset: CSV must have 'query' column"},
    )
    rows = await _dataset_profile_ground_truth_rows(
        telemetry.get_provider(tenant_id=tenant), tenant, name
    )
    assert [(row["query"], row["expected_videos"]) for row in rows] == [
        ("sunset over the sea", ["sunset_reel"]),
        ("boats", ["harbour_tour"]),
    ]


def _temporal_query(index):
    return f"what happened before the goal in match number {index}"


def _plain_query(index):
    return f"red car {index}"


def _seed_training_searches(tenant, temporal_profile, plain_profile, count=20):
    for index in range(count):
        record_search(tenant, _temporal_query(index), temporal_profile, "hybrid", ["A"])
        record_search(tenant, _plain_query(index), plain_profile, "hybrid", ["B"])


async def test_the_recommender_trains_stores_and_predicts(telemetry, app):
    tenant = _tenant("recommender")
    _seed_training_searches(tenant, "video_colpali_temporal", "video_xclip_plain")
    telemetry.force_flush(timeout_millis=10000)
    async with _client(app) as client:
        await _until(client, _searches_path(tenant), _searches(40))
        before = await client.get(f"/admin/tenant/{tenant}/profile-selection/model")
        unpredicted = await client.post(
            f"/admin/tenant/{tenant}/profile-selection/predict",
            json={"query": "anything"},
        )
        analysis = await client.get(
            f"/admin/tenant/{tenant}/profile-selection/analysis?lookback_days=1"
        )
        trained = await client.post(
            f"/admin/tenant/{tenant}/profile-selection/train",
            json={"lookback_days": 1},
        )
        after = await client.get(f"/admin/tenant/{tenant}/profile-selection/model")
        temporal = await client.post(
            f"/admin/tenant/{tenant}/profile-selection/predict",
            json={"query": _temporal_query(99)},
        )
        plain = await client.post(
            f"/admin/tenant/{tenant}/profile-selection/predict",
            json={"query": _plain_query(99)},
        )

    assert before.json() == {"trained": False, "profiles": []}
    assert (unpredicted.status_code, unpredicted.json()) == (
        404,
        {"detail": f"Tenant {tenant} has no trained profile recommender."},
    )
    usage = analysis.json()
    assert usage["search_spans"] == 40
    assert usage["profile_usage"] == {
        "attributes.profile": {"video_colpali_temporal": 20, "video_xclip_plain": 20}
    }
    assert [q["column"] for q in usage["quality"]] == ["attributes.top_score"]
    assert usage["quality"][0]["statistics"]["count"] == 40.0
    assert usage["profile_quality"] == [
        {
            "profile_column": "attributes.profile",
            "quality_column": "attributes.top_score",
            "rows": [
                {"profile": "video_colpali_temporal", "mean": 1.0, "count": 20},
                {"profile": "video_xclip_plain", "mean": 1.0, "count": 20},
            ],
        }
    ]
    assert trained.status_code == 200, trained.text
    model = trained.json()
    assert {k: model[k] for k in ("train_accuracy", "test_accuracy", "samples")} == {
        "train_accuracy": 1.0,
        "test_accuracy": 1.0,
        "samples": 40,
    }
    assert (model["features"], model["profiles"]) == (
        6,
        ["video_colpali_temporal", "video_xclip_plain"],
    )
    importances = [entry["importance"] for entry in model["feature_importance"]]
    assert sorted(e["feature"] for e in model["feature_importance"]) == sorted(
        FEATURE_NAMES
    )
    assert importances == sorted(importances, reverse=True)
    assert after.json() == {
        "trained": True,
        "profiles": ["video_colpali_temporal", "video_xclip_plain"],
    }
    query = _temporal_query(99)
    assert temporal.json()["profile"] == "video_colpali_temporal"
    assert temporal.json()["features"] == {
        "query_length": float(len(query)),
        "word_count": 9.0,
        "has_temporal_keywords": 1.0,
        "has_spatial_keywords": 0.0,
        "has_object_keywords": 1.0,
        "avg_word_length": sum(len(w) for w in query.split()) / 9,
    }
    assert plain.json()["profile"] == "video_xclip_plain"


async def test_recommenders_trained_at_once_learn_only_their_tenants_spans(
    telemetry, app, phoenix_proxy
):
    tenants = [_tenant("trainconcurrent"), _tenant("trainconcurrent")]
    profiles = {
        tenants[0]: ("first_temporal", "first_plain"),
        tenants[1]: ("second_temporal", "second_plain"),
    }
    for tenant, (temporal, plain) in profiles.items():
        _seed_training_searches(tenant, temporal, plain, count=10)
    telemetry.force_flush(timeout_millis=10000)
    async with _client(app) as client:
        for tenant in tenants:
            await _until(client, _searches_path(tenant), _searches(20))
        barrier = threading.Barrier(2, timeout=60)
        held = []

        def hold_training_reads(method, path, body):
            # Both trainings read their spans before either is answered.
            if "spans" in path and "annotations" not in path and len(held) < 2:
                held.append(path)
                barrier.wait()
            return None

        phoenix_proxy.intercept = hold_training_reads
        trained = await asyncio.gather(
            *(
                client.post(
                    f"/admin/tenant/{tenant}/profile-selection/train",
                    json={"lookback_days": 1},
                )
                for tenant in tenants
            )
        )
        phoenix_proxy.intercept = None
        stored = [
            (await client.get(f"/admin/tenant/{tenant}/profile-selection/model")).json()
            for tenant in tenants
        ]
    assert len(held) == 2
    assert [r.json()["profiles"] for r in trained] == [
        sorted(profiles[tenant]) for tenant in tenants
    ]
    assert stored == [
        {"trained": True, "profiles": sorted(profiles[tenant])} for tenant in tenants
    ]


async def test_too_few_labelled_searches_cannot_train(telemetry, app):
    tenant = _tenant("fewsearches")
    _seed_training_searches(tenant, "temporal", "plain", count=2)
    telemetry.force_flush(timeout_millis=10000)
    async with _client(app) as client:
        await _until(client, _searches_path(tenant), _searches(4))
        response = await client.post(
            f"/admin/tenant/{tenant}/profile-selection/train",
            json={"lookback_days": 1},
        )
    assert (response.status_code, response.json()) == (
        422,
        {
            "detail": "Cannot train the recommender: Insufficient training data: "
            "4 samples (need 10)"
        },
    )


async def test_optimization_metrics_score_routing_per_agent(telemetry, app):
    tenant = _tenant("metrics")
    # The gateway records no processing_time: each decision is timed by its
    # span, 200, 300, 400, 500 and 100 ms.
    for minutes, duration_ms in ((1, 200), (2, 300), (3, 400)):
        record_routing(
            telemetry, tenant, "search_agent", 0.9, duration_ms, minutes_ago=minutes
        )
    record_routing(
        telemetry, tenant, "search_agent", 0.3, 500, minutes_ago=4, failed=True
    )
    record_routing(
        telemetry,
        tenant,
        "summarizer_agent",
        0.3,
        100,
        minutes_ago=5,
        within_request=False,
    )
    record_trace(telemetry, tenant, "evaluation.run", 10, minutes_ago=6)
    record_trace(telemetry, tenant, "cogniverse.optimization", 10, minutes_ago=7)
    record_trace(telemetry, tenant, "cogniverse.optimization", 10, minutes_ago=8)
    telemetry.force_flush(timeout_millis=10000)
    path = f"/admin/tenant/{tenant}/optimization-metrics?lookback_days=7"
    async with _client(app) as client:
        body = await _until(
            client,
            path,
            lambda b: b["routing"] is not None and b["routing"]["total_decisions"] == 5,
        )
        bad_window = await client.get(
            f"/admin/tenant/{tenant}/optimization-metrics?lookback_days=91"
        )
    precision = 3 / 4
    assert body["routing"] == {
        "accuracy": 3 / 5,
        "total_decisions": 5,
        "avg_latency_ms": 300.0,
        "confidence_calibration": 1.0,
        "per_agent": [
            {
                "agent": "search_agent",
                "precision": precision,
                "recall": 1.0,
                "f1": 2 * precision / (precision + 1.0),
            },
            {"agent": "summarizer_agent", "precision": 0.0, "recall": 0.0, "f1": 0.0},
        ],
    }
    # Each recorded root also writes a child step span.
    assert body["evaluation"] == {"spans": 2, "queries": 0}
    assert body["training"] == [
        {"date": datetime.now(timezone.utc).date().isoformat(), "runs": 4}
    ]
    assert bad_window.status_code == 422


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/search-annotations?lookback_hours=1", None),
        ("GET", "/search-annotations/count", None),
        (
            "POST",
            "/search-annotations/0123456789abcdef",
            {"kind": "thumbs", "value": 1},
        ),
        ("POST", "/golden-dataset", {"min_rating": 0.8, "lookback_days": 1}),
        ("GET", "/profile-selection/analysis", None),
        ("GET", "/optimization-metrics", None),
    ],
)
async def test_an_unreadable_telemetry_backend_answers_502(
    app, phoenix_proxy, method, path, payload
):
    tenant = _tenant("outage")
    phoenix_proxy.intercept = lambda m, p, body: (503, {"detail": "down"})
    async with _client(app) as client:
        response = await client.request(
            method, f"/admin/tenant/{tenant}{path}", json=payload
        )
    body = response.json()
    assert (response.status_code, body["detail"]["error"]) == (
        502,
        "telemetry_unavailable",
    )
    assert body["detail"]["message"].endswith(f"of tenant {tenant}.")


async def test_unreachable_dataset_and_model_stores_answer_502(app, phoenix_proxy):
    tenant = _tenant("storeoutage")
    phoenix_proxy.intercept = lambda m, p, body: (503, {"detail": "down"})
    async with _client(app) as client:
        datasets = await client.get(f"/admin/tenant/{tenant}/datasets")
        model = await client.get(f"/admin/tenant/{tenant}/profile-selection/model")
        upload = await client.post(
            f"/admin/tenant/{tenant}/datasets",
            data={"name": "golden"},
            files={"file": ("g.csv", io.BytesIO(b"query\nq\n"), "text/csv")},
        )
    assert [
        (r.status_code, r.json()["detail"]["error"]) for r in (datasets, model, upload)
    ] == [
        (502, "telemetry_unavailable"),
        (502, "artifact_store_unavailable"),
        (502, "dataset_store_unavailable"),
    ]


async def test_a_telemetry_read_that_does_not_answer_in_time_says_slow_not_empty(
    app, phoenix_proxy, monkeypatch
):
    tenant = _tenant("slow")
    monkeypatch.setattr(optimization_framework, "TELEMETRY_READ_TIMEOUT_S", 1.0)

    def hold(method, path, body):
        time.sleep(3)
        return None

    phoenix_proxy.intercept = hold
    async with _client(app) as client:
        response = await client.get(_searches_path(tenant))
    assert (response.status_code, response.json()["detail"]["message"]) == (
        504,
        f"Telemetry did not return the recorded searches of tenant {tenant} "
        "within 1s; the store is slow, not empty.",
    )
