"""The evaluation-dataset and telemetry-window routes against real Phoenix.

Datasets are written through the production dataset store (and
``DatasetManager``, the eval CLI's writer), which records the owning tenant.
Spans are recorded by their producers' writers with fixed values. The runtime
reads Phoenix through a forwarding proxy, so a test can fail or slow a read.
"""

from __future__ import annotations

import asyncio
import threading
import time
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_evaluation.data.datasets import INPUT_KEYS, OUTPUT_KEYS, DatasetManager
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_PROFILE_SELECTION,
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.optimization_cli import emit_ab_compare_span
from cogniverse_runtime.routers import telemetry_metrics
from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    ab_result,
    record_ab_compare,
    record_profile_selection,
    record_search,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]

SUNSET, RED_CAR, DOG = "sunset over the sea", "a red car", "dog on a beach"
QUERIES = [
    {"query": SUNSET, "expected_videos": ["sunset"]},
    {"query": RED_CAR, "expected_videos": ["red_car", "garage"]},
    {"query": DOG, "expected_videos": ["dog"]},
]


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
    app.include_router(telemetry_metrics.router, prefix="/admin/tenant")
    yield app
    phoenix_proxy.intercept = None


@pytest.fixture()
def store(phoenix_container):
    return PhoenixDatasetStore(http_endpoint=phoenix_container["http_endpoint"])


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://runtime",
        timeout=300,
    )


def _tenant(prefix="evalds"):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _frame(queries):
    return pd.DataFrame(
        [
            {
                "query": q["query"],
                "category": "general",
                "expected_videos": ",".join(q["expected_videos"]),
            }
            for q in queries
        ]
    )


async def _create(store, tenant, name, queries=QUERIES):
    return await store.create_dataset(
        name,
        _frame(queries),
        {"input_keys": INPUT_KEYS, "output_keys": OUTPUT_KEYS, "tenant_id": tenant},
    )


async def _summary(store, dataset_id):
    return next(d for d in await store.describe_datasets() if d.id == dataset_id)


def _listed(summary):
    return {
        "id": summary.id,
        "name": summary.name,
        "example_count": summary.example_count,
        "created_at": summary.created_at.isoformat(),
        "description": summary.description,
    }


class TestDatasets:
    async def test_a_tenant_lists_only_its_own_datasets_newest_first(
        self, app, store, monkeypatch
    ):
        tenant, other = _tenant(), _tenant()
        suffix = uuid4().hex[:8]
        older = await _create(store, tenant, f"golden-old-{suffix}", QUERIES[:1])
        # Written the way the eval CLI writes it.
        newer = await asyncio.to_thread(
            DatasetManager(tenant_id=tenant, dataset_store=store).create_from_queries,
            QUERIES,
            f"golden-new-{suffix}",
            "three queries",
        )
        await _create(store, other, f"golden-other-{suffix}")
        await store.create_dataset(
            f"unowned-{suffix}",
            _frame(QUERIES),
            {"input_keys": INPUT_KEYS, "output_keys": OUTPUT_KEYS},
        )
        monkeypatch.setenv("PHOENIX_UI_URL", "http://phoenix.example:6006/")
        async with _client(app) as client:
            response = await client.get(f"/admin/tenant/{tenant}/evaluation/datasets")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "phoenix_url": "http://phoenix.example:6006",
            "datasets": [
                _listed(await _summary(store, newer)),
                _listed(await _summary(store, older)),
            ],
        }
        assert [d["example_count"] for d in response.json()["datasets"]] == [3, 1]
        assert response.json()["datasets"][0]["description"] == "three queries"

    async def test_the_tenants_optimization_artifacts_are_not_evaluation_datasets(
        self, app, store, telemetry
    ):
        """The artifact datasets a golden-set upload and the optimizers write
        for the tenant, newer than its evaluation dataset, are left out: the
        newest dataset listed is the evaluation dataset."""
        tenant, other = _tenant(), _tenant()
        evaluation = await _create(store, tenant, f"golden-{uuid4().hex[:8]}")
        artifacts = ArtifactManager(telemetry.get_provider(tenant_id=tenant), tenant)
        await artifacts.save_blob("config", "golden_set_ground_truth", "[]")
        await artifacts.save_blob("model", "profile_selection", "{}")
        await artifacts.save_prompts("query_enhancement", {"system": "Expand."})
        await artifacts.save_prompts_versioned(
            "query_enhancement", {"system": "Expand."}
        )
        await ArtifactManager(telemetry.get_provider(tenant_id=other), other).save_blob(
            "config", "golden_set_ground_truth", "[]"
        )
        written = [
            d.name
            for d in await store.describe_datasets()
            if d.tenant_id == tenant and d.id != evaluation
        ]
        assert sorted(written) == sorted(
            [
                f"dspy-config-{tenant}-golden_set_ground_truth--r1",
                f"dspy-model-{tenant}-profile_selection--r1",
                f"dspy-prompts-{tenant}-query_enhancement",
                f"dspy-prompts-{tenant}-query_enhancement-v1",
            ]
        )
        artifact = next(
            d
            for d in await store.describe_datasets()
            if d.name == f"dspy-prompts-{tenant}-query_enhancement"
        )

        async with _client(app) as client:
            listing = await client.get(f"/admin/tenant/{tenant}/evaluation/datasets")
            scored = await client.get(
                f"/admin/tenant/{tenant}/evaluation/dataset",
                params={"dataset_id": artifact.id},
            )
        assert listing.status_code == 200, listing.text
        assert listing.json()["datasets"] == [
            _listed(await _summary(store, evaluation))
        ]
        assert (scored.status_code, scored.json()["detail"]) == (
            404,
            f"Tenant {tenant} has no dataset {artifact.id}.",
        )

    async def test_without_a_phoenix_ui_address_no_link_is_offered(
        self, app, store, monkeypatch
    ):
        tenant = _tenant()
        await _create(store, tenant, f"golden-{uuid4().hex[:8]}")
        monkeypatch.delenv("PHOENIX_UI_URL", raising=False)
        async with _client(app) as client:
            response = await client.get(f"/admin/tenant/{tenant}/evaluation/datasets")
        assert response.json()["phoenix_url"] is None

    async def test_a_dataset_scores_the_tenants_searches_of_its_queries(
        self, app, store, telemetry
    ):
        tenant = _tenant()
        dataset_id = await _create(store, tenant, f"golden-{uuid4().hex[:8]}")
        record_search(
            tenant, SUNSET, "video_colpali", "hybrid", ["beach.mp4", "sunset.mp4"]
        )
        record_search(
            tenant,
            RED_CAR,
            "video_colpali",
            "hybrid",
            ["red_car.mp4", "beach.mp4", "garage.mov", "a.mp4", "b.mp4", "c.mp4"],
        )
        record_search(tenant, SUNSET, "video_colpali", "bm25", ["sunset.mp4"])
        record_search(tenant, RED_CAR, "audio", "bm25", [], error="backend down")
        telemetry.force_flush(timeout_millis=10000)

        path = (
            f"/admin/tenant/{tenant}/evaluation/dataset"
            f"?dataset_id={dataset_id}&lookback_hours=2"
        )
        async with _client(app) as client:
            deadline = time.monotonic() + 90
            while True:
                response = await client.get(path)
                assert response.status_code == 200, response.text
                body = response.json()
                if (len(body["queries"]), body["failed_searches"]) == (3, 1):
                    break
                assert time.monotonic() < deadline, body
                await asyncio.sleep(2)
        assert body["dataset"] == _listed(await _summary(store, dataset_id))
        assert body["golden_queries"] == 3
        assert body["unsearched_queries"] == [DOG]
        assert body["unscored_searches"] == 0
        assert [
            (
                s["profile"],
                s["strategy"],
                s["queries"],
                round(s["mrr"], 3),
                round(s["recall_at_1"], 3),
                round(s["recall_at_5"], 3),
                s["success_rate"],
            )
            for s in body["strategies"]
        ] == [
            ("video_colpali", "bm25", 1, 1.0, 1.0, 1.0, 1.0),
            ("video_colpali", "hybrid", 2, 0.75, 0.25, 1.0, 0.5),
        ]

    async def test_another_tenants_dataset_is_not_found(self, app, store):
        owner, intruder = _tenant(), _tenant()
        dataset_id = await _create(store, owner, f"golden-{uuid4().hex[:8]}")
        async with _client(app) as client:
            response = await client.get(
                f"/admin/tenant/{intruder}/evaluation/dataset",
                params={"dataset_id": dataset_id},
            )
        assert (response.status_code, response.json()["detail"]) == (
            404,
            f"Tenant {intruder} has no dataset {dataset_id}.",
        )

    async def test_concurrent_listings_of_two_tenants_keep_each_tenants_datasets(
        self, app, store
    ):
        first, second = _tenant(), _tenant()
        names = {
            first: {f"first-{uuid4().hex[:8]}" for _ in range(2)},
            second: {f"second-{uuid4().hex[:8]}"},
        }
        for tenant, owned in names.items():
            for name in owned:
                await _create(store, tenant, name)
        barrier = threading.Barrier(6)

        async with _client(app) as client:

            async def listing(tenant):
                await asyncio.to_thread(barrier.wait)
                response = await client.get(
                    f"/admin/tenant/{tenant}/evaluation/datasets"
                )
                return tenant, {d["name"] for d in response.json()["datasets"]}

            results = await asyncio.gather(*(listing(t) for t in [first, second] * 3))
        assert [owned == names[tenant] for tenant, owned in results] == [True] * 6

    async def test_an_unreadable_dataset_store_is_an_error_not_no_datasets(
        self, app, store, phoenix_proxy
    ):
        tenant = _tenant()
        await _create(store, tenant, f"golden-{uuid4().hex[:8]}")
        phoenix_proxy.intercept = lambda method, path, body: (
            (503, {"detail": "down"}) if path.startswith("/v1/datasets") else None
        )
        async with _client(app) as client:
            response = await client.get(f"/admin/tenant/{tenant}/evaluation/datasets")
        assert (response.status_code, response.json()["detail"]) == (
            502,
            {
                "error": "dataset_store_unavailable",
                "message": f"Could not list the datasets of tenant {tenant}.",
                "failure": "DatasetStoreUnavailableError",
                "tenant_id": tenant,
            },
        )


class TestTelemetryWindow:
    async def test_profile_selection_names_its_project_and_counts_every_span(
        self, app, telemetry
    ):
        tenant = _tenant("windowprofile")
        record_profile_selection(telemetry, tenant, "video", 100)
        record_profile_selection(telemetry, tenant, None, 100)
        telemetry.force_flush(timeout_millis=10000)
        path = f"/admin/tenant/{tenant}/telemetry/profile-selection?lookback_hours=720"
        async with _client(app) as client:
            deadline = time.monotonic() + 90
            while True:
                body = (await client.get(path)).json()
                if body["spans"] == 2 or time.monotonic() > deadline:
                    break
                await asyncio.sleep(2)
        assert body == {
            "project": telemetry.config.get_project_name(tenant),
            "spans": 2,
            "modalities": [
                {
                    "modality": "video",
                    "count": 1,
                    "p50_ms": 100.0,
                    "p95_ms": 100.0,
                    "p99_ms": 100.0,
                    "success_rate": 1.0,
                }
            ],
        }

    async def test_rlm_ab_reads_sub_hour_and_month_long_windows(self, app, telemetry):
        tenant = _tenant("windowrlm")
        tracer = telemetry._get_tracer_for_project(tenant, None)
        emit_ab_compare_span(
            tracer, ab_result("now", "q now", 10.0, 1, 0.1, False), tenant, "ds"
        )
        record_ab_compare(
            telemetry,
            tenant,
            ab_result("half-hour", "q half hour", 20.0, 2, 0.2, False),
            "ds",
            minutes_ago=30,
        )
        record_ab_compare(
            telemetry,
            tenant,
            ab_result("ten-days", "q ten days", 30.0, 3, 0.3, True),
            "ds",
            minutes_ago=240 * 60,
        )
        telemetry.force_flush(timeout_millis=10000)

        async def ab_ids(client, hours):
            response = await client.get(
                f"/admin/tenant/{tenant}/telemetry/rlm-ab",
                params={"lookback_hours": hours},
            )
            assert response.status_code == 200, response.text
            return [row["ab_id"] for row in response.json()["comparisons"]]

        async with _client(app) as client:
            deadline = time.monotonic() + 90
            while len(await ab_ids(client, 720)) < 3 and time.monotonic() < deadline:
                await asyncio.sleep(2)
            assert await ab_ids(client, 720) == ["now", "half-hour", "ten-days"]
            assert await ab_ids(client, 239) == ["now", "half-hour"]
            assert await ab_ids(client, 0.25) == ["now"]
            refused = await client.get(
                f"/admin/tenant/{tenant}/telemetry/rlm-ab",
                params={"lookback_hours": 0.05},
            )
        assert refused.status_code == 422
        assert refused.json()["detail"][0]["msg"] == (
            "Input should be greater than or equal to 0.1"
        )

    async def test_a_tenant_without_a_telemetry_provider_is_told_so(
        self, app, telemetry, monkeypatch
    ):
        tenant = _tenant("windowunconfigured")
        monkeypatch.setattr(telemetry.config, "provider", "absent")
        async with _client(app) as client:
            profile = await client.get(
                f"/admin/tenant/{tenant}/telemetry/profile-selection"
            )
            rlm = await client.get(f"/admin/tenant/{tenant}/telemetry/rlm-ab")
        assert [(r.status_code, r.json()["detail"]) for r in (profile, rlm)] == [
            (
                503,
                {
                    "error": "telemetry_unconfigured",
                    "message": "No telemetry provider could be built for tenant "
                    f"{tenant}, so the {SPAN_NAME_PROFILE_SELECTION} spans cannot "
                    "be read; the runtime log names the cause.",
                    "failure": "ValueError",
                    "tenant_id": tenant,
                },
            ),
            (
                503,
                {
                    "error": "telemetry_unconfigured",
                    "message": "No telemetry provider could be built for tenant "
                    f"{tenant}, so the rlm.ab_compare spans cannot be read; the "
                    "runtime log names the cause.",
                    "failure": "ValueError",
                    "tenant_id": tenant,
                },
            ),
        ]

    async def test_a_slow_store_is_reported_as_slow_not_as_failed(
        self, app, phoenix_proxy, monkeypatch
    ):
        tenant = _tenant("windowslow")
        monkeypatch.setattr(telemetry_metrics, "SPAN_READ_BUDGET_S", 0.5)

        def slow(method, path, body):
            if "/spans" in path:
                time.sleep(2)
            return None

        phoenix_proxy.intercept = slow
        async with _client(app) as client:
            response = await client.get(
                f"/admin/tenant/{tenant}/telemetry/profile-selection"
            )
        assert (response.status_code, response.json()["detail"]) == (
            504,
            {
                "error": "telemetry_slow",
                "message": "The telemetry store did not return the "
                f"{SPAN_NAME_PROFILE_SELECTION} spans of tenant {tenant} within "
                "0.5 s. It is slow, not empty; retry shortly.",
                "failure": "TimeoutError",
                "tenant_id": tenant,
            },
        )

    async def test_concurrent_slow_reads_each_answer_slow(
        self, app, phoenix_proxy, monkeypatch
    ):
        """Every request held past the budget answers 504 on its own; one
        held read neither blocks nor answers for another."""
        monkeypatch.setattr(telemetry_metrics, "SPAN_READ_BUDGET_S", 0.5)
        held = []

        def slow(method, path, body):
            if "/spans" in path:
                held.append(path)
                time.sleep(2)
            return None

        phoenix_proxy.intercept = slow
        tenants = [_tenant("windowslowc") for _ in range(4)]
        async with _client(app) as client:
            started = time.monotonic()
            responses = await asyncio.gather(
                *(client.get(f"/admin/tenant/{t}/telemetry/rlm-ab") for t in tenants)
            )
            elapsed = time.monotonic() - started
        assert [
            (r.status_code, r.json()["detail"]["tenant_id"]) for r in responses
        ] == [(504, t) for t in tenants]
        assert elapsed < 1.9
        assert len(held) >= 4
