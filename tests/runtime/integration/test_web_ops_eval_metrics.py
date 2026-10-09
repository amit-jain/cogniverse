"""The web client's Evaluation (golden set and Phoenix datasets), Profile
metrics and RLM A/B views, driven in Chromium against real Phoenix.

Datasets are written through the production dataset store, which records the
owning tenant; spans by their producers' writers with fixed values. The
runtime reads Phoenix through a forwarding proxy, so a test can fail or slow
a read.
"""

from __future__ import annotations

import re
import time
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_evaluation.data.datasets import INPUT_KEYS, OUTPUT_KEYS
from cogniverse_foundation.telemetry.config import (
    SPAN_NAME_PROFILE_SELECTION,
    BatchExportConfig,
    TelemetryConfig,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_foundation.telemetry.span_metrics import AB_COMPARE_SPAN_NAME
from cogniverse_runtime.routers import admin, telemetry_metrics
from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore
from tests.utils.approval_review import review_config_manager, run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    ab_result,
    record_ab_compare,
    record_profile_selection,
    record_search,
)
from tests.utils.web_client import recording_telemetry_sink, serve_web
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

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


@pytest.fixture(scope="module")
def runtime_url(phoenix_container, schema_loader, workflow_state_redis_url, telemetry):
    with serve_ops_runtime(
        review_config_manager(phoenix_container, workflow_state_redis_url),
        schema_loader,
        workflow_state_redis_url,
    ) as url:
        yield url


@pytest.fixture()
def web_url(built_client, runtime_url, phoenix_proxy):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []
    phoenix_proxy.intercept = None


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture()
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


@pytest.fixture()
def runtime_reads(page):
    """Every runtime path the page asks the web server for, in order."""
    paths: list[str] = []
    page.on(
        "request",
        lambda request: (
            paths.append(request.url.split("/ui-api/runtime", 1)[1])
            if "/ui-api/runtime/" in request.url
            else None
        ),
    )
    return paths


@pytest.fixture()
def store(phoenix_container):
    return PhoenixDatasetStore(http_endpoint=phoenix_container["http_endpoint"])


def _tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _show(page: Page, web_url: str, view: str, heading: str, action: str, tenant):
    page.goto(f"{web_url}/#/ops/{view}")
    expect(page.get_by_role("heading", name=heading, level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name=action).click()
    # A choice becomes the active tenant once the runtime's registry has
    # answered; the view starts over for it then.
    expect(
        page.get_by_role("region", name="Active tenant").locator("p.current-tenant")
    ).to_have_text(f"Current tenant: {tenant}")


def _region(page: Page, name: str):
    return page.get_by_role("region", name=name, exact=True)


def _facts(container):
    return dict(
        zip(
            container.locator("dt").all_inner_texts(),
            container.locator("dd").all_inner_texts(),
            strict=True,
        )
    )


def _rows(container, table: str):
    return [
        row.locator("td").all_inner_texts()
        for row in container.get_by_role("table", name=table, exact=True)
        .locator("tbody tr")
        .all()
    ]


def _lookback(container, hours):
    form = container.get_by_role("form", name="Lookback")
    form.get_by_label("Lookback (hours)").fill(str(hours))
    form.get_by_role("button", name="Apply").click()


def _plot(page: Page, title: str):
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    expect(area).to_have_class(re.compile(r"\bjs-plotly-plot\b"))
    return area.evaluate(
        "el => el.data.map(t => ({type: t.type, labels: t.labels, values: t.values}))"
    )


def _create_dataset(store, tenant, name, queries=QUERIES):
    frame = pd.DataFrame(
        [
            {
                "query": q["query"],
                "category": "general",
                "expected_videos": ",".join(q["expected_videos"]),
            }
            for q in queries
        ]
    )
    dataset_id = run_in_own_loop(
        store.create_dataset(
            name,
            frame,
            {"input_keys": INPUT_KEYS, "output_keys": OUTPUT_KEYS, "tenant_id": tenant},
        )
    )
    summaries = run_in_own_loop(store.describe_datasets())
    return next(s for s in summaries if s.id == dataset_id)


def _until(page: Page, region, done, timeout=90.0):
    """Refresh ``region`` until ``done()`` holds (Phoenix serves spans after
    a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, page.locator("main").inner_text()
        region.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)


class TestPhoenixDatasets:
    def test_a_tenants_dataset_is_chosen_linked_and_scored(
        self, page, web_url, store, telemetry, monkeypatch, runtime_reads
    ):
        tenant, other = _tenant("webds"), _tenant("webdsother")
        suffix = uuid4().hex[:8]
        older = _create_dataset(store, tenant, f"older-{suffix}", QUERIES[:1])
        time.sleep(1.1)
        newer = _create_dataset(store, tenant, f"newer-{suffix}")
        _create_dataset(store, other, f"other-{suffix}")
        monkeypatch.setenv("PHOENIX_UI_URL", "http://phoenix.example:6006")
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
        record_search(tenant, SUNSET, "audio", "bm25", ["x.mp4", "y.mp4"])
        telemetry.force_flush(timeout_millis=10000)

        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
        page.get_by_role("tablist", name="Evaluate against").get_by_role(
            "tab", name="Phoenix datasets"
        ).click()
        listing = _region(page, f"Phoenix datasets of {tenant}")
        dataset = listing.get_by_role("combobox", name="Dataset")
        expect(dataset.locator("option")).to_have_text(
            [f"newer-{suffix} (3 examples)", f"older-{suffix} (1 examples)"]
        )
        assert _facts(listing.locator('dl[aria-label="Dataset details"]')) == {
            "Dataset examples": "3",
            "Created": newer.created_at.isoformat().split("T")[0],
        }
        expect(listing.get_by_role("link", name="View in Phoenix")).to_have_attribute(
            "href", f"http://phoenix.example:6006/datasets/{newer.id}"
        )

        scored = _region(page, f"Evaluation of newer-{suffix}")
        scores = page.get_by_role("table", name="Scores by profile and strategy")
        _until(
            page,
            scored,
            lambda: len(_rows(page, "Scores by profile and strategy")) == 3,
        )
        expect(scores).to_be_visible()
        assert [row[:3] for row in _rows(page, "Scores by profile and strategy")] == [
            ["audio", "bm25", "1"],
            ["video_colpali", "bm25", "1"],
            ["video_colpali", "hybrid", "2"],
        ]
        results = _region(page, "Query results")
        profiles = results.get_by_role("tablist", name="Profiles")
        expect(profiles.get_by_role("tab")).to_have_text(["audio", "video_colpali"])
        profiles.get_by_role("tab", name="video_colpali").click()
        strategies = results.get_by_role("tablist", name="Strategies of video_colpali")
        expect(strategies.get_by_role("tab")).to_have_text(["bm25", "hybrid"])
        strategies.get_by_role("tab", name="hybrid").click()
        assert _facts(
            results.locator('dl[aria-label="Summary of video_colpali / hybrid"]')
        ) == {
            "MRR": "75.0%",
            "Recall@1": "25.0%",
            "Recall@5": "100.0%",
            "Queries": "2",
        }
        badges = results.get_by_role("table", name="Query results").locator(
            "tbody tr td.score"
        )
        expect(badges).to_have_text(
            ["0.500", "0.000", "1.000", "1.000", "0.500", "1.000"]
        )
        assert [badge.get_attribute("class") for badge in badges.all()] == [
            "score fair",
            "score poor",
            "score good",
            "score good",
            "score fair",
            "score good",
        ]
        profiles.get_by_role("tab", name="audio").click()
        assert _rows(results, "Query results")[0][:6] == [
            SUNSET,
            "sunset",
            "✗ x\n✗ y",
            "0.000",
            "0.000",
            "0.000",
        ]

        _lookback(scored, 5)
        expect(
            scored.get_by_role("form", name="Lookback").get_by_role("alert")
        ).to_have_count(0)
        expect(
            page.get_by_role("table", name="Scores by profile and strategy")
        ).to_be_visible()
        assert (
            f"/admin/tenant/{tenant}/evaluation/dataset?dataset_id="
            f"{newer.id}&lookback_hours=5"
        ) in [p.replace("%3A", ":").replace("%3D", "=") for p in runtime_reads]
        _lookback(scored, 2161)
        expect(
            scored.get_by_role("form", name="Lookback").get_by_role("alert")
        ).to_have_text("Lookback must be a number of hours from 1 to 2160.")

        dataset.select_option(label=f"older-{suffix} (1 examples)")
        assert _facts(listing.locator('dl[aria-label="Dataset details"]')) == {
            "Dataset examples": "1",
            "Created": older.created_at.isoformat().split("T")[0],
        }
        older_scores = _region(page, f"Evaluation of older-{suffix}")
        expect(
            older_scores.locator('dl[aria-label="Evaluation summary"] dd')
        ).to_have_text(["1", "1", "0", "0", "0"])

    def test_the_tenants_optimization_artifacts_are_not_offered(
        self, page, web_url, store, telemetry
    ):
        """Artifact datasets written after the evaluation dataset (a golden-set
        upload, an optimizer's prompts) are not listed, so the evaluation
        dataset is the one chosen."""
        tenant = _tenant("webdsartifacts")
        suffix = uuid4().hex[:8]
        evaluation = _create_dataset(store, tenant, f"eval-{suffix}")
        time.sleep(1.1)
        artifacts = ArtifactManager(telemetry.get_provider(tenant_id=tenant), tenant)
        run_in_own_loop(artifacts.save_blob("config", "golden_set_ground_truth", "[]"))
        run_in_own_loop(
            artifacts.save_prompts("query_enhancement", {"system": "Expand."})
        )
        newest = run_in_own_loop(store.describe_datasets())[0]
        assert newest.name == f"dspy-prompts-{tenant}-query_enhancement"

        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
        page.get_by_role("tablist", name="Evaluate against").get_by_role(
            "tab", name="Phoenix datasets"
        ).click()
        listing = _region(page, f"Phoenix datasets of {tenant}")
        dataset = listing.get_by_role("combobox", name="Dataset")
        expect(dataset.locator("option")).to_have_text([f"eval-{suffix} (3 examples)"])
        expect(dataset).to_have_value(evaluation.id)
        expect(_region(page, f"Evaluation of eval-{suffix}")).to_be_visible()

    def test_a_tenant_without_datasets_is_told_and_no_link_is_offered(
        self, page, web_url, store, monkeypatch
    ):
        tenant = _tenant("webdsnone")
        _create_dataset(store, _tenant("webdsneighbour"), f"n-{uuid4().hex[:8]}")
        monkeypatch.delenv("PHOENIX_UI_URL", raising=False)
        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
        page.get_by_role("tab", name="Phoenix datasets").click()
        listing = _region(page, f"Phoenix datasets of {tenant}")
        expect(listing.locator("p.muted")).to_have_text(
            f"No datasets of tenant {tenant} were found in Phoenix."
        )
        expect(listing.get_by_role("combobox", name="Dataset")).to_have_count(0)

        owner = _tenant("webdsnolink")
        _create_dataset(store, owner, f"l-{uuid4().hex[:8]}")
        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", owner)
        page.get_by_role("tab", name="Phoenix datasets").click()
        listing = _region(page, f"Phoenix datasets of {owner}")
        expect(listing.locator("p.muted")).to_have_text(
            "No Phoenix address is configured, so the dataset is not linked."
        )
        expect(listing.get_by_role("link")).to_have_count(0)

    def test_a_refused_listing_is_an_error_not_no_datasets(
        self, page, web_url, phoenix_proxy
    ):
        tenant = _tenant("webdsdown")
        phoenix_proxy.intercept = lambda method, path, body: (
            (503, {"detail": "down"}) if path.startswith("/v1/datasets") else None
        )
        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
        page.get_by_role("tab", name="Phoenix datasets").click()
        listing = _region(page, f"Phoenix datasets of {tenant}")
        expect(listing.get_by_role("alert")).to_have_text(
            f"Could not list the datasets of tenant {tenant}."
        )
        expect(listing.locator("p.muted")).to_have_count(0)


@pytest.fixture()
def upload_golden(runtime_url, phoenix_container, monkeypatch):
    monkeypatch.setattr(admin, "_phoenix_endpoints", {})
    admin.set_phoenix_endpoints(
        phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
    )

    def upload(tenant):
        response = httpx.put(
            f"{runtime_url}/admin/tenants/{tenant}/golden_set_ground_truth",
            json=QUERIES,
            timeout=60,
        )
        assert (response.status_code, response.json()["row_count"]) == (200, 3)

    return upload


class TestGoldenSet:
    def test_a_window_without_searches_names_the_tenant_and_the_hours(
        self, page, web_url, upload_golden
    ):
        tenant = _tenant("webgoldenquiet")
        upload_golden(tenant)
        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
        panel = _region(page, f"Golden set evaluation of {tenant}")
        expect(panel.locator("p.muted")).to_have_text(
            "No searches of the golden queries were recorded for tenant "
            f"{tenant} in the last 168 hours."
        )
        _lookback(panel, 2160)
        expect(panel.locator("p.muted")).to_have_text(
            "No searches of the golden queries were recorded for tenant "
            f"{tenant} in the last 2160 hours."
        )
        expect(page.get_by_role("region", name="Query results")).to_have_count(0)

    def test_an_evaluation_is_kept_for_a_minute_and_refresh_reads_again(
        self, page, web_url, upload_golden, runtime_reads
    ):
        tenant = _tenant("webgoldencache")
        upload_golden(tenant)
        golden = f"/admin/tenant/{tenant}/evaluation/golden?lookback_hours=168"
        _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
        panel = _region(page, f"Golden set evaluation of {tenant}")
        expect(panel.locator("p.muted")).to_be_visible()
        tabs = page.get_by_role("tablist", name="Evaluate against")
        tabs.get_by_role("tab", name="Phoenix datasets").click()
        expect(_region(page, f"Phoenix datasets of {tenant}")).to_be_visible()
        tabs.get_by_role("tab", name="Golden set").click()
        expect(panel.locator("p.muted")).to_be_visible()
        assert [p.replace("%3A", ":") for p in runtime_reads].count(golden) == 1
        panel.get_by_role("button", name="Refresh").click()
        expect(panel.locator("p.muted")).to_be_visible()
        page.wait_for_timeout(500)
        assert [p.replace("%3A", ":") for p in runtime_reads].count(golden) == 2


class TestProfileMetrics:
    def test_cards_pie_and_a_month_long_window(
        self, page, web_url, telemetry, runtime_reads
    ):
        tenant = _tenant("webpmcards")
        for duration in (100, 200, 300, 400):
            record_profile_selection(telemetry, tenant, "video", duration)
        record_profile_selection(telemetry, tenant, "image", 50)
        telemetry.force_flush(timeout_millis=10000)
        _show(
            page, web_url, "profile-metrics", "Profile metrics", "Show metrics", tenant
        )
        panel = _region(page, f"Profile selections of {tenant}")
        _lookback(panel, 720)
        cards = panel.get_by_role("list", name="Per-modality metrics")
        _until(
            page, panel, lambda: cards.count() == 1 and cards.locator("li").count() == 2
        )
        expect(cards.locator("li")).to_have_text(
            ["VIDEO4P95 385 ms", "IMAGE1P95 50 ms"]
        )
        assert _plot(page, "Queries per modality") == [
            {"type": "pie", "labels": ["video", "image"], "values": [4, 1]}
        ]
        assert (
            f"/admin/tenant/{tenant}/telemetry/profile-selection?lookback_hours=720"
            in [p.replace("%3A", ":") for p in runtime_reads]
        )
        _lookback(panel, 721)
        expect(
            panel.get_by_role("form", name="Lookback").get_by_role("alert")
        ).to_have_text("Lookback must be a number of hours from 1 to 720.")

    def test_an_empty_window_names_the_project_and_the_agent_to_drive(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webpmempty")
        _show(
            page, web_url, "profile-metrics", "Profile metrics", "Show metrics", tenant
        )
        panel = _region(page, f"Profile selections of {tenant}")
        expect(panel.locator("p.muted")).to_have_text(
            f"No {SPAN_NAME_PROFILE_SELECTION} spans in "
            f"{telemetry.config.get_project_name(tenant)} for the last 24 hours. "
            "Drive traffic through profile_selection_agent first."
        )

    def test_selections_without_a_modality_are_told_apart_from_none(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webpmnomodality")
        record_profile_selection(telemetry, tenant, None, 100)
        record_profile_selection(telemetry, tenant, None, 120)
        telemetry.force_flush(timeout_millis=10000)
        _show(
            page, web_url, "profile-metrics", "Profile metrics", "Show metrics", tenant
        )
        panel = _region(page, f"Profile selections of {tenant}")
        expected = (
            f"2 {SPAN_NAME_PROFILE_SELECTION} spans in this window, but none names "
            "a modality. Verify ProfileSelectionAgent is recording it."
        )
        _until(page, panel, lambda: panel.locator("p.muted").inner_text() == expected)
        expect(panel.get_by_role("list", name="Per-modality metrics")).to_have_count(0)

    def test_a_slow_store_reads_as_slow_and_an_unconfigured_one_as_unconfigured(
        self, page, web_url, phoenix_proxy, telemetry, monkeypatch
    ):
        tenant = _tenant("webpmslow")
        monkeypatch.setattr(telemetry_metrics, "SPAN_READ_BUDGET_S", 0.5)

        def slow(method, path, body):
            if "/spans" in path:
                time.sleep(2)
            return None

        phoenix_proxy.intercept = slow
        _show(
            page, web_url, "profile-metrics", "Profile metrics", "Show metrics", tenant
        )
        panel = _region(page, f"Profile selections of {tenant}")
        slow_notice = panel.locator("p.alert.warning")
        expect(slow_notice).to_have_text(
            "The telemetry store did not return the "
            f"{SPAN_NAME_PROFILE_SELECTION} spans of tenant {tenant} within 0.5 s. "
            "It is slow, not empty; retry shortly."
        )
        expect(slow_notice).to_have_attribute("role", "status")
        expect(panel.get_by_role("alert")).to_have_count(0)
        expect(panel.locator("p.muted")).to_have_count(0)

        phoenix_proxy.intercept = None
        monkeypatch.setattr(telemetry.config, "provider", "absent")
        other = _tenant("webpmunconfigured")
        _show(
            page, web_url, "profile-metrics", "Profile metrics", "Show metrics", other
        )
        expect(
            _region(page, f"Profile selections of {other}").get_by_role("alert")
        ).to_have_text(
            f"No telemetry provider could be built for tenant {other}, so the "
            f"{SPAN_NAME_PROFILE_SELECTION} spans cannot be read; the runtime log "
            "names the cause."
        )

    def test_selections_are_kept_for_half_a_minute_and_refresh_reads_again(
        self, page, web_url, telemetry, runtime_reads
    ):
        first, second = _tenant("webpmcache"), _tenant("webpmcache")
        read = f"/admin/tenant/{first}/telemetry/profile-selection?lookback_hours=24"
        _show(
            page, web_url, "profile-metrics", "Profile metrics", "Show metrics", first
        )
        for tenant in (first, second, first):
            chooser = page.get_by_role("form", name="Choose tenant")
            chooser.get_by_label("Tenant ID").fill(tenant)
            chooser.get_by_role("button", name="Show metrics").click()
            expect(
                _region(page, f"Profile selections of {tenant}").locator("p.muted")
            ).to_be_visible()
        assert [p.replace("%3A", ":") for p in runtime_reads].count(read) == 1
        _region(page, f"Profile selections of {first}").get_by_role(
            "button", name="Refresh"
        ).click()
        page.wait_for_timeout(500)
        assert [p.replace("%3A", ":") for p in runtime_reads].count(read) == 2


class TestRlmAb:
    def test_windows_from_minutes_to_weeks_and_the_ab_id(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webrlmwindow")
        for ab_id, minutes in (("ab-now", 1), ("ab-hour", 50), ("ab-week", 190 * 60)):
            record_ab_compare(
                telemetry,
                tenant,
                ab_result(ab_id, f"q {ab_id}", 100.0, 10, 0.1, False),
                "lectures",
                minutes_ago=minutes,
            )
        telemetry.force_flush(timeout_millis=10000)
        _show(page, web_url, "rlm-ab", "RLM A/B", "Show comparisons", tenant)
        panel = _region(page, f"RLM A/B comparisons of {tenant}")
        expect(panel.locator("p.caption")).to_have_text(
            "Spans recorded by cogniverse-optim --mode ab-compare. Each row is one "
            "query and context from the input dataset, answered with and without "
            "RLM; both answers share one A/B id."
        )

        def ids():
            return [row[0] for row in _rows(page, "Comparisons")]

        _lookback(panel, 200)
        _until(
            page,
            panel,
            lambda: (
                page.get_by_role("table", name="Comparisons", exact=True).count() == 1
                and len(ids()) == 3
            ),
        )
        assert ids() == ["ab-now", "ab-hour", "ab-week"]
        _lookback(panel, 0.5)
        expect(
            page.get_by_role("table", name="Comparisons", exact=True).locator(
                "tbody tr"
            )
        ).to_have_count(1)
        assert ids() == ["ab-now"]
        _lookback(panel, 0.05)
        expect(
            panel.get_by_role("form", name="Lookback").get_by_role("alert")
        ).to_have_text("Lookback must be a number of hours from 0.1 to 720.")

    def test_a_down_store_and_an_unconfigured_one_are_errors(
        self, page, web_url, phoenix_proxy, telemetry, monkeypatch
    ):
        tenant = _tenant("webrlmdown")
        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        _show(page, web_url, "rlm-ab", "RLM A/B", "Show comparisons", tenant)
        panel = _region(page, f"RLM A/B comparisons of {tenant}")
        expect(panel.get_by_role("alert")).to_have_text(
            f"Could not read the {AB_COMPARE_SPAN_NAME} spans of tenant {tenant}."
        )
        expect(panel.locator("p.muted")).to_have_count(0)

        phoenix_proxy.intercept = None
        monkeypatch.setattr(telemetry.config, "provider", "absent")
        other = _tenant("webrlmunconfigured")
        _show(page, web_url, "rlm-ab", "RLM A/B", "Show comparisons", other)
        expect(
            _region(page, f"RLM A/B comparisons of {other}").get_by_role("alert")
        ).to_have_text(
            f"No telemetry provider could be built for tenant {other}, so the "
            f"{AB_COMPARE_SPAN_NAME} spans cannot be read; the runtime log names "
            "the cause."
        )
