"""The web client's Profile metrics, RLM A/B, Analytics, Evaluation and Routing
evaluation views, driven in Chromium against spans in real Phoenix.

Spans are recorded the way their producers record them, with fixed values
(``tests/utils/telemetry_metric_spans``). The runtime reads them through a
forwarding proxy in front of Phoenix's HTTP API, so a test can fail the reads.
"""

from __future__ import annotations

import re
import statistics
import time
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    AutoAnnotation,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.optimization_cli import emit_ab_compare_span
from cogniverse_runtime.routers import admin
from tests.utils.approval_review import review_config_manager, run_in_own_loop
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    SEARCH,
    ab_result,
    record_profile_selection,
    record_routing,
    record_sample_traces,
    record_search,
)
from tests.utils.web_client import (
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

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


def _tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _show(page: Page, web_url: str, view: str, heading: str, action: str, tenant):
    page.goto(f"{web_url}/#/ops/{view}")
    expect(page.get_by_role("heading", name=heading, level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name=action).click()


def _table(page: Page, table: str):
    return page.get_by_role("table", name=table, exact=True)


def _rows(page: Page, table: str):
    return [
        row.locator("td").all_inner_texts()
        for row in _table(page, table).locator("tbody tr").all()
    ]


def _rows_until(page: Page, panel: str, table: str, want: int, timeout=90.0):
    """The rows of ``table`` once it shows ``want``, refreshing ``panel``
    meanwhile (Phoenix serves spans after a short indexing delay)."""
    region = page.get_by_role("region", name=panel, exact=True)
    deadline = time.monotonic() + timeout
    while True:
        expect(region.locator("table, dl, p.muted, .alert").first).to_be_visible()
        rows = _rows(page, table) if _table(page, table).count() else []
        if len(rows) == want or time.monotonic() > deadline:
            return rows
        region.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)


def _bars(page: Page, title: str):
    figure = page.get_by_role("figure", name=title, exact=True)
    return list(
        zip(
            figure.locator(".bar-label").all_inner_texts(),
            figure.locator(".bar-value").all_inner_texts(),
            strict=True,
        )
    )


def test_profile_metrics_show_each_modality(page, web_url, telemetry):
    tenant = _tenant("webprofile")
    for duration in (100, 200, 300, 400):
        record_profile_selection(telemetry, tenant, "video", duration)
    record_profile_selection(telemetry, tenant, "image", 50)
    record_profile_selection(telemetry, tenant, "image", 150, failed=True)
    telemetry.force_flush(timeout_millis=10000)

    _show(page, web_url, "profile-metrics", "Profile metrics", "Show metrics", tenant)
    assert _rows_until(
        page, f"Profile selections of {tenant}", "Selections by modality", 2
    ) == [
        ["video", "4", "250.0 ms", "385.0 ms", "397.0 ms", "100.0%"],
        ["image", "2", "100.0 ms", "145.0 ms", "149.0 ms", "50.0%"],
    ]
    assert _bars(page, "Selections per modality") == [("video", "4"), ("image", "2")]
    assert _bars(page, "P95 latency per modality") == [
        ("video", "385.0 ms"),
        ("image", "145.0 ms"),
    ]


def test_profile_metrics_show_an_outage_rather_than_an_empty_window(
    page, web_url, phoenix_proxy
):
    tenant = _tenant("webprofileoutage")
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    _show(page, web_url, "profile-metrics", "Profile metrics", "Show metrics", tenant)
    panel = page.get_by_role(
        "region", name=f"Profile selections of {tenant}", exact=True
    )
    expect(panel.get_by_role("alert")).to_have_text(
        f"Could not read the cogniverse.profile_selection spans of tenant {tenant}."
    )
    expect(panel.locator("p.muted")).to_have_count(0)


def test_rlm_ab_shows_averages_datasets_and_comparisons(page, web_url, telemetry):
    tenant = _tenant("webrlmab")
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

    _show(page, web_url, "rlm-ab", "RLM A/B", "Show comparisons", tenant)
    rows_shown = _rows_until(page, f"RLM A/B comparisons of {tenant}", "Comparisons", 3)
    assert [[row[0], *row[2:]] for row in rows_shown] == [
        ["ab-3", "third question", "podcasts", "+600.0", "+10.0", "-0.500", "no"],
        ["ab-2", "second question", "lectures", "+400.0", "+50.0", "+0.250", "yes"],
        ["ab-1", "first question", "lectures", "+200.0", "+30.0", "+0.250", "no"],
    ]
    expect(page.get_by_role("region", name="Comparisons", exact=True)).to_be_visible()
    averages = page.locator('dl[aria-label="Comparison averages"]')
    terms = averages.locator("dt").all_inner_texts()
    values = averages.locator("dd").all_inner_texts()
    assert dict(zip(terms, values, strict=True)) == {
        "Comparisons": "3",
        "Latency change with RLM": "+400.0 ms",
        "Token change with RLM": "+30.0",
        "Judge score change with RLM": "0.000",
        "RLM fell back": "33.3%",
    }
    assert _rows(page, "Comparisons per dataset") == [
        ["lectures", "2", "+300.0", "+40.0", "+0.250"],
        ["podcasts", "1", "+600.0", "+10.0", "-0.500"],
    ]
    assert _bars(page, "Average latency change per dataset") == [
        ("lectures", "+300.0 ms"),
        ("podcasts", "+600.0 ms"),
    ]


def test_rlm_ab_with_no_comparisons_says_how_to_record_them(page, web_url, telemetry):
    tenant = _tenant("webrlmabempty")
    _show(page, web_url, "rlm-ab", "RLM A/B", "Show comparisons", tenant)
    panel = page.get_by_role(
        "region", name=f"RLM A/B comparisons of {tenant}", exact=True
    )
    expect(panel.locator("p.muted")).to_have_text(
        "No rlm.ab_compare spans in this window. Run cogniverse-optim --mode "
        f"ab-compare --tenant-id {tenant} --queries-dataset <name> to populate."
    )


def _plot(page: Page, title: str):
    """The series Plotly drew in the chart ``title``."""
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    expect(area).to_have_class(re.compile(r"\bjs-plotly-plot\b"))
    return area.evaluate(
        """el => el.data.map(t => {
            const series = {name: t.name ?? null, type: t.type};
            for (const key of ['x', 'y', 'z']) if (t[key] !== undefined) series[key] = t[key];
            return series;
        })"""
    )


def _facts(page: Page, label: str):
    facts = page.locator(f'dl[aria-label="{label}"]')
    return dict(
        zip(
            facts.locator("dt").all_inner_texts(),
            facts.locator("dd").all_inner_texts(),
            strict=True,
        )
    )


def _section(page: Page, name: str):
    page.get_by_role("navigation", name="Analytics sections").get_by_role(
        "button", name=name, exact=True
    ).click()
    return page.get_by_role("region", name=name, exact=True)


def _show_traces(page, web_url, tenant, want):
    _show(page, web_url, "analytics", "Analytics", "Show traces", tenant)
    region = page.get_by_role("region", name=f"Traces of {tenant}", exact=True)
    deadline = time.monotonic() + 90
    while True:
        expect(region.locator("dl, p.muted, .alert").first).to_be_visible()
        facts = _facts(page, "Trace summary") if region.locator("dl").count() else {}
        if facts.get("Traces") == str(want) or time.monotonic() > deadline:
            return region, facts
        region.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)


def _buckets(rows, width_ms):
    buckets = {}
    for row in rows:
        start = pd.Timestamp(row["start_time"]).value // 1_000_000
        key = start // width_ms * width_ms
        buckets.setdefault(key, []).append(row["duration_ms"].expected)
    return {
        pd.Timestamp(key, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z"): v
        for key, v in sorted(buckets.items())
    }


def test_analytics_charts_and_explores_a_tenants_traces(page, web_url, telemetry):
    tenant = _tenant("webtraces")
    rows = record_sample_traces(telemetry, tenant)
    telemetry.force_flush(timeout_millis=10000)

    region, facts = _show_traces(page, web_url, tenant, 6)
    assert facts == {
        "Traces": "6",
        "Succeeded": "5 (83.3%)",
        "Mean latency": "341.7 ms",
        "P95 latency": "850.0 ms",
    }

    overview = page.get_by_role("region", name="Overview", exact=True)
    assert _rows(page, "Traces by operation") == [
        [SEARCH, "5", "83.3%", "400.0 ms", "880.0 ms", "20.0%"],
        ["agent.dispatch", "1", "16.7%", "50.0 ms", "50.0 ms", "0.0%"],
    ]
    expect(overview).to_be_visible()
    assert _bars(page, "Latency percentiles") == [
        ("Min", "50.0 ms"),
        ("P50", "250.0 ms"),
        ("P75", "375.0 ms"),
        ("P90", "700.0 ms"),
        ("P95", "850.0 ms"),
        ("P99", "970.0 ms"),
        ("Max", "1000.0 ms"),
    ]

    _section(page, "Time series")
    buckets = _buckets(rows, 300_000)
    assert _plot(page, "Traces over time") == [
        {
            "name": "Traces",
            "type": "bar",
            "x": list(buckets),
            "y": [len(v) for v in buckets.values()],
        }
    ]
    latency = _plot(page, "Latency over time")
    assert [(s["name"], s["x"]) for s in latency] == [
        ("Mean", list(buckets)),
        ("P50", list(buckets)),
        ("P95", list(buckets)),
    ]
    assert latency[0]["y"] == pytest.approx([sum(v) / len(v) for v in buckets.values()])

    _section(page, "Distribution")
    page.get_by_label("Group by").select_option("operation")
    expect(page.get_by_role("figure", name="Latency histogram")).to_be_visible()
    page.wait_for_function(
        """() => document.querySelector('figure[aria-label="Latency histogram"] .plot-area')
            ?.data?.length === 2"""
    )
    assert _plot(page, "Latency histogram") == [
        {"name": "agent.dispatch", "type": "histogram", "x": [50.0]},
        {
            "name": SEARCH,
            "type": "histogram",
            "x": [r["duration_ms"].expected for r in rows if r["operation"] == SEARCH],
        },
    ]

    _section(page, "Heatmap")
    page.get_by_label("Columns").select_option("operation")
    page.get_by_label("Rows").select_option("status")
    assert _plot(page, "Mean latency by status and operation") == [
        {
            "name": None,
            "type": "heatmap",
            "x": ["agent.dispatch", SEARCH],
            "y": ["failed", "succeeded"],
            "z": [[None, 1000.0], [50.0, 250.0]],
        }
    ]

    outliers = _section(page, "Outliers")
    expect(outliers.locator("p.muted").first).to_have_text(
        "Traces outside -250.0 ms to 750.0 ms (1.5 times the interquartile "
        "range beyond the quartiles)."
    )
    assert [row[1:] for row in _rows(page, "Outlier traces")] == [
        [
            SEARCH,
            "1000.0 ms",
            "failed: backend down",
            "video_colpali",
            "bm25",
            rows[4]["trace_id"],
        ]
    ]

    _section(page, "Trace explorer")
    assert [row[-1] for row in _rows(page, "Traces")] == [r["trace_id"] for r in rows]
    page.get_by_label("Trace ID or operation", exact=True).fill("DISPATCH")
    assert [row[1:] for row in _rows(page, "Traces")] == [
        [
            "agent.dispatch",
            "50.0 ms",
            "succeeded",
            "audio",
            "semantic",
            rows[5]["trace_id"],
        ]
    ]
    page.get_by_label("Trace ID or operation", exact=True).fill("")
    page.get_by_label("Order").select_option("Slowest first")
    assert [row[2] for row in _rows(page, "Traces")] == [
        "1000.0 ms",
        "400.0 ms",
        "300.0 ms",
        "200.0 ms",
        "100.0 ms",
        "50.0 ms",
    ]

    filters = region.get_by_role("form", name="Trace filters")
    filters.get_by_label("Strategies").select_option(["bm25", "semantic"])
    filters.get_by_role("button", name="Apply filters").click()
    expect(page.locator('dl[aria-label="Trace summary"] dd').first).to_have_text("2")
    assert _facts(page, "Trace summary") == {
        "Traces": "2",
        "Succeeded": "1 (50.0%)",
        "Mean latency": "525.0 ms",
        "P95 latency": "952.5 ms",
    }


def test_analytics_finds_the_root_causes_of_the_filtered_traces(
    page, web_url, telemetry
):
    tenant = _tenant("webrootcauses")
    rows = record_sample_traces(telemetry, tenant)
    telemetry.force_flush(timeout_millis=10000)
    slow = next(row["trace_id"] for row in rows if row["duration_ms"] == 400)
    region, _ = _show_traces(page, web_url, tenant, 6)

    causes = _section(page, "Root causes")
    form = causes.get_by_role("form", name="Find root causes")
    form.get_by_label("Slow percentile").fill("100")
    form.get_by_role("button", name="Find root causes").click()
    expect(form.get_by_role("alert")).to_have_text(
        "The slow percentile must be a whole number from 50 to 99."
    )
    form.get_by_label("Slow percentile").fill("75")
    form.get_by_role("button", name="Find root causes").click()
    expect(page.locator('dl[aria-label="Root cause summary"]')).to_be_visible()
    assert _facts(page, "Root cause summary") == {
        "Traces analyzed": "6",
        "Total issues": "2",
        "Failed": "1 (16.7%)",
        "Slow": "1 (slower than 300.0 ms)",
    }
    hypotheses = causes.get_by_role("list", name="Hypotheses")
    expect(hypotheses.locator("summary")).to_have_text(
        [
            f"Operation '{SEARCH}' experiencing performance degradation "
            "(90.0% confidence, performance)",
            "Profile 'video_colpali' has performance issues "
            "(85.0% confidence, configuration)",
        ]
    )
    hypotheses.locator("summary").first.click()
    expect(hypotheses.locator("details").first).to_contain_text(
        f"Affected traces: {slow}"
    )
    assert _rows(page, "Recommendations") == [
        [
            "medium",
            "performance",
            "Optimize slow operations",
            "Profile slow operations; Add caching where appropriate; Consider "
            "asynchronous processing",
            f"Optimize '{SEARCH}' operation or increase resources",
        ],
        [
            "medium",
            "configuration",
            "Review configuration settings",
            "Check Profile 'video_colpali' has performance issues",
            "Review 'video_colpali' configuration and resource allocation",
        ],
    ]

    filters = region.get_by_role("form", name="Trace filters")
    filters.get_by_label("Strategies").select_option(["bm25"])
    filters.get_by_role("button", name="Apply filters").click()
    expect(page.locator('dl[aria-label="Trace summary"] dd').first).to_have_text("1")
    causes = _section(page, "Root causes")
    form = causes.get_by_role("form", name="Find root causes")
    form.get_by_label("Include slow traces").uncheck()
    form.get_by_role("button", name="Find root causes").click()
    expect(
        causes.get_by_text("No root cause stands out in these traces.")
    ).to_be_visible()
    assert _facts(page, "Root cause summary") == {
        "Traces analyzed": "1",
        "Total issues": "1",
        "Failed": "1 (100.0%)",
        "Slow": "0",
    }


def test_analytics_shows_an_outage_rather_than_no_traces(page, web_url, phoenix_proxy):
    tenant = _tenant("webtracesoutage")
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    _show(page, web_url, "analytics", "Analytics", "Show traces", tenant)
    region = page.get_by_role("region", name=f"Traces of {tenant}", exact=True)
    expect(region.get_by_role("alert")).to_have_text(
        f"Could not read the traces of tenant {tenant}. Refresh to retry."
    )
    expect(region.get_by_text("No traces match in this window.")).to_have_count(0)


def test_analytics_of_a_quiet_tenant_says_there_are_no_traces(page, web_url, telemetry):
    tenant = _tenant("webtracesempty")
    _show(page, web_url, "analytics", "Analytics", "Show traces", tenant)
    region = page.get_by_role("region", name=f"Traces of {tenant}", exact=True)
    expect(region.locator("p.muted")).to_have_text("No traces match in this window.")
    expect(page.get_by_role("navigation", name="Analytics sections")).to_have_count(0)


SUNSET, RED_CAR, DOG = "sunset over the sea", "a red car", "dog on a beach"


@pytest.fixture()
def upload_golden(runtime_url, phoenix_container, monkeypatch):
    """Uploads a tenant's golden set through the runtime's admin route."""
    monkeypatch.setattr(admin, "_phoenix_endpoints", {})
    admin.set_phoenix_endpoints(
        phoenix_container["http_endpoint"], phoenix_container["grpc_endpoint"]
    )

    def upload(tenant):
        response = httpx.put(
            f"{runtime_url}/admin/tenants/{tenant}/golden_set_ground_truth",
            json=[
                {"query": SUNSET, "expected_videos": ["sunset"]},
                {"query": RED_CAR, "expected_videos": ["red_car", "garage"]},
                {"query": DOG, "expected_videos": ["dog"]},
            ],
            timeout=60,
        )
        assert (response.status_code, response.json()["row_count"]) == (200, 3)

    return upload


def _show_evaluation(page, web_url, tenant):
    _show(page, web_url, "evaluation", "Evaluation", "Evaluate", tenant)
    return page.get_by_role(
        "region", name=f"Golden set evaluation of {tenant}", exact=True
    )


def test_evaluation_scores_a_tenants_searches_of_its_golden_set(
    page, web_url, telemetry, upload_golden
):
    tenant = _tenant("webgolden")
    upload_golden(tenant)
    sunset = record_search(
        tenant, SUNSET, "video_colpali", "hybrid", ["beach.mp4", "sunset.mp4"]
    )
    red_car = record_search(
        tenant,
        RED_CAR,
        "video_colpali",
        "hybrid",
        ["red_car.mp4", "beach.mp4", "garage.mov", "a.mp4", "b.mp4", "c.mp4"],
    )
    sunset_bm25 = record_search(tenant, SUNSET, "video_colpali", "bm25", ["sunset.mp4"])
    record_search(tenant, RED_CAR, "audio", "bm25", [], error="backend down")
    telemetry.force_flush(timeout_millis=10000)

    _show_evaluation(page, web_url, tenant)
    assert _rows_until(
        page,
        f"Golden set evaluation of {tenant}",
        "Scores by profile and strategy",
        2,
    ) == [
        [
            "video_colpali",
            "bm25",
            "1",
            "1.000",
            "1.000",
            "1.000",
            "1.000",
            "1.000",
            "100.0%",
        ],
        [
            "video_colpali",
            "hybrid",
            "2",
            "0.750",
            "0.775",
            "0.250",
            "1.000",
            "0.450",
            "50.0%",
        ],
    ]
    summary = {
        "Golden queries": "3",
        "Searched": "2",
        "Not searched": "1",
        "Failed searches": "1",
        "Unscored searches": "0",
    }
    # The failed search may reach Phoenix's index after the others.
    panel = page.get_by_role(
        "region", name=f"Golden set evaluation of {tenant}", exact=True
    )
    deadline = time.monotonic() + 60
    while _facts(page, "Evaluation summary") != summary and time.monotonic() < deadline:
        panel.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)
    assert _facts(page, "Evaluation summary") == summary
    assert _plot(page, "Success by profile and strategy") == [
        {
            "name": None,
            "type": "heatmap",
            "x": ["bm25", "hybrid"],
            "y": ["video_colpali"],
            "z": [[1, 0.5]],
        }
    ]
    assert page.get_by_role("list", name="Golden queries not searched").locator(
        "li"
    ).all_inner_texts() == [DOG]

    results = page.get_by_role("region", name="Query results", exact=True)
    strategies = results.get_by_role("tablist", name="Strategies of video_colpali")
    strategies.get_by_role("tab", name="hybrid", exact=True).click()
    assert _rows(page, "Query results") == [
        [
            SUNSET,
            "sunset",
            "✗ beach\n✓ sunset",
            "0.500",
            "0.000",
            "1.000",
            _local(page, sunset),
        ],
        [
            RED_CAR,
            "red_car, garage",
            "✓ red_car\n✗ beach\n✓ garage\n✗ a\n✗ b",
            "1.000",
            "0.500",
            "1.000",
            _local(page, red_car),
        ],
    ]
    strategies.get_by_role("tab", name="bm25", exact=True).click()
    assert _rows(page, "Query results") == [
        [
            SUNSET,
            "sunset",
            "✓ sunset",
            "1.000",
            "1.000",
            "1.000",
            _local(page, sunset_bm25),
        ]
    ]


def _local(page: Page, search):
    """The browser's local rendering of a recorded search's start time."""
    return page.evaluate(
        "iso => new Date(iso).toLocaleString()",
        search[1].isoformat(timespec="milliseconds"),
    )


def test_evaluation_of_a_tenant_without_a_golden_set_says_how_to_upload_one(
    page, web_url
):
    tenant = _tenant("webnogolden")
    panel = _show_evaluation(page, web_url, tenant)
    expect(panel.get_by_role("alert")).to_have_text(
        f"Tenant {tenant} has no golden set. Upload one with "
        f"PUT /admin/tenants/{tenant}/golden_set_ground_truth."
    )
    expect(panel.locator('dl[aria-label="Evaluation summary"]')).to_have_count(0)


def test_evaluation_shows_an_outage_rather_than_no_searches(
    page, web_url, phoenix_proxy, upload_golden
):
    tenant = _tenant("webgoldenoutage")
    upload_golden(tenant)

    def fail_span_reads(method, path, body):
        return (503, {"detail": "down"}) if "/spans" in path else None

    phoenix_proxy.intercept = fail_span_reads
    panel = _show_evaluation(page, web_url, tenant)
    expect(panel.get_by_role("alert")).to_have_text(
        f"Could not read the {SEARCH} spans of tenant {tenant}."
    )
    expect(panel.locator("p.muted")).to_have_count(0)


def _show_routing(page, web_url, tenant):
    _show(page, web_url, "routing", "Routing evaluation", "Show decisions", tenant)
    return page.get_by_role("region", name=f"Routing decisions of {tenant}", exact=True)


def _routing_decisions(runtime_url, tenant, want, timeout=90.0):
    """The tenant's decisions as the runtime serves them, once it serves
    ``want`` (Phoenix serves spans after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while True:
        response = httpx.get(
            f"{runtime_url}/admin/tenant/{tenant}/routing-decisions",
            params={"lookback_hours": 1},
            timeout=60,
        )
        assert response.status_code == 200, response.text
        decisions = response.json()["decisions"]
        if len(decisions) == want or time.monotonic() > deadline:
            return decisions


def _decision_rows(page: Page):
    # The last cell holds the row's buttons.
    return [row[:-1] for row in _rows(page, "Decisions")]


def test_routing_evaluation_reviews_a_tenants_decisions(
    page, web_url, runtime_url, telemetry
):
    tenant = _tenant("webrouting")
    confident = record_routing(
        telemetry, tenant, "search_agent", 0.9, 100, minutes_ago=3
    )
    failed = record_routing(
        telemetry, tenant, "search_agent", 0.3, 300, minutes_ago=2, failed=True
    )
    boundary = record_routing(
        telemetry, tenant, "summarizer_agent", 0.65, 200, minutes_ago=1
    )
    telemetry.force_flush(timeout_millis=10000)
    served = _routing_decisions(runtime_url, tenant, 3)
    assert [d["span_id"] for d in served] == [boundary, failed, confident]
    storage = AnnotationStorage(tenant_id=tenant)
    run_in_own_loop(
        storage.store_llm_annotation(
            failed,
            AutoAnnotation(
                span_id=failed,
                label=AnnotationLabel.WRONG_ROUTING,
                confidence=0.8,
                reasoning="Summaries belong to the summarizer.",
                suggested_correct_agent="summarizer_agent",
                requires_human_review=True,
            ),
        )
    )
    started = {
        d["span_id"]: page.evaluate(
            "iso => new Date(iso).toLocaleString()", d["start_time"]
        )
        for d in served
    }
    unlabelled_row = [
        started[boundary],
        "a query for summarizer_agent",
        "summarizer_agent",
        "0.65",
        "success",
        "200.0 ms",
        "Unlabelled",
    ]
    llm_row = [
        started[failed],
        "a query for search_agent",
        "search_agent",
        "0.30",
        "failure",
        "300.0 ms",
        "wrong_routing (should be summarizer_agent)\nLLM",
    ]
    confident_row = [
        started[confident],
        "a query for search_agent",
        "search_agent",
        "0.90",
        "success",
        "100.0 ms",
        "Unlabelled",
    ]

    panel = _show_routing(page, web_url, tenant)
    deadline = time.monotonic() + 60
    while True:
        expect(panel.locator("dl").first).to_be_visible()
        if _decision_rows(page) == [unlabelled_row, llm_row, confident_row]:
            break
        assert time.monotonic() < deadline, _decision_rows(page)
        panel.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)
    assert _facts(page, "Routing summary") == {
        "Decisions": "3",
        "Succeeded": "2",
        "Failed": "1",
        "Ambiguous": "0",
        "Unreadable": "0",
        "Accuracy": "66.7%",
        "Confidence calibration": f"{statistics.correlation([0.65, 0.3, 0.9], [1, 0, 1]):.2f}",
        "Latency mean": "200.0 ms",
        "Latency p50": "200.0 ms",
        "Latency p95": "290.0 ms",
    }
    assert _rows(page, "Decisions by agent") == [
        ["search_agent", "2", "1", "1", "0", "50.0%", "0.60", "200.0 ms"],
        ["summarizer_agent", "1", "1", "0", "0", "100.0%", "0.65", "200.0 ms"],
    ]
    assert _plot(page, "Confidence by outcome") == [
        {"name": "success", "type": "histogram", "x": [0.65, 0.9]},
        {"name": "failure", "type": "histogram", "x": [0.3]},
        {"name": "ambiguous", "type": "histogram", "x": []},
    ]
    assert _plot(page, "Success rate by confidence") == [
        {
            "name": "Success rate",
            "type": "scatter",
            "x": ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
            "y": [None, 0, None, 1, 1],
        }
    ]
    hours = sorted(
        {
            pd.Timestamp(d["start_time"]).floor("h").strftime("%Y-%m-%dT%H:00:00Z")
            for d in served
        }
    )
    per_hour = [
        [
            d
            for d in served
            if pd.Timestamp(d["start_time"]).floor("h").strftime("%Y-%m-%dT%H:00:00Z")
            == hour
        ]
        for hour in hours
    ]
    assert _plot(page, "Decisions per hour") == [
        {
            "name": "Decisions",
            "type": "bar",
            "x": hours,
            "y": [len(ds) for ds in per_hour],
        },
        {
            "name": "Succeeded",
            "type": "bar",
            "x": hours,
            "y": [sum(d["outcome"] == "success" for d in ds) for ds in per_hour],
        },
    ]

    decisions = page.get_by_role("region", name="Decisions", exact=True)
    decisions.get_by_label("Show").select_option("LLM labels to review")
    assert _decision_rows(page) == [llm_row]
    approve = decisions.get_by_role("button", name=f"Approve the LLM label of {failed}")
    approve.click()
    expect(decisions.get_by_role("alert")).to_have_text(
        "Enter your name as the reviewer first."
    )
    panel.get_by_label("Reviewer").fill("dana")
    approve.click()
    expect(page.get_by_role("status")).to_have_text(
        f"Approved the LLM label of {failed}."
    )
    expect(decisions.get_by_text("No decisions match.")).to_be_visible()
    decisions.get_by_label("Show").select_option("Reviewed")
    assert _decision_rows(page) == [
        [
            *llm_row[:-1],
            "wrong_routing (should be summarizer_agent)\nLLM, approved by dana",
        ]
    ]

    decisions.get_by_label("Show").select_option("All decisions")
    decisions.get_by_role("button", name=f"Relabel {boundary}").click()
    form = page.get_by_role("form", name=f"Label {boundary}")
    form.get_by_role("combobox").select_option("wrong")
    form.get_by_label("Reasoning").fill("Needs a search.")
    form.get_by_label("Should have gone to").fill("search_agent")
    form.get_by_role("button", name="Save label").click()
    expect(page.get_by_role("status")).to_have_text(f"Labelled {boundary} wrong.")
    expect(form).to_have_count(0)
    assert _decision_rows(page) == [
        [*unlabelled_row[:-1], "wrong (should be search_agent)\ndana"],
        [
            *llm_row[:-1],
            "wrong_routing (should be summarizer_agent)\nLLM, approved by dana",
        ],
        confident_row,
    ]

    stored = {
        span_id: run_in_own_loop(storage.get_annotation(span_id))
        for span_id in (failed, boundary)
    }
    assert {
        span_id: (
            annotation["label"],
            {
                key: annotation["metadata"].get(key)
                for key in (
                    "annotator",
                    "human_reviewed",
                    "approved_by",
                    "suggested_agent",
                    "reasoning",
                )
            },
        )
        for span_id, annotation in stored.items()
    } == {
        failed: (
            "wrong_routing",
            {
                "annotator": "llm",
                "human_reviewed": True,
                "approved_by": "dana",
                "suggested_agent": "summarizer_agent",
                "reasoning": "Summaries belong to the summarizer.",
            },
        ),
        boundary: (
            "wrong",
            {
                "annotator": "dana",
                "human_reviewed": True,
                "approved_by": None,
                "suggested_agent": "search_agent",
                "reasoning": "Needs a search.",
            },
        ),
    }


def test_routing_evaluation_of_a_quiet_tenant_says_there_are_no_decisions(
    page, web_url, telemetry
):
    tenant = _tenant("webquietrouting")
    panel = _show_routing(page, web_url, tenant)
    expect(
        panel.get_by_text("No routing decisions were recorded in this window.")
    ).to_be_visible()
    assert _facts(page, "Routing summary")["Decisions"] == "0"
    expect(page.get_by_role("table", name="Decisions")).to_have_count(0)


def test_routing_evaluation_shows_an_outage_rather_than_no_decisions(
    page, web_url, phoenix_proxy
):
    tenant = _tenant("webroutingoutage")

    def fail_span_reads(method, path, body):
        return (503, {"detail": "down"}) if "/spans" in path else None

    phoenix_proxy.intercept = fail_span_reads
    panel = _show_routing(page, web_url, tenant)
    expect(panel.get_by_role("alert")).to_have_text(
        f"Could not read the routing decisions of tenant {tenant}."
    )
    expect(
        panel.get_by_text("No routing decisions were recorded in this window.")
    ).to_have_count(0)
