"""The web client's Analytics view beyond its charts' defaults: the time range,
refresh, raw data, export, the extra distributions and outlier metrics, the
explorer's scopes and pages, and the root-cause details, driven in Chromium
against spans in real Phoenix.

Spans are recorded the way their producers record them, with fixed values
(``tests/utils/telemetry_metric_spans``). The runtime reads them through a
forwarding proxy in front of Phoenix's HTTP API, so a test can slow or fail
the reads.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
import time
from uuid import uuid4

import httpx
import pandas as pd
import pytest
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from tests.utils.approval_review import review_config_manager
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    SEARCH,
    record_sample_traces,
    record_trace,
)
from tests.utils.web_client import recording_telemetry_sink, serve_web
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

# The CSV columns, in the order the view writes them.
TRACE_COLUMNS = [
    "trace_id",
    "span_id",
    "start_time",
    "duration_ms",
    "operation",
    "succeeded",
    "profile",
    "strategy",
    "error",
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


@pytest.fixture(scope="module")
def runtime_url(phoenix_container, schema_loader, workflow_state_redis_url, telemetry):
    """The operations routes, with Phoenix links pointing at the Phoenix the
    spans are in."""
    os.environ["PHOENIX_UI_URL"] = phoenix_container["http_endpoint"]
    try:
        with serve_ops_runtime(
            review_config_manager(phoenix_container, workflow_state_redis_url),
            schema_loader,
            workflow_state_redis_url,
        ) as url:
            yield url
    finally:
        del os.environ["PHOENIX_UI_URL"]


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
    context = browser.new_context(
        timezone_id="UTC", locale="en-US", accept_downloads=True
    )
    page = context.new_page()
    yield page
    context.close()


def _tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _region(page: Page, name: str):
    return page.get_by_role("region", name=name, exact=True)


def _table(page: Page, table: str):
    return page.get_by_role("table", name=table, exact=True)


def _rows(page: Page, table: str):
    return [
        row.locator("td").all_inner_texts()
        for row in _table(page, table).locator("tbody tr").all()
    ]


def _facts(page: Page, label: str):
    facts = page.locator(f'dl[aria-label="{label}"]')
    return dict(
        zip(
            facts.locator("dt").all_inner_texts(),
            facts.locator("dd").all_inner_texts(),
            strict=True,
        )
    )


def _plot(page: Page, title: str, *keys: str):
    """The series Plotly drew in the chart ``title``, with ``keys`` of each."""
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    expect(area).to_have_class(re.compile(r"\bjs-plotly-plot\b"))
    return area.evaluate(
        "(el, keys) => el.data.map(t => Object.fromEntries(keys.map(k => [k, t[k] ?? null])))",
        list(keys),
    )


def _layout(page: Page, title: str, expression: str):
    area = page.get_by_role("figure", name=title, exact=True).locator(".plot-area")
    expect(area).to_have_class(re.compile(r"\bjs-plotly-plot\b"))
    return area.evaluate(f"el => {expression}")


def _section(page: Page, name: str):
    page.get_by_role("navigation", name="Analytics sections").get_by_role(
        "button", name=name, exact=True
    ).click()
    return _region(page, name)


def _show_traces(page: Page, web_url: str, tenant: str, want: int):
    """The Analytics view of ``tenant`` once it counts ``want`` traces
    (Phoenix serves spans after a short indexing delay)."""
    page.goto(f"{web_url}/#/ops/analytics")
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Show traces").click()
    region = _region(page, f"Traces of {tenant}")
    deadline = time.monotonic() + 90
    while True:
        expect(
            region.locator(
                'dl[aria-label="Trace summary"], p.muted, [role=alert]'
            ).first
        ).to_be_visible()
        facts = _facts(page, "Trace summary") if region.locator("dl").count() else {}
        if facts.get("Traces") == str(want) or time.monotonic() > deadline:
            assert facts.get("Traces") == str(want), facts
            return region
        region.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(2000)


def _apply(region, **fields):
    filters = region.get_by_role("form", name="Trace filters")
    for label, value in fields.items():
        control = filters.get_by_label(label)
        if isinstance(value, list):
            control.select_option(value)
        elif control.evaluate("el => el.tagName") == "SELECT":
            control.select_option(value)
        else:
            control.fill(value)
    filters.get_by_role("button", name="Apply filters").click()


def _trace_count(page: Page, want: str):
    expect(page.locator('dl[aria-label="Trace summary"] dd').first).to_have_text(
        want, timeout=60_000
    )


def _hour_rows(rows):
    hours = {}
    for row in rows:
        hour = pd.Timestamp(row["start_time"]).floor("h")
        requests, failed = hours.get(hour, (0, 0))
        hours[hour] = (requests + 1, failed + (row["error"] is not None))
    return [
        [
            hour.strftime("%Y-%m-%d %H:00"),
            str(requests),
            str(failed),
            f"{failed / requests * 100:.1f}%",
            "no",
        ]
        for hour, (requests, failed) in sorted(hours.items())
    ]


class TestWindowAndRefresh:
    def test_a_preset_and_a_custom_window_choose_the_traces_read(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webwindow")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        region = _show_traces(page, web_url, tenant, 6)
        expect(region.get_by_label("Last refreshed")).to_have_text(
            re.compile(r"^Last 6 hours; last refreshed \d{1,2}:\d{2}:\d{2} [AP]M\.$")
        )

        _apply(region, Window="Last 15 minutes")
        expect(region.get_by_label("Last refreshed")).to_contain_text(
            "Last 15 minutes; last refreshed"
        )
        _trace_count(page, "6")

        # Minute-precision bounds around the traces 3 and 4 minutes old.
        start = pd.Timestamp(rows[3]["start_time"]).floor("min")
        end = pd.Timestamp(rows[2]["start_time"]).floor("min") + pd.Timedelta(minutes=1)
        filters = region.get_by_role("form", name="Trace filters")
        filters.get_by_label("Window").select_option("Custom range")
        filters.get_by_label("Start (UTC)").fill(start.strftime("%Y-%m-%dT%H:%M"))
        filters.get_by_label("End (UTC)").fill(end.strftime("%Y-%m-%dT%H:%M"))
        filters.get_by_role("button", name="Apply filters").click()
        _trace_count(page, "2")
        expect(region.get_by_label("Last refreshed")).to_contain_text(
            f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} UTC"
        )
        _section(page, "Trace explorer")
        assert [row[-1] for row in _rows(page, "Traces")] == [
            rows[2]["trace_id"],
            rows[3]["trace_id"],
        ]

        # A reversed range never reaches the runtime.
        filters.get_by_label("Start (UTC)").fill(end.strftime("%Y-%m-%dT%H:%M"))
        filters.get_by_label("End (UTC)").fill(start.strftime("%Y-%m-%dT%H:%M"))
        filters.get_by_role("button", name="Apply filters").click()
        expect(region.get_by_role("alert")).to_have_text(
            "The custom range must start before it ends."
        )

    def test_auto_refresh_reads_new_traces_without_a_click(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webautorefresh")
        record_trace(telemetry, tenant, SEARCH, 100, minutes_ago=2)
        telemetry.force_flush(timeout_millis=10000)
        region = _show_traces(page, web_url, tenant, 1)
        refresh = region.get_by_label("Refresh and data")
        refresh.get_by_label("Every (seconds)").fill("5")
        refresh.get_by_label("Refresh automatically").check()
        record_trace(telemetry, tenant, SEARCH, 300, minutes_ago=1)
        telemetry.force_flush(timeout_millis=10000)
        _trace_count(page, "2")
        assert _facts(page, "Trace summary")["Mean latency"] == "200.0 ms"

    def test_a_slow_read_says_the_figures_may_be_stale_until_it_answers(
        self, page, web_url, telemetry, phoenix_proxy
    ):
        tenant = _tenant("webslowread")
        record_trace(telemetry, tenant, SEARCH, 100, minutes_ago=2)
        telemetry.force_flush(timeout_millis=10000)
        region = _show_traces(page, web_url, tenant, 1)

        def slow_span_reads(method, path, body):
            if "spans" in path:
                time.sleep(8)
            return None

        phoenix_proxy.intercept = slow_span_reads
        region.get_by_role("button", name="Refresh").click()
        slow = region.get_by_label("Slow read")
        expect(slow).to_have_text(
            re.compile(
                r"^The telemetry backend is slow to answer; the figures below are from "
                r"\d{1,2}:\d{2}:\d{2} [AP]M and may be stale\. Still reading…$"
            ),
            timeout=8_000,
        )
        # The figures read before stay up meanwhile.
        assert _facts(page, "Trace summary")["Traces"] == "1"
        expect(slow).to_have_count(0, timeout=30_000)
        phoenix_proxy.intercept = None

    def test_an_outage_names_the_retry(self, page, web_url, phoenix_proxy):
        tenant = _tenant("webretry")
        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        page.goto(f"{web_url}/#/ops/analytics")
        chooser = page.get_by_role("form", name="Choose tenant")
        chooser.get_by_label("Tenant ID").fill(tenant)
        chooser.get_by_role("button", name="Show traces").click()
        expect(_region(page, f"Traces of {tenant}").get_by_role("alert")).to_have_text(
            f"Could not read the traces of tenant {tenant}. Refresh to retry."
        )


class TestFiltersAndData:
    def test_regex_operation_and_profile_filters(self, page, web_url, telemetry):
        tenant = _tenant("webfilters")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        region = _show_traces(page, web_url, tenant, 6)

        _apply(region, **{"Operation (regular expression)": r"^agent\.dispatch$"})
        _trace_count(page, "1")
        _apply(region, **{"Operation (regular expression)": "", "Profiles": ["audio"]})
        _trace_count(page, "1")
        _section(page, "Trace explorer")
        assert [row[-1] for row in _rows(page, "Traces")] == [rows[5]["trace_id"]]
        _apply(region, Profiles=["video_colpali"])
        _trace_count(page, "5")

        _apply(region, **{"Operation (regular expression)": "search("})
        expect(region.get_by_role("alert")).to_have_text(
            "operation is not a valid regular expression: missing ), unterminated "
            "subpattern at position 6"
        )

    def test_raw_data_and_the_three_exports_carry_every_trace(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webexport")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        region = _show_traces(page, web_url, tenant, 6)

        refresh = region.get_by_label("Refresh and data")
        refresh.get_by_label("Show raw data").check()
        raw = _rows(page, "Raw traces")
        assert [
            [r[0], r[1], r[2], r[3], float(r[4]), r[5], r[6], r[7], r[8]] for r in raw
        ] == [
            [
                row["start_time"],
                row["trace_id"],
                row["span_id"],
                row["operation"],
                pytest.approx(row["duration_ms"].expected),
                "true" if row["succeeded"] else "false",
                row["profile"] or "",
                row["strategy"] or "",
                row["error"] or "",
            ]
            for row in rows
        ]

        with page.expect_download() as download:
            refresh.get_by_role("button", name="Download CSV").click()
        assert download.value.suggested_filename == (
            f"traces-{tenant.replace(':', '_')}.csv"
        )
        with open(download.value.path(), newline="") as handle:
            written = list(csv.reader(handle))
        assert written[0] == TRACE_COLUMNS
        assert [line[:3] + line[4:] for line in written[1:]] == [
            [
                row["trace_id"],
                row["span_id"],
                row["start_time"],
                row["operation"],
                "true" if row["succeeded"] else "false",
                row["profile"] or "",
                row["strategy"] or "",
                row["error"] or "",
            ]
            for row in rows
        ]
        assert [float(line[3]) for line in written[1:]] == [
            pytest.approx(row["duration_ms"].expected) for row in rows
        ]

        with page.expect_download() as download:
            refresh.get_by_role("button", name="Download JSON").click()
        report = json.loads(open(download.value.path()).read())
        assert (
            report["tenant"],
            report["filters"],
            [t["trace_id"] for t in report["analytics"]["traces"]],
            report["analytics"]["statistics"]["requests"],
            "rootCauses" in report,
        ) == (
            tenant,
            {"operation": "", "profiles": [], "strategies": []},
            [row["trace_id"] for row in rows],
            6,
            False,
        )

        causes = _section(page, "Root causes")
        causes.get_by_role("button", name="Find root causes").click()
        expect(page.locator('dl[aria-label="Root cause summary"]')).to_be_visible()
        with page.expect_download() as download:
            refresh.get_by_role("button", name="Download JSON").click()
        report = json.loads(open(download.value.path()).read())
        assert (report["rootCauses"]["traces"], report["rootCauses"]["failed"]) == (
            6,
            1,
        )

        with page.expect_download() as download:
            refresh.get_by_role("button", name="Download HTML").click()
        html = open(download.value.path()).read()
        assert re.findall(r"<caption>([^<]*)</caption>", html) == [
            "Summary",
            "Traces by operation",
            "Traces",
            "Root causes",
            "Recommendations",
        ]
        assert f"<h1>Traces of {tenant}</h1>" in html
        assert [row["trace_id"] for row in rows] == re.findall(
            r"<tr><td>([0-9a-f]{32})</td>", html
        )


class TestCharts:
    def test_overview_distribution_and_heatmap_extras(self, page, web_url, telemetry):
        tenant = _tenant("webcharts")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        region = _show_traces(page, web_url, tenant, 6)
        # 5 of 6 succeeded: 83.3% against the 95% target.
        expect(region.get_by_label("Success target")).to_have_text(
            "11.7 points below the 95.0% success target."
        )
        assert _plot(page, "Operation share", "type", "labels", "values", "hole") == [
            {
                "type": "pie",
                "labels": [SEARCH, "agent.dispatch"],
                "values": [5, 1],
                "hole": 0.4,
            }
        ]

        _section(page, "Time series")
        _region(page, "Time series").get_by_label("Bucket").select_option("1 min")
        per_minute = {}
        for row in rows:
            minute = pd.Timestamp(row["start_time"]).floor("min")
            per_minute[minute] = per_minute.get(minute, 0) + 1
        page.wait_for_function(
            """n => document.querySelector('figure[aria-label="Traces over time"] .plot-area')
                ?.data?.[0]?.x?.length === n""",
            arg=len(per_minute),
        )
        assert _plot(page, "Traces over time", "x", "y") == [
            {
                "x": [m.strftime("%Y-%m-%dT%H:%M:00.000Z") for m in sorted(per_minute)],
                "y": [per_minute[m] for m in sorted(per_minute)],
            }
        ]

        distribution = _section(page, "Distribution")
        expect(distribution.locator("details.explainer summary")).to_have_text(
            "Reading these charts"
        )
        durations = [row["duration_ms"].expected for row in rows]
        assert _plot(page, "Latency density", "type", "name", "y") == [
            {"type": "violin", "name": "all traces", "y": durations}
        ]
        assert _plot(page, "Latency spread", "type", "name", "x") == [
            {"type": "box", "name": "all traces", "x": durations}
        ]
        ordered = sorted(durations)
        assert _plot(page, "Cumulative latency", "type", "x", "y") == [
            {
                "type": "scatter",
                "x": ordered,
                "y": pytest.approx([(i + 1) / 6 for i in range(6)]),
            }
        ]
        assert _layout(
            page, "Cumulative latency", "el.layout.annotations.map(a => [a.text, a.x])"
        ) == [
            ["P50", pytest.approx(250.0)],
            ["P90", pytest.approx(700.0)],
            ["P95", pytest.approx(850.0)],
            ["P99", pytest.approx(970.0)],
        ]
        distribution.get_by_label("Group by").select_option("status")
        page.wait_for_function(
            """() => document.querySelector('figure[aria-label="Latency density"] .plot-area')
                ?.data?.length === 2"""
        )
        assert _plot(page, "Latency density", "name", "y") == [
            {"name": "failed", "y": [1000.0]},
            {"name": "succeeded", "y": [d for d in durations if d != 1000.0]},
        ]

        heatmap = _section(page, "Heatmap")
        expect(heatmap.locator("details.explainer summary")).to_have_text(
            "About profile and strategy"
        )
        heatmap.get_by_label("Columns").select_option("operation")
        heatmap.get_by_label("Rows").select_option("operation")
        expect(heatmap.locator("p.muted")).to_have_text(
            "Pick different fields for columns and rows."
        )
        expect(
            page.get_by_role("figure", name=re.compile("^Mean latency by"))
        ).to_have_count(0)

    def test_latency_and_hourly_error_rate_outliers(self, page, web_url, telemetry):
        tenant = _tenant("weboutliers")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        _show_traces(page, web_url, tenant, 6)

        outliers = _section(page, "Outliers")
        expect(outliers.locator("details.explainer summary")).to_have_text(
            "How outliers are found"
        )
        points = _plot(page, "Latency outliers", "name", "x", "y")
        assert [(s["name"], s["y"]) for s in points] == [
            ("Within bounds", [100.0, 200.0, 300.0, 400.0, 50.0]),
            ("Outliers", [1000.0]),
        ]
        assert points[1]["x"] == [rows[4]["start_time"]]
        assert _layout(
            page, "Latency outliers", "el.layout.shapes.map(s => s.y0)"
        ) == pytest.approx([750.0, 250.0, 850.0, 970.0])

        outliers.get_by_label("Metric").select_option("Hourly error rate")
        expected = _hour_rows(rows)
        assert _rows(page, "Error rate by hour") == expected
        expect(outliers.locator("p.muted")).to_have_text(
            "Error-rate outliers need at least four hours with traces."
        )
        assert _plot(page, "Error rate outliers", "name", "y") == [
            {
                "name": "Within bounds",
                "y": pytest.approx([int(r[2]) / int(r[1]) * 100 for r in expected]),
            },
            {"name": "Outliers", "y": []},
        ]

    def test_a_tenant_with_three_traces_has_no_outlier_bounds(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webfewtraces")
        for age in (1, 2, 3):
            record_trace(telemetry, tenant, SEARCH, 100 * age, minutes_ago=age)
        telemetry.force_flush(timeout_millis=10000)
        _show_traces(page, web_url, tenant, 3)
        outliers = _section(page, "Outliers")
        expect(outliers.locator("p.muted")).to_have_text(
            "Outliers need at least four traces."
        )


class TestExplorer:
    def test_search_scope_sorts_and_pages(self, page, web_url, telemetry):
        tenant = _tenant("webexplore")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        _show_traces(page, web_url, tenant, 6)
        explorer = _section(page, "Trace explorer")
        count = explorer.locator("p.muted")
        expect(count).to_have_text("6 traces")

        explorer.get_by_label("Search in").select_option("Trace ID")
        explorer.get_by_label("Trace ID", exact=True).fill(rows[4]["trace_id"][:12])
        expect(count).to_have_text("1 trace")
        assert [row[-1] for row in _rows(page, "Traces")] == [rows[4]["trace_id"]]
        explorer.get_by_label("Trace ID", exact=True).fill("dispatch")
        expect(count).to_have_text("0 traces")
        explorer.get_by_label("Search in").select_option("Operation")
        explorer.get_by_label("Operation", exact=True).fill("dispatch")
        expect(count).to_have_text("1 trace")
        explorer.get_by_label("Operation", exact=True).fill("")

        explorer.get_by_label("Order").select_option("Operation A-Z")
        assert [row[-1] for row in _rows(page, "Traces")] == [
            rows[5]["trace_id"],
            *[row["trace_id"] for row in rows[:5]],
        ]
        explorer.get_by_label("Order").select_option("Failed first")
        assert [row[-1] for row in _rows(page, "Traces")] == [
            rows[4]["trace_id"],
            *[row["trace_id"] for row in rows if row["error"] is None],
        ]

    def test_traces_page_twenty_at_a_time(self, page, web_url, telemetry):
        tenant = _tenant("webpages")
        recorded = [
            record_trace(telemetry, tenant, SEARCH, 10 + i, minutes_ago=i * 0.5 + 0.5)
            for i in range(25)
        ]
        telemetry.force_flush(timeout_millis=10000)
        _show_traces(page, web_url, tenant, 25)
        explorer = _section(page, "Trace explorer")
        pager = explorer.locator(".pager")
        expect(pager.locator("span")).to_have_text("Page 1 of 2")
        assert [row[-1] for row in _rows(page, "Traces")] == [
            trace_id for trace_id, _, _ in recorded[:20]
        ]
        expect(pager.get_by_role("button", name="Previous")).to_be_disabled()
        pager.get_by_role("button", name="Next").click()
        expect(pager.locator("span")).to_have_text("Page 2 of 2")
        assert [row[-1] for row in _rows(page, "Traces")] == [
            trace_id for trace_id, _, _ in recorded[20:]
        ]
        expect(pager.get_by_role("button", name="Next")).to_be_disabled()


class TestRootCauses:
    def test_root_cause_details_name_the_failures_and_slow_traces(
        self, page, web_url, telemetry, phoenix_container
    ):
        tenant = _tenant("webrcadetail")
        rows = record_sample_traces(telemetry, tenant)
        telemetry.force_flush(timeout_millis=10000)
        slow = rows[3]["trace_id"]
        _show_traces(page, web_url, tenant, 6)
        project = telemetry.config.get_project_name(tenant)
        project_id = httpx.get(
            f"{phoenix_container['http_endpoint']}/v1/projects/{project}"
        ).json()["data"]["id"]
        project_url = f"{phoenix_container['http_endpoint']}/projects/{project_id}"

        causes = _section(page, "Root causes")
        expect(causes.locator("details.explainer summary").first).to_have_text(
            "Phoenix query reference"
        )
        expect(causes.locator("p.muted")).to_have_text(
            "Find root causes runs over the traces the filters above keep."
        )
        form = causes.get_by_role("form", name="Find root causes")
        # Successful 50-400 ms: P95 380 ms; all six traces: P95 850 ms.
        expect(causes.get_by_label("Slow threshold")).to_have_text(
            "P95 of the successful traces: 380.0 ms; successful traces slower than "
            "this are flagged. P95 of all traces, failures included, is 850.0 ms."
        )
        form.get_by_label("Slow percentile").fill("75")
        expect(causes.get_by_label("Slow threshold")).to_have_text(
            "P75 of the successful traces: 300.0 ms; successful traces slower than "
            "this are flagged."
        )
        form.get_by_role("button", name="Find root causes").click()
        expect(page.locator('dl[aria-label="Root cause summary"]')).to_be_visible()
        assert _facts(page, "Root cause summary") == {
            "Traces analyzed": "6",
            "Total issues": "2",
            "Failed": "1 (16.7%)",
            "Slow": "1 (slower than 300.0 ms)",
        }
        expect(causes.get_by_label("Threshold comparison")).to_have_text(
            "P75 of the successful traces is 300.0 ms; P75 of all traces, failures "
            "included, is 375.0 ms. Failed traces do not count toward the slow "
            "threshold."
        )

        first = causes.get_by_role("list", name="Hypotheses").locator("details").first
        first.locator("summary").click()
        expect(first).to_contain_text(
            f"Suggested action: Optimize '{SEARCH}' operation or increase resources"
        )
        expect(first).to_contain_text(f"1 affected trace; sample: {slow[:8]}…")
        expect(
            first.get_by_label(re.compile("^Phoenix query for the traces of"))
        ).to_have_text(f'trace_id == "{slow}"')
        expect(
            first.get_by_role("link", name="Open the project in Phoenix")
        ).to_have_attribute("href", project_url)

        failures = causes.get_by_role("region", name="Failure analysis")
        expect(
            failures.get_by_role("link", name="View all 1 failed traces in Phoenix")
        ).to_have_attribute("href", project_url)
        expect(
            failures.get_by_label("Phoenix query for the failed traces")
        ).to_have_text('status_code == "ERROR"')
        assert [
            _rows(page, name)
            for name in (
                "Error types",
                "Failed operations",
                "Failed profiles",
                "Failed strategies",
            )
        ] == [
            [["unknown", "1", "100.0%"]],
            [[SEARCH, "1", "100.0%"]],
            [["video_colpali", "1", "100.0%"]],
            [["bm25", "1", "100.0%"]],
        ]
        hours = {}
        for row in rows:
            hour = pd.Timestamp(row["start_time"]).hour
            requests, fails = hours.get(hour, (0, 0))
            hours[hour] = (requests + 1, fails + (row["error"] is not None))
        assert _rows(page, "Hours with failures") == [
            [f"{hour:02d}:00", str(r), str(f), f"{f / r * 100:.1f}%"]
            for hour, (r, f) in sorted(hours.items())
            if f / r > 0.1
        ]

        performance = causes.get_by_role("region", name="Performance analysis")
        expect(
            performance.get_by_role("link", name="View 1 slow traces in Phoenix")
        ).to_have_attribute("href", project_url)
        expect(
            performance.get_by_label("Phoenix query for the slow traces")
        ).to_have_text("latency_ms > 300")
        assert _rows(page, "Slow operations") == [
            [SEARCH, "1", "400.0 ms", "400.0 ms", "400.0 ms"]
        ]
        normal = [50, 100, 200, 300]
        assert _facts(page, "Degradation") == {
            "Slow mean": "400.0 ms (σ 0.0 ms)",
            "Other successful mean": (
                f"162.5 ms (σ {statistics.pstdev(normal):.1f} ms)"
            ),
            "Slowdown": f"{400 / 162.5:.2f}×",
        }
        assert [_rows(page, "Slow profiles"), _rows(page, "Slow strategies")] == [
            [["video_colpali", "1", "100.0%"]],
            [["hybrid", "1", "100.0%"]],
        ]

    def test_a_failure_burst_reads_as_a_time_range_with_its_query(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webburst")
        recorded = [
            record_trace(
                telemetry,
                tenant,
                SEARCH,
                100,
                minutes_ago=age,
                error="request timed out",
            )
            for age in (3, 2, 1)
        ]
        record_trace(telemetry, tenant, SEARCH, 50, minutes_ago=4)
        telemetry.force_flush(timeout_millis=10000)
        _show_traces(page, web_url, tenant, 4)
        causes = _section(page, "Root causes")
        causes.get_by_label("Include slow traces").uncheck()
        causes.get_by_role("button", name="Find root causes").click()
        expect(page.locator('dl[aria-label="Root cause summary"]')).to_be_visible()

        starts = [pd.Timestamp(start, unit="ns", tz="UTC") for _, _, start in recorded]
        first, last = starts[0].to_pydatetime(), starts[2].to_pydatetime()
        same_day = first.strftime("%b %-d, %Y") == last.strftime("%b %-d, %Y")
        clock = lambda moment: moment.strftime("%-I:%M:%S %p")  # noqa: E731
        readable = (
            f"{first.strftime('%b %-d, %Y')}, {clock(first)} - {clock(last)}"
            if same_day
            else f"{first.strftime('%b %-d, %Y')}, {clock(first)} - "
            f"{last.strftime('%b %-d, %Y')}, {clock(last)}"
        )
        burst = causes.get_by_role("list", name="Hypotheses").locator(
            "details", has_text="Failure burst detected"
        )
        burst.locator("summary").click()
        expect(burst.get_by_role("list", name="Evidence").locator("li")).to_have_text(
            ["3 failures in 2.0 minutes", f"Time range: {readable}"]
        )
        assert [row[:4] for row in _rows(page, "Failure bursts")] == [
            ["1", "3", "2.0 min", readable]
        ]
        expect(causes.get_by_label("Phoenix query for burst 1")).to_have_text(
            f'timestamp >= "{first.isoformat()}" and timestamp <= "{last.isoformat()}"'
        )
        assert _rows(page, "Error types") == [["timeout", "3", "100.0%"]]

    def test_a_clean_window_says_so_and_a_section_switch_keeps_the_analysis(
        self, page, web_url, telemetry, phoenix_proxy
    ):
        tenant = _tenant("webrcaclean")
        for age in (1, 2, 3):
            record_trace(telemetry, tenant, SEARCH, 100, minutes_ago=age)
        telemetry.force_flush(timeout_millis=10000)
        _show_traces(page, web_url, tenant, 3)
        causes = _section(page, "Root causes")
        causes.get_by_label("Include slow traces").uncheck()
        causes.get_by_role("button", name="Find root causes").click()
        expect(causes.locator("p.ok-line")).to_have_text(
            "No failures or slow traces among the 3 analyzed: the failure rate is 0%."
        )
        reads = len(phoenix_proxy.requests)
        _section(page, "Overview")
        causes = _section(page, "Root causes")
        expect(causes.locator("p.ok-line")).to_have_text(
            "No failures or slow traces among the 3 analyzed: the failure rate is 0%."
        )
        assert len(phoenix_proxy.requests) == reads
