"""The web client's Profile metrics and RLM A/B views, driven in Chromium
against spans in real Phoenix.

Spans are recorded the way their producers record them, with fixed values
(``tests/utils/telemetry_metric_spans``). The runtime reads them through a
forwarding proxy in front of Phoenix's HTTP API, so a test can fail the reads.
"""

from __future__ import annotations

import re
import time
from uuid import uuid4

import pandas as pd
import pytest
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.optimization_cli import emit_ab_compare_span
from tests.utils.approval_review import review_config_manager
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import (
    SEARCH,
    ab_result,
    record_profile_selection,
    record_sample_traces,
)
from tests.utils.web_client import (
    build_web_client,
    install_web_client,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

KEY = "web-ops-harness-key"


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
def built_client(tmp_path_factory):
    return build_web_client(install_web_client(tmp_path_factory.mktemp("web_ops")))


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
            built_client, runtime_url, KEY, telemetry_url=sink_url, built=True
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
    expect(panel.get_by_text("No profile selections in this window.")).to_have_count(0)


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
    assert [row[1:] for row in rows_shown] == [
        ["third question", "podcasts", "+600.0", "+10.0", "-0.500", "no"],
        ["second question", "lectures", "+400.0", "+50.0", "+0.250", "yes"],
        ["first question", "lectures", "+200.0", "+30.0", "+0.250", "no"],
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
        "No comparisons in this window. Run cogniverse-optim --mode ab-compare "
        "for this tenant to record some."
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
    page.get_by_label("Trace ID or operation").fill("DISPATCH")
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
    page.get_by_label("Trace ID or operation").fill("")
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


def test_analytics_shows_an_outage_rather_than_no_traces(page, web_url, phoenix_proxy):
    tenant = _tenant("webtracesoutage")
    phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
    _show(page, web_url, "analytics", "Analytics", "Show traces", tenant)
    region = page.get_by_role("region", name=f"Traces of {tenant}", exact=True)
    expect(region.get_by_role("alert")).to_have_text(
        f"Could not read the traces of tenant {tenant}."
    )
    expect(region.get_by_text("No traces match in this window.")).to_have_count(0)


def test_analytics_of_a_quiet_tenant_says_there_are_no_traces(page, web_url, telemetry):
    tenant = _tenant("webtracesempty")
    _show(page, web_url, "analytics", "Analytics", "Show traces", tenant)
    region = page.get_by_role("region", name=f"Traces of {tenant}", exact=True)
    expect(region.locator("p.muted")).to_have_text("No traces match in this window.")
    expect(page.get_by_role("navigation", name="Analytics sections")).to_have_count(0)
