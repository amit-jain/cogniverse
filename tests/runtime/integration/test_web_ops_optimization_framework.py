"""The web client's Optimization framework view, driven in Chromium.

Its routes run on a real uvicorn socket over real Phoenix (read through a
fault proxy), the production approval store on Phoenix and Redis, and a real
Argo API server. No workflow controller runs there: a synthetic run's pod is
played by the optimization CLI's own submit step against the real approval
store, and the Workflow's status is the one Argo's controller recorded for a
real run of the chart's template, carrying that outcome.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.unified_config import ApprovalConfig
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime.config_loader import WorkflowSettings, get_workflow_settings
from cogniverse_runtime.optimization_cli import submit_synthetic_outcome
from cogniverse_sdk.document import source_title_key
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER
from cogniverse_synthetic.schemas import SAMPLING_STRATEGIES, SyntheticDataResponse
from tests.utils.approval_review import review_config_manager, run_in_own_loop
from tests.utils.argo_api import (
    argo_api_server,
    recorded_optimizer_run,
    set_workflow_status,
)
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.k8s_api_server import _kubectl
from tests.utils.telemetry_metric_spans import (
    record_routing,
    record_search,
    record_trace,
)
from tests.utils.web_client import recording_telemetry_sink, serve_web
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

# The page polls every 5 s; Phoenix serves spans after a short indexing delay.
POLL_TIMEOUT_MS = 60_000
VIEW = "Optimization framework"


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
def argo(tmp_path_factory):
    with argo_api_server(tmp_path_factory.mktemp("web-framework-argo")) as cluster:
        yield cluster


@pytest.fixture(autouse=True)
def workflow_settings(argo):
    get_workflow_settings._instance = WorkflowSettings(
        api_url=argo["url"],
        namespace="cogniverse",
        job_template="cogniverse-job-runner",
        optimization_template="cogniverse-optimization-runner",
    )
    yield
    del get_workflow_settings._instance


@pytest.fixture(scope="module")
def review_config(phoenix_container, phoenix_proxy, workflow_state_redis_url):
    """The approval store reads Phoenix through ``phoenix_proxy`` too."""
    return review_config_manager(
        phoenix_container, workflow_state_redis_url, telemetry_url=phoenix_proxy.url
    )


@pytest.fixture(scope="module")
def runtime_url(review_config, schema_loader, workflow_state_redis_url, telemetry):
    with serve_ops_runtime(
        review_config, schema_loader, workflow_state_redis_url
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
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    yield page
    context.close()


def _tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _open(page: Page, web_url: str, tenant: str, tab: str) -> None:
    page.goto(f"{web_url}/#/ops/optimization-framework")
    expect(page.get_by_role("heading", name=VIEW, level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Open").click()
    page.get_by_role("navigation", name="Optimization sections").get_by_role(
        "button", name=tab, exact=True
    ).click()


def _fact(scope, term: str):
    return scope.locator(f"dt:text-is('{term}') + dd")


def _cells(table):
    return [
        row.locator("td").all_inner_texts() for row in table.locator("tbody tr").all()
    ]


def _workflow(argo, name: str) -> dict:
    got = _kubectl(
        argo["kubeconfig"],
        "get",
        "workflow.argoproj.io",
        name,
        "-n",
        "cogniverse",
        "-o",
        "json",
    )
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def _parameters(argo, name: str) -> dict:
    return {
        p["name"]: p["value"]
        for p in _workflow(argo, name)["spec"]["arguments"]["parameters"]
    }


def _finish(argo, name: str, outcome: dict | None, case: str = "succeeded") -> None:
    """Record ``name`` as Argo's controller recorded a run of ``case`` whose
    optimizer pod wrote ``outcome``."""
    set_workflow_status(
        argo["kubeconfig"], name, recorded_optimizer_run(case, name, outcome)
    )


SUNSET = "sunset over the sea"
HARBOUR = "boats in the harbour"
SUNSET_TITLES = ["Sunset Reel", "Beach Walk", "Night Sky", "Pier", "Waves", "Gulls"]


def _fetch_until(page: Page, want: int, timeout=90.0) -> None:
    form = page.get_by_role("form", name="Fetch searches")
    deadline = time.monotonic() + timeout
    while True:
        form.get_by_role("button", name="Fetch search results").click()
        status = page.get_by_role("status").filter(has_text="Fetched")
        expect(status).to_be_visible()
        if (
            status.inner_text() == f"Fetched {want} search results."
            or time.monotonic() > deadline
        ):
            return
        page.wait_for_timeout(2000)


class TestSearchAnnotationsAndGoldenDataset:
    def test_a_reviewer_rates_searches_builds_and_exports_a_golden_dataset(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webannotate")
        record_search(tenant, SUNSET, "video_colpali", "hybrid", SUNSET_TITLES)
        time.sleep(0.01)
        record_search(tenant, HARBOUR, "video_xclip", "bm25", ["Harbour Tour", "Ferry"])
        telemetry.force_flush(timeout_millis=10000)

        _open(page, web_url, tenant, "Search annotations")
        page.get_by_label("Annotation type").select_option("stars")
        _fetch_until(page, 2)
        queue = page.get_by_role("region", name="Annotation queue (2 results)")
        expect(queue.get_by_label("Page").locator("option")).to_have_text(["1"])
        sunset = queue.get_by_role("group", name=f"Result 2: {SUNSET}")
        sunset.locator("summary").click()
        expect(_fact(sunset, "Profile")).to_have_text("video_colpali")
        expect(_fact(sunset, "Strategy")).to_have_text("hybrid")
        expect(
            sunset.get_by_role("list", name="Top results").locator("li")
        ).to_have_text(SUNSET_TITLES[:5])
        rating = sunset.get_by_role("form", name="Your annotation")
        rating.get_by_label("Rating").fill("5")
        rating.get_by_label("Notes (optional)").fill("the reel is the answer")
        rating.get_by_role("button", name="Save rating").click()
        expect(rating.get_by_role("status").filter(has_text="Saved")).to_have_text(
            "Saved: 5 stars"
        )
        expect(_fact(sunset, "Current rating")).to_have_text(
            "positive (1.00) — the reel is the answer"
        )

        page.get_by_label("Annotation type").select_option("thumbs")
        harbour = queue.get_by_role("group", name=f"Result 1: {HARBOUR}")
        harbour.locator("summary").click()
        harbour.get_by_role("button", name="Bad").click()
        expect(harbour.get_by_role("status")).to_have_text("Saved: Thumbs down")

        page.get_by_role("navigation", name="Optimization sections").get_by_role(
            "button", name="Golden dataset"
        ).click()
        build = page.get_by_role("form", name="Build golden dataset")
        build.get_by_label("Lookback days").fill("1")
        deadline = time.monotonic() + 90
        while True:
            build.get_by_role("button", name="Build golden dataset").click()
            built = page.get_by_role("status").filter(has_text="Built a golden dataset")
            expect(built).to_be_visible()
            if built.inner_text() == "Built a golden dataset of 1 queries.":
                break
            assert time.monotonic() < deadline, built.inner_text()
            page.wait_for_timeout(2000)
        sample = page.get_by_role("table", name="Golden dataset sample")
        assert _cells(sample) == [[SUNSET, "5", "1.00"]]

        export = page.get_by_role("region", name="Export golden dataset")
        filename = f"golden_dataset_{tenant.replace(':', '_')}_{datetime.now(timezone.utc):%Y%m%d}.json"
        expect(export.get_by_label("Filename")).to_have_value(filename)
        with page.expect_download() as download:
            export.get_by_role("button", name="Download JSON").click()
        assert download.value.suggested_filename == filename
        exported = json.loads(open(download.value.path()).read())
        keys = [source_title_key(title) for title in SUNSET_TITLES[:5]]
        assert list(exported) == [SUNSET]
        assert (
            exported[SUNSET]["expected_videos"],
            exported[SUNSET]["relevance_scores"],
            exported[SUNSET]["avg_relevance"],
            exported[SUNSET]["profile"],
        ) == (
            keys,
            {key: 1 / (rank + 1) for rank, key in enumerate(keys)},
            1.0,
            "video_colpali",
        )

        page.get_by_role("navigation", name="Optimization sections").get_by_role(
            "button", name="Overview"
        ).click()
        expect(_fact(page, "Golden dataset size")).to_have_text("1")
        expect(_fact(page, "Total annotations")).to_have_text("2")
        expect(_fact(page, "Last optimization")).to_have_text("Never")

    def test_an_outage_reads_as_an_outage_not_as_no_searches(
        self, page, web_url, phoenix_proxy
    ):
        tenant = _tenant("weboutage")
        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        _open(page, web_url, tenant, "Search annotations")
        form = page.get_by_role("form", name="Fetch searches")
        form.get_by_role("button", name="Fetch search results").click()
        expect(form.get_by_role("alert")).to_have_text(
            f"Could not read the recorded searches of tenant {tenant}."
        )
        expect(page.get_by_text("Fetched")).to_have_count(0)
        page.get_by_role("navigation", name="Optimization sections").get_by_role(
            "button", name="Metrics"
        ).click()
        expect(
            page.get_by_role("region", name="Optimization metrics").get_by_role("alert")
        ).to_have_text(
            f"Optimization metrics are unavailable: Could not read the spans of tenant {tenant}."
        )


ENHANCEMENTS = [
    {
        "query": f"pytorch tutorial {index}",
        "enhanced_query": f"pytorch framework tutorial {index}",
        "expansion_terms": ["framework"],
        "synonyms": ["deep learning library"],
        "context": "PyTorch education",
        "reasoning": f"Named the framework for tutorial {index}",
    }
    for index in range(2)
]


def _response(optimizer: str, data: list[dict]) -> SyntheticDataResponse:
    return SyntheticDataResponse(
        optimizer=optimizer,
        schema_name="QueryEnhancementExampleSchema",
        count=len(data),
        selected_profiles=["video_colpali_smol500_mv_frame", "document_text_semantic"],
        profile_selection_reasoning="Frame and text profiles cover the tutorials",
        data=data,
        metadata={"generation_time_ms": 1250},
    )


def _pod_outcome(review_config, telemetry, tenant, human_review):
    """What the synthetic run's pod does after generation: the CLI's submit
    step against the tenant's real approval store."""
    storage = ApprovalStorageImpl.from_system_config(review_config, telemetry, tenant)
    outcome = run_in_own_loop(
        submit_synthetic_outcome(
            storage,
            tenant_id=tenant,
            optimizer_type="query_enhancement",
            response=_response("query_enhancement", ENHANCEMENTS),
            human_review=human_review,
        )
    )
    return {"status": "success", "results": {"query_enhancement": outcome}}


def _submit_synthetic(client, tenant: str) -> str:
    response = client.post(
        f"/admin/tenant/{tenant}/optimize",
        json={
            "mode": "synthetic",
            "optimizers": ["query_enhancement"],
            "options": {"count": 2},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["workflow_name"]


class TestSyntheticRun:
    def test_an_operator_generates_follows_and_exports_a_synthetic_run(
        self, page, web_url, argo, review_config, telemetry
    ):
        tenant = _tenant("websynthetic")
        _open(page, web_url, tenant, "Synthetic data")
        form = page.get_by_role("form", name="Generate synthetic data")
        expect(
            form.get_by_label("Optimizer", exact=True).locator("option")
        ).to_have_text(sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER))
        expect(form.get_by_label("Optimizer", exact=True)).to_have_value("profile")
        threshold = ApprovalConfig().confidence_threshold
        expect(_fact(form, "Auto-approval threshold")).to_have_text(f"{threshold:.2f}")
        expect(_fact(form, "Expected review rate")).to_have_text(
            f"~{round((1 - threshold) * 100)}%"
        )
        form.get_by_label("Optimizer", exact=True).select_option("query_enhancement")
        form.get_by_label("Examples to generate").fill("2")
        form.get_by_label("Backend sample size").fill("30")
        form.locator("summary", has_text="Advanced options").click()
        expect(form.get_by_label("Sampling strategy").locator("option")).to_have_text(
            ["The optimizer's own", *sorted(SAMPLING_STRATEGIES)]
        )
        form.get_by_label("Sampling strategy").select_option("temporal_recent")
        form.get_by_label("Max profiles").fill("4")
        form.get_by_role("button", name="Generate synthetic data").click()
        notice = form.get_by_role("status")
        expect(notice).to_contain_text("Generating 2 examples for query_enhancement: ")
        name = re.fullmatch(
            r"Generating 2 examples for query_enhancement: (\S+)\.", notice.inner_text()
        ).group(1)
        assert _parameters(argo, name) == {
            "mode": "synthetic",
            "tenant-id": tenant,
            "lookback-hours": "48",
            "agents": "query_enhancement",
            "options": json.dumps(
                {
                    "count": 2,
                    "vespa_sample_size": 30,
                    "strategy": "temporal_recent",
                    "max_profiles": 4,
                    "human_review": True,
                },
                separators=(",", ":"),
            ),
        }
        results = page.get_by_role("region", name=f"Results of {name}")
        expect(results.get_by_role("status")).to_have_text(
            "The run is Pending; results appear when it finishes."
        )

        _finish(argo, name, _pod_outcome(review_config, telemetry, tenant, True))
        outcome = results.get_by_role("region", name="Outcome for query_enhancement")
        expect(outcome.get_by_role("status").first).to_have_text(
            "Generated 2 examples: 0 auto-approved, 2 awaiting review.",
            timeout=POLL_TIMEOUT_MS,
        )
        expect(_fact(outcome, "Auto-approved")).to_have_text("0")
        expect(_fact(outcome, "Pending review")).to_have_text("2")
        expect(_fact(outcome, "Average confidence")).to_have_text("0.00")
        expect(outcome.locator("p", has_text="Profile selection:")).to_have_text(
            "Profile selection: Frame and text profiles cover the tutorials"
        )
        expect(_fact(outcome, "Schema")).to_have_text("QueryEnhancementExampleSchema")
        expect(_fact(outcome, "Generation time")).to_have_text("1250ms")
        expect(_fact(outcome, "Profiles used")).to_have_text("2")
        expect(
            outcome.get_by_role("list", name="Selected profiles").locator("li")
        ).to_have_text(["video_colpali_smol500_mv_frame", "document_text_semantic"])
        sample = outcome.get_by_role("table", name="Sample generated examples")
        expect(sample.get_by_role("columnheader")).to_have_text(list(ENHANCEMENTS[0]))
        assert _cells(sample) == [
            [
                example["query"],
                example["enhanced_query"],
                '["framework"]',
                '["deep learning library"]',
                "PyTorch education",
                example["reasoning"],
            ]
            for example in ENHANCEMENTS
        ]
        first = outcome.get_by_role(
            "group", name="Item 1/2 - Confidence: 0.00 - pytorch tutorial 0"
        )
        expect(_fact(first, "Generated query")).to_have_text("pytorch tutorial 0")
        expect(_fact(first, "Reasoning")).to_have_text(
            "Named the framework for tutorial 0"
        )
        expect(_fact(first, "Entities")).to_have_text("No entities")
        expect(_fact(first, "Retries")).to_have_text("0")
        expect(_fact(first, "Band")).to_have_text("Very low")
        expect(outcome.get_by_role("link", name="Approvals")).to_have_attribute(
            "href", "#/ops/approvals"
        )

        export = outcome.get_by_role("group", name="Export query_enhancement")
        expect(export.get_by_label("Filename")).to_have_value(
            re.compile(r"^synthetic_query_enhancement_\d{8}_\d{6}\.json$")
        )
        with page.expect_download() as download:
            export.get_by_role("button", name="Download JSON").click()
        exported = json.loads(open(download.value.path()).read())
        assert {k: v for k, v in exported.items() if k != "metadata"} == {
            "optimizer": "query_enhancement",
            "schema_name": "QueryEnhancementExampleSchema",
            "count": 2,
            "selected_profiles": [
                "video_colpali_smol500_mv_frame",
                "document_text_semantic",
            ],
            "profile_selection_reasoning": "Frame and text profiles cover the tutorials",
            "data": ENHANCEMENTS,
        }
        assert {k: v for k, v in exported["metadata"].items() if k != "batch_id"} == {
            "generation_time_ms": 1250,
            "auto_approved": 0,
            "pending_review": 2,
        }

        runs = page.get_by_role("region", name="Synthetic runs")
        runs.get_by_role("button", name="Refresh").click()
        expect(runs.get_by_label("Synthetic run").locator("option")).to_have_text(
            ["Choose a run", f"{name} (Succeeded)"]
        )

    def test_without_review_every_example_is_approved_and_a_failure_is_named(
        self, page, web_url, argo, review_config, telemetry
    ):
        tenant = _tenant("websynthauto")
        _open(page, web_url, tenant, "Synthetic data")
        form = page.get_by_role("form", name="Generate synthetic data")
        form.get_by_label("Optimizer", exact=True).select_option("query_enhancement")
        form.get_by_label("Examples to generate").fill("2")
        form.get_by_label("Human-in-the-loop review").uncheck()
        expect(
            form.get_by_text(
                "Without review every generated example is approved for training."
            )
        ).to_be_visible()
        form.get_by_role("button", name="Generate synthetic data").click()
        notice = form.get_by_role("status")
        expect(notice).to_contain_text("Generating 2 examples for query_enhancement: ")
        name = notice.inner_text().split(": ")[1].rstrip(".")
        assert json.loads(_parameters(argo, name)["options"])["human_review"] is False

        printed = _pod_outcome(review_config, telemetry, tenant, False)
        printed["results"]["routing"] = {
            "status": "failed",
            "error": "GLiNER inference endpoint is required for synthetic routing",
        }
        printed["status"] = "failed"
        _finish(argo, name, printed, "mixed")
        results = page.get_by_role("region", name=f"Results of {name}")
        approved = results.get_by_role("region", name="Outcome for query_enhancement")
        expect(approved.get_by_role("status").first).to_have_text(
            "Generated 2 examples: 2 auto-approved, 0 awaiting review.",
            timeout=POLL_TIMEOUT_MS,
        )
        expect(approved.get_by_role("status").last).to_have_text(
            "All items were approved automatically; no review is needed."
        )
        failed = results.get_by_role("region", name="Outcome for routing")
        expect(failed.get_by_role("alert")).to_have_text(
            "Generation for routing failed: GLiNER inference endpoint is required for synthetic routing"
        )

    def test_a_failed_a_killed_and_a_cancelled_run_each_say_what_happened(
        self, page, web_url, runtime_url, argo
    ):
        tenant = _tenant("websynthend")
        with httpx.Client(base_url=runtime_url, timeout=120) as client:
            raised, killed, cancelled = (
                _submit_synthetic(client, tenant) for _ in range(3)
            )
            stopped = client.post(
                f"/admin/tenant/{tenant}/optimize/runs/{cancelled}/cancel"
            )
            assert stopped.status_code == 200, stopped.text
        _finish(argo, raised, None, "raised")
        _finish(argo, killed, None, "killed")
        _finish(argo, cancelled, None, "cancelled_pending")

        _open(page, web_url, tenant, "Synthetic data")
        runs = page.get_by_role("region", name="Synthetic runs")
        chooser = runs.get_by_label("Synthetic run")
        expect(chooser.locator("option")).to_have_count(4)
        # The runs list orders them; here each carries its phase.
        assert sorted(chooser.locator("option").all_inner_texts()) == sorted(
            [
                "Choose a run",
                f"{raised} (Failed)",
                f"{killed} (Failed)",
                f"{cancelled} (Cancelled)",
            ]
        )

        chooser.select_option(raised)
        results = page.get_by_role("region", name=f"Results of {raised}")
        expect(results.get_by_role("alert")).to_have_text(
            "The run failed: ValueError: synthetic optimizer types have no "
            "approved training-data consumer: ['bogus']"
        )
        expect(results.locator("section")).to_have_count(0)

        chooser.select_option(killed)
        results = page.get_by_role("region", name=f"Results of {killed}")
        expect(results.get_by_role("alert")).to_have_text(
            f"The run's results are unavailable: Run {killed} ended Failed without "
            "reporting its outcome: Error (exit code 137)."
        )

        chooser.select_option(cancelled)
        results = page.get_by_role("region", name=f"Results of {cancelled}")
        expect(results.locator("p.muted")).to_have_text(
            "The run was cancelled before it reported an outcome."
        )
        expect(results.get_by_role("alert")).to_have_count(0)


class TestModuleOptimization:
    def test_an_operator_trains_the_unified_module_on_an_uploaded_golden_set(
        self, page, web_url, argo, tmp_path
    ):
        tenant = _tenant("webmodule")
        _open(page, web_url, tenant, "Module optimization")
        form = page.get_by_role("form", name="Optimize a module")
        expect(form.get_by_label("Module to optimize").locator("option")).to_have_text(
            ["Routing", "Workflow", "Unified"]
        )
        form.get_by_label("Module to optimize").select_option("unified")
        expect(form.get_by_text("Routing, then workflow.")).to_be_visible()
        form.get_by_label("Max iterations").fill("7")
        form.get_by_label("Lookback hours").fill("6")
        form.get_by_label("Use synthetic data").uncheck()
        golden = form.get_by_role("group", name="Golden dataset")
        expect(
            golden.get_by_text(
                "The tenant has no datasets yet. Upload a CSV to create one."
            )
        ).to_be_visible()
        csv = tmp_path / "golden.csv"
        csv.write_text(
            'query,expected_videos\nsunset over the sea,sunset_reel\nboats,"harbour,ferry"\n'
        )
        upload = golden.get_by_role("group", name="Upload CSV dataset")
        upload.get_by_label("CSV dataset").set_input_files(str(csv))
        upload.get_by_label("Dataset name").fill("golden_eval")
        upload.get_by_role("button", name="Upload dataset").click()
        dataset = f"golden_eval-{tenant}"
        expect(upload.get_by_role("status")).to_have_text(
            f"Created dataset {dataset} with 2 queries."
        )
        expect(golden.get_by_label("Telemetry dataset")).to_have_value(dataset)
        expect(_fact(golden, "Dataset size")).to_have_text("2")
        form.get_by_role("button", name="Submit module optimization").click()
        submitted = page.get_by_role(
            "region", name=re.compile(r"^Run manual-optimize-unified-")
        )
        expect(submitted.get_by_role("status")).to_contain_text(
            "Submitted manual-optimize-unified-"
        )
        name = (
            submitted.get_by_role("status")
            .inner_text()
            .removeprefix("Submitted ")
            .rstrip(".")
        )
        assert _parameters(argo, name) == {
            "mode": "unified",
            "tenant-id": tenant,
            "lookback-hours": "6",
            "options": json.dumps(
                {
                    "max_iterations": 7,
                    "use_synthetic_data": False,
                    "dataset_name": dataset,
                },
                separators=(",", ":"),
            ),
        }
        expect(_fact(submitted, "Phase")).to_have_text("Pending")
        expect(
            submitted.get_by_role("link", name="Optimization runs")
        ).to_have_attribute("href", "#/ops/optimization")

    def test_a_dataset_store_outage_offers_the_manual_name(
        self, page, web_url, argo, phoenix_proxy
    ):
        tenant = _tenant("webmoduleoutage")
        phoenix_proxy.intercept = lambda method, path, body: (503, {"detail": "down"})
        _open(page, web_url, tenant, "Module optimization")
        form = page.get_by_role("form", name="Optimize a module")
        form.get_by_label("Use synthetic data").uncheck()
        golden = form.get_by_role("group", name="Golden dataset")
        expect(golden.get_by_role("alert")).to_have_text(
            f"The tenant's datasets are unavailable: Could not read the datasets of tenant {tenant}."
        )
        golden.get_by_label("Dataset name (manual)").fill(f"golden_eval-{tenant}")
        form.get_by_role("button", name="Submit module optimization").click()
        submitted = page.get_by_role(
            "region", name=re.compile(r"^Run manual-optimize-routing-")
        )
        name = (
            submitted.get_by_role("status")
            .inner_text()
            .removeprefix("Submitted ")
            .rstrip(".")
        )
        assert json.loads(_parameters(argo, name)["options"]) == {
            "max_iterations": 100,
            "use_synthetic_data": False,
            "dataset_name": f"golden_eval-{tenant}",
        }


def _temporal(index):
    return f"what happened before the goal in match number {index}"


class TestProfileSelection:
    def test_the_recommender_is_analyzed_trained_and_asked(
        self, page, web_url, telemetry, runtime_url
    ):
        tenant = _tenant("webrecommend")
        for index in range(20):
            record_search(
                tenant, _temporal(index), "video_colpali_temporal", "hybrid", ["A"]
            )
            record_search(
                tenant, f"red car {index}", "video_xclip_plain", "hybrid", ["B"]
            )
        telemetry.force_flush(timeout_millis=10000)

        _open(page, web_url, tenant, "Profile selection")
        analyze = page.get_by_role("form", name="Analyze search spans")
        analyze.get_by_label("Lookback days").fill("1")
        deadline = time.monotonic() + 90
        while True:
            analyze.get_by_role("button", name="Analyze search spans").click()
            found = page.get_by_role("status").filter(has_text="search spans")
            expect(found).to_be_visible()
            if found.inner_text() == "Found 40 search spans.":
                break
            assert time.monotonic() < deadline, found.inner_text()
            page.wait_for_timeout(2000)
        usage = page.get_by_role("figure", name="attributes.profile", exact=True)
        assert list(
            zip(
                usage.locator(".bar-label").all_inner_texts(),
                usage.locator(".bar-value").all_inner_texts(),
                strict=True,
            )
        ) == [("video_colpali_temporal", "20"), ("video_xclip_plain", "20")]
        assert _cells(
            page.get_by_role("table", name="attributes.profile by attributes.top_score")
        ) == [["video_colpali_temporal", "1", "20"], ["video_xclip_plain", "1", "20"]]

        training = page.get_by_role("region", name="Train profile selector")
        training.get_by_role("button", name="Train profile selector model").click()
        expect(training.get_by_role("status").last).to_have_text(
            "Model trained and stored: 40 samples, 6 features, 2 profiles "
            "(video_colpali_temporal, video_xclip_plain).",
            timeout=POLL_TIMEOUT_MS,
        )
        expect(_fact(training, "Training accuracy")).to_have_text("100.0%")
        expect(_fact(training, "Test accuracy")).to_have_text("100.0%")
        expect(_fact(training, "Samples")).to_have_text("40")
        expect(training.get_by_role("status").first).to_have_text(
            f"A trained model is stored for {tenant}: video_colpali_temporal, video_xclip_plain."
        )

        query = _temporal(99)
        with httpx.Client(base_url=runtime_url, timeout=60) as client:
            predicted = client.post(
                f"/admin/tenant/{tenant}/profile-selection/predict",
                json={"query": query},
            ).json()
        assert predicted["profile"] == "video_colpali_temporal"
        ask = page.get_by_role("form", name="Predict a profile")
        ask.get_by_label("Test query").fill(query)
        ask.get_by_role("button", name="Load model and predict").click()
        region = page.get_by_role("region", name="Test prediction")
        expect(region.get_by_role("status")).to_have_text(
            "Recommended profile: video_colpali_temporal "
            f"(confidence: {predicted['confidence'] * 100:.1f}%)"
        )
        assert _cells(region.get_by_role("table", name="Extracted features")) == [
            ["query_length", str(len(query))],
            ["word_count", "9"],
            ["has_temporal_keywords", "1"],
            ["has_spatial_keywords", "0"],
            ["has_object_keywords", "1"],
            ["avg_word_length", f"{sum(len(w) for w in query.split()) / 9:.2f}"],
        ]


class TestMetricsAndReranking:
    def test_routing_scores_evaluation_and_training_activity_show(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webmetrics")
        # Timed by their spans only, as the gateway records them.
        for minutes, duration_ms in ((1, 200), (2, 300), (3, 400)):
            record_routing(
                telemetry,
                tenant,
                "search_agent",
                0.9,
                duration_ms,
                minutes_ago=minutes,
            )
        record_routing(
            telemetry, tenant, "search_agent", 0.3, 500, minutes_ago=4, failed=True
        )
        record_trace(telemetry, tenant, "cogniverse.optimization", 10, minutes_ago=5)
        telemetry.force_flush(timeout_millis=10000)

        _open(page, web_url, tenant, "Metrics")
        panel = page.get_by_role("region", name="Routing optimization metrics")
        deadline = time.monotonic() + 90
        while not (
            panel.count()
            and _fact(panel, "Total decisions").count()
            and _fact(panel, "Total decisions").inner_text() == "4"
        ):
            assert time.monotonic() < deadline
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Refresh metrics").click()
        expect(_fact(panel, "Routing accuracy")).to_have_text("75.0%")
        expect(_fact(panel, "Average routing latency")).to_have_text("350ms")
        expect(_fact(panel, "Confidence calibration")).to_have_text("1.000")
        assert _cells(panel.get_by_role("table", name="Per-agent performance")) == [
            ["search_agent", "0.750", "1.000", f"{2 * 0.75 / 1.75:.3f}"]
        ]
        expect(
            page.get_by_role("region", name="Search quality metrics").get_by_text(
                "No search evaluation spans. Run search evaluations to see NDCG metrics."
            )
        ).to_be_visible()
        expect(
            page.get_by_role("figure", name="Training activity over time")
        ).to_be_visible()

    def test_reranking_counts_the_annotations_against_its_minimum(
        self, page, web_url, telemetry
    ):
        tenant = _tenant("webrerank")
        _open(page, web_url, tenant, "Reranking")
        panel = page.get_by_role("region", name="Reranking optimization")
        expect(_fact(panel, "Current annotations")).to_have_text("0")
        expect(panel.get_by_role("status")).to_have_text(
            "50 more annotations are needed to reach 50."
        )
        panel.get_by_label("Minimum annotations").fill("0")
        expect(panel.get_by_role("status")).to_have_text(
            "0 annotations meet the minimum of 0. No reranker trainer exists; the annotations "
            "feed the golden dataset and the profile recommender."
        )


class TestSyntheticResultsRoute:
    """The synthetic results route on the same runtime, without a browser."""

    def _submit(self, client, tenant):
        return _submit_synthetic(client, tenant)

    def test_every_way_a_run_ends_reads_as_argo_recorded_it(self, runtime_url, argo):
        tenant = _tenant("synthends")
        with httpx.Client(base_url=runtime_url, timeout=120) as client:
            names = {
                case: self._submit(client, tenant)
                for case in (
                    "mixed",
                    "raised",
                    "killed",
                    "cancelled_running",
                    "cancelled_pending",
                    "garbled",
                )
            }
            for case in ("cancelled_running", "cancelled_pending"):
                stopped = client.post(
                    f"/admin/tenant/{tenant}/optimize/runs/{names[case]}/cancel"
                )
                assert stopped.status_code == 200, stopped.text
            _finish(
                argo,
                names["mixed"],
                {
                    "status": "failed",
                    "results": {
                        "query_enhancement": {
                            "status": "no_data",
                            "examples_generated": 0,
                            "schema_name": "QueryEnhancementExampleSchema",
                            "selected_profiles": ["document_text_semantic"],
                            "profile_selection_reasoning": "Text covers it",
                            "generation_time_ms": 812.4,
                        },
                        "routing": {
                            "status": "failed",
                            "error": "SyntheticDataService generated 0 examples "
                            "but request count is 2",
                        },
                    },
                },
                "mixed",
            )
            for case in ("raised", "killed", "cancelled_running", "cancelled_pending"):
                _finish(argo, names[case], None, case)
            garbled = recorded_optimizer_run("succeeded", names["garbled"])
            [node] = garbled["nodes"].values()
            node["outputs"]["parameters"][0]["value"] = "Segmentation fault"
            set_workflow_status(argo["kubeconfig"], names["garbled"], garbled)
            read = {
                case: client.get(
                    f"/admin/tenant/{tenant}/optimize/runs/{name}/synthetic"
                )
                for case, name in names.items()
            }

        def body(case, **fields):
            return {
                "workflow_name": names[case],
                "settled": True,
                "status": None,
                "error": None,
                "parameters": {
                    "optimizers": ["query_enhancement"],
                    "count": 2,
                    "vespa_sample_size": 200,
                    "strategy": None,
                    "max_profiles": 3,
                    "human_review": True,
                },
                "outcomes": [],
                **fields,
            }

        empty = {
            "batch_id": None,
            "auto_approved": 0,
            "pending_review": 0,
            "avg_confidence": None,
            "items": [],
        }
        garbled = read.pop("garbled")
        assert (garbled.status_code, garbled.json()["detail"]["error"]) == (
            502,
            "synthetic_result_unreadable",
        )
        assert {case: (r.status_code, r.json()) for case, r in read.items()} == {
            "mixed": (
                200,
                body(
                    "mixed",
                    phase="Failed",
                    status="failed",
                    outcomes=[
                        {
                            **empty,
                            "optimizer": "query_enhancement",
                            "status": "no_data",
                            "error": None,
                            "schema_name": "QueryEnhancementExampleSchema",
                            "selected_profiles": ["document_text_semantic"],
                            "profile_selection_reasoning": "Text covers it",
                            "generation_time_ms": 812.4,
                            "examples_generated": 0,
                        },
                        {
                            **empty,
                            "optimizer": "routing",
                            "status": "failed",
                            "error": "SyntheticDataService generated 0 examples "
                            "but request count is 2",
                            "schema_name": None,
                            "selected_profiles": [],
                            "profile_selection_reasoning": None,
                            "generation_time_ms": None,
                            "examples_generated": 0,
                        },
                    ],
                ),
            ),
            "raised": (
                200,
                body(
                    "raised",
                    phase="Failed",
                    status="failed",
                    error="ValueError: synthetic optimizer types have no approved "
                    "training-data consumer: ['bogus']",
                ),
            ),
            "killed": (
                502,
                {
                    "detail": f"Run {names['killed']} ended Failed without "
                    "reporting its outcome: Error (exit code 137)."
                },
            ),
            "cancelled_running": (
                200,
                body("cancelled_running", phase="Cancelled"),
            ),
            "cancelled_pending": (
                200,
                body("cancelled_pending", phase="Cancelled"),
            ),
        }

    def test_runs_read_at_once_each_answer_their_own_batch(
        self, runtime_url, argo, review_config, telemetry
    ):
        tenants = [_tenant("synthconcurrent"), _tenant("synthconcurrent")]
        with httpx.Client(base_url=runtime_url, timeout=120) as client:
            names = [self._submit(client, tenant) for tenant in tenants]
        printed = {}
        for tenant, name, review in zip(tenants, names, (True, False), strict=True):
            printed[name] = _pod_outcome(review_config, telemetry, tenant, review)
            _finish(argo, name, printed[name])

        async def read_all():
            async with httpx.AsyncClient(base_url=runtime_url, timeout=120) as client:
                return await asyncio.gather(
                    *(
                        client.get(
                            f"/admin/tenant/{tenant}/optimize/runs/{name}/synthetic"
                        )
                        for tenant, name in zip(tenants * 3, names * 3, strict=True)
                    )
                )

        responses = run_in_own_loop(read_all())
        assert [r.status_code for r in responses] == [200] * 6
        for response, name in zip(responses, names * 3, strict=True):
            [outcome] = response.json()["outcomes"]
            batch = printed[name]["results"]["query_enhancement"]["batch_id"]
            assert (
                outcome["batch_id"],
                [item["item_id"] for item in outcome["items"]],
                [item["status"] for item in outcome["items"]],
            ) == (
                batch,
                [f"{batch}_0", f"{batch}_1"],
                ["pending_review"] * 2 if name == names[0] else ["auto_approved"] * 2,
            )

    def test_an_unsettled_run_a_missing_batch_and_a_dead_store_are_named(
        self, runtime_url, argo, review_config, telemetry, phoenix_proxy
    ):
        tenant = _tenant("synthfaults")
        with httpx.Client(base_url=runtime_url, timeout=120) as client:
            name = self._submit(client, tenant)
            pending = client.get(
                f"/admin/tenant/{tenant}/optimize/runs/{name}/synthetic"
            )
            _finish(
                argo,
                name,
                {
                    "status": "success",
                    "results": {
                        "query_enhancement": {
                            "status": "success",
                            "batch_id": "synthetic_query_enhancement_missing",
                        }
                    },
                },
            )
            missing = client.get(
                f"/admin/tenant/{tenant}/optimize/runs/{name}/synthetic"
            )
            other = client.get(
                f"/admin/tenant/{_tenant('synthother')}/optimize/runs/{name}/synthetic"
            )
            _finish(argo, name, _pod_outcome(review_config, telemetry, tenant, True))
            phoenix_proxy.intercept = lambda method, path, body: (
                503,
                {"detail": "down"},
            )
            down = client.get(f"/admin/tenant/{tenant}/optimize/runs/{name}/synthetic")
            phoenix_proxy.intercept = None

        assert (
            pending.status_code,
            pending.json()["settled"],
            pending.json()["outcomes"],
        ) == (
            200,
            False,
            [],
        )
        assert pending.json()["parameters"] == {
            "optimizers": ["query_enhancement"],
            "count": 2,
            "vespa_sample_size": 200,
            "strategy": None,
            "max_profiles": 3,
            "human_review": True,
        }
        assert (missing.status_code, missing.json()) == (
            502,
            {
                "detail": "The run reported synthetic batch synthetic_query_enhancement_missing, "
                f"which the approval store of tenant {tenant} does not hold."
            },
        )
        assert (other.status_code, other.json()) == (
            404,
            {"detail": "Workflow not found"},
        )
        assert (down.status_code, down.json()["detail"]["error"]) == (
            502,
            "approval_store_unavailable",
        )
