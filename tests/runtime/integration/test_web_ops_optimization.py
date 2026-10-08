"""The web client's Optimization runs view, driven in Chromium.

The built client's Node server forwards to the runtime's tenant router on a
real uvicorn socket, which submits, lists, cancels and retries Workflows on a
real Argo API server. No workflow controller runs there, so a test sets each
phase a controller would report and watches the page pick it up.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_runtime.config_loader import WorkflowSettings, get_workflow_settings
from cogniverse_runtime.routers import tenant as tenant_router
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER
from tests.utils.argo_api import argo_api_server, set_workflow_status
from tests.utils.k8s_api_server import _kubectl
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import register_tenant, serve_ops_runtime

pytestmark = [pytest.mark.integration]

# The page polls every 5 s and argo-server serves from an informer cache.
POLL_TIMEOUT_MS = 30_000


def _configure_workflow(api_url):
    get_workflow_settings._instance = WorkflowSettings(
        api_url=api_url,
        namespace="cogniverse",
        job_template="cogniverse-job-runner",
        optimization_template="cogniverse-optimization-runner",
    )


@pytest.fixture(scope="module")
def argo(tmp_path_factory):
    with argo_api_server(tmp_path_factory.mktemp("web-argo")) as cluster:
        yield cluster


@pytest.fixture(autouse=True)
def workflow_settings(argo):
    _configure_workflow(argo["url"])
    yield
    del get_workflow_settings._instance


@pytest.fixture(scope="module")
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        config_manager, schema_loader, workflow_state_redis_url
    ) as url:
        yield url


@pytest.fixture()
def web_url(built_client, runtime_url):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []


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


def _runs_view(page: Page, web_url: str, tenant: str) -> None:
    register_tenant(tenant)
    page.goto(f"{web_url}/#/ops/optimization")
    expect(
        page.get_by_role("heading", name="Optimization runs", level=1)
    ).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Show runs").click()
    expect(
        page.get_by_role("region", name=f"Optimization runs of {tenant}")
    ).to_be_visible()


def _start(page: Page, mode: str) -> str:
    form = page.get_by_role("form", name="Start optimization")
    form.get_by_label("Mode").select_option(mode)
    return _start_submitted(page, mode)


def _start_submitted(page: Page, mode: str) -> str:
    """Click Start run on the filled form; the started run's name."""
    page.get_by_role("form", name="Start optimization").get_by_role(
        "button", name="Start run"
    ).click()
    notice = page.get_by_role("status")
    expect(notice).to_contain_text(f"Started a {mode} run: manual-optimize-{mode}-")
    return re.fullmatch(
        rf"Started a {mode} run: (manual-optimize-{mode}-\w+)\.", notice.inner_text()
    ).group(1)


def _row(page: Page, tenant: str, name: str):
    return (
        page.get_by_role("region", name=f"Optimization runs of {tenant}")
        .get_by_role("row")
        .filter(has=page.get_by_role("button", name=name, exact=True))
    )


def _nodes(name: str, phase: str) -> dict:
    """A run's node tree as Argo's controller records it: the root steps
    node, its optimizer pod in ``phase``, and the profile step it skipped."""
    pod, skipped = f"{name}-1", f"{name}-2"
    return {
        name: {
            "id": name,
            "name": name,
            "displayName": name,
            "type": "Steps",
            "templateName": "main",
            "phase": phase,
            "children": [pod, skipped],
        },
        pod: {
            "id": pod,
            "name": f"{name}[0].run-optimizer",
            "displayName": "run-optimizer",
            "type": "Pod",
            "templateName": "run-optimizer",
            "boundaryID": name,
            "phase": phase,
        },
        skipped: {
            "id": skipped,
            "name": f"{name}[1].profile",
            "displayName": "profile",
            "type": "Skipped",
            "templateName": "run-optimizer",
            "boundaryID": name,
            "phase": "Omitted",
        },
    }


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


class TestSyntheticRun:
    def test_a_synthetic_run_carries_the_chosen_optimizers_and_lookback(
        self, page, web_url, argo
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        form = page.get_by_role("form", name="Start optimization")
        form.get_by_label("Mode").select_option("synthetic")
        choices = form.get_by_role("group", name="Generate training data for")
        expect(choices.get_by_role("checkbox")).to_have_count(
            len(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
        )
        expect(choices.locator("label")).to_have_text(
            sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
        )

        form.get_by_role("button", name="Start run").click()
        expect(form.get_by_role("alert")).to_have_text(
            "Choose the optimizers to generate data for."
        )
        form.get_by_label("Lookback hours").fill("0")
        choices.get_by_label("profile").check()
        form.get_by_role("button", name="Start run").click()
        expect(form.get_by_role("alert")).to_have_text(
            "Lookback hours must be a number above 0."
        )

        form.get_by_label("Lookback hours").fill("12")
        choices.get_by_label("routing").check()
        name = _start_submitted(page, "synthetic")
        stored = _workflow(argo, name)
        assert {
            p["name"]: p["value"] for p in stored["spec"]["arguments"]["parameters"]
        } == {
            "mode": "synthetic",
            "tenant-id": tenant,
            "lookback-hours": "12",
            "agents": "profile,routing",
        }


class TestOptimizationRuns:
    def test_an_operator_starts_watches_cancels_and_retries_a_run(
        self, page, web_url, argo
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        runs = page.get_by_role("region", name=f"Optimization runs of {tenant}")
        expect(runs.get_by_text(f"No optimization runs for {tenant}.")).to_be_visible()
        expect(
            page.get_by_role("form", name="Start optimization")
            .get_by_label("Mode")
            .locator("option")
        ).to_have_text(sorted(tenant_router._MANUAL_OPTIMIZE_MODES))

        name = _start(page, "simba")
        stored = _workflow(argo, name)
        assert (
            stored["metadata"]["labels"]["cogniverse.ai/mode"],
            {p["name"]: p["value"] for p in stored["spec"]["arguments"]["parameters"]}[
                "tenant-id"
            ],
        ) == ("simba", tenant)
        expect(_row(page, tenant, name).get_by_role("cell")).to_have_text(
            [name, "simba", "manual", "Pending", "—", "—"], timeout=POLL_TIMEOUT_MS
        )
        detail = page.get_by_role("region", name=f"Run {name}")
        expect(detail.locator("dt:text-is('Phase') + dd")).to_have_text("Pending")

        # The page follows the run as a controller would move it.
        set_workflow_status(
            argo["kubeconfig"],
            name,
            {
                "phase": "Running",
                "startedAt": "2026-10-04T09:00:00Z",
                "nodes": _nodes(name, "Running"),
            },
        )
        expect(_row(page, tenant, name).get_by_role("cell")).to_have_text(
            [name, "simba", "manual", "Running", "2026-10-04 09:00 UTC", "—"],
            timeout=POLL_TIMEOUT_MS,
        )
        steps = detail.get_by_role("table", name="Steps")
        expect(steps.get_by_role("cell")).to_have_text(
            ["run-optimizer", "Running", "profile", "Skipped"],
            timeout=POLL_TIMEOUT_MS,
        )
        expect(steps.get_by_role("columnheader")).to_have_text(["Step", "Phase"])

        detail.get_by_role("button", name="Cancel run").click()
        expect(page.get_by_role("status")).to_have_text(
            f"Cancelled {name}; Argo reports Running."
        )
        assert _workflow(argo, name)["spec"]["shutdown"] == "Terminate"

        set_workflow_status(
            argo["kubeconfig"],
            name,
            {
                "phase": "Failed",
                "startedAt": "2026-10-04T09:00:00Z",
                "finishedAt": "2026-10-04T09:10:00Z",
                "message": "Stopped with strategy 'Terminate'",
                "nodes": _nodes(name, "Failed"),
            },
        )
        detail = page.get_by_role("region", name=f"Run {name}")
        expect(detail.locator("dt:text-is('Phase') + dd")).to_have_text(
            "Failed", timeout=POLL_TIMEOUT_MS
        )
        expect(detail.locator("dt:text-is('Message') + dd")).to_have_text(
            "Stopped with strategy 'Terminate'"
        )
        expect(detail.get_by_role("button", name="Cancel run")).to_have_count(0)
        detail.get_by_role("button", name="Retry failed steps").click()
        expect(page.get_by_role("status")).to_contain_text(
            f"Retried {name}; Argo reports ", timeout=POLL_TIMEOUT_MS
        )
        retried = _workflow(argo, name)
        expect(page.get_by_role("status")).to_have_text(
            f"Retried {name}; Argo reports {retried['status'].get('phase') or 'no phase'}."
        )


class TestConcurrency:
    def test_two_operators_starting_runs_at_once_each_get_their_own(
        self, browser, web_url, argo
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        modes = ["simba", "profile"]
        contexts = [browser.new_context() for _ in modes]
        pages = [context.new_page() for context in contexts]
        try:
            for page, mode in zip(pages, modes):
                _runs_view(page, web_url, tenant)
                page.get_by_role("form", name="Start optimization").get_by_label(
                    "Mode"
                ).select_option(mode)
            for page in pages:
                page.get_by_role("form", name="Start optimization").get_by_role(
                    "button", name="Start run"
                ).click(no_wait_after=True)
            names = []
            for page, mode in zip(pages, modes):
                notice = page.get_by_role("status")
                expect(notice).to_contain_text(f"Started a {mode} run: ")
                names.append(
                    re.fullmatch(
                        r"Started a \S+ run: (\S+)\.", notice.inner_text()
                    ).group(1)
                )
            assert [
                _workflow(argo, name)["metadata"]["labels"]["cogniverse.ai/mode"]
                for name in names
            ] == modes
            for page in pages:
                page.get_by_role(
                    "region", name=f"Optimization runs of {tenant}"
                ).get_by_role("button", name="Refresh").click()
                for name in names:
                    expect(_row(page, tenant, name)).to_have_count(
                        1, timeout=POLL_TIMEOUT_MS
                    )
        finally:
            for context in contexts:
                context.close()


class TestFaultContract:
    def test_an_unreachable_argo_reads_as_an_outage_not_an_empty_history(
        self, page, web_url
    ):
        _configure_workflow(f"http://127.0.0.1:{free_port()}")
        tenant = "acme:production"
        _runs_view(page, web_url, tenant)
        runs = page.get_by_role("region", name=f"Optimization runs of {tenant}")
        expect(runs.get_by_role("alert")).to_have_text(
            f"Argo could not list the workflows of tenant {tenant}; retry."
        )
        expect(runs.get_by_role("table")).to_have_count(0)
        expect(runs.get_by_text(f"No optimization runs for {tenant}.")).to_have_count(0)
        form = page.get_by_role("form", name="Start optimization")
        form.get_by_role("button", name="Start run").click()
        expect(form.get_by_role("alert")).to_have_text(
            "The Argo API did not answer; retry."
        )

    def test_a_deployment_without_argo_says_so(self, page, web_url):
        _configure_workflow(None)
        tenant = "acme:production"
        _runs_view(page, web_url, tenant)
        expect(
            page.get_by_role(
                "region", name=f"Optimization runs of {tenant}"
            ).get_by_role("alert")
        ).to_have_text("Argo is not configured on this deployment.")
