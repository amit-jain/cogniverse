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
from tests.utils.argo_api import argo_api_server, set_workflow_status
from tests.utils.k8s_api_server import _kubectl
from tests.utils.web_client import (
    build_web_client,
    free_port,
    install_web_client,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration]

KEY = "web-ops-harness-key"
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
def built_client(tmp_path_factory):
    return build_web_client(install_web_client(tmp_path_factory.mktemp("web_ops")))


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
            built_client, runtime_url, KEY, telemetry_url=sink_url, built=True
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
    form.get_by_role("button", name="Start run").click()
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
                "nodes": {
                    "a": {
                        "type": "Pod",
                        "displayName": "run-optimizer",
                        "phase": "Running",
                    },
                    "b": {
                        "type": "Skipped",
                        "displayName": "profile",
                        "phase": "Omitted",
                    },
                },
            },
        )
        expect(_row(page, tenant, name).get_by_role("cell")).to_have_text(
            [name, "simba", "manual", "Running", "2026-10-04 09:00 UTC", "—"],
            timeout=POLL_TIMEOUT_MS,
        )
        expect(
            detail.get_by_role("table", name="Steps").get_by_role("row")
        ).to_have_text(
            ["Step Phase", "run-optimizer Running", "profile Skipped"],
            timeout=POLL_TIMEOUT_MS,
        )

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
                "finishedAt": "2026-10-04T09:10:00Z",
                "message": "Stopped with strategy 'Terminate'",
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
            f"Retried {name}; Argo reports "
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
