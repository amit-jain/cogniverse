"""The web client's Optimization runs view, driven in Chromium.

The built client's Node server forwards to the runtime's tenant router on a
real uvicorn socket, which submits, lists, cancels and retries Workflows on a
real Argo API server. No workflow controller runs there, so a test sets each
phase a controller would report and watches the page pick it up.

Uploaded training examples go through the approval store on real Phoenix and
Redis, and are read back the way an optimization run reads them. The report
streams from the production dispatcher.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_agents.approval import ApprovalStorageImpl
from cogniverse_agents.approval.approval_storage import (
    validate_approved_dataset_record,
)
from cogniverse_core.approval.interfaces import approved_synthetic_dataset_name
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
from cogniverse_runtime.config_loader import WorkflowSettings, get_workflow_settings
from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data
from cogniverse_runtime.routers import agents, approvals, training_examples
from cogniverse_runtime.routers import tenant as tenant_router
from cogniverse_runtime.routers.optimization_report import REPORT_AGENT, REPORT_QUERY
from cogniverse_synthetic.approval.uploads import upload_templates
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER
from tests.utils.approval_review import review_config_manager, run_in_own_loop
from tests.utils.argo_api import argo_api_server, set_workflow_status
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.k8s_api_server import _kubectl
from tests.utils.memory_store import InMemoryConfigStore
from tests.utils.web_client import (
    free_port,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import serve_ops_runtime

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]

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
        expect(
            page.get_by_role("region", name="Run an optimization").locator("p.last-run")
        ).to_have_text(f"Last run: {name} (mode: simba)")
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


class TestRunLifecycle:
    def test_a_run_waiting_on_the_tenant_mutex_says_what_it_waits_for(
        self, page, web_url, argo
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        name = _start(page, "profile")
        set_workflow_status(
            argo["kubeconfig"],
            name,
            {
                "phase": "Pending",
                "synchronization": {
                    "mutex": {
                        "waiting": [
                            {
                                "mutex": "cogniverse/optimize-mutex",
                                "holder": "cogniverse/manual-optimize-simba-1",
                            }
                        ]
                    }
                },
            },
        )
        detail = page.get_by_role("region", name=f"Run {name}")
        expect(detail.locator("dt:text-is('Waiting') + dd")).to_have_text(
            "Waiting for another optimization to release the mutex: "
            "cogniverse/optimize-mutex",
            timeout=POLL_TIMEOUT_MS,
        )

    def test_the_last_run_line_selects_its_run(self, page, web_url, argo):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        first = _start(page, "simba")
        second = _start(page, "profile")
        _row(page, tenant, first).get_by_role("button", name=first).click()
        expect(page.get_by_role("region", name=f"Run {first}")).to_be_visible()
        page.get_by_role("region", name="Run an optimization").get_by_role(
            "button", name=second
        ).click()
        expect(page.get_by_role("region", name=f"Run {second}")).to_be_visible()
        expect(page.get_by_role("region", name=f"Run {first}")).to_have_count(0)

    def test_a_run_argo_deleted_reads_as_expired_not_as_an_error(
        self, page, web_url, argo
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        name = _start(page, "simba")
        detail = page.get_by_role("region", name=f"Run {name}")
        expect(detail.locator("dt:text-is('Phase') + dd")).to_have_text("Pending")
        deleted = _kubectl(
            argo["kubeconfig"],
            "delete",
            "workflow.argoproj.io",
            name,
            "-n",
            "cogniverse",
        )
        assert deleted.returncode == 0, deleted.stderr
        gone = (
            f"Run {name} no longer exists; Argo deleted it when its "
            "time-to-live expired."
        )
        # argo-server serves from an informer cache that sees the delete late.
        deadline = time.monotonic() + POLL_TIMEOUT_MS / 1000
        while True:
            detail.get_by_role("button", name="Cancel run").click()
            alert = detail.get_by_role("alert")
            expect(alert).to_have_count(1)
            if alert.inner_text() == gone or time.monotonic() > deadline:
                break
            page.wait_for_timeout(1000)
        expect(alert).to_have_text(gone)
        detail.get_by_role("button", name="Refresh").click()
        expect(detail.get_by_role("alert")).to_have_text(gone)
        expect(detail.locator("dl.facts")).to_have_count(0)

    def test_a_status_argo_cannot_answer_shows_an_outage_not_stale_facts(
        self, page, web_url, argo
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        name = _start(page, "simba")
        detail = page.get_by_role("region", name=f"Run {name}")
        expect(detail.locator("dt:text-is('Phase') + dd")).to_have_text("Pending")
        _configure_workflow(f"http://127.0.0.1:{free_port()}")
        detail.get_by_role("button", name="Refresh").click()
        expect(detail.get_by_role("alert")).to_have_text(
            "The Argo API did not answer; retry."
        )
        expect(detail.locator("dl.facts")).to_have_count(0)
        expect(detail.get_by_role("button", name="Cancel run")).to_have_count(0)


# Two query-enhancement examples an operator wrote.
KILN = [
    {
        "query": "kiln firing schedule",
        "enhanced_query": "stoneware kiln firing schedule cone six bisque",
        "expansion_terms": ["cone six", "bisque"],
        "synonyms": ["kiln program"],
        "context": "ceramics",
        "reasoning": "names the ware, the cone and the firing stage",
    },
    {
        "query": "glaze crazing",
        "enhanced_query": "glaze crazing cooling quartz inversion",
        "expansion_terms": ["quartz inversion"],
        "synonyms": ["glaze cracking"],
        "context": "ceramics",
        "reasoning": "crazing follows from cooling through quartz inversion",
    },
]
UPLOAD_TIMEOUT_MS = 180_000


@pytest.fixture()
def uploads(
    config_manager,
    phoenix_container,
    workflow_state_redis_url,
    telemetry_manager_with_phoenix,
):
    """The upload and approval routes on the approval store over real
    Phoenix and Redis."""
    review = review_config_manager(phoenix_container, workflow_state_redis_url)
    training_examples.set_config_manager(review)
    approvals.set_config_manager(review)
    yield {
        "phoenix": phoenix_container,
        "telemetry": telemetry_manager_with_phoenix,
        "review": review,
    }
    training_examples.set_config_manager(config_manager)
    approvals.set_config_manager(config_manager)


def _provider(uploads, tenant: str):
    """``tenant``'s telemetry provider on the healthy Phoenix endpoint."""
    return ApprovalStorageImpl.from_system_config(
        uploads["review"], uploads["telemetry"], tenant
    ).provider


def _trained_on(uploads, tenant: str) -> list[dict]:
    """The query-enhancement examples an optimization run of ``tenant`` reads
    from its approved training dataset."""
    return run_in_own_loop(
        _load_approved_synthetic_data(
            _provider(uploads, tenant), tenant, "query_enhancement"
        )
    )


def _trained_on_until(uploads, tenant: str, count: int, timeout=60.0) -> list:
    """``_trained_on`` once it holds ``count`` examples (Phoenix serves dataset
    rows after a short indexing delay)."""
    deadline = time.monotonic() + timeout
    while True:
        examples = _trained_on(uploads, tenant)
        if len(examples) == count or time.monotonic() > deadline:
            return examples
        time.sleep(2)


def _dataset_rows(uploads, tenant: str) -> list[dict]:
    try:
        frame = run_in_own_loop(
            _provider(uploads, tenant).datasets.get_dataset(
                name=approved_synthetic_dataset_name(tenant)
            )
        )
    except DatasetNotFoundError:
        return []
    return [
        validate_approved_dataset_record(
            row["input"],
            tenant_id=tenant,
            dataset_name=approved_synthetic_dataset_name(tenant),
            position=position,
        )
        for position, (_, row) in enumerate(frame.iterrows())
    ]


def _examples_panel(page: Page):
    return page.get_by_role("region", name="Upload training examples")


def _choose_files(page: Page, *paths: Path) -> None:
    _examples_panel(page).get_by_label("Examples files (JSON)").set_input_files(
        list(paths)
    )


def _file(page: Page, name: str):
    return _examples_panel(page).get_by_role("region", name=f"File {name}")


def _uploaded_batch(page: Page, count: int, name: str, tenant: str) -> str:
    notice = page.get_by_role("status")
    prefix = (
        f"Approved {count} query_enhancement examples from {name} into "
        f"{approved_synthetic_dataset_name(tenant)} as batch "
    )
    expect(notice).to_contain_text(prefix, timeout=UPLOAD_TIMEOUT_MS)
    return re.fullmatch(
        re.escape(prefix) + r"(upload_query_enhancement_[0-9a-f]{32})\.",
        notice.inner_text(),
    ).group(1)


def _write(path: Path, content) -> Path:
    path.write_text(json.dumps(content))
    return path


class TestTrainingExamples:
    def test_an_operator_uploads_examples_an_optimization_run_then_reads(
        self, page, web_url, runtime_url, uploads, tmp_path
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        panel = _examples_panel(page)
        expect(panel.get_by_label("Template optimizer").locator("option")).to_have_text(
            sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
        )
        panel.get_by_label("Template optimizer").select_option("entity_extraction")
        with page.expect_download() as downloaded:
            panel.get_by_role("button", name="Download template").click()
        assert downloaded.value.suggested_filename == "entity_extraction_examples.json"
        assert json.loads(Path(downloaded.value.path()).read_text()) == {
            "optimizer": "entity_extraction",
            "examples": [upload_templates()["entity_extraction"]["example"]],
        }

        legacy = _write(
            tmp_path / "routing_examples.json", {"good_routes": [], "bad_routes": []}
        )
        kiln = _write(
            tmp_path / "kiln.json", {"optimizer": "query_enhancement", "examples": KILN}
        )
        _choose_files(page, legacy, kiln)
        expect(_file(page, "routing_examples.json").get_by_role("alert")).to_have_text(
            'Unexpected keys: good_routes, bad_routes. "optimizer" must be one of '
            "entity_extraction, profile, query_enhancement, routing. "
            '"examples" must be a non-empty list.'
        )
        expect(
            _file(page, "routing_examples.json").get_by_role("button")
        ).to_have_count(0)
        expect(_file(page, "kiln.json").locator("p.muted")).to_have_text(
            "Valid query_enhancement examples file (2 examples)."
        )
        _file(page, "kiln.json").get_by_text("Preview: kiln.json").click()
        assert json.loads(
            _file(page, "kiln.json").get_by_label("Preview of kiln.json").inner_text()
        ) == {"optimizer": "query_enhancement", "examples": KILN}

        _file(page, "kiln.json").get_by_role("button", name="Upload kiln.json").click()
        expect(_file(page, "kiln.json").get_by_role("alert")).to_have_text(
            "Enter your name under Uploaded by first."
        )
        panel.get_by_label("Uploaded by").fill("operator@example.com")
        _file(page, "kiln.json").get_by_role("button", name="Upload kiln.json").click()
        batch = _uploaded_batch(page, 2, "kiln.json", tenant)
        expect(
            _file(page, "kiln.json").get_by_role("button", name="Uploaded")
        ).to_be_disabled()

        assert _trained_on_until(uploads, tenant, 2) == [
            {**KILN[0], "example_id": f"approved:{batch}_0"},
            {**KILN[1], "example_id": f"approved:{batch}_1"},
        ]
        rows = _dataset_rows(uploads, tenant)
        assert [
            (
                row["item_id"],
                row["status"],
                row["metadata.decision"]["reviewer"],
                row["metadata.decision"]["feedback"],
                row["context.source_file"],
            )
            for row in rows
        ] == [
            (
                f"{batch}_{index}",
                "approved",
                "operator@example.com",
                "Uploaded from kiln.json",
                "kiln.json",
            )
            for index in range(2)
        ]
        queue = httpx.get(f"{runtime_url}/admin/tenant/{tenant}/approvals", timeout=60)
        assert (queue.status_code, queue.json()) == (200, {"items": []})

    def test_values_the_schema_refuses_are_named_and_nothing_is_stored(
        self, page, web_url, uploads, tmp_path
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        same = _write(
            tmp_path / "same.json",
            {
                "optimizer": "query_enhancement",
                "examples": [KILN[0], dict(KILN[1], enhanced_query="glaze crazing")],
            },
        )
        _choose_files(page, same)
        _examples_panel(page).get_by_label("Uploaded by").fill("operator@example.com")
        _file(page, "same.json").get_by_role("button", name="Upload same.json").click()
        expect(_file(page, "same.json").get_by_role("alert")).to_have_text(
            "1 of 2 QueryEnhancementExampleSchema examples are invalid: "
            "examples[1] enhanced_query must differ from query"
        )
        assert _dataset_rows(uploads, tenant) == []

    def test_two_operators_uploading_at_once_each_land_their_own_batch(
        self, browser, web_url, uploads, tmp_path
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        kiln = _write(
            tmp_path / "kiln.json", {"optimizer": "query_enhancement", "examples": KILN}
        )
        contexts = [browser.new_context() for _ in range(2)]
        pages = [context.new_page() for context in contexts]
        try:
            for page, operator in zip(pages, ("first", "second")):
                _runs_view(page, web_url, tenant)
                _choose_files(page, kiln)
                _examples_panel(page).get_by_label("Uploaded by").fill(
                    f"{operator}@example.com"
                )
            for page in pages:
                _file(page, "kiln.json").get_by_role(
                    "button", name="Upload kiln.json"
                ).click(no_wait_after=True)
            batches = [_uploaded_batch(page, 2, "kiln.json", tenant) for page in pages]
        finally:
            for context in contexts:
                context.close()
        assert len(set(batches)) == 2
        assert sorted(
            example["example_id"] for example in _trained_on_until(uploads, tenant, 4)
        ) == sorted(
            f"approved:{batch}_{index}" for batch in batches for index in (0, 1)
        )

    def test_a_dataset_write_failing_part_way_leaves_the_rest_awaiting_review(
        self, page, web_url, runtime_url, uploads, workflow_state_redis_url, tmp_path
    ):
        """Phoenix refuses every dataset write after the first: the first
        example is trained on, the second waits in the review queue."""
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        writes = []

        def refuse_after_first_write(method, path, _body):
            if method == "POST" and path.startswith("/v1/datasets/upload"):
                writes.append(path)
                if len(writes) > 1:
                    return 503, {"detail": "injected dataset outage"}
            return None

        with InterceptFaultProxy(
            uploads["phoenix"]["http_endpoint"], refuse_after_first_write
        ) as proxy:
            training_examples.set_config_manager(
                review_config_manager(
                    uploads["phoenix"],
                    workflow_state_redis_url,
                    telemetry_url=proxy.url,
                )
            )
            kiln = _write(
                tmp_path / "kiln.json",
                {"optimizer": "query_enhancement", "examples": KILN},
            )
            _runs_view(page, web_url, tenant)
            _choose_files(page, kiln)
            _examples_panel(page).get_by_label("Uploaded by").fill(
                "operator@example.com"
            )
            _file(page, "kiln.json").get_by_role(
                "button", name="Upload kiln.json"
            ).click()
            alert = _file(page, "kiln.json").get_by_role("alert")
            expect(alert).to_contain_text(
                "1 of 2 uploaded examples were approved into the training "
                "dataset; the rest of batch upload_query_enhancement_",
                timeout=UPLOAD_TIMEOUT_MS,
            )
            batch = re.fullmatch(
                r"1 of 2 uploaded examples were approved into the training "
                r"dataset; the rest of batch (upload_query_enhancement_[0-9a-f]{32}) "
                r"await review in the approval queue\.",
                alert.inner_text(),
            ).group(1)
        assert _trained_on_until(uploads, tenant, 1) == [
            {**KILN[0], "example_id": f"approved:{batch}_0"}
        ]
        deadline = time.monotonic() + 60
        while True:
            queue = httpx.get(
                f"{runtime_url}/admin/tenant/{tenant}/approvals", timeout=60
            ).json()["items"]
            if [item["item_id"] for item in queue] == [f"{batch}_1"]:
                break
            assert time.monotonic() < deadline, queue
            time.sleep(2)
        assert (queue[0]["status"], queue[0]["data"]) == ("pending_review", KILN[1])


# The shipped document profiles one embedding service serves.
_SHIPPED_PROFILES = json.loads((REPO_ROOT / "configs" / "config.json").read_text())[
    "backend"
]["profiles"]
DOCUMENT_PROFILES = sorted(
    name
    for name, data in _SHIPPED_PROFILES.items()
    if str(data.get("type") or "").lower() == "document"
    and (data.get("inference_services") or {}).get("embedding") == "colbert_pylate"
)


@pytest.fixture()
def report_agent(vespa_instance, schema_loader):
    """A dispatcher over a config store of its own, with a registry that
    ``register(tenant)`` registers ``detailed_report_agent`` in, the way the
    runtime registers its configured agents, after configuring the shipped
    document profiles for ``tenant`` without deploying their schemas."""
    report_config = ConfigManager(store=InMemoryConfigStore())
    report_config.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            inference_service_urls={"colbert_pylate": "http://127.0.0.1:9"},
        )
    )
    registry = AgentRegistry(tenant_id="default", config_manager=report_config)
    agents.set_agent_registry(registry)
    agents.set_agent_dependencies(report_config, schema_loader)
    shipped = json.loads((REPO_ROOT / "configs" / "config.json").read_text())["agents"][
        REPORT_AGENT
    ]

    def register(tenant: str) -> bool:
        for name in DOCUMENT_PROFILES:
            report_config.add_backend_profile(
                BackendProfileConfig.from_dict(name, _SHIPPED_PROFILES[name]),
                tenant_id=tenant,
            )
        return registry.register_agent_from_data(
            {
                "name": REPORT_AGENT,
                "url": shipped["url"],
                "capabilities": shipped["capabilities"],
                "streams_answer_tokens": shipped["streams_answer_tokens"],
                "health_endpoint": "/health",
                "process_endpoint": f"/agents/{REPORT_AGENT}/process",
            }
        )

    yield register
    agents.set_agent_registry(
        AgentRegistry(tenant_id="default", config_manager=report_config)
    )
    agents.set_agent_dependencies(None, None)


class TestReport:
    def test_a_report_streams_from_the_report_agent_and_downloads_as_json(
        self, page, web_url, runtime_url, report_agent
    ):
        tenant = f"webopt{uuid.uuid4().hex[:8]}:main"
        _runs_view(page, web_url, tenant)
        panel = page.get_by_role("region", name="Optimization report")
        expect(panel.locator("p.muted")).to_have_text(
            f"{REPORT_AGENT} is not registered with the runtime, so no report can "
            "be generated."
        )
        expect(panel.get_by_role("button", name="Generate report")).to_be_disabled()
        refused = httpx.post(
            f"{runtime_url}/admin/tenant/{tenant}/optimize/report", timeout=60
        )
        assert (refused.status_code, refused.json()) == (
            404,
            {
                "detail": f"Agent '{REPORT_AGENT}' is not registered, so no report "
                "can be generated."
            },
        )

        assert report_agent(tenant) is True
        page.reload()
        _runs_view(page, web_url, tenant)
        panel = page.get_by_role("region", name="Optimization report")
        expect(panel.locator("p.muted")).to_have_text(
            f"{REPORT_AGENT} is registered with the runtime and writes the report."
        )
        dispatcher = agents.get_dispatcher()
        grounding = run_in_own_loop(
            dispatcher._resolve_answer_search_results(
                REPORT_QUERY, tenant, {"tenant_id": tenant}, top_k=20
            )
        )
        assert grounding.nothing_to_search is True
        expected = json.loads(
            json.dumps(dispatcher._nothing_to_search_report(tenant, grounding))
        )

        panel.get_by_role("button", name="Generate report").click()
        expect(
            panel.get_by_label("Report", exact=True).locator("p.report-text")
        ).to_have_text(grounding.unanswerable_text(tenant), timeout=POLL_TIMEOUT_MS)
        expect(panel.get_by_role("alert")).to_have_count(0)
        with page.expect_download() as downloaded:
            panel.get_by_role("button", name="Download report").click()
        assert re.fullmatch(
            r"optimization_report_\d{8}_\d{6}\.json",
            downloaded.value.suggested_filename,
        )
        assert json.loads(Path(downloaded.value.path()).read_text()) == expected
