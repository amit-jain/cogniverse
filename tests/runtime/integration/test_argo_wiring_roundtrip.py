"""Round-trip integration test for scheduled-job workflow submission.

``POST /admin/tenant/{tenant}/jobs`` must submit a CronWorkflow when the
workflow engine is configured (``WORKFLOW_API_URL`` set), and persist without
submitting when it is not. Exercised through the real router + ConfigManager;
the submission-shape tests mock the HTTP boundary
(``_submit_cron_workflow`` / ``_delete_cron_workflow``), while the failed-commit
tests drive the real Argo API over a fixture-owned Kubernetes API server with
real Vespa traffic through a refusing proxy.
"""

import asyncio
import copy
import inspect
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import unquote

import httpx
import jsonschema
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.config_loader import WorkflowSettings, get_workflow_settings
from cogniverse_runtime.routers import tenant
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.k8s_api_server import (
    CRONWORKFLOW_CRD,
    _kubectl,
    start_k8s_api_server,
    stop_k8s_api_server,
)

_ARGO_SCHEMA_PATH = (
    Path(__file__).parent / "schemas" / "argo_cronworkflow_min.schema.json"
)


def _argo_schema() -> dict:
    return json.loads(_ARGO_SCHEMA_PATH.read_text())


def _configure_workflow(api_url=None):
    get_workflow_settings._instance = WorkflowSettings(
        api_url=api_url,
        namespace="cogniverse",
        job_template="cogniverse-job-runner",
        optimization_template="cogniverse-optimization-runner",
    )


@pytest.fixture(autouse=True)
def reset_workflow_settings():
    yield
    if hasattr(get_workflow_settings, "_instance"):
        del get_workflow_settings._instance


@pytest.fixture
def tenant_client(config_manager):
    """Mount the tenant router with a real ConfigManager wired to test Vespa."""
    tenant.set_config_manager(config_manager)
    app = FastAPI()
    app.include_router(tenant.router, prefix="/admin/tenant")
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
class TestWorkflowSubmissionRoundTrip:
    def test_configured_then_post_submits_workflow(self, tenant_client):
        _configure_workflow(api_url="http://argo-server:2746")
        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_submit:
            resp = tenant_client.post(
                "/admin/tenant/test_argo_tenant/jobs",
                json={
                    "name": "weekly_news_brief",
                    "schedule": "0 9 * * 1",
                    "query": "latest AI papers",
                },
            )
            assert resp.status_code == 200, resp.text
            mock_submit.assert_awaited_once()
            manifest = mock_submit.call_args[0][0]
            assert manifest["kind"] == "CronWorkflow"
            assert manifest["spec"]["schedule"] == "0 9 * * 1"
            assert (
                manifest["spec"]["workflowSpec"]["workflowTemplateRef"]["name"]
                == "cogniverse-job-runner"
            )

    def test_unconfigured_then_post_persists_without_submitting(self, tenant_client):
        _configure_workflow(api_url=None)
        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_submit:
            resp = tenant_client.post(
                "/admin/tenant/test_argo_no_url/jobs",
                json={"name": "test_job", "schedule": "0 9 * * *", "query": "test"},
            )
            assert resp.status_code == 200
            mock_submit.assert_not_awaited()

        resp = tenant_client.get("/admin/tenant/test_argo_no_url/jobs")
        jobs = resp.json().get("jobs", [])
        assert any(j.get("name") == "test_job" for j in jobs), (
            "Job should be persisted even when the workflow engine is unconfigured"
        )

    def test_delete_job_removes_cron_workflow_and_tombstones(self, tenant_client):
        _configure_workflow(api_url="http://argo-server:2746")
        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ):
            created = tenant_client.post(
                "/admin/tenant/test_argo_del/jobs",
                json={"name": "j", "schedule": "0 9 * * *", "query": "q"},
            )
        assert created.status_code == 200, created.text
        job_id = created.json()["job_id"]

        with patch(
            "cogniverse_runtime.routers.tenant._delete_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_delete:
            resp = tenant_client.delete(f"/admin/tenant/test_argo_del/jobs/{job_id}")
            assert resp.status_code == 200, resp.text
            mock_delete.assert_awaited_once_with(
                tenant._cron_workflow_name("test_argo_del", job_id),
                get_workflow_settings().namespace,
            )

        resp2 = tenant_client.delete(f"/admin/tenant/test_argo_del/jobs/{job_id}")
        assert resp2.status_code == 404

        listed = tenant_client.get("/admin/tenant/test_argo_del/jobs").json()["jobs"]
        assert all(j["job_id"] != job_id for j in listed)


@pytest.mark.integration
class TestManifestMatchesArgoSchema:
    """Every manifest the router hands to Argo must satisfy the vendored
    minimal CronWorkflow/Workflow schema, so a field rename or misspelled key
    fails here instead of being silently dropped by the Argo API server."""

    def test_built_cron_workflow_validates_against_argo_schema(self):
        _configure_workflow(api_url="http://argo-server:2746")
        manifest = tenant._build_cron_workflow(
            tenant_id="acme:prod",
            job_id="ab12cd34",
            schedule="0 9 * * 1",
            namespace="cogniverse",
        )
        jsonschema.validate(instance=manifest, schema=_argo_schema())
        assert manifest["metadata"]["name"] == "tenant-job-acme-prod-ab12cd34"
        assert manifest["metadata"]["labels"] == {
            "app": "cogniverse",
            "tenant": "acme-prod",
            "job-id": "ab12cd34",
        }
        assert manifest["spec"]["workflowSpec"]["workflowTemplateRef"] == {
            "name": "cogniverse-job-runner"
        }
        assert manifest["spec"]["workflowSpec"]["arguments"]["parameters"] == [
            {"name": "job-id", "value": "ab12cd34"},
            {"name": "tenant-id", "value": "acme:prod"},
        ]

    def test_route_submitted_cron_manifest_validates(self, tenant_client):
        _configure_workflow(api_url="http://argo-server:2746")
        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_submit:
            resp = tenant_client.post(
                "/admin/tenant/test_argo_schema/jobs",
                json={
                    "name": "nightly_digest",
                    "schedule": "30 6 * * *",
                    "query": "digest",
                },
            )
            assert resp.status_code == 200, resp.text
            job_id = resp.json()["job_id"]
            manifest = mock_submit.call_args[0][0]
        jsonschema.validate(instance=manifest, schema=_argo_schema())
        assert manifest["metadata"]["name"] == f"tenant-job-test-argo-schema-{job_id}"
        assert manifest["spec"]["schedule"] == "30 6 * * *"

    def test_built_optimization_workflow_validates(self):
        _configure_workflow(api_url="http://argo-server:2746")
        manifest = tenant._build_optimization_workflow_manifest(
            tenant_id="acme:prod", mode="simba", namespace="cogniverse"
        )
        jsonschema.validate(instance=manifest, schema=_argo_schema())
        assert manifest["metadata"]["generateName"] == "manual-optimize-simba-"
        assert manifest["spec"]["workflowTemplateRef"] == {
            "name": "cogniverse-optimization-runner"
        }
        assert manifest["spec"]["arguments"]["parameters"] == [
            {"name": "mode", "value": "simba"},
            {"name": "tenant-id", "value": "acme:prod"},
            {"name": "lookback-hours", "value": "48"},
        ]

    def test_misspelled_workflow_template_ref_fails_validation(self):
        """Pin the schema's strictness: a manifest carrying
        ``workflowTemplateReff`` (the field rename Argo would silently drop)
        must be rejected."""
        _configure_workflow(api_url="http://argo-server:2746")
        manifest = tenant._build_cron_workflow(
            tenant_id="acme:prod",
            job_id="ab12cd34",
            schedule="0 9 * * 1",
            namespace="cogniverse",
        )
        broken = copy.deepcopy(manifest)
        spec = broken["spec"]["workflowSpec"]
        spec["workflowTemplateReff"] = spec.pop("workflowTemplateRef")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=_argo_schema())

    def test_invalid_metadata_name_fails_validation(self):
        _configure_workflow(api_url="http://argo-server:2746")
        manifest = tenant._build_cron_workflow(
            tenant_id="acme:prod",
            job_id="ab12cd34",
            schedule="0 9 * * 1",
            namespace="cogniverse",
        )
        broken = copy.deepcopy(manifest)
        broken["metadata"]["name"] = "Tenant_Job:Bad"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=_argo_schema())


@pytest.fixture(scope="module")
def argo_cluster(tmp_path_factory):
    """Run the real Argo API against a fixture-owned Kubernetes datastore."""
    cluster = start_k8s_api_server(tmp_path_factory.mktemp("job-argo"))
    kubeconfig = Path(cluster["kubeconfig"])
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    name = f"cogniverse-test-argo-{os.getpid()}-{port}"
    try:
        for plural, kind, namespaced in (
            ("workflows", "Workflow", True),
            ("workflowtemplates", "WorkflowTemplate", True),
            ("clusterworkflowtemplates", "ClusterWorkflowTemplate", False),
        ):
            crd = copy.deepcopy(CRONWORKFLOW_CRD)
            crd["metadata"]["name"] = f"{plural}.argoproj.io"
            crd["spec"]["scope"] = "Namespaced" if namespaced else "Cluster"
            crd["spec"]["names"] = {
                "kind": kind,
                "listKind": f"{kind}List",
                "plural": plural,
                "singular": plural[:-1],
            }
            applied = _kubectl(
                kubeconfig, "apply", "-f", "-", input_text=json.dumps(crd)
            )
            assert applied.returncode == 0, applied.stderr
            ready = _kubectl(
                kubeconfig,
                "wait",
                "--for=condition=Established",
                f"crd/{plural}.argoproj.io",
                "--timeout=60s",
                timeout=70,
            )
            assert ready.returncode == 0, ready.stderr
        # argo-server reads the controller's ConfigMap on startup and exits
        # fatally without it; the defaults are what this fixture needs.
        applied = _kubectl(
            kubeconfig,
            "apply",
            "-f",
            "-",
            input_text=json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": "workflow-controller-configmap",
                        "namespace": "cogniverse",
                    },
                }
            ),
        )
        assert applied.returncode == 0, applied.stderr
        for template_name, entrypoint in (
            ("cogniverse-job-runner", "job"),
            ("cogniverse-optimization-runner", "run-optimizer"),
        ):
            template = {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "WorkflowTemplate",
                "metadata": {"name": template_name, "namespace": "cogniverse"},
                "spec": {
                    "entrypoint": entrypoint,
                    "templates": [
                        {
                            "name": entrypoint,
                            "container": {
                                "image": "alpine:3.20",
                                "command": ["true"],
                            },
                        }
                    ],
                },
            }
            applied = _kubectl(
                kubeconfig, "apply", "-f", "-", input_text=json.dumps(template)
            )
            assert applied.returncode == 0, applied.stderr
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--label",
                f"cogniverse-test-owner-pid={os.getpid()}",
                "--network",
                "host",
                "--user",
                "0:0",
                "-v",
                f"{kubeconfig}:/kubeconfig:ro",
                "quay.io/argoproj/argocli:v3.7.3",
                "server",
                "--kubeconfig",
                "/kubeconfig",
                "--namespace",
                "cogniverse",
                "--namespaced",
                "--auth-mode",
                "server",
                "--secure=false",
                "--port",
                str(port),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{url}/api/v1/info", timeout=2)
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(0.2)
        else:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True
            )
            pytest.fail(
                f"Argo server did not become ready: {logs.stdout}\n{logs.stderr}"
            )
        yield {"url": url, "kubeconfig": kubeconfig}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        stop_k8s_api_server(cluster["container"])


@pytest.fixture(scope="module")
def argo_server(argo_cluster):
    """The Argo API base URL of the fixture-owned cluster."""
    return argo_cluster["url"]


_CONFIG_SCHEMA = (
    inspect.signature(VespaConfigStore.__init__).parameters["schema_name"].default
)
_FEED_PREFIX = f"/document/v1/{_CONFIG_SCHEMA}/{_CONFIG_SCHEMA}/docid/"


def _job_feed(method: str, path: str) -> tuple[str, str] | None:
    """``(tenant_id, job_id)`` this request feeds a job row for, else None.

    Gating on the Document v1 path rather than on parsed body fields: pyvespa
    gzips a body over 1 KiB, and the write's first leg is a ``POST /search/``
    with no body at all, so a body-reading gate can never see the feed it is
    meant to refuse.
    """
    if method != "POST" or not path.startswith(_FEED_PREFIX):
        return None
    doc_id = unquote(path[len(_FEED_PREFIX) :].split("?")[0])
    if not doc_id.startswith(f"{_CONFIG_SCHEMA}::"):
        return None
    config_id = doc_id[len(_CONFIG_SCHEMA) + 2 :].rsplit("::", 1)[0]
    parts = config_id.split(":")
    if len(parts) != 5 or parts[3] != tenant._JOBS_SERVICE:
        return None
    return f"{parts[0]}:{parts[1]}", parts[4].removeprefix("job_")


@pytest.fixture
def job_commit_boundary(vespa_instance, argo_server, monkeypatch):
    """Forward real Vespa traffic, refusing one tenant's job feed on demand."""
    state: dict = {
        "failed_tenant": None,
        "hold": False,
        "entered": threading.Event(),
        "release": threading.Event(),
        "refused_jobs": [],
    }

    def refuse_one_tenants_job_feed(method, path, _body):
        target = _job_feed(method, path)
        if target is None or target[0] != state["failed_tenant"]:
            return None
        state["refused_jobs"].append(target[1])
        state["entered"].set()
        if state["hold"]:
            assert state["release"].wait(60) is True
        return 503, b'{"message":"injected config feed refusal"}'

    with InterceptFaultProxy(
        vespa_instance["base_url"], refuse_one_tenants_job_feed
    ) as proxy:
        cm = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://127.0.0.1",
                backend_port=int(proxy.url.rsplit(":", 1)[1]),
            )
        )
        monkeypatch.setattr(tenant, "_config_manager", cm)
        _configure_workflow(api_url=argo_server)
        app = FastAPI()
        app.include_router(tenant.router, prefix="/admin/tenant")
        try:
            yield app, cm, state, proxy
        finally:
            state["release"].set()


def _seen(proxy) -> list[tuple[str, str]]:
    """Every request the proxy forwarded, for a failure message."""
    return [(method, path) for method, path, _body in proxy.requests]


async def _actual_cron_names(argo_url: str, tenant_id: str) -> list[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{argo_url}/api/v1/cron-workflows/cogniverse",
            params={
                "listOptions.labelSelector": (
                    f"tenant={tenant._sanitize_label_value(tenant_id)}"
                )
            },
        )
    assert response.status_code == 200, response.text
    return sorted(row["metadata"]["name"] for row in response.json().get("items") or [])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_job_commit_removes_real_schedule_before_retry(
    job_commit_boundary, argo_server
):
    app, cm, state, proxy = job_commit_boundary
    tenant_id = f"jobfail{uuid.uuid4().hex[:8]}:production"
    body = {
        "name": "daily-report",
        "schedule": "0 9 * * *",
        "query": "Read launch notes",
    }
    state["failed_tenant"] = tenant_id
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://job-test",
    ) as client:
        failed = await client.post(f"/admin/tenant/{tenant_id}/jobs", json=body)
        assert failed.status_code == 500
        # The version allocation re-reads and re-feeds, so the same job id can
        # be refused more than once; exactly one job must have been attempted.
        assert len(set(state["refused_jobs"])) == 1, _seen(proxy)
        assert await _actual_cron_names(argo_server, tenant_id) == []
        assert (await client.get(f"/admin/tenant/{tenant_id}/jobs")).json() == {
            "jobs": []
        }
        state["failed_tenant"] = None
        created = await client.post(f"/admin/tenant/{tenant_id}/jobs", json=body)
        assert created.status_code == 200, created.text
        result = created.json()
        assert await _actual_cron_names(argo_server, tenant_id) == [
            tenant._cron_workflow_name(tenant_id, result["job_id"])
        ]
        assert cm.get_config_value(
            tenant_id=tenant_id,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
            config_key=f"job_{result['job_id']}",
        ) == {
            "job_id": result["job_id"],
            **body,
            "post_actions": [],
            "created_at": result["created_at"],
        }
        assert (await client.get(f"/admin/tenant/{tenant_id}/jobs")).json() == {
            "jobs": [{**result, "status": "active"}]
        }
        deleted = await client.delete(
            f"/admin/tenant/{tenant_id}/jobs/{result['job_id']}"
        )
        assert deleted.json() == {"status": "deleted", "job_id": result["job_id"]}
        assert await _actual_cron_names(argo_server, tenant_id) == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_job_compensation_does_not_remove_concurrent_tenant_schedule(
    job_commit_boundary, argo_server
):
    app, cm, state, proxy = job_commit_boundary
    failed_tenant = f"jobfail{uuid.uuid4().hex[:8]}:production"
    peer_tenant = f"jobpeer{uuid.uuid4().hex[:8]}:production"
    state["failed_tenant"] = failed_tenant
    state["hold"] = True
    body = {
        "name": "daily-report",
        "schedule": "0 9 * * *",
        "query": "Read launch notes",
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://job-test",
    ) as client:
        failing = asyncio.create_task(
            client.post(f"/admin/tenant/{failed_tenant}/jobs", json=body)
        )
        try:
            assert await asyncio.to_thread(state["entered"].wait, 60) is True, _seen(
                proxy
            )
            peer = await client.post(f"/admin/tenant/{peer_tenant}/jobs", json=body)
            assert peer.status_code == 200, peer.text
            result = peer.json()
            assert await _actual_cron_names(argo_server, failed_tenant) == [
                tenant._cron_workflow_name(failed_tenant, state["refused_jobs"][0])
            ]
        finally:
            state["release"].set()
        assert (await failing).status_code == 500
        assert await _actual_cron_names(argo_server, failed_tenant) == []
        assert await _actual_cron_names(argo_server, peer_tenant) == [
            tenant._cron_workflow_name(peer_tenant, result["job_id"])
        ]
        assert (await client.get(f"/admin/tenant/{failed_tenant}/jobs")).json() == {
            "jobs": []
        }
        assert (await client.get(f"/admin/tenant/{peer_tenant}/jobs")).json() == {
            "jobs": [{**result, "status": "active"}]
        }
        assert cm.get_config_value(
            tenant_id=peer_tenant,
            scope=ConfigScope.SYSTEM,
            service="tenant_jobs",
            config_key=f"job_{result['job_id']}",
        ) == {
            "job_id": result["job_id"],
            **body,
            "post_actions": [],
            "created_at": result["created_at"],
        }


_OPTIMIZE_TENANT = "optruns:alpha"
_OTHER_TENANT = "optruns:beta"
_CRON_LABEL = "workflows.argoproj.io/cron-workflow"


def _cron_spawned_workflow(name: str, tenant_id: str, template: str, status: dict):
    """A Workflow shaped like one Argo's CronWorkflow controller spawns.

    The controller copies the CronWorkflow's ``workflowSpec`` and stamps the
    ``workflows.argoproj.io/cron-workflow`` label; it does NOT copy the
    CronWorkflow's own labels, which is why the runtime cannot select these
    by ``cogniverse.ai/tenant``.
    """
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "name": name,
            "namespace": "cogniverse",
            "labels": {_CRON_LABEL: f"cogniverse-{name.rsplit('-', 1)[0]}"},
        },
        "spec": {
            "entrypoint": "pipeline",
            "arguments": {"parameters": [{"name": "tenant-id", "value": tenant_id}]},
            "templates": [
                {
                    "name": "pipeline",
                    "steps": [
                        [
                            {
                                "name": "run",
                                "templateRef": {
                                    "name": template,
                                    "template": "run-optimizer",
                                },
                            }
                        ]
                    ],
                }
            ],
        },
        "status": status,
    }


def _apply(kubeconfig, manifest) -> None:
    applied = _kubectl(kubeconfig, "apply", "-f", "-", input_text=json.dumps(manifest))
    assert applied.returncode == 0, applied.stderr


def _set_status(kubeconfig, name: str, status: dict) -> None:
    patched = _kubectl(
        kubeconfig,
        "patch",
        "workflow.argoproj.io",
        name,
        "-n",
        "cogniverse",
        "--type=merge",
        "-p",
        json.dumps({"status": status}),
    )
    assert patched.returncode == 0, patched.stderr


def _runs_when_settled(client, tenant_id: str, expected: int, **params) -> list:
    """Poll the route until Argo's cache reports ``expected`` runs with a phase.

    The fixture runs no workflow controller, so the phases under test are the
    ones the test patched in; argo-server serves them from an informer cache
    that syncs a moment after the patch lands.
    """
    deadline = time.monotonic() + 60
    last: list = []
    while time.monotonic() < deadline:
        response = client.get(f"/admin/tenant/{tenant_id}/optimize/runs", params=params)
        assert response.status_code == 200, response.text
        last = response.json()["runs"]
        if len(last) == expected and all(run["phase"] for run in last):
            return last
        time.sleep(0.5)
    return last


@pytest.fixture(scope="module")
def optimize_runs_seeded(argo_cluster):
    """Seed this module's optimization runs into the real Argo cluster once.

    Module-scoped: re-seeding per test would leave each earlier test's runs
    behind and the exact-list assertions would drift with execution order.
    """
    _configure_workflow(api_url=argo_cluster["url"])
    kubeconfig = argo_cluster["kubeconfig"]
    app = FastAPI()
    app.include_router(tenant.router, prefix="/admin/tenant")
    with TestClient(app) as client:
        simba = client.post(
            f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize", json={"mode": "simba"}
        )
        assert simba.status_code == 200, simba.text
        gateway = client.post(
            f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize",
            json={"mode": "gateway-thresholds"},
        )
        assert gateway.status_code == 200, gateway.text
        other = client.post(
            f"/admin/tenant/{_OTHER_TENANT}/optimize", json={"mode": "simba"}
        )
        assert other.status_code == 200, other.text

        simba_name = simba.json()["workflow_name"]
        gateway_name = gateway.json()["workflow_name"]
        _set_status(
            kubeconfig,
            simba_name,
            {
                "phase": "Succeeded",
                "startedAt": "2026-09-16T10:00:00Z",
                "finishedAt": "2026-09-16T10:05:00Z",
            },
        )
        _set_status(
            kubeconfig,
            gateway_name,
            {"phase": "Running", "startedAt": "2026-09-16T11:00:00Z"},
        )
        _set_status(
            kubeconfig,
            other.json()["workflow_name"],
            {
                "phase": "Succeeded",
                "startedAt": "2026-09-16T12:00:00Z",
                "finishedAt": "2026-09-16T12:01:00Z",
            },
        )
        # A scheduled optimization run for this tenant, and a scheduled
        # tenant-JOB run that must not be mistaken for one.
        _apply(
            kubeconfig,
            _cron_spawned_workflow(
                "agent-optimization-1758009600",
                _OPTIMIZE_TENANT,
                "cogniverse-optimization-runner",
                {
                    "phase": "Failed",
                    "startedAt": "2026-09-16T09:00:00Z",
                    "finishedAt": "2026-09-16T09:30:00Z",
                },
            ),
        )
        _apply(
            kubeconfig,
            _cron_spawned_workflow(
                "tenant-job-1758009600",
                _OPTIMIZE_TENANT,
                "cogniverse-job-runner",
                {
                    "phase": "Succeeded",
                    "startedAt": "2026-09-16T09:45:00Z",
                    "finishedAt": "2026-09-16T09:46:00Z",
                },
            ),
        )
        yield {"simba": simba_name, "gateway": gateway_name}


@pytest.fixture
def optimize_runs_env(argo_cluster, optimize_runs_seeded):
    """Tenant router mounted against the real Argo API holding the seeded runs.

    Returns the TestClient and the two manual run names.
    """
    _configure_workflow(api_url=argo_cluster["url"])
    app = FastAPI()
    app.include_router(tenant.router, prefix="/admin/tenant")
    with TestClient(app) as client:
        yield client, optimize_runs_seeded["simba"], optimize_runs_seeded["gateway"]


@pytest.mark.integration
class TestOptimizationRunListing:
    """``GET /{tenant}/optimize/runs`` lists a tenant's optimization runs."""

    def test_manual_and_scheduled_runs_listed_newest_first(self, optimize_runs_env):
        client, simba_name, gateway_name = optimize_runs_env

        runs = _runs_when_settled(client, _OPTIMIZE_TENANT, 3)

        assert runs == [
            {
                "workflow_name": gateway_name,
                "mode": "gateway-thresholds",
                "trigger": "manual",
                "phase": "Running",
                "started_at": "2026-09-16T11:00:00Z",
                "finished_at": None,
            },
            {
                "workflow_name": simba_name,
                "mode": "simba",
                "trigger": "manual",
                "phase": "Succeeded",
                "started_at": "2026-09-16T10:00:00Z",
                "finished_at": "2026-09-16T10:05:00Z",
            },
            {
                "workflow_name": "agent-optimization-1758009600",
                "mode": None,
                "trigger": "scheduled",
                "phase": "Failed",
                "started_at": "2026-09-16T09:00:00Z",
                "finished_at": "2026-09-16T09:30:00Z",
            },
        ]

    def test_page_size_caps_the_response(self, optimize_runs_env):
        client, _simba_name, gateway_name = optimize_runs_env

        _runs_when_settled(client, _OPTIMIZE_TENANT, 3)
        response = client.get(
            f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize/runs", params={"limit": 1}
        )
        assert response.status_code == 200, response.text
        assert [run["workflow_name"] for run in response.json()["runs"]] == [
            gateway_name
        ]

    def test_another_tenants_runs_are_not_listed(self, optimize_runs_env):
        client, _simba_name, _gateway_name = optimize_runs_env

        runs = _runs_when_settled(client, _OTHER_TENANT, 1)
        assert [(run["mode"], run["trigger"]) for run in runs] == [("simba", "manual")]

    def test_argo_outage_answers_503_not_an_empty_list(self, optimize_runs_env):
        client, _simba_name, _gateway_name = optimize_runs_env

        with socket.socket() as closed:
            closed.bind(("127.0.0.1", 0))
            dead_port = closed.getsockname()[1]
        _configure_workflow(api_url=f"http://127.0.0.1:{dead_port}")

        response = client.get(f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize/runs")
        assert response.status_code == 503, response.text
        body = response.json()
        assert list(body) == ["detail"]
        assert body["detail"].startswith("Argo API unreachable:")

    def test_unconfigured_argo_answers_503(self, optimize_runs_env):
        client, _simba_name, _gateway_name = optimize_runs_env
        _configure_workflow(api_url=None)

        response = client.get(f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize/runs")
        assert response.status_code == 503, response.text
        assert response.json() == {
            "detail": "Argo is not configured on this deployment."
        }


class _BadArgoHandler(BaseHTTPRequestHandler):
    """Answers every list with the scripted status and body."""

    status = 500
    body = b'{"message":"argo-server is restarting"}'
    content_type = "application/json"

    def log_message(self, *_args):
        return

    def do_GET(self):
        self.send_response(self.status)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


@pytest.fixture
def failing_argo():
    """A stand-in Argo endpoint whose responses the test scripts."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BadArgoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.mark.integration
class TestOptimizationRunListingFaults:
    """Argo answering badly must raise, never read as "no runs"."""

    def _client(self, api_url):
        _configure_workflow(api_url=api_url)
        app = FastAPI()
        app.include_router(tenant.router, prefix="/admin/tenant")
        return TestClient(app)

    def test_a_rejecting_argo_answers_503(self, failing_argo):
        _BadArgoHandler.status = 500
        _BadArgoHandler.body = b'{"message":"argo-server is restarting"}'
        _BadArgoHandler.content_type = "application/json"
        with self._client(failing_argo) as client:
            response = client.get(f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize/runs")
        assert response.status_code == 503, response.text
        assert response.json() == {
            "detail": (
                'Argo list failed (500): {"message":"argo-server is restarting"}'
            )
        }

    def test_an_html_body_on_200_answers_503(self, failing_argo):
        _BadArgoHandler.status = 200
        _BadArgoHandler.body = b"<html><body>502 Bad Gateway</body></html>"
        _BadArgoHandler.content_type = "text/html"
        try:
            with self._client(failing_argo) as client:
                response = client.get(f"/admin/tenant/{_OPTIMIZE_TENANT}/optimize/runs")
        finally:
            _BadArgoHandler.status = 500
            _BadArgoHandler.body = b'{"message":"argo-server is restarting"}'
            _BadArgoHandler.content_type = "application/json"
        assert response.status_code == 503, response.text
        assert response.json() == {
            "detail": (
                "Argo list returned a non-JSON body: "
                "<html><body>502 Bad Gateway</body></html>"
            )
        }
