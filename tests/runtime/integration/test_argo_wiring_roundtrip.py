"""Round-trip integration test for scheduled-job workflow submission.

``POST /admin/tenant/{tenant}/jobs`` must submit a CronWorkflow when the
workflow engine is configured (``WORKFLOW_API_URL`` set), and persist without
submitting when it is not. Exercised through the real router + ConfigManager;
the HTTP boundary (``_submit_cron_workflow`` / ``_delete_cron_workflow``) is
mocked.
"""

import asyncio
import copy
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
def argo_server(tmp_path_factory):
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
        template = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "WorkflowTemplate",
            "metadata": {"name": "cogniverse-job-runner", "namespace": "cogniverse"},
            "spec": {
                "entrypoint": "job",
                "templates": [
                    {
                        "name": "job",
                        "container": {"image": "alpine:3.20", "command": ["true"]},
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
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        stop_k8s_api_server(cluster["container"])


@pytest.fixture
def job_commit_boundary(vespa_instance, argo_server, monkeypatch):
    """Forward real Vespa traffic, refusing one tenant's config feed on demand."""
    state = {
        "failed_tenant": None,
        "hold": False,
        "entered": threading.Event(),
        "release": threading.Event(),
        "refused_jobs": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def forward(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            fields = json.loads(body).get("fields", {}) if body else {}
            if (
                self.command == "POST"
                and fields.get("service") == "tenant_jobs"
                and fields.get("tenant_id") == state["failed_tenant"]
            ):
                state["refused_jobs"].append(fields["config_key"].removeprefix("job_"))
                state["entered"].set()
                if state["hold"]:
                    assert state["release"].wait(60) is True
                response_body = b'{"message":"injected config feed refusal"}'
                self.send_response(503)
            else:
                response = requests.request(
                    self.command,
                    vespa_instance["base_url"] + self.path,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                response_body = response.content
                self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        do_GET = forward
        do_POST = forward
        do_DELETE = forward

        def log_message(self, *args):
            pass

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=proxy.server_port
        )
    )
    monkeypatch.setattr(tenant, "_config_manager", cm)
    _configure_workflow(api_url=argo_server)
    app = FastAPI()
    app.include_router(tenant.router, prefix="/admin/tenant")
    try:
        yield app, cm, state
    finally:
        state["release"].set()
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)


async def _actual_cron_names(argo_url: str, tenant_id: str) -> list[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{argo_url}/api/v1/cron-workflows/cogniverse",
            params={
                "listOptions.labelSelector": f"tenant={tenant._sanitize_label_value(tenant_id)}"
            },
        )
    assert response.status_code == 200, response.text
    return sorted(row["metadata"]["name"] for row in response.json().get("items") or [])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_job_commit_removes_real_schedule_before_retry(
    job_commit_boundary, argo_server
):
    app, cm, state = job_commit_boundary
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
        assert len(set(state["refused_jobs"])) == 1
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
    app, cm, state = job_commit_boundary
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
            assert await asyncio.to_thread(state["entered"].wait, 60) is True
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
