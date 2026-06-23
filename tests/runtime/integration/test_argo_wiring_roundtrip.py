"""Round-trip integration test for scheduled-job workflow submission.

``POST /admin/tenant/{tenant}/jobs`` must submit a CronWorkflow when the
workflow engine is configured (``WORKFLOW_API_URL`` set), and persist without
submitting when it is not. Exercised through the real router + ConfigManager;
the HTTP boundary (``_submit_cron_workflow`` / ``_delete_cron_workflow``) is
mocked.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.config_loader import WorkflowSettings, get_workflow_settings
from cogniverse_runtime.routers import tenant


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
