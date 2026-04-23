"""Round-trip integration test for Argo CronWorkflow wiring.

Verifies that the runtime startup helper ``_wire_argo_from_environment``
actually populates the tenant router's globals from env vars, and that
``POST /admin/tenant/{tenant}/jobs`` then results in a real Argo
CronWorkflow submission.

Audit fix #3 — before this fix, ``set_argo_config()`` was defined but
never called from main.py's lifespan, so ``_argo_api_url`` stayed
``None`` and POST /jobs silently dropped the CronWorkflow submission.
The pre-fix tests bypassed the bug by directly mutating
``tenant._argo_api_url`` from inside the test instead of going through
the env var → lifespan → set_argo_config wiring.

This test exercises the real chain.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.main import _wire_argo_from_environment
from cogniverse_runtime.routers import tenant


@pytest.fixture(autouse=True)
def reset_argo_state():
    """Snapshot and restore tenant._argo_api_url around each test."""
    original_url = tenant._argo_api_url
    original_ns = tenant._argo_namespace
    yield
    tenant._argo_api_url = original_url
    tenant._argo_namespace = original_ns


@pytest.fixture
def tenant_client(config_manager):
    """Mount the tenant router with a real ConfigManager wired to test Vespa."""
    tenant.set_config_manager(config_manager)
    app = FastAPI()
    app.include_router(tenant.router, prefix="/admin/tenant")
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
class TestArgoWiringRoundTrip:
    def test_helper_populates_globals_from_env(self, monkeypatch):
        """The lifespan helper must read BACKEND_PORT-style env vars and
        call set_argo_config so the router globals end up populated.
        Pre-fix this never happened — globals stayed None forever."""
        monkeypatch.setenv("ARGO_API_URL", "http://argo-server:2746")
        monkeypatch.setenv("ARGO_NAMESPACE", "test_ns")

        tenant._argo_api_url = None
        tenant._argo_namespace = "stale"

        _wire_argo_from_environment()

        assert tenant._argo_api_url == "http://argo-server:2746"
        assert tenant._argo_namespace == "test_ns"

    def test_helper_handles_missing_env_var_gracefully(self, monkeypatch):
        """When ARGO_API_URL is unset the helper must explicitly set
        ``_argo_api_url`` to None so create_job degrades to persist-only
        mode rather than crashing."""
        monkeypatch.delenv("ARGO_API_URL", raising=False)
        monkeypatch.delenv("ARGO_NAMESPACE", raising=False)

        tenant._argo_api_url = "stale"

        _wire_argo_from_environment()

        assert tenant._argo_api_url is None
        assert tenant._argo_namespace == "cogniverse"

    def test_full_roundtrip_env_var_then_post_submits_workflow(
        self, monkeypatch, tenant_client
    ):
        """End-to-end: set ARGO_API_URL, run the lifespan helper, POST a
        job through the real router, assert the CronWorkflow submission
        was actually attempted with the right manifest.

        This is the test that would have caught the audit bug — the
        previous test directly mutated ``tenant._argo_api_url`` and
        bypassed the helper entirely."""
        monkeypatch.setenv("ARGO_API_URL", "http://argo-server:2746")

        _wire_argo_from_environment()
        assert tenant._argo_api_url == "http://argo-server:2746"

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

    def test_full_roundtrip_no_env_var_skips_workflow(self, monkeypatch, tenant_client):
        """Symmetric round trip: with no ARGO_API_URL, POST /jobs must
        still persist the job to ConfigStore but NOT call submit. The
        bug had this path silently dropping ALL jobs because the global
        was always None."""
        monkeypatch.delenv("ARGO_API_URL", raising=False)

        _wire_argo_from_environment()
        assert tenant._argo_api_url is None

        with patch(
            "cogniverse_runtime.routers.tenant._submit_cron_workflow",
            new_callable=AsyncMock,
        ) as mock_submit:
            resp = tenant_client.post(
                "/admin/tenant/test_argo_no_url/jobs",
                json={
                    "name": "test_job",
                    "schedule": "0 9 * * *",
                    "query": "test",
                },
            )
            assert resp.status_code == 200
            mock_submit.assert_not_awaited()

        # And the job WAS persisted — verify by listing jobs.
        resp = tenant_client.get("/admin/tenant/test_argo_no_url/jobs")
        jobs = resp.json().get("jobs", [])
        assert any(j.get("name") == "test_job" for j in jobs), (
            "Job should be persisted to ConfigStore even when Argo is unconfigured"
        )
