"""E2E tests for tenant extensibility on k3d."""

import httpx
import pytest

from tests.e2e.conftest import RUNTIME, TENANT_ID, skip_if_no_runtime


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantInstructions:
    def test_set_and_get_instructions(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.put(
                f"/admin/tenant/{TENANT_ID}/instructions",
                json={"text": "E2E test: prefer summaries over reports"},
            )
            assert resp.status_code == 200
            assert "E2E test" in resp.json()["text"]

            resp = client.get(f"/admin/tenant/{TENANT_ID}/instructions")
            assert resp.status_code == 200
            assert resp.json()["text"] == "E2E test: prefer summaries over reports"

    def test_delete_instructions(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            client.put(
                f"/admin/tenant/{TENANT_ID}/instructions",
                json={"text": "temp instructions"},
            )
            resp = client.delete(f"/admin/tenant/{TENANT_ID}/instructions")
            assert resp.status_code == 200
            assert resp.json()["status"] == "cleared"


@pytest.mark.e2e
@skip_if_no_runtime
class TestTenantJobs:
    def test_create_list_delete_job(self):
        with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
            resp = client.post(
                f"/admin/tenant/{TENANT_ID}/jobs",
                json={
                    "name": "e2e_test_job",
                    "schedule": "0 12 * * *",
                    "query": "e2e test search",
                    "post_actions": ["save to wiki"],
                },
            )
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]

            resp = client.get(f"/admin/tenant/{TENANT_ID}/jobs")
            assert resp.status_code == 200
            jobs = resp.json()["jobs"]
            assert any(j["job_id"] == job_id for j in jobs)

            resp = client.delete(f"/admin/tenant/{TENANT_ID}/jobs/{job_id}")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"
