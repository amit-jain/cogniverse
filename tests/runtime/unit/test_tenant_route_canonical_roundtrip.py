"""Bare-tenant round-trips through the real tenant routes.

The instructions and jobs write/read paths canonicalize the tenant id to
``org:tenant`` before keying the config store. A route that read with the raw
path param would look up an empty namespace for a bare tenant id and 404/empty
on every value it just wrote. This drives the REAL routes over ASGITransport
against a real in-memory ConfigStore with a BARE tenant id — the fast-lane
guard the mock-store tests cannot provide (a mock store returns canned rows for
any key and so hides a write/read key mismatch).
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.config_loader import WorkflowSettings, get_workflow_settings
from cogniverse_runtime.routers import tenant as tenant_router
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

# Deliberately NOT the canonical org:tenant form.
BARE_TENANT = "acme"


@pytest.fixture
def app_cm():
    store = InMemoryConfigStore()
    store.initialize()
    cm = ConfigManager(store=store)
    saved_cm = tenant_router._config_manager
    tenant_router.set_config_manager(cm)
    # No Argo configured, so create_job just persists to the config store.
    get_workflow_settings._instance = WorkflowSettings(api_url=None)

    app = FastAPI()
    app.include_router(tenant_router.router)
    try:
        yield app, cm
    finally:
        tenant_router._config_manager = saved_cm
        if hasattr(get_workflow_settings, "_instance"):
            del get_workflow_settings._instance


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    )


@pytest.mark.asyncio
async def test_instructions_round_trip_with_bare_tenant(app_cm):
    app, _cm = app_cm
    async with _client(app) as client:
        put = await client.put(
            f"/{BARE_TENANT}/instructions",
            json={"text": "always cite the source video"},
        )
        assert put.status_code == 200

        got = await client.get(f"/{BARE_TENANT}/instructions")
        assert got.status_code == 200
        assert got.json()["text"] == "always cite the source video"


@pytest.mark.asyncio
async def test_jobs_round_trip_with_bare_tenant(app_cm):
    app, _cm = app_cm
    async with _client(app) as client:
        created = await client.post(
            f"/{BARE_TENANT}/jobs",
            json={
                "name": "nightly summary",
                "schedule": "0 2 * * *",
                "query": "summarize new videos",
                "post_actions": [],
            },
        )
        assert created.status_code == 200
        job_id = created.json()["job_id"]

        listed = await client.get(f"/{BARE_TENANT}/jobs")
        assert listed.status_code == 200
        job_ids = [j["job_id"] for j in listed.json()["jobs"]]
        assert job_id in job_ids
