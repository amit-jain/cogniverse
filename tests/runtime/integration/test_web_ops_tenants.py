"""The web client's Tenants view, driven in Chromium against real Vespa.

The client is built the way a deployment builds it and served by its own Node
server, which forwards to the runtime's tenant admin router on a real uvicorn
socket over the real tenant registry and config store. Every action the view
offers is taken through the page, and its outcome is read back from the
runtime, not from the page alone.
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.tenant_tiers import read_tenant_tier
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
)
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_runtime.routers import agents
from tests.utils.web_client import (
    build_web_client,
    free_port,
    install_web_client,
    recording_telemetry_sink,
    serve_app,
    serve_web,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "web-ops-harness-key"
# A tenant's base schemas deploy on creation; that is the slow step.
DEPLOY_TIMEOUT_MS = 240_000


@pytest.fixture(scope="module")
def built_client(tmp_path_factory):
    return build_web_client(install_web_client(tmp_path_factory.mktemp("web_ops")))


@pytest.fixture(scope="module")
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    previous = (tm._config_manager, tm._schema_loader)
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    agents.set_agent_registry(
        AgentRegistry(tenant_id="default", config_manager=config_manager)
    )

    @asynccontextmanager
    async def cluster_events(_app):
        # The one worker on its own cluster-events channel, started on the
        # server's loop as the runtime starts its own, so tenant deletes
        # reach it.
        events = ClusterEvents(
            workflow_state_redis_url,
            "web-ops-test-worker",
            {"tenant_deleted": tm.release_deleted_tenant},
            channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        )
        await events.start()
        tm.set_cluster_events(events)
        try:
            yield
        finally:
            tm.set_cluster_events(None)
            await events.close()

    app = FastAPI(lifespan=cluster_events)
    app.include_router(tm.router, prefix="/admin")
    app.include_router(agents.router, prefix="/agents")
    with serve_app(app) as url:
        yield url
    tm.set_config_manager(previous[0])
    tm.set_schema_loader(previous[1])
    BackendRegistry.get_instance().clear_instances()


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


def _tenants_view(page: Page, web_url: str) -> None:
    page.goto(f"{web_url}/#/ops/tenants")
    expect(page.get_by_role("heading", name="Tenants", level=1)).to_be_visible()


def _create_org(page: Page, org_id: str, name: str) -> None:
    form = page.get_by_role("form", name="Create organization")
    form.get_by_label("Organization ID").fill(org_id)
    form.get_by_label("Name").fill(name)
    form.get_by_role("button", name="Create organization").click()
    expect(form.get_by_role("status")).to_have_text(f"Created organization {org_id}.")


def _create_tenant(page: Page, org_id: str, name: str) -> None:
    form = page.get_by_role("form", name="Create tenant")
    form.get_by_label("Organization").fill(org_id)
    form.get_by_label("Tenant name").fill(name)
    form.get_by_role("button", name="Create tenant").click()
    expect(form.get_by_role("status")).to_contain_text(
        f"Created {org_id}:{name} with schemas ", timeout=DEPLOY_TIMEOUT_MS
    )


def _org_row(page: Page, org_id: str):
    return (
        page.get_by_role("region", name="Organizations")
        .get_by_role("row")
        .filter(has=page.get_by_role("button", name=org_id, exact=True))
    )


def _tenant_row(page: Page, org_id: str, tenant_id: str):
    return (
        page.get_by_role("region", name=f"Tenants of {org_id}")
        .get_by_role("row")
        .filter(has=page.get_by_role("cell", name=tenant_id, exact=True))
    )


class TestTenantLifecycle:
    def test_an_operator_creates_tiers_and_deletes_a_tenant(
        self, page, web_url, runtime_url, config_manager
    ):
        org_id = f"webops{uuid.uuid4().hex[:8]}"
        tenant_id = f"{org_id}:production"
        _tenants_view(page, web_url)

        _create_org(page, org_id, "Web Ops Org")
        row = _org_row(page, org_id)
        expect(row.get_by_role("cell")).to_have_text(
            [
                org_id,
                "Web Ops Org",
                "active",
                "0",
                re.compile(r".+ by admin$"),
                "Delete",
            ]
        )
        stored = httpx.get(f"{runtime_url}/admin/organizations/{org_id}").json()
        assert (stored["org_name"], stored["created_by"], stored["status"]) == (
            "Web Ops Org",
            "admin",
            "active",
        )

        row.get_by_role("button", name=org_id).click()
        tenants = page.get_by_role("region", name=f"Tenants of {org_id}")
        expect(tenants.get_by_text(f"No tenants in {org_id}.")).to_be_visible()

        _create_tenant(page, org_id, "production")
        created = httpx.get(f"{runtime_url}/admin/tenants/{tenant_id}").json()
        status = page.get_by_role("form", name="Create tenant").get_by_role("status")
        expect(status).to_have_text(
            f"Created {tenant_id} with schemas "
            f"{', '.join(created['schemas_deployed'])}."
        )
        tenant_row = _tenant_row(page, org_id, tenant_id)
        expect(tenant_row.get_by_role("cell").nth(2)).to_have_text(
            ", ".join(created["schemas_deployed"])
        )
        expect(_org_row(page, org_id).get_by_role("cell").nth(3)).to_have_text("1")

        tier = tenant_row.get_by_label(f"Router tier for {tenant_id}")
        expect(tier).to_have_value(DEFAULT_ROUTER_TIER)
        expect(tier.locator("option")).to_have_text(sorted(ROUTER_TIERS))
        tier.select_option("pro")
        expect(tier).to_be_enabled()
        assert read_tenant_tier(config_manager, tenant_id) == "pro"
        page.reload()
        _org_row(page, org_id).get_by_role("button", name=org_id).click()
        expect(
            _tenant_row(page, org_id, tenant_id).get_by_label(
                f"Router tier for {tenant_id}"
            )
        ).to_have_value("pro")

        tenant_row = _tenant_row(page, org_id, tenant_id)
        tenant_row.get_by_role("button", name="Delete").click()
        confirm = tenant_row.get_by_role("button", name="Delete tenant")
        expect(confirm).to_be_disabled()
        tenant_row.get_by_label(f"Type {tenant_id} to delete this tenant").fill(
            f"{org_id}:producti"
        )
        expect(confirm).to_be_disabled()
        tenant_row.get_by_label(f"Type {tenant_id} to delete this tenant").fill(
            tenant_id
        )
        confirm.click()
        # Deleting an organization's last tenant removes the organization too.
        expect(page.get_by_role("status").first).to_have_text(
            f"Deleted tenant {tenant_id} and organization {org_id}, which had "
            "no tenants left.",
            timeout=DEPLOY_TIMEOUT_MS,
        )
        expect(_org_row(page, org_id)).to_have_count(0)
        expect(page.get_by_role("region", name=f"Tenants of {org_id}")).to_have_count(0)
        assert httpx.get(f"{runtime_url}/admin/tenants/{tenant_id}").status_code == 404
        assert (
            httpx.get(f"{runtime_url}/admin/organizations/{org_id}").status_code == 404
        )

    def test_a_refused_create_shows_the_runtime_reason(
        self, page, web_url, runtime_url
    ):
        org_id = f"webops{uuid.uuid4().hex[:8]}"
        _tenants_view(page, web_url)
        _create_org(page, org_id, "Twice")

        form = page.get_by_role("form", name="Create organization")
        form.get_by_label("Organization ID").fill(org_id)
        form.get_by_label("Name").fill("Twice")
        form.get_by_role("button", name="Create organization").click()
        expect(form.get_by_role("alert")).to_have_text(
            f"Organization {org_id} already exists"
        )

        tenant_form = page.get_by_role("form", name="Create tenant")
        tenant_form.get_by_label("Organization").fill(org_id)
        tenant_form.get_by_label("Tenant name").fill("has-hyphen")
        tenant_form.get_by_role("button", name="Create tenant").click()
        expect(tenant_form.get_by_role("alert")).to_have_text(
            "Invalid tenant_name 'has-hyphen': only alphanumeric and underscore allowed"
        )
        assert (
            httpx.get(f"{runtime_url}/admin/tenants/{org_id}:has-hyphen").status_code
            == 404
        )

        org_row = _org_row(page, org_id)
        org_row.get_by_role("button", name="Delete").click()
        org_row.get_by_label(f"Type {org_id} to delete this organization").fill(org_id)
        org_row.get_by_role("button", name="Delete organization").click()
        expect(page.get_by_role("status").first).to_have_text(
            f"Deleted organization {org_id} and its 0 tenant(s)."
        )
        expect(_org_row(page, org_id)).to_have_count(0)
        assert (
            httpx.get(f"{runtime_url}/admin/organizations/{org_id}").status_code == 404
        )


class TestConcurrency:
    def test_two_operators_tiering_different_tenants_each_land(
        self, browser, web_url, runtime_url, config_manager
    ):
        """Two pages set tiers on two tenants at once; each write lands on its
        own tenant, and each page shows its own tenant's tier."""
        org_id = f"webops{uuid.uuid4().hex[:8]}"
        tenant_ids = [f"{org_id}:alpha", f"{org_id}:beta"]
        for tenant_id in tenant_ids:
            created = httpx.post(
                f"{runtime_url}/admin/tenants",
                json={"tenant_id": tenant_id, "created_by": "web-ops-test"},
                timeout=DEPLOY_TIMEOUT_MS / 1000,
            )
            assert created.status_code == 200, created.text

        contexts = [browser.new_context() for _ in tenant_ids]
        pages = [context.new_page() for context in contexts]
        try:
            for page in pages:
                _tenants_view(page, web_url)
                _org_row(page, org_id).get_by_role("button", name=org_id).click()
            pickers = [
                _tenant_row(page, org_id, tenant_id).get_by_label(
                    f"Router tier for {tenant_id}"
                )
                for page, tenant_id in zip(pages, tenant_ids)
            ]
            for picker in pickers:
                expect(picker).to_have_value(DEFAULT_ROUTER_TIER)
            # Both selects fire before either page waits on its write.
            pickers[0].select_option("pro", no_wait_after=True)
            pickers[1].select_option("free", no_wait_after=True)
            expect(pickers[0]).to_have_value("pro")
            expect(pickers[1]).to_have_value("free")
            expect(pickers[0]).to_be_enabled()
            expect(pickers[1]).to_be_enabled()
        finally:
            for context in contexts:
                context.close()
        with ThreadPoolExecutor(2) as pool:
            tiers = list(
                pool.map(lambda t: read_tenant_tier(config_manager, t), tenant_ids)
            )
        assert tiers == ["pro", "free"]
        for tenant_id in tenant_ids:
            deleted = httpx.delete(
                f"{runtime_url}/admin/tenants/{tenant_id}",
                timeout=DEPLOY_TIMEOUT_MS / 1000,
            )
            assert deleted.status_code == 200, deleted.text


class TestFaultContract:
    def test_a_down_runtime_reads_as_an_error_not_an_empty_list(
        self, page, built_client
    ):
        dead_runtime = f"http://127.0.0.1:{free_port()}"
        with recording_telemetry_sink() as (sink_url, _):
            with serve_web(
                built_client, dead_runtime, KEY, telemetry_url=sink_url, built=True
            ) as url:
                _tenants_view(page, url)
                organizations = page.get_by_role("region", name="Organizations")
                expect(organizations.get_by_role("alert")).to_have_text(
                    f"The Cogniverse runtime at {dead_runtime} did not answer "
                    "(TypeError)."
                )
                expect(organizations.get_by_role("table")).to_have_count(0)
                expect(
                    organizations.get_by_text("No organizations yet.")
                ).to_have_count(0)
