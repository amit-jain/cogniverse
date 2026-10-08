"""The web client's shell, driven in Chromium: the active tenant every view
and agent acts for, the tenant gate against the runtime's real tenant registry
on Vespa, the agents' status, and two browser sessions on different tenants
chatting at once.

The runtime serves its tenant admin, harness-key admin (over the Vespa config
store), health, agents and AG-UI routers on a real uvicorn socket. The web
server mints each tenant's harness key through that admin, so the runtime
resolves every run's tenant from the key the run carries.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_PERSIST_FAILURE_CAPACITY,
    CONVERSATION_SAVE_LEASE_S,
    AgentDispatcher,
)
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.harness_keys import HarnessKeyStore
from cogniverse_runtime.routers import admin, ag_ui, agents, health, openai_compat
from cogniverse_runtime.session_state import ContinuationStore, ConversationLedger
from cogniverse_runtime.shared_state import connect_shared_state_redis
from cogniverse_runtime.task_events import TaskEventStore
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.memory_store import InMemoryConfigStore
from tests.utils.web_client import (
    browse_as,
    recording_telemetry_sink,
    serve_app,
    serve_web,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

# A tenant's base schemas deploy on registration; that is the slow step.
DEPLOY_TIMEOUT_S = 240
ORG = f"shell{uuid.uuid4().hex[:6]}"
ALPHA = f"{ORG}:{ORG}"
BETA = f"{ORG}:beta"

# Runs of the shell agent wait inside the agent until ``expected`` of them
# have entered, so they are in flight at once.
_barrier = {"expected": 1, "entered": 0}
_barrier_lock = threading.Lock()


class ShellDeps(AgentDeps):
    pass


class ShellInput(AgentInput):
    query: str = ""
    tenant_id: str = ""
    conversation_history: list = []


class ShellOutput(AgentOutput):
    summary: str = ""
    results: list = []


class ShellSearchAgent(AgentBase[ShellInput, ShellOutput, ShellDeps]):
    """Answers once every run of the scenario is inside, with one hit titled
    after the run's tenant."""

    async def _process_impl(self, input: ShellInput) -> ShellOutput:
        with _barrier_lock:
            _barrier["entered"] += 1
        deadline = time.monotonic() + 60
        while _barrier["entered"] < _barrier["expected"]:
            if time.monotonic() > deadline:
                raise RuntimeError("the other session's run never arrived")
            await asyncio.sleep(0.01)
        return ShellOutput(
            summary=f"[{input.tenant_id}] {input.query}",
            results=[
                {
                    "id": f"{input.tenant_id}-hit",
                    "document_id": f"id:video:video::{input.tenant_id}-hit",
                    "score": 1.0,
                    "metadata": {"video_title": f"clip of {input.tenant_id}"},
                }
            ],
        )


# Not "search_agent", which the dispatcher builds as the real SearchAgent.
AGENT = "shell_agent"


@pytest.fixture(scope="module")
def runtime(config_manager, schema_loader, vespa_instance, workflow_state_redis_url):
    """The runtime's shell-facing routers on a real socket; the tenant
    registry and harness keys live in the shared Vespa."""
    registry_config = ConfigManager(store=InMemoryConfigStore())
    registry = AgentRegistry(tenant_id=ALPHA, config_manager=registry_config)
    registry.register_agent(
        AgentEndpoint(name=AGENT, url="http://localhost:8000", capabilities=["web"])
    )
    ConfigLoader.AGENT_CLASSES[AGENT] = f"{__name__}:ShellSearchAgent"
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=registry_config, schema_loader=None
    )
    dispatcher._conversation_store_factory = lambda tenant_id: None

    @asynccontextmanager
    async def lifespan(_app):
        redis = await connect_shared_state_redis(workflow_state_redis_url)
        prefix = f"test:shell:{uuid.uuid4().hex}"
        # A tenant delete releases the tenant on every worker over the
        # cluster events channel and cancels its tasks.
        events = ClusterEvents(
            workflow_state_redis_url,
            f"web-shell-test-{uuid.uuid4().hex[:8]}",
            {"tenant_deleted": tm.release_deleted_tenant},
            channel=f"cogniverse:test-events:{uuid.uuid4().hex[:8]}",
        )
        await events.start()
        task_events = TaskEventStore(redis)
        task_events.start()
        tm.set_cluster_events(events)
        tm.set_task_event_store(task_events)
        openai_compat.set_continuation_store(
            ContinuationStore(redis, key_prefix=prefix)
        )
        dispatcher.set_conversation_ledger(
            ConversationLedger(
                redis,
                save_lease_s=CONVERSATION_SAVE_LEASE_S,
                failure_capacity=CONVERSATION_PERSIST_FAILURE_CAPACITY,
                key_prefix=f"{prefix}:conversation",
            )
        )
        try:
            yield
        finally:
            tm.set_cluster_events(None)
            tm.set_task_event_store(None)
            await task_events.close()
            await events.close()
            dispatcher.set_conversation_ledger(None)
            openai_compat.set_continuation_store(None)
            await redis.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.backend_base_url = vespa_instance["base_url"]
    app.include_router(health.router)
    app.include_router(agents.router, prefix="/agents")
    app.include_router(ag_ui.router, prefix="/ag-ui")
    app.include_router(admin.router, prefix="/admin")
    app.include_router(tm.router, prefix="/admin")
    previous = (tm._config_manager, tm._schema_loader)
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    admin.set_config_manager(config_manager)
    agents.set_agent_registry(registry)
    openai_compat.set_dispatcher_provider(lambda: dispatcher)
    keys = HarnessKeyStore(config_manager.store)
    openai_compat.set_key_resolver(keys.resolve)
    health._reset_probe_state()
    try:
        with serve_app(app) as url:
            yield SimpleNamespace(
                url=url, app=app, keys=keys, vespa_url=vespa_instance["base_url"]
            )
    finally:
        openai_compat.set_dispatcher_provider(None)
        openai_compat.set_key_resolver(None)
        admin.reset_dependencies()
        tm.set_config_manager(previous[0])
        tm.set_schema_loader(previous[1])
        ConfigLoader.AGENT_CLASSES.pop(AGENT, None)
        health._reset_probe_state()


@pytest.fixture(scope="module")
def tenants(runtime):
    """Two registered tenants of one organization."""
    for tenant in (ALPHA, BETA):
        created = httpx.post(
            f"{runtime.url}/admin/tenants",
            json={"tenant_id": tenant, "created_by": "web-shell-test"},
            timeout=DEPLOY_TIMEOUT_S,
        )
        assert created.status_code == 200, created.text
    yield ALPHA, BETA
    for tenant in (ALPHA, BETA):
        deleted = httpx.delete(
            f"{runtime.url}/admin/tenants/{tenant}", timeout=DEPLOY_TIMEOUT_S
        )
        assert deleted.status_code == 200, deleted.text


@pytest.fixture()
def web_url(built_client, runtime, tenants):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime.url, telemetry_url=sink_url, built=True
        ) as url:
            yield url
        assert received == []


# A hostname that is not localhost, as the cluster ingress serves the client
# over plain http: the page is not a secure context there.
INGRESS_HOST = "cogniverse.test"
THREAD = re.compile(
    rf"#/agents/{AGENT}/[0-9a-f]{{8}}-[0-9a-f]{{4}}-4[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}$"
)


@pytest.fixture(scope="module")
def playwright_driver():
    with sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="module")
def browser(playwright_driver):
    browser = playwright_driver.chromium.launch()
    yield browser
    browser.close()


@pytest.fixture(scope="module")
def ingress_browser(playwright_driver):
    """Chromium resolving ``INGRESS_HOST`` to this host."""
    browser = playwright_driver.chromium.launch(
        args=[f"--host-resolver-rules=MAP {INGRESS_HOST} 127.0.0.1"]
    )
    yield browser
    browser.close()


@pytest.fixture()
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()


def _use_tenant(page: Page, tenant: str) -> None:
    form = page.get_by_role("form", name="Active tenant")
    form.get_by_label("Active tenant").fill(tenant)
    form.get_by_role("button", name="Use").click()


def _current(page: Page):
    return page.get_by_role("region", name="Active tenant").locator("p.current-tenant")


def _assistant_messages(page: Page):
    return page.get_by_test_id("copilot-assistant-message")


def _page_errors(page: Page) -> list:
    """Every error the page throws and does not catch from now on."""
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


def _send(page: Page, text: str) -> None:
    box = page.get_by_placeholder("Ask Shell…")
    box.fill(text)
    box.press("Enter")


class TestActiveTenant:
    def test_the_active_tenant_is_shared_by_every_view_and_survives_a_reload(
        self, page, web_url, tenants
    ):
        alpha, beta = tenants
        page.goto(f"{web_url}/#/ops/analytics")
        chooser = page.get_by_role("form", name="Choose tenant")
        expect(page.get_by_role("region", name="Tenant", exact=True)).to_contain_text(
            "Every view reads one tenant. Choose a registered tenant; register a "
            "new one in the Tenants view (POST /admin/tenants) first."
        )
        expect(_current(page)).to_have_text("No tenant chosen.")
        expect(page.locator("nav .sidebar-footer")).to_have_text(
            "Cogniverse operations and agents"
        )

        # The simple form names the organization's own tenant; the runtime's
        # canonical form becomes the active tenant.
        _use_tenant(page, ORG)
        expect(_current(page)).to_have_text(f"Current tenant: {alpha}")
        expect(
            page.get_by_role("region", name=f"Traces of {alpha}", exact=True)
        ).to_be_visible()
        expect(chooser.get_by_label("Tenant ID")).to_have_value(alpha)

        for view, heading in (("approvals", "Approvals"), ("rlm-ab", "RLM A/B")):
            page.goto(f"{web_url}/#/ops/{view}")
            expect(page.get_by_role("heading", name=heading, level=1)).to_be_visible()
            expect(
                page.get_by_role("form", name="Choose tenant").get_by_label("Tenant ID")
            ).to_have_value(alpha)

        page.reload()
        expect(_current(page)).to_have_text(f"Current tenant: {alpha}")
        page.goto(f"{web_url}/#/ops/analytics")
        expect(
            page.get_by_role("region", name=f"Traces of {alpha}", exact=True)
        ).to_be_visible()

        # A choice in a view's own form switches every view, and the view
        # starts over for the new tenant.
        chooser.get_by_label("Tenant ID").fill(beta)
        chooser.get_by_role("button", name="Show traces").click()
        expect(_current(page)).to_have_text(f"Current tenant: {beta}")
        expect(
            page.get_by_role("region", name=f"Traces of {beta}", exact=True)
        ).to_be_visible()
        expect(
            page.get_by_role("region", name=f"Traces of {alpha}", exact=True)
        ).to_have_count(0)

    def test_an_unknown_tenant_is_refused_and_the_active_one_kept(
        self, page, web_url, tenants
    ):
        alpha, _ = tenants
        page.goto(f"{web_url}/#/ops/analytics")
        _use_tenant(page, alpha)
        expect(_current(page)).to_have_text(f"Current tenant: {alpha}")
        typo = f"{ORG}:typo"
        _use_tenant(page, typo)
        notice = page.get_by_role("region", name="Active tenant").get_by_role("alert")
        expect(notice).to_have_text(
            f"Tenant {typo} is not registered. Register it in the Tenants view "
            "(or with POST /admin/tenants) first, or choose a registered tenant."
        )
        expect(_current(page)).to_have_text(f"Current tenant: {alpha}")
        expect(
            page.get_by_role("region", name=f"Traces of {typo}", exact=True)
        ).to_have_count(0)

    def test_a_registry_that_cannot_answer_is_not_read_as_an_unknown_tenant(
        self, page, built_client, runtime, tenants
    ):
        """The registry's 503 is passed on as a warning, and the tenant is
        used; an unknown tenant behind the same server is still refused."""
        _, beta = tenants
        with InterceptFaultProxy(runtime.url) as proxy:
            proxy.intercept = lambda method, path, body: (
                (503, {"detail": "Tenant registry temporarily unavailable"})
                if method == "GET" and path == f"/admin/tenants/{quote(beta, safe='')}"
                else None
            )
            with recording_telemetry_sink() as (sink_url, _):
                with serve_web(
                    built_client, proxy.url, telemetry_url=sink_url, built=True
                ) as web_url:
                    page.goto(f"{web_url}/#/ops/analytics")
                    _use_tenant(page, beta)
                    expect(_current(page)).to_have_text(f"Current tenant: {beta}")
                    expect(
                        page.get_by_role("region", name="Active tenant").locator(
                            "p.alert.warning"
                        )
                    ).to_have_text(
                        f"The runtime could not confirm tenant {beta} is registered "
                        "(Tenant registry temporarily unavailable); its views read "
                        "as empty if it is not."
                    )
                    expect(
                        page.get_by_role("region", name=f"Traces of {beta}", exact=True)
                    ).to_be_visible()


class TestAgents:
    def test_each_agent_shows_whether_it_can_serve(self, page, web_url, runtime):
        page.goto(f"{web_url}/#/ops/analytics")
        status = page.get_by_label("Shell is online")
        expect(status).to_have_text("online")
        expect(status).to_have_attribute("title", "Registry health: unknown")

        runtime.app.state.backend_base_url = "http://127.0.0.1:9"
        health._reset_probe_state()
        try:
            page.reload()
            offline = page.get_by_label("Shell is offline")
            expect(offline).to_have_text("offline")
            expect(offline).to_have_attribute(
                "title", "The runtime is unhealthy: backend unreachable (ConnectError)."
            )
        finally:
            runtime.app.state.backend_base_url = runtime.vespa_url
            health._reset_probe_state()

    def test_a_chat_waits_for_a_tenant(self, page, web_url):
        page.goto(f"{web_url}/#/agents/{AGENT}")
        expect(page.locator("main .notice")).to_have_text(
            "Choose the active tenant in the sidebar before talking to an agent. "
            "Agents run for that tenant only."
        )
        expect(page.get_by_placeholder("Ask Shell…")).to_have_count(0)

    def test_two_sessions_on_different_tenants_see_only_their_own_results(
        self, browser, web_url, runtime, tenants
    ):
        """Both sessions' runs are inside the agent at once; each session
        shows its own tenant's reply and hit, and a session that switches
        tenant shows none of the other tenant's results."""
        alpha, beta = tenants
        contexts = [browser.new_context() for _ in range(2)]
        pages = [context.new_page() for context in contexts]
        try:
            for page, tenant in zip(pages, (alpha, beta), strict=True):
                page.goto(f"{web_url}/#/agents/{AGENT}")
                _use_tenant(page, tenant)
                expect(_current(page)).to_have_text(f"Current tenant: {tenant}")
                expect(page.get_by_placeholder("Ask Shell…")).to_be_visible()
            _barrier.update(expected=2, entered=0)
            for page, tenant in zip(pages, (alpha, beta), strict=True):
                _send(page, f"question from {tenant}")
            for page, tenant in zip(pages, (alpha, beta), strict=True):
                expect(_assistant_messages(page)).to_have_text(
                    [f"[{tenant}] question from {tenant}"], timeout=60_000
                )
                expect(page.locator(".result-title")).to_have_text(
                    [f"clip of {tenant}"]
                )
                expect(page.get_by_label("Conversation")).to_have_text(
                    f"{tenant} · 2 messages"
                )
            assert _barrier["entered"] == 2

            first = pages[0]
            _use_tenant(first, beta)
            expect(_current(first)).to_have_text(f"Current tenant: {beta}")
            expect(first.get_by_label("Conversation")).to_have_text(
                f"{beta} · 0 messages"
            )
            expect(first.locator(".result-card", has_text=alpha)).to_have_count(0)
            expect(_assistant_messages(first)).to_have_count(0)
        finally:
            for context in contexts:
                context.close()
        name = re.compile(r"^cogniverse-web ")
        for tenant in (alpha, beta):
            assert [
                key["tenant_id"]
                for key in runtime.keys.list(tenant)["keys"]
                if name.match(key["name"]) and not key["revoked"]
            ] == [tenant]


class TestShell:
    def test_each_operations_view_says_what_it_is_for(self, page, web_url):
        header = page.locator("header.workspace-header")
        for view, heading, description in (
            (
                "ingestion",
                "Ingestion",
                "Interactive testing and configuration of ingestion pipelines with "
                "different processing profiles.",
            ),
            (
                "tenants",
                "Tenants",
                "Create and delete organizations and their tenants, and set each "
                "tenant's router tier.",
            ),
        ):
            page.goto(f"{web_url}/#/ops/{view}")
            expect(header.get_by_role("heading", level=1)).to_have_text(heading)
            expect(header.locator("p.view-description")).to_have_text(description)

    def test_a_malformed_tenant_is_refused_and_the_active_one_kept(
        self, page, web_url, tenants
    ):
        alpha, _ = tenants
        page.goto(f"{web_url}/#/ops/tenants")
        _use_tenant(page, alpha)
        expect(_current(page)).to_have_text(f"Current tenant: {alpha}")
        _use_tenant(page, "a:b:c")
        notice = page.get_by_role("region", name="Active tenant").get_by_role("alert")
        expect(notice).to_have_text(
            "Tenant a:b:c cannot be used: Tenant ID 'a:b:c' is malformed: use "
            "'<org>:<tenant>' or '<tenant>', with no empty part."
        )
        expect(_current(page)).to_have_text(f"Current tenant: {alpha}")


class TestIngress:
    """The client served over plain http on a hostname other than localhost,
    as the cluster ingress serves it: not a secure context."""

    def test_choosing_a_tenant_opens_the_agent_and_a_run_answers(
        self, ingress_browser, web_url, tenants
    ):
        alpha, _ = tenants
        context = ingress_browser.new_context()
        page = context.new_page()
        errors = _page_errors(page)
        try:
            page.goto(f"{web_url.replace('127.0.0.1', INGRESS_HOST)}/")
            assert page.evaluate(
                "[window.isSecureContext, typeof crypto.randomUUID]"
            ) == [False, "undefined"]
            _use_tenant(page, alpha)
            expect(_current(page)).to_have_text(f"Current tenant: {alpha}")
            expect(page.get_by_placeholder("Ask Shell…")).to_be_visible()
            expect(page).to_have_url(THREAD)
            _barrier.update(expected=1, entered=0)
            _send(page, "over the ingress")
            expect(_assistant_messages(page)).to_have_text(
                [f"[{alpha}] over the ingress"], timeout=60_000
            )
            first = page.url
            page.get_by_role("button", name="New conversation").click()
            expect(page).not_to_have_url(first)
            expect(page).to_have_url(THREAD)
            expect(page.get_by_label("Conversation")).to_have_text(
                f"{alpha} · 0 messages"
            )
        finally:
            context.close()
        assert errors == []

    def test_an_agent_address_without_a_thread_opens_a_new_one(
        self, ingress_browser, web_url, tenants
    ):
        alpha, _ = tenants
        context = ingress_browser.new_context()
        browse_as(context, alpha)
        page = context.new_page()
        errors = _page_errors(page)
        try:
            page.goto(f"{web_url.replace('127.0.0.1', INGRESS_HOST)}/#/agents/{AGENT}")
            expect(page.get_by_placeholder("Ask Shell…")).to_be_visible()
            expect(page).to_have_url(THREAD)
        finally:
            context.close()
        assert errors == []
