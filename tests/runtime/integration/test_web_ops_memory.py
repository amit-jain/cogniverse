"""The web client's Memory view, driven in Chromium against Mem0 on real Vespa.

The built client's Node server forwards to the runtime's tenant memory routes
on a real uvicorn socket. Each tenant's Mem0 manager stores in real Vespa
through a fault proxy; its embedder is ``serve_token_embedder``, which speaks
DenseOn's ``/v1/embeddings`` contract with a vector that depends only on the
words of the text, so search order is known in advance. Every action is taken
through the page and its outcome read back from the store.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from cogniverse_core.memory.manager import Mem0MemoryManager, affirm_memory_profile
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.vespa_test_helpers import deploy_tenant_schema
from tests.utils.web_client import (
    build_web_client,
    install_web_client,
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import (
    serve_ops_runtime,
    serve_token_embedder,
    token_embedding,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "web-ops-harness-key"
TENANTS = ("webmem:alpha", "webmem:beta")
SYSTEM_NOTE = (
    "A system namespace: the runtime manages these memories, so they are "
    "read-only here."
)


@pytest.fixture(scope="module")
def memory_store(vespa_instance, config_manager):
    """A Mem0 manager per tenant on real Vespa, reached through a fault proxy."""
    affirm_memory_profile(config_manager)
    with (
        InterceptFaultProxy(vespa_instance["base_url"]) as proxy,
        serve_token_embedder() as embedder_url,
    ):
        endpoints = dict(vespa_instance, http_port=proxy.port)
        managers = {}
        for tenant_id in TENANTS:
            Mem0MemoryManager._instances.pop(tenant_id, None)
            deploy_tenant_schema(
                endpoints,
                tenant_id=tenant_id,
                base_schema_name="agent_memories",
                config_manager=config_manager,
            )
            manager = Mem0MemoryManager(tenant_id)
            manager.initialize(
                backend_host="http://127.0.0.1",
                backend_port=proxy.port,
                backend_config_port=vespa_instance["config_port"],
                base_schema_name="agent_memories",
                llm_model="memory-view-test-unused",
                embedding_model="lightonai/DenseOn",
                llm_base_url="http://127.0.0.1:9",
                embedder_base_url=embedder_url,
                auto_create_schema=False,
                config_manager=config_manager,
                schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            )
            managers[tenant_id] = manager
        yield managers, proxy
        for tenant_id in TENANTS:
            Mem0MemoryManager._instances.pop(tenant_id, None)


@pytest.fixture()
def store(memory_store):
    managers, proxy = memory_store
    yield managers, proxy
    proxy.intercept = None
    for manager in managers.values():
        rows = manager.memory.get_all(user_id=manager.tenant_id, limit=None)
        for row in rows["results"]:
            manager.memory.delete(row["id"])


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


def _seed(manager, rows):
    """Seed ``(id, namespace, text, category, created_at, archived)`` rows."""
    manager.memory.vector_store.insert(
        [token_embedding(text) for _, _, text, _, _, _ in rows],
        [
            {
                "data": text,
                "user_id": manager.tenant_id,
                "agent_id": namespace,
                "created_at": created_at,
                "archived": archived,
                **({"category": category} if category else {}),
            }
            for _, namespace, text, category, created_at, archived in rows
        ],
        [mid for mid, *_ in rows],
    )


def _rows(manager, namespace):
    """The namespace's live rows by ID."""
    return {
        row["id"]: row
        for row in manager.get_all_memories(manager.tenant_id, namespace, limit=None)
    }


def _show(page: Page, web_url: str, tenant: str, namespace: str | None = None):
    page.goto(f"{web_url}/#/ops/memory")
    expect(page.get_by_role("heading", name="Memory", level=1)).to_be_visible()
    chooser = page.get_by_role("form", name="Choose tenant")
    chooser.get_by_label("Tenant ID").fill(tenant)
    chooser.get_by_role("button", name="Show memories").click()
    if namespace:
        _namespace(page, namespace)


def _namespace(page: Page, namespace: str):
    form = page.get_by_role("form", name="Choose namespace")
    form.get_by_label("Namespace").fill(namespace)
    form.get_by_role("button", name="Show").click()


def _panel(page: Page, tenant: str, namespace: str):
    return page.get_by_role("region", name=f"Memories of {namespace} in {tenant}")


def _table_rows(panel):
    return panel.get_by_role("table", name="Memories").locator("tbody tr")


def _cells(manager, namespace, ids):
    """The table cells for ``ids`` in a writable namespace, row by row, from
    the stored rows."""
    rows = _rows(manager, namespace)
    return [
        cell
        for mid in ids
        for cell in (
            rows[mid]["memory"],
            (rows[mid].get("metadata") or {}).get("category") or "—",
            str(rows[mid]["created_at"]),
            mid,
            "Delete",
        )
    ]


class TestMemoryView:
    def test_an_operator_searches_adds_deletes_and_clears_memories(
        self, page, web_url, store
    ):
        managers, _ = store
        tenant = TENANTS[0]
        manager = managers[tenant]
        _seed(
            manager,
            [
                (
                    "m-dark",
                    "_user_memories",
                    "prefers dark mode in every editor",
                    "ui",
                    1700000300,
                    False,
                ),
                (
                    "m-zone",
                    "_user_memories",
                    "timezone is utc plus five",
                    "locale",
                    1700000200,
                    False,
                ),
                ("m-old", "_user_memories", "an archived note", None, 1700000100, True),
                (
                    "m-strategy",
                    "_strategy_store",
                    "route video questions to colpali",
                    None,
                    1700000000,
                    False,
                ),
                (
                    "m-agent-1",
                    "search_agent",
                    "user liked the lecture clips",
                    None,
                    1700000000,
                    False,
                ),
                (
                    "m-agent-2",
                    "search_agent",
                    "user skips intros",
                    None,
                    1700000001,
                    False,
                ),
            ],
        )

        _show(page, web_url, tenant)
        panel = _panel(page, tenant, "_user_memories")
        expect(panel.get_by_text("2 live, 1 archived.")).to_be_visible()
        expect(_table_rows(panel).locator("td")).to_have_text(
            _cells(manager, "_user_memories", ["m-dark", "m-zone"])
        )

        search = panel.get_by_role("form", name="Search memories")
        search.get_by_label("Query").fill("utc timezone")
        search.get_by_role("button", name="Search").click()
        expect(_table_rows(panel).locator("td:nth-child(4)")).to_have_text(
            ["m-zone", "m-dark"]
        )
        search.get_by_role("button", name="Show all").click()
        expect(_table_rows(panel).locator("td:nth-child(4)")).to_have_text(
            ["m-dark", "m-zone"]
        )

        add = page.get_by_role("form", name="Add memory")
        add.get_by_label("Memory").fill("always cite the source video")
        add.get_by_label("Category").fill("style")
        add.get_by_label("Metadata (JSON)").fill("{not json")
        add.get_by_role("button", name="Save memory").click()
        expect(add.get_by_role("alert")).to_have_text("Metadata is not valid JSON.")
        assert set(_rows(manager, "_user_memories")) == {"m-dark", "m-zone"}

        add.get_by_label("Metadata (JSON)").fill('{"source": "web"}')
        add.get_by_role("button", name="Save memory").click()
        notice = page.get_by_role("status")
        expect(notice).to_contain_text("Saved memory ")
        saved_id = (
            notice.inner_text()
            .removeprefix("Saved memory ")
            .removesuffix(" to _user_memories.")
        )
        saved = _rows(manager, "_user_memories")[saved_id]
        assert (
            saved["memory"],
            {key: saved["metadata"][key] for key in ("category", "source")},
            saved["agent_id"],
        ) == (
            "always cite the source video",
            {"category": "style", "source": "web"},
            "_user_memories",
        )
        panel = _panel(page, tenant, "_user_memories")
        expect(panel.get_by_text("3 live, 1 archived.")).to_be_visible()
        expect(_table_rows(panel).locator("td:nth-child(4)")).to_have_text(
            [saved_id, "m-dark", "m-zone"]
        )

        panel.get_by_role("button", name="Delete memory m-zone").click()
        panel.get_by_role("button", name="Confirm delete of m-zone").click()
        expect(page.get_by_role("status")).to_have_text("Deleted memory m-zone.")
        assert set(_rows(manager, "_user_memories")) == {saved_id, "m-dark"}

        _namespace(page, "_strategy_store")
        system = _panel(page, tenant, "_strategy_store")
        expect(system.get_by_text(f"1 live, 0 archived. {SYSTEM_NOTE}")).to_be_visible()
        expect(_table_rows(system).locator("td:nth-child(4)")).to_have_text(
            ["m-strategy"]
        )
        expect(
            system.get_by_role("button", name="Delete memory m-strategy")
        ).to_have_count(0)
        expect(system.get_by_role("group", name="Clear namespace")).to_have_count(0)
        expect(page.get_by_role("form", name="Add memory")).to_have_count(0)

        _namespace(page, "search_agent")
        agent = _panel(page, tenant, "search_agent")
        expect(_table_rows(agent).locator("td:nth-child(4)")).to_have_text(
            ["m-agent-2", "m-agent-1"]
        )
        clear = agent.get_by_role("group", name="Clear namespace")
        clear.get_by_role("button", name="Clear every memory of search_agent").click()
        clear.get_by_label("Type search_agent to clear its memories").fill(
            "search_agen"
        )
        expect(clear.get_by_role("button", name="Clear search_agent")).to_be_disabled()
        clear.get_by_label("Type search_agent to clear its memories").fill(
            "search_agent"
        )
        clear.get_by_role("button", name="Clear search_agent").click()
        expect(page.get_by_role("status")).to_have_text(
            "Cleared every memory of search_agent."
        )
        agent = _panel(page, tenant, "search_agent")
        expect(agent.get_by_text("0 live, 0 archived.")).to_be_visible()
        expect(agent.get_by_text("No memories in search_agent.")).to_be_visible()
        assert _rows(manager, "search_agent") == {}
        assert set(_rows(manager, "_user_memories")) == {saved_id, "m-dark"}
        assert set(_rows(manager, "_strategy_store")) == {"m-strategy"}


class TestConcurrency:
    def test_two_operators_adding_at_once_each_land_in_their_own_tenant(
        self, browser, web_url, store
    ):
        managers, proxy = store
        texts = {tenant: f"note from the {tenant} operator" for tenant in TENANTS}
        barrier = threading.Barrier(2, timeout=30)
        arrived = []
        lock = threading.Lock()

        def intercept(method, path, body):
            # Hold each tenant's first write until both have arrived, so the
            # two saves are in flight together.
            if method != "POST" or not path.startswith("/document/v1/"):
                return None
            tenant = next(t for t in TENANTS if t.replace(":", "_") in path)
            with lock:
                first = tenant not in arrived
                if first:
                    arrived.append(tenant)
            if first:
                barrier.wait()
            return None

        contexts = [browser.new_context() for _ in TENANTS]
        pages = [context.new_page() for context in contexts]
        try:
            for page, tenant in zip(pages, TENANTS):
                _show(page, web_url, tenant, "concurrent_agent")
                add = page.get_by_role("form", name="Add memory")
                add.get_by_label("Memory").fill(texts[tenant])
            proxy.intercept = intercept
            for page in pages:
                page.get_by_role("form", name="Add memory").get_by_role(
                    "button", name="Save memory"
                ).click(no_wait_after=True)
            for page in pages:
                expect(page.get_by_role("status")).to_contain_text(
                    "to concurrent_agent."
                )
            proxy.intercept = None
            assert sorted(arrived) == sorted(TENANTS)
            assert {
                tenant: [
                    row["memory"]
                    for row in _rows(managers[tenant], "concurrent_agent").values()
                ]
                for tenant in TENANTS
            } == {tenant: [texts[tenant]] for tenant in TENANTS}
        finally:
            proxy.intercept = None
            for context in contexts:
                context.close()


class TestFaultContract:
    def test_a_store_outage_reads_as_an_outage_not_an_empty_namespace(
        self, page, web_url, store
    ):
        managers, proxy = store
        tenant = TENANTS[0]
        _seed(
            managers[tenant],
            [
                (
                    "m-kept",
                    "_user_memories",
                    "kept through the outage",
                    None,
                    1700000000,
                    False,
                )
            ],
        )

        def intercept(method, path, body):
            if method == "POST" and path.startswith("/search/"):
                return 503, {"message": "injected search outage"}

        proxy.intercept = intercept
        _show(page, web_url, tenant)
        panel = _panel(page, tenant, "_user_memories")
        message = f"Could not read the memories of _user_memories for tenant {tenant}."
        expect(panel.get_by_role("alert")).to_have_text([message, message])
        expect(panel.get_by_role("table")).to_have_count(0)
        expect(panel.get_by_text("No memories in _user_memories.")).to_have_count(0)
        expect(panel.get_by_text("0 live, 0 archived.")).to_have_count(0)

        proxy.intercept = None
        panel.get_by_role("button", name="Refresh").click()
        expect(panel.get_by_text("1 live, 0 archived.")).to_be_visible()
        expect(_table_rows(panel).locator("td:nth-child(4)")).to_have_text(["m-kept"])
