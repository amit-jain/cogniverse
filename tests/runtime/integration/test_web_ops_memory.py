"""The web client's Memory view, driven in Chromium against Mem0 on real Vespa.

The built client's Node server forwards to the runtime's tenant memory routes
on a real uvicorn socket. Each tenant's Mem0 manager stores in real Vespa
through a fault proxy; its embedder is ``serve_token_embedder``, which speaks
DenseOn's ``/v1/embeddings`` contract with a vector that depends only on the
words of the text, so search order is known in advance. Every action is taken
through the page and its outcome read back from the store.
"""

from __future__ import annotations

import json
import math
import threading

import httpx
import pytest
from playwright.sync_api import Page, expect, sync_playwright

from tests.utils.web_client import (
    recording_telemetry_sink,
    serve_web,
)
from tests.utils.web_ops import (
    memory_on_vespa,
    register_tenant,
    serve_ops_runtime,
    token_embedding,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANTS = ("webmem:alpha", "webmem:beta")
SYSTEM_NOTE = (
    "A system namespace: the runtime manages these memories, so they are "
    "read-only here."
)


@pytest.fixture(scope="module")
def memory_store(vespa_instance, config_manager):
    """A Mem0 manager per tenant on real Vespa, reached through a fault proxy."""
    with memory_on_vespa(vespa_instance, config_manager, TENANTS) as (managers, proxy):
        yield managers, proxy


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
def runtime_url(config_manager, schema_loader, workflow_state_redis_url):
    with serve_ops_runtime(
        config_manager, schema_loader, workflow_state_redis_url
    ) as url:
        yield url


@pytest.fixture()
def web_url(built_client, runtime_url):
    with recording_telemetry_sink() as (sink_url, received):
        with serve_web(
            built_client, runtime_url, telemetry_url=sink_url, built=True
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
    register_tenant(tenant)
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
            rows[mid].get("updated_at") or "—",
            mid,
            "Details",
            "Delete",
        )
    ]


def _score(query: str, text: str) -> str:
    """Vespa's closeness of the two texts' embeddings, as the table shows it."""
    distance = math.dist(token_embedding(query), token_embedding(text))
    return f"{1 / (1 + distance):.3f}"


def _facts(panel):
    return panel.locator('dl[aria-label="Memory store facts"] dd')


# The listing's ID column; a search adds the score after the memory.
ID_COLUMN = "td:nth-child(5)"
SEARCH_ID_COLUMN = "td:nth-child(6)"


class TestMemoryView:
    def test_an_operator_searches_adds_deletes_and_clears_memories(
        self, page, web_url, runtime_url, store
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
        expect(_facts(panel)).to_have_text(
            [tenant, "_user_memories", "healthy: the store answers reads"]
        )
        expect(_table_rows(panel).locator("td")).to_have_text(
            _cells(manager, "_user_memories", ["m-dark", "m-zone"])
        )
        details = panel.get_by_label("Details of memory m-dark")
        details.locator("summary").click()
        shown = json.loads(details.locator("pre").inner_text())
        stored = _rows(manager, "_user_memories")["m-dark"]
        assert shown == {
            "id": "m-dark",
            "memory": "prefers dark mode in every editor",
            "type": "preference",
            "owned": True,
            "category": "ui",
            "metadata": stored["metadata"],
            "created_at": str(stored["created_at"]),
            "updated_at": None,
            "score": None,
        }
        assert stored["metadata"]["category"] == "ui"

        search = panel.get_by_role("form", name="Search memories")
        search.get_by_label("Query").fill("utc timezone")
        search.get_by_role("button", name="Search").click()
        expect(_table_rows(panel).locator(SEARCH_ID_COLUMN)).to_have_text(
            ["m-zone", "m-dark"]
        )
        expect(_table_rows(panel).locator("td:nth-child(2)")).to_have_text(
            [
                _score("utc timezone", "timezone is utc plus five"),
                _score("utc timezone", "prefers dark mode in every editor"),
            ]
        )
        searched = httpx.get(
            f"{runtime_url}/admin/tenant/{tenant}/memories",
            params={"agent_name": "_user_memories", "q": "utc timezone", "limit": 1},
        ).json()["memories"]
        assert [(m["id"], f"{m['score']:.3f}") for m in searched] == [
            ("m-zone", _score("utc timezone", "timezone is utc plus five"))
        ]
        search.get_by_label("Results").fill("1")
        search.get_by_role("button", name="Search").click()
        expect(_table_rows(panel).locator(SEARCH_ID_COLUMN)).to_have_text(["m-zone"])
        search.get_by_label("Results").fill("0")
        search.get_by_role("button", name="Search").click()
        expect(search.get_by_role("alert")).to_have_text(
            "Results must be a whole number from 1 to 200."
        )
        # Show all is not refused over the invalid count: it lists with the
        # count in use (1) and puts it back in the box.
        search.get_by_role("button", name="Show all").click()
        expect(_table_rows(panel).locator(ID_COLUMN)).to_have_text(["m-dark"])
        expect(search.get_by_label("Results")).to_have_value("1")
        expect(search.get_by_role("alert")).to_have_count(0)
        search.get_by_label("Results").fill("20")
        search.get_by_role("button", name="Search").click()
        expect(_table_rows(panel).locator(ID_COLUMN)).to_have_text(["m-dark", "m-zone"])

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
        assert json.loads(page.get_by_label("Saved memory").inner_text()) == {
            "status": "saved",
            "id": saved_id,
            "type": "preference",
            "agent_name": "_user_memories",
            "category": "style",
            "kind": None,
        }
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
        expect(_table_rows(panel).locator(ID_COLUMN)).to_have_text(
            [saved_id, "m-dark", "m-zone"]
        )
        expect(_table_rows(panel).nth(0).locator("td:nth-child(4)")).to_have_text(
            _rows(manager, "_user_memories")[saved_id].get("updated_at") or "—"
        )

        panel.get_by_role("button", name="Delete memory m-zone").click()
        panel.get_by_role("button", name="Confirm delete of m-zone").click()
        expect(page.get_by_role("status")).to_have_text("Deleted memory m-zone.")
        assert set(_rows(manager, "_user_memories")) == {saved_id, "m-dark"}

        _namespace(page, "_strategy_store")
        system = _panel(page, tenant, "_strategy_store")
        expect(system.get_by_text(f"1 live, 0 archived. {SYSTEM_NOTE}")).to_be_visible()
        expect(_table_rows(system).locator(ID_COLUMN)).to_have_text(["m-strategy"])
        expect(
            system.get_by_role("button", name="Delete memory m-strategy")
        ).to_have_count(0)
        expect(system.get_by_role("group", name="Clear namespace")).to_have_count(0)
        expect(page.get_by_role("form", name="Add memory")).to_have_count(0)

        _namespace(page, "search_agent")
        agent = _panel(page, tenant, "search_agent")
        expect(_table_rows(agent).locator(ID_COLUMN)).to_have_text(
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
        search = agent.get_by_role("form", name="Search memories")
        search.get_by_label("Query").fill("lecture clips")
        search.get_by_role("button", name="Search").click()
        expect(agent.get_by_text("No memories match “lecture clips”.")).to_be_visible()
        expect(agent.get_by_role("table")).to_have_count(0)


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
        expect(_facts(panel).nth(2)).to_have_text(
            "unhealthy: The memory store did not answer a read (VespaError)."
        )

        proxy.intercept = None
        panel.get_by_role("button", name="Check health").click()
        expect(_facts(panel).nth(2)).to_have_text("healthy: the store answers reads")
        panel.get_by_role("button", name="Refresh").click()
        expect(panel.get_by_text("1 live, 0 archived.")).to_be_visible()
        expect(_table_rows(panel).locator(ID_COLUMN)).to_have_text(["m-kept"])
