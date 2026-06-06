"""Real-boundary round-trips for RuntimeClient wiki/instructions/jobs CRUD.

The unit suite in tests/messaging/unit/test_runtime_client_crud.py mocks
RuntimeClient's httpx transport and only asserts the URL + payload the
client *builds*. That proves nothing about whether those URLs resolve to a
live route or whether the round-trip actually persists and retrieves data.

Here RuntimeClient drives a real FastAPI app (wiki + tenant routers) over
httpx.ASGITransport, backed by a real Vespa container. Each method's URL
must resolve to the real handler and the stored data must come back out.
"""

import httpx
import pytest
from cogniverse_messaging.runtime_client import RuntimeClient
from fastapi import FastAPI

from cogniverse_agents.wiki.wiki_manager import WikiManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.routers import tenant as tenant_router
from cogniverse_runtime.routers import wiki as wiki_router

TENANT_ID = "rc_crud"


@pytest.fixture(scope="module")
def crud_app(vespa_instance, config_manager, schema_loader):
    """FastAPI app mounting wiki + tenant routers wired to real Vespa.

    Mirrors main.py: wiki gets a per-tenant factory that deploys a
    wiki_pages_<tenant> schema and binds the manager to the *canonical*
    schema name (get_tenant_schema_name), and the tenant router gets the
    real Vespa-backed ConfigManager.
    """
    BackendRegistry._backend_instances.clear()

    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id="test:unit",
        config={
            "backend": {
                "url": "http://localhost",
                "port": vespa_instance["http_port"],
                "config_port": vespa_instance["config_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    managers: dict = {}

    def factory(tenant_id: str) -> WikiManager:
        if tenant_id in managers:
            return managers[tenant_id]
        try:
            backend.schema_registry.deploy_schema(
                tenant_id=tenant_id, base_schema_name="wiki_pages"
            )
        except Exception:
            pass
        mgr = WikiManager(
            backend=backend,
            tenant_id=tenant_id,
            schema_name=backend.get_tenant_schema_name(tenant_id, "wiki_pages"),
        )
        managers[tenant_id] = mgr
        return mgr

    original_wiki_factory = wiki_router._wiki_manager_factory
    original_config_manager = tenant_router._config_manager
    wiki_router.set_wiki_manager_factory(factory)
    tenant_router.set_config_manager(config_manager)

    app = FastAPI()
    app.include_router(wiki_router.router, prefix="/wiki")
    app.include_router(tenant_router.router, prefix="/admin/tenant")

    try:
        yield app
    finally:
        wiki_router._wiki_manager_factory = original_wiki_factory
        tenant_router._config_manager = original_config_manager


@pytest.fixture
async def runtime_client(crud_app):
    """RuntimeClient whose transport is the in-process crud_app."""
    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=crud_app),
        base_url="http://runtime",
        timeout=60.0,
    )
    try:
        yield rc
    finally:
        await rc._client.aclose()


async def _retry(coro_factory, predicate, attempts: int = 10, delay: float = 1.0):
    """Await coro_factory() until predicate(result) is true or attempts run out."""
    import asyncio

    result = None
    for _ in range(attempts):
        result = await coro_factory()
        if predicate(result):
            return result
        await asyncio.sleep(delay)
    return result


@pytest.mark.integration
class TestRuntimeClientWikiRoundTrip:
    @pytest.mark.asyncio
    async def test_save_topic_index_lint_delete(self, runtime_client):
        rc = runtime_client

        saved = await rc.save_wiki_session(
            tenant_id=TENANT_ID,
            query="What is reinforcement learning?",
            response={"answer": "Reinforcement learning trains agents via reward."},
            entities=["reinforcement_learning"],
            agent_name="search_agent",
        )
        assert saved.get("status") == "saved", saved
        assert saved["doc_id"].startswith("wiki_session_")

        # get_wiki_topic resolves to the real route and returns the stored topic.
        topic = await _retry(
            lambda: rc.get_wiki_topic(
                tenant_id=TENANT_ID, slug="reinforcement_learning"
            ),
            lambda r: r.get("status") != "error",
        )
        assert topic.get("status") != "error", topic
        assert topic["page_type"] == "topic"
        assert topic["title"] == "reinforcement_learning"
        assert "reward" in topic["content"]

        # lint resolves and returns the full report contract shape.
        lint = await rc.lint_wiki(tenant_id=TENANT_ID)
        assert lint.get("status") != "error", lint
        assert {
            "orphan_pages",
            "stale_pages",
            "empty_pages",
            "total_pages",
            "issues_found",
        }.issubset(lint.keys())

        # index resolves and returns the content envelope.
        index = await rc.get_wiki_index(tenant_id=TENANT_ID)
        assert index.get("status") != "error", index
        assert "content" in index

        # delete removes the topic; a subsequent fetch is a real 404.
        deleted = await rc.delete_wiki_topic(
            tenant_id=TENANT_ID, slug="reinforcement_learning"
        )
        assert deleted.get("status") == "deleted", deleted

        gone = await _retry(
            lambda: rc.get_wiki_topic(
                tenant_id=TENANT_ID, slug="reinforcement_learning"
            ),
            lambda r: r.get("status") == "error",
        )
        assert gone.get("status") == "error"
        assert gone.get("status_code") == 404


@pytest.mark.integration
class TestRuntimeClientInstructionsRoundTrip:
    @pytest.mark.asyncio
    async def test_set_then_get_instructions(self, runtime_client):
        rc = runtime_client
        text = "Always cite sources and answer in two sentences."

        set_resp = await rc.set_instructions(tenant_id=TENANT_ID, text=text)
        assert set_resp.get("status") != "error", set_resp
        assert set_resp["text"] == text
        assert set_resp["updated_at"]

        got = await rc.get_instructions(tenant_id=TENANT_ID)
        assert got.get("status") != "error", got
        assert got["text"] == text
        assert got["updated_at"] == set_resp["updated_at"]


@pytest.mark.integration
class TestRuntimeClientJobsRoundTrip:
    @pytest.mark.asyncio
    async def test_create_list_delete_job(self, runtime_client):
        rc = runtime_client

        created = await rc.create_job(
            tenant_id=TENANT_ID,
            name="weekly_news",
            schedule="0 9 * * 1",
            query="latest AI research",
            post_actions=["save to wiki"],
        )
        assert created.get("status") == "created", created
        job_id = created["job_id"]
        assert created["name"] == "weekly_news"
        assert created["schedule"] == "0 9 * * 1"
        assert created["query"] == "latest AI research"
        assert created["post_actions"] == ["save to wiki"]

        listed = await _retry(
            lambda: rc.list_jobs(tenant_id=TENANT_ID),
            lambda r: (
                r.get("status") != "error"
                and any(j["job_id"] == job_id for j in r.get("jobs", []))
            ),
        )
        jobs = listed.get("jobs", [])
        match = next((j for j in jobs if j["job_id"] == job_id), None)
        assert match is not None, listed
        assert match["name"] == "weekly_news"
        assert match["query"] == "latest AI research"
        assert match["post_actions"] == ["save to wiki"]

        deleted = await rc.delete_job(tenant_id=TENANT_ID, job_id=job_id)
        assert deleted.get("status") != "error", deleted

        after = await _retry(
            lambda: rc.list_jobs(tenant_id=TENANT_ID),
            lambda r: (
                r.get("status") != "error"
                and all(j["job_id"] != job_id for j in r.get("jobs", []))
            ),
        )
        assert all(j["job_id"] != job_id for j in after.get("jobs", []))
