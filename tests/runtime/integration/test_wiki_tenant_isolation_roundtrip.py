"""Per-tenant wiki isolation against real Vespa.

Two tenants posting wiki pages must resolve distinct ``WikiManager``
instances bound to distinct ``wiki_pages_<canonical>`` schemas, driven
through the real wiki router's per-tenant factory.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.wiki.wiki_manager import WikiManager
from cogniverse_runtime.routers import wiki as wiki_router


@pytest.fixture
def per_tenant_wiki_app(
    vespa_instance, config_manager, schema_loader, served_semantic_embedder
):
    """Mount the wiki router with a per-tenant factory backed by real Vespa.

    Mirrors the production wiring in main.py:
    - One factory function that yields a WikiManager per tenant
    - Each tenant gets its own wiki_pages_<tenant> schema in Vespa
    """
    from cogniverse_core.registries.backend_registry import BackendRegistry

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
            backend_resolver=lambda: backend,
            tenant_id=tenant_id,
            schema_name=backend.get_tenant_schema_name(tenant_id, "wiki_pages"),
        )
        managers[tenant_id] = mgr
        return mgr

    original_factory = wiki_router._wiki_manager_factory
    wiki_router.set_wiki_manager_factory(factory)

    app = FastAPI()
    app.include_router(wiki_router.router, prefix="/wiki")

    try:
        with TestClient(app) as client:
            yield client, managers
    finally:
        wiki_router._wiki_manager_factory = original_factory


@pytest.mark.integration
class TestWikiTenantIsolationRoundTrip:
    def test_two_tenants_get_distinct_managers(self, per_tenant_wiki_app):
        """Saving for tenant_a then tenant_b materialises two distinct
        WikiManager instances."""
        client, managers = per_tenant_wiki_app

        client.post(
            "/wiki/save",
            json={
                "query": "alpha tenant query",
                "response": {"answer": "alpha tenant answer"},
                "tenant_id": "wiki_tenant_a",
            },
        )
        client.post(
            "/wiki/save",
            json={
                "query": "beta tenant query",
                "response": {"answer": "beta tenant answer"},
                "tenant_id": "wiki_tenant_b",
            },
        )

        assert "wiki_tenant_a" in managers
        assert "wiki_tenant_b" in managers
        assert managers["wiki_tenant_a"] is not managers["wiki_tenant_b"]
        assert (
            managers["wiki_tenant_a"]._tenant_id != managers["wiki_tenant_b"]._tenant_id
        )

    def test_managers_use_distinct_schemas(self, per_tenant_wiki_app):
        """Each tenant's WikiManager binds to its own canonical
        ``wiki_pages_<canonical>`` schema, not a shared default."""
        client, managers = per_tenant_wiki_app

        client.post(
            "/wiki/save",
            json={
                "query": "alpha",
                "response": {"answer": "first"},
                "tenant_id": "schema_iso_a",
            },
        )
        client.post(
            "/wiki/save",
            json={
                "query": "beta",
                "response": {"answer": "second"},
                "tenant_id": "schema_iso_b",
            },
        )

        assert (
            managers["schema_iso_a"]._schema_name
            == "wiki_pages_schema_iso_a_schema_iso_a"
        )
        assert (
            managers["schema_iso_b"]._schema_name
            == "wiki_pages_schema_iso_b_schema_iso_b"
        )
        assert (
            managers["schema_iso_a"]._schema_name
            != managers["schema_iso_b"]._schema_name
        )

    def test_factory_caches_one_manager_per_tenant(self, per_tenant_wiki_app):
        """Two saves to the same tenant must reuse the same manager instance.
        If the factory built a new manager every call, that would defeat the
        per-tenant cache and waste schema deployments."""
        client, managers = per_tenant_wiki_app

        client.post(
            "/wiki/save",
            json={
                "query": "first",
                "response": {"answer": "x"},
                "tenant_id": "wiki_cache_test",
            },
        )
        first_mgr = managers["wiki_cache_test"]

        client.post(
            "/wiki/save",
            json={
                "query": "second",
                "response": {"answer": "y"},
                "tenant_id": "wiki_cache_test",
            },
        )
        second_mgr = managers["wiki_cache_test"]

        assert first_mgr is second_mgr


@pytest.mark.integration
class TestARunNeverRetrievesItsOwnPages:
    """The dispatcher files an orchestrated run once, after it succeeded. A
    step of the run files nothing, so the tenant's wiki holds nothing of the
    run for a later step to retrieve while it is still running."""

    QUERY = "compare the harbour clips and write a report"
    ANSWER = "Both harbour clips show the same tugboat; the second is at dawn."
    ENTITIES = [
        {"text": "Tugboat Hercules", "type": "VEHICLE", "context": "the tugboat"},
        {"text": "Harbour Lighthouse", "type": "LOCATION", "context": "the lighthouse"},
        {"text": "Dawn Shift", "type": "EVENT", "context": "at dawn"},
    ]

    def _dispatcher(self, config_manager):
        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        dispatcher = AgentDispatcher(
            agent_registry=AgentRegistry(
                tenant_id="test:unit", config_manager=config_manager
            ),
            config_manager=config_manager,
            schema_loader=None,
        )
        for name, capabilities in (
            ("entity_extraction_agent", ["entity_extraction"]),
            ("orchestrator_agent", ["orchestration"]),
        ):
            dispatcher._registry.register_agent(
                AgentEndpoint(
                    name=name, url="http://localhost:8003", capabilities=capabilities
                )
            )

        async def extract(agent_name, query, context, tenant_id):
            return {
                "status": "success",
                "agent": "entity_extraction_agent",
                "entities": self.ENTITIES,
                "entity_count": len(self.ENTITIES),
                "has_entities": True,
                "dominant_types": ["VEHICLE"],
            }

        async def run(query, context, tenant_id):
            return {
                "status": "success",
                "agent": "orchestrator_agent",
                "message": "Orchestrated the comparison",
                "orchestration_result": {
                    "query": query,
                    "agent_results": {
                        "entity_extraction_agent": await extract(
                            "entity_extraction_agent", query, context, tenant_id
                        ),
                    },
                    "final_output": {"status": "success", "answer": self.ANSWER},
                },
            }

        dispatcher._execute_generic_agent = extract
        dispatcher._execute_orchestration_task = run
        return dispatcher

    @pytest.mark.asyncio
    async def test_steps_file_nothing_and_the_finished_run_is_searchable(
        self, per_tenant_wiki_app, config_manager
    ):
        import asyncio
        import uuid

        from cogniverse_agents.orchestrator_agent import step_context

        _, managers = per_tenant_wiki_app
        tenant = f"wikirun{uuid.uuid4().hex[:8]}:t1"
        dispatcher = self._dispatcher(config_manager)

        async def drain():
            while dispatcher._background_tasks:
                await asyncio.gather(*list(dispatcher._background_tasks))

        step = await dispatcher.dispatch(
            "entity_extraction_agent", self.QUERY, context=step_context(tenant)
        )
        await drain()
        after_step = dict(managers)

        result = await dispatcher.dispatch(
            "orchestrator_agent", self.QUERY, context={"tenant_id": tenant}
        )
        await drain()

        assert step["entities"] == self.ENTITIES
        assert tenant not in after_step
        assert result["answer"] == self.ANSWER
        hits = await asyncio.to_thread(managers[tenant].search, "tugboat dawn", 10)
        [index] = [hit for hit in hits if hit["page_type"] == "index"]
        assert index["content"].splitlines()[:3] == [
            f"# Wiki Index — {tenant}",
            "",
            "_4 pages: 3 topics, 1 sessions._",
        ]
        assert sorted(
            (hit["page_type"], hit["title"], hit["content"])
            for hit in hits
            if hit is not index
        ) == [
            ("session", f"Session — {self.QUERY}", self.ANSWER),
            ("topic", "Dawn Shift", self.ANSWER),
            ("topic", "Harbour Lighthouse", self.ANSWER),
            ("topic", "Tugboat Hercules", self.ANSWER),
        ]
