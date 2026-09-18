"""A dispatch's config-store reads run off the serving event loop."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from requests.exceptions import HTTPError

import cogniverse_vespa  # noqa: F401 — registers the vespa backend
from cogniverse_agents.document_agent import _STRATEGY_SCHEMAS
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa._yql import yql_quote
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.runtime.integration.test_schema_event_loop import assert_responsive
from tests.utils.http_fault_proxy import HTTPFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

SCHEMA = "config_metadata"
"""The store schema ``VespaConfigStore`` visits; asserted against the store."""

QUERY = "find PDF documents about washing dishes"

SESSION_ID = "loopstall-config-read"
"""Session the dispatch runs under; the route seeds ``request_id`` from it, so
the error envelope's request id is the one this test sent."""


def backend_scope_visit(method: str, path: str, _body: bytes) -> bool:
    """THE read ``get_backend`` makes: the system tenant's backend-scope visit.

    ``ConfigUtils.get('backend_type')`` resolves the backend scope for the
    ``system`` tenant the metadata backend is registered under, which
    ``VespaConfigStore`` serves with one ``document/v1`` visit whose
    ``selection`` names that tenant, that scope and the ``backend``
    service. No other read in a dispatch carries it.
    """
    parsed = urlparse(path)
    if method != "GET":
        return False
    if parsed.path != f"/document/v1/{SCHEMA}/{SCHEMA}/docid/":
        return False
    expected = (
        f"{SCHEMA}.tenant_id == {yql_quote(canonical_tenant_id('system'))} and "
        f"{SCHEMA}.scope == {yql_quote(ConfigScope.BACKEND.value)} and "
        f"{SCHEMA}.service == {yql_quote('backend')}"
    )
    return parse_qs(parsed.query).get("selection") == [expected]


@pytest.fixture
def dispatch_env(shared_vespa, monkeypatch):
    """The agents router on a real dispatcher, reading a real Vespa through a
    proxy that can hold one config-store read open."""

    def upstream(path):
        port = (
            shared_vespa["config_port"]
            if path.startswith(("/application/", "/config/"))
            else shared_vespa["http_port"]
        )
        return f"http://127.0.0.1:{port}"

    with HTTPFaultProxy(upstream) as proxy:
        store = VespaConfigStore(
            backend_url="http://127.0.0.1", backend_port=proxy.port
        )
        assert store.schema_name == SCHEMA
        cm = ConfigManager(store=store)
        cm.set_system_config(
            SystemConfig(backend_url="http://127.0.0.1", backend_port=proxy.port)
        )
        loader = FilesystemSchemaLoader(Path("configs/schemas"))
        tenant = f"loopstall:t{uuid.uuid4().hex[:8]}"
        registry = AgentRegistry(tenant_id=tenant, config_manager=cm)
        registry.register_agent(
            AgentEndpoint(
                name="document_agent",
                url="http://localhost:8000",
                capabilities=["document_analysis", "pdf_processing"],
            )
        )
        dispatcher = AgentDispatcher(
            agent_registry=registry, config_manager=cm, schema_loader=loader
        )
        monkeypatch.setattr(tenant_manager, "_backend", None)
        monkeypatch.setattr(tenant_manager, "_config_manager", cm)
        monkeypatch.setattr(tenant_manager, "_schema_loader", loader)
        monkeypatch.setattr(agents, "_dispatcher", dispatcher)

        app = FastAPI()
        app.include_router(agents.router, prefix="/agents")

        @app.get("/heartbeat")
        async def heartbeat():
            return {"status": "responsive"}

        yield SimpleNamespace(proxy=proxy, cm=cm, tenant=tenant, app=app)
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_document_dispatch_answers_peers_while_the_backend_config_read_is_held(
    dispatch_env, failure
):
    """The replica answers an unrelated request while ``get_backend``'s
    config-store read is open.

    Once the read completes the dispatch reports the tenant's undeployed
    document schemas; a read the store refuses ends the turn naming the
    failure and the request, never a zero-result answer that reads as
    "this tenant has no documents".
    """
    env = dispatch_env
    env.proxy.arm(backend_scope_visit, failure=failure)
    async with AsyncClient(
        transport=ASGITransport(app=env.app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/agents/document_agent/process",
                json={
                    "agent_name": "document_agent",
                    "query": QUERY,
                    "context": {"tenant_id": env.tenant},
                    "session_id": SESSION_ID,
                    "top_k": 5,
                },
            )
        )
        await assert_responsive(client, task, env.proxy)
        response = await task

    if failure:
        assert response.status_code == 500, response.text
        assert response.json() == {
            "detail": (
                f"Agent 'document_agent' failed with {HTTPError.__name__} "
                f"(request_id={SESSION_ID}). See runtime logs for detail."
            )
        }
        return
    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": (
            f"No document schema deployed for tenant '{env.tenant}': "
            f"expected one of {sorted(_STRATEGY_SCHEMAS.values())}"
        )
    }
