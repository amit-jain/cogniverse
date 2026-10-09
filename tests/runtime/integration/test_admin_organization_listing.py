"""``GET /admin/organizations`` over the real tenant registry in Vespa.

The registry is read through a forwarding proxy, so a test can make Vespa
time a query out the way it does under load: a registry that does not answer
is a 503 naming the outage, never a 500 server fault, whether the
organization query or one organization's tenant query is the one that times
out.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.admin import tenant_manager as tm
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

CREATED_AT = 1757000000000
SEEDED_SCHEMAS = ["video_colpali_smol500_mv_frame"]

# Vespa's answer to a query that ran out of its time budget.
VESPA_TIMEOUT = (
    504,
    {
        "root": {
            "id": "toplevel",
            "relevance": 1.0,
            "fields": {"totalCount": 0},
            "errors": [
                {
                    "code": 12,
                    "summary": "Timed out",
                    "message": "Search request timed out after 0.5 seconds.",
                }
            ],
        }
    },
)


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader):
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    BackendRegistry.get_instance().clear_instances()


@pytest.fixture
def registry_proxy(wired_tenant_manager, config_manager, vespa_instance):
    """The tenant registry read through a proxy in front of the real Vespa."""
    http_port = vespa_instance["http_port"]
    registry = BackendRegistry.get_instance()
    with InterceptFaultProxy(f"http://localhost:{http_port}") as proxy:
        system = config_manager.get_system_config()
        system.backend_port = proxy.port
        config_manager.set_system_config(system)
        registry.clear_instances()
        try:
            yield proxy
        finally:
            proxy.intercept = None
            system = config_manager.get_system_config()
            system.backend_port = http_port
            config_manager.set_system_config(system)
            registry.clear_instances()


def _client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=tm.app, raise_app_exceptions=False),
        base_url="http://registry-test",
    )


def _seed_organization() -> str:
    org_id = f"orglist{uuid.uuid4().hex[:8]}"
    backend = tm.get_backend()
    assert backend.create_metadata_document(
        schema="organization_metadata",
        doc_id=org_id,
        fields={
            "org_id": org_id,
            "org_name": "Listing Org",
            "created_at": CREATED_AT,
            "created_by": "orglist-test",
            "status": "active",
            "tenant_count": 0,
        },
    )
    assert backend.create_metadata_document(
        schema="tenant_metadata",
        doc_id=f"{org_id}:production",
        fields={
            "tenant_full_id": f"{org_id}:production",
            "org_id": org_id,
            "tenant_name": "production",
            "created_at": CREATED_AT,
            "created_by": "orglist-test",
            "status": "active",
            "schemas_deployed": SEEDED_SCHEMAS,
        },
    )
    return org_id


def _times_out(schema: str):
    """Answer every registry query of ``schema`` as a Vespa timeout."""

    def intercept(method: str, path: str, body: bytes):
        if method == "POST" and path.startswith("/search/"):
            if f"from {schema} " in json.loads(body or b"{}").get("yql", ""):
                return VESPA_TIMEOUT
        return None

    return intercept


async def test_a_healthy_registry_lists_each_organization_with_its_tenants(
    registry_proxy,
):
    org_id = _seed_organization()
    async with _client() as client:
        response = await client.get("/admin/organizations")
    assert response.status_code == 200, response.text
    listed = {o["org_id"]: o for o in response.json()["organizations"]}
    assert listed[org_id] == {
        "org_id": org_id,
        "org_name": "Listing Org",
        "created_at": CREATED_AT,
        "created_by": "orglist-test",
        "status": "active",
        "tenant_count": 1,
        "config": {},
    }


async def test_an_organization_query_timeout_is_a_503_naming_the_registry(
    registry_proxy,
):
    _seed_organization()
    registry_proxy.intercept = _times_out("organization_metadata")
    async with _client() as client:
        response = await client.get("/admin/organizations")
    assert (response.status_code, response.json()) == (
        503,
        {
            "detail": {
                "error": "organization_registry_unavailable",
                "message": "The organization registry did not answer, so the "
                "organizations could not be listed; retry.",
                "failure": "VespaError",
            }
        },
    )


async def test_a_tenant_query_timeout_is_the_tenant_registrys_503(registry_proxy):
    _seed_organization()
    registry_proxy.intercept = _times_out("tenant_metadata")
    async with _client() as client:
        listing = await client.get("/admin/organizations")
    assert (listing.status_code, listing.json()) == (
        503,
        {"detail": "Tenant registry temporarily unavailable"},
    )


async def test_the_listing_answers_again_once_the_registry_does(registry_proxy):
    org_id = _seed_organization()
    registry_proxy.intercept = _times_out("organization_metadata")
    async with _client() as client:
        down = await client.get("/admin/organizations")
        registry_proxy.intercept = None
        up = await client.get("/admin/organizations")
    assert down.status_code == 503
    assert up.status_code == 200, up.text
    assert org_id in {o["org_id"] for o in up.json()["organizations"]}
