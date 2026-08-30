"""Real VespaBackend tenant-scoped metadata queries scope exactly once.

Memory list and provenance fetch now pass the base schema plus tenant_id, and
the backend resolves the tenant-specific physical schema one time before
sending YQL. This test drives the real VespaBackend and VespaSchemaManager
with a recording registry and a stub Vespa app so the seam is exercised
without a cluster.
"""

from __future__ import annotations

import pytest

from cogniverse_core.memory.backend_vector_store import BackendVectorStore
from cogniverse_core.memory.provenance_store import ProvenanceStore
from cogniverse_foundation.config.unified_config import BackendConfig
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "flywheel_org:production"


class RecordingRegistry:
    """Schema registry double that records the exact lookup contract."""

    deployed_bases = {"agent_memories", "provenance"}

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def schema_exists(self, tenant_id: str, base_schema_name: str) -> bool:
        self.calls.append((tenant_id, base_schema_name))
        return base_schema_name in self.deployed_bases


class StubResponse:
    status_code = 200
    json = {"root": {"children": [], "coverage": {"degraded": False}}}


class StubApp:
    def __init__(self):
        self.bodies: list[dict] = []

    def query(self, body):
        self.bodies.append(dict(body))
        return StubResponse()


class Stub:
    pass


def test_tenant_scoped_metadata_queries_scope_once():
    cfg = BackendConfig(tenant_id=TENANT, url="http://vespa.invalid", port=8080)
    backend = VespaBackend(
        backend_config=cfg,
        schema_loader=Stub(),
        config_manager=Stub(),
    )
    registry = RecordingRegistry()
    backend.schema_registry = registry
    backend.schema_manager = VespaSchemaManager(
        backend_endpoint="http://vespa.invalid",
        backend_port=19071,
        schema_registry=registry,
    )
    app = StubApp()
    backend._metadata_vespa_app = lambda: app

    memory_schema = backend.get_tenant_schema_name(TENANT, "agent_memories")
    provenance_schema = backend.get_tenant_schema_name(TENANT, "provenance")

    store = BackendVectorStore(
        collection_name=memory_schema,
        backend_client=backend,
        embedding_model_dims=8,
        tenant_id=TENANT,
        profile="agent_memories",
    )
    assert (
        store._list_page(
            "true", limit=10, offset=0, order_by="created_at desc, id desc"
        )
        == []
    )

    provenance = ProvenanceStore(backend=backend, tenant_id=TENANT)
    assert provenance.get("m1") is None

    assert registry.calls == [
        (TENANT, "agent_memories"),
        (TENANT, "provenance"),
    ]
    assert app.bodies[0]["yql"] == (
        f"select * from {memory_schema} where true order by created_at desc, id desc"
    )
    assert app.bodies[1]["yql"] == (
        f'select * from {provenance_schema} where memory_id in ("m1") '
        f'and tenant_id contains "{TENANT}" limit 100'
    )
