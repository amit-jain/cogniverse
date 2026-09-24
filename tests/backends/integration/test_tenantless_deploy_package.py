"""A deploy that carries no tenant schema activates no document type of its own.

pyvespa gives an ApplicationPackage built without schemas a default document
type named after the application. Nothing registers it, so once it is live
every later deploy refuses it as an unknown schema.
"""

from pathlib import Path

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def vespa_instance(second_vespa):
    """An owned container holding only the metadata schemas."""
    return second_vespa


def _backend(ports):
    BackendRegistry.clear_instances()
    store = VespaConfigStore(backend_port=ports["http_port"])
    return BackendRegistry.get_instance().get_search_backend(
        name="vespa",
        config={
            "backend": {
                "url": "http://localhost",
                "port": ports["http_port"],
                "config_port": ports["config_port"],
            }
        },
        config_manager=ConfigManager(store=store),
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )


def test_a_deploy_without_tenant_schemas_adds_no_document_type(vespa_instance):
    backend = _backend(vespa_instance)
    manager = backend.schema_manager
    before = set(manager.list_deployed_document_types(raise_on_failure=True))
    assert "cogniverse" not in before

    assert backend.deploy_schemas([]) is True

    assert set(manager.list_deployed_document_types(raise_on_failure=True)) == before
    schema = backend.schema_registry.deploy_schema("tenantless:acme", "agent_memories")
    assert schema == "agent_memories_tenantless_acme"
    assert set(manager.list_deployed_document_types(raise_on_failure=True)) == (
        before | {schema}
    )
