"""The tenant's deployed base schemas, read from the schema registry.

Servability asks this: a profile whose embedding service resolves but whose
tenant schema was never deployed (or was reconciled away) must not be reported
as deployed, and a storage outage must never read as "nothing is deployed".
"""

import threading
from types import SimpleNamespace

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_core.registries.schema_deployment_intents import (
    SchemaDeploymentIntents,
)
from cogniverse_core.registries.schema_registry import (
    SchemaRegistry,
    tenant_deployed_schema_names,
)
from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import (
    InMemoryConfigStore,
    register_deployed_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:prod"
OTHER_TENANT = "rival:prod"


def _config_manager() -> ConfigManager:
    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


def _registry(config_manager: ConfigManager) -> SchemaRegistry:
    """The production registry; ``register_schema`` never touches the backend."""
    return SchemaRegistry(
        config_manager=config_manager,
        backend=SimpleNamespace(),
        schema_loader=SimpleNamespace(),
    )


def _full(base: str, tenant_id: str) -> str:
    return f"{base}_{canonical_tenant_id(tenant_id).replace(':', '_')}"


def test_registered_schemas_are_reported_per_tenant():
    config_manager = _config_manager()
    register_deployed_schema(config_manager, TENANT, "document_text")
    register_deployed_schema(config_manager, TENANT, "lateon_mv")
    register_deployed_schema(config_manager, OTHER_TENANT, "wiki_pages")

    assert tenant_deployed_schema_names(config_manager, TENANT) == frozenset(
        {"document_text", "lateon_mv"}
    )
    assert tenant_deployed_schema_names(config_manager, OTHER_TENANT) == frozenset(
        {"wiki_pages"}
    )
    assert tenant_deployed_schema_names(config_manager, "never:used") == frozenset()


def test_uncanonical_tenant_id_reads_the_canonical_rows():
    config_manager = _config_manager()
    register_deployed_schema(config_manager, "acme", "document_text")

    assert tenant_deployed_schema_names(config_manager, "acme") == frozenset(
        {"document_text"}
    )
    assert tenant_deployed_schema_names(
        config_manager, canonical_tenant_id("acme")
    ) == frozenset({"document_text"})


def test_unregistered_schema_stops_being_reported():
    config_manager = _config_manager()
    registry = _registry(config_manager)
    register_deployed_schema(config_manager, TENANT, "document_text")
    register_deployed_schema(config_manager, TENANT, "lateon_mv")

    registry.unregister_schema(TENANT, "lateon_mv")

    assert tenant_deployed_schema_names(config_manager, TENANT) == frozenset(
        {"document_text"}
    )


def test_an_activation_in_flight_owns_its_name_before_the_row_lands():
    """A prepared intent is an activation this tenant owns right now.

    Registration lands seconds after activation; a window in which the tenant's
    own deploy reads as "not deployed" would un-advertise a profile mid-deploy.
    """
    config_manager = _config_manager()
    intents = SchemaDeploymentIntents(config_manager.store)
    intents.prepare(
        {
            "tenant_id": TENANT,
            "base_schema_name": "lateon_mv",
            "full_schema_name": _full("lateon_mv", TENANT),
            "schema_definition": '{"name": "%s"}' % _full("lateon_mv", TENANT),
            "config": {},
            "deployment_time": "2026-09-09T00:00:00+00:00",
        },
        grace_s=90,
    )

    assert tenant_deployed_schema_names(config_manager, TENANT) == frozenset(
        {"lateon_mv"}
    )
    assert tenant_deployed_schema_names(config_manager, OTHER_TENANT) == frozenset()


def test_registry_read_outage_raises_and_never_reads_as_nothing_deployed():
    config_manager = _config_manager()
    register_deployed_schema(config_manager, TENANT, "document_text")
    outage = ConnectionError("config store unreachable")

    def _fail(**kwargs):
        raise outage

    config_manager.store.list_all_configs = _fail

    with pytest.raises(RegistryStorageError) as failure:
        tenant_deployed_schema_names(config_manager, TENANT)

    assert str(failure.value) == (
        f"Cannot read deployed schemas for tenant {canonical_tenant_id(TENANT)!r}: "
        "ConnectionError: config store unreachable"
    )
    assert failure.value.__cause__ is outage


def test_intent_read_outage_raises_rather_than_dropping_in_flight_names():
    config_manager = _config_manager()
    register_deployed_schema(config_manager, TENANT, "document_text")
    rows = config_manager.store.list_all_configs

    def _fail_intents(*, scope, service):
        if service == "schema_deployment_intents":
            raise ConnectionError("config store unreachable")
        return rows(scope=scope, service=service)

    config_manager.store.list_all_configs = _fail_intents

    with pytest.raises(RegistryStorageError) as failure:
        tenant_deployed_schema_names(config_manager, TENANT)

    assert str(failure.value) == (
        "Cannot read deployment intents: config store unreachable"
    )


def test_concurrent_readers_each_see_the_full_registered_set():
    """One shared store, N concurrent readers: no reader sees a torn subset."""
    config_manager = _config_manager()
    for base in ("document_text", "lateon_mv", "wiki_pages"):
        register_deployed_schema(config_manager, TENANT, base)

    readers = 16
    barrier = threading.Barrier(readers, timeout=30)
    results: list[frozenset] = []
    results_lock = threading.Lock()

    def _read() -> None:
        barrier.wait()
        names = tenant_deployed_schema_names(config_manager, TENANT)
        with results_lock:
            results.append(names)

    threads = [threading.Thread(target=_read) for _ in range(readers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert (
        results == [frozenset({"document_text", "lateon_mv", "wiki_pages"})] * readers
    )
