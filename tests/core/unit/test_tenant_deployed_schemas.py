"""The tenant's deployed base schemas, read from the schema registry.

Servability asks this: a profile whose embedding service resolves but whose
tenant schema was never deployed (or was reconciled away) must not be reported
as deployed, and a storage outage must never read as "nothing is deployed".
"""

import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_core.registries.schema_deployment_intents import (
    SchemaDeploymentIntents,
)
from cogniverse_core.registries.schema_registry import (
    SCHEMA_REGISTRY_SERVICE,
    DeployedSchemaNames,
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


def _count_reads(config_manager) -> Counter:
    reads: Counter = Counter()
    lock = threading.Lock()
    original = config_manager.store.list_all_configs

    def _counting(*, scope, service):
        with lock:
            reads[service] += 1
        return original(scope=scope, service=service)

    config_manager.store.list_all_configs = _counting
    return reads


def _registration(base: str, tenant_id: str = TENANT) -> dict:
    return {
        "tenant_id": canonical_tenant_id(tenant_id),
        "base_schema_name": base,
        "full_schema_name": _full(base, tenant_id),
        "schema_definition": '{"name": "%s"}' % _full(base, tenant_id),
        "config": {},
        "deployment_time": "2026-09-09T00:00:00+00:00",
    }


def test_every_registry_write_drops_the_tenant_entry_in_every_reader():
    """Removals are what an entry can hide, so each write is followed by a
    question the stale entry would answer wrongly, or by a cached name whose
    re-read proves the entry was dropped."""
    config_manager = _config_manager()
    registry = _registry(config_manager)
    intents = SchemaDeploymentIntents(config_manager.store)
    registry.register_schema(**_registration("document_text"))
    registry.register_schema(**_registration("wiki_pages", OTHER_TENANT))
    reads = _count_reads(config_manager)
    readers = [
        DeployedSchemaNames(config_manager, ttl_s=3600),
        DeployedSchemaNames(ConfigManager(store=config_manager.store), ttl_s=3600),
    ]
    steps: list = []

    def ask(label, tenant, base):
        before = reads[SCHEMA_REGISTRY_SERVICE]
        answers = [reader(tenant, base) for reader in readers]
        steps.append((label, answers, reads[SCHEMA_REGISTRY_SERVICE] - before))

    ask("warm", TENANT, "document_text")
    ask("warm other", OTHER_TENANT, "wiki_pages")
    ask("cached", TENANT, "document_text")
    registry.register_schema(**_registration("lateon_mv"))
    ask("after register", TENANT, "document_text")
    pending = intents.prepare(_registration("video_frames"), grace_s=90)
    ask("after prepare", TENANT, "document_text")
    intents.retire(pending)
    ask("after retire", TENANT, "video_frames")
    registry.unregister_schema(TENANT, "lateon_mv")
    ask("after unregister", TENANT, "lateon_mv")
    completing = intents.prepare(_registration("audio_content"), grace_s=90)
    ask("after second prepare", TENANT, "audio_content")
    intents.complete(completing)
    ask("after complete", TENANT, "audio_content")
    ask("other untouched", OTHER_TENANT, "wiki_pages")

    assert steps == [
        ("warm", [True, True], 2),
        ("warm other", [True, True], 2),
        ("cached", [True, True], 0),
        ("after register", [True, True], 2),
        ("after prepare", [True, True], 2),
        ("after retire", [False, False], 2),
        ("after unregister", [False, False], 2),
        ("after second prepare", [True, True], 2),
        ("after complete", [False, False], 2),
        ("other untouched", [True, True], 0),
    ]
    assert reads == Counter(
        {SCHEMA_REGISTRY_SERVICE: 16, "schema_deployment_intents": 16}
    )


def test_another_process_deletion_is_seen_after_the_ttl_and_registration_at_once():
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    config_manager = _config_manager()
    registry = _registry(config_manager)
    registry.register_schema(**_registration("document_text"))
    registry.register_schema(**_registration("wiki_pages", OTHER_TENANT))
    reads = _count_reads(config_manager)
    ttl = 0.3
    reader = DeployedSchemaNames(config_manager, ttl_s=ttl)

    def out_of_band(tenant, base, **fields):
        config_manager.store.set_config(
            tenant_id=canonical_tenant_id(tenant),
            scope=ConfigScope.SCHEMA,
            service=SCHEMA_REGISTRY_SERVICE,
            config_key=f"schema_{base}",
            config_value={**_registration(base, tenant), **fields},
        )

    warm = (reader(TENANT, "document_text"), reader(OTHER_TENANT, "wiki_pages"))
    filled = time.monotonic()
    out_of_band(TENANT, "document_text", deleted=True)
    within_ttl = reader(TENANT, "document_text")
    checked_after = time.monotonic() - filled
    reads_within_ttl = Counter(reads)
    out_of_band(OTHER_TENANT, "lateon_mv")
    registered = reader(OTHER_TENANT, "lateon_mv")
    reads_after_registration = Counter(reads)
    time.sleep(max(0.0, filled + ttl + 0.05 - time.monotonic()))
    after_ttl = reader(TENANT, "document_text")

    assert checked_after < ttl
    assert warm == (True, True)
    assert within_ttl is True
    assert reads_within_ttl == Counter(
        {SCHEMA_REGISTRY_SERVICE: 2, "schema_deployment_intents": 2}
    )
    assert registered is True
    assert reads_after_registration == Counter(
        {SCHEMA_REGISTRY_SERVICE: 3, "schema_deployment_intents": 3}
    )
    assert after_ttl is False
    assert reads == Counter(
        {SCHEMA_REGISTRY_SERVICE: 4, "schema_deployment_intents": 4}
    )


def test_an_undeployed_schema_reads_on_every_call_and_concurrent_calls_share_it():
    config_manager = _config_manager()
    register_deployed_schema(config_manager, TENANT, "document_text")
    rows = config_manager.store.list_all_configs
    calls: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def _parked_first_read(*, scope, service):
        calls.append(service)
        if len(calls) == 1:
            entered.set()
            release.wait(timeout=30)
        return rows(scope=scope, service=service)

    config_manager.store.list_all_configs = _parked_first_read
    reader = DeployedSchemaNames(config_manager, ttl_s=3600)
    callers = 8
    barrier = threading.Barrier(callers, timeout=30)
    answers: list = []
    lock = threading.Lock()

    def _ask() -> None:
        barrier.wait()
        answer = reader(TENANT, "lateon_mv")
        with lock:
            answers.append(answer)

    threads = [threading.Thread(target=_ask) for _ in range(callers)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=30)
    # Every caller is past the barrier and parked on the one read in flight.
    time.sleep(0.5)
    release.set()
    for thread in threads:
        thread.join(timeout=30)
    shared = list(calls)
    again = [reader(TENANT, "lateon_mv") for _ in range(3)]
    deployed = reader(TENANT, "document_text")

    assert answers == [False] * callers
    assert shared == [SCHEMA_REGISTRY_SERVICE, "schema_deployment_intents"]
    assert again == [False, False, False]
    assert deployed is True
    assert calls == [SCHEMA_REGISTRY_SERVICE, "schema_deployment_intents"] * 4


def test_ttl_must_be_positive():
    config_manager = _config_manager()
    with pytest.raises(ValueError) as zero:
        DeployedSchemaNames(config_manager, ttl_s=0)
    with pytest.raises(ValueError) as negative:
        DeployedSchemaNames(config_manager, ttl_s=-1.5)

    assert str(zero.value) == "ttl_s must be positive, got 0"
    assert str(negative.value) == "ttl_s must be positive, got -1.5"


def test_concurrent_misses_share_one_failed_read_and_cache_nothing():
    config_manager = _config_manager()
    register_deployed_schema(config_manager, TENANT, "document_text")
    rows = config_manager.store.list_all_configs
    calls: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def _slow_outage(*, scope, service):
        calls.append(service)
        entered.set()
        release.wait(timeout=30)
        raise ConnectionError("config store unreachable")

    config_manager.store.list_all_configs = _slow_outage
    reader = DeployedSchemaNames(config_manager, ttl_s=3600)
    callers = 8
    barrier = threading.Barrier(callers, timeout=30)
    outcomes: list = []
    lock = threading.Lock()

    def _read() -> None:
        barrier.wait()
        try:
            answer = reader(TENANT, "document_text")
        except RegistryStorageError as exc:
            answer = str(exc)
        with lock:
            outcomes.append(answer)

    threads = [threading.Thread(target=_read) for _ in range(callers)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=30)
    # Every caller is past the barrier and parked on the one read in flight.
    time.sleep(0.2)
    release.set()
    for thread in threads:
        thread.join(timeout=30)
    config_manager.store.list_all_configs = rows
    recovered = reader(TENANT, "document_text")

    message = (
        f"Cannot read deployed schemas for tenant {canonical_tenant_id(TENANT)!r}: "
        "ConnectionError: config store unreachable"
    )
    assert outcomes == [message] * callers
    assert calls == [SCHEMA_REGISTRY_SERVICE]
    assert recovered is True
