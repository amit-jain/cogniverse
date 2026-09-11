"""A search answers "is this tenant's schema deployed?" from the schema
registry through a per-tenant cache of deployed names (a missing name is
re-read before refusing), and the backend registry refuses a cache hit wired
to another source.

Real Vespa (the session's test-owned container) and a real ``VespaConfigStore``
throughout: the deployed tenants' schemas are deployed through the schema
registry and their documents fed to Vespa; the undeployed tenants have the same
profile configured and no schema. Store reads are counted at the store the
backend reads, by method and service.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import requests
from vespa.exceptions import VespaError

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID, canonical_tenant_id
from cogniverse_core.registries.backend_registry import (
    BackendBindingConflictError,
    BackendRegistry,
)
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_core.registries.schema_deployment_intents import (
    _SERVICE as INTENTS_SERVICE,
)
from cogniverse_core.registries.schema_registry import (
    DEPLOYED_SCHEMAS_TTL_S,
    SCHEMA_REGISTRY_SERVICE,
    DeployedSchemaNames,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.interfaces.backend import SchemaNotDeployedError
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

pytestmark = [pytest.mark.integration]

BASE_SCHEMA = "agent_memories"
DEPLOYED_TENANTS = 4
UNDEPLOYED_PER_DEPLOYED = 3
SEARCHES_PER_TENANT = 4
DEAD_STORE_PORT = 29073
WARM_SEARCHES = 50
REFUSED_SEARCHES = 10
# One lookup: the registry rows and the deployment journal.
ONE_LOOKUP = Counter(
    {
        ("list_all_configs", SCHEMA_REGISTRY_SERVICE): 1,
        ("list_all_configs", INTENTS_SERVICE): 1,
    }
)
# A tenant's first search also reads its scoped configs, which ConfigManager
# caches for its scoped_config_cache_ttl_s.
FIRST_SEARCH_READS = ONE_LOOKUP + Counter(
    {
        ("get_config", "backend"): 1,
        ("get_config", "telemetry"): 1,
        ("get_config", "gateway_agent"): 1,
    }
)


def _config_manager(http_port):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    manager = ConfigManager(
        store=VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
    )
    manager.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=http_port)
    )
    return manager


def _backend_config(instance):
    return {
        "backend": {
            "url": "http://localhost",
            "config_port": instance["config_port"],
            "port": instance["http_port"],
        }
    }


def _wait_queryable(http_port, tenant_schema):
    base = f"http://localhost:{http_port}"
    for _ in range(60):
        probe = requests.get(
            f"{base}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if probe.status_code == 200 and "errors" not in probe.json().get("root", {}):
            return
        time.sleep(1)
    pytest.fail(f"{base}: schema {tenant_schema} never activated")


def _wait_visible(http_port, tenant_schema, doc_ids):
    """Until a search over the tenant's schema returns exactly ``doc_ids``."""
    base = f"http://localhost:{http_port}"
    seen: list = []
    for _ in range(60):
        response = requests.get(
            f"{base}/search/",
            params={"yql": f"select id from {tenant_schema} where true", "hits": 10},
            timeout=5,
        )
        root = response.json().get("root", {})
        seen = sorted(hit["fields"]["id"] for hit in root.get("children", []))
        if response.status_code == 200 and seen == sorted(doc_ids):
            return
        time.sleep(1)
    pytest.fail(f"{tenant_schema} never served exactly {sorted(doc_ids)}; last {seen}")


def _feed(http_port, tenant_schema, tenant_id, doc_id, vector):
    put = requests.post(
        f"http://localhost:{http_port}/document/v1/content/{tenant_schema}/docid/{doc_id}",
        json={
            "fields": {
                "id": doc_id,
                "text": f"existence marker {doc_id}",
                "embedding": vector.tolist(),
                "user_id": "test_user",
                "agent_id": "test_agent",
                "metadata_": json.dumps({"tenant_id": tenant_id}),
                "created_at": int(time.time() * 1000),
            }
        },
        timeout=10,
    )
    assert put.status_code == 200, put.text[:300]


@pytest.fixture(scope="module")
def corpus(vespa_instance):
    """Deployed tenants each hold two documents; every undeployed tenant's id
    extends a deployed tenant's id, so a lookup matching by prefix or ignoring
    the tenant answers "deployed" for it."""
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    registry = BackendRegistry.get_instance()
    registry.clear_instances()
    http_port = vespa_instance["http_port"]
    config_manager = _config_manager(http_port)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    run = uuid.uuid4().hex[:8]
    profile_name = f"sxr_profile_{run}"
    rng = np.random.default_rng(23)

    deployed = [f"sxr{run}d{index}" for index in range(DEPLOYED_TENANTS)]
    parent_of = {
        f"{tenant}x{suffix}": tenant
        for tenant in deployed
        for suffix in range(UNDEPLOYED_PER_DEPLOYED)
    }
    undeployed = list(parent_of)
    for tenant in deployed + undeployed:
        config_manager.add_backend_profile(
            BackendProfileConfig(
                profile_name=profile_name,
                type="document",
                schema_name=BASE_SCHEMA,
                embedding_model="lightonai/DenseOn",
                embedding_type="single_vector",
                schema_config={"embedding_dims": 768},
            ),
            tenant_id=tenant,
            service="backend",
        )

    seeded = {}
    for tenant in deployed:
        ingestion = registry.get_ingestion_backend(
            name="vespa",
            tenant_id=tenant,
            config=_backend_config(vespa_instance),
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        ingestion.schema_registry.deploy_schema(
            tenant_id=tenant, base_schema_name=BASE_SCHEMA
        )
        tenant_schema = ingestion.get_tenant_schema_name(tenant, BASE_SCHEMA)
        _wait_queryable(http_port, tenant_schema)
        query_vector = rng.random(768).astype(np.float32)
        near = f"{tenant}_near"
        far = f"{tenant}_far"
        _feed(http_port, tenant_schema, tenant, near, query_vector)
        _feed(http_port, tenant_schema, tenant, far, -query_vector)
        _wait_visible(http_port, tenant_schema, [near, far])
        seeded[tenant] = {"vector": query_vector, "ids": [near, far]}

    registry.clear_instances()
    yield {
        "instance": vespa_instance,
        "config_manager": config_manager,
        "schema_loader": schema_loader,
        "profile_name": profile_name,
        "deployed": deployed,
        "undeployed": undeployed,
        "parent_of": parent_of,
        "seeded": seeded,
    }
    registry.clear_instances()


def _search_backend(env, config_manager=None, schema_loader=None):
    return BackendRegistry.get_instance().get_search_backend(
        name="vespa",
        config=_backend_config(env["instance"]),
        config_manager=config_manager or env["config_manager"],
        schema_loader=schema_loader or env["schema_loader"],
    )


def _query(env, tenant, vector):
    return {
        "query": "existence marker",
        "type": "document",
        "profile": env["profile_name"],
        "strategy": "semantic_search",
        "tenant_id": tenant,
        "top_k": 10,
        "query_embeddings": vector,
    }


def _not_deployed_message(env, tenant):
    schema = f"{BASE_SCHEMA}_{canonical_tenant_id(tenant).replace(':', '_')}"
    return (
        f"Tenant '{tenant}' has no deployed schema '{schema}' (base schema "
        f"'{BASE_SCHEMA}', profile '{env['profile_name']}'); deploy it before "
        "searching this profile"
    )


@pytest.fixture
def constructions(monkeypatch):
    """Every VespaBackend the real registry builds, by the tenant it serves."""
    from cogniverse_vespa.backend import VespaBackend

    built: list[str] = []
    lock = threading.Lock()
    original_init = VespaBackend.__init__

    def recording_init(self, backend_config, schema_loader=None, config_manager=None):
        with lock:
            built.append(backend_config.tenant_id)
        original_init(
            self,
            backend_config,
            schema_loader=schema_loader,
            config_manager=config_manager,
        )

    monkeypatch.setattr(VespaBackend, "__init__", recording_init)
    return built


@pytest.fixture
def store_reads(corpus, monkeypatch):
    """Every read the corpus config store serves, as (method, service)."""
    store = corpus["config_manager"].store
    reads: Counter = Counter()
    lock = threading.Lock()
    for method in (
        "get_config",
        "get_config_history",
        "list_configs",
        "list_all_configs",
    ):
        original = getattr(store, method)

        def counting(*args, _method=method, _original=original, **kwargs):
            with lock:
                reads[(_method, kwargs.get("service"))] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, method, counting)
    return reads


def _counted_manager(env):
    """A ConfigManager over the corpus store with nothing cached yet, holding
    tenant scoped configs as long as a deployed-schema entry lives."""
    from cogniverse_foundation.config.manager import ConfigManager

    return ConfigManager(
        store=env["config_manager"].store,
        scoped_config_cache_ttl_s=DEPLOYED_SCHEMAS_TTL_S,
    )


def _direct_backend(env, reader):
    from cogniverse_vespa.search_backend import VespaSearchBackend

    return VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": env["instance"]["http_port"],
            "profiles": {
                env["profile_name"]: {
                    "type": "document",
                    "schema_name": BASE_SCHEMA,
                    "embedding_model": "lightonai/DenseOn",
                    "embedding_type": "single_vector",
                }
            },
        },
        schema_loader=env["schema_loader"],
        is_schema_deployed=reader,
    )


def _profiled_tenant(env, label):
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    tenant = f"sxr{label}{uuid.uuid4().hex[:8]}"
    env["config_manager"].add_backend_profile(
        BackendProfileConfig(
            profile_name=env["profile_name"],
            type="document",
            schema_name=BASE_SCHEMA,
            embedding_model="lightonai/DenseOn",
            embedding_type="single_vector",
            schema_config={"embedding_dims": 768},
        ),
        tenant_id=tenant,
        service="backend",
    )
    return tenant


def _ingestion(env, tenant):
    return BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config=_backend_config(env["instance"]),
        config_manager=env["config_manager"],
        schema_loader=env["schema_loader"],
    )


def _deploy_and_seed(env, tenant, ingestion):
    """Deploy the tenant's schema through the registry, feed two documents."""
    http_port = env["instance"]["http_port"]
    ingestion.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name=BASE_SCHEMA
    )
    tenant_schema = ingestion.get_tenant_schema_name(tenant, BASE_SCHEMA)
    _wait_queryable(http_port, tenant_schema)
    vector = np.random.default_rng(29).random(768).astype(np.float32)
    ids = [f"{tenant}_near", f"{tenant}_far"]
    _feed(http_port, tenant_schema, tenant, ids[0], vector)
    _feed(http_port, tenant_schema, tenant, ids[1], -vector)
    _wait_visible(http_port, tenant_schema, ids)
    return vector, ids


class TestExistenceReadBuildsNoBackend:
    def test_searches_answer_from_the_registry_and_build_no_ingestion_backend(
        self, corpus, constructions
    ):
        env = corpus
        endpoint = f"http://localhost:{env['instance']['http_port']}"
        BackendRegistry.get_instance().clear_instances()
        backend = _search_backend(env)
        built_for_search = list(constructions)

        results = {}
        for tenant in env["deployed"]:
            for _ in range(SEARCHES_PER_TENANT):
                hits = backend.search(
                    _query(env, tenant, env["seeded"][tenant]["vector"])
                )
                results.setdefault(tenant, []).append([h.document.id for h in hits])
        undeployed = env["undeployed"][0]
        with pytest.raises(SchemaNotDeployedError) as refused:
            backend.search(
                _query(env, undeployed, env["seeded"][env["deployed"][0]]["vector"])
            )

        assert built_for_search == [SYSTEM_TENANT_ID]
        assert constructions == [SYSTEM_TENANT_ID]
        assert BackendRegistry._backend_instances.keys() == [f"search_vespa@{endpoint}"]
        assert results == {
            tenant: [env["seeded"][tenant]["ids"]] * SEARCHES_PER_TENANT
            for tenant in env["deployed"]
        }
        assert str(refused.value) == _not_deployed_message(env, undeployed)


class TestConcurrentTenantsGetTheirOwnAnswer:
    def test_sixteen_tenants_searching_at_once(
        self, corpus, constructions, store_reads, monkeypatch
    ):
        import cogniverse_core.registries.schema_registry as schema_registry_module

        env = corpus
        lookups: list[str] = []
        lookups_lock = threading.Lock()
        read = schema_registry_module.tenant_deployed_schema_names

        def recorded_read(config_manager, tenant_id):
            with lookups_lock:
                lookups.append(tenant_id)
            return read(config_manager, tenant_id)

        monkeypatch.setattr(
            schema_registry_module, "tenant_deployed_schema_names", recorded_read
        )
        endpoint = f"http://localhost:{env['instance']['http_port']}"
        BackendRegistry.get_instance().clear_instances()
        backend = _search_backend(env, config_manager=_counted_manager(env))
        backend.get_embedding_requirements(BASE_SCHEMA)
        tenants = env["deployed"] + env["undeployed"]
        assert len(tenants) == 16
        store_reads.clear()
        began = time.monotonic()
        # Undeployed tenants query with a deployed tenant's vector, so a
        # lookup that answered "deployed" for them would return real hits.
        vector_of = {
            tenant: env["seeded"][env["parent_of"].get(tenant, tenant)]["vector"]
            for tenant in tenants
        }
        start = threading.Barrier(len(tenants) * SEARCHES_PER_TENANT)
        outcome: dict[str, list] = {tenant: [] for tenant in tenants}
        lock = threading.Lock()

        def search(tenant):
            start.wait(timeout=60)
            try:
                answer = [
                    h.document.id
                    for h in backend.search(_query(env, tenant, vector_of[tenant]))
                ]
            except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                answer = (type(exc).__name__, str(exc))
            with lock:
                outcome[tenant].append(answer)

        threads = [
            threading.Thread(target=search, args=(tenant,))
            for tenant in tenants
            for _ in range(SEARCHES_PER_TENANT)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
        elapsed = time.monotonic() - began
        reads_during_searches = Counter(store_reads)
        lookups_during_searches = list(lookups)
        reader = backend._vespa_search_backend._is_schema_deployed
        entries = {tenant: names for tenant, (names, _) in reader._entries.items()}
        lookups.clear()
        afterwards = {tenant: reader(tenant, BASE_SCHEMA) for tenant in tenants}
        canonical = {tenant: canonical_tenant_id(tenant) for tenant in tenants}

        assert [thread.is_alive() for thread in threads] == [False] * len(threads)
        assert elapsed < DEPLOYED_SCHEMAS_TTL_S, (
            f"64 searches took {elapsed:.1f}s, past the {DEPLOYED_SCHEMAS_TTL_S}s "
            "entry lifetime this pin counts reads within"
        )
        # Every deployed tenant's four searches share one read or hit its
        # entry; an undeployed tenant re-reads on each refusal that finds no
        # read in flight, so only the deployed tenants' count is fixed.
        assert Counter(
            tenant
            for tenant in lookups_during_searches
            if tenant in {canonical[t] for t in env["deployed"]}
        ) == Counter({canonical[t]: 1 for t in env["deployed"]})
        assert {
            key: count
            for key, count in reads_during_searches.items()
            if key[0] == "get_config"
        } == {key: len(tenants) for key in FIRST_SEARCH_READS if key[0] == "get_config"}
        assert {
            key: count
            for key, count in reads_during_searches.items()
            if key[0] == "list_all_configs"
        } == {key: len(lookups_during_searches) for key in ONE_LOOKUP}
        assert entries == {
            canonical[tenant]: frozenset({BASE_SCHEMA}) for tenant in env["deployed"]
        }
        assert afterwards == {
            **{tenant: True for tenant in env["deployed"]},
            **{tenant: False for tenant in env["undeployed"]},
        }
        assert lookups == [canonical[tenant] for tenant in env["undeployed"]]
        assert constructions == [SYSTEM_TENANT_ID]
        assert BackendRegistry._backend_instances.keys() == [f"search_vespa@{endpoint}"]
        assert outcome == {
            **{
                tenant: [env["seeded"][tenant]["ids"]] * SEARCHES_PER_TENANT
                for tenant in env["deployed"]
            },
            **{
                tenant: [("SchemaNotDeployedError", _not_deployed_message(env, tenant))]
                * SEARCHES_PER_TENANT
                for tenant in env["undeployed"]
            },
        }


class TestRegistryOutageIsNotNotDeployed:
    def test_dead_registry_store_raises_named_error_before_querying(self, corpus):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_vespa.config.config_store import VespaConfigStore
        from cogniverse_vespa.search_backend import VespaSearchBackend

        env = corpus
        tenant = env["deployed"][0]
        dead_manager = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://127.0.0.1", backend_port=DEAD_STORE_PORT
            )
        )
        profiles = {
            env["profile_name"]: {
                "type": "document",
                "schema_name": BASE_SCHEMA,
                "embedding_model": "lightonai/DenseOn",
                "embedding_type": "single_vector",
            }
        }
        backend = VespaSearchBackend(
            config={
                "url": "http://localhost",
                "port": env["instance"]["http_port"],
                "profiles": profiles,
            },
            schema_loader=env["schema_loader"],
            is_schema_deployed=DeployedSchemaNames(dead_manager),
        )
        try:
            with pytest.raises(RegistryStorageError) as failure:
                backend.search(_query(env, tenant, env["seeded"][tenant]["vector"]))
        finally:
            backend.close()

        assert type(failure.value) is RegistryStorageError
        assert str(failure.value).startswith(
            f"Cannot read deployed schemas for tenant '{canonical_tenant_id(tenant)}': "
            "ConfigStoreUnavailableError: Failed to read Vespa config visit after 5 "
            "attempts over "
        )
        assert f"port={DEAD_STORE_PORT}" in str(failure.value)
        assert type(failure.value.__cause__) is ConfigStoreUnavailableError
        assert backend.metrics.total_searches == 0


class TestCacheHitDependencyBinding:
    """The real VespaBackend through the real registry: the instance handed
    out is the one wired to the requester's config store and schema source."""

    def test_same_sources_share_one_instance(self, corpus):
        env = corpus
        BackendRegistry.get_instance().clear_instances()
        first = _search_backend(env)
        same_objects = _search_backend(env)
        fresh_objects = _search_backend(
            env,
            config_manager=_config_manager(env["instance"]["http_port"]),
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas").resolve()),
        )

        assert same_objects is first
        assert fresh_objects is first
        assert first.config_manager is env["config_manager"]
        assert first.schema_loader is env["schema_loader"]

    def test_other_config_store_is_refused_naming_both(self, corpus):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from tests.utils.memory_store import InMemoryConfigStore

        env = corpus
        BackendRegistry.get_instance().clear_instances()
        http_port = env["instance"]["http_port"]
        endpoint = f"http://localhost:{http_port}"
        _search_backend(env)
        other_store = InMemoryConfigStore()
        other_manager = ConfigManager(store=other_store)
        other_manager.set_system_config(
            SystemConfig(backend_url="http://localhost", backend_port=http_port)
        )

        with pytest.raises(BackendBindingConflictError) as refused:
            _search_backend(env, config_manager=other_manager)

        assert str(refused.value) == (
            f"Cached backend search_vespa@{endpoint} is bound to a config_manager "
            f"reading ('vespa', '{endpoint}', 'config_metadata'); the requester's "
            f"reads ('memory', {id(other_store)}). Backends are shared per "
            "endpoint; requesters at one endpoint must read one config_manager "
            "source."
        )

    def test_other_schema_directory_is_refused_naming_both(self, corpus, tmp_path):
        env = corpus
        BackendRegistry.get_instance().clear_instances()
        endpoint = f"http://localhost:{env['instance']['http_port']}"
        _search_backend(env)

        with pytest.raises(BackendBindingConflictError) as refused:
            _search_backend(env, schema_loader=FilesystemSchemaLoader(tmp_path))

        assert str(refused.value) == (
            f"Cached backend search_vespa@{endpoint} is bound to a schema_loader "
            f"reading ('filesystem', {str(Path('configs/schemas').resolve())!r}); "
            f"the requester's reads ('filesystem', {str(tmp_path.resolve())!r}). "
            "Backends are shared per endpoint; requesters at one endpoint must "
            "read one schema_loader source."
        )

    def test_loopback_spellings_of_one_store_share_one_instance(self, corpus):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_vespa.config.config_store import VespaConfigStore

        env = corpus
        BackendRegistry.get_instance().clear_instances()
        http_port = env["instance"]["http_port"]
        first = _search_backend(env)
        respelled_store = VespaConfigStore(
            backend_url="http://127.0.0.1/", backend_port=http_port
        )
        respelled = _search_backend(
            env,
            config_manager=ConfigManager(store=respelled_store),
            schema_loader=FilesystemSchemaLoader(Path("configs/../configs/schemas")),
        )

        assert respelled is first
        assert respelled_store.source == env["config_manager"].store.source
        assert respelled_store.source == (
            "vespa",
            f"http://localhost:{http_port}",
            "config_metadata",
        )


class TestWarmSearchesReadNothing:
    def test_warm_searches_do_no_store_reads(self, corpus, store_reads):
        env = corpus
        tenant = env["deployed"][0]
        vector = env["seeded"][tenant]["vector"]
        BackendRegistry.get_instance().clear_instances()
        manager = _counted_manager(env)
        backend = _search_backend(env, config_manager=manager)
        backend.get_embedding_requirements(BASE_SCHEMA)
        reader = backend._vespa_search_backend._is_schema_deployed

        store_reads.clear()
        began = time.monotonic()
        first = [h.document.id for h in backend.search(_query(env, tenant, vector))]
        miss_reads = Counter(store_reads)
        store_reads.clear()
        warm = [
            [h.document.id for h in backend.search(_query(env, tenant, vector))]
            for _ in range(WARM_SEARCHES)
        ]
        elapsed = time.monotonic() - began

        assert type(reader) is DeployedSchemaNames
        assert reader.ttl_s == DEPLOYED_SCHEMAS_TTL_S
        assert reader.config_manager is manager
        assert elapsed < DEPLOYED_SCHEMAS_TTL_S, (
            f"{WARM_SEARCHES + 1} searches took {elapsed:.1f}s, past the "
            f"{DEPLOYED_SCHEMAS_TTL_S}s entry lifetime"
        )
        assert first == env["seeded"][tenant]["ids"]
        assert miss_reads == FIRST_SEARCH_READS
        assert warm == [env["seeded"][tenant]["ids"]] * WARM_SEARCHES
        assert store_reads == Counter()


class TestUndeployedSchemaIsReadEverySearch:
    def test_each_refusal_costs_one_lookup_and_caches_nothing(
        self, corpus, store_reads
    ):
        env = corpus
        tenant = env["undeployed"][0]
        vector = env["seeded"][env["parent_of"][tenant]]["vector"]
        BackendRegistry.get_instance().clear_instances()
        backend = _search_backend(env, config_manager=_counted_manager(env))
        backend.get_embedding_requirements(BASE_SCHEMA)
        reader = backend._vespa_search_backend._is_schema_deployed

        store_reads.clear()
        refusals = []
        for _ in range(REFUSED_SEARCHES):
            with pytest.raises(SchemaNotDeployedError) as refused:
                backend.search(_query(env, tenant, vector))
            refusals.append(str(refused.value))

        assert refusals == [_not_deployed_message(env, tenant)] * REFUSED_SEARCHES
        assert store_reads == FIRST_SEARCH_READS + Counter(
            {key: REFUSED_SEARCHES - 1 for key in ONE_LOOKUP}
        )
        assert reader._entries == {}


class TestInProcessDeployAndDelete:
    def test_deploy_is_seen_at_once_and_a_delete_drops_the_entry(
        self, corpus, store_reads
    ):
        env = corpus
        tenant = _profiled_tenant(env, "inproc")
        # An hour-long entry: only invalidation can make the deletion visible.
        backend = _direct_backend(env, DeployedSchemaNames(env["config_manager"], 3600))
        probe = _query(env, tenant, np.ones(768, dtype=np.float32))
        try:
            store_reads.clear()
            with pytest.raises(SchemaNotDeployedError) as before:
                backend.search(probe)
            with pytest.raises(SchemaNotDeployedError) as again:
                backend.search(probe)
            reads_before = Counter(store_reads)

            ingestion = _ingestion(env, tenant)
            vector, ids = _deploy_and_seed(env, tenant, ingestion)
            query = _query(env, tenant, vector)
            store_reads.clear()
            after_deploy = [h.document.id for h in backend.search(query)]
            reads_after_deploy = Counter(store_reads)
            store_reads.clear()
            cached = [h.document.id for h in backend.search(query)]
            reads_cached = Counter(store_reads)

            # The production delete: Vespa drops the schema, then its row is
            # tombstoned; a stale entry would send the query on to Vespa.
            ingestion.delete_schema(BASE_SCHEMA, tenant_id=tenant)
            store_reads.clear()
            with pytest.raises(SchemaNotDeployedError) as after_delete:
                backend.search(query)
            reads_after_delete = Counter(store_reads)
        finally:
            backend.close()
            BackendRegistry.get_instance().clear_instances()

        assert str(before.value) == _not_deployed_message(env, tenant)
        assert str(again.value) == _not_deployed_message(env, tenant)
        assert reads_before == Counter({key: 2 for key in ONE_LOOKUP})
        assert after_deploy == ids
        assert reads_after_deploy == ONE_LOOKUP
        assert cached == ids
        assert reads_cached == Counter()
        assert str(after_delete.value) == _not_deployed_message(env, tenant)
        assert reads_after_delete == ONE_LOOKUP


_OTHER_PROCESS = """
import json, sys
from pathlib import Path

import cogniverse_vespa  # noqa: F401
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore

http_port, config_port = int(sys.argv[1]), int(sys.argv[2])
action, tenant, payload = sys.argv[3], sys.argv[4], json.loads(sys.argv[5])
manager = ConfigManager(
    store=VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
)
ingestion = BackendRegistry.get_instance().get_ingestion_backend(
    name="vespa",
    tenant_id=tenant,
    config={"backend": {"url": "http://localhost", "port": http_port, "config_port": config_port}},
    config_manager=manager,
    schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
)
print("ready", flush=True)
sys.stdin.readline()
if action == "register":
    ingestion.schema_registry.register_schema(**payload)
else:
    ingestion.delete_schema(payload["base_schema_name"], tenant_id=tenant)
print("done", flush=True)
"""

# Long enough to cover the other process's delete (a redeploy plus its
# convergence); the before-TTL check fails loudly if it ever is not.
CROSS_PROCESS_TTL_S = 60.0


def _other_process(env, action, tenant, payload):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _OTHER_PROCESS,
            str(env["instance"]["http_port"]),
            str(env["instance"]["config_port"]),
            action,
            tenant,
            json.dumps(payload),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run(other):
    other.stdin.write("go\n")
    other.stdin.flush()
    assert other.stdout.readline() == "done\n", other.stderr.read()


def _wait_removed(http_port, tenant_schema):
    """Until Vespa itself refuses a query naming the schema."""
    base = f"http://localhost:{http_port}"
    for _ in range(120):
        probe = requests.get(
            f"{base}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if "errors" in probe.json().get("root", {}):
            return
        time.sleep(1)
    pytest.fail(f"{base}: schema {tenant_schema} still resolves after its delete")


def _vespa_refusal(exc):
    """Each Vespa error as (code, summary, message up to its valid-refs list)."""
    return [
        (error["code"], error["summary"], error["message"].split(". Valid source")[0])
        for error in ast.literal_eval(str(exc))
    ]


class TestOtherProcessDeleteIsSeenAfterTheTtl:
    def test_deleted_by_another_process_reaches_vespa_until_the_ttl(
        self, corpus, store_reads
    ):
        env = corpus
        tenant = _profiled_tenant(env, "xdel")
        ingestion = _ingestion(env, tenant)
        vector, ids = _deploy_and_seed(env, tenant, ingestion)
        tenant_schema = ingestion.get_tenant_schema_name(tenant, BASE_SCHEMA)
        BackendRegistry.get_instance().clear_instances()
        other = _other_process(env, "delete", tenant, {"base_schema_name": BASE_SCHEMA})
        backend = _direct_backend(
            env, DeployedSchemaNames(env["config_manager"], CROSS_PROCESS_TTL_S)
        )
        query = _query(env, tenant, vector)
        try:
            assert other.stdout.readline() == "ready\n", other.stderr.read()
            filled = time.monotonic()
            warm = [h.document.id for h in backend.search(query)]
            _run(other)
            _wait_removed(env["instance"]["http_port"], tenant_schema)
            store_reads.clear()
            with pytest.raises(VespaError) as from_vespa:
                backend.search(query)
            reads_within_ttl = Counter(store_reads)
            checked_after = time.monotonic() - filled

            time.sleep(max(0.0, filled + CROSS_PROCESS_TTL_S + 1.0 - time.monotonic()))
            store_reads.clear()
            with pytest.raises(SchemaNotDeployedError) as after_ttl:
                backend.search(query)
            reads_after_ttl = Counter(store_reads)
        finally:
            backend.close()
            other.kill()
            other.wait(timeout=30)

        assert checked_after < CROSS_PROCESS_TTL_S, (
            f"the within-TTL search ran {checked_after:.1f}s after the fill, past "
            f"the {CROSS_PROCESS_TTL_S}s entry lifetime"
        )
        assert warm == ids
        assert reads_within_ttl == Counter()
        assert type(from_vespa.value) is VespaError
        assert _vespa_refusal(from_vespa.value) == [
            (
                4,
                "Invalid query parameter",
                f"Could not resolve source ref '{tenant_schema}'",
            )
        ]
        assert str(after_ttl.value) == _not_deployed_message(env, tenant)
        assert reads_after_ttl == ONE_LOOKUP


class TestOtherProcessRegistrationIsSeenAtOnce:
    def test_registered_by_another_process_is_seen_by_the_next_search(
        self, corpus, store_reads
    ):
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        env = corpus
        tenant = _profiled_tenant(env, "xreg")
        ingestion = _ingestion(env, tenant)
        vector, ids = _deploy_and_seed(env, tenant, ingestion)
        row = (
            env["config_manager"]
            .store.get_config(
                tenant_id=canonical_tenant_id(tenant),
                scope=ConfigScope.SCHEMA,
                service=SCHEMA_REGISTRY_SERVICE,
                config_key=f"schema_{BASE_SCHEMA}",
            )
            .config_value
        )
        # The schema stays live in Vespa with its registration withdrawn, and
        # the tenant keeps another registered name, so its entry is a
        # non-empty set that lacks the schema the search asks for.
        ingestion.schema_registry.unregister_schema(tenant, BASE_SCHEMA)
        other_base = "wiki_pages"
        other_full = ingestion.get_tenant_schema_name(tenant, other_base)
        ingestion.schema_registry.register_schema(
            tenant_id=tenant,
            base_schema_name=other_base,
            full_schema_name=other_full,
            schema_definition=json.dumps(
                {**env["schema_loader"].load_schema(other_base), "name": other_full}
            ),
        )
        BackendRegistry.get_instance().clear_instances()
        other = _other_process(env, "register", tenant, row)
        reader = DeployedSchemaNames(env["config_manager"], 3600)
        backend = _direct_backend(env, reader)
        query = _query(env, tenant, vector)
        try:
            assert other.stdout.readline() == "ready\n", other.stderr.read()
            store_reads.clear()
            other_cached = reader(tenant, other_base)
            with pytest.raises(SchemaNotDeployedError) as before:
                backend.search(query)
            entry_before = reader._entries[canonical_tenant_id(tenant)][0]
            reads_before = Counter(store_reads)
            _run(other)
            store_reads.clear()
            after = [h.document.id for h in backend.search(query)]
            reads_after = Counter(store_reads)
        finally:
            backend.close()
            other.kill()
            other.wait(timeout=30)
            # A live schema without its row blocks every later deploy in the
            # session, so the row is restored whatever happened above.
            cleanup = _ingestion(env, tenant).schema_registry
            cleanup.register_schema(**row)
            cleanup.unregister_schema(tenant, other_base)
            BackendRegistry.get_instance().clear_instances()

        assert other_cached is True
        assert str(before.value) == _not_deployed_message(env, tenant)
        assert entry_before == frozenset({other_base})
        assert reads_before == Counter({key: 2 for key in ONE_LOOKUP})
        assert after == ids
        assert reads_after == ONE_LOOKUP


class TestReaderFaultContract:
    def test_the_store_down_raises_on_every_read_and_never_serves_stale(
        self, corpus, monkeypatch
    ):
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        env = corpus
        tenant = env["deployed"][0]
        ttl = 1.0
        reader = DeployedSchemaNames(env["config_manager"], ttl)
        warm = reader(tenant, BASE_SCHEMA)
        filled = time.monotonic()
        store = env["config_manager"].store
        failed_reads: list = []

        def store_down(*args, **kwargs):
            failed_reads.append(kwargs.get("service"))
            raise ConfigStoreUnavailableError(
                f"store paused for tenant rows under {ConfigScope.SCHEMA.value}"
            )

        monkeypatch.setattr(store, "list_all_configs", store_down)
        within_ttl = reader(tenant, BASE_SCHEMA)
        reads_within_ttl = list(failed_reads)
        with pytest.raises(RegistryStorageError) as absent_name:
            reader(tenant, "wiki_pages")
        time.sleep(max(0.0, filled + ttl + 0.2 - time.monotonic()))
        with pytest.raises(RegistryStorageError) as expired:
            reader(tenant, BASE_SCHEMA)
        with pytest.raises(RegistryStorageError) as retried:
            reader(tenant, BASE_SCHEMA)

        message = (
            f"Cannot read deployed schemas for tenant '{canonical_tenant_id(tenant)}'"
            ": ConfigStoreUnavailableError: store paused for tenant rows under "
            f"{ConfigScope.SCHEMA.value}"
        )
        assert warm is True
        assert within_ttl is True
        assert reads_within_ttl == []
        assert str(absent_name.value) == message
        assert str(expired.value) == message
        assert type(expired.value.__cause__) is ConfigStoreUnavailableError
        assert str(retried.value) == message
        assert failed_reads == [SCHEMA_REGISTRY_SERVICE] * 3


class TestDeleteDuringAReadIsNotCached:
    def test_unregistration_landing_mid_read_is_seen_by_the_next_call(
        self, corpus, monkeypatch
    ):
        env = corpus
        tenant = _profiled_tenant(env, "race")
        ingestion = _ingestion(env, tenant)
        full_name = ingestion.get_tenant_schema_name(tenant, BASE_SCHEMA)
        ingestion.schema_registry.register_schema(
            tenant_id=tenant,
            base_schema_name=BASE_SCHEMA,
            full_schema_name=full_name,
            schema_definition=json.dumps(
                {**env["schema_loader"].load_schema(BASE_SCHEMA), "name": full_name}
            ),
        )
        reader = DeployedSchemaNames(env["config_manager"], 3600)
        store = env["config_manager"].store
        original = store.list_all_configs
        rows_read = threading.Event()
        unregistered = threading.Event()
        calls: list = []

        def list_all_configs(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(kwargs.get("service"))
            if len(calls) == 1:
                rows_read.set()
                assert unregistered.wait(timeout=60)
            return result

        monkeypatch.setattr(store, "list_all_configs", list_all_configs)
        in_flight: dict = {}
        reading = threading.Thread(
            target=lambda: in_flight.setdefault("answer", reader(tenant, BASE_SCHEMA))
        )
        reading.start()
        try:
            assert rows_read.wait(timeout=60)
            ingestion.schema_registry.unregister_schema(tenant, BASE_SCHEMA)
        finally:
            unregistered.set()
            reading.join(timeout=60)
        calls_for_the_stale_read = list(calls)
        next_answer = reader(tenant, BASE_SCHEMA)
        BackendRegistry.get_instance().clear_instances()

        assert in_flight == {"answer": True}
        assert calls_for_the_stale_read == [SCHEMA_REGISTRY_SERVICE, INTENTS_SERVICE]
        assert next_answer is False
        assert calls == [
            SCHEMA_REGISTRY_SERVICE,
            INTENTS_SERVICE,
            SCHEMA_REGISTRY_SERVICE,
            INTENTS_SERVICE,
        ]
