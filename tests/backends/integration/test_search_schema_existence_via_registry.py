"""A search answers "is this tenant's schema deployed?" from the schema
registry, and the backend registry refuses a cache hit wired to another source.

Real Vespa (the session's test-owned container) and a real ``VespaConfigStore``
throughout: the deployed tenants' schemas are deployed through the schema
registry and their documents fed to Vespa; the undeployed tenants have the same
profile configured and no schema.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID, canonical_tenant_id
from cogniverse_core.registries.backend_registry import (
    BackendBindingConflictError,
    BackendRegistry,
)
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_core.registries.schema_registry import tenant_deployed_schema_names
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.interfaces.backend import SchemaNotDeployedError
from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

pytestmark = [pytest.mark.integration]

BASE_SCHEMA = "agent_memories"
DEPLOYED_TENANTS = 4
UNDEPLOYED_PER_DEPLOYED = 3
SEARCHES_PER_TENANT = 4
DEAD_STORE_PORT = 29073


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
    def test_sixteen_tenants_searching_at_once(self, corpus, constructions):
        env = corpus
        endpoint = f"http://localhost:{env['instance']['http_port']}"
        BackendRegistry.get_instance().clear_instances()
        backend = _search_backend(env)
        tenants = env["deployed"] + env["undeployed"]
        assert len(tenants) == 16
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

        assert [thread.is_alive() for thread in threads] == [False] * len(threads)
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
            deployed_schema_names=partial(tenant_deployed_schema_names, dead_manager),
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
