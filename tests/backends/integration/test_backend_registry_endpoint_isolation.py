"""Two Vespa clusters, two backends: each answers only from its own.

The registry hands out process-shared search backends. Keyed by name
alone, the second caller's config was discarded and its queries went to
the first caller's cluster — a cross-cluster read that returns real,
wrong documents rather than an error.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.backend_registry import (
    BackendRegistry,
    configure_tenant_cache_capacity,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

DEAD_VESPA_PORT = 29074


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


def _seed(instance, config_manager, schema_loader, tenant_id, marker, vector):
    """Deploy agent_memories for the tenant and PUT one marked document."""
    registry = BackendRegistry.get_instance()
    ingestion = registry.get_ingestion_backend(
        name="vespa",
        tenant_id=tenant_id,
        config=_backend_config(instance),
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    ingestion.schema_registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name="agent_memories"
    )
    tenant_schema = ingestion.get_tenant_schema_name(tenant_id, "agent_memories")

    base = f"http://localhost:{instance['http_port']}"
    for _ in range(40):
        probe = requests.get(
            f"{base}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if probe.status_code == 200 and "errors" not in probe.json().get("root", {}):
            break
        time.sleep(1)
    else:
        pytest.fail(f"{base}: schema {tenant_schema} never activated")

    doc_id = f"sbe_{marker}"
    put = requests.post(
        f"{base}/document/v1/content/{tenant_schema}/docid/{doc_id}",
        json={
            "fields": {
                "id": doc_id,
                "text": f"cluster marker {marker}",
                "embedding": vector.tolist(),
                "user_id": "test_user",
                "agent_id": "test_agent",
                "metadata_": json.dumps({"tenant_id": tenant_id}),
                "created_at": int(time.time() * 1000),
            }
        },
        timeout=10,
    )
    assert put.status_code in (200, 201), put.text[:300]
    return doc_id


@pytest.fixture(scope="module")
def two_clusters(vespa_instance, second_vespa):
    """One tenant + one profile per cluster, each holding a distinct doc."""
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    tenant_id = f"sbe_iso_{uuid.uuid4().hex[:8]}"
    profile_name = f"sbe_profile_{uuid.uuid4().hex[:8]}"
    rng = np.random.default_rng(11)
    vector = rng.random(768).astype(np.float32)

    clusters = {}
    for label, instance in (("alpha", vespa_instance), ("bravo", second_vespa)):
        config_manager = _config_manager(instance["http_port"])
        config_manager.add_backend_profile(
            BackendProfileConfig(
                profile_name=profile_name,
                type="document",
                schema_name="agent_memories",
                embedding_model="lightonai/DenseOn",
                embedding_type="single_vector",
                schema_config={"embedding_dims": 768},
            ),
            tenant_id=tenant_id,
            service="backend",
        )
        marker = f"{label}_{uuid.uuid4().hex[:10]}"
        doc_id = _seed(
            instance, config_manager, schema_loader, tenant_id, marker, vector
        )
        clusters[label] = {
            "instance": instance,
            "config_manager": config_manager,
            "doc_id": doc_id,
        }

    time.sleep(3)
    yield {
        "clusters": clusters,
        "schema_loader": schema_loader,
        "tenant_id": tenant_id,
        "profile_name": profile_name,
        "vector": vector,
    }
    BackendRegistry._backend_instances.clear()


def _search_backend(env, label):
    cluster = env["clusters"][label]
    return BackendRegistry.get_instance().get_search_backend(
        name="vespa",
        config=_backend_config(cluster["instance"]),
        config_manager=cluster["config_manager"],
        schema_loader=env["schema_loader"],
    )


def _hit_ids(env, backend):
    results = backend.search(
        {
            "query": "cluster marker",
            "type": "document",
            "profile": env["profile_name"],
            "strategy": "semantic_search",
            "tenant_id": env["tenant_id"],
            "top_k": 10,
            "query_embeddings": env["vector"],
        }
    )
    return [r.document.id for r in results]


@pytest.mark.integration
class TestEachBackendQueriesItsOwnCluster:
    def test_two_endpoints_yield_two_backends_each_reading_its_own_document(
        self, two_clusters
    ):
        env = two_clusters
        alpha = _search_backend(env, "alpha")
        bravo = _search_backend(env, "bravo")

        assert alpha is not bravo
        assert _hit_ids(env, alpha) == [env["clusters"]["alpha"]["doc_id"]]
        assert _hit_ids(env, bravo) == [env["clusters"]["bravo"]["doc_id"]]

    def test_repeat_request_for_one_endpoint_reuses_that_instance(self, two_clusters):
        env = two_clusters
        assert _search_backend(env, "alpha") is _search_backend(env, "alpha")
        assert _search_backend(env, "bravo") is _search_backend(env, "bravo")


@pytest.mark.integration
class TestDeadEndpointDoesNotPoisonTheLiveOne:
    def test_dead_endpoint_search_fails_naming_it_and_alpha_keeps_serving(
        self, two_clusters
    ):
        env = two_clusters
        alpha = _search_backend(env, "alpha")
        alpha_before = _hit_ids(env, alpha)

        dead = BackendRegistry.get_instance().get_search_backend(
            name="vespa",
            config={
                "backend": {
                    "url": "http://127.0.0.1",
                    "config_port": DEAD_VESPA_PORT,
                    "port": DEAD_VESPA_PORT,
                }
            },
            config_manager=env["clusters"]["alpha"]["config_manager"],
            schema_loader=env["schema_loader"],
        )
        assert dead is not alpha

        with pytest.raises(Exception) as excinfo:
            dead.search(
                {
                    "query": "cluster marker",
                    "type": "document",
                    "profile": env["profile_name"],
                    "strategy": "semantic_search",
                    "tenant_id": env["tenant_id"],
                    "top_k": 10,
                    "query_embeddings": env["vector"],
                }
            )
        assert str(DEAD_VESPA_PORT) in str(excinfo.value)

        assert (
            BackendRegistry.get_instance().get_search_backend(
                name="vespa",
                config=_backend_config(env["clusters"]["alpha"]["instance"]),
                config_manager=env["clusters"]["alpha"]["config_manager"],
                schema_loader=env["schema_loader"],
            )
            is alpha
        )
        assert _hit_ids(env, alpha) == alpha_before


@pytest.mark.integration
class TestEvictionRefusesTheHolder:
    def test_holder_of_an_evicted_backend_is_refused_naming_its_endpoint(
        self, two_clusters
    ):
        """An agent keeps the backend the registry handed it. Another tenant's
        traffic fills the cache, the agent's instance is evicted and closed,
        and its next search must be refused naming the endpoint it can no
        longer reach — not rebuild its clients unnoticed and serve from
        outside the cache.

        The holder reads its own document first, so the refusal can only come
        from the eviction the evictor thread caused.

        Capacity is 1: a search occupies only its search backend's entry
        (the tenant's schema is read from the schema registry, not through a
        cached backend), so the evictor's backend displaces the holder's.
        """
        from cogniverse_sdk.interfaces.backend import BackendClosedError
        from cogniverse_vespa.backend import VespaBackend

        env = two_clusters
        alpha_port = env["clusters"]["alpha"]["instance"]["http_port"]
        bravo_port = env["clusters"]["bravo"]["instance"]["http_port"]
        alpha_endpoint = f"http://localhost:{alpha_port}"
        registry = BackendRegistry.get_instance()
        capacity_before = BackendRegistry._backend_instances.capacity
        registry.clear_instances()
        configure_tenant_cache_capacity(1)

        closed_endpoints: list[str] = []
        original_close = VespaBackend.close

        def counting_close(backend):
            closed_endpoints.append(f"{backend._url}:{backend._port}")
            original_close(backend)

        holder_holds = threading.Barrier(2)
        eviction_done = threading.Event()
        outcome: dict = {}

        def holder():
            backend = _search_backend(env, "alpha")
            outcome["served_before_eviction"] = _hit_ids(env, backend)
            outcome["keys_while_warm"] = sorted(
                BackendRegistry._backend_instances.keys()
            )
            holder_holds.wait(timeout=120)
            eviction_done.wait(timeout=180)
            try:
                outcome["returned"] = _hit_ids(env, backend)
            except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                outcome["error"] = exc

        def evictor():
            holder_holds.wait(timeout=120)
            outcome["evictor_served"] = _hit_ids(env, _search_backend(env, "bravo"))
            outcome["keys_after_eviction"] = sorted(
                BackendRegistry._backend_instances.keys()
            )
            eviction_done.set()

        VespaBackend.close = counting_close
        try:
            threads = [
                threading.Thread(target=holder),
                threading.Thread(target=evictor),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=300)
        finally:
            VespaBackend.close = original_close
            registry.clear_instances()
            configure_tenant_cache_capacity(capacity_before)

        assert [thread.is_alive() for thread in threads] == [False, False]
        assert outcome["served_before_eviction"] == [env["clusters"]["alpha"]["doc_id"]]
        assert outcome["evictor_served"] == [env["clusters"]["bravo"]["doc_id"]]
        assert outcome["keys_while_warm"] == [f"search_vespa@{alpha_endpoint}"]
        assert outcome["keys_after_eviction"] == [
            f"search_vespa@http://localhost:{bravo_port}"
        ]
        assert closed_endpoints == [alpha_endpoint]
        assert set(outcome) == {
            "served_before_eviction",
            "keys_while_warm",
            "evictor_served",
            "keys_after_eviction",
            "error",
        }
        assert type(outcome["error"]) is BackendClosedError
        assert str(outcome["error"]) == (
            f"VespaBackend for {alpha_endpoint} is closed; its clients were "
            f"released. Obtain a fresh instance from the backend registry."
        )


class _PausingSchemaLoader(FilesystemSchemaLoader):
    """The real loader, with a rendezvous on one thread's first load.

    ``VespaSearchBackend.get_search_results`` loads the profile's schema
    before it resolves the tenant's deployed schema, so blocking here
    parks a real search mid-flight — the backend checked out, the query
    not yet sent — which is the window the other tenants' traffic has to
    survive.
    """

    def __init__(self, base_path, thread_name, reached, resume):
        super().__init__(base_path)
        self._thread_name = thread_name
        self._reached = reached
        self._resume = resume
        self._paused_once = False

    def load_schema(self, schema_name):
        loaded = super().load_schema(schema_name)
        if threading.current_thread().name == self._thread_name and (
            not self._paused_once
        ):
            self._paused_once = True
            self._reached.wait(timeout=120)
            self._resume.wait(timeout=240)
        return loaded


@pytest.mark.integration
class TestCheckedOutBackendOutlivesOtherTenants:
    def test_a_search_in_flight_survives_a_burst_of_other_tenants(self, two_clusters):
        """A query holds the backend it runs on against eviction.

        Other tenants' traffic inserts into the bounded cache the search
        backend lives in; enough of it during one query would evict and close
        the instance running it, and the query would die on a released
        connection pool. The instance serving a query is checked out:
        eviction takes the least-recently-used free entry instead.
        """
        from cogniverse_vespa.backend import VespaBackend

        env = two_clusters
        alpha = env["clusters"]["alpha"]
        alpha_endpoint = f"http://localhost:{alpha['instance']['http_port']}"
        tenant_id = env["tenant_id"]
        search_key = f"search_vespa@{alpha_endpoint}"
        registry = BackendRegistry.get_instance()
        capacity_before = BackendRegistry._backend_instances.capacity
        registry.clear_instances()
        configure_tenant_cache_capacity(2)

        holder_name = "lease-holder"
        reached_search = threading.Barrier(2)
        may_finish = threading.Event()
        loader = _PausingSchemaLoader(
            Path("configs/schemas"), holder_name, reached_search, may_finish
        )
        burst_tenants = [f"{tenant_id}_burst{index}" for index in (1, 2, 3)]

        closed: list[str] = []
        closed_lock = threading.Lock()
        original_close = VespaBackend.close

        def counting_close(backend):
            with closed_lock:
                closed.append(f"{backend._tenant_id}@{backend._url}:{backend._port}")
            original_close(backend)

        outcome: dict = {}

        def holder():
            backend = registry.get_search_backend(
                name="vespa",
                config=_backend_config(alpha["instance"]),
                config_manager=alpha["config_manager"],
                schema_loader=loader,
            )
            try:
                outcome["returned"] = _hit_ids(env, backend)
            except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                outcome["error"] = exc

        VespaBackend.close = counting_close
        try:
            thread = threading.Thread(target=holder, name=holder_name)
            thread.start()
            reached_search.wait(timeout=120)

            checkouts_mid_search = BackendRegistry._backend_instances.lease_count(
                search_key
            )
            keys_mid_search = sorted(BackendRegistry._backend_instances.keys())

            for burst_tenant in burst_tenants:
                registry.get_ingestion_backend(
                    name="vespa",
                    tenant_id=burst_tenant,
                    config=_backend_config(alpha["instance"]),
                    config_manager=alpha["config_manager"],
                    schema_loader=loader,
                )
            with closed_lock:
                closed_during_burst = list(closed)
            keys_after_burst = sorted(BackendRegistry._backend_instances.keys())

            may_finish.set()
            thread.join(timeout=300)
        finally:
            VespaBackend.close = original_close
            may_finish.set()
            registry.clear_instances()
            configure_tenant_cache_capacity(capacity_before)

        assert thread.is_alive() is False
        assert set(outcome) == {"returned"}
        assert outcome["returned"] == [alpha["doc_id"]]
        assert checkouts_mid_search == 1
        assert keys_mid_search == [search_key]
        assert closed_during_burst == [
            f"{burst_tenants[0]}@{alpha_endpoint}",
            f"{burst_tenants[1]}@{alpha_endpoint}",
        ]
        assert keys_after_burst == [
            f"backend_vespa_{burst_tenants[2]}@{alpha_endpoint}",
            search_key,
        ]

    def test_a_failed_search_gives_its_checkout_back(self, two_clusters):
        """A search that raises releases the backend it checked out.

        A checkout leaked on the error path pins the instance forever: it
        is never evicted, the cache stays one entry over capacity for the
        life of the process, and its connection pool is never released.
        """
        from cogniverse_vespa.backend import VespaBackend

        env = two_clusters
        alpha = env["clusters"]["alpha"]
        alpha_endpoint = f"http://localhost:{alpha['instance']['http_port']}"
        dead_endpoint = f"http://127.0.0.1:{DEAD_VESPA_PORT}"
        dead_key = f"search_vespa@{dead_endpoint}"
        registry = BackendRegistry.get_instance()
        capacity_before = BackendRegistry._backend_instances.capacity
        registry.clear_instances()
        configure_tenant_cache_capacity(1)

        closed: list[str] = []
        original_close = VespaBackend.close

        def counting_close(backend):
            closed.append(f"{backend._tenant_id}@{backend._url}:{backend._port}")
            original_close(backend)

        VespaBackend.close = counting_close
        try:
            dead = registry.get_search_backend(
                name="vespa",
                config={
                    "backend": {
                        "url": "http://127.0.0.1",
                        "config_port": DEAD_VESPA_PORT,
                        "port": DEAD_VESPA_PORT,
                    }
                },
                config_manager=alpha["config_manager"],
                schema_loader=env["schema_loader"],
            )
            with pytest.raises(Exception) as excinfo:
                _hit_ids(env, dead)

            checkouts_after_failure = BackendRegistry._backend_instances.lease_count(
                dead_key
            )
            keys_after_failure = sorted(BackendRegistry._backend_instances.keys())
            closed_after_failure = list(closed)

            served = _hit_ids(env, _search_backend(env, "alpha"))
            keys_after_pressure = sorted(BackendRegistry._backend_instances.keys())
            closed_after_pressure = list(closed)
        finally:
            VespaBackend.close = original_close
            registry.clear_instances()
            configure_tenant_cache_capacity(capacity_before)

        assert str(DEAD_VESPA_PORT) in str(excinfo.value)
        assert checkouts_after_failure == 0
        assert closed_after_failure == []
        assert keys_after_failure == [dead_key]
        assert served == [alpha["doc_id"]]
        # Alpha's backend could only displace the dead one if the failed
        # search released its checkout.
        assert closed_after_pressure == [f"{SYSTEM_TENANT_ID}@{dead_endpoint}"]
        assert keys_after_pressure == [f"search_vespa@{alpha_endpoint}"]
