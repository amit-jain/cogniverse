"""Two Vespa clusters, two backends: each answers only from its own.

The registry hands out process-shared search backends. Keyed by name
alone, the second caller's config was discarded and its queries went to
the first caller's cluster — a cross-cluster read that returns real,
wrong documents rather than an error.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.registries.backend_registry import BackendRegistry
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
