"""Profiles added at runtime resolve per tenant on the cached search backend.

The shared ``VespaSearchBackend`` holds the profiles it was built with. A
profile registered later through ``ConfigManager.add_backend_profile`` (the
path the admin router uses) reaches a search only through the searching
tenant's config, read per request:

    1. The cached backend refuses the profile before it is registered.
    2. Once registered for a tenant, that tenant's search resolves it and
       returns a real document fed into the tenant-scoped schema.
    3. Another tenant's search refuses it and does not list it.
    4. Once deleted, the tenant's search refuses it again.

The backend's own profiles never change along the way.
"""

from __future__ import annotations

import ast
import logging
import uuid
from pathlib import Path

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from tests.utils.async_polling import wait_for_vespa_indexing

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def temp_config_manager(vespa_instance):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
    )
    return cm


@pytest.fixture(scope="module")
def schema_loader():
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.fixture
def clean_registry():
    BackendRegistry._backend_instances.clear()
    yield
    BackendRegistry._backend_instances.clear()


def _search_backend(registry, vespa_instance, config_manager, schema_loader):
    return registry.get_search_backend(
        name="vespa",
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


def _refused_listing(search_backend, query_dict: dict) -> list:
    """The profiles a "not found" refusal lists for this search."""
    with pytest.raises(ValueError) as refused:
        search_backend.search(query_dict=query_dict)
    prefix = (
        f"Requested profile '{query_dict['profile']}' not found. Available profiles: "
    )
    message = str(refused.value)
    assert message.startswith(prefix), message
    return ast.literal_eval(message[len(prefix) :])


def _hit_ids(results) -> list[str]:
    return [result.document.id for result in results]


def _searched_profiles(config_manager, tenant_id: str) -> dict:
    """The tenant's profiles as every search reads them per request."""
    from cogniverse_foundation.config.utils import get_config

    return get_config(tenant_id=tenant_id, config_manager=config_manager).get(
        "backend"
    )["profiles"]


@pytest.mark.integration
def test_a_profile_registered_for_a_tenant_at_runtime_is_searchable_by_it_alone(
    vespa_instance, temp_config_manager, schema_loader, clean_registry
):
    """Real ingest + real search + content assertion.

    Uses the `agent_memories` schema (768-dim dense single-vector) so no
    model is needed at test time — the query uses the SAME vector as the
    document, so distance is 0 and it is the top hit.
    """
    import json
    import time

    import numpy as np
    import requests

    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    registry = BackendRegistry.get_instance()
    tenant_id = f"dyn_roundtrip_{uuid.uuid4().hex[:8]}"
    other_tenant = f"dyn_other_{uuid.uuid4().hex[:8]}"
    profile_name = f"mem_probe_{uuid.uuid4().hex[:8]}"

    search_backend = _search_backend(
        registry, vespa_instance, temp_config_manager, schema_loader
    )

    rng = np.random.default_rng(42)
    vector = rng.random(768).astype(np.float32).tolist()
    unique_token = f"zxqv_{uuid.uuid4().hex[:12]}"

    def query(tenant: str, top_k: int = 5) -> dict:
        return {
            "query": unique_token,
            "type": "document",
            "profile": profile_name,
            "strategy": "semantic_search",
            "tenant_id": tenant,
            "top_k": top_k,
            "query_embeddings": np.asarray(vector, dtype=np.float32),
        }

    # --- Not registered yet: the tenant's search refuses it ---------
    assert profile_name not in _refused_listing(search_backend, query(tenant_id))
    # The search built the backend's own profiles; nothing below changes them.
    built_with = dict(search_backend.profiles)

    # --- Register the profile for the searching tenant -------------
    profile = BackendProfileConfig(
        profile_name=profile_name,
        type="document",
        schema_name="agent_memories",
        embedding_model="lightonai/DenseOn",
        embedding_type="single_vector",
        schema_config={"embedding_dims": 768},
    )
    temp_config_manager.add_backend_profile(
        profile, tenant_id=tenant_id, service="backend"
    )

    # --- Deploy agent_memories schema for our tenant via registry ---
    ingestion_backend = registry.get_ingestion_backend(
        name="vespa",
        tenant_id=tenant_id,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=temp_config_manager,
        schema_loader=schema_loader,
    )
    ingestion_backend.schema_registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name="agent_memories"
    )
    tenant_schema = ingestion_backend.get_tenant_schema_name(
        tenant_id, "agent_memories"
    )

    # The ApplicationPackage deploy is async — the config server accepts it
    # immediately but content nodes need a few seconds to activate.
    vespa_url = f"http://localhost:{vespa_instance['http_port']}"
    for _ in range(30):
        check = requests.get(
            f"{vespa_url}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if check.status_code == 200 and "errors" not in check.json().get("root", {}):
            break
        time.sleep(1)
    else:
        pytest.fail(
            f"Vespa never activated schema {tenant_schema} after 30s — "
            "deploy_schema returned success but content cluster didn't apply it."
        )

    doc_id = f"dyn_probe_{unique_token}"
    put_resp = requests.post(
        f"{vespa_url}/document/v1/content/{tenant_schema}/docid/{doc_id}",
        json={
            "fields": {
                "id": doc_id,
                "text": f"document containing the unique token {unique_token}",
                "embedding": vector,
                "user_id": "test_user",
                "agent_id": "test_agent",
                "metadata_": json.dumps({"tenant_id": tenant_id}),
                "created_at": int(time.time() * 1000),
            }
        },
        timeout=10,
    )
    assert put_resp.status_code == 200, (
        f"Direct Vespa PUT failed: {put_resp.status_code} {put_resp.text[:300]}"
    )
    wait_for_vespa_indexing(delay=3)

    # --- The tenant's search resolves it: the document is the only hit
    assert _hit_ids(search_backend.search(query_dict=query(tenant_id))) == [doc_id]
    # Vespa's default query profile rejects hits > 400 unless the built query
    # raises maxHits/maxOffset per request.
    assert _hit_ids(search_backend.search(query_dict=query(tenant_id, 1000))) == [
        doc_id
    ]

    # --- Another tenant neither resolves nor sees it -----------------
    from cogniverse_foundation.config.utils import get_config

    visible_to_other = list(
        {
            **built_with,
            **get_config(tenant_id=other_tenant, config_manager=temp_config_manager)
            .get("backend")
            .get("profiles"),
        }
    )
    assert profile_name not in visible_to_other
    assert _refused_listing(search_backend, query(other_tenant)) == visible_to_other

    # --- Deleted: the tenant's search refuses it again ---------------
    assert (
        temp_config_manager.delete_backend_profile(
            profile_name, tenant_id=tenant_id, service="backend"
        )
        is True
    )
    assert profile_name not in _refused_listing(search_backend, query(tenant_id))
    assert search_backend.profiles == built_with


@pytest.mark.integration
def test_a_profile_updated_for_a_tenant_resolves_with_its_merged_fields(
    vespa_instance, temp_config_manager, schema_loader, clean_registry
):
    """What a tenant's search resolves for a runtime profile follows every
    add, update and delete of it, field for field."""
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    registry = BackendRegistry.get_instance()
    tenant_id = f"dyn_fields_{uuid.uuid4().hex[:8]}"
    target_profile = f"dyn_probe_{uuid.uuid4().hex[:8]}"

    search_backend = _search_backend(
        registry, vespa_instance, temp_config_manager, schema_loader
    )

    def probe() -> dict:
        return {
            "query": "probe",
            "type": "document",
            "profile": target_profile,
            "tenant_id": tenant_id,
            "top_k": 1,
        }

    assert target_profile not in _refused_listing(search_backend, probe())
    built_with = dict(search_backend.profiles)

    temp_config_manager.add_backend_profile(
        BackendProfileConfig(
            profile_name=target_profile,
            type="document",
            schema_name="document_text",
            embedding_model="lightonai/DenseOn",
            embedding_type="single_vector",
            schema_config={"embedding_dims": 768},
        ),
        tenant_id=tenant_id,
        service="backend",
    )

    resolved = _searched_profiles(temp_config_manager, tenant_id)[target_profile]
    assert resolved["schema_name"] == "document_text"
    assert resolved["embedding_model"] == "lightonai/DenseOn"
    assert resolved["embedding_type"] == "single_vector"
    assert resolved["schema_config"] == {"embedding_dims": 768}
    assert resolved["type"] == "document"

    temp_config_manager.update_backend_profile(
        target_profile,
        {"embedding_model": "updated/DenseOn"},
        base_tenant_id=tenant_id,
        target_tenant_id=tenant_id,
        service="backend",
    )
    assert _searched_profiles(temp_config_manager, tenant_id)[target_profile] == {
        **resolved,
        "embedding_model": "updated/DenseOn",
    }

    assert (
        temp_config_manager.delete_backend_profile(
            target_profile, tenant_id=tenant_id, service="backend"
        )
        is True
    )
    assert target_profile not in _searched_profiles(temp_config_manager, tenant_id)
    assert target_profile not in _refused_listing(search_backend, probe())
    assert search_backend.profiles == built_with
