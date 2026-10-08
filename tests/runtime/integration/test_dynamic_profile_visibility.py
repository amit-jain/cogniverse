"""Runtime: POST /admin/profiles → backend.search() returns ingested doc.

Real integration, real content assertion. A profile posted through the HTTP
admin API is immediately searchable by its tenant on the shared search
backend, which resolves the searching tenant's profiles from the config store
per request; another tenant's search neither resolves nor lists it, and once
DELETE /admin/profiles removes it the tenant's search refuses it.

We bypass the /search/ HTTP endpoint because it routes through
`QueryEncoderFactory` which only knows ColPali/ColQwen/ColBERT/
X-CLIP encoders — not generic text embedders like
DenseOn. The backend search layer accepts pre-computed
query_embeddings directly, so we call it that way and assert the
ingested document is returned by id.
"""

from __future__ import annotations

import ast
import json
import time
import uuid

import numpy as np
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.routers import admin, search
from tests.utils.async_polling import wait_for_vespa_indexing


@pytest.fixture
def clean_backend_registry():
    BackendRegistry._backend_instances.clear()
    yield
    BackendRegistry._backend_instances.clear()


@pytest.fixture
def wired_app(
    vespa_instance,
    config_manager,
    schema_loader,
    real_telemetry,
    clean_backend_registry,
    profile_change_events,
):
    """FastAPI app with the admin and search routers wired as runtime startup
    wires them."""
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.include_router(search.router, prefix="/search")

    app.dependency_overrides[search.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = lambda: (
        schema_loader
    )

    with TestClient(app) as client:
        yield client


def _wait_for_vespa_schema(vespa_url: str, schema: str, timeout: float = 120.0) -> None:
    """Poll until ``schema`` is queryable.

    The 30s default that was here before regularly missed the deadline on
    sessions where the test container had served many earlier writes; the
    content cluster's prepareandactivate cycle for a freshly-pushed schema
    routinely runs 45-60s under load. 120s gives the round-trip room
    without making the green path noticeably slower (it returns as soon
    as the first query succeeds).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = requests.get(
            f"{vespa_url}/search/",
            params={"yql": f"select * from {schema} where true limit 0"},
            timeout=5,
        )
        if resp.status_code == 200 and "errors" not in resp.json().get("root", {}):
            return
        time.sleep(1)
    raise AssertionError(f"Vespa never activated schema {schema} after {timeout}s")


def _refused_listing(live_backend, query_dict: dict) -> list:
    """The profiles a "not found" refusal lists for this search."""
    with pytest.raises(ValueError) as refused:
        live_backend.search(query_dict=query_dict)
    prefix = (
        f"Requested profile '{query_dict['profile']}' not found. Available profiles: "
    )
    message = str(refused.value)
    assert message.startswith(prefix), message
    return ast.literal_eval(message[len(prefix) :])


@pytest.mark.integration
def test_admin_profile_post_makes_backend_search_return_ingested_doc(
    wired_app, vespa_instance, config_manager, schema_loader
):
    """POST /admin/profiles → PUT doc → backend.search() → assert doc id.

    Full round-trip with positive content assertion against the real
    VespaSearchBackend accessed after HTTP profile registration.
    """
    tenant_id = f"http_be_{uuid.uuid4().hex[:8]}"
    other_tenant = f"http_be_other_{uuid.uuid4().hex[:8]}"
    profile_name = f"http_be_probe_{uuid.uuid4().hex[:8]}"

    # --- Cold cached backend has no target profile ---------------------
    registry = BackendRegistry.get_instance()
    live_backend = registry.get_search_backend(
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
    rng = np.random.default_rng(42)
    vector = rng.random(768).astype(np.float32).tolist()
    unique_token = f"zxqv_{uuid.uuid4().hex[:12]}"

    def query(tenant: str) -> dict:
        return {
            "query": unique_token,
            "type": "document",
            "profile": profile_name,
            "strategy": "semantic_search",
            "tenant_id": tenant,
            "top_k": 5,
            "query_embeddings": np.asarray(vector, dtype=np.float32),
        }

    # --- Not registered yet: the tenant's search refuses it ---------
    assert profile_name not in _refused_listing(live_backend, query(tenant_id))
    # The search built the backend's own profiles; nothing below changes them.
    built_with = dict(live_backend.profiles)

    # --- POST /admin/profiles with schema deploy --------------------
    create_resp = wired_app.post(
        "/admin/profiles",
        json={
            "profile_name": profile_name,
            "tenant_id": tenant_id,
            "type": "document",
            "schema_name": "agent_memories",
            "embedding_model": "lightonai/DenseOn",
            "pipeline_config": {},
            "strategies": {},
            "embedding_type": "single_vector",
            "model_loader": "xclip",
            "schema_config": {"embedding_dims": 768},
            "deploy_schema": True,
        },
    )
    assert create_resp.status_code == 201, (
        f"admin /profiles failed: {create_resp.status_code} {create_resp.text}"
    )
    created = create_resp.json()
    assert created["schema_deployed"] is True
    tenant_schema = created["tenant_schema_name"]
    assert tenant_schema == f"agent_memories_{tenant_id}_{tenant_id}"

    # --- The tenant's search resolves it from the store, per request --
    resolved = (
        get_config(tenant_id=tenant_id, config_manager=config_manager)
        .get("backend")
        .get("profiles")[profile_name]
    )
    assert resolved["schema_name"] == "agent_memories"
    assert resolved["embedding_type"] == "single_vector"
    assert live_backend.profiles == built_with

    # --- Wait for Vespa content cluster to apply schema -------------
    vespa_url = f"http://localhost:{vespa_instance['http_port']}"
    _wait_for_vespa_schema(vespa_url, tenant_schema)

    # --- PUT a real document with deterministic 768-dim vector ------
    doc_id = f"http_be_probe_{unique_token}"

    put_resp = requests.post(
        f"{vespa_url}/document/v1/content/{tenant_schema}/docid/{doc_id}",
        json={
            "fields": {
                "id": doc_id,
                "text": f"document containing the unique token {unique_token}",
                "embedding": vector,
                "user_id": "http_be_user",
                "agent_id": "http_be_agent",
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

    # --- backend.search() with identical vector → the doc is the only hit
    results = live_backend.search(query_dict=query(tenant_id))
    assert [result.document.id for result in results] == [doc_id], (
        f"backend.search for profile {profile_name!r} on schema "
        f"{tenant_schema!r} did not return exactly the ingested document."
    )

    # --- Another tenant neither resolves nor sees it -----------------
    visible_to_other = list(
        {
            **built_with,
            **get_config(tenant_id=other_tenant, config_manager=config_manager)
            .get("backend")
            .get("profiles"),
        }
    )
    assert profile_name not in visible_to_other
    assert _refused_listing(live_backend, query(other_tenant)) == visible_to_other

    # --- DELETE /admin/profiles/<name>: the tenant's search refuses it --
    del_resp = wired_app.delete(
        f"/admin/profiles/{profile_name}", params={"tenant_id": tenant_id}
    )
    assert del_resp.status_code == 200, del_resp.text
    assert profile_name not in _refused_listing(live_backend, query(tenant_id))
    assert live_backend.profiles == built_with
