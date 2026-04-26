"""Runtime: POST /admin/profiles → backend.search() returns ingested doc.

Real integration, real content assertion. The point of this test is
to prove the runtime-startup wiring (ConfigManager +
profile_change_listener set up as `main.py` does) makes a profile
posted through the HTTP admin API immediately available to the
backend search layer.

We bypass the /search/ HTTP endpoint because it routes through
`QueryEncoderFactory` which only knows ColPali/ColQwen/ColBERT/
VideoPrism encoders — not generic text embedders like
DenseOn. The backend search layer accepts pre-computed
query_embeddings directly, so we call it that way and assert the
ingested document is returned by id.

Flow:
  1. Cached VespaSearchBackend starts WITHOUT our target profile.
  2. POST /admin/profiles (deploy_schema=True) — triggers the
     profile_change_listener fanout and a Vespa schema deploy.
  3. Wait for Vespa content cluster to activate the new schema.
  4. PUT a real document with a deterministic 768-dim vector to the
     tenant-scoped Vespa schema.
  5. backend.search(query_dict=..., query_embeddings=same_vector) —
     asserts the ingested document id appears in the results.
  6. DELETE /admin/profiles/<name> and verify removal from the
     cached backend.
"""

from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.registries.backend_registry import BackendRegistry
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
):
    """FastAPI app with admin router and profile_change_listener wired,
    mirroring runtime startup."""

    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    config_manager.set_profile_change_listener(listener)

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

    config_manager.set_profile_change_listener(None)


def _wait_for_vespa_schema(vespa_url: str, schema: str, timeout: float = 30.0) -> None:
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


@pytest.mark.integration
def test_admin_profile_post_makes_backend_search_return_ingested_doc(
    wired_app, vespa_instance, config_manager, schema_loader
):
    """POST /admin/profiles → PUT doc → backend.search() → assert doc id.

    Full round-trip with positive content assertion against the real
    VespaSearchBackend accessed after HTTP profile registration.
    """
    tenant_id = f"http_be_{uuid.uuid4().hex[:8]}"
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
    assert profile_name not in live_backend.profiles

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
            "schema_config": {"embedding_dims": 768},
            "deploy_schema": True,
        },
    )
    assert create_resp.status_code in (200, 201), (
        f"admin /profiles failed: {create_resp.status_code} {create_resp.text}"
    )
    created = create_resp.json()
    assert created["schema_deployed"] is True
    tenant_schema = created["tenant_schema_name"]
    assert tenant_schema

    # --- Listener fanout landed on cached backend -------------------
    live = live_backend.profiles.get(profile_name)
    assert live is not None, (
        "POST /admin/profiles didn't propagate to cached search backend"
    )
    assert live["schema_name"] == "agent_memories"
    assert live["embedding_type"] == "single_vector"

    # --- Wait for Vespa content cluster to apply schema -------------
    vespa_url = f"http://localhost:{vespa_instance['http_port']}"
    _wait_for_vespa_schema(vespa_url, tenant_schema)

    # --- PUT a real document with deterministic 768-dim vector ------
    rng = np.random.default_rng(42)
    vector = rng.random(768).astype(np.float32).tolist()
    unique_token = f"zxqv_{uuid.uuid4().hex[:12]}"
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
    assert put_resp.status_code in (200, 201), (
        f"Direct Vespa PUT failed: {put_resp.status_code} {put_resp.text[:300]}"
    )

    wait_for_vespa_indexing(delay=3)

    # --- backend.search() with identical vector → doc is top hit ----
    results = live_backend.search(
        query_dict={
            "query": unique_token,
            "type": "document",
            "profile": profile_name,
            "strategy": "semantic_search",
            "tenant_id": tenant_id,
            "top_k": 5,
            "query_embeddings": np.asarray(vector, dtype=np.float32),
        }
    )
    assert isinstance(results, list)
    assert len(results) > 0, (
        f"backend.search returned zero hits for profile {profile_name!r} "
        f"on schema {tenant_schema!r}. Doc {doc_id!r} should be top hit."
    )

    hit_ids: list[str] = []
    for r in results:
        rid = getattr(r, "id", None) or getattr(r, "document_id", None)
        if rid is None and hasattr(r, "document") and r.document is not None:
            rid = getattr(r.document, "id", None)
        if rid is None and isinstance(r, dict):
            rid = (
                r.get("id")
                or r.get("document_id")
                or ((r.get("document") or {}).get("id"))
            )
        if rid:
            hit_ids.append(str(rid))

    assert doc_id in hit_ids, (
        f"backend.search returned {len(results)} hits but none matched "
        f"the ingested document id {doc_id!r}. Hit ids: {hit_ids!r}. "
        "Document ingested after HTTP profile registration was not "
        "retrievable via backend search — visibility gap regressed."
    )

    # --- DELETE /admin/profiles/<name> removes from cached backend --
    del_resp = wired_app.delete(
        f"/admin/profiles/{profile_name}", params={"tenant_id": tenant_id}
    )
    assert del_resp.status_code in (200, 204)
    assert profile_name not in live_backend.profiles
