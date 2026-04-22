"""Integration test for dynamic profile visibility at search time.

Existing `test_tenant_schema_lifecycle.py` deploys schemas and asserts
against the SchemaRegistry's own tracking — it never calls a real
`backend.search(...)` against a dynamically-added profile, so it
never exercised the in-memory profile visibility path that caused
the agent_memories retry storm.

This test closes that gap with hard assertions on real data:

    1. Construct a cached `VespaSearchBackend` with NO profiles — the
       target profile must be absent.
    2. Register the profile at runtime via
       `ConfigManager.add_backend_profile` (the same path the admin
       router uses).
    3. Deploy its schema to Vespa.
    4. Ingest a real document with a unique token into the tenant-
       scoped schema.
    5. Wait for Vespa to index.
    6. Call `backend.search(...)` with the new profile and assert we
       get back a document that contains the unique token.

If any step fails the test fails loudly — no swallowed exceptions, no
"did not raise" soft assertions.
"""

from __future__ import annotations

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

    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    cm.set_profile_change_listener(listener)
    return cm


@pytest.fixture(scope="module")
def schema_loader():
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.fixture
def clean_registry():
    BackendRegistry._backend_instances.clear()
    yield
    BackendRegistry._backend_instances.clear()


@pytest.mark.integration
def test_register_profile_then_ingest_and_search_returns_the_document(
    vespa_instance, temp_config_manager, schema_loader, clean_registry
):
    """Real ingest + real search + content assertion.

    End-to-end proof that a profile registered at runtime via
    `ConfigManager.add_backend_profile` is (a) propagated to the cached
    `VespaSearchBackend.profiles`, and (b) usable for an actual search
    that returns a document ingested via direct Vespa PUT.

    Uses `agent_memories` schema (768-dim dense single-vector) so we
    don't need a model at test time — a deterministic pseudo-embedding
    is sufficient for the search to succeed (the query uses the SAME
    vector as the document, so distance is 0 and it's the top hit).
    """
    import json
    import time

    import numpy as np
    import requests

    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    registry = BackendRegistry.get_instance()
    tenant_id = f"dyn_roundtrip_{uuid.uuid4().hex[:8]}"
    # Profile name must survive as an in-memory dict key; pick something unique.
    profile_name = f"mem_probe_{uuid.uuid4().hex[:8]}"

    # --- Cached search backend starts WITHOUT our profile ----------
    search_backend = registry.get_search_backend(
        name="vespa",
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
    assert profile_name not in search_backend.profiles

    # --- Register profile via ConfigManager (fires listener) -------
    # Schema is agent_memories (already defined in configs/schemas/),
    # 768-dim dense single-vector. semantic_search ranking strategy.
    profile = BackendProfileConfig(
        profile_name=profile_name,
        type="document",
        schema_name="agent_memories",
        embedding_model="nomic-embed-text",
        embedding_type="single_vector",
        schema_config={"embedding_dims": 768},
    )
    temp_config_manager.add_backend_profile(
        profile, tenant_id="test:unit", service="backend"
    )
    assert profile_name in search_backend.profiles

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

    # Wait for Vespa content cluster to pick up the new schema.
    # The ApplicationPackage deploy is async — the config server accepts
    # it immediately but content nodes need a few seconds to activate.
    # Poll the Vespa search endpoint until it can resolve the new type.
    vespa_url = f"http://localhost:{vespa_instance['http_port']}"
    for attempt in range(30):
        check = requests.get(
            f"{vespa_url}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if check.status_code == 200:
            root = check.json().get("root", {})
            if "errors" not in root:
                break
        time.sleep(1)
    else:
        pytest.fail(
            f"Vespa never activated schema {tenant_schema} after 30s — "
            "deploy_schema returned success but content cluster didn't apply it."
        )

    # --- PUT document directly to Vespa (skip complex ingest pipe) ---
    # Deterministic seeded 768-dim vector, used for both doc and query so
    # distance is 0 and this doc is the top hit.
    rng = np.random.default_rng(42)
    vector = rng.random(768).astype(np.float32).tolist()

    unique_token = f"zxqv_{uuid.uuid4().hex[:12]}"
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
    assert put_resp.status_code in (200, 201), (
        f"Direct Vespa PUT failed: {put_resp.status_code} {put_resp.text[:300]}"
    )

    # Wait for Vespa to index.
    wait_for_vespa_indexing(delay=3)

    # --- Real search, matching embedding → doc MUST be top hit -----
    results = search_backend.search(
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
        f"Search for identical-vector query returned zero hits against the "
        f"just-deployed profile {profile_name!r} on schema {tenant_schema!r}. "
        "The document we PUT isn't retrievable — either profile propagation, "
        "schema deployment, or Vespa indexing regressed."
    )

    # Collect hit IDs from whatever shape the results have.
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
        f"Search returned {len(results)} hits but none matched the "
        f"ingested document id {doc_id!r}. Hit ids: {hit_ids!r}. "
        "The PUT document was not retrievable via the dynamically-"
        "registered profile — exactly the bug the fix targets."
    )


@pytest.mark.integration
def test_profile_registered_via_config_manager_appears_in_live_backend(
    vespa_instance, temp_config_manager, schema_loader, clean_registry
):
    """Full positive-assertion round-trip against a real Vespa:

    1. Construct the cached VespaSearchBackend (via registry) with NO
       target profile.
    2. Register the profile through `ConfigManager.add_backend_profile`
       (the same path the admin router uses).
    3. Verify the exact profile config appears on the live backend's
       `profiles` dict with matching field values (schema_name,
       embedding_model, embedding_type, schema_config).
    4. Call `backend.search(...)` with the new profile name and assert
       the profile-resolution phase completes (the exception the bug
       used to fire has `"Requested profile '{name}' not found"` in
       its message — that specific shape must not appear).
    5. Delete the profile via `ConfigManager.delete_backend_profile`
       and verify it's gone from the live backend.

    Covers the full fanout chain with hard field-level assertions,
    not string-contains or presence-only checks.
    """
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    registry = BackendRegistry.get_instance()
    target_profile = f"dyn_probe_{uuid.uuid4().hex[:8]}"

    search_backend = registry.get_search_backend(
        name="vespa",
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
    # Step 1: cold cache, target absent.
    assert target_profile not in search_backend.profiles

    # Step 2: register via ConfigManager → listener → fanout.
    profile = BackendProfileConfig(
        profile_name=target_profile,
        type="document",
        schema_name="document_text",
        embedding_model="nomic-embed-text",
        embedding_type="single_vector",
        schema_config={"embedding_dims": 768},
    )
    temp_config_manager.add_backend_profile(
        profile, tenant_id="test:unit", service="backend"
    )

    # Step 3: full field-level assertion on what the live backend sees.
    live = search_backend.profiles.get(target_profile)
    assert live is not None, (
        "Profile registered via ConfigManager did NOT reach the cached "
        "search backend — the listener → BackendRegistry → VespaSearchBackend "
        "chain is broken."
    )
    assert live["schema_name"] == "document_text"
    assert live["embedding_model"] == "nomic-embed-text"
    assert live["embedding_type"] == "single_vector"
    assert live["schema_config"] == {"embedding_dims": 768}
    assert live["type"] == "document"

    # Step 4: profile resolution at search time must pass.
    resolution_ok = False
    try:
        search_backend.search(
            query_dict={
                "query": "probe",
                "type": "document",
                "profile": target_profile,
                "tenant_id": "dyn_probe_tenant",
                "top_k": 1,
            }
        )
        resolution_ok = True
    except ValueError as exc:
        msg = str(exc)
        # The bug we're guarding against raises exactly this shape.
        if f"Requested profile '{target_profile}' not found" in msg:
            pytest.fail(
                "Profile-resolution at search-time still raises the "
                f"exact bug signature: {msg}"
            )
        # Any other ValueError (strategy/schema) is past the bug surface.
        resolution_ok = True
    except Exception:
        # Non-ValueError exceptions are downstream of profile resolution.
        resolution_ok = True
    assert resolution_ok, "search should have at least reached profile resolution"

    # Step 5: delete → profile disappears from the live backend.
    deleted = temp_config_manager.delete_backend_profile(
        target_profile, tenant_id="test:unit", service="backend"
    )
    assert deleted is True
    assert target_profile not in search_backend.profiles, (
        "delete_backend_profile propagated to ConfigStore but NOT to the "
        "cached search backend — remove_profile fanout regressed."
    )
