"""Admin signature-variant route logic (canonicalization, merge, validation).

GET/PUT ``/admin/tenants/{t}/signature_variants`` route serialization and
validation, driven in-process with the in-memory config store so the route
logic runs without Docker. The real Vespa persistence round-trip +
cross-process dispatcher resolution live in
tests/runtime/integration/test_signature_variant_persistence.py. Both the write
and the read canonicalize the tenant id, so a selection stored for one spelling
resolves for the canonical form.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def store():
    previous = admin._config_manager
    store = InMemoryConfigStore()
    admin.set_config_manager(ConfigManager(store=store))
    yield store
    admin.set_config_manager(previous)


@pytest.fixture
def client(store) -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    return TestClient(app)


class TestSignatureVariantEndpoints:
    def test_get_empty_for_new_tenant(self, client: TestClient):
        resp = client.get("/admin/tenants/acme/signature_variants")
        assert resp.status_code == 200
        assert resp.json()["selections"] == {}

    def test_put_one_agent_persists_for_get(self, client: TestClient):
        resp = client.put(
            "/admin/tenants/acme/signature_variants/search_agent",
            json={"variant_id": "with_jurisdiction"},
        )
        assert resp.status_code == 200
        assert resp.json()["selections"] == {"search_agent": "with_jurisdiction"}
        # GET reflects.
        again = client.get("/admin/tenants/acme/signature_variants").json()
        assert again["selections"] == {"search_agent": "with_jurisdiction"}

    def test_put_multiple_agents_keeps_each(self, client: TestClient):
        client.put(
            "/admin/tenants/acme/signature_variants/search_agent",
            json={"variant_id": "with_jurisdiction"},
        )
        client.put(
            "/admin/tenants/acme/signature_variants/summarizer_agent",
            json={"variant_id": "concise"},
        )
        body = client.get("/admin/tenants/acme/signature_variants").json()
        assert body["selections"] == {
            "search_agent": "with_jurisdiction",
            "summarizer_agent": "concise",
        }

    def test_empty_variant_id_rejected(self, client: TestClient):
        resp = client.put(
            "/admin/tenants/acme/signature_variants/search_agent",
            json={"variant_id": ""},
        )
        assert resp.status_code == 400

    def test_selection_resolves_across_id_spellings(self, client: TestClient):
        # Stored under the canonical id; a bare-id GET resolves the same blob.
        client.put(
            "/admin/tenants/acme/signature_variants/search_agent",
            json={"variant_id": "with_jurisdiction"},
        )
        raw = client.get("/admin/tenants/acme/signature_variants").json()
        canonical = client.get("/admin/tenants/acme:acme/signature_variants").json()
        assert raw["selections"] == {"search_agent": "with_jurisdiction"}
        assert canonical["selections"] == {"search_agent": "with_jurisdiction"}

    def test_put_stores_the_record_under_the_canonical_tenant(
        self, client: TestClient, store
    ):
        resp = client.put(
            "/admin/tenants/acme/signature_variants/search_agent",
            json={"variant_id": "with_jurisdiction"},
        )
        again = client.put(
            "/admin/tenants/acme/signature_variants/search_agent",
            json={"variant_id": "with_jurisdiction"},
        )

        assert resp.json() == {
            "tenant_id": "acme",
            "selections": {"search_agent": "with_jurisdiction"},
        }
        assert again.json() == resp.json()
        history = store.get_config_history(
            "acme:acme", ConfigScope.SYSTEM, "admin_overrides", "signature_variants"
        )
        # A PUT that changes nothing writes no new version.
        assert [(e.version, e.config_value) for e in history] == [
            (1, {"search_agent": "with_jurisdiction"})
        ]
        assert (
            store.get_config(
                "acme", ConfigScope.SYSTEM, "admin_overrides", "signature_variants"
            )
            is None
        )
