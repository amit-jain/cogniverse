"""Admin signature-variant endpoints (in-memory override store, no persistence).

GET/PUT ``/admin/tenants/{t}/signature_variants`` round-trip through the
in-memory ``_signature_variant_overrides`` dict — no Vespa or Phoenix — so they
belong in the fast gate rather than behind the integration suite's Docker
fixtures. Both the write and the read canonicalize the tenant id, so a variant
selection stored for one spelling resolves for the canonical form.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin._reset_admin_overrides_for_tests()
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
