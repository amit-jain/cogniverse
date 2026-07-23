"""Admin signature-variant route logic (canonicalization, merge, validation).

GET/PUT ``/admin/tenants/{t}/signature_variants`` route serialization and
validation, driven in-process with an in-memory artifact-store double so the
route logic runs without Docker. The real Phoenix persistence round-trip +
cold-replica dispatcher resolution live in
tests/runtime/integration/test_signature_variant_persistence.py. Both the write
and the read canonicalize the tenant id, so a selection stored for one spelling
resolves for the canonical form.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _InMemoryArtifactManager:
    """Per-tenant in-memory blob store double for the route-logic unit tests."""

    def __init__(self, blobs: dict, tenant: str):
        self._blobs = blobs
        self._tenant = tenant

    async def save_blob(self, kind: str, key: str, raw: str) -> None:
        self._blobs[(self._tenant, kind, key)] = raw

    async def load_blob(self, kind: str, key: str):
        return self._blobs.get((self._tenant, kind, key))


@pytest.fixture
def client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    blobs: dict = {}
    monkeypatch.setattr(
        admin,
        "_build_artifact_manager",
        lambda key: _InMemoryArtifactManager(blobs, key),
    )
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
