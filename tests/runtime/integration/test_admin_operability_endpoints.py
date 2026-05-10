"""Admin endpoints for pin quota / signature variant / canary.

Without these endpoints, operators had no way to reach pin quotas,
signature variants, or canary promote/retire without writing custom
Python. The audit flagged this gap directly.

This test mounts the real `admin` router on a FastAPI TestClient and
hits each endpoint. The canary endpoints round-trip through real
Phoenix (docker-managed); the pin-quota and variant endpoints use the
in-memory override store (no persistence layer for those keys yet).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import admin

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    admin._reset_admin_overrides_for_tests()
    return TestClient(app)


@pytest.fixture
def phoenix_env(phoenix_container, monkeypatch):
    """Point the canary endpoints at the docker-managed Phoenix on 16006."""
    monkeypatch.setenv("PHOENIX_HTTP_ENDPOINT", "http://localhost:16006")
    monkeypatch.setenv("PHOENIX_GRPC_ENDPOINT", "localhost:14317")
    yield


# ----- pin-quota endpoints ----------------------------------------------


class TestPinQuotaEndpoints:
    def test_get_returns_defaults_when_unset(self, client: TestClient):
        resp = client.get("/admin/tenants/acme/pin_quotas")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "acme"
        # Defaults from PinQuotas dataclass.
        assert body["quotas"]["user"] >= 0
        assert body["quotas"]["tenant_admin"] >= 0

    def test_put_updates_one_field_keeps_others(self, client: TestClient):
        # Set a baseline.
        baseline = client.get("/admin/tenants/acme/pin_quotas").json()["quotas"]
        # Update only the user quota.
        resp = client.put(
            "/admin/tenants/acme/pin_quotas",
            json={"user": 99},
        )
        assert resp.status_code == 200
        updated = resp.json()["quotas"]
        assert updated["user"] == 99
        # Other fields preserved.
        assert updated["tenant_admin"] == baseline["tenant_admin"]
        # GET reflects the put.
        again = client.get("/admin/tenants/acme/pin_quotas").json()["quotas"]
        assert again["user"] == 99

    def test_negative_quota_rejected(self, client: TestClient):
        resp = client.put(
            "/admin/tenants/acme/pin_quotas",
            json={"user": -1},
        )
        assert resp.status_code == 400
        assert "must be >= 0" in resp.text


# ----- signature-variant endpoints ------------------------------------------


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


# ----- canary endpoints (real Phoenix) --------------------------------------


class TestCanaryEndpoints:
    def test_promote_then_retire_round_trip(self, client: TestClient, phoenix_env):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_telemetry_phoenix.provider import PhoenixProvider

        tenant_id = f"p4can_{uuid.uuid4().hex[:8]}"
        # Seed a versioned dataset so promote_to_canary has something to promote.
        provider = PhoenixProvider()
        provider.initialize(
            {
                "tenant_id": tenant_id,
                "http_endpoint": "http://localhost:16006",
                "grpc_endpoint": "localhost:14317",
            }
        )
        am = ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)
        import asyncio

        asyncio.run(
            am.save_prompts_versioned("search_agent", {"system": "CANARY_VIA_ADMIN"})
        )

        # Promote.
        promote_resp = client.post(
            f"/admin/tenants/{tenant_id}/canary/search_agent/promote",
            json={"version": 1, "traffic_pct": 25},
        )
        assert promote_resp.status_code == 200, promote_resp.text
        state = promote_resp.json()["state"]
        assert state["canary"]["version"] == 1
        assert state["canary"]["traffic_pct"] == 25

        # Retire.
        retire_resp = client.post(
            f"/admin/tenants/{tenant_id}/canary/search_agent/retire",
            params={"reason": "test_retire"},
        )
        assert retire_resp.status_code == 200, retire_resp.text
        retired_state = retire_resp.json()["state"]
        assert retired_state["canary"] is None
        assert any(
            r.get("version") == 1 and r.get("reason") == "test_retire"
            for r in retired_state["retired"]
        )

    def test_invalid_traffic_pct_returns_400(self, client: TestClient, phoenix_env):
        tenant_id = f"p4can_{uuid.uuid4().hex[:8]}"
        # No need to seed — promote_to_canary validates traffic_pct first.
        resp = client.post(
            f"/admin/tenants/{tenant_id}/canary/search_agent/promote",
            json={"version": 1, "traffic_pct": 0},  # invalid: must be > 0
        )
        assert resp.status_code == 400
