"""Admin endpoints for pin quota / signature variant / canary.

Without these endpoints, operators had no way to reach pin quotas,
signature variants, or canary promote/retire without writing custom
Python.

This test mounts the real `admin` router on a FastAPI TestClient and
hits each endpoint. The canary and pin-quota endpoints round-trip
through real Phoenix (docker-managed); the variant endpoints use the
in-memory override store (no persistence layer for those keys yet).
"""

from __future__ import annotations

import time
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
    # Context-managed so every request shares one portal event loop — the
    # write-behind queue's worker lives on the loop that served the PUT and
    # must survive across requests, as it does under a real server.
    with TestClient(app) as managed:
        yield managed


@pytest.fixture
def phoenix_env(phoenix_container):
    """Point the canary/pin-quota endpoints at the docker-managed Phoenix
    (per-pid port). The router reads the module state wired at startup via
    set_phoenix_endpoints, not the process environment."""
    saved = dict(admin._phoenix_endpoints)
    admin.set_phoenix_endpoints(
        phoenix_container["http_endpoint"], phoenix_container["otlp_endpoint"]
    )
    yield
    admin.set_phoenix_endpoints(saved["http_endpoint"], saved["grpc_endpoint"])


# ----- pin-quota endpoints ----------------------------------------------


class TestPinQuotaEndpoints:
    def test_get_returns_defaults_when_unset(self, client: TestClient, phoenix_env):
        tenant = f"pinq_{uuid.uuid4().hex[:8]}"
        resp = client.get(f"/admin/tenants/{tenant}/pin_quotas")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"tenant_id", "quotas", "pending_write"}
        assert body["tenant_id"] == tenant
        # Defaults from PinQuotas dataclass; org_admin None (unlimited) -> -1.
        assert body["quotas"] == {"user": 50, "tenant_admin": 500, "org_admin": -1}

    def test_put_updates_one_field_keeps_others(self, client: TestClient, phoenix_env):
        tenant = f"pinq_{uuid.uuid4().hex[:8]}"
        # Set a baseline.
        baseline = client.get(f"/admin/tenants/{tenant}/pin_quotas").json()["quotas"]
        # Update only the user quota.
        resp = client.put(
            f"/admin/tenants/{tenant}/pin_quotas",
            json={"user": 99},
        )
        assert resp.status_code == 200
        updated = resp.json()["quotas"]
        assert updated["user"] == 99
        # Other fields preserved.
        assert updated["tenant_admin"] == baseline["tenant_admin"]
        assert updated["org_admin"] == baseline["org_admin"]
        # GET reflects the put, field for field.
        again = client.get(f"/admin/tenants/{tenant}/pin_quotas").json()["quotas"]
        assert again["user"] == 99
        assert again == updated

    def test_put_survives_process_restart(self, client: TestClient, phoenix_env):
        """A PUT must persist durably, not just in the process cache. Clearing
        the cache simulates a runtime restart; the value must reload from the
        store on the next GET."""
        tenant = f"pinq_{uuid.uuid4().hex[:8]}"
        put = client.put(
            f"/admin/tenants/{tenant}/pin_quotas",
            json={"user": 7, "tenant_admin": 3},
        )
        assert put.status_code == 200
        put_body = put.json()
        assert put_body["tenant_id"] == tenant
        assert put_body["quotas"] == {"user": 7, "tenant_admin": 3, "org_admin": -1}
        assert put_body["pending_write"] is True

        # Persistence is write-behind: wait for the accepted write to land
        # (pending_write is the reportable settle signal), then simulate a
        # fresh process by dropping the write-through cache so the next read
        # must hit the durable store.
        deadline = time.monotonic() + 30
        while client.get(f"/admin/tenants/{tenant}/pin_quotas").json()["pending_write"]:
            assert time.monotonic() < deadline, "pin-quota write never landed"
            time.sleep(0.05)
        admin._reset_admin_overrides_for_tests()

        reloaded = client.get(f"/admin/tenants/{tenant}/pin_quotas").json()["quotas"]
        assert reloaded["user"] == 7
        assert reloaded["tenant_admin"] == 3
        assert reloaded["org_admin"] == -1

    def test_negative_quota_rejected(self, client: TestClient):
        # Validation happens before any store access, so no Phoenix needed.
        resp = client.put(
            f"/admin/tenants/pinq_{uuid.uuid4().hex[:8]}/pin_quotas",
            json={"user": -1},
        )
        assert resp.status_code == 400
        assert "must be >= 0" in resp.text


# ----- signature-variant endpoints ------------------------------------------


# Signature-variant endpoints (in-memory override store, no Phoenix) moved to
# tests/runtime/unit/test_admin_signature_variants.py so the fast gate covers
# them; the pin-quota and canary endpoints below round-trip real Phoenix.


# ----- canary endpoints (real Phoenix) --------------------------------------


class TestCanaryEndpoints:
    def test_promote_then_retire_round_trip(
        self, client: TestClient, phoenix_env, phoenix_container
    ):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_telemetry_phoenix.provider import PhoenixProvider

        tenant_id = f"p4can_{uuid.uuid4().hex[:8]}"
        # Seed a versioned dataset so promote_to_canary has something to promote.
        provider = PhoenixProvider()
        provider.initialize(
            {
                "tenant_id": tenant_id,
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["otlp_endpoint"],
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
