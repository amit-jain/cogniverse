"""In-process tests for the admin router's memory, invite, restore, and
stats endpoints.

Each test mounts the real router and executes the live handler body,
stubbing only the outermost boundary (Mem0MemoryManager, ConfigManager,
BackendRegistry, pin service) and asserting the exact calls the handler
makes against it plus the exact response body.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import admin as admin_router


@pytest.fixture(autouse=True)
def _stub_pin_quota_store(monkeypatch):
    """Pin/promote enforcement now warms quotas from the durable artifact store
    (real Phoenix in prod). These in-process route tests have no Phoenix, so
    stub the factory with an in-memory blob store — the quota values are not
    what these tests assert (the store round-trip is covered by
    test_pin_quota_enforcement_reads_blob.py). Tests that need specific quota
    values override this in their own body."""

    class _InMemoryAM:
        _blobs: dict = {}

        def __init__(self, tenant):
            self._tenant = tenant

        async def load_blob(self, kind, key):
            return self._blobs.get((self._tenant, kind, key))

        async def save_blob(self, kind, key, raw):
            self._blobs[(self._tenant, kind, key)] = raw

    _InMemoryAM._blobs = {}
    monkeypatch.setattr(
        admin_router, "_build_artifact_manager", lambda key: _InMemoryAM(key)
    )
    admin_router._reset_admin_overrides_for_tests()
    yield
    admin_router._reset_admin_overrides_for_tests()


def _make_stub_manager_class(delete_results=None):
    """Build a fresh Mem0MemoryManager stand-in class that records every
    constructor + method call. ``delete_results`` maps agent_name → bool."""

    class _StubMemoryManager:
        instances: list = []

        def __init__(self, tenant_id):
            self.tenant_id = tenant_id
            self.memory = object()  # truthy → backend initialised
            self.delete_calls: list = []
            self.clear_calls: list = []
            type(self).instances.append(self)

        def delete_memory(self, memory_id, tenant_id, agent_name):
            self.delete_calls.append(
                {
                    "memory_id": memory_id,
                    "tenant_id": tenant_id,
                    "agent_name": agent_name,
                }
            )
            return (delete_results or {}).get(agent_name, False)

        def clear_agent_memory(self, tenant_id, agent_name):
            self.clear_calls.append({"tenant_id": tenant_id, "agent_name": agent_name})

    return _StubMemoryManager


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAdminDeleteMemory:
    def test_namespace_iteration_order_is_pinned(self):
        assert admin_router._ADMIN_ALL_NAMESPACES == [
            "_user_memories",
            "_strategy_store",
        ]

    def test_delete_found_in_second_namespace(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class(
            delete_results={"_user_memories": False, "_strategy_store": True}
        )
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod/mem-123")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "memory_id": "mem-123"}

        (mgr,) = stub_cls.instances
        assert mgr.tenant_id == "acme:prod"
        assert mgr.delete_calls == [
            {
                "memory_id": "mem-123",
                "tenant_id": "acme:prod",
                "agent_name": "_user_memories",
            },
            {
                "memory_id": "mem-123",
                "tenant_id": "acme:prod",
                "agent_name": "_strategy_store",
            },
        ]

    def test_delete_short_circuits_on_first_namespace_hit(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class(
            delete_results={"_user_memories": True, "_strategy_store": True}
        )
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod/mem-1st")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted", "memory_id": "mem-1st"}
        (mgr,) = stub_cls.instances
        assert [c["agent_name"] for c in mgr.delete_calls] == ["_user_memories"]

    def test_delete_unknown_memory_returns_404_after_both_namespaces(
        self, client, monkeypatch
    ):
        stub_cls = _make_stub_manager_class(delete_results={})
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod/mem-404")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Memory mem-404 not found"}
        (mgr,) = stub_cls.instances
        assert [c["agent_name"] for c in mgr.delete_calls] == [
            "_user_memories",
            "_strategy_store",
        ]

    def test_uninitialised_memory_backend_returns_503(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class()

        def _init(self, tenant_id):
            self.tenant_id = tenant_id
            self.memory = None
            self.delete_calls = []
            self.clear_calls = []
            stub_cls.instances.append(self)

        stub_cls.__init__ = _init
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod/mem-x")
        assert resp.status_code == 503
        assert resp.json() == {"detail": "Memory backend not initialised"}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAdminClearMemories:
    def test_clear_all_iterates_every_namespace(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class()
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared", "type": "all"}
        (mgr,) = stub_cls.instances
        assert mgr.tenant_id == "acme:prod"
        assert mgr.clear_calls == [
            {"tenant_id": "acme:prod", "agent_name": "_user_memories"},
            {"tenant_id": "acme:prod", "agent_name": "_strategy_store"},
        ]

    def test_clear_preference_targets_only_user_memories(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class()
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod", params={"type": "preference"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared", "type": "preference"}
        (mgr,) = stub_cls.instances
        assert mgr.clear_calls == [
            {"tenant_id": "acme:prod", "agent_name": "_user_memories"}
        ]

    def test_clear_strategy_targets_only_strategy_store(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class()
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod", params={"type": "strategy"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared", "type": "strategy"}
        (mgr,) = stub_cls.instances
        assert mgr.clear_calls == [
            {"tenant_id": "acme:prod", "agent_name": "_strategy_store"}
        ]

    def test_clear_unknown_type_returns_400_without_clearing(self, client, monkeypatch):
        stub_cls = _make_stub_manager_class()
        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", stub_cls
        )

        resp = client.delete("/admin/memories/acme:prod", params={"type": "bogus"})
        assert resp.status_code == 400
        assert resp.json() == {"detail": "Unknown memory type: bogus"}
        (mgr,) = stub_cls.instances
        assert mgr.clear_calls == []


@pytest.mark.unit
@pytest.mark.ci_fast
class TestMessagingInvite:
    def test_invite_persists_blob_with_computed_expiry(self):
        from cogniverse_sdk.interfaces.config_store import ConfigScope

        cm = MagicMock(name="config_manager")
        app = FastAPI()
        app.include_router(admin_router.router, prefix="/admin")
        app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
            cm
        )

        before = datetime.now(timezone.utc) + timedelta(hours=6)
        with TestClient(app) as client:
            resp = client.post(
                "/admin/messaging/invite",
                json={"tenant_id": "acme:prod", "expires_in_hours": 6},
            )
        after = datetime.now(timezone.utc) + timedelta(hours=6)

        assert resp.status_code == 200
        body = resp.json()
        token = body["token"]
        assert body == {"token": token, "tenant_id": "acme:prod"}
        assert len(token) == 32
        int(token, 16)  # uuid4().hex — 32 hex chars

        cm.set_config_value.assert_called_once()
        kwargs = cm.set_config_value.call_args.kwargs
        blob = kwargs.pop("config_value")
        assert kwargs == {
            "tenant_id": "_system",
            "scope": ConfigScope.SYSTEM,
            "service": "messaging_gateway",
            "config_key": f"invite_token_{token}",
        }
        expires_at = datetime.fromisoformat(blob.pop("expires_at"))
        assert before <= expires_at <= after
        assert blob == {"tenant_id": "acme:prod", "token": token, "used": False}

    def test_invite_default_ttl_is_24_hours(self):
        cm = MagicMock(name="config_manager")
        app = FastAPI()
        app.include_router(admin_router.router, prefix="/admin")
        app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
            cm
        )

        before = datetime.now(timezone.utc) + timedelta(hours=24)
        with TestClient(app) as client:
            resp = client.post(
                "/admin/messaging/invite", json={"tenant_id": "acme:prod"}
            )
        after = datetime.now(timezone.utc) + timedelta(hours=24)

        assert resp.status_code == 200
        blob = cm.set_config_value.call_args.kwargs["config_value"]
        expires_at = datetime.fromisoformat(blob["expires_at"])
        assert before <= expires_at <= after


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRestoreMemory:
    def test_restore_clears_archived_flag(self, client, monkeypatch):
        mm = MagicMock(name="memory_manager")
        mm.restore_archived_memory.return_value = True
        pin_service_calls = []

        def _get_pin_service(tenant_id):
            pin_service_calls.append(tenant_id)
            return SimpleNamespace(_mm=mm)

        monkeypatch.setattr(admin_router, "_get_pin_service", _get_pin_service)

        resp = client.post("/admin/tenants/acme:prod/memories/mem-77/restore")
        assert resp.status_code == 200
        assert resp.json() == {
            "tenant_id": "acme:prod",
            "memory_id": "mem-77",
            "restored": True,
        }
        assert pin_service_calls == ["acme:prod"]
        mm.restore_archived_memory.assert_called_once_with("mem-77")

    def test_restore_unknown_or_unarchived_memory_returns_404(
        self, client, monkeypatch
    ):
        mm = MagicMock(name="memory_manager")
        mm.restore_archived_memory.return_value = False
        monkeypatch.setattr(
            admin_router,
            "_get_pin_service",
            lambda tenant_id: SimpleNamespace(_mm=mm),
        )

        resp = client.post("/admin/tenants/acme:prod/memories/mem-gone/restore")
        assert resp.status_code == 404
        assert resp.json() == {
            "detail": "memory mem-gone not found or not in archived state"
        }
        mm.restore_archived_memory.assert_called_once_with("mem-gone")


class _StubStatsBackend:
    """Sync ``get_statistics`` matching the Backend interface every registered
    backend implements — pinned against the real VespaBackend type below so
    this stub cannot drift into a shape production never builds."""

    def __init__(self):
        self.get_statistics_calls = 0

    def get_statistics(self):
        self.get_statistics_calls += 1
        return {"backend": "vespa", "status": "healthy", "search_enabled": True}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSystemStats:
    def _app(self, monkeypatch, backend=None):
        cm = MagicMock(name="config_manager")
        sl = MagicMock(name="schema_loader")
        registry = MagicMock(name="backend_registry")
        registry.list_backends.return_value = ["vespa"]
        registry.get_ingestion_backend.return_value = backend
        registry_cls = MagicMock(name="BackendRegistry")
        registry_cls.get_instance.return_value = registry
        monkeypatch.setattr(admin_router, "BackendRegistry", registry_cls)

        app = FastAPI()
        app.include_router(admin_router.router, prefix="/admin")
        app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
            cm
        )
        app.dependency_overrides[admin_router.get_schema_loader_dependency] = lambda: sl
        return app, cm, sl, registry

    def test_general_stats_top_level_keys_and_types(self, monkeypatch):
        app, _, _, registry = self._app(monkeypatch)
        with TestClient(app) as client:
            before = datetime.now()
            resp = client.get("/admin/system/stats")
            after = datetime.now()

        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"registered_backends", "timestamp"}
        assert body["registered_backends"] == ["vespa"]
        stamp = datetime.fromisoformat(body["timestamp"])
        assert before <= stamp <= after
        registry.get_ingestion_backend.assert_not_called()

    def test_backend_stats_merge_into_response(self, monkeypatch):
        backend = _StubStatsBackend()
        app, cm, sl, registry = self._app(monkeypatch, backend=backend)
        with TestClient(app) as client:
            resp = client.get(
                "/admin/system/stats",
                params={"tenant_id": "acme:prod", "backend": "vespa"},
            )

        assert resp.status_code == 200
        body = resp.json()
        stamp = body.pop("timestamp")
        datetime.fromisoformat(stamp)
        assert body == {
            "registered_backends": ["vespa"],
            "backend": "vespa",
            "tenant_id": "acme:prod",
            "backend_type": "_StubStatsBackend",
            "status": "healthy",
            "search_enabled": True,
        }
        assert backend.get_statistics_calls == 1
        registry.get_ingestion_backend.assert_called_once_with(
            "vespa",
            tenant_id="acme:prod",
            config_manager=cm,
            schema_loader=sl,
        )

    def test_stub_contract_matches_real_backend_type(self):
        """The route dispatches on the sync Backend-interface method. The
        real VespaBackend exposes ``get_statistics`` and has NO ``get_stats``
        — a stub with any other method name or an async signature exercises a
        type production never builds, and the backend-specific stats path
        would ship dead (501 on every real request) behind a green test."""
        import inspect

        from cogniverse_vespa.backend import VespaBackend

        assert not hasattr(VespaBackend, "get_stats")
        assert callable(VespaBackend.get_statistics)
        assert not inspect.iscoroutinefunction(VespaBackend.get_statistics)
        assert not hasattr(_StubStatsBackend, "get_stats")
        assert not inspect.iscoroutinefunction(_StubStatsBackend.get_statistics)

    def test_backend_without_statistics_method_returns_501(self, monkeypatch):
        class _NoStats:
            pass

        app, _, _, _ = self._app(monkeypatch, backend=_NoStats())
        with TestClient(app) as client:
            resp = client.get(
                "/admin/system/stats",
                params={"tenant_id": "acme:prod", "backend": "vespa"},
            )
        assert resp.status_code == 501
        assert "get_statistics" in resp.json()["detail"]


def _trusted_row(memory_id: str, score: float = 0.5, endorsements: int = 2):
    from cogniverse_core.memory.trust import TrustRecord, attach_trust_to_metadata

    trust = TrustRecord(
        score=score,
        initial_score=score,
        decayed_at="2026-07-01T00:00:00+00:00",
        endorsements=endorsements,
    )
    return {
        "id": memory_id,
        "memory": "acme ships on fridays",
        "metadata": attach_trust_to_metadata({}, trust),
    }


def _pin_service_with_mm(monkeypatch, rows):
    """Patch _get_pin_service with a stub whose ._mm serves ``rows`` and
    records memory.update calls — the exact seam the endorse/promote/restore
    handlers reuse."""
    mm = MagicMock(name="memory_manager")
    mm.memory.get_all.return_value = {"results": rows}
    svc = SimpleNamespace(_mm=mm)
    monkeypatch.setattr(admin_router, "_get_pin_service", lambda tenant_id: svc)
    return mm


@pytest.mark.unit
@pytest.mark.ci_fast
class TestEndorseMemoryRoute:
    def test_endorse_bumps_trust_and_persists(self, client, monkeypatch):
        mm = _pin_service_with_mm(monkeypatch, [_trusted_row("mem-1")])

        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-1/endorse",
            json={"endorser_role": "org_admin", "actor_id": "ops@acme"},
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "memory_id": "mem-1",
            "new_score": 0.7,
            "endorsements": 3,
        }
        update = mm.memory.update.call_args.kwargs
        assert update["memory_id"] == "mem-1"
        assert update["data"] == "acme ships on fridays"
        assert update["metadata"]["trust"]["score"] == 0.7
        assert update["metadata"]["trust"]["endorsements"] == 3

    def test_unknown_role_400_before_backend(self, client, monkeypatch):
        mm = _pin_service_with_mm(monkeypatch, [_trusted_row("mem-1")])
        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-1/endorse",
            json={"endorser_role": "superuser", "actor_id": "ops"},
        )
        assert resp.status_code == 400
        assert "superuser" in resp.json()["detail"]
        mm.memory.get_all.assert_not_called()

    def test_unknown_memory_404(self, client, monkeypatch):
        _pin_service_with_mm(monkeypatch, [_trusted_row("other")])
        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-x/endorse",
            json={"endorser_role": "user", "actor_id": "ops"},
        )
        assert resp.status_code == 404

    def test_backend_outage_maps_to_503_not_success(self, client, monkeypatch):
        mm = _pin_service_with_mm(monkeypatch, [])
        mm.memory.get_all.side_effect = ConnectionError("vespa down")
        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-1/endorse",
            json={"endorser_role": "user", "actor_id": "ops"},
        )
        assert resp.status_code == 503
        assert "vespa down" in resp.json()["detail"]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPromoteToOrgTrunkRoute:
    def _stub_federation(self, monkeypatch, *, denied=False):
        calls = {}

        class _StubFederation:
            def __init__(self, memory_manager_factory, registry):
                calls["registry"] = registry

            def promote_to_org_trunk(self, **kwargs):
                calls["promote"] = kwargs
                if denied:
                    from cogniverse_core.memory.federation import (
                        FederationDeniedError,
                    )

                    raise FederationDeniedError("role user may not promote")
                return SimpleNamespace(
                    source_memory_id=kwargs["source_memory"]["id"],
                    promoted_memory_id="trunk-mem-9",
                    org_trunk_tenant_id="acme:__org_trunk__",
                )

        monkeypatch.setattr(
            "cogniverse_core.memory.federation.FederationService", _StubFederation
        )
        return calls

    def test_promote_returns_trunk_ids(self, client, monkeypatch):
        _pin_service_with_mm(monkeypatch, [_trusted_row("mem-7")])
        calls = self._stub_federation(monkeypatch)

        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-7/promote_to_org_trunk",
            json={"actor_role": "org_admin", "actor_id": "ops@acme"},
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "source_tenant_id": "acme:prod",
            "source_memory_id": "mem-7",
            "promoted_memory_id": "trunk-mem-9",
            "org_trunk_tenant_id": "acme:__org_trunk__",
        }
        assert calls["promote"]["source_tenant_id"] == "acme:prod"
        assert calls["promote"]["actor_id"] == "ops@acme"
        assert calls["promote"]["actor_role"].value == "org_admin"

    def test_denied_maps_to_403(self, client, monkeypatch):
        _pin_service_with_mm(monkeypatch, [_trusted_row("mem-7")])
        self._stub_federation(monkeypatch, denied=True)
        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-7/promote_to_org_trunk",
            json={"actor_role": "user", "actor_id": "u1"},
        )
        assert resp.status_code == 403


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPinRoutes:
    def _svc(self, monkeypatch):
        svc = MagicMock(name="pin_service")
        monkeypatch.setattr(admin_router, "_get_pin_service", lambda tenant_id: svc)
        return svc

    def _record(self, target="mem-3"):
        from cogniverse_core.memory.pinning import Pinnable

        return SimpleNamespace(
            memory_id="pin-1",
            target_memory_id=target,
            target_kind="entity_fact",
            pinned_by=Pinnable("tenant_admin"),
            pinned_by_actor="ops@acme",
        )

    def test_pin_returns_persisted_record(self, client, monkeypatch):
        svc = self._svc(monkeypatch)
        svc.pin.return_value = self._record()

        resp = client.post(
            "/admin/tenants/acme:prod/memories/mem-3/pin",
            json={
                "target_kind": "entity_fact",
                "pinned_by": "tenant_admin",
                "actor_id": "ops@acme",
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {
            "memory_id": "pin-1",
            "target_memory_id": "mem-3",
            "target_kind": "entity_fact",
            "pinned_by": "tenant_admin",
            "pinned_by_actor": "ops@acme",
        }
        assert svc.pin.call_args.kwargs["target_memory_id"] == "mem-3"
        assert svc.pin.call_args.kwargs["tenant_id"] == "acme:prod"

    def test_pin_quota_maps_to_429_and_authority_to_403(self, client, monkeypatch):
        from cogniverse_core.memory.pinning import (
            PinAuthorityError,
            PinQuotaExceededError,
        )

        svc = self._svc(monkeypatch)
        svc.pin.side_effect = PinQuotaExceededError("user quota 5 reached")
        resp = client.post(
            "/admin/tenants/acme:prod/memories/m/pin",
            json={"target_kind": "k", "pinned_by": "user", "actor_id": "u"},
        )
        assert resp.status_code == 429

        svc.pin.side_effect = PinAuthorityError("user may not pin org kind")
        resp = client.post(
            "/admin/tenants/acme:prod/memories/m/pin",
            json={"target_kind": "k", "pinned_by": "user", "actor_id": "u"},
        )
        assert resp.status_code == 403

    def test_unpin_reports_removed_count(self, client, monkeypatch):
        svc = self._svc(monkeypatch)
        svc.unpin.return_value = 2
        resp = client.request(
            "DELETE",
            "/admin/tenants/acme:prod/memories/mem-3/pin",
            json={"requester_role": "org_admin", "actor_id": "ops"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "tenant_id": "acme:prod",
            "target_memory_id": "mem-3",
            "removed": 2,
        }

    def test_list_pins_serialises_records(self, client, monkeypatch):
        svc = self._svc(monkeypatch)
        svc.list_pins.return_value = [self._record()]
        resp = client.get("/admin/tenants/acme:prod/pins")
        assert resp.status_code == 200
        assert resp.json() == {
            "tenant_id": "acme:prod",
            "pins": [
                {
                    "memory_id": "pin-1",
                    "target_memory_id": "mem-3",
                    "target_kind": "entity_fact",
                    "pinned_by": "tenant_admin",
                    "pinned_by_actor": "ops@acme",
                }
            ],
        }


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSessionRoutes:
    def _stub_manager_cls(self, monkeypatch, drop_result):
        class _StubMgr:
            instances: list = []
            _instances: dict = {}

            def __init__(self, tenant_id):
                self.tenant_id = tenant_id
                self.memory = object()
                self.drop_calls: list = []
                type(self).instances.append(self)

            def drop_session(self, session_id, registry):
                self.drop_calls.append(session_id)
                return dict(drop_result)

        monkeypatch.setattr(
            "cogniverse_core.memory.manager.Mem0MemoryManager", _StubMgr
        )
        return _StubMgr

    def test_drop_session_reports_per_kind_counts(self, client, monkeypatch):
        stub_cls = self._stub_manager_cls(
            monkeypatch, {"chat_turn": 3, "scratchpad": 1}
        )
        resp = client.delete("/admin/tenants/acme:prod/sessions/sess-9")
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "dropped",
            "tenant_id": "acme:prod",
            "session_id": "sess-9",
            "deleted_by_kind": {"chat_turn": 3, "scratchpad": 1},
            "total_deleted": 4,
        }
        (mgr,) = stub_cls.instances
        assert mgr.tenant_id == "acme:prod"
        assert mgr.drop_calls == ["sess-9"]

    def test_close_session_sweeps_warm_tenants(self, client, monkeypatch):
        stub_cls = self._stub_manager_cls(monkeypatch, {"chat_turn": 2})
        warm_a = stub_cls("acme:prod")
        warm_b = stub_cls("beta:beta")
        stub_cls._instances = {"acme:prod": warm_a, "beta:beta": warm_b}

        resp = client.post("/admin/sessions/sess-9/close")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "closed"
        assert body["session_id"] == "sess-9"
        assert body["total_deleted"] == 4
        assert body["per_tenant"] == {
            "acme:prod": {"chat_turn": 2},
            "beta:beta": {"chat_turn": 2},
        }
        assert warm_a.drop_calls == ["sess-9"]
        assert warm_b.drop_calls == ["sess-9"]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCanaryRoutes:
    def _am(self, monkeypatch):
        am = MagicMock(name="artifact_manager")
        monkeypatch.setattr(
            admin_router, "_build_artifact_manager", lambda tenant_id: am
        )
        return am

    def test_promote_canary_returns_state(self, client, monkeypatch):
        am = self._am(monkeypatch)

        async def _promote(agent_type, version, traffic_pct):
            return {"canary_version": version, "traffic_pct": traffic_pct}

        am.promote_to_canary = _promote
        resp = client.post(
            "/admin/tenants/acme:prod/canary/summarizer_agent/promote",
            json={"version": 4, "traffic_pct": 25},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "tenant_id": "acme:prod",
            "agent_type": "summarizer_agent",
            "state": {"canary_version": 4, "traffic_pct": 25},
        }

    def test_promote_unknown_version_maps_to_400(self, client, monkeypatch):
        am = self._am(monkeypatch)

        async def _promote(agent_type, version, traffic_pct):
            raise ValueError("no artefact at version 99")

        am.promote_to_canary = _promote
        resp = client.post(
            "/admin/tenants/acme:prod/canary/summarizer_agent/promote",
            json={"version": 99, "traffic_pct": 10},
        )
        assert resp.status_code == 400
        assert "version 99" in resp.json()["detail"]

    def test_retire_canary_returns_state(self, client, monkeypatch):
        am = self._am(monkeypatch)

        async def _retire(agent_type, reason):
            return {"canary_version": None, "retired_reason": reason}

        am.retire_canary = _retire
        resp = client.post(
            "/admin/tenants/acme:prod/canary/summarizer_agent/retire",
            params={"reason": "bad_quality"},
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "tenant_id": "acme:prod",
            "agent_type": "summarizer_agent",
            "state": {"canary_version": None, "retired_reason": "bad_quality"},
        }
