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
