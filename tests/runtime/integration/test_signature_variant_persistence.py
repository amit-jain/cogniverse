"""Admin pin-quota and signature-variant writes reach every process intact.

Each process holds its own admin router state and its own config-store
session. These stand independent router copies — each wired to its own real
Vespa session, as two worker processes are — on one Vespa and pin that a PUT
answered by one is what the other reads, that concurrent PUTs on either never
erase each other's fields, and that a store failing mid-PUT answers 503 with
nothing written.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import threading
import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import admin as admin_router
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.http_fault_proxy import InterceptFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KINDS = ["pin_quotas", "signature_variants"]
INITIAL = {
    "pin_quotas": {"user": 1, "tenant_admin": 2, "org_admin": -1},
    "signature_variants": {"search_agent": "initial"},
}
# (path suffix, body, field it sets) for three distinct partial PUTs per kind.
PUTS = {
    "pin_quotas": [
        ("", {"user": 7}, ("user", 7)),
        ("", {"tenant_admin": 9}, ("tenant_admin", 9)),
        ("", {"org_admin": 12}, ("org_admin", 12)),
    ],
    "signature_variants": [
        ("/search_agent", {"variant_id": "search-v2"}, ("search_agent", "search-v2")),
        (
            "/summarizer_agent",
            {"variant_id": "summary-v3"},
            ("summarizer_agent", "summary-v3"),
        ),
        (
            "/detailed_report_agent",
            {"variant_id": "report-v4"},
            ("detailed_report_agent", "report-v4"),
        ),
    ],
}
FIELD = {"pin_quotas": "quotas", "signature_variants": "selections"}


def _tenant() -> str:
    name = f"sigvarpersist{uuid.uuid4().hex[:8]}"
    return f"{name}:{name}"


def _session(port: int, host: str = "http://localhost") -> VespaConfigStore:
    return VespaConfigStore(backend_url=host, backend_port=port)


def _record(store: VespaConfigStore, tenant: str, kind: str):
    return store.get_config(tenant, ConfigScope.SYSTEM, "admin_overrides", kind)


def _seed(store: VespaConfigStore, tenant: str, kind: str) -> None:
    store.set_config(
        tenant, ConfigScope.SYSTEM, "admin_overrides", kind, dict(INITIAL[kind])
    )


def _with(kind: str, *sets) -> dict:
    return {**INITIAL[kind], **dict(sets)}


@pytest.fixture
def store(vespa_instance):
    store = _session(vespa_instance["http_port"])
    yield store
    store.close()


def _load_replica(store: VespaConfigStore):
    name = f"cogniverse_runtime.routers.state_replica_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, admin_router.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.set_config_manager(ConfigManager(store=store))
    return module


@pytest.fixture
def replicas(vespa_instance):
    """Independent admin router state, each on its own config-store session."""
    sessions = [_session(vespa_instance["http_port"]) for _ in range(2)]
    modules = [_load_replica(session) for session in sessions]
    yield modules
    for module, session in zip(modules, sessions):
        sys.modules.pop(module.__name__)
        session.close()


@pytest.fixture
def dispatching_replica(vespa_instance):
    """The admin router the dispatcher reads, on its own session."""
    previous = admin_router._config_manager
    session = _session(vespa_instance["http_port"])
    admin_router.set_config_manager(ConfigManager(store=session))
    yield admin_router
    admin_router.set_config_manager(previous)
    session.close()


def _client(module):
    app = FastAPI()
    app.include_router(module.router, prefix="/admin")
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://replica")


class _Overlay:
    """The artefact manager the dispatcher builds its overlay from."""

    async def load_for_request(self, agent_name, *, request_seed, variant_id):
        return {
            "prompts": None,
            "served_from": "default",
            "version": None,
            "variant_id": variant_id,
        }


async def _served_variant(agent_name: str, tenant: str) -> dict:
    dispatcher = object.__new__(AgentDispatcher)
    dispatcher._artifact_manager_factory = lambda tenant_id: _Overlay()
    overlay = await dispatcher.resolve_artefact_for_request(
        agent_name, tenant, "seed-1"
    )
    return {
        "variant_id": overlay["variant_id"],
        "variant_lookup_status": overlay["variant_lookup_status"],
    }


@pytest.mark.asyncio
async def test_a_variant_put_on_one_replica_is_served_by_another(
    replicas, dispatching_replica, store
):
    tenant = _tenant()
    writer, _ = replicas
    async with _client(writer) as client:
        resp = await client.put(
            f"/admin/tenants/{tenant}/signature_variants/search_agent",
            json={"variant_id": "search_v2"},
        )
    assert resp.status_code == 200
    assert resp.json() == {
        "tenant_id": tenant,
        "selections": {"search_agent": "search_v2"},
    }

    # The dispatcher on another process (cold) serves the stored variant.
    assert await _served_variant("search_agent", tenant) == {
        "variant_id": "search_v2",
        "variant_lookup_status": "loaded",
    }
    # An agent the tenant never selected still resolves to the default.
    assert await _served_variant("summarizer_agent", tenant) == {
        "variant_id": "default",
        "variant_lookup_status": "loaded",
    }
    stored = _record(store, tenant, "signature_variants")
    assert (stored.version, stored.config_value) == (1, {"search_agent": "search_v2"})


@pytest.mark.asyncio
async def test_a_warm_dispatcher_serves_another_replicas_put_within_the_bound(
    replicas, dispatching_replica, monkeypatch
):
    """A selection this process already serves is re-read once it is
    SIGNATURE_VARIANT_REFRESH_S old: the next request starts the refresh and
    the one after serves the other replica's PUT."""
    from cogniverse_foundation.caching.refreshing_cache import RefreshingCache

    clock = {"now": 1000.0}
    monkeypatch.setattr(
        admin_router,
        "_signature_variant_cache",
        RefreshingCache(
            name="signature-variants",
            refresh_after_s=admin_router.SIGNATURE_VARIANT_REFRESH_S,
            max_staleness_s=admin_router.SIGNATURE_VARIANT_MAX_STALENESS_S,
            max_entries=16,
            clock=lambda: clock["now"],
        ),
    )
    tenant = _tenant()
    writer, _ = replicas
    async with _client(writer) as client:
        await client.put(
            f"/admin/tenants/{tenant}/signature_variants/search_agent",
            json={"variant_id": "v1"},
        )
        assert (await _served_variant("search_agent", tenant))["variant_id"] == "v1"
        await client.put(
            f"/admin/tenants/{tenant}/signature_variants/search_agent",
            json={"variant_id": "v2"},
        )

    clock["now"] += admin_router.SIGNATURE_VARIANT_REFRESH_S - 0.5
    assert (await _served_variant("search_agent", tenant))["variant_id"] == "v1"
    clock["now"] += 0.5
    # This request starts the background refresh and is answered from memory.
    assert (await _served_variant("search_agent", tenant))["variant_id"] == "v1"
    for thread in threading.enumerate():
        if thread.name == "signature-variants-refresh":
            thread.join(timeout=30)
    assert (await _served_variant("search_agent", tenant))["variant_id"] == "v2"


@pytest.mark.asyncio
async def test_a_put_on_this_replica_is_served_by_its_dispatcher_at_once(
    dispatching_replica,
):
    tenant = _tenant()
    async with _client(dispatching_replica) as client:
        await client.put(
            f"/admin/tenants/{tenant}/signature_variants/search_agent",
            json={"variant_id": "v1"},
        )
        assert (await _served_variant("search_agent", tenant))["variant_id"] == "v1"
        await client.put(
            f"/admin/tenants/{tenant}/signature_variants/search_agent",
            json={"variant_id": "v2"},
        )

    assert (await _served_variant("search_agent", tenant))["variant_id"] == "v2"


@pytest.mark.asyncio
async def test_second_agent_selection_merges_not_replaces(replicas, store):
    tenant = _tenant()
    first, second = replicas
    async with _client(first) as a, _client(second) as b:
        await a.put(
            f"/admin/tenants/{tenant}/signature_variants/search_agent",
            json={"variant_id": "search_v2"},
        )
        resp = await b.put(
            f"/admin/tenants/{tenant}/signature_variants/summarizer_agent",
            json={"variant_id": "sum_v3"},
        )
        read_on_first = await a.get(f"/admin/tenants/{tenant}/signature_variants")

    expected = {"search_agent": "search_v2", "summarizer_agent": "sum_v3"}
    assert resp.status_code == 200
    assert resp.json()["selections"] == expected
    assert read_on_first.json() == {"tenant_id": tenant, "selections": expected}
    assert _record(store, tenant, "signature_variants").config_value == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_warm_replica_partial_put_preserves_other_replica_fields(
    replicas, store, kind
):
    tenant = _tenant()
    first, second = replicas
    _seed(store, tenant, kind)
    path = f"/admin/tenants/{tenant}/{kind}"
    (first_suffix, first_body, first_set), (second_suffix, second_body, second_set) = (
        PUTS[kind][:2]
    )
    async with _client(first) as a, _client(second) as b:
        warm = await b.get(path)
        assert warm.status_code == 200
        assert warm.json()[FIELD[kind]] == INITIAL[kind]
        response = await a.put(path + first_suffix, json=first_body)
        assert response.status_code == 200
        assert response.json()[FIELD[kind]] == _with(kind, first_set)
        response = await b.put(path + second_suffix, json=second_body)
        assert response.status_code == 200
        expected = _with(kind, first_set, second_set)
        assert response.json() == {"tenant_id": tenant, FIELD[kind]: expected}

    stored = _record(store, tenant, kind)
    assert (stored.version, stored.config_value) == (3, expected)


class _InterleavedStore(VespaConfigStore):
    """A real session whose first ``hold`` reads of the record wait on a shared
    barrier, so PUTs on different replicas all read the same version before
    any of them writes."""

    def __init__(self, port: int, barrier: threading.Barrier, kind: str) -> None:
        super().__init__(backend_url="http://localhost", backend_port=port)
        self._barrier = barrier
        self._kind = kind
        self.held = 0

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        entry = super().get_config(tenant_id, scope, service, config_key, version)
        if config_key == self._kind and self.held == 0:
            self.held = 1
            self._barrier.wait(timeout=30)
        return entry


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_concurrent_puts_on_three_replicas_each_keep_their_field(
    vespa_instance, store, kind
):
    """Three replicas read the same stored version, then all write: two lose
    the compare-and-set, re-read and merge, so every field lands."""
    tenant = _tenant()
    _seed(store, tenant, kind)
    barrier = threading.Barrier(3)
    sessions = [
        _InterleavedStore(vespa_instance["http_port"], barrier, kind) for _ in range(3)
    ]
    modules = [_load_replica(session) for session in sessions]
    path = f"/admin/tenants/{tenant}/{kind}"
    try:
        clients = [_client(module) for module in modules]
        responses = await asyncio.gather(
            *[
                client.put(path + suffix, json=body)
                for client, (suffix, body, _) in zip(clients, PUTS[kind])
            ]
        )
        for client in clients:
            await client.aclose()
    finally:
        for module, session in zip(modules, sessions):
            sys.modules.pop(module.__name__)
            session.close()

    expected = _with(kind, *[field for _, _, field in PUTS[kind]])
    assert [session.held for session in sessions] == [1, 1, 1]
    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [response.json()[FIELD[kind]] for response in responses].count(expected) == 1
    stored = _record(store, tenant, kind)
    assert (stored.version, stored.config_value) == (4, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_concurrent_puts_on_one_replica_each_keep_their_field(
    replicas, store, kind
):
    tenant = _tenant()
    _seed(store, tenant, kind)
    replica, _ = replicas
    path = f"/admin/tenants/{tenant}/{kind}"
    async with _client(replica) as client:
        responses = await asyncio.gather(
            *[client.put(path + suffix, json=body) for suffix, body, _ in PUTS[kind]]
        )
        read_back = await client.get(path)

    expected = _with(kind, *[field for _, _, field in PUTS[kind]])
    assert [response.status_code for response in responses] == [200, 200, 200]
    assert read_back.json() == {"tenant_id": tenant, FIELD[kind]: expected}
    stored = _record(store, tenant, kind)
    assert (stored.version, stored.config_value) == (4, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a_put_the_store_cannot_complete_answers_503_and_writes_nothing(
    vespa_instance, store, kind, caplog
):
    """A PUT is answered only once it is stored: an unreadable store and a
    refused write both answer a typed 503, the stored record is unchanged,
    and the next PUT after recovery merges onto it. The store's error goes
    to the runtime log, not the body."""
    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
    tenant = _tenant()
    _seed(store, tenant, kind)
    path = f"/admin/tenants/{tenant}/{kind}"
    prefix = "pin-quota" if kind == "pin_quotas" else "signature-variant"
    (first_suffix, first_body, first_set), (second_suffix, second_body, second_set) = (
        PUTS[kind][:2]
    )
    with InterceptFaultProxy(
        f"http://localhost:{vespa_instance['http_port']}"
    ) as proxy:
        session = _session(proxy.port, host="http://127.0.0.1")
        replica = _load_replica(session)
        try:
            async with _client(replica) as client:
                proxy.intercept = lambda method, url, body: (
                    503,
                    {"message": "store offline"},
                )
                unreadable = await client.put(path + first_suffix, json=first_body)
                assert _record(store, tenant, kind).version == 1

                proxy.intercept = lambda method, url, body: (
                    (503, {"message": "write refused"})
                    if method in ("POST", "PUT")
                    else None
                )
                refused = await client.put(path + first_suffix, json=first_body)
                served_while_refusing = await client.get(path)

                proxy.intercept = None
                recovered = await client.put(path + second_suffix, json=second_body)
        finally:
            sys.modules.pop(replica.__name__)
            session.close()

    def unavailable(failure):
        return {
            "detail": {
                "error": "store_unavailable",
                "message": f"The {prefix} store did not answer; retry.",
                "failure": failure,
                "store": prefix,
                "tenant_id": tenant,
            }
        }

    assert unreadable.status_code == 503
    assert unreadable.json() == unavailable("ConfigStoreUnavailableError")
    assert refused.status_code == 503
    assert refused.json() == unavailable("VespaError")
    unreadable_cause, refused_cause = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ]
    assert unreadable_cause.startswith(
        "store_unavailable: ConfigStoreUnavailableError: Failed to read Vespa "
        "config visit after 5 attempts over "
    ), unreadable_cause
    assert refused_cause == "store_unavailable: VespaError: write refused"
    assert served_while_refusing.json() == {
        "tenant_id": tenant,
        FIELD[kind]: INITIAL[kind],
    }
    assert recovered.status_code == 200
    assert recovered.json()[FIELD[kind]] == _with(kind, second_set)
    assert first_set not in recovered.json()[FIELD[kind]].items()
    stored = _record(store, tenant, kind)
    assert (stored.version, stored.config_value) == (2, _with(kind, second_set))
