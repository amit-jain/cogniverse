"""The admin tier routes against the real tenant registry and config store.

Tier is a per-tenant attribute an operator sets through the product. These
route the real FastAPI app over real Vespa: a set is read back exactly, an
unknown tenant is a 404, a value the router binds no group for is refused with
the vocabulary in the message, a store outage is a 503 rather than a tenant
silently reading as default, and eight concurrent sets on distinct tenants each
land on their own tenant.

A set reaches every runtime worker process before it is answered: another
worker process that holds the tenant's previous tier serves the new one on its
next read, and a worker that does not confirm dropping it fails the set with a
503 naming the stored tier.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import uuid

import httpx
import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.tenant_tiers import (
    read_tenant_tier,
    tenant_tier_reader,
)
from cogniverse_foundation.config.unified_config import (
    DEFAULT_ROUTER_TIER,
    ROUTER_TIERS,
)
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_vespa.config.config_store import VespaConfigStore

pytestmark = pytest.mark.integration

# Nothing listens here; the CI-parity default for a dead backend port.
DEAD_VESPA_PORT = 29071

CREATED_AT = 1757000000000
SEEDED_SCHEMAS = ["video_colpali_smol500_mv_frame"]


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader):
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    BackendRegistry.get_instance().clear_instances()


def _tier_channel() -> str:
    return f"cogniverse:test-tier-events:{uuid.uuid4().hex[:8]}"


async def _route_worker(redis_url: str, channel: str, worker_id: str = "route-worker"):
    """The worker process serving the routes: subscribed as main.py wires it."""
    events = ClusterEvents(
        redis_url,
        worker_id,
        {"tenant_tier_set": lambda payload: tm.release_tenant_tier(payload)},
        channel=channel,
    )
    await events.start()
    return events


@pytest.fixture
async def tier_channel(shared_state_redis_url, monkeypatch):
    """The cluster-events channel tier sets publish on; yields its name."""
    channel = _tier_channel()
    events = await _route_worker(shared_state_redis_url, channel)
    monkeypatch.setattr(tm, "_cluster_events", events)
    yield channel
    await events.close()


@pytest.fixture
def client(wired_tenant_manager, tier_channel):
    transport = httpx.ASGITransport(app=tm.app)
    return httpx.AsyncClient(transport=transport, base_url="http://tier-test")


def _seed_tenant() -> str:
    org_id = f"tier{uuid.uuid4().hex[:8]}"
    tenant_id = f"{org_id}:production"
    backend = tm.get_backend()
    stored = backend.create_metadata_document(
        schema="tenant_metadata",
        doc_id=tenant_id,
        fields={
            "tenant_full_id": tenant_id,
            "org_id": org_id,
            "tenant_name": "production",
            "created_at": CREATED_AT,
            "created_by": "tier-test",
            "status": "active",
            "schemas_deployed": SEEDED_SCHEMAS,
        },
    )
    assert stored is True
    return tenant_id


class TestTierRoundTrip:
    async def test_a_tenant_with_no_tier_reads_as_the_default(self, client):
        tenant_id = _seed_tenant()
        async with client as c:
            response = await c.get(f"/admin/tenants/{tenant_id}/tier")
        assert response.status_code == 200
        assert response.json() == {
            "tenant_id": tenant_id,
            "tier": DEFAULT_ROUTER_TIER,
        }

    async def test_every_tier_set_is_read_back_exactly(self, client, config_manager):
        tenant_id = _seed_tenant()
        async with client as c:
            for tier in sorted(ROUTER_TIERS):
                put = await c.put(
                    f"/admin/tenants/{tenant_id}/tier", json={"tier": tier}
                )
                assert put.status_code == 200
                assert put.json() == {"tenant_id": tenant_id, "tier": tier}

                got = await c.get(f"/admin/tenants/{tenant_id}/tier")
                assert got.status_code == 200
                assert got.json() == {"tenant_id": tenant_id, "tier": tier}
                assert read_tenant_tier(config_manager, tenant_id) == tier

    async def test_every_listed_tier_is_settable(self, client, config_manager):
        tenant_id = _seed_tenant()
        async with client as c:
            listed = await c.get("/admin/router-tiers")
            assert listed.status_code == 200
            assert listed.json() == {
                "tiers": sorted(ROUTER_TIERS),
                "default": DEFAULT_ROUTER_TIER,
            }
            for tier in listed.json()["tiers"]:
                response = await c.put(
                    f"/admin/tenants/{tenant_id}/tier", json={"tier": tier}
                )
                assert response.status_code == 200
                assert read_tenant_tier(config_manager, tenant_id) == tier

    async def test_the_simple_form_addresses_the_canonical_tenant(self, client):
        tenant_id = _seed_tenant()
        org_id = tenant_id.split(":")[0]
        simple = f"{org_id}:{org_id}"
        # A tenant whose org and name match is the simple form's canonical id.
        tm.get_backend().create_metadata_document(
            schema="tenant_metadata",
            doc_id=simple,
            fields={
                "tenant_full_id": simple,
                "org_id": org_id,
                "tenant_name": org_id,
                "created_at": CREATED_AT,
                "created_by": "tier-test",
                "status": "active",
                "schemas_deployed": SEEDED_SCHEMAS,
            },
        )
        async with client as c:
            put = await c.put(f"/admin/tenants/{org_id}/tier", json={"tier": "pro"})
            assert put.status_code == 200
            assert put.json() == {"tenant_id": simple, "tier": "pro"}
            got = await c.get(f"/admin/tenants/{simple}/tier")
            assert got.json() == {"tenant_id": simple, "tier": "pro"}


class TestRefusals:
    async def test_a_tier_outside_the_vocabulary_is_422_naming_the_set(self, client):
        tenant_id = _seed_tenant()
        async with client as c:
            response = await c.put(
                f"/admin/tenants/{tenant_id}/tier", json={"tier": "gold"}
            )
        assert response.status_code == 422
        assert response.json() == {
            "detail": (
                f"Unknown router tier 'gold'. Valid tiers: {sorted(ROUTER_TIERS)}"
            )
        }

    async def test_a_refused_tier_leaves_the_stored_tier_untouched(
        self, client, config_manager
    ):
        tenant_id = _seed_tenant()
        async with client as c:
            await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"})
            await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "gold"})
            got = await c.get(f"/admin/tenants/{tenant_id}/tier")
        assert got.json() == {"tenant_id": tenant_id, "tier": "pro"}
        assert read_tenant_tier(config_manager, tenant_id) == "pro"

    async def test_an_unknown_tenant_is_404_on_both_verbs(self, client):
        missing = f"nosuch{uuid.uuid4().hex[:8]}:production"
        async with client as c:
            got = await c.get(f"/admin/tenants/{missing}/tier")
            put = await c.put(f"/admin/tenants/{missing}/tier", json={"tier": "pro"})
        assert got.status_code == 404
        assert got.json() == {"detail": f"Tenant {missing} not found"}
        assert put.status_code == 404
        assert put.json() == {"detail": f"Tenant {missing} not found"}


class TestFaultContract:
    async def test_a_dead_config_store_is_503_not_a_default_tier(
        self, client, wired_tenant_manager
    ):
        tenant_id = _seed_tenant()
        dead = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://localhost", backend_port=DEAD_VESPA_PORT
            )
        )
        previous = tm._config_manager
        try:
            async with client as c:
                tm._config_manager = dead
                got = await c.get(f"/admin/tenants/{tenant_id}/tier")
        finally:
            tm._config_manager = previous
        assert got.status_code == 503
        assert got.json() == {"detail": "Tenant registry temporarily unavailable"}


class TestConcurrency:
    async def test_eight_concurrent_sets_on_distinct_tenants_land_on_their_own(
        self, client, config_manager
    ):
        """Eight PUTs in flight on one loop, one per tenant: each tenant ends on
        the tier its own request named, and every response says so."""
        import asyncio

        tiers = sorted(ROUTER_TIERS)
        tenants = [_seed_tenant() for _ in range(8)]
        expected = {t: tiers[i % len(tiers)] for i, t in enumerate(tenants)}

        async with client as c:
            puts = await asyncio.gather(
                *(
                    c.put(f"/admin/tenants/{t}/tier", json={"tier": expected[t]})
                    for t in tenants
                )
            )
            gets = await asyncio.gather(
                *(c.get(f"/admin/tenants/{t}/tier") for t in tenants)
            )

        assert [r.status_code for r in puts] == [200] * 8
        assert [r.json() for r in puts] == [
            {"tenant_id": t, "tier": expected[t]} for t in tenants
        ]
        assert [r.json() for r in gets] == [
            {"tenant_id": t, "tier": expected[t]} for t in tenants
        ]
        assert {t: read_tenant_tier(config_manager, t) for t in tenants} == expected


# Another runtime worker process: its own tier reader over the same config
# store, subscribed to the tier channel as main.py subscribes every worker.
# Each stdin line names tenants; it answers their tiers as it serves them now.
_OTHER_WORKER = """
import asyncio, sys
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.tenant_tiers import tenant_tier_reader
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.cluster_events import ClusterEvents
from cogniverse_vespa.config.config_store import VespaConfigStore

async def main(redis_url, channel, port):
    reader = tenant_tier_reader(ConfigManager(store=VespaConfigStore(
        backend_url="http://localhost", backend_port=int(port))))
    events = ClusterEvents(redis_url, "other-worker", {
        "tenant_tier_set": lambda p: tenant_manager.release_tenant_tier(p)},
        channel=channel)
    await events.start()
    print("ready", flush=True)
    loop = asyncio.get_running_loop()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        tenants = line.split()
        tiers = await asyncio.to_thread(lambda: [reader(t) for t in tenants])
        print(" ".join(tiers), flush=True)
    await events.close()

asyncio.run(main(*sys.argv[1:]))
"""


class _OtherWorker:
    def __init__(self, redis_url: str, channel: str, vespa_port: int):
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _OTHER_WORKER, redis_url, channel, str(vespa_port)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = self._proc.stdout.readline().strip()
        if ready != "ready":
            self.close()
            raise AssertionError(f"other worker did not subscribe: {ready!r}")

    async def tiers(self, *tenants: str) -> list[str]:
        def ask() -> list[str]:
            self._proc.stdin.write(" ".join(tenants) + "\n")
            self._proc.stdin.flush()
            return self._proc.stdout.readline().split()

        return await asyncio.to_thread(ask)

    def close(self) -> None:
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


@pytest.fixture
def other_worker(shared_state_redis_url, tier_channel, vespa_instance):
    worker = _OtherWorker(
        shared_state_redis_url, tier_channel, vespa_instance["http_port"]
    )
    yield worker
    worker.close()


class TestEveryWorkerServesTheSetTier:
    async def test_another_worker_holding_the_old_tier_serves_the_new_one_next(
        self, client, other_worker, config_manager
    ):
        """The other worker reads the tenant as free and holds it; a pro set
        answered by the route worker is what that worker serves next."""
        tenant_id = _seed_tenant()
        async with client as c:
            put = await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "free"})
            assert put.json() == {"tenant_id": tenant_id, "tier": "free"}
            assert await other_worker.tiers(tenant_id) == ["free"]

            put = await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"})

        assert put.status_code == 200
        assert put.json() == {"tenant_id": tenant_id, "tier": "pro"}
        assert await other_worker.tiers(tenant_id) == ["pro"]
        assert tenant_tier_reader(config_manager)(tenant_id) == "pro"

    async def test_every_tier_in_turn_is_served_by_the_other_worker_next(
        self, client, other_worker
    ):
        tenant_id = _seed_tenant()
        walk = sorted(ROUTER_TIERS)
        walk.append(walk[0])
        served = []
        async with client as c:
            for tier in walk:
                put = await c.put(
                    f"/admin/tenants/{tenant_id}/tier", json={"tier": tier}
                )
                assert put.status_code == 200
                served += await other_worker.tiers(tenant_id)
        assert served == walk


class TestTierSetConcurrency:
    async def test_eight_concurrent_sets_each_reach_the_other_worker(
        self, client, other_worker
    ):
        """The other worker holds eight tenants; eight sets in flight at once,
        each publishing its own event, leave it serving every new tier."""
        tiers = sorted(ROUTER_TIERS)
        tenants = [_seed_tenant() for _ in range(8)]
        before = {t: tiers[i % len(tiers)] for i, t in enumerate(tenants)}
        after = {t: tiers[(i + 1) % len(tiers)] for i, t in enumerate(tenants)}

        async with client as c:
            for t in tenants:
                assert (
                    await c.put(f"/admin/tenants/{t}/tier", json={"tier": before[t]})
                ).status_code == 200
            assert await other_worker.tiers(*tenants) == [before[t] for t in tenants]

            puts = await asyncio.gather(
                *(
                    c.put(f"/admin/tenants/{t}/tier", json={"tier": after[t]})
                    for t in tenants
                )
            )

        assert [r.status_code for r in puts] == [200] * 8
        assert [r.json() for r in puts] == [
            {"tenant_id": t, "tier": after[t]} for t in tenants
        ]
        assert await other_worker.tiers(*tenants) == [after[t] for t in tenants]

    async def test_a_read_in_flight_across_the_set_is_not_held(
        self, client, vespa_instance
    ):
        """A worker's store read that returned the old tier before the set but
        finishes after the worker dropped the tenant answers its own caller
        only: the worker's next read serves the set tier."""
        tenant_id = _seed_tenant()
        read_done = threading.Event()
        release = threading.Event()

        class GatedStore(VespaConfigStore):
            def get_config(self, *args, **kwargs):
                entry = super().get_config(*args, **kwargs)
                if not release.is_set():
                    read_done.set()
                    release.wait(30)
                return entry

        reader = tenant_tier_reader(
            ConfigManager(
                store=GatedStore(
                    backend_url="http://localhost",
                    backend_port=vespa_instance["http_port"],
                )
            )
        )
        async with client as c:
            put = await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "free"})
            assert put.status_code == 200
            in_flight = asyncio.create_task(asyncio.to_thread(reader, tenant_id))
            assert await asyncio.to_thread(read_done.wait, 30) is True
            put = await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"})
            release.set()
            first = await in_flight

        assert put.json() == {"tenant_id": tenant_id, "tier": "pro"}
        assert first == "free"
        assert await asyncio.to_thread(reader, tenant_id) == "pro"


class TestTierSetFaultContract:
    async def test_a_worker_that_does_not_confirm_fails_the_set_with_503(
        self, client, shared_state_redis_url, tier_channel, config_manager, monkeypatch
    ):
        """The tier is stored, the answering worker dropped it, and the caller
        is told that a worker did not, so the set can be retried."""
        tenant_id = _seed_tenant()
        monkeypatch.setattr(tm, "TENANT_TIER_ACK_TIMEOUT_S", 2.0)
        release = threading.Event()

        def stuck(payload):
            release.wait(30)
            return {}

        silent = ClusterEvents(
            shared_state_redis_url,
            "silent-worker",
            {"tenant_tier_set": stuck},
            channel=tier_channel,
        )
        await silent.start()
        held = tenant_tier_reader(config_manager)
        try:
            async with client as c:
                release.set()
                put = await c.put(
                    f"/admin/tenants/{tenant_id}/tier", json={"tier": "free"}
                )
                assert put.status_code == 200
                assert held(tenant_id) == "free"
                release.clear()
                put = await c.put(
                    f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"}
                )
        finally:
            release.set()
            await silent.close()

        assert put.status_code == 503
        assert put.json() == {
            "detail": (
                f"Tier pro is stored for {tenant_id}, but not every runtime worker "
                "dropped the tier it held; retry the set."
            )
        }
        assert read_tenant_tier(config_manager, tenant_id) == "pro"
        assert held(tenant_id) == "pro"

    async def test_an_unreachable_channel_fails_the_set_with_503(
        self, wired_tenant_manager, own_redis, config_manager, monkeypatch
    ):
        redis_url, pause, resume = own_redis
        events = await _route_worker(redis_url, _tier_channel())
        monkeypatch.setattr(tm, "_cluster_events", events)
        tenant_id = _seed_tenant()
        transport = httpx.ASGITransport(app=tm.app)
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://tier-test"
            ) as c:
                pause()
                try:
                    put = await c.put(
                        f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"}
                    )
                finally:
                    resume()
        finally:
            await events.close()

        assert put.status_code == 503
        assert put.json() == {
            "detail": (
                f"Tier pro is stored for {tenant_id}, but not every runtime worker "
                "dropped the tier it held; retry the set."
            )
        }
        assert read_tenant_tier(config_manager, tenant_id) == "pro"

    async def test_a_set_without_the_channel_wired_stores_nothing(
        self, wired_tenant_manager, config_manager, monkeypatch
    ):
        monkeypatch.setattr(tm, "_cluster_events", None)
        tenant_id = _seed_tenant()
        transport = httpx.ASGITransport(app=tm.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://tier-test"
        ) as c:
            with pytest.raises(RuntimeError) as raised:
                await c.put(f"/admin/tenants/{tenant_id}/tier", json={"tier": "pro"})
        assert str(raised.value) == (
            "Tenant tier sets need the cluster events channel wired"
        )
        assert read_tenant_tier(config_manager, tenant_id) == DEFAULT_ROUTER_TIER
