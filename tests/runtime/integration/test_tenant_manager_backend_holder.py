"""Tenant reads survive the registry closing the backend they resolved.

The runtime resolves a system backend at startup and then clears the
registry cache when deployment env overrides change SystemConfig
(main.py, step 7) — which closes every cached instance. A module holding
that handle across requests answers BackendClosedError, and every
tenant-scoped request 503s for the pod's lifetime. tenant_manager
re-resolves through the registry per call and holds a checkout for the
operation so nothing closes under it mid-write.
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
import uuid

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.admin.models import CreateOrganizationRequest
from cogniverse_sdk.interfaces.backend import BackendClosedError

pytestmark = pytest.mark.integration

# Nothing listens here; the CI-parity default for a dead backend port.
DEAD_VESPA_PORT = 29071

# A warm resolve is a SystemConfig read plus a dict lookup in the registry's
# LRU. Measured over 1000 calls against the shared Vespa with the k3d cluster
# resident: median 415us, p95 451us, max 714us. The budget is 2ms — under a
# Vespa round trip (~2-5ms) and far under a rebuild (~50ms+), so a resolve
# that starts round-tripping or rebuilding fails here.
RESOLVE_BUDGET_S = 0.002

CREATED_AT = 1757000000000
SEEDED_SCHEMAS = ["video_colpali_smol500_mv_frame"]


@pytest.fixture
def wired_tenant_manager(config_manager, schema_loader):
    """tenant_manager wired to the test Vespa, module seams restored after."""
    previous_config_manager = tm._config_manager
    previous_schema_loader = tm._schema_loader
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(schema_loader)
    yield tm
    tm.set_config_manager(previous_config_manager)
    tm.set_schema_loader(previous_schema_loader)
    # Leave no closed instance behind for the next module.
    BackendRegistry.get_instance().clear_instances()


def _seed_org_and_tenant(org_name="Holder Org", created_by="holder-test"):
    """Create the org through the real route and PUT one tenant row."""
    org_id = f"holder{uuid.uuid4().hex[:8]}"
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
            "created_by": created_by,
            "status": "active",
            "schemas_deployed": SEEDED_SCHEMAS,
        },
    )
    assert stored is True
    return org_id, tenant_id, backend


def _tenant_row(tenant):
    return (
        tenant.tenant_full_id,
        tenant.org_id,
        tenant.tenant_name,
        tenant.created_at,
        tenant.created_by,
        tenant.status,
        tenant.schemas_deployed,
    )


def _expected_row(org_id, tenant_id, created_by="holder-test"):
    return (
        tenant_id,
        org_id,
        "production",
        CREATED_AT,
        created_by,
        "active",
        SEEDED_SCHEMAS,
    )


@pytest.mark.asyncio
async def test_tenant_read_serves_after_the_registry_closes_that_instance(
    wired_tenant_manager,
):
    """The production sequence: resolve, registry clear (which closes), read."""
    org_id, tenant_id, resolved = _seed_org_and_tenant()

    created = await tm.create_organization(
        CreateOrganizationRequest(
            org_id=org_id, org_name="Holder Org", created_by="holder-test"
        )
    )
    assert (
        created.org_id,
        created.org_name,
        created.created_by,
        created.status,
        created.tenant_count,
    ) == (org_id, "Holder Org", "holder-test", "active", 0)

    BackendRegistry.get_instance().clear_instances()

    # The instance the module resolved is genuinely dead now: this is the
    # exact error the runtime pod served on every tenant-scoped request.
    with pytest.raises(BackendClosedError) as closed:
        resolved.get_metadata_document(schema="tenant_metadata", doc_id=tenant_id)
    assert "is closed" in str(closed.value)

    tenant = await tm.get_tenant_internal(tenant_id)
    assert _tenant_row(tenant) == _expected_row(org_id, tenant_id)

    organization = await tm.get_organization_internal(org_id)
    assert (
        organization.org_id,
        organization.org_name,
        organization.created_by,
        organization.status,
    ) == (org_id, "Holder Org", "holder-test", "active")

    # Resolution went back to the registry rather than to a held handle.
    assert tm.get_backend() is not resolved


@pytest.mark.asyncio
async def test_a_read_in_flight_is_closed_only_after_it_releases(
    wired_tenant_manager, monkeypatch
):
    """Barrier-executed interleaving: clear() lands mid-read.

    The checkout defers the close, so the read finishes on a live pool and
    the instance closes exactly once, after the release.
    """
    org_id, tenant_id, backend = _seed_org_and_tenant()

    closed_instances: list[int] = []
    real_close = type(backend).close

    def counting_close(self):
        closed_instances.append(id(self))
        return real_close(self)

    monkeypatch.setattr(type(backend), "close", counting_close, raising=True)

    read_entered = threading.Event()
    clear_returned = threading.Event()
    real_get = backend.get_metadata_document
    closes_seen_during_read: list[int] = []

    def barrier_get(**kwargs):
        read_entered.set()
        assert clear_returned.wait(30), "the evictor never ran"
        return real_get(**kwargs)

    monkeypatch.setattr(backend, "get_metadata_document", barrier_get, raising=True)

    def evict():
        assert read_entered.wait(30), "the read never reached the backend"
        BackendRegistry.get_instance().clear_instances()
        closes_seen_during_read.extend(closed_instances)
        clear_returned.set()

    evictor = threading.Thread(target=evict, name="evictor")
    evictor.start()
    try:
        tenant = await tm.get_tenant_internal(tenant_id)
    finally:
        evictor.join(30)
        assert evictor.is_alive() is False

    assert _tenant_row(tenant) == _expected_row(org_id, tenant_id)
    # Nothing closed while the read held its checkout...
    assert closes_seen_during_read == []
    # ...and the deferred close ran exactly once on release.
    assert closed_instances == [id(backend)]


@pytest.mark.asyncio
async def test_a_dead_backend_is_a_503_outage_not_a_missing_tenant(
    wired_tenant_manager, schema_loader, caplog
):
    """Vespa genuinely down: 503 naming the outage, never 404, never closed."""
    org_id, tenant_id, _ = _seed_org_and_tenant()
    BackendRegistry.get_instance().clear_instances()

    dead = ConfigManager(store=wired_tenant_manager._config_manager.store)
    dead.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=DEAD_VESPA_PORT)
    )
    tm.set_config_manager(dead)

    with caplog.at_level("ERROR"):
        with pytest.raises(tm.HTTPException) as exc:
            await tm.get_tenant_internal(tenant_id)

    assert exc.value.status_code == 503
    assert exc.value.detail == "Tenant registry temporarily unavailable"
    # The cause names the unreachable endpoint, not a released instance.
    assert str(DEAD_VESPA_PORT) in caplog.text
    assert "is closed" not in caplog.text


def test_resolving_the_backend_per_call_stays_within_its_measured_budget(
    wired_tenant_manager,
):
    """A warm resolve is the registry LRU hit, not a rebuild.

    The measured median is printed; the budget above it is tight enough that
    a resolve which rebuilds or round-trips instead of hitting the LRU fails.
    """
    tm.get_backend()

    samples: list[float] = []
    for _ in range(1000):
        started = time.perf_counter()
        tm.get_backend()
        samples.append(time.perf_counter() - started)

    median = statistics.median(samples)
    print(
        f"get_backend(): median={median * 1e6:.1f}us "
        f"p95={statistics.quantiles(samples, n=20)[-1] * 1e6:.1f}us "
        f"max={max(samples) * 1e6:.1f}us"
    )
    assert median < RESOLVE_BUDGET_S
    # One instance for the whole run: resolving per call must not rebuild.
    assert tm.get_backend() is tm.get_backend()


@pytest.mark.asyncio
async def test_concurrent_tenant_reads_share_one_instance_and_all_serve(
    wired_tenant_manager,
):
    """N concurrent reads: one cached instance, every read the exact row."""
    org_id, tenant_id, backend = _seed_org_and_tenant()

    results = await asyncio.gather(
        *[tm.get_tenant_internal(tenant_id) for _ in range(12)]
    )

    assert [_tenant_row(row) for row in results] == [
        _expected_row(org_id, tenant_id)
    ] * 12
    assert tm.get_backend() is backend


@pytest.mark.asyncio
async def test_the_shared_schema_registry_rebinds_after_a_registry_clear(
    wired_tenant_manager,
):
    """The process-wide SchemaRegistry must not keep a closed backend.

    BackendFactory builds one SchemaRegistry around the first backend and the
    registry shares it with every later instance. A clear closes that backend
    while the shared registry keeps deploying through it, so every tenant
    create after a clear would fail on a released client.
    """
    _, tenant_id, first = _seed_org_and_tenant()
    stale_schema_registry = first.schema_registry
    assert stale_schema_registry is BackendRegistry._shared_schema_registry

    BackendRegistry.get_instance().clear_instances()

    # What the stale registry would deploy through.
    with pytest.raises(BackendClosedError):
        stale_schema_registry._backend.get_metadata_document(
            schema="tenant_metadata", doc_id=tenant_id
        )

    rebuilt = tm.get_backend()
    assert rebuilt is not first
    assert rebuilt.schema_registry is not stale_schema_registry
    assert rebuilt.schema_registry._backend is rebuilt
    assert BackendRegistry._shared_schema_registry is rebuilt.schema_registry
    # The rebound registry's backend serves.
    assert (
        rebuilt.get_metadata_document(schema="tenant_metadata", doc_id=tenant_id)[
            "tenant_full_id"
        ]
        == tenant_id
    )


def test_resolving_without_an_injected_config_manager_is_not_a_binding_conflict(
    schema_loader,
):
    """A standalone process injects no ConfigManager (optimization_cli.py:3040).

    It then gets the module's own, built once: a fresh ConfigManager per
    resolve would build a VespaConfigStore per tenant read, and it is the
    binding the registry compares a cache hit against
    (backend_registry.py:226).
    """
    import cogniverse_runtime.admin.tenant_manager as tm_module

    previous_config_manager = tm_module._config_manager
    previous_schema_loader = tm_module._schema_loader
    previous_fallback = tm_module._fallback_config_manager
    tm.set_config_manager(None)
    tm.set_schema_loader(schema_loader)
    tm_module._fallback_config_manager = None
    BackendRegistry.get_instance().clear_instances()
    try:
        first = tm.get_backend()
        built_once = tm_module._fallback_config_manager
        second = tm.get_backend()

        assert second is first
        assert tm_module._fallback_config_manager is built_once
        assert tm._default_config_manager() is built_once
        with tm.metadata_backend() as leased:
            assert leased is first
    finally:
        tm_module._fallback_config_manager = previous_fallback
        tm.set_config_manager(previous_config_manager)
        tm.set_schema_loader(previous_schema_loader)
        BackendRegistry.get_instance().clear_instances()
