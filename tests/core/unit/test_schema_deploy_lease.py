"""The deployment lease admits one writer and refuses an expired one.

The lease is what keeps two processes from each replacing the whole Vespa
application package from its own snapshot. These pin the state machine over a
shared store; the two-process behaviour against real Vespa is pinned by
tests/backends/integration/test_schema_deployment_serialization.py.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_core.registries import schema_deploy_lease
from cogniverse_core.registries.schema_deploy_lease import (
    DeploymentLeaseLost,
    SchemaDeployLease,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _lease(store, **kwargs):
    return SchemaDeployLease(store, **kwargs)


def test_only_one_of_two_concurrent_holders_takes_the_lease():
    store = InMemoryConfigStore()
    first, second = _lease(store, wait_seconds=0), _lease(store, wait_seconds=0)
    ready = threading.Barrier(2)

    def acquire(lease):
        ready.wait(timeout=10)
        try:
            lease.acquire()
            return lease.holder
        except TimeoutError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(acquire, [first, second]))

    winners = [
        outcome for outcome in outcomes if outcome in (first.holder, second.holder)
    ]
    assert len(winners) == 1
    record = store.get_config(
        tenant_id="__system__",
        scope=schema_deploy_lease.ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    assert record.config_value["holder"] == winners[0]
    losers = [outcome for outcome in outcomes if outcome not in winners]
    assert losers == [
        f"Vespa deployment lease still held by {winners[0]!r} after 0s; refusing "
        f"to replace the application package concurrently with another deployer"
    ]


def test_release_hands_the_lease_to_the_next_holder():
    store = InMemoryConfigStore()
    first, second = _lease(store, wait_seconds=0), _lease(store, wait_seconds=0)
    first.acquire()
    with pytest.raises(TimeoutError):
        second.acquire()
    first.release()
    assert second.acquire() is second
    assert second.renew() is None


def test_expired_holder_is_replaced_and_then_refused():
    store = InMemoryConfigStore()
    expiring = _lease(store, lease_seconds=0.0, wait_seconds=0)
    expiring.acquire()

    successor = _lease(store, wait_seconds=0)
    assert successor.acquire() is successor

    with pytest.raises(DeploymentLeaseLost) as caught:
        expiring.renew()
    assert str(caught.value) == "Vespa deployment lease expired or was replaced"

    # The refused holder must not take the lease away on its way out.
    expiring.release()
    record = store.get_config(
        tenant_id="__system__",
        scope=schema_deploy_lease.ConfigScope.SCHEMA,
        service="schema_deploy_lease",
        config_key="application",
    )
    assert record.config_value["holder"] == successor.holder


def test_a_store_outage_fails_the_acquire_instead_of_granting_it():
    class _Down(InMemoryConfigStore):
        def get_config(self, *args, **kwargs):
            raise ConnectionError("config store unreachable")

    with pytest.raises(ConnectionError, match="config store unreachable"):
        _lease(_Down(), wait_seconds=0).acquire()
