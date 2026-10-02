"""Partial admin config updates must survive a peer process's own update.

The pin-quota PUT applies only the fields the request names and merges them,
with compare-and-set, onto the tenant's stored record. A merge onto anything
but the record as stored erased a second process's PUT of a different field,
while both PUTs answered 200.

The peer here is a second *process* whatever the runtime's worker count: a
``kubectl exec`` python in the runtime pod that mounts the same admin router on
the pod's config store and drives the same route through it. The
authoritative read is a third process — this test — reading the stored record
from the cluster's Vespa config store.
"""

from __future__ import annotations

import json
import subprocess
import time

import httpx
import pytest

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.cluster import IN_POD_TELEMETRY_PRELUDE, KUBECTL_CONTEXT, RUNTIME
from tests.e2e.tenants import register_tenant_and_wait, unique_id

pytestmark = pytest.mark.e2e

NAMESPACE = "cogniverse"
RUNTIME_DEPLOYMENT = "deploy/cogniverse-runtime"
RUNTIME_CONTAINER = "runtime"
VESPA_HTTP_PORT = 33080

PEER_READY = "__PEER_READY__"
PEER_RESULT = "__PEER_RESULT__"

# One process, warmed before the window under test opens, that issues exactly
# one partial update when told to. It builds its config manager the way the
# runtime entrypoint does, so it talks to the same store the live runtime does.
_PEER_SOURCE = (
    IN_POD_TELEMETRY_PRELUDE
    + """
import asyncio, json, sys
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.routers import admin

admin.set_config_manager(create_default_config_manager())
app = FastAPI()
app.include_router(admin.router, prefix="/admin")


async def main():
    print({ready!r}, flush=True)
    sys.stdin.readline()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://peer"
    ) as client:
        response = await client.put({path!r}, json={body!r})
        payload = {{"status": response.status_code, "body": response.json()}}
    print({result!r} + json.dumps(payload), flush=True)


asyncio.run(main())
"""
)


class _PeerReplica:
    """A second runtime process holding one partial update until released."""

    def __init__(self, tenant_id: str, body: dict) -> None:
        source = _PEER_SOURCE.format(
            ready=PEER_READY,
            result=PEER_RESULT,
            path=f"/admin/tenants/{tenant_id}/pin_quotas",
            body=body,
        )
        self._proc = subprocess.Popen(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "-n",
                NAMESPACE,
                "exec",
                "-i",
                RUNTIME_DEPLOYMENT,
                "-c",
                RUNTIME_CONTAINER,
                "--",
                "python3",
                "-c",
                source,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait_ready(self, timeout_s: float = 300.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if line == "":
                break
            if line.strip() == PEER_READY:
                return
        self.kill()
        pytest.fail(
            "the peer runtime process never reported itself ready inside "
            f"{timeout_s:.0f}s; stderr={self._stderr()[:1000]!r}",
            pytrace=False,
        )

    def release_and_collect(self, timeout_s: float = 300.0) -> dict:
        self._proc.stdin.write("go\n")
        self._proc.stdin.flush()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if line == "":
                break
            if line.startswith(PEER_RESULT):
                payload = json.loads(line[len(PEER_RESULT) :])
                self._proc.wait(timeout=60)
                return payload
        self.kill()
        pytest.fail(
            "the peer runtime process never reported its update inside "
            f"{timeout_s:.0f}s; stderr={self._stderr()[:1000]!r}",
            pytrace=False,
        )

    def _stderr(self) -> str:
        try:
            return self._proc.stderr.read() or ""
        except Exception:
            return ""

    def kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


@pytest.fixture
def owned_quota_tenant():
    tenant_id = unique_id("prode2epipe") + ":t1"
    register_tenant_and_wait(tenant_id, created_by="e2e-test")
    return tenant_id


def _stored_quotas(tenant_id: str):
    """The tenant's pin-quota record as the cluster's config store holds it."""
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
    )
    try:
        entry = store.get_config(
            canonical_tenant_id(tenant_id),
            ConfigScope.SYSTEM,
            "admin_overrides",
            "pin_quotas",
        )
    finally:
        store.close()
    assert entry is not None, (
        f"no stored pin-quota record for {tenant_id!r}; the route answered 200 "
        "without storing anything"
    )
    return entry


def _put_quotas(client: httpx.Client, tenant_id: str, body: dict) -> dict:
    resp = client.put(f"/admin/tenants/{tenant_id}/pin_quotas", json=body)
    assert resp.status_code == 200, resp.text[:500]
    payload = resp.json()
    assert set(payload) == {"tenant_id", "quotas"}, payload
    return payload


def _read_quotas(client: httpx.Client, tenant_id: str) -> dict:
    resp = client.get(f"/admin/tenants/{tenant_id}/pin_quotas")
    assert resp.status_code == 200, resp.text[:500]
    return resp.json()


SEED_QUOTAS = {"user": 1, "tenant_admin": 2, "org_admin": -1}


class TestPartialUpdatesAcrossProcesses:
    def test_a_peer_process_field_survives_this_replicas_update(
        self, owned_quota_tenant
    ):
        """Two processes updating different fields inside one window leave both
        fields at their new values.

        Before the merge read the store, the field the peer had already
        accepted was replaced by this replica's stale copy of it: the
        document settled at ``{"user": 1, "tenant_admin": 9, "org_admin": -1}``
        where the first PUT had set ``user`` to 7.
        """
        tenant_id = owned_quota_tenant
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            _put_quotas(client, tenant_id, dict(SEED_QUOTAS))
            settled = _read_quotas(client, tenant_id)
            assert settled["quotas"] == SEED_QUOTAS, settled
            assert _stored_quotas(tenant_id).version == 1

            peer = _PeerReplica(tenant_id, {"tenant_admin": 9})
            try:
                peer.wait_ready()
                accepted = _put_quotas(client, tenant_id, {"user": 7})
                assert accepted["quotas"] == {**SEED_QUOTAS, "user": 7}, accepted
                peer_result = peer.release_and_collect()
            finally:
                peer.kill()

            assert peer_result["status"] == 200, peer_result
            assert peer_result["body"]["quotas"] == {
                "user": 7,
                "tenant_admin": 9,
                "org_admin": -1,
            }, peer_result
            served = _read_quotas(client, tenant_id)
            assert served["quotas"] == {"user": 7, "tenant_admin": 9, "org_admin": -1}

        stored = _stored_quotas(tenant_id)
        assert stored.config_value == {
            "user": 7,
            "tenant_admin": 9,
            "org_admin": -1,
        }
        assert stored.version == 3

    def test_an_update_merges_onto_the_store_not_a_held_copy(self, owned_quota_tenant):
        """An update issued after this replica served a copy of a field a peer
        has since changed keeps the peer's value, and the peer's value is what
        this replica serves at once.

        Before the merge read the store, this replica replayed its held
        ``user`` and the document settled at
        ``{"user": 1, "tenant_admin": 8, "org_admin": -1}``.
        """
        tenant_id = owned_quota_tenant
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            _put_quotas(client, tenant_id, dict(SEED_QUOTAS))

            peer = _PeerReplica(tenant_id, {"user": 5})
            try:
                peer.wait_ready()
                warmed = client.get(f"/admin/tenants/{tenant_id}/pin_quotas")
                assert warmed.status_code == 200, warmed.text[:500]
                assert warmed.json()["quotas"] == SEED_QUOTAS, warmed.json()
                peer_result = peer.release_and_collect()
            finally:
                peer.kill()
            assert peer_result["status"] == 200, peer_result

            served = _read_quotas(client, tenant_id)
            assert served["quotas"] == {**SEED_QUOTAS, "user": 5}, served
            accepted = _put_quotas(client, tenant_id, {"tenant_admin": 8})
            assert accepted["quotas"] == {"user": 5, "tenant_admin": 8, "org_admin": -1}

        stored = _stored_quotas(tenant_id)
        assert stored.config_value == {
            "user": 5,
            "tenant_admin": 8,
            "org_admin": -1,
        }
        assert stored.version == 3
