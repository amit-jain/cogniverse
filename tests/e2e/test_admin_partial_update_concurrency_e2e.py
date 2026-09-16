"""Partial admin config updates must survive a peer process's own update.

The pin-quota PUT applies only the fields the request names and merges them
onto the tenant's stored document. When the merge read the process-local TTL
cache instead of the store, a second replica's PUT of a different field was
erased by the first replica's persist, while both PUTs answered 200.

The e2e cluster runs one runtime replica, so the peer here is a second
*process*: a ``kubectl exec`` python in the runtime pod that mounts the same
admin router and drives the same route through it. The authoritative read is a
third process — this test — reading the durable blob through the shipped
artifact manager against the cluster's Phoenix.
"""

from __future__ import annotations

import json
import subprocess
import time

import httpx
import pytest

from cogniverse_agents.optimizer.artifact_manager import (
    _BLOB_RING_SLOTS,
    ArtifactManager,
)
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_runtime.routers.admin import (
    _PIN_QUOTA_BLOB_KEY,
    _PIN_QUOTA_BLOB_KIND,
    _PIN_QUOTA_CACHE_TTL_S,
)
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.e2e.conftest import (
    IN_POD_TELEMETRY_PRELUDE,
    KUBECTL_CONTEXT,
    PHOENIX_URL,
    RUNTIME,
    register_tenant_and_wait,
    run_async,
    unique_id,
)

pytestmark = pytest.mark.e2e

NAMESPACE = "cogniverse"
RUNTIME_DEPLOYMENT = "deploy/cogniverse-runtime"
RUNTIME_CONTAINER = "runtime"
PHOENIX_GRPC = "localhost:33317"

PEER_READY = "__PEER_READY__"
PEER_RESULT = "__PEER_RESULT__"

# One process, warmed before the window under test opens, that issues exactly
# one partial update when told to. It resolves the Phoenix endpoints the way
# the runtime entrypoint does, so it talks to the same store the live replica
# does.
_PEER_SOURCE = (
    IN_POD_TELEMETRY_PRELUDE
    + """
import asyncio, json, sys
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.routers import admin

system_config = create_default_config_manager().get_system_config()
admin.set_phoenix_endpoints(
    system_config.telemetry_url, system_config.telemetry_collector_endpoint
)
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
    await admin._blob_write_queue.flush()
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


def _artifact_manager(tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": PHOENIX_URL,
            "grpc_endpoint": PHOENIX_GRPC,
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


def _durable_quotas(tenant_id: str) -> dict:
    manager = _artifact_manager(canonical_tenant_id(tenant_id))
    raw = run_async(manager.load_blob(_PIN_QUOTA_BLOB_KIND, _PIN_QUOTA_BLOB_KEY))
    assert isinstance(raw, str), (
        f"no durable pin-quota blob for {tenant_id!r}; the route answered 200 "
        "without persisting anything"
    )
    return json.loads(raw)


def _committed_revision(tenant_id: str) -> int:
    """The serving revision the ring currently answers with."""
    manager = _artifact_manager(canonical_tenant_id(tenant_id))
    revisions = []
    for slot in range(_BLOB_RING_SLOTS):
        record = run_async(
            manager._read_blob_slot(_PIN_QUOTA_BLOB_KIND, _PIN_QUOTA_BLOB_KEY, slot)
        )
        if record is not None:
            revisions.append(record["revision"])
    assert revisions, (
        f"the pin-quota ring holds no revision for {tenant_id!r} after a PUT "
        "that answered 200"
    )
    return max(revisions)


def _put_quotas(client: httpx.Client, tenant_id: str, body: dict) -> dict:
    resp = client.put(f"/admin/tenants/{tenant_id}/pin_quotas", json=body)
    assert resp.status_code == 200, resp.text[:500]
    payload = resp.json()
    assert payload["pending_write"] is True, payload
    return payload


def _settle(client: httpx.Client, tenant_id: str, timeout_s: float = 180.0) -> dict:
    """Block until the live replica reports its accepted write persisted."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/admin/tenants/{tenant_id}/pin_quotas")
        assert resp.status_code == 200, resp.text[:500]
        payload = resp.json()
        if payload["pending_write"] is False:
            return payload
        time.sleep(0.5)
    pytest.fail(
        f"the pin-quota write for {tenant_id!r} never landed within {timeout_s:.0f}s",
        pytrace=False,
    )


SEED_QUOTAS = {"user": 1, "tenant_admin": 2, "org_admin": -1}


class TestPartialUpdatesAcrossProcesses:
    def test_a_peer_process_field_survives_this_replicas_update(
        self, owned_quota_tenant
    ):
        """Two processes updating different fields inside one write-behind
        window leave both fields at their new values.

        Before the merge read the store, the field the peer had already
        accepted was replaced by this replica's stale copy of it: the
        document settled at ``{"user": 1, "tenant_admin": 9, "org_admin": -1}``
        where the first PUT had set ``user`` to 7.
        """
        tenant_id = owned_quota_tenant
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            _put_quotas(client, tenant_id, dict(SEED_QUOTAS))
            settled = _settle(client, tenant_id)
            assert settled["quotas"] == SEED_QUOTAS, settled
            assert _committed_revision(tenant_id) == 1

            peer = _PeerReplica(tenant_id, {"tenant_admin": 9})
            try:
                peer.wait_ready()
                accepted = _put_quotas(client, tenant_id, {"user": 7})
                assert accepted["quotas"] == {**SEED_QUOTAS, "user": 7}, accepted
                peer_result = peer.release_and_collect()
            finally:
                peer.kill()

            assert peer_result["status"] == 200, peer_result
            assert peer_result["body"]["pending_write"] is True, peer_result
            _settle(client, tenant_id)

        assert _durable_quotas(tenant_id) == {
            "user": 7,
            "tenant_admin": 9,
            "org_admin": -1,
        }
        assert _committed_revision(tenant_id) == 3

    def test_an_update_merges_onto_the_store_not_this_replicas_cache(
        self, owned_quota_tenant
    ):
        """An update issued while this replica holds a warm cached copy of a
        field a peer has since changed keeps the peer's value.

        Before the merge skipped the cache, this replica replayed its cached
        ``user`` and the document settled at
        ``{"user": 1, "tenant_admin": 8, "org_admin": -1}``.
        """
        tenant_id = owned_quota_tenant
        with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
            _put_quotas(client, tenant_id, dict(SEED_QUOTAS))
            _settle(client, tenant_id)

            peer = _PeerReplica(tenant_id, {"user": 5})
            try:
                peer.wait_ready()
                warmed = client.get(f"/admin/tenants/{tenant_id}/pin_quotas")
                assert warmed.status_code == 200, warmed.text[:500]
                assert warmed.json()["quotas"] == SEED_QUOTAS, warmed.json()
                warmed_at = time.monotonic()
                peer_result = peer.release_and_collect()
            finally:
                peer.kill()
            assert peer_result["status"] == 200, peer_result

            accepted = _put_quotas(client, tenant_id, {"tenant_admin": 8})
            cache_age = time.monotonic() - warmed_at
            assert cache_age < _PIN_QUOTA_CACHE_TTL_S, (
                f"the peer's update took {cache_age:.1f}s, past the "
                f"{_PIN_QUOTA_CACHE_TTL_S}s cache window this test exists to "
                "cross, so a stale merge would not be observable here"
            )
            assert accepted["quotas"] == {"user": 5, "tenant_admin": 8, "org_admin": -1}
            _settle(client, tenant_id)

        assert _durable_quotas(tenant_id) == {
            "user": 5,
            "tenant_admin": 8,
            "org_admin": -1,
        }
        assert _committed_revision(tenant_id) == 3
