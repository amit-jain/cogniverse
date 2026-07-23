"""In-process coverage of POST /ingestion/upload's success envelope.

The success branch (the 202-style async envelope, the wait=true synchronous
result, and the force=true idempotency bypass) had assertions only in
docker-backed integration + e2e; a no-Docker ci_fast run never pinned it —
the exact surface where a force/idempotency regression once shipped. These
mount the router with minio_client.upload_bytes + enqueue_ingestion stubbed
and assert the response body plus that wait/force reach enqueue_ingestion.

wait/wait_timeout/force are FastAPI Query params: they MUST be sent in the
query string. Posting them as multipart form fields silently applies the
defaults (the footgun the audit flagged) — the last test pins that contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.ingestion_worker import minio_client, submit_api
from cogniverse_runtime.ingestion_worker import redis_client as redis_client_mod
from cogniverse_runtime.routers import ingestion as ingestion_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_SOURCE_URL = "s3://bucket/acme/v.mp4"


@pytest.fixture
def upload_client(monkeypatch):
    """Router mounted with a configured backend and stubbed minio/redis/queue.

    ``captured`` records the kwargs the route passes to enqueue_ingestion, and
    ``enqueue_result`` is the result the stub returns (a test sets its
    final_event to exercise the wait=true synchronous branch).
    """
    app = FastAPI()
    app.include_router(ingestion_router.router, prefix="/ingestion")

    sys_cfg = SimpleNamespace(
        search_backend="vespa",
        redis_url="redis://stub",
        minio_endpoint="minio://stub",
    )
    cm = MagicMock()
    cm.get_system_config.return_value = sys_cfg
    app.dependency_overrides[ingestion_router.get_config_manager_dependency] = lambda: (
        cm
    )

    async def _tenant_ok(tenant_id: str) -> None:
        return None

    monkeypatch.setattr(ingestion_router, "assert_tenant_exists", _tenant_ok)
    monkeypatch.setattr(
        minio_client, "upload_bytes", lambda *a, **k: _SOURCE_URL, raising=True
    )

    async def _get_redis(url):
        return MagicMock()

    monkeypatch.setattr(redis_client_mod, "get_redis", _get_redis, raising=True)

    captured: dict = {}
    state = {
        "result": SimpleNamespace(
            ingest_id="ing-1",
            sha="sha-abc",
            state="queued",
            existing=False,
            final_event=None,
        )
    }

    async def _enqueue(redis, **kwargs):
        captured.update(kwargs)
        return state["result"]

    monkeypatch.setattr(submit_api, "enqueue_ingestion", _enqueue, raising=True)

    with TestClient(app) as client:
        yield client, captured, state


def _post(client, query="", data=None):
    return client.post(
        f"/ingestion/upload{query}",
        files={"file": ("v.mp4", b"video-bytes", "video/mp4")},
        data={"tenant_id": "acme:acme", **(data or {})},
    )


def test_async_default_returns_queued_envelope(upload_client):
    client, captured, _ = upload_client

    resp = _post(client)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ingest_id"] == "ing-1"
    assert body["sha"] == "sha-abc"
    assert body["state"] == "queued"
    assert body["existing"] is False
    assert body["status"] == "queued"
    assert body["source_url"] == _SOURCE_URL
    assert body["filename"] == "v.mp4"
    # Defaults reached the queue: not a synchronous wait, no idempotency bypass.
    assert captured["wait"] is False
    assert captured["force"] is False
    assert captured["wait_timeout"] == 300


def test_wait_and_force_query_params_drive_synchronous_success(upload_client):
    client, captured, state = upload_client
    state["result"] = SimpleNamespace(
        ingest_id="ing-2",
        sha="sha-xyz",
        state="complete",
        existing=False,
        final_event={
            "result": {
                "video_id": "vid-9",
                "chunks": 3,
                "documents_fed": 5,
                "graph_nodes": 2,
                "graph_edges": 1,
            }
        },
    )

    resp = _post(client, query="?wait=true&force=true")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert body["video_id"] == "vid-9"
    assert body["chunks_created"] == 3
    assert body["documents_fed"] == 5
    assert body["graph_nodes"] == 2
    assert body["graph_edges"] == 1
    # The query params were honored, not the defaults.
    assert captured["wait"] is True
    assert captured["force"] is True


def test_wait_and_force_as_form_fields_are_ignored(upload_client):
    """Contract guard: wait/force are Query params. Posting them as form fields
    applies the defaults (the route runs async, force off) — send them in the
    query string or they silently do nothing."""
    client, captured, _ = upload_client

    resp = _post(client, data={"wait": "true", "force": "true"})

    assert resp.status_code == 200, resp.text
    assert captured["wait"] is False
    assert captured["force"] is False


def test_minio_outage_returns_503_not_500(upload_client, monkeypatch):
    """A MinIO/S3 outage mid-transfer (botocore ClientError/connection error)
    is a transient backend fault, not a server bug — the route must return 503,
    not the opaque 500 the default handler gives for a non-RuntimeError."""
    from botocore.exceptions import EndpointConnectionError

    client, captured, _ = upload_client

    def _boom(*a, **k):
        raise EndpointConnectionError(endpoint_url="minio://stub")

    monkeypatch.setattr(minio_client, "upload_bytes", _boom, raising=True)

    resp = _post(client)

    assert resp.status_code == 503, resp.text
    assert "object store" in resp.json()["detail"]["message"]
    # The object never made it to the queue.
    assert captured == {}


def test_redis_outage_during_enqueue_returns_503(upload_client, monkeypatch):
    """The object uploaded but Redis is unreachable while enqueueing — a
    retryable outage, not a 500. A retry re-enqueues idempotently."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    client, _captured, _ = upload_client

    async def _boom(redis, **kwargs):
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr(submit_api, "enqueue_ingestion", _boom, raising=True)

    resp = _post(client)

    assert resp.status_code == 503, resp.text
    assert "queue" in resp.json()["detail"]["message"]


def test_empty_profile_rejected_before_any_upload(upload_client, monkeypatch):
    """An empty profile used to fail only at idempotency hashing — AFTER the
    whole multipart body had been copied into the object store — and as a
    500. It must be a 400 before a single byte is transferred."""
    client, captured, _state = upload_client

    uploads: list = []
    monkeypatch.setattr(
        minio_client,
        "upload_bytes",
        lambda *a, **k: uploads.append(a) or _SOURCE_URL,
        raising=True,
    )

    resp = _post(client, data={"profile": "   "})

    assert resp.status_code == 400, resp.text
    assert "profile" in resp.json()["detail"]
    assert uploads == []
    assert captured == {}


def test_org_id_combined_with_simple_tenant(upload_client):
    """A separately-supplied org_id must combine with a simple tenant_id into
    the canonical org:tenant form, not be silently dropped (which routed the
    upload into the wrong namespace)."""
    client, captured, _ = upload_client

    resp = client.post(
        "/ingestion/upload",
        files={"file": ("v.mp4", b"video-bytes", "video/mp4")},
        data={"tenant_id": "research", "org_id": "acme"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["tenant_id"] == "acme:research"


def test_org_id_ignored_when_tenant_already_qualified(upload_client):
    """An org_id is ignored when the tenant_id is already org:tenant form."""
    client, captured, _ = upload_client

    resp = client.post(
        "/ingestion/upload",
        files={"file": ("v.mp4", b"video-bytes", "video/mp4")},
        data={"tenant_id": "acme:research", "org_id": "globex"},
    )

    assert resp.status_code == 200, resp.text
    assert captured["tenant_id"] == "acme:research"
