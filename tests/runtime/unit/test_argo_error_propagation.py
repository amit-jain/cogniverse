"""Argo CronWorkflow submit/delete must surface failure to the caller.

The pre-fix helpers ate every error (network, 4xx, 5xx) and logged. The
``create_job`` route then returned 200 with status="created" and a
persisted ConfigStore row even when the cluster had rejected the
manifest — the schedule never fired but the user saw "created", and
``delete_job`` happily tombstoned the row even though the schedule kept
firing on the cluster.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import cogniverse_runtime.routers.tenant as tenant_router


class _FakeAsyncClient:
    """Minimal stand-in for ``httpx.AsyncClient`` covering ``.post``/``.delete``.

    Configurable per call to return either a status code + body, or to raise
    a connection error. Awaitable ``__aenter__`` / ``__aexit__`` so the
    ``async with`` block in the SUT works.
    """

    # Faithful httpx surface: the shared-client cache checks ``is_closed``
    # before reuse.
    is_closed = False

    def __init__(self, *, post_status=None, delete_status=None, raise_on_call=False):
        self._post_status = post_status
        self._delete_status = delete_status
        self._raise = raise_on_call

    def __call__(self, *_, **__):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, *args, **kwargs):
        if self._raise:
            raise ConnectionError("argo unreachable")
        return _FakeResponse(self._post_status, "rejected by argo")

    async def delete(self, *args, **kwargs):
        if self._raise:
            raise ConnectionError("argo unreachable")
        return _FakeResponse(self._delete_status, "rejected by argo")


class _FakeResponse:
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.text = body


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, fake: _FakeAsyncClient) -> None:
    import httpx as _httpx

    monkeypatch.setattr(_httpx, "AsyncClient", fake)


def _set_argo_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from cogniverse_runtime.config_loader import WorkflowSettings

    settings = WorkflowSettings(
        api_url="http://argo.test",
        namespace="argo",
        job_template="cogniverse-job-runner",
        optimization_template="cogniverse-optimization-runner",
    )
    monkeypatch.setattr(tenant_router, "get_workflow_settings", lambda: settings)


def _build_manifest() -> dict:
    return tenant_router._build_cron_workflow(
        tenant_id="acme",
        job_id="abc12345",
        schedule="0 * * * *",
        namespace="argo",
    )


@pytest.mark.asyncio
async def test_submit_cron_workflow_raises_on_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argo_endpoint(monkeypatch)
    _patch_httpx(monkeypatch, _FakeAsyncClient(post_status=422))
    manifest = _build_manifest()
    with pytest.raises(HTTPException) as excinfo:
        await tenant_router._submit_cron_workflow(manifest)
    assert excinfo.value.status_code == 503
    assert "Argo rejected" in excinfo.value.detail


@pytest.mark.asyncio
async def test_submit_cron_workflow_raises_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argo_endpoint(monkeypatch)
    _patch_httpx(monkeypatch, _FakeAsyncClient(raise_on_call=True))
    manifest = _build_manifest()
    with pytest.raises(HTTPException) as excinfo:
        await tenant_router._submit_cron_workflow(manifest)
    assert excinfo.value.status_code == 503
    assert "Argo unreachable" in excinfo.value.detail


@pytest.mark.asyncio
async def test_submit_cron_workflow_succeeds_on_201(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argo_endpoint(monkeypatch)
    _patch_httpx(monkeypatch, _FakeAsyncClient(post_status=201))
    manifest = _build_manifest()
    # Must not raise.
    await tenant_router._submit_cron_workflow(manifest)


@pytest.mark.asyncio
async def test_delete_cron_workflow_raises_on_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_argo_endpoint(monkeypatch)
    _patch_httpx(monkeypatch, _FakeAsyncClient(delete_status=500))
    with pytest.raises(HTTPException) as excinfo:
        await tenant_router._delete_cron_workflow("tenant-job-acme-abc", "argo")
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_delete_cron_workflow_404_is_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """404 means the workflow is already gone — the desired end state."""
    _set_argo_endpoint(monkeypatch)
    _patch_httpx(monkeypatch, _FakeAsyncClient(delete_status=404))
    # Must not raise.
    await tenant_router._delete_cron_workflow("tenant-job-acme-abc", "argo")


def test_shared_argo_client_never_crosses_loops(monkeypatch) -> None:
    """A new event loop must never inherit another loop's cached client —
    an ``id(loop)``-keyed cache collides when a torn-down loop's id is
    reused, handing out a client bound to a dead loop (in the sweep, a
    previous test's fake)."""
    import asyncio
    import gc

    created = []

    class _TrackingFake(_FakeAsyncClient):
        def __call__(self, *args, **kwargs):
            instance = _FakeAsyncClient(delete_status=404)
            created.append(instance)
            return instance

    _set_argo_endpoint(monkeypatch)
    _patch_httpx(monkeypatch, _TrackingFake())

    async def _get():
        return await tenant_router._shared_argo_client()

    first = asyncio.run(_get())
    gc.collect()  # tear down the first loop so its id can be reused
    second = asyncio.run(_get())

    assert first is not second
    assert len(created) == 2


class TestArgoClientAcceptsTheInClusterCertificate:
    """The shared client must reach argo-server's self-signed endpoint.

    argo-server runs in secure mode behind a self-signed in-cluster
    certificate. A default-verifying client raises
    ``CERTIFICATE_VERIFY_FAILED`` on connect, ``_submit_cron_workflow`` maps
    that to 503, and no scheduled job can ever be created. This drives a real
    local HTTPS server holding a self-signed cert — no mock — so the client's
    TLS posture is exercised rather than asserted from a flag.
    """

    @staticmethod
    def _self_signed_server():
        import http.server
        import ssl
        import tempfile
        import threading
        from pathlib import Path

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "argo-server.argo.svc")]
        )
        import datetime as _dt

        now = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(days=1))
            .not_valid_after(now + _dt.timedelta(days=3650))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]), False
            )
            .sign(key, hashes.SHA256())
        )
        tmp = Path(tempfile.mkdtemp())
        (tmp / "c.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        (tmp / "k.pem").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )

        class _H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"metadata":{"name":"ok"}}')

            def log_message(self, *a):  # keep pytest output clean
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(tmp / "c.pem", tmp / "k.pem")
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_post_to_a_self_signed_https_endpoint_succeeds(self):
        import asyncio

        srv = self._self_signed_server()
        port = srv.server_address[1]
        try:

            async def _post():
                client = await tenant_router._shared_argo_client()
                return await client.post(
                    f"https://127.0.0.1:{port}/api/v1/cron-workflows/cogniverse",
                    json={"namespace": "cogniverse"},
                )

            response = asyncio.run(_post())
        finally:
            srv.shutdown()

        assert response.status_code == 200
        assert response.json() == {"metadata": {"name": "ok"}}
