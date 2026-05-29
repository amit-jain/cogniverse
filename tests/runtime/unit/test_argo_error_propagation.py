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
    monkeypatch.setattr(tenant_router, "_argo_api_url", "http://argo.test")
    monkeypatch.setattr(tenant_router, "_argo_namespace", "argo")


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
