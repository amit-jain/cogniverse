"""Tenant/org registry reads distinguish a backend outage from 'not found'.

get_tenant_internal / get_organization_internal caught every exception and
returned None, so a Vespa outage read as 'tenant not found' → assert_tenant_exists
404'd every search/ingestion/graph request during a blip (a permanent-looking
'tenant deleted'), and create_tenant/create_organization read a live record as
missing and could clobber it. A backend failure must now surface 503; a genuine
miss still returns None.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.asyncio
async def test_get_tenant_internal_raises_503_on_outage(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.get_metadata_document.side_effect = RuntimeError("Vespa unreachable")
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    with pytest.raises(HTTPException) as exc:
        await tm.get_tenant_internal("acme:prod")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_get_tenant_internal_returns_none_on_genuine_not_found(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.get_metadata_document.return_value = None  # genuine 404
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    assert await tm.get_tenant_internal("acme:prod") is None


@pytest.mark.asyncio
async def test_get_organization_internal_raises_503_on_outage(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.get_metadata_document.side_effect = RuntimeError("Vespa unreachable")
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    with pytest.raises(HTTPException) as exc:
        await tm.get_organization_internal("acme")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_assert_tenant_exists_surfaces_503_not_404_on_outage(monkeypatch):
    """The whole point: the guard on every tenant-scoped request must return a
    retryable 503 on outage, not a 404 that reads as 'tenant deregistered'."""
    from cogniverse_core.common import tenant_utils

    # Fresh tenant id so the positive cache doesn't short-circuit.
    tid = "outagetest:outagetest"
    tenant_utils._TENANT_EXISTS_CACHE.pop(tid, None)

    async def _raise_503(_canonical):
        raise HTTPException(status_code=503, detail="registry down")

    monkeypatch.setattr(
        "cogniverse_runtime.admin.tenant_manager.get_tenant_internal", _raise_503
    )

    with pytest.raises(HTTPException) as exc:
        await tenant_utils.assert_tenant_exists(tid)
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_list_organizations_internal_raises_on_outage(monkeypatch):
    """A whole-sweep [] on outage let the cleanup cron report success while
    doing nothing — it must raise so the cron fails loudly instead."""
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.query_metadata_documents.side_effect = RuntimeError("Vespa 503")
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    with pytest.raises(HTTPException) as exc:
        await tm.list_organizations_internal()
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_list_tenants_for_org_internal_raises_on_outage(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.query_metadata_documents.side_effect = RuntimeError("Vespa 503")
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    with pytest.raises(HTTPException) as exc:
        await tm.list_tenants_for_org_internal("acme")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_list_tenants_for_org_internal_returns_tenants_on_success(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.query_metadata_documents.return_value = [
        {"tenant_full_id": "acme:prod", "org_id": "acme", "tenant_name": "prod"},
    ]
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    tenants = await tm.list_tenants_for_org_internal("acme")
    assert [t.tenant_full_id for t in tenants] == ["acme:prod"]


def _delete_seam(monkeypatch, *, remaining_tenants, org_exists=True):
    """Wire delete_tenant_internal's collaborators for org-lifecycle tests."""
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.schema_manager._schema_registry = None
    backend.schema_manager.list_deployed_document_types.return_value = []
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _tenant(_tid):
        return MagicMock()

    async def _org(_org_id):
        return MagicMock() if org_exists else None

    async def _remaining(_org_id):
        return remaining_tenants

    monkeypatch.setattr(tm, "get_tenant_internal", _tenant)
    monkeypatch.setattr(tm, "get_organization_internal", _org)
    monkeypatch.setattr(tm, "list_tenants_for_org_internal", _remaining)
    return tm, backend


def _org_deletes(backend):
    return [
        c
        for c in backend.delete_metadata_document.call_args_list
        if c.kwargs.get("schema") == "organization_metadata"
    ]


@pytest.mark.asyncio
async def test_deleting_last_tenant_deletes_the_auto_created_org(monkeypatch):
    """Tenant create auto-creates the org; deleting the org's last tenant
    must remove it again, or every provision/teardown cycle leaks an org."""
    tm, backend = _delete_seam(monkeypatch, remaining_tenants=[])

    result = await tm.delete_tenant_internal("acme:acme")

    assert result["status"] == "deleted"
    assert result["organization_deleted"] is True
    org_calls = _org_deletes(backend)
    assert len(org_calls) == 1
    assert org_calls[0].kwargs["doc_id"] == "acme"


@pytest.mark.asyncio
async def test_org_with_remaining_tenants_is_kept(monkeypatch):
    tm, backend = _delete_seam(monkeypatch, remaining_tenants=[MagicMock()])

    result = await tm.delete_tenant_internal("acme:acme")

    assert result["status"] == "deleted"
    assert result["organization_deleted"] is False
    assert _org_deletes(backend) == []


@pytest.mark.asyncio
async def test_org_cleanup_failure_keeps_tenant_delete_successful(monkeypatch, caplog):
    """The tenant IS deleted by the time org cleanup runs; a backend blip
    there must warn and report organization_deleted false, not turn the
    whole delete into an error."""
    import logging

    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.schema_manager._schema_registry = None
    backend.schema_manager.list_deployed_document_types.return_value = []
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _tenant(_tid):
        return MagicMock()

    async def _boom(_org_id):
        raise HTTPException(status_code=503, detail="registry unavailable")

    monkeypatch.setattr(tm, "get_tenant_internal", _tenant)
    monkeypatch.setattr(tm, "list_tenants_for_org_internal", _boom)

    with caplog.at_level(logging.WARNING):
        result = await tm.delete_tenant_internal("acme:acme")

    assert result["status"] == "deleted"
    assert result["organization_deleted"] is False
    assert "Organization cleanup" in caplog.text


@pytest.mark.asyncio
async def test_org_delete_reporting_failure_is_not_claimed_deleted(monkeypatch, caplog):
    """delete_metadata_document returns False on a non-200 without raising;
    the response must not claim organization_deleted in that case."""
    import logging

    tm, backend = _delete_seam(monkeypatch, remaining_tenants=[])
    backend.delete_metadata_document.return_value = False

    with caplog.at_level(logging.WARNING):
        result = await tm.delete_tenant_internal("acme:acme")

    assert result["status"] == "deleted"
    assert result["organization_deleted"] is False
    assert len(_org_deletes(backend)) == 1
    assert "may remain" in caplog.text


@pytest.mark.asyncio
async def test_schema_delete_runs_off_the_event_loop(monkeypatch):
    """Each schema-removal redeploy takes minutes of Vespa deploy +
    convergence; run inline it blocks the whole runtime API (health
    included) for the duration of a tenant delete."""
    import threading
    from types import SimpleNamespace

    from cogniverse_runtime.admin import tenant_manager as tm

    loop_thread = threading.get_ident()
    delete_threads: list[int] = []

    backend = MagicMock()
    registry = MagicMock()
    registry.get_tenant_schemas.return_value = [
        SimpleNamespace(base_schema_name="video_x")
    ]
    backend.schema_manager._schema_registry = registry
    backend.schema_manager.list_deployed_document_types.return_value = []

    def _record_delete(tid, base_name):
        delete_threads.append(threading.get_ident())
        return f"video_x_{tid.replace(':', '_')}"

    backend.schema_manager.delete_schema.side_effect = _record_delete
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _tenant(_tid):
        return MagicMock()

    async def _org(_org_id):
        return None

    async def _remaining(_org_id):
        return []

    monkeypatch.setattr(tm, "get_tenant_internal", _tenant)
    monkeypatch.setattr(tm, "get_organization_internal", _org)
    monkeypatch.setattr(tm, "list_tenants_for_org_internal", _remaining)

    result = await tm.delete_tenant_internal("acme:acme")

    assert result["schemas_deleted"] >= 1
    assert delete_threads, "delete_schema was never called"
    assert all(t != loop_thread for t in delete_threads)
