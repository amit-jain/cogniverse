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
    backend.schema_manager.delete_tenant_schemas.return_value = []
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
async def test_tenant_schemas_dropped_in_one_offloaded_redeploy(monkeypatch):
    """All of a tenant's schemas go in ONE redeploy (delete_tenant_schemas),
    run off the event loop — the per-schema loop did one multi-minute
    redeploy per schema, blocking the whole runtime API for each."""
    import threading

    from cogniverse_runtime.admin import tenant_manager as tm

    loop_thread = threading.get_ident()
    call_threads: list[int] = []

    backend = MagicMock()

    def _record_bulk(tid):
        call_threads.append(threading.get_ident())
        assert tid == "acme:acme"
        return ["video_x_acme_acme", "wiki_pages_acme_acme"]

    backend.schema_manager.delete_tenant_schemas.side_effect = _record_bulk
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

    assert result["schemas_deleted"] == 2
    assert result["deleted_schemas"] == [
        "video_x_acme_acme",
        "wiki_pages_acme_acme",
    ]
    backend.schema_manager.delete_tenant_schemas.assert_called_once_with("acme:acme")
    backend.schema_manager.delete_schema.assert_not_called()
    assert call_threads, "delete_tenant_schemas was never called"
    assert all(t != loop_thread for t in call_threads)


@pytest.mark.asyncio
async def test_schema_drop_failure_keeps_the_tenant(monkeypatch):
    """A refused or failed redeploy propagates: the tenant record stays and
    the delete is retryable — schemas are never silently stranded behind a
    deleted tenant record."""
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.schema_manager.delete_tenant_schemas.side_effect = RuntimeError(
        "Cannot enumerate Vespa-deployed schemas"
    )
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _tenant(_tid):
        return MagicMock()

    monkeypatch.setattr(tm, "get_tenant_internal", _tenant)

    with pytest.raises(RuntimeError, match="Cannot enumerate"):
        await tm.delete_tenant_internal("acme:acme")

    backend.delete_metadata_document.assert_not_called()


@pytest.mark.asyncio
async def test_raw_form_input_resolves_to_one_canonical_pass(monkeypatch):
    """A raw-form tenant id goes through exactly one canonical
    delete_tenant_schemas pass — the registry APIs canonicalize on both
    reads and writes, so no raw-keyed follow-up work exists."""
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.schema_manager.delete_tenant_schemas.return_value = ["video_x_acme_acme"]
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

    result = await tm.delete_tenant_internal("acme")

    backend.schema_manager.delete_tenant_schemas.assert_called_once_with("acme:acme")
    backend.schema_manager.delete_schema.assert_not_called()
    assert result["deleted_schemas"] == ["video_x_acme_acme"]
    assert result["tenant_full_id"] == "acme:acme"


@pytest.mark.asyncio
async def test_org_delete_continues_past_one_failing_tenant(monkeypatch):
    """delete_organization tolerates a single tenant's failed delete: the
    remaining tenants and the org record still go, and the response names
    exactly the tenants that were deleted."""
    from types import SimpleNamespace

    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _org(_org_id):
        return MagicMock()

    async def _tenants(_org_id):
        return [
            SimpleNamespace(tenant_full_id="acme:one"),
            SimpleNamespace(tenant_full_id="acme:two"),
        ]

    deleted: list[str] = []

    async def _delete_tenant(tid):
        if tid == "acme:one":
            raise RuntimeError("redeploy refused")
        deleted.append(tid)
        return {"status": "deleted"}

    monkeypatch.setattr(tm, "get_organization_internal", _org)
    monkeypatch.setattr(tm, "list_tenants_for_org_internal", _tenants)
    monkeypatch.setattr(tm, "delete_tenant_internal", _delete_tenant)

    result = await tm.delete_organization("acme")

    assert result["status"] == "deleted"
    assert result["tenants_deleted"] == 1
    assert result["deleted_tenant_ids"] == ["acme:two"]
    assert deleted == ["acme:two"]
    org_deletes = [
        c
        for c in backend.delete_metadata_document.call_args_list
        if c.kwargs.get("schema") == "organization_metadata"
    ]
    assert len(org_deletes) == 1
    assert org_deletes[0].kwargs["doc_id"] == "acme"
