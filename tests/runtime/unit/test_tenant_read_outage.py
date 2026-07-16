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
