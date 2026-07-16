"""create_tenant must not create an active tenant when schema deploy fails.

A tenant persisted with schemas_deployed=[] silently accepts writes that then
hit an undeployed doc type — every ingest/search fails or grinds the feed-retry
loop, and graph upsert reports success having saved nothing. A deploy failure
must raise 502 and leave no tenant_metadata document.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.asyncio
async def test_create_tenant_fails_loud_when_schema_deploy_fails(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tm

    fake_backend = MagicMock()
    fake_backend.schema_registry.deploy_schema.side_effect = RuntimeError(
        "Vespa config server unreachable"
    )
    monkeypatch.setattr(tm, "get_backend", lambda: fake_backend)

    async def _no_existing_tenant(_tid):
        return None

    async def _org_exists(_oid):
        return MagicMock()  # org already present -> skip auto-create

    monkeypatch.setattr(tm, "get_tenant_internal", _no_existing_tenant)
    monkeypatch.setattr(tm, "get_organization_internal", _org_exists)

    request = tm.CreateTenantRequest(tenant_id="acme:prod", created_by="admin")

    with pytest.raises(HTTPException) as exc:
        await tm.create_tenant(request)

    assert exc.value.status_code == 502
    assert "schema deploy failed" in str(exc.value.detail)

    # The tenant_metadata document must NOT have been written.
    written_schemas = [
        call.kwargs.get("schema")
        for call in fake_backend.create_metadata_document.call_args_list
    ]
    assert "tenant_metadata" not in written_schemas
