"""create_tenant must roll back every partial write.

A tenant create that fails after schemas deploy leaves live state behind unless
the deployed tenant schemas and the auto-created org metadata are removed too.
The contract here is:
- transport timeouts are retried without rolling back;
- schema-deploy and tenant-write failures roll back the tenant schemas and org;
- missing schema_manager during rollback is logged loudly.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from requests import exceptions as requests_exceptions

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_MISSING = object()


class _RecordingSchemaRegistry:
    def __init__(self, outcomes):
        self.calls: list[tuple[str, str]] = []
        self._outcomes = list(outcomes)

    def deploy_schema(self, tenant_id, base_schema_name):
        self.calls.append((tenant_id, base_schema_name))
        if not self._outcomes:
            raise AssertionError("deploy_schema called more times than expected")

        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _RecordingSchemaManager:
    def __init__(self):
        self.calls: list[str] = []

    def delete_tenant_schemas(self, tenant_id):
        self.calls.append(tenant_id)
        return [f"rolled_back:{tenant_id}"]


class _RecordingBackend:
    def __init__(self, *, deploy_outcomes, create_outcomes, schema_manager=_MISSING):
        self.schema_registry = _RecordingSchemaRegistry(deploy_outcomes)
        self.schema_manager = (
            _RecordingSchemaManager() if schema_manager is _MISSING else schema_manager
        )
        self.create_metadata_calls: list[tuple[str, str, dict]] = []
        self.delete_metadata_calls: list[tuple[str, str]] = []
        self._create_outcomes = list(create_outcomes)

    def create_metadata_document(self, *, schema, doc_id, fields):
        self.create_metadata_calls.append((schema, doc_id, dict(fields)))
        if not self._create_outcomes:
            raise AssertionError(
                "create_metadata_document called more times than expected"
            )

        outcome = self._create_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def delete_metadata_document(self, *, schema, doc_id):
        self.delete_metadata_calls.append((schema, doc_id))
        return True


def _prepare_create_tenant(monkeypatch, backend, *, org_exists=False):
    from cogniverse_runtime.admin import tenant_manager as tm

    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _no_sleep(_seconds):
        return None

    async def _tenant(_tid):
        return None

    async def _org(_org_id):
        return object() if org_exists else None

    monkeypatch.setattr(tm.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(tm.time, "time", lambda: 1234.567)
    monkeypatch.setattr(tm, "get_tenant_internal", _tenant)
    monkeypatch.setattr(tm, "get_organization_internal", _org)
    return tm


@pytest.mark.asyncio
async def test_create_tenant_retries_transport_timeout_without_rollback(monkeypatch):
    from cogniverse_runtime.admin import tenant_manager as tenant_manager_mod

    backend = _RecordingBackend(
        deploy_outcomes=[
            requests_exceptions.Timeout("server took too long"),
            None,
        ],
        create_outcomes=[True, True],
    )
    tenant_manager = _prepare_create_tenant(monkeypatch, backend)

    tenant = await tenant_manager.create_tenant(
        tenant_manager_mod.CreateTenantRequest(
            tenant_id="acme:prod", created_by="admin"
        )
    )

    assert tenant == tenant_manager_mod.Tenant(
        tenant_full_id="acme:prod",
        org_id="acme",
        tenant_name="prod",
        created_at=1234567,
        created_by="admin",
        status="active",
        schemas_deployed=["video_colpali_smol500_mv_frame"],
    )
    assert backend.schema_registry.calls == [
        ("acme:prod", "video_colpali_smol500_mv_frame"),
        ("acme:prod", "video_colpali_smol500_mv_frame"),
    ]
    assert backend.schema_manager.calls == []
    assert backend.create_metadata_calls == [
        (
            "organization_metadata",
            "acme",
            {
                "org_id": "acme",
                "org_name": "Acme",
                "created_at": 1234567,
                "created_by": "admin",
                "status": "active",
                "tenant_count": 0,
            },
        ),
        (
            "tenant_metadata",
            "acme:prod",
            {
                "tenant_full_id": "acme:prod",
                "org_id": "acme",
                "tenant_name": "prod",
                "created_at": 1234567,
                "created_by": "admin",
                "status": "active",
                "schemas_deployed": ["video_colpali_smol500_mv_frame"],
            },
        ),
    ]
    assert backend.delete_metadata_calls == []


@pytest.mark.asyncio
async def test_create_tenant_rolls_back_schema_and_org_after_deploy_failure(
    monkeypatch,
):
    from cogniverse_runtime.admin import tenant_manager as tenant_manager_mod

    backend = _RecordingBackend(
        deploy_outcomes=[
            None,
            RuntimeError("schema validation failed"),
        ],
        create_outcomes=[True],
    )
    tenant_manager = _prepare_create_tenant(monkeypatch, backend)

    request = tenant_manager_mod.CreateTenantRequest(
        tenant_id="acme:prod",
        created_by="admin",
        base_schemas=[
            "video_colpali_smol500_mv_frame",
            "document_visual",
        ],
    )

    with pytest.raises(HTTPException) as exc:
        await tenant_manager.create_tenant(request)

    assert exc.value.status_code == 500
    assert exc.value.detail == "schema validation failed"
    assert backend.schema_registry.calls == [
        ("acme:prod", "video_colpali_smol500_mv_frame"),
        ("acme:prod", "document_visual"),
    ]
    assert backend.schema_manager.calls == ["acme:prod"]
    assert backend.create_metadata_calls == [
        (
            "organization_metadata",
            "acme",
            {
                "org_id": "acme",
                "org_name": "Acme",
                "created_at": 1234567,
                "created_by": "admin",
                "status": "active",
                "tenant_count": 0,
            },
        ),
    ]
    assert backend.delete_metadata_calls == [("organization_metadata", "acme")]


@pytest.mark.asyncio
async def test_create_tenant_rolls_back_schema_and_org_after_metadata_failure(
    monkeypatch,
):
    from cogniverse_runtime.admin import tenant_manager as tenant_manager_mod

    backend = _RecordingBackend(
        deploy_outcomes=[None],
        create_outcomes=[True, False],
    )
    tenant_manager = _prepare_create_tenant(monkeypatch, backend)

    request = tenant_manager_mod.CreateTenantRequest(
        tenant_id="acme:prod", created_by="admin"
    )

    with pytest.raises(HTTPException) as exc:
        await tenant_manager.create_tenant(request)

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to create tenant acme:prod in backend"
    assert backend.schema_registry.calls == [
        ("acme:prod", "video_colpali_smol500_mv_frame"),
    ]
    assert backend.schema_manager.calls == ["acme:prod"]
    assert backend.create_metadata_calls == [
        (
            "organization_metadata",
            "acme",
            {
                "org_id": "acme",
                "org_name": "Acme",
                "created_at": 1234567,
                "created_by": "admin",
                "status": "active",
                "tenant_count": 0,
            },
        ),
        (
            "tenant_metadata",
            "acme:prod",
            {
                "tenant_full_id": "acme:prod",
                "org_id": "acme",
                "tenant_name": "prod",
                "created_at": 1234567,
                "created_by": "admin",
                "status": "active",
                "schemas_deployed": ["video_colpali_smol500_mv_frame"],
            },
        ),
    ]
    assert backend.delete_metadata_calls == [("organization_metadata", "acme")]


@pytest.mark.asyncio
async def test_create_tenant_logs_when_schema_manager_missing_during_rollback(
    monkeypatch,
    caplog,
):
    from cogniverse_runtime.admin import tenant_manager as tenant_manager_mod

    backend = _RecordingBackend(
        deploy_outcomes=[
            None,
            RuntimeError("schema validation failed"),
        ],
        create_outcomes=[True],
        schema_manager=None,
    )
    tenant_manager = _prepare_create_tenant(monkeypatch, backend)

    request = tenant_manager_mod.CreateTenantRequest(
        tenant_id="acme:prod",
        created_by="admin",
        base_schemas=[
            "video_colpali_smol500_mv_frame",
            "document_visual",
        ],
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as exc:
            await tenant_manager.create_tenant(request)

    assert exc.value.detail == "schema validation failed"
    assert backend.delete_metadata_calls == [("organization_metadata", "acme")]
    assert (
        "backend.schema_manager is unavailable after deploying 1 schema(s)"
        in caplog.text
    )
