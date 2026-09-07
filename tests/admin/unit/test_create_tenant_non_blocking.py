"""Tenant deployment yields the event loop until the entire package is ready."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.admin.models import CreateTenantRequest

pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_create_tenant_keeps_loop_responsive_during_deploy(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def deploy(tenant_id, base_schema_names):
        calls.append((tenant_id, base_schema_names))
        entered.set()
        assert release.wait(timeout=5) is True

    fake = SimpleNamespace(
        get_metadata_document=lambda schema, doc_id: None,
        create_metadata_document=lambda schema, doc_id, fields: True,
        schema_registry=SimpleNamespace(deploy_schemas=deploy),
    )
    monkeypatch.setattr(tenant_manager, "backend", fake)
    bases = ["agent_memories", "provenance", "wiki_pages"]
    creation = asyncio.create_task(
        tenant_manager.create_tenant(
            CreateTenantRequest(
                tenant_id="acme:prod", created_by="t", base_schemas=bases
            )
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2) is True
        assert creation.done() is False
        health = await tenant_manager.health_check()
        assert health["status"] == "healthy"
    finally:
        release.set()
        tenant = await creation
    assert tenant.tenant_full_id == "acme:prod"
    assert tenant.schemas_deployed == bases
    assert calls == [("acme:prod", bases)]
