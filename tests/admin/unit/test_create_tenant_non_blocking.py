"""create_tenant must offload the blocking Vespa schema deploy off the loop.

deploy_schema issues a synchronous, no-timeout app redeploy (requests.post +
sleep backoff). Run inline on the event loop it froze every tenant's requests
and health probes for the 5-60s deploy. It is now
``await asyncio.to_thread(deploy_schema, ...)``. The proof is a ticker coroutine
that can only keep advancing while the loop stays free during the deploy sleep.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.admin.models import CreateTenantRequest

pytestmark = [pytest.mark.unit]


@pytest.mark.asyncio
async def test_create_tenant_keeps_loop_responsive_during_deploy():
    def _slow_deploy_schema(tenant_id, base_schema_name):
        time.sleep(0.4)  # stands in for the no-timeout requests.post + backoff

    fake = SimpleNamespace(
        get_metadata_document=lambda schema, doc_id: None,
        create_metadata_document=lambda schema, doc_id, fields: True,
        schema_registry=SimpleNamespace(deploy_schema=_slow_deploy_schema),
    )
    tenant_manager.backend = fake

    ticks = 0
    stop = False

    async def ticker():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        tenant = await tenant_manager.create_tenant(
            CreateTenantRequest(tenant_id="acme:prod", created_by="t")
        )
    finally:
        stop = True
        ticker_task.cancel()

    assert tenant.tenant_full_id == "acme:prod"
    assert tenant.schemas_deployed == ["video_colpali_smol500_mv_frame"]
    # 0.4s deploy at 0.01s tick spacing => ~40 ticks while the loop is free.
    # Inline on the loop the sleep blocks scheduling and ticks stay near 0.
    assert ticks >= 20, f"event loop was blocked by deploy_schema: only {ticks} ticks"
