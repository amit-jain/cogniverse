"""Real Mem0 (Vespa-backed) round-trip for the messaging UserTenantMapper.

Regression: register_user wrote the user→tenant mapping under the user's own
tenant partition, while get_tenant_id read from the SYSTEM partition. Mem0 hard
partitions on user_id, so the lookup never saw the write and every registered
Telegram user looked unregistered. This exercises the real Mem0 store, not a
mock — register then look up against actual Vespa-backed memory.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from cogniverse_messaging.auth import GATEWAY_AGENT_NAME, UserTenantMapper

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model


def _build_manager(*, shared_memory_vespa, shared_denseon) -> Mem0MemoryManager:
    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    mm = Mem0MemoryManager(tenant_id=SYSTEM_TENANT_ID)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
    )
    return mm


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_then_lookup_resolves_tenant_real_mem0(
    shared_memory_vespa, shared_denseon
):
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    mapper = UserTenantMapper(mm)

    # Unique user id so the shared SYSTEM partition can't collide across runs.
    user_id = f"tg{uuid.uuid4().hex[:10]}"
    tenant = "acme:production"

    assert mapper.register_user("telegram", user_id, tenant) is True

    # The mapping is queryable from the SYSTEM partition (where the lookup
    # runs before the tenant is known) and resolves to the registered tenant.
    resolved = mapper.get_tenant_id("telegram", user_id)
    assert resolved == tenant

    # An unregistered user resolves to None against the same real store.
    assert mapper.get_tenant_id("telegram", f"tg{uuid.uuid4().hex[:10]}") is None

    # The mapping landed in the SYSTEM partition, not the tenant's own.
    sys_hits = mm.search_memory(
        query=f"User {user_id} on telegram tenant mapping",
        tenant_id=SYSTEM_TENANT_ID,
        agent_name=GATEWAY_AGENT_NAME,
        top_k=5,
    )
    assert any(user_id in (h.get("memory", "")) for h in sys_hits)
