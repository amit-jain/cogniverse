"""PinService against real Mem0 + real Vespa.

Verifies pinning composes with the actual storage backend: pin records
persist, list_pins reads them back, unpin removes them, and quota
enforcement works against Vespa-stored counts (not in-memory counters).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.pinning import (
    PinAuthorityError,
    PinNotFoundError,
    PinQuotaExceededError,
    PinQuotas,
    PinService,
)
from cogniverse_core.memory.schema import (
    KnowledgeSchema,
    Pinnable,
    build_default_registry,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "pin_int_tenant"


@pytest.fixture(scope="module")
def pin_environment(shared_memory_vespa, shared_denseon):
    """Real Mem0 manager + a fresh registry for pinning tests."""
    Mem0MemoryManager._instances.clear()

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
    mm = Mem0MemoryManager(tenant_id=TENANT)
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

    registry = build_default_registry()
    # Allow user-pinning of conversation_turn for the user-quota test.
    registry.register(
        KnowledgeSchema(
            kind="user_pinnable_kind",
            pinnable_by=Pinnable.USER,
            provenance_required=False,
        ),
        replace=True,
    )

    yield mm, registry

    # Best-effort teardown.
    try:
        mm.clear_agent_memory(TENANT, "_pinning")
        mm.clear_agent_memory(TENANT, "pin_int_agent")
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _seed_target(mm, kind: str) -> str:
    return mm.add_memory(
        content=f"target memory for kind={kind}",
        tenant_id=TENANT,
        agent_name="pin_int_agent",
        metadata={"kind": kind},
        infer=False,
    )


def test_round_trip_pin_unpin_against_real_vespa(pin_environment):
    """Pin → list_pins → unpin → list_pins, all through real Vespa I/O."""
    mm, registry = pin_environment
    service = PinService(mm, registry)

    target_id = _seed_target(mm, "tenant_instruction")

    record = service.pin(
        target_memory_id=target_id,
        target_kind="tenant_instruction",
        pinned_by=Pinnable.TENANT_ADMIN,
        actor_id="admin_alpha",
        tenant_id=TENANT,
    )
    assert record.target_memory_id == target_id
    assert service.is_pinned(target_id, TENANT) is True

    listed = service.list_pins(TENANT)
    matching = [r for r in listed if r.target_memory_id == target_id]
    assert len(matching) == 1
    assert matching[0].pinned_by is Pinnable.TENANT_ADMIN
    assert matching[0].pinned_by_actor == "admin_alpha"

    removed = service.unpin(
        target_memory_id=target_id,
        requester=Pinnable.TENANT_ADMIN,
        actor_id="admin_alpha",
        tenant_id=TENANT,
    )
    assert removed == 1
    assert service.is_pinned(target_id, TENANT) is False


def test_quota_enforced_against_real_vespa_counts(pin_environment):
    """User quota counts persisted pin_records, not in-memory state."""
    mm, registry = pin_environment
    service = PinService(mm, registry, quotas=PinQuotas(user=2, tenant_admin=10))

    pinned_targets = []
    for i in range(2):
        tid = _seed_target(mm, "user_pinnable_kind")
        pinned_targets.append(tid)
        service.pin(
            target_memory_id=tid,
            target_kind="user_pinnable_kind",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id=TENANT,
        )

    # Third pin must bounce — quota lookup goes through the real Vespa search.
    third = _seed_target(mm, "user_pinnable_kind")
    with pytest.raises(PinQuotaExceededError):
        service.pin(
            target_memory_id=third,
            target_kind="user_pinnable_kind",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id=TENANT,
        )

    # Cleanup so subsequent tests start fresh.
    for tid in pinned_targets:
        try:
            service.unpin(
                target_memory_id=tid,
                requester=Pinnable.USER,
                actor_id="alice",
                tenant_id=TENANT,
            )
        except PinNotFoundError:
            pass


def test_authority_violation_persists_no_pin_record(pin_environment):
    """A rejected pin must not leave any artefact in Vespa."""
    mm, registry = pin_environment
    service = PinService(mm, registry)

    target_id = _seed_target(mm, "tenant_instruction")
    with pytest.raises(PinAuthorityError):
        service.pin(
            target_memory_id=target_id,
            target_kind="tenant_instruction",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id=TENANT,
        )

    # No pin records should reference this target.
    assert service.is_pinned(target_id, TENANT) is False
    listed = [r for r in service.list_pins(TENANT) if r.target_memory_id == target_id]
    assert listed == []
