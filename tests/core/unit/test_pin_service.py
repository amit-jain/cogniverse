"""Unit tests for PinService — quotas, authority, override semantics (A.6)."""

from __future__ import annotations

import pytest

from cogniverse_core.memory.pinning import (
    PIN_AGENT_NAME,
    PIN_RECORD_KIND,
    PinAuthorityError,
    PinNotFoundError,
    PinQuotaExceededError,
    PinQuotas,
    PinService,
)
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
    build_default_registry,
)


class FakeManager:
    """In-memory Mem0MemoryManager-shaped stub used by PinService tests."""

    def __init__(self) -> None:
        self._next = 0
        # store: memory_id -> {content, agent_name, tenant_id, metadata}
        self.store: dict[str, dict] = {}

    def add_memory(
        self,
        *,
        content,
        tenant_id,
        agent_name,
        metadata=None,
        infer=False,
    ) -> str:
        self._next += 1
        mid = f"m{self._next}"
        self.store[mid] = {
            "content": content,
            "tenant_id": tenant_id,
            "agent_name": agent_name,
            "metadata": dict(metadata or {}),
        }
        return mid

    def delete_memory(self, *, memory_id, tenant_id, agent_name):
        return self.store.pop(memory_id, None) is not None

    def get_all_memories(self, *, tenant_id, agent_name):
        return [
            {"id": mid, "memory": v["content"], "metadata": v["metadata"]}
            for mid, v in self.store.items()
            if v["agent_name"] == agent_name and v["tenant_id"] == tenant_id
        ]


@pytest.fixture
def registry() -> KnowledgeRegistry:
    return build_default_registry()


@pytest.fixture
def manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def service(manager, registry) -> PinService:
    return PinService(manager, registry)


class TestPinQuotasFromTenantConfig:
    def test_defaults_when_no_config(self):
        q = PinQuotas.from_tenant_config(None)
        assert q.user == 50
        assert q.tenant_admin == 500
        assert q.org_admin is None  # unlimited

    def test_overrides_via_metadata(self):
        class TC:
            metadata = {"pin_quota": {"user": 5, "tenant_admin": 50}}

        q = PinQuotas.from_tenant_config(TC())
        assert q.user == 5
        assert q.tenant_admin == 50
        assert q.org_admin is None

    def test_org_admin_can_be_capped(self):
        class TC:
            metadata = {"pin_quota": {"org_admin": 10}}

        q = PinQuotas.from_tenant_config(TC())
        assert q.org_admin == 10


class TestPinAuthority:
    def test_user_cannot_pin_tenant_instruction(self, service, manager):
        # tenant_instruction requires tenant_admin floor.
        target_id = manager.add_memory(
            content="seed",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "tenant_instruction"},
        )
        with pytest.raises(PinAuthorityError):
            service.pin(
                target_memory_id=target_id,
                target_kind="tenant_instruction",
                pinned_by=Pinnable.USER,
                actor_id="alice",
                tenant_id="t1",
            )

    def test_tenant_admin_can_pin_tenant_instruction(self, service, manager):
        target_id = manager.add_memory(
            content="seed",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "tenant_instruction"},
        )
        rec = service.pin(
            target_memory_id=target_id,
            target_kind="tenant_instruction",
            pinned_by=Pinnable.TENANT_ADMIN,
            actor_id="admin1",
            tenant_id="t1",
        )
        assert rec.target_memory_id == target_id
        # And it shows up in list_pins / is_pinned.
        assert service.is_pinned(target_id, "t1")
        assert any(r.target_memory_id == target_id for r in service.list_pins("t1"))

    def test_user_can_pin_conversation_turn(self, service, manager, registry):
        # conversation_turn pinnable_by=USER per the seed registry.
        # Override to make sure the seeded value really is USER.
        registry.register(
            KnowledgeSchema(
                kind="conversation_turn",
                pinnable_by=Pinnable.USER,
                provenance_required=False,
            ),
            replace=True,
        )
        target_id = manager.add_memory(
            content="seed",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "conversation_turn"},
        )
        rec = service.pin(
            target_memory_id=target_id,
            target_kind="conversation_turn",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id="t1",
        )
        assert rec.pinned_by is Pinnable.USER


class TestQuotaEnforcement:
    def test_user_quota_blocks_after_n_pins(self, manager, registry):
        registry.register(
            KnowledgeSchema(
                kind="my_kind",
                pinnable_by=Pinnable.USER,
                provenance_required=False,
            ),
            replace=True,
        )
        service = PinService(
            manager, registry, quotas=PinQuotas(user=2, tenant_admin=10)
        )

        for i in range(2):
            tid = manager.add_memory(
                content=f"seed{i}",
                tenant_id="t1",
                agent_name="x",
                metadata={"kind": "my_kind"},
            )
            service.pin(
                target_memory_id=tid,
                target_kind="my_kind",
                pinned_by=Pinnable.USER,
                actor_id="alice",
                tenant_id="t1",
            )

        # third pin should bounce.
        third = manager.add_memory(
            content="seed_extra",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "my_kind"},
        )
        with pytest.raises(PinQuotaExceededError):
            service.pin(
                target_memory_id=third,
                target_kind="my_kind",
                pinned_by=Pinnable.USER,
                actor_id="alice",
                tenant_id="t1",
            )

    def test_org_admin_unlimited_by_default(self, manager, registry):
        service = PinService(manager, registry, quotas=PinQuotas())
        # Pin many targets as org_admin — should not hit a cap.
        for i in range(20):
            tid = manager.add_memory(
                content=f"s{i}",
                tenant_id="t1",
                agent_name="x",
                metadata={"kind": "tenant_instruction"},
            )
            service.pin(
                target_memory_id=tid,
                target_kind="tenant_instruction",
                pinned_by=Pinnable.ORG_ADMIN,
                actor_id="oadm",
                tenant_id="t1",
            )
        assert service.quota_used(Pinnable.ORG_ADMIN, "t1") == 20


class TestOverrideSemantics:
    def test_org_admin_can_override_existing_pin(self, manager, registry):
        # Override schema so user can pin too.
        registry.register(
            KnowledgeSchema(
                kind="dup_target",
                pinnable_by=Pinnable.USER,
                provenance_required=False,
            ),
            replace=True,
        )
        service = PinService(manager, registry)
        target_id = manager.add_memory(
            content="seed",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "dup_target"},
        )
        # User pins first.
        first = service.pin(
            target_memory_id=target_id,
            target_kind="dup_target",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id="t1",
        )
        assert service.is_pinned(target_id, "t1")

        # Org admin pins same target — must replace, not error.
        second = service.pin(
            target_memory_id=target_id,
            target_kind="dup_target",
            pinned_by=Pinnable.ORG_ADMIN,
            actor_id="oadm",
            tenant_id="t1",
        )
        assert second.pinned_by is Pinnable.ORG_ADMIN
        # The original user pin record must be gone.
        all_pins = service.list_pins("t1")
        assert len(all_pins) == 1
        assert all_pins[0].pinned_by is Pinnable.ORG_ADMIN
        # First record id must no longer be in the manager store.
        assert first.memory_id not in manager.store

    def test_user_cannot_override_existing_pin(self, manager, registry):
        registry.register(
            KnowledgeSchema(
                kind="dup_target",
                pinnable_by=Pinnable.USER,
                provenance_required=False,
            ),
            replace=True,
        )
        service = PinService(manager, registry)
        target_id = manager.add_memory(
            content="seed",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "dup_target"},
        )
        service.pin(
            target_memory_id=target_id,
            target_kind="dup_target",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id="t1",
        )
        # Another user can't override the first user's pin.
        with pytest.raises(PinAuthorityError):
            service.pin(
                target_memory_id=target_id,
                target_kind="dup_target",
                pinned_by=Pinnable.USER,
                actor_id="bob",
                tenant_id="t1",
            )


class TestUnpin:
    def test_user_can_only_unpin_own(self, manager, registry):
        registry.register(
            KnowledgeSchema(
                kind="dup",
                pinnable_by=Pinnable.USER,
                provenance_required=False,
            ),
            replace=True,
        )
        service = PinService(manager, registry)
        tid = manager.add_memory(
            content="x",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "dup"},
        )
        service.pin(
            target_memory_id=tid,
            target_kind="dup",
            pinned_by=Pinnable.USER,
            actor_id="alice",
            tenant_id="t1",
        )
        # Bob cannot unpin alice's pin.
        with pytest.raises(PinAuthorityError):
            service.unpin(
                target_memory_id=tid,
                requester=Pinnable.USER,
                actor_id="bob",
                tenant_id="t1",
            )
        # Alice can.
        removed = service.unpin(
            target_memory_id=tid,
            requester=Pinnable.USER,
            actor_id="alice",
            tenant_id="t1",
        )
        assert removed == 1
        assert not service.is_pinned(tid, "t1")

    def test_unpin_missing_raises(self, service):
        with pytest.raises(PinNotFoundError):
            service.unpin(
                target_memory_id="m_doesnt_exist",
                requester=Pinnable.ORG_ADMIN,
                actor_id="oadm",
                tenant_id="t1",
            )

    def test_org_admin_can_unpin_any(self, manager, registry):
        service = PinService(manager, registry)
        tid = manager.add_memory(
            content="x",
            tenant_id="t1",
            agent_name="x",
            metadata={"kind": "tenant_instruction"},
        )
        service.pin(
            target_memory_id=tid,
            target_kind="tenant_instruction",
            pinned_by=Pinnable.TENANT_ADMIN,
            actor_id="tadm",
            tenant_id="t1",
        )
        # Org admin removes a tenant-admin pin.
        removed = service.unpin(
            target_memory_id=tid,
            requester=Pinnable.ORG_ADMIN,
            actor_id="oadm",
            tenant_id="t1",
        )
        assert removed == 1


class TestPinRecordIsolation:
    """pin_records must not pollute normal-agent search results."""

    def test_pin_records_live_under_pin_agent_name(self, manager, registry):
        service = PinService(manager, registry)
        tid = manager.add_memory(
            content="payload",
            tenant_id="t1",
            agent_name="search_agent",
            metadata={"kind": "external_doc"},
        )
        service.pin(
            target_memory_id=tid,
            target_kind="external_doc",
            pinned_by=Pinnable.TENANT_ADMIN,
            actor_id="tadm",
            tenant_id="t1",
        )
        # search_agent's bucket must have ONLY the original payload (not the pin record).
        agent_rows = manager.get_all_memories(tenant_id="t1", agent_name="search_agent")
        assert len(agent_rows) == 1
        assert agent_rows[0]["id"] == tid

        # The pin record lives under PIN_AGENT_NAME (sentinel).
        pin_rows = manager.get_all_memories(tenant_id="t1", agent_name=PIN_AGENT_NAME)
        assert len(pin_rows) == 1
        assert pin_rows[0]["metadata"]["kind"] == PIN_RECORD_KIND
