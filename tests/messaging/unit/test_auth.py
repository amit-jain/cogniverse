"""Unit tests for invite token auth and user-tenant mapping.

InviteTokenManager runs against a real in-memory ConfigStore through the
real ConfigManager: generate/mark-used (writes) and validate (read) must
agree on the stored key. The previous mocked store returned a canned entry
for any key, which hid a write/read key mismatch that invalidated every
token.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from cogniverse_messaging.auth import InviteTokenManager, UserTenantMapper

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_sdk.interfaces.config_store import ConfigScope


@pytest.fixture
def token_manager(config_manager_memory):
    return InviteTokenManager(config_manager_memory)


def _write_token_value(config_manager, token: str, value) -> None:
    """Store a token value through the same manager path generate_token uses."""
    config_manager.set_config_value(
        tenant_id="_system",
        scope=ConfigScope.SYSTEM,
        service="messaging_gateway",
        config_key=f"invite_token_{token}",
        config_value=value,
    )


class TestInviteTokenManager:
    def test_generate_token_shape(self, token_manager):
        token = token_manager.generate_token("acme:alice")
        assert len(token) == 32
        assert all(c in "0123456789abcdef" for c in token)

    def test_generate_then_validate_round_trip(self, token_manager):
        token = token_manager.generate_token("acme:alice")
        assert token_manager.validate_token(token) == "acme:alice"

    def test_validate_unknown_token(self, token_manager):
        assert token_manager.validate_token("nonexistent") is None

    def test_mark_used_then_validate_rejects(self, token_manager):
        token = token_manager.generate_token("acme:alice")
        assert token_manager.validate_token(token) == "acme:alice"

        token_manager.mark_token_used(token, "acme:alice")
        assert token_manager.validate_token(token) is None

    def test_mark_used_stores_tz_aware_used_at(
        self, token_manager, config_manager_memory
    ):
        before = datetime.now(timezone.utc)
        token = token_manager.generate_token("acme:alice")
        token_manager.mark_token_used(token, "acme:alice")
        after = datetime.now(timezone.utc)

        value = config_manager_memory.get_config_value(
            tenant_id="_system",
            scope=ConfigScope.SYSTEM,
            service="messaging_gateway",
            config_key=f"invite_token_{token}",
        )
        assert value["used"] is True
        assert value["tenant_id"] == "acme:alice"
        used_at = datetime.fromisoformat(value["used_at"])
        assert used_at.tzinfo is not None
        assert used_at.utcoffset() == timedelta(0)
        assert before <= used_at <= after

    def test_expired_token_rejected(self, token_manager):
        token = token_manager.generate_token("acme:alice", expires_in_hours=-1)
        assert token_manager.validate_token(token) is None

    def test_naive_expiry_compared_as_utc(self, token_manager, config_manager_memory):
        """A naive stored expiry is treated as UTC; the old naive-vs-aware
        comparison raised TypeError and rejected a valid token."""
        _write_token_value(
            config_manager_memory,
            "naivetok",
            {
                "tenant_id": "acme:alice",
                "token": "naivetok",
                "expires_at": "2099-12-31T23:59:59",
                "used": False,
            },
        )
        assert token_manager.validate_token("naivetok") == "acme:alice"

    def test_json_string_value_parsed(self, token_manager, config_manager_memory):
        """VespaConfigStore can hand back config_value as a JSON string;
        validate must parse it rather than reject the token."""
        _write_token_value(
            config_manager_memory,
            "strtok",
            json.dumps(
                {
                    "tenant_id": "acme:alice",
                    "token": "strtok",
                    "expires_at": "2099-12-31T23:59:59+00:00",
                    "used": False,
                }
            ),
        )
        assert token_manager.validate_token("strtok") == "acme:alice"

    def test_validate_fails_closed_on_store_outage(self):
        """A config-store outage validates to None (token treated as not
        valid) instead of crashing the /start handler."""
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        class OutageStore(InMemoryConfigStore):
            def get_config(self, *args, **kwargs):
                raise ConnectionError("config store unreachable")

        store = OutageStore()
        store.initialize()
        manager = InviteTokenManager(ConfigManager(store=store))
        assert manager.validate_token("anytoken") is None


class TestUserTenantMapper:
    @pytest.fixture
    def memory_manager(self):
        from unittest.mock import MagicMock

        mm = MagicMock()
        mm.add_memory.return_value = "mem_123"
        mm.search_memory.return_value = []
        return mm

    def test_register_user(self, memory_manager):
        mapper = UserTenantMapper(memory_manager)
        result = mapper.register_user("telegram", "12345", "acme:alice")

        assert result is True
        memory_manager.add_memory.assert_called_once()

        call_kwargs = memory_manager.add_memory.call_args.kwargs
        assert "12345" in call_kwargs["content"]
        assert "telegram" in call_kwargs["content"]
        # The mapping must be written to the SYSTEM partition (what
        # get_tenant_id reads), NOT the user's own tenant — the lookup runs
        # before the tenant is known. The real tenant is preserved in the
        # content text and metadata. Stored verbatim (infer=False) so the
        # substring parse in get_tenant_id is reliable.
        assert call_kwargs["tenant_id"] == SYSTEM_TENANT_ID
        assert call_kwargs["infer"] is False
        assert "acme:alice" in call_kwargs["content"]
        assert call_kwargs["metadata"]["tenant_id"] == "acme:alice"

    def test_get_tenant_id_reads_system_partition(self, memory_manager):
        """get_tenant_id must enumerate the SYSTEM partition (where
        register_user writes), not some other tenant — the read/write
        partitions must match, and the match is exact on metadata."""
        memory_manager.get_all_memories.return_value = [
            {
                "memory": "User 12345 on telegram is mapped to tenant acme:alice",
                "metadata": {
                    "type": "user_mapping",
                    "platform": "telegram",
                    "external_user_id": "12345",
                    "tenant_id": "acme:alice",
                },
            }
        ]

        mapper = UserTenantMapper(memory_manager)
        result = mapper.get_tenant_id("telegram", "12345")
        assert result == "acme:alice"
        assert memory_manager.get_all_memories.call_args.kwargs["tenant_id"] == (
            SYSTEM_TENANT_ID
        )

    def test_register_then_lookup_round_trip(self):
        """Regression: with a partition-faithful Mem0 model (search only
        returns memories written to the SAME (tenant_id, agent_name) partition,
        as the real Mem0 hard-partitions), register_user → get_tenant_id must
        resolve the tenant. On the old code the write went to 'acme:alice' and
        the read to the SYSTEM partition, so the lookup returned nothing."""

        class _PartitionedMemory:
            def __init__(self):
                self.store = {}

            def add_memory(
                self, content, tenant_id, agent_name, metadata=None, infer=True
            ):
                self.store.setdefault((tenant_id, agent_name), []).append(
                    {"memory": content, "metadata": metadata or {}}
                )
                return "mem_1"

            def get_all_memories(self, tenant_id, agent_name):
                return self.store.get((tenant_id, agent_name), [])

        mapper = UserTenantMapper(_PartitionedMemory())
        assert mapper.register_user("telegram", "12345", "acme:alice") is True
        assert mapper.get_tenant_id("telegram", "12345") == "acme:alice"
        # A different, unregistered user resolves to None.
        assert mapper.get_tenant_id("telegram", "99999") is None

    def test_get_tenant_id_not_found(self, memory_manager):
        memory_manager.get_all_memories.return_value = []

        mapper = UserTenantMapper(memory_manager)
        result = mapper.get_tenant_id("telegram", "99999")
        assert result is None
