"""Unit tests for invite token auth and user-tenant mapping."""

from unittest.mock import MagicMock

import pytest
from cogniverse_messaging.auth import InviteTokenManager, UserTenantMapper


class TestInviteTokenManager:
    @pytest.fixture
    def config_manager(self):
        cm = MagicMock()
        cm.set_config = MagicMock()
        cm.get_all_configs = MagicMock(return_value=[])
        return cm

    def test_generate_token(self, config_manager):
        manager = InviteTokenManager(config_manager)
        token = manager.generate_token("acme:alice")

        assert len(token) == 32  # uuid4 hex
        config_manager.set_config.assert_called_once()

        call_kwargs = config_manager.set_config.call_args.kwargs
        assert call_kwargs["tenant_id"] == "acme:alice"
        assert call_kwargs["scope"] == "messaging"
        assert "invite_token_" in call_kwargs["config_key"]

    def test_validate_valid_token(self, config_manager):
        token = "abc123"
        config_manager.get_all_configs.return_value = [
            {
                "config_value": {
                    "token": token,
                    "tenant_id": "acme:alice",
                    "used": False,
                    "expires_at": "2099-12-31T23:59:59",
                }
            }
        ]

        manager = InviteTokenManager(config_manager)
        result = manager.validate_token(token)
        assert result == "acme:alice"

    def test_validate_used_token(self, config_manager):
        config_manager.get_all_configs.return_value = [
            {
                "config_value": {
                    "token": "abc123",
                    "tenant_id": "acme:alice",
                    "used": True,
                }
            }
        ]

        manager = InviteTokenManager(config_manager)
        result = manager.validate_token("abc123")
        assert result is None

    def test_validate_expired_token(self, config_manager):
        config_manager.get_all_configs.return_value = [
            {
                "config_value": {
                    "token": "abc123",
                    "tenant_id": "acme:alice",
                    "used": False,
                    "expires_at": "2020-01-01T00:00:00",
                }
            }
        ]

        manager = InviteTokenManager(config_manager)
        result = manager.validate_token("abc123")
        assert result is None

    def test_validate_unknown_token(self, config_manager):
        config_manager.get_all_configs.return_value = []

        manager = InviteTokenManager(config_manager)
        result = manager.validate_token("nonexistent")
        assert result is None

    def test_mark_token_used(self, config_manager):
        manager = InviteTokenManager(config_manager)
        manager.mark_token_used("abc123", "acme:alice")

        config_manager.set_config.assert_called_once()
        call_value = config_manager.set_config.call_args.kwargs["config_value"]
        assert call_value["used"] is True


class TestUserTenantMapper:
    @pytest.fixture
    def memory_manager(self):
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
        assert call_kwargs["tenant_id"] == "acme:alice"

    def test_get_tenant_id_found(self, memory_manager):
        memory_manager.search_memory.return_value = [
            {"memory": "User 12345 on telegram is mapped to tenant acme:alice"}
        ]

        mapper = UserTenantMapper(memory_manager)
        result = mapper.get_tenant_id("telegram", "12345")
        assert result == "acme:alice"

    def test_get_tenant_id_not_found(self, memory_manager):
        memory_manager.search_memory.return_value = []

        mapper = UserTenantMapper(memory_manager)
        result = mapper.get_tenant_id("telegram", "99999")
        assert result is None
