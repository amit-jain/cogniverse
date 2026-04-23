"""Invite token authentication and user-tenant mapping.

Tokens stored in ConfigStore (VespaConfigStore). User→tenant mappings
stored in Mem0 with agent_name="_messaging_gateway".
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_sdk.interfaces.config_store import ConfigScope

logger = logging.getLogger(__name__)

GATEWAY_AGENT_NAME = "_messaging_gateway"


class InviteTokenManager:
    """Manages invite tokens for Telegram user registration."""

    def __init__(self, config_manager):
        self.config_manager = config_manager

    def generate_token(self, tenant_id: str, expires_in_hours: int = 24) -> str:
        """Generate a new invite token for a tenant."""
        token = uuid.uuid4().hex
        expiry = (datetime.utcnow() + timedelta(hours=expires_in_hours)).isoformat()

        self.config_manager.set_config_value(
            tenant_id="_system",
            scope=ConfigScope.SYSTEM,
            service="messaging_gateway",
            config_key=f"invite_token_{token}",
            config_value={
                "tenant_id": tenant_id,
                "token": token,
                "expires_at": expiry,
                "used": False,
            },
        )

        logger.info(f"Generated invite token for tenant {tenant_id}")
        return token

    def validate_token(self, token: str) -> Optional[str]:
        """Validate an invite token and return the tenant_id if valid.

        Returns None if token is invalid, expired, or already used.
        Searches the _system tenant's config store for invite tokens.
        """
        try:
            entry = self.config_manager.store.get_config(
                tenant_id="_system",
                scope=ConfigScope.SYSTEM,
                service="messaging_gateway",
                config_key=f"invite_token_{token}",
            )

            if entry is None:
                return None

            value = entry.config_value
            if isinstance(value, str):
                import json

                try:
                    value = json.loads(value)
                except Exception:
                    return None

            if value.get("used"):
                logger.warning(f"Token already used: {token[:8]}...")
                return None

            expires_at = value.get("expires_at", "")
            if expires_at:
                expiry = datetime.fromisoformat(expires_at)
                if datetime.utcnow() > expiry:
                    logger.warning(f"Token expired: {token[:8]}...")
                    return None

            return value.get("tenant_id")

        except Exception as e:
            logger.error(f"Token validation failed: {e}")

        return None

    def mark_token_used(self, token: str, tenant_id: str) -> None:
        """Mark a token as used after successful registration."""
        try:
            self.config_manager.set_config_value(
                tenant_id="_system",
                scope=ConfigScope.SYSTEM,
                service="messaging_gateway",
                config_key=f"invite_token_{token}",
                config_value={
                    "tenant_id": tenant_id,
                    "token": token,
                    "used": True,
                    "used_at": datetime.utcnow().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to mark token as used: {e}")


class UserTenantMapper:
    """Maps external messaging user IDs to Cogniverse tenant IDs via Mem0."""

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def register_user(
        self, platform: str, external_user_id: str, tenant_id: str
    ) -> bool:
        """Register a mapping from external user to tenant."""
        content = (
            f"User {external_user_id} on {platform} is mapped to tenant {tenant_id}"
        )
        metadata = {
            "type": "user_mapping",
            "platform": platform,
            "external_user_id": str(external_user_id),
            "tenant_id": tenant_id,
        }

        try:
            self.memory_manager.add_memory(
                content=content,
                tenant_id=tenant_id,
                agent_name=GATEWAY_AGENT_NAME,
                metadata=metadata,
            )
            logger.info(f"Registered {platform} user {external_user_id} → {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register user mapping: {e}")
            return False

    def get_tenant_id(self, platform: str, external_user_id: str) -> Optional[str]:
        """Look up tenant_id for an external user."""
        try:
            results = self.memory_manager.search_memory(
                query=f"User {external_user_id} on {platform} tenant mapping",
                tenant_id=SYSTEM_TENANT_ID,
                agent_name=GATEWAY_AGENT_NAME,
                top_k=5,
            )

            for result in results:
                memory_text = result.get("memory", "")
                if str(external_user_id) in memory_text and platform in memory_text:
                    # Extract tenant_id from the memory text
                    if "mapped to tenant" in memory_text:
                        parts = memory_text.split("mapped to tenant ")
                        if len(parts) > 1:
                            return parts[1].strip()
        except Exception as e:
            logger.error(f"Failed to look up user mapping: {e}")

        return None
