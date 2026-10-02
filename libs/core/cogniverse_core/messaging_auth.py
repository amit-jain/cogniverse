"""Invite token authentication and user-tenant mapping for messaging surfaces.

Tokens stored in ConfigStore (VespaConfigStore). User→tenant mappings
stored in Mem0 with agent_name="_messaging_gateway".
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID, canonical_tenant_id
from cogniverse_sdk.interfaces.config_store import ConfigScope

logger = logging.getLogger(__name__)

GATEWAY_AGENT_NAME = "_messaging_gateway"


def _mapping_session_key(platform: str, external_user_id: str) -> str:
    """Promoted filter key for one user→tenant mapping.

    Stored as ``session_id`` (a filterable Vespa field) so ``get_tenant_id``
    narrows to the single mapping server-side instead of scanning the
    partition's capped 100-row page.
    """
    return f"{platform}:{external_user_id}"


class InviteTokenManager:
    """Manages invite tokens for Telegram user registration.

    A token is redeemed in two compare-and-set steps on its config record,
    so every process and replica agrees on who redeemed it: ``claim_token``
    binds an unused token to one external user, and ``mark_token_used``
    consumes it for that user. Only the bound user can complete or retry a
    redemption; every other user is refused from the moment of the claim.
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager

    def generate_token(self, tenant_id: str, expires_in_hours: int = 24) -> str:
        """Generate a new invite token for a tenant."""
        token = uuid.uuid4().hex
        expiry = (
            datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
        ).isoformat()

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
        """Return the tenant_id of a token no user has claimed or used.

        Returns None if the token is unknown, expired, claimed or used.
        Raises on a config-store outage — flattening that to None told the
        user their perfectly good token was invalid. Reads through the
        ConfigManager so the lookup key gets the same tenant
        canonicalization as generate_token's write — a raw store read with
        "_system" looks up a key nobody writes.
        """
        value = self.config_manager.get_config_value(
            tenant_id="_system",
            scope=ConfigScope.SYSTEM,
            service="messaging_gateway",
            config_key=f"invite_token_{token}",
        )
        if value is None or value.get("claimed_by") is not None:
            return None
        return self._redeemable_tenant(token, value)

    def claim_token(
        self, token: str, platform: str, external_user_id: str
    ) -> Optional[str]:
        """Bind an unused token to one external user; return its tenant_id.

        Returns None if the token is unknown, expired, used, or bound to a
        different user. A token already bound to this user returns its
        tenant_id again, so a registration that failed after the claim can
        be retried by the same user. Raises on a config-store outage and on
        ``ConfigWriteConflictError``.
        """
        claimant = _claimant(platform, external_user_id)
        claimed: dict = {}

        def claim(entry):
            claimed.clear()
            if entry is None:
                return None
            value = entry.config_value
            holder = value.get("claimed_by")
            if holder is not None and holder != claimant:
                logger.warning(f"Token claimed by another user: {token[:8]}...")
                return None
            tenant_id = self._redeemable_tenant(token, value)
            if tenant_id is None:
                return None
            claimed["tenant_id"] = tenant_id
            if holder == claimant:
                return None
            return {
                **value,
                "claimed_by": claimant,
                "claimed_at": datetime.now(timezone.utc).isoformat(),
            }

        self.config_manager.store.update_config(*_token_coordinates(token), claim)
        return claimed.get("tenant_id")

    def mark_token_used(self, token: str, platform: str, external_user_id: str) -> bool:
        """Consume a token this user claimed, after their mapping is stored.

        Returns False when the write fails or the token is not bound to this
        user — logged; the user is already registered at that point, so a
        failure here must not undo the registration. The token stays bound to
        this user, so no other user can redeem it either way.
        """
        claimant = _claimant(platform, external_user_id)
        consumed: dict = {}

        def consume(entry):
            consumed.clear()
            if entry is None or entry.config_value.get("claimed_by") != claimant:
                return None
            consumed["ok"] = True
            if entry.config_value.get("used"):
                return None
            return {
                **entry.config_value,
                "used": True,
                "used_at": datetime.now(timezone.utc).isoformat(),
            }

        try:
            self.config_manager.store.update_config(*_token_coordinates(token), consume)
        except Exception as e:
            logger.error(f"Failed to mark token as used: {e}")
            return False
        if not consumed:
            logger.error(f"Token {token[:8]}... is not claimed by this user")
            return False
        return True

    @staticmethod
    def _redeemable_tenant(token: str, value: dict) -> Optional[str]:
        """The token's tenant_id unless it is used or expired."""
        if value.get("used"):
            logger.warning(f"Token already used: {token[:8]}...")
            return None

        expires_at = value.get("expires_at", "")
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expiry:
                logger.warning(f"Token expired: {token[:8]}...")
                return None

        return value.get("tenant_id")


def _claimant(platform: str, external_user_id: str) -> dict:
    return {"platform": platform, "external_user_id": str(external_user_id)}


def _token_coordinates(token: str) -> tuple:
    """Store coordinates of a token record, as generate_token writes it."""
    return (
        canonical_tenant_id("_system"),
        ConfigScope.SYSTEM,
        "messaging_gateway",
        f"invite_token_{token}",
    )


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
            "session_id": _mapping_session_key(platform, str(external_user_id)),
        }

        try:
            # Store the mapping in the SYSTEM partition, NOT the user's own
            # tenant: get_tenant_id() runs before the tenant is known (that is
            # what it resolves), and Mem0 hard-partitions on user_id. Writing
            # under tenant_id here put the mapping in a partition the lookup
            # never searches, so every registered user looked unregistered.
            # The real tenant is preserved in the content text and metadata.
            self.memory_manager.add_memory(
                content=content,
                tenant_id=SYSTEM_TENANT_ID,
                agent_name=GATEWAY_AGENT_NAME,
                metadata=metadata,
                # Store verbatim: get_tenant_id parses "mapped to tenant <id>"
                # out of the content by substring, so the LLM extraction pass
                # (infer=True) must not be allowed to reword it — and a curated
                # mapping needs no extraction.
                infer=False,
            )
            logger.info(f"Registered {platform} user {external_user_id} → {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register user mapping: {e}")
            return False

    def get_tenant_id(self, platform: str, external_user_id: str) -> Optional[str]:
        """Look up tenant_id for an external user by exact metadata match.

        Narrows to this user's mapping server-side on the stamped
        ``session_id`` key, then confirms ``platform`` + ``external_user_id``
        by equality. Filtering server-side is what keeps a mapping past the
        partition's 100-row page reachable — a plain enumerate-and-scan sees
        only an arbitrary capped page and reports a registered user as
        unregistered once the partition grows past it.

        Raises on a Mem0 outage — flattening that to None made every
        registered user look unregistered for the outage's duration.
        """
        memories = self.memory_manager.get_all_memories(
            tenant_id=SYSTEM_TENANT_ID,
            agent_name=GATEWAY_AGENT_NAME,
            filters={"session_id": _mapping_session_key(platform, external_user_id)},
        )

        for mem in memories:
            meta = mem.get("metadata") or {}
            if (
                meta.get("type") == "user_mapping"
                and meta.get("platform") == platform
                and str(meta.get("external_user_id")) == str(external_user_id)
            ):
                return meta.get("tenant_id")
        return None
