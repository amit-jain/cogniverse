"""Hashed harness credentials and immutable revocations in ConfigStore."""

import hashlib
import re
import secrets
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID, canonical_tenant_id
from cogniverse_sdk.interfaces.config_store import ConfigScope, ImmutableConfigStore

_SERVICE = "harness_keys"
_REVOCATIONS = "harness_key_revocations"
MAX_TTL_SECONDS = 7 * 24 * 3600


class HarnessKeyNotFoundError(LookupError):
    """The presented credential is absent, revoked or expired."""


def generate_key() -> tuple[str, str]:
    plaintext = f"cgv-{secrets.token_urlsafe(32)}"
    return plaintext, hashlib.sha256(plaintext.encode()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HarnessKeyStore:
    """Canonical tenant credentials; only create returns plaintext.

    A key created with ``ttl_seconds`` stops authenticating at its
    ``expires_at``; one created without never expires.
    """

    def __init__(
        self,
        config_store: ImmutableConfigStore,
        now: Callable[[], datetime] = _utc_now,
    ):
        self._store = config_store
        self._now = now

    def create(self, tenant_id: str, name: str, ttl_seconds: int | None = None) -> dict:
        tenant_id = canonical_tenant_id(tenant_id)
        if not name.strip():
            raise ValueError("name must not be blank")
        if ttl_seconds is not None and not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
            raise ValueError(
                f"ttl_seconds must be from 1 to {MAX_TTL_SECONDS}, got {ttl_seconds}"
            )
        plaintext, key_hash = generate_key()
        created_at = self._now()
        record = {
            "tenant_id": tenant_id,
            "name": name,
            "created_at": created_at.isoformat(),
            "expires_at": (
                None
                if ttl_seconds is None
                else (created_at + timedelta(seconds=ttl_seconds)).isoformat()
            ),
            "revoked": False,
        }
        self._store.put_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            _SERVICE,
            key_hash,
            record,
        )
        return {"key": plaintext, **self._public(key_hash, record)}

    @staticmethod
    def _public(key_hash: str, record: dict) -> dict:
        return {
            "key_hash": key_hash,
            "key_prefix": key_hash[:12],
            "tenant_id": record["tenant_id"],
            "name": record["name"],
            "created_at": record["created_at"],
            "expires_at": record.get("expires_at"),
            "revoked": record["revoked"],
        }

    def _expired(self, record: dict) -> bool:
        expires_at = record.get("expires_at")
        return expires_at is not None and self._now() >= datetime.fromisoformat(
            expires_at
        )

    def _revoked(self, key_hash: str) -> bool:
        entry = self._store.get_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            _REVOCATIONS,
            key_hash,
        )
        if entry is None:
            return False
        if entry.config_value != {"revoked": True}:
            raise ValueError(f"Invalid revocation record for {key_hash}")
        return True

    def resolve(self, plaintext: str) -> str:
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        entry = self._store.get_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            _SERVICE,
            key_hash,
        )
        if (
            entry is None
            or self._expired(entry.config_value)
            or self._revoked(key_hash)
        ):
            raise HarnessKeyNotFoundError("Harness key not found")
        return canonical_tenant_id(entry.config_value["tenant_id"])

    def revoke(self, key_hash: str) -> bool:
        """Idempotently revoke a hash with one write and one confirming read."""
        if re.fullmatch(r"[0-9a-f]{64}", key_hash) is None:
            raise ValueError("key_hash must be 64 lowercase hexadecimal characters")
        self._store.put_immutable_config(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            _REVOCATIONS,
            key_hash,
            {"revoked": True},
        )
        return True

    def _list(
        self, tenant_id: str | None, page_size: int, continuation: str | None
    ) -> dict:
        entries, continuation = self._store.list_immutable_configs(
            SYSTEM_TENANT_ID,
            ConfigScope.SYSTEM,
            _SERVICE,
            page_size=page_size,
            continuation=continuation,
        )
        records = []
        for entry in entries:
            record = entry.config_value
            if tenant_id is None or record["tenant_id"] == tenant_id:
                records.append(
                    self._public(
                        entry.config_key,
                        {
                            **record,
                            "revoked": self._expired(record)
                            or self._revoked(entry.config_key),
                        },
                    )
                )
        return {"keys": records, "continuation": continuation}

    def list(
        self,
        tenant_id: str | None = None,
        *,
        page_size: int = 100,
        continuation: str | None = None,
    ) -> dict:
        """Return one bounded page; an empty page may have a continuation."""
        tenant_id = canonical_tenant_id(tenant_id) if tenant_id is not None else None
        return self._list(tenant_id, page_size, continuation)

    def revoke_tenant(self, tenant_id: str) -> int:
        tenant_id = canonical_tenant_id(tenant_id)
        continuation = None
        count = 0
        while True:
            page = self._list(tenant_id, 100, continuation)
            for record in page["keys"]:
                if not record["revoked"]:
                    self.revoke(record["key_hash"])
                    count += 1
            continuation = page["continuation"]
            if continuation is None:
                return count
