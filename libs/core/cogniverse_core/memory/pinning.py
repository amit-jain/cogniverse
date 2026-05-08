"""Pinning service — A.6 of the harness plan.

Pinned memories are immune to lifecycle cleanup, trust decay, and any future
curator pass. The user-confirmed semantics:

  * ``pin_quota.user`` (default 50) — per-user limit for end-user pins.
  * ``pin_quota.tenant_admin`` (default 500) — tenant-admin limit for
    promoting tenant-wide knowledge.
  * ``pin_quota.org_admin`` — unlimited; org admins can override any pin
    and bump per-tenant quotas.

Implementation note: Mem0's ``update()`` only accepts content, not metadata,
so we cannot flip a flag on the target memory in place. Instead, a *pin
record* is a separate memory of kind ``pin_record`` whose content references
the target by id. This composes cleanly with the existing search/cleanup
plumbing — no metadata-update workaround needed — and gives the future
curator (and audit) a real history of pin actions per role.

A.7 will read pin records when building the per-schema cleanup batch so
pinned targets are skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    Pinnable,
    SchemaViolationError,
)

if TYPE_CHECKING:
    from cogniverse_core.memory.manager import Mem0MemoryManager

logger = logging.getLogger(__name__)

PIN_RECORD_KIND = "pin_record"
PIN_AGENT_NAME = "_pinning"  # Sentinel agent_name so pin_records do not
# mix into normal-agent search results.

# Default quotas. Org admins can bump per-tenant via TenantConfig metadata
# under the "pin_quota" key (see PinQuotas.from_tenant_config).
_DEFAULT_USER_QUOTA = 50
_DEFAULT_TENANT_ADMIN_QUOTA = 500


class PinQuotaExceededError(RuntimeError):
    """Raised when a pin would push the requester's quota above its limit."""


class PinAuthorityError(SchemaViolationError):
    """Raised when the requester's role is below the schema's pin floor."""


class PinNotFoundError(LookupError):
    """Raised when an unpin/inspect call references an unpinned memory."""


@dataclass(frozen=True)
class PinQuotas:
    """Per-role quotas for a tenant.

    ``None`` means unlimited. Org admin is unlimited by default; tenant admins
    can be capped via tenant config; user quotas are always capped (a small
    user-pinned set is a feature, not a free-for-all).
    """

    user: Optional[int] = _DEFAULT_USER_QUOTA
    tenant_admin: Optional[int] = _DEFAULT_TENANT_ADMIN_QUOTA
    org_admin: Optional[int] = None  # unlimited

    def limit_for(self, role: Pinnable) -> Optional[int]:
        if role is Pinnable.USER:
            return self.user
        if role is Pinnable.TENANT_ADMIN:
            return self.tenant_admin
        if role is Pinnable.ORG_ADMIN:
            return self.org_admin
        # Pinnable.NOBODY — has no quota; treat as 0.
        return 0

    @classmethod
    def from_tenant_config(cls, tenant_config: Optional[Any]) -> "PinQuotas":
        """Read overrides from TenantConfig.metadata['pin_quota'].

        Schema: ``{"user": 50, "tenant_admin": 500, "org_admin": null}``.
        Missing keys keep defaults. Use ``None``/``null`` for unlimited.
        """
        if tenant_config is None:
            return cls()
        meta = getattr(tenant_config, "metadata", {}) or {}
        overrides = meta.get("pin_quota") or {}
        return cls(
            user=overrides.get("user", _DEFAULT_USER_QUOTA),
            tenant_admin=overrides.get("tenant_admin", _DEFAULT_TENANT_ADMIN_QUOTA),
            org_admin=overrides.get("org_admin", None),
        )


@dataclass(frozen=True)
class PinRecord:
    """In-memory view of a pin_record memory."""

    memory_id: str  # ID of the pin_record memory itself
    target_memory_id: str  # ID of the memory being pinned
    pinned_by: Pinnable
    target_kind: str
    pinned_by_actor: str  # opaque actor id (user id, admin id) for audit


class PinService:
    """Pin / unpin / quota / inspection over Mem0+Vespa.

    Args:
        memory_manager: Live Mem0MemoryManager bound to the tenant.
        registry: KnowledgeRegistry that knows the target's pinnable_by floor.
        quotas: PinQuotas (default = constants above; build via
            ``PinQuotas.from_tenant_config(tc)`` to honour admin overrides).
    """

    def __init__(
        self,
        memory_manager: "Mem0MemoryManager",
        registry: KnowledgeRegistry,
        quotas: Optional[PinQuotas] = None,
    ) -> None:
        self._mm = memory_manager
        self._registry = registry
        self._quotas = quotas or PinQuotas()

    # --- public API -------------------------------------------------------

    def pin(
        self,
        target_memory_id: str,
        target_kind: str,
        pinned_by: Pinnable,
        actor_id: str,
        tenant_id: str,
    ) -> PinRecord:
        """Pin a memory, raising on quota or authority violations.

        Org admins can pin even when an existing pin from a lower role exists
        (override semantics). Tenant admins / users cannot override an
        existing pin from a higher role.
        """
        schema = self._registry.get(target_kind)
        # Pin operations only assert authority — provenance was checked when
        # the target memory was originally written.
        try:
            schema.validate_pin_authority(pinned_by)
        except SchemaViolationError as exc:
            raise PinAuthorityError(str(exc)) from exc

        # Override semantics: org admins can replace any existing pin. Lower
        # roles must yield to existing higher-role pins.
        existing = self._find_pin_records(target_memory_id, tenant_id)
        if existing:
            if pinned_by is not Pinnable.ORG_ADMIN:
                # Lower role attempting to pin something already pinned —
                # reject with PinAuthorityError so the caller knows it is
                # not a quota issue.
                raise PinAuthorityError(
                    f"target_memory_id={target_memory_id} is already pinned "
                    f"by a higher-or-equal role; org_admin can override"
                )
            # ORG_ADMIN override: drop existing pin records first.
            for rec in existing:
                self._mm.delete_memory(
                    memory_id=rec.memory_id,
                    tenant_id=tenant_id,
                    agent_name=PIN_AGENT_NAME,
                )

        # Enforce quota for the requesting role.
        limit = self._quotas.limit_for(pinned_by)
        if limit is not None:
            used = self.quota_used(pinned_by, tenant_id)
            if used >= limit:
                raise PinQuotaExceededError(
                    f"pin quota exhausted for role={pinned_by.value} "
                    f"(used={used}, limit={limit}); ask an org admin to bump"
                )

        content = (
            f"pin: target_memory_id={target_memory_id} "
            f"target_kind={target_kind} pinned_by={pinned_by.value} "
            f"actor={actor_id}"
        )
        metadata = {
            "kind": PIN_RECORD_KIND,
            "target_memory_id": target_memory_id,
            "target_kind": target_kind,
            "pinned_by": pinned_by.value,
            "actor_id": actor_id,
        }
        memory_id = self._mm.add_memory(
            content=content,
            tenant_id=tenant_id,
            agent_name=PIN_AGENT_NAME,
            metadata=metadata,
            infer=False,
        )
        return PinRecord(
            memory_id=memory_id,
            target_memory_id=target_memory_id,
            pinned_by=pinned_by,
            target_kind=target_kind,
            pinned_by_actor=actor_id,
        )

    def unpin(
        self,
        target_memory_id: str,
        requester: Pinnable,
        actor_id: str,
        tenant_id: str,
    ) -> int:
        """Remove pin records for a target. Returns number of records removed.

        Authority rules:
          * org_admin can unpin anything;
          * tenant_admin can unpin tenant_admin+user pins;
          * user can only unpin pins they themselves created (pinned_by=user
            AND actor_id matches).
        """
        records = self._find_pin_records(target_memory_id, tenant_id)
        if not records:
            raise PinNotFoundError(
                f"no pin records found for target_memory_id={target_memory_id}"
            )

        removable = [r for r in records if self._can_unpin(requester, actor_id, r)]
        if not removable:
            raise PinAuthorityError(
                f"requester role={requester.value} actor={actor_id} cannot "
                f"unpin existing pin records (held by higher role or "
                f"different user)"
            )

        for rec in removable:
            self._mm.delete_memory(
                memory_id=rec.memory_id,
                tenant_id=tenant_id,
                agent_name=PIN_AGENT_NAME,
            )
        return len(removable)

    def is_pinned(self, target_memory_id: str, tenant_id: str) -> bool:
        return bool(self._find_pin_records(target_memory_id, tenant_id))

    def list_pins(self, tenant_id: str) -> List[PinRecord]:
        """Return all pin records for a tenant — used by A.7 cleanup to skip
        pinned targets, and by admins for audit."""
        return self._all_pin_records(tenant_id)

    def quota_used(self, role: Pinnable, tenant_id: str) -> int:
        return sum(
            1 for rec in self._all_pin_records(tenant_id) if rec.pinned_by is role
        )

    def get_quotas(self) -> PinQuotas:
        return self._quotas

    # --- internals --------------------------------------------------------

    def _all_pin_records(self, tenant_id: str) -> List[PinRecord]:
        rows = self._mm.get_all_memories(tenant_id=tenant_id, agent_name=PIN_AGENT_NAME)
        return [r for r in (self._row_to_record(row) for row in rows) if r]

    def _find_pin_records(
        self, target_memory_id: str, tenant_id: str
    ) -> List[PinRecord]:
        return [
            rec
            for rec in self._all_pin_records(tenant_id)
            if rec.target_memory_id == target_memory_id
        ]

    @staticmethod
    def _row_to_record(row: Dict[str, Any]) -> Optional[PinRecord]:
        meta = row.get("metadata") or {}
        if isinstance(meta, str):
            import json

            try:
                meta = json.loads(meta)
            except Exception:
                return None
        if meta.get("kind") != PIN_RECORD_KIND:
            return None
        try:
            return PinRecord(
                memory_id=row.get("id", ""),
                target_memory_id=meta["target_memory_id"],
                pinned_by=Pinnable(meta["pinned_by"]),
                target_kind=meta.get("target_kind", "unknown"),
                pinned_by_actor=meta.get("actor_id", "unknown"),
            )
        except (KeyError, ValueError):
            logger.warning(
                "Skipping malformed pin_record %s; metadata=%r",
                row.get("id"),
                meta,
            )
            return None

    @staticmethod
    def _can_unpin(requester: Pinnable, actor_id: str, record: PinRecord) -> bool:
        if requester is Pinnable.ORG_ADMIN:
            return True
        if requester is Pinnable.TENANT_ADMIN:
            # Tenant admin can unpin user + tenant_admin pins, not org-admin pins.
            return record.pinned_by in (Pinnable.USER, Pinnable.TENANT_ADMIN)
        if requester is Pinnable.USER:
            # User can only remove their own pin record.
            return (
                record.pinned_by is Pinnable.USER and record.pinned_by_actor == actor_id
            )
        return False
