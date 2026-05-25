"""Federation: org trunk + tenant overlays.

Multi-tenant orgs need to share a "trunk" of knowledge while letting each
tenant overlay tenant-specific facts. This module gives:

  * a federated **read** that pulls candidates from both the tenant's own
    Mem0 namespace and the org's trunk, dedups by ``subject_key`` with
    the tenant overlay winning;
  * an admin-gated **promote** that copies a tenant memory into the org
    trunk so every tenant under the org can see it.

Storage: a separate tenant_id ``<org>:_org_trunk`` stands in for the
"org trunk" Vespa schema. Mem0+Vespa already isolate per-tenant_id, so
no new backend wiring is required — the trunk just looks like a tenant
that no end-user owns directly. The convention is enforced here so that
``KnowledgeSchema.sensitivity`` rules line up:

  * ``tenant_private`` memories never get promoted (refused)
  * ``org_shared`` memories are eligible for promotion
  * ``global_shared`` is reserved for a future cross-org channel

ACLs: the read path only ever reads from the caller's tenant + that
tenant's org trunk, so leakage across orgs is prevented at query time.
The promotion path requires a ``tenant_admin`` or ``org_admin`` actor
(checked against schema's ``pinnable_by`` floor — admins who can pin can
also promote). User-supplied promotions are rejected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cogniverse_core.common.tenant_utils import parse_tenant_id
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    Pinnable,
    SchemaViolationError,
    Sensitivity,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ORG_TRUNK_TENANT_SUFFIX = "_org_trunk"


def org_trunk_tenant_id(tenant_id: str) -> str:
    """Return the canonical org-trunk tenant_id for ``tenant_id``.

    >>> org_trunk_tenant_id("acme:production")
    'acme:_org_trunk'
    >>> org_trunk_tenant_id("acme")
    'acme:_org_trunk'
    """
    org_id, _ = parse_tenant_id(tenant_id)
    return f"{org_id}:{ORG_TRUNK_TENANT_SUFFIX}"


class FederationDeniedError(SchemaViolationError):
    """Raised when a promotion is rejected by the schema sensitivity gate."""


class ACLRejected(SchemaViolationError):
    """Raised when a caller's role / tenant scope is invalid for a federated read.

    The federated-query and cross-tenant-comparison agents both gate their
    read paths on the caller's role and the tenant set belonging to the
    caller's org; both raise this when that gate fails.
    """


def _read_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            return json.loads(meta) or {}
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _subject_key(memory: Dict[str, Any]) -> Optional[str]:
    return _read_metadata(memory).get("subject_key")


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a single tenant→org-trunk promotion."""

    source_memory_id: str
    promoted_memory_id: str
    org_trunk_tenant_id: str


class FederationService:
    """Federated read + admin-gated promote operations.

    Args:
        memory_manager_factory: Callable ``tenant_id -> Mem0MemoryManager``.
            Cogniverse's Mem0MemoryManager is a per-tenant singleton, so
            this is typically just the class itself; tests can inject a
            fake factory that returns stubs keyed by tenant_id.
        registry: KnowledgeRegistry used to look up sensitivity / pin floor
            during a promotion.
    """

    def __init__(
        self,
        memory_manager_factory,
        registry: KnowledgeRegistry,
    ) -> None:
        self._mm_factory = memory_manager_factory
        self._registry = registry

    # --- read path ---------------------------------------------------------

    def federated_get_all(
        self,
        tenant_id: str,
        agent_name: str,
    ) -> List[Dict[str, Any]]:
        """Return tenant + org-trunk memories, deduped by subject_key.

        Tenant overlays win when the same subject_key appears on both
        sides — a tenant can shadow an org claim without modifying the
        trunk. Memories without a ``subject_key`` are kept as-is from
        whichever side they came from (no dedup possible).
        """
        tenant_rows = self._fetch(tenant_id, agent_name)
        org_rows = self._fetch(org_trunk_tenant_id(tenant_id), agent_name)

        # Tag origin so callers know where each memory came from.
        for r in tenant_rows:
            r.setdefault("_federation_origin", "tenant")
        for r in org_rows:
            r.setdefault("_federation_origin", "org_trunk")

        # Dedup: prefer tenant on subject_key collisions.
        chosen: Dict[str, Dict[str, Any]] = {}
        unsubjected: List[Dict[str, Any]] = []
        for row in tenant_rows + org_rows:
            subject = _subject_key(row)
            if not subject:
                unsubjected.append(row)
                continue
            if subject in chosen:
                # tenant rows come first → existing entry stays (tenant wins).
                continue
            chosen[subject] = row
        return list(chosen.values()) + unsubjected

    def _fetch(self, tenant_id: str, agent_name: str) -> List[Dict[str, Any]]:
        try:
            mm = self._mm_factory(tenant_id)
        except Exception as exc:
            logger.debug("Federation: factory(%s) failed: %s", tenant_id, exc)
            return []
        if mm is None or not getattr(mm, "memory", None):
            return []
        try:
            return list(mm.get_all_memories(tenant_id=tenant_id, agent_name=agent_name))
        except Exception as exc:
            logger.debug("Federation: get_all_memories(%s) failed: %s", tenant_id, exc)
            return []

    # --- promotion path ----------------------------------------------------

    def promote_to_org_trunk(
        self,
        *,
        source_tenant_id: str,
        source_memory: Dict[str, Any],
        actor_role: Pinnable,
        actor_id: str,
    ) -> PromotionResult:
        """Copy a memory into the org trunk so every tenant in the org sees it.

        Raises:
            FederationDeniedError: when the schema's ``sensitivity`` is
                ``tenant_private`` (never promotable) or the actor's role
                is below the schema's ``pinnable_by`` floor.
        """
        meta = _read_metadata(source_memory)
        kind = str(meta.get("kind") or "")
        if not kind:
            raise FederationDeniedError(
                "memory has no metadata.kind; cannot determine sensitivity policy"
            )

        schema = self._registry.get(kind)
        if schema.sensitivity is Sensitivity.TENANT_PRIVATE:
            raise FederationDeniedError(
                f"kind={kind!r} is tenant_private; promotion to the org "
                "trunk is forbidden by the schema sensitivity policy"
            )

        # Authority: a promoter must be at-or-above the schema's pin floor.
        # Tenant admins can promote anything pinnable by tenant admins;
        # org admins can promote anything.
        try:
            schema.validate_pin_authority(actor_role)
        except SchemaViolationError as exc:
            raise FederationDeniedError(
                f"actor_role={actor_role.value} insufficient for kind={kind!r}: {exc}"
            ) from exc

        target_tenant = org_trunk_tenant_id(source_tenant_id)
        try:
            target_mm = self._mm_factory(target_tenant)
        except Exception as exc:
            raise FederationDeniedError(
                f"could not open org trunk for {target_tenant}: {exc}"
            ) from exc

        new_metadata = dict(meta)
        # Mark provenance: every promoted record carries a federation
        # provenance stamp so audit can tell trunk-original vs promoted.
        new_metadata["promoted_from_tenant"] = source_tenant_id
        new_metadata["promoted_by"] = actor_id
        new_metadata["promoted_by_role"] = actor_role.value

        promoted_id = target_mm.add_memory(
            content=source_memory.get("memory") or source_memory.get("content") or "",
            tenant_id=target_tenant,
            agent_name=source_memory.get("agent_name") or "_promoted",
            metadata=new_metadata,
            infer=False,
        )
        return PromotionResult(
            source_memory_id=str(source_memory.get("id") or ""),
            promoted_memory_id=str(promoted_id or ""),
            org_trunk_tenant_id=target_tenant,
        )
