"""CrossTenantComparisonAgent.

Compares knowledge across multiple tenants within the *same* org. Built on
top of the federation read path: each per-tenant read goes through
``FederationService.federated_get_all`` so the org trunk is included
automatically and ACLs are enforced (the agent never reaches across orgs).

Use cases:
  * "How do our subsidiaries describe X?"
  * "Which tenants have a fact about subject_key Y, and how do they
    differ?"
  * Audit trail for org-trunk vs tenant-overlay drift.

The agent is read-only and does not call out to an LLM in V1. RLM-capable
summarisation can plug in later via the same pattern as the other agents.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import parse_tenant_id
from cogniverse_core.memory.federation import FederationService
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    Pinnable,
    SchemaViolationError,
    build_default_registry,
)

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8023


def _read_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
    import json as _json

    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            return _json.loads(meta) or {}
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


class CrossTenantComparisonInput(AgentInput):
    """What to compare and which tenants to compare across."""

    tenant_id: Optional[str] = Field(None, description="Caller tenant identifier")
    query: str = Field(
        "",
        description=(
            "Optional context label for telemetry; the comparison is driven "
            "by ``subject_key`` + ``tenant_ids``."
        ),
    )
    subject_key: str = Field(
        ...,
        description=(
            "Canonical subject identifier whose claims will be compared across tenants."
        ),
    )
    tenant_ids: List[str] = Field(
        ...,
        description=(
            "Tenants to compare; all must belong to the same org as the "
            "caller (enforced before any read happens)."
        ),
        min_length=2,
    )
    actor_role: str = Field(
        "tenant_admin",
        description=(
            "Caller's role for ACL purposes: must be tenant_admin or "
            "org_admin. Lower roles are rejected."
        ),
    )
    actor_id: str = Field(..., description="Caller actor id (for audit)")
    agent_name_filter: Optional[str] = Field(
        None,
        description=(
            "When provided, restrict the per-tenant federated read to this "
            "agent_name namespace. Defaults to ``_promoted`` so the org "
            "trunk's promoted records are included alongside tenant data."
        ),
    )


class TenantViewOut(AgentInput):
    """Per-tenant view of the subject."""

    tenant_id: str
    matching_memory_ids: List[str] = []
    excerpts: List[str] = []
    origin_tags: List[str] = []  # one entry per matching_memory_ids


class CrossTenantComparisonOutput(AgentOutput):
    subject_key: str = Field(..., description="The compared subject")
    tenant_views: List[TenantViewOut] = Field(default_factory=list)
    distinct_signatures_count: int = Field(
        0,
        description=(
            "Number of distinct content signatures observed across all "
            "tenant views — 1 means all tenants agree."
        ),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossTenantComparisonDeps(AgentDeps):
    pass


class _ACLRejected(SchemaViolationError):
    """Raised internally when the caller's role / tenant scope is invalid."""


class CrossTenantComparisonAgent(
    MemoryAwareMixin,
    A2AAgent[
        CrossTenantComparisonInput,
        CrossTenantComparisonOutput,
        CrossTenantComparisonDeps,
    ],
):
    """A2A agent that compares per-tenant views of a subject across an org."""

    def __init__(
        self,
        deps: CrossTenantComparisonDeps,
        memory_manager_factory=None,
        registry: Optional[KnowledgeRegistry] = None,
        config_manager=None,
        port: int = _DEFAULT_PORT,
    ):
        config = A2AAgentConfig(
            agent_name="cross_tenant_comparison_agent",
            agent_description=(
                "Compares per-tenant views of a subject across all tenants "
                "in an org via the FederationService read path."
            ),
            capabilities=[
                "cross_tenant_comparison",
                "audit",
                "federation_consumer",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        self._config_manager = config_manager
        self._registry = registry or build_default_registry()
        # Lazy default: when no factory is injected, the dispatcher's
        # standard Mem0MemoryManager singleton is used. Tests inject a stub.
        self._mm_factory = memory_manager_factory

    async def _process_impl(
        self, input: CrossTenantComparisonInput
    ) -> CrossTenantComparisonOutput:
        # ACL: caller must be tenant_admin or org_admin.
        try:
            actor_role = Pinnable(input.actor_role.lower())
        except ValueError:
            raise _ACLRejected(
                f"unknown actor_role={input.actor_role!r}; valid: "
                f"{[p.value for p in Pinnable]}"
            )
        if actor_role not in (Pinnable.TENANT_ADMIN, Pinnable.ORG_ADMIN):
            raise _ACLRejected(
                f"actor_role={actor_role.value} cannot run cross-tenant "
                "comparisons; tenant_admin or org_admin required"
            )

        # ACL: every requested tenant must be in the caller's org.
        if input.tenant_id:
            caller_org, _ = parse_tenant_id(input.tenant_id)
            for tid in input.tenant_ids:
                org_id, _ = parse_tenant_id(tid)
                if org_id != caller_org:
                    raise _ACLRejected(
                        f"tenant_id={tid!r} belongs to org={org_id!r} but "
                        f"caller is in org={caller_org!r}; cross-org "
                        "comparison is forbidden"
                    )

        if self._mm_factory is None:
            # Default to the standard Mem0MemoryManager singleton; in
            # production the dispatcher injects a different factory.
            from cogniverse_core.memory.manager import Mem0MemoryManager

            self._mm_factory = lambda tid: Mem0MemoryManager(tenant_id=tid)

        federation = FederationService(self._mm_factory, self._registry)
        agent_filter = input.agent_name_filter or "_promoted"

        tenant_views: List[TenantViewOut] = []
        all_signatures: set = set()

        for tid in input.tenant_ids:
            view_rows = federation.federated_get_all(tid, agent_filter)
            matching = [
                r
                for r in view_rows
                if _read_metadata(r).get("subject_key") == input.subject_key
            ]
            view_ids = [str(r.get("id") or "") for r in matching]
            view_excerpts = [
                str(r.get("memory") or r.get("content") or "")[:200] for r in matching
            ]
            view_origins = [
                str(r.get("_federation_origin") or "tenant") for r in matching
            ]

            for r in matching:
                content = str(r.get("memory") or r.get("content") or "").strip().lower()
                all_signatures.add(content)

            tenant_views.append(
                TenantViewOut(
                    tenant_id=tid,
                    matching_memory_ids=view_ids,
                    excerpts=view_excerpts,
                    origin_tags=view_origins,
                )
            )

        return CrossTenantComparisonOutput(
            subject_key=input.subject_key,
            tenant_views=tenant_views,
            distinct_signatures_count=len(all_signatures),
            metadata={
                "tenants_compared": len(input.tenant_ids),
                "actor_role": actor_role.value,
                "actor_id": input.actor_id,
                "agent_name_filter": agent_filter,
            },
        )
