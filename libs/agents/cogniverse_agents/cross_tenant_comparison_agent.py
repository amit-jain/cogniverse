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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

if TYPE_CHECKING:
    from cogniverse_agents.graph.graph_manager import GraphManager

logger = logging.getLogger(__name__)


def _node_ids_for(manager: "GraphManager") -> set:
    """Collect node_ids from a GraphManager by scanning all nodes."""
    ids: set = set()
    for node_fields in manager._visit(doc_type="node", top_k=500):
        doc_id = str(node_fields.get("doc_id") or "")
        if doc_id.startswith("kg_node_"):
            ids.add(doc_id.split("_", 3)[-1])
    return ids


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
        from cogniverse_agents._mm_factory import make_mm_factory

        self._config_manager = config_manager
        self._registry = registry or build_default_registry()
        self._mm_factory = make_mm_factory(memory_manager_factory)
        self._graph_managers: Dict[str, "GraphManager"] = {}
        self._trunk_graph_manager: Optional["GraphManager"] = None

    def set_graph_managers(
        self,
        per_tenant: Dict[str, "GraphManager"],
        trunk: Optional["GraphManager"] = None,
    ) -> None:
        """Bind per-tenant graph managers plus an optional org-trunk manager."""
        self._graph_managers = dict(per_tenant)
        self._trunk_graph_manager = trunk

    def compare(self, tenant_a: str, tenant_b: str) -> Dict[str, Dict[str, Any]]:
        """Diff the node sets of two tenants and (optionally) the org trunk.

        Returns ``{"diff": {"shared": [...sorted node_ids...], "tenant_only":
        {<tenant_a>: [...], <tenant_b>: [...]}, "trunk_only": [...]}}``.
        Diff is computed over node_ids only — Edge / Mention deltas are out
        of scope for this method and ``MultiDocumentSynthesisAgent`` covers
        per-segment evidence.
        """
        mgr_a = self._graph_managers.get(tenant_a)
        mgr_b = self._graph_managers.get(tenant_b)
        if mgr_a is None or mgr_b is None:
            raise RuntimeError(
                "CrossTenantComparisonAgent.compare requires graph managers for "
                f"both tenants; missing: "
                f"{[t for t, m in [(tenant_a, mgr_a), (tenant_b, mgr_b)] if m is None]}"
            )
        ids_a = _node_ids_for(mgr_a)
        ids_b = _node_ids_for(mgr_b)
        trunk_ids: set = (
            _node_ids_for(self._trunk_graph_manager)
            if self._trunk_graph_manager is not None
            else set()
        )
        shared = sorted(ids_a & ids_b)
        only_a = sorted(ids_a - ids_b - trunk_ids)
        only_b = sorted(ids_b - ids_a - trunk_ids)
        trunk_only = sorted(trunk_ids - (ids_a | ids_b))
        return {
            "diff": {
                "shared": shared,
                "tenant_only": {tenant_a: only_a, tenant_b: only_b},
                "trunk_only": trunk_only,
            }
        }

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
