"""FederatedQueryAgent.

Issues a single user query against multiple tenants under the same org and
merges the results, with org-trunk records included via federation.
Differs from CrossTenantComparisonAgent: that agent *compares* tenant
views of one subject; this agent *answers* a free-text query by aggregating
matches across tenants.

ACL contract (mirrors CrossTenantComparisonAgent):
  * caller's role must be ``tenant_admin`` or ``org_admin``
  * every queried tenant must belong to the caller's org

Read-only — does not write to memory. RLM-capable when the merged result
context is large.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_core.common.tenant_utils import parse_tenant_id
from cogniverse_core.memory.federation import ACLRejected, FederationService
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    Pinnable,
    build_default_registry,
)

if TYPE_CHECKING:
    from cogniverse_agents.graph.graph_manager import GraphManager

logger = logging.getLogger(__name__)


def _parse_mentions(raw: Any) -> List[Dict[str, Any]]:
    """Parse the JSON-string ``mentions`` field on a Node row."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return []
    elif isinstance(raw, list):
        parsed = raw
    else:
        return []
    return [m for m in parsed if isinstance(m, dict)]


_DEFAULT_PORT = 8024


class FederatedQueryInput(AgentInput):
    tenant_id: Optional[str] = Field(None, description="Caller tenant identifier")
    query: str = Field(..., description="Free-text query to issue across tenants")
    tenant_ids: List[str] = Field(
        ...,
        description=(
            "Tenants to search; all must belong to the caller's org. "
            "Federated reads include each tenant's org-trunk automatically."
        ),
        min_length=1,
    )
    actor_role: str = Field(
        "tenant_admin",
        description="tenant_admin or org_admin (lower roles rejected)",
    )
    actor_id: str = Field(..., description="Caller actor id (audit)")
    agent_name_filter: Optional[str] = Field(
        None,
        description=(
            "When set, restrict each per-tenant federated read to this "
            "agent_name namespace. Defaults to ``_promoted``."
        ),
    )
    top_k_per_tenant: int = Field(
        20,
        ge=1,
        le=200,
        description="Cap on rows fetched from each tenant before merging",
    )
    rlm: Optional[RLMOptions] = Field(
        None,
        description=(
            "Optional RLM summariser. When enabled (or auto-detected past "
            "the threshold), the merged context is summarised."
        ),
    )


class FederatedHitOut(AgentInput):
    tenant_id: str
    memory_id: str
    excerpt: str
    origin: str  # "tenant" | "org_trunk"


class FederatedQueryOutput(AgentOutput):
    query: str = Field(..., description="The query that was answered")
    hits: List[FederatedHitOut] = Field(default_factory=list)
    summary: Optional[str] = Field(
        None,
        description="LLM-summarised answer when ``rlm`` ran",
    )
    used_rlm: bool = Field(False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FederatedQueryDeps(AgentDeps):
    pass


def _matches_query(memory: Dict[str, Any], query: str) -> bool:
    """Tiny case-insensitive substring match.

    The plan reserves richer semantic matching (embedding cosine over
    Mem0's vector) for a follow-up; V1 uses substring so the agent has
    a deterministic, testable contract today.
    """
    if not query:
        return True
    q = query.lower()
    content = (memory.get("memory") or memory.get("content") or "").lower()
    return q in content


class FederatedQueryAgent(
    MemoryAwareMixin,
    A2AAgent[FederatedQueryInput, FederatedQueryOutput, FederatedQueryDeps],
):
    """A2A agent that answers a query by reading federated rows across tenants."""

    def __init__(
        self,
        deps: FederatedQueryDeps,
        memory_manager_factory=None,
        registry: Optional[KnowledgeRegistry] = None,
        llm_config=None,
        config_manager=None,
        port: int = _DEFAULT_PORT,
    ):
        config = A2AAgentConfig(
            agent_name="federated_query_agent",
            agent_description=(
                "Answers a free-text query by aggregating federated reads "
                "across multiple tenants in the same org with ACL "
                "enforcement and an optional RLM summariser."
            ),
            capabilities=[
                "federated_query",
                "audit",
                "federation_consumer",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        from cogniverse_agents._mm_factory import make_mm_factory

        from cogniverse_agents._llm_resolution import resolve_llm_config

        self._config_manager = config_manager
        self._registry = registry or build_default_registry()
        self._mm_factory = make_mm_factory(memory_manager_factory)
        # Fall back to the system primary LM via config_manager when no
        # explicit llm_config was passed; gives the constructor param a real
        # consumer instead of being a dead injection point.
        self._llm_config = resolve_llm_config(llm_config, config_manager)
        self._graph_managers: Dict[str, "GraphManager"] = {}

    def set_graph_managers(self, graph_managers: Dict[str, "GraphManager"]) -> None:
        """Bind one GraphManager per tenant/overlay name for ``.query``."""
        self._graph_managers = dict(graph_managers)

    def query(
        self, text: str, tenants_or_overlays: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Federated graph-level query that dedupes nodes across sources.

        For each bound (tenant|overlay) name in ``tenants_or_overlays``, scans
        the bound GraphManager's nodes and keeps the ones whose ``name``
        contains ``text`` (case-insensitive). Results are deduplicated by
        ``node_id`` (the normalised name) with sources merged. Each result:
        ``{node_id, sources (sorted unique), merged_mentions_count (count of
        distinct ``(source_doc_id, segment_id)`` tuples across all sources)}``.
        Output list ordered by ``node_id`` ascending.
        """
        if not self._graph_managers:
            raise RuntimeError(
                "FederatedQueryAgent.query requires graph managers — call "
                "set_graph_managers({name: GraphManager, ...}) first."
            )
        needle = (text or "").lower()
        accum: Dict[str, Dict[str, Any]] = {}
        for name in tenants_or_overlays:
            mgr = self._graph_managers.get(name)
            if mgr is None:
                continue
            for node_fields in mgr._visit(doc_type="node", top_k=500):
                node_name = str(node_fields.get("name") or "")
                if needle and needle not in node_name.lower():
                    continue
                doc_id = str(node_fields.get("doc_id") or "")
                # doc_id format: ``kg_node_{tenant}_{node_id}``; the trailing
                # node_id is the normalised name. Recover it from the suffix.
                node_id = (
                    doc_id.split("_", 3)[-1] if doc_id.startswith("kg_node_") else ""
                )
                if not node_id:
                    continue
                entry = accum.setdefault(
                    node_id,
                    {"node_id": node_id, "sources": set(), "mention_keys": set()},
                )
                entry["sources"].add(name)
                for m in _parse_mentions(node_fields.get("mentions")):
                    entry["mention_keys"].add(
                        (
                            str(m.get("source_doc_id") or ""),
                            str(m.get("segment_id") or ""),
                        )
                    )
        results: List[Dict[str, Any]] = []
        for node_id in sorted(accum):
            entry = accum[node_id]
            results.append(
                {
                    "node_id": node_id,
                    "sources": sorted(entry["sources"]),
                    "merged_mentions_count": len(entry["mention_keys"]),
                }
            )
        return {"results": results}

    async def _process_impl(self, input: FederatedQueryInput) -> FederatedQueryOutput:
        # ACL: role + cross-org checks (mirror CrossTenantComparisonAgent).
        try:
            actor_role = Pinnable(input.actor_role.lower())
        except ValueError:
            raise ACLRejected(
                f"unknown actor_role={input.actor_role!r}; valid: "
                f"{[p.value for p in Pinnable]}"
            )
        if actor_role not in (Pinnable.TENANT_ADMIN, Pinnable.ORG_ADMIN):
            raise ACLRejected(
                f"actor_role={actor_role.value} cannot issue federated "
                "queries; tenant_admin or org_admin required"
            )

        if input.tenant_id:
            caller_org, _ = parse_tenant_id(input.tenant_id)
            for tid in input.tenant_ids:
                org_id, _ = parse_tenant_id(tid)
                if org_id != caller_org:
                    raise ACLRejected(
                        f"tenant_id={tid!r} belongs to org={org_id!r} but "
                        f"caller is in org={caller_org!r}; cross-org query "
                        "is forbidden"
                    )

        federation = FederationService(self._mm_factory, self._registry)
        agent_filter = input.agent_name_filter or "_promoted"

        hits: List[FederatedHitOut] = []
        per_tenant_counts: Dict[str, int] = {}
        for tid in input.tenant_ids:
            rows = federation.federated_get_all(tid, agent_filter)
            matched = [r for r in rows if _matches_query(r, input.query)][
                : input.top_k_per_tenant
            ]
            per_tenant_counts[tid] = len(matched)
            for r in matched:
                hits.append(
                    FederatedHitOut(
                        tenant_id=tid,
                        memory_id=str(r.get("id") or ""),
                        excerpt=str(r.get("memory", ""))[:200],
                        origin=str(r.get("_federation_origin") or "tenant"),
                    )
                )

        used_rlm = False
        summary: Optional[str] = None
        rlm_options = input.rlm
        if rlm_options is not None and hits:
            block = self._format_hits_for_summary(hits)
            if rlm_options.should_use_rlm(len(block)):
                summary = await self._summarise_with_rlm(
                    input.query, block, rlm_options
                )
                used_rlm = True

        return FederatedQueryOutput(
            query=input.query,
            hits=hits,
            summary=summary,
            used_rlm=used_rlm,
            metadata={
                "tenants_queried": len(input.tenant_ids),
                "hit_count": len(hits),
                "per_tenant_counts": per_tenant_counts,
                "actor_role": actor_role.value,
                "actor_id": input.actor_id,
                "agent_name_filter": agent_filter,
            },
        )

    @staticmethod
    def _format_hits_for_summary(hits: List[FederatedHitOut]) -> str:
        lines: List[str] = []
        for h in hits:
            lines.append(f"[{h.tenant_id}/{h.origin}] {h.memory_id}: {h.excerpt}")
        return "\n".join(lines)

    async def _summarise_with_rlm(
        self, query: str, block: str, rlm_options: RLMOptions
    ) -> str:
        from cogniverse_agents.inference.rlm_inference import build_rlm_from_options

        rlm = build_rlm_from_options(self._llm_config, rlm_options)
        result = rlm.process(query=query, context=block)
        return result.answer
