"""Direct HTTP access to the knowledge-system agents.

The orchestrator's planner can only fill a generic 5-field input
(query, tenant_id, entities, relationships, enhanced_query) on dispatch.
Knowledge agents take richer inputs — memory_id (audit, citation),
subject_key (KG traversal, temporal), conflict_member_ids
(reconciliation), tenant_ids + actor_role (cross-tenant, federated),
documents (multi-document synthesis), windows (temporal). Without
dedicated routes these agents would Pydantic-error on the first
dispatched call.

Each agent gets its own route under ``/admin/tenants/{t}/knowledge/...``
whose request body matches the agent's input model exactly. Operators,
audit / compliance UIs, and admin scripts use these routes; the
orchestrator-routing path stays available for the agents whose input
shape can be filled from a generic query.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_factory(_tid: str):
    """Lazy memory manager factory — returns the per-tenant singleton.

    Uses the same lazy-init pathway the tenant router relies on so
    admin routes are reachable on dev/k3d topologies where Mem0 isn't
    pre-wired at startup.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager

    mm = Mem0MemoryManager(_tid)
    if not mm.memory:
        from cogniverse_runtime.routers import tenant as _tenant_router

        _tenant_router._lazy_init_memory(mm, _tid)
    if not mm.memory:
        raise HTTPException(
            status_code=503,
            detail=f"Memory backend not initialised for tenant {_tid}",
        )
    return mm


def _build_default_registry():
    from cogniverse_core.memory.schema import build_default_registry

    return build_default_registry()


def _inject_memory(agent, tenant_id: str, agent_name: str):
    """Wire a per-tenant Mem0MemoryManager onto an agent that takes no
    memory_manager_factory in its constructor.

    Several knowledge agents (citation_tracing, kg_traversal,
    multi_document_synthesis) have no factory parameter — they receive
    the manager via the MemoryAwareMixin attributes that the runtime
    usually populates. The admin path needs the same wiring.
    """
    mm = _build_factory(tenant_id)
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = tenant_id
    agent._memory_agent_name = agent_name
    return mm


# --- AuditExplanationAgent --------------------------------------------


class AuditExplainRequest(BaseModel):
    answer_memory_id: str = Field(..., min_length=1)
    include_trust: bool = True
    include_contradictions: bool = True
    max_chain_depth: int = Field(10, ge=1, le=25)
    max_chain_nodes: int = Field(100, ge=1, le=500)


@router.post("/tenants/{tenant_id}/knowledge/audit/explain")
async def audit_explain(tenant_id: str, body: AuditExplainRequest) -> Dict[str, Any]:
    """explain why a system answer was produced (read-only)."""
    from cogniverse_agents.audit_explanation_agent import (
        AuditExplanationAgent,
        AuditExplanationDeps,
        AuditExplanationInput,
    )

    agent = AuditExplanationAgent(
        deps=AuditExplanationDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
    )
    out = await agent._process_impl(
        AuditExplanationInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- CitationTracingAgent ---------------------------------------------


class CitationTraceRequest(BaseModel):
    memory_id: str = Field(..., min_length=1)
    max_depth: int = Field(10, ge=1, le=25)
    max_nodes: int = Field(100, ge=1, le=500)


@router.post("/tenants/{tenant_id}/knowledge/citations/trace")
async def citation_trace(tenant_id: str, body: CitationTraceRequest) -> Dict[str, Any]:
    """walk the provenance chain back to primary sources (read-only)."""
    from cogniverse_agents.citation_tracing_agent import (
        CitationTracingAgent,
        CitationTracingDeps,
        CitationTracingInput,
    )

    agent = CitationTracingAgent(deps=CitationTracingDeps(tenant_id=tenant_id))
    _inject_memory(agent, tenant_id, "citation_tracing_agent")
    out = await agent._process_impl(
        CitationTracingInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- KnowledgeSummarizationAgent --------------------------------------


class KnowledgeSummarizeRequest(BaseModel):
    subject_keys: List[str] = Field(..., min_length=1)
    kinds: Optional[List[str]] = None
    agent_name_filter: Optional[str] = None
    title: str = Field("Subject summary")
    actor_role: str = Field("user")
    actor_id: str = Field("admin")
    promote: bool = False


@router.post("/tenants/{tenant_id}/knowledge/summarize")
async def knowledge_summarize(
    tenant_id: str, body: KnowledgeSummarizeRequest
) -> Dict[str, Any]:
    """distill a subject slice into a structured summary."""
    from cogniverse_agents.knowledge_summarization_agent import (
        KnowledgeSummarizationAgent,
        KnowledgeSummarizationDeps,
        KnowledgeSummarizationInput,
    )

    agent = KnowledgeSummarizationAgent(
        deps=KnowledgeSummarizationDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
        registry=_build_default_registry(),
    )
    out = await agent._process_impl(
        KnowledgeSummarizationInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- ContradictionReconciliationAgent ---------------------------------


class ContradictionReconcileRequest(BaseModel):
    target_kind: str = Field(..., min_length=1)
    conflict_member_ids: List[str] = Field(..., min_length=1)
    policy_override: Optional[str] = None


@router.post("/tenants/{tenant_id}/knowledge/contradictions/reconcile")
async def contradiction_reconcile(
    tenant_id: str, body: ContradictionReconcileRequest
) -> Dict[str, Any]:
    """apply schema policy to a conflict set (write-capable)."""
    from cogniverse_agents.contradiction_reconciliation_agent import (
        ContradictionReconciliationAgent,
        ContradictionReconciliationDeps,
        ContradictionReconciliationInput,
    )

    agent = ContradictionReconciliationAgent(
        deps=ContradictionReconciliationDeps(tenant_id=tenant_id),
        registry=_build_default_registry(),
    )
    # Inject the per-tenant manager via the mixin attribute path; this
    # agent's constructor doesn't take a factory.
    mm = _build_factory(tenant_id)
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = tenant_id
    agent._memory_agent_name = "contradiction_reconciliation_agent"
    out = await agent._process_impl(
        ContradictionReconciliationInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- MultiDocumentSynthesisAgent --------------------------------------


class MultiDocSynthesizeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[Dict[str, Any]] = Field(..., min_length=1)
    actor_role: str = Field("user")
    actor_id: str = Field("admin")


@router.post("/tenants/{tenant_id}/knowledge/synthesis/multi_doc")
async def multi_doc_synthesize(
    tenant_id: str, body: MultiDocSynthesizeRequest
) -> Dict[str, Any]:
    """synthesise an answer across N documents with citations."""
    from cogniverse_agents.multi_document_synthesis_agent import (
        MultiDocSynthesisAgent,
        MultiDocSynthesisDeps,
        MultiDocSynthesisInput,
    )

    agent = MultiDocSynthesisAgent(deps=MultiDocSynthesisDeps(tenant_id=tenant_id))
    _inject_memory(agent, tenant_id, "multi_document_synthesis_agent")
    out = await agent._process_impl(
        MultiDocSynthesisInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- KGTraversalAgent -------------------------------------------------


class KGTraverseRequest(BaseModel):
    start_subject_key: str = Field(..., min_length=1)
    relation_filter: Optional[List[str]] = None
    max_depth: int = Field(3, ge=1, le=10)
    max_nodes: int = Field(50, ge=1, le=500)


@router.post("/tenants/{tenant_id}/knowledge/kg/traverse")
async def kg_traverse(tenant_id: str, body: KGTraverseRequest) -> Dict[str, Any]:
    """walk the entity / kg graph from a starting subject (read-only)."""
    from cogniverse_agents.kg_traversal_agent import (
        KGTraversalDeps,
        KGTraversalInput,
        KnowledgeGraphTraversalAgent,
    )

    agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id=tenant_id))
    _inject_memory(agent, tenant_id, "kg_traversal_agent")
    # KGTraversalInput uses ``relation_allowlist`` / ``max_edges``; the
    # public route field names are ``relation_filter`` / ``max_nodes``
    # for symmetry with the citation-trace request shape. Translate
    # before construction so the agent input doesn't reject extra keys.
    out = await agent._process_impl(
        KGTraversalInput(
            tenant_id=tenant_id,
            start_subject_key=body.start_subject_key,
            relation_allowlist=body.relation_filter,
            max_depth=body.max_depth,
            max_edges=body.max_nodes,
        )
    )
    return out.model_dump()


# --- CrossTenantComparisonAgent ---------------------------------------


class CrossTenantCompareRequest(BaseModel):
    subject_key: str = Field(..., min_length=1)
    tenant_ids: List[str] = Field(..., min_length=2)
    # actor_role must be a Pinnable enum value (tenant_admin / org_admin)
    # — the agent rejects everything else via _ACLRejected. Default to
    # tenant_admin so the route's documented default actually works.
    actor_role: str = Field("tenant_admin")
    actor_id: str = Field("admin")
    agent_name_filter: Optional[str] = Field(
        None,
        description=(
            "Restrict each per-tenant federated read to this agent_name "
            "namespace. Defaults to ``_promoted`` (matches federation writes)."
        ),
    )


@router.post("/tenants/{tenant_id}/knowledge/cross_tenant/compare")
async def cross_tenant_compare(
    tenant_id: str, body: CrossTenantCompareRequest
) -> Dict[str, Any]:
    """compare knowledge across org tenants for a subject (admin)."""
    from cogniverse_agents.cross_tenant_comparison_agent import (
        CrossTenantComparisonAgent,
        CrossTenantComparisonDeps,
        CrossTenantComparisonInput,
    )
    from cogniverse_agents.cross_tenant_comparison_agent import (
        _ACLRejected as _ACLRejectedCT,
    )

    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
        registry=_build_default_registry(),
    )
    try:
        out = await agent._process_impl(
            CrossTenantComparisonInput(tenant_id=tenant_id, **body.model_dump())
        )
    except _ACLRejectedCT as exc:
        raise HTTPException(403, str(exc)) from exc
    return out.model_dump()


# --- FederatedQueryAgent ----------------------------------------------


class FederatedQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_ids: List[str] = Field(..., min_length=1)
    # Same Pinnable enum constraint as cross_tenant — default to a
    # value that actually passes the agent's ACL.
    actor_role: str = Field("tenant_admin")
    actor_id: str = Field("admin")
    # FederatedQueryInput's field is ``top_k_per_tenant``; expose it
    # under the public name ``top_k`` and translate at dispatch.
    top_k: int = Field(10, ge=1, le=200)
    agent_name_filter: Optional[str] = Field(None)


@router.post("/tenants/{tenant_id}/knowledge/federated/query")
async def federated_query(
    tenant_id: str, body: FederatedQueryRequest
) -> Dict[str, Any]:
    """issue a single query against multiple tenants (admin, read-only)."""
    from cogniverse_agents.federated_query_agent import (
        FederatedQueryAgent,
        FederatedQueryDeps,
        FederatedQueryInput,
    )
    from cogniverse_agents.federated_query_agent import (
        _ACLRejected as _ACLRejectedFQ,
    )

    agent = FederatedQueryAgent(
        deps=FederatedQueryDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
        registry=_build_default_registry(),
    )
    try:
        out = await agent._process_impl(
            FederatedQueryInput(
                tenant_id=tenant_id,
                query=body.query,
                tenant_ids=body.tenant_ids,
                actor_role=body.actor_role,
                actor_id=body.actor_id,
                top_k_per_tenant=body.top_k,
                agent_name_filter=body.agent_name_filter,
            )
        )
    except _ACLRejectedFQ as exc:
        raise HTTPException(403, str(exc)) from exc
    return out.model_dump()


# --- TemporalReasoningAgent -------------------------------------------


class TemporalReasonRequest(BaseModel):
    subject_key: str = Field(..., min_length=1)
    windows: List[Dict[str, Any]] = Field(..., min_length=1)


@router.post("/tenants/{tenant_id}/knowledge/temporal/reason")
async def temporal_reason(
    tenant_id: str, body: TemporalReasonRequest
) -> Dict[str, Any]:
    """compare knowledge of a subject across time windows (read-only)."""
    from cogniverse_agents.temporal_reasoning_agent import (
        TemporalReasoningAgent,
        TemporalReasoningDeps,
        TemporalReasoningInput,
    )

    agent = TemporalReasoningAgent(
        deps=TemporalReasoningDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
    )
    out = await agent._process_impl(
        TemporalReasoningInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()
