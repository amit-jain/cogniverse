"""C3 admin endpoints — direct HTTP access to the 9 knowledge-system agents.

Audit found that 8 of 9 C3 agents were orchestrator-unreachable: even
with ``enabled: true`` in config, the dispatcher's
``_execute_generic_agent`` only fills 5 input fields (query, tenant_id,
entities, relationships, enhanced_query) so any agent requiring extra
fields (memory_id, subject_key, conflict_member_ids, …) Pydantic-errors
on first call. The orchestrator's planner cannot synthesise these
fields from a free-form user query.

These endpoints close the gap: each C3 agent gets a dedicated admin
route under ``/admin/tenants/{t}/c3/...`` whose request body matches the
agent's input model exactly. Operators (and audit / compliance UIs)
get a documented HTTP surface; the orchestrator-routing path remains
available for the agents that can be filled from a generic query
(future work — see ``test_c3_agent_reachability.py`` for the planner
discovery contract).
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

    Several C3 agents (citation_tracing, kg_traversal, multi_document_synthesis)
    have no factory parameter — they're meant to receive the manager via
    the MemoryAwareMixin attributes that the runtime usually populates.
    The admin path needs the same wiring.
    """
    mm = _build_factory(tenant_id)
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = tenant_id
    agent._memory_agent_name = agent_name
    return mm


# --- C3.9 AuditExplanationAgent --------------------------------------------


class AuditExplainRequest(BaseModel):
    answer_memory_id: str = Field(..., min_length=1)
    include_trust: bool = True
    include_contradictions: bool = True
    max_chain_depth: int = Field(10, ge=1, le=25)
    max_chain_nodes: int = Field(100, ge=1, le=500)


@router.post("/tenants/{tenant_id}/c3/audit/explain")
async def audit_explain(tenant_id: str, body: AuditExplainRequest) -> Dict[str, Any]:
    """C3.9 — explain why a system answer was produced (read-only)."""
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


# --- C3.5 CitationTracingAgent ---------------------------------------------


class CitationTraceRequest(BaseModel):
    memory_id: str = Field(..., min_length=1)
    max_depth: int = Field(10, ge=1, le=25)
    max_nodes: int = Field(100, ge=1, le=500)


@router.post("/tenants/{tenant_id}/c3/citations/trace")
async def citation_trace(tenant_id: str, body: CitationTraceRequest) -> Dict[str, Any]:
    """C3.5 — walk the provenance chain back to primary sources (read-only)."""
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


# --- C3.8 KnowledgeSummarizationAgent --------------------------------------


class KnowledgeSummarizeRequest(BaseModel):
    subject_keys: List[str] = Field(..., min_length=1)
    kinds: Optional[List[str]] = None
    agent_name_filter: Optional[str] = None
    title: str = Field("Subject summary")
    actor_role: str = Field("user")
    actor_id: str = Field("admin")
    promote: bool = False


@router.post("/tenants/{tenant_id}/c3/knowledge/summarize")
async def knowledge_summarize(
    tenant_id: str, body: KnowledgeSummarizeRequest
) -> Dict[str, Any]:
    """C3.8 — distill a subject slice into a structured summary."""
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


# --- C3.4 ContradictionReconciliationAgent ---------------------------------


class ContradictionReconcileRequest(BaseModel):
    target_kind: str = Field(..., min_length=1)
    conflict_member_ids: List[str] = Field(..., min_length=1)
    policy_override: Optional[str] = None


@router.post("/tenants/{tenant_id}/c3/contradictions/reconcile")
async def contradiction_reconcile(
    tenant_id: str, body: ContradictionReconcileRequest
) -> Dict[str, Any]:
    """C3.4 — apply schema policy to a conflict set (write-capable)."""
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


# --- C3.1 MultiDocumentSynthesisAgent --------------------------------------


class MultiDocSynthesizeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: List[Dict[str, Any]] = Field(..., min_length=1)
    actor_role: str = Field("user")
    actor_id: str = Field("admin")


@router.post("/tenants/{tenant_id}/c3/synthesis/multi_doc")
async def multi_doc_synthesize(
    tenant_id: str, body: MultiDocSynthesizeRequest
) -> Dict[str, Any]:
    """C3.1 — synthesise an answer across N documents with citations."""
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


# --- C3.2 KGTraversalAgent -------------------------------------------------


class KGTraverseRequest(BaseModel):
    start_subject_key: str = Field(..., min_length=1)
    relation_filter: Optional[List[str]] = None
    max_depth: int = Field(3, ge=1, le=10)
    max_nodes: int = Field(50, ge=1, le=500)


@router.post("/tenants/{tenant_id}/c3/kg/traverse")
async def kg_traverse(tenant_id: str, body: KGTraverseRequest) -> Dict[str, Any]:
    """C3.2 — walk the entity / kg graph from a starting subject (read-only)."""
    from cogniverse_agents.kg_traversal_agent import (
        KGTraversalAgent,
        KGTraversalDeps,
        KGTraversalInput,
    )

    agent = KGTraversalAgent(deps=KGTraversalDeps(tenant_id=tenant_id))
    _inject_memory(agent, tenant_id, "kg_traversal_agent")
    out = await agent._process_impl(
        KGTraversalInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- C3.3 CrossTenantComparisonAgent ---------------------------------------


class CrossTenantCompareRequest(BaseModel):
    subject_key: str = Field(..., min_length=1)
    tenant_ids: List[str] = Field(..., min_length=2)
    actor_role: str = Field("admin")
    actor_id: str = Field("admin")


@router.post("/tenants/{tenant_id}/c3/cross_tenant/compare")
async def cross_tenant_compare(
    tenant_id: str, body: CrossTenantCompareRequest
) -> Dict[str, Any]:
    """C3.3 — compare knowledge across org tenants for a subject (admin)."""
    from cogniverse_agents.cross_tenant_comparison_agent import (
        CrossTenantComparisonAgent,
        CrossTenantComparisonDeps,
        CrossTenantComparisonInput,
    )

    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
        registry=_build_default_registry(),
    )
    out = await agent._process_impl(
        CrossTenantComparisonInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- C3.7 FederatedQueryAgent ----------------------------------------------


class FederatedQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_ids: List[str] = Field(..., min_length=1)
    actor_role: str = Field("admin")
    actor_id: str = Field("admin")
    top_k: int = Field(10, ge=1, le=200)


@router.post("/tenants/{tenant_id}/c3/federated/query")
async def federated_query(
    tenant_id: str, body: FederatedQueryRequest
) -> Dict[str, Any]:
    """C3.7 — issue a single query against multiple tenants (admin, read-only)."""
    from cogniverse_agents.federated_query_agent import (
        FederatedQueryAgent,
        FederatedQueryDeps,
        FederatedQueryInput,
    )

    agent = FederatedQueryAgent(
        deps=FederatedQueryDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
        registry=_build_default_registry(),
    )
    out = await agent._process_impl(
        FederatedQueryInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- C3.6 TemporalReasoningAgent -------------------------------------------


class TemporalReasonRequest(BaseModel):
    subject_key: str = Field(..., min_length=1)
    windows: List[Dict[str, Any]] = Field(..., min_length=1)


@router.post("/tenants/{tenant_id}/c3/temporal/reason")
async def temporal_reason(
    tenant_id: str, body: TemporalReasonRequest
) -> Dict[str, Any]:
    """C3.6 — compare knowledge of a subject across time windows (read-only)."""
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
