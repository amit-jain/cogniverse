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

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cogniverse_foundation.config.manager import ConfigManager

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
        from cogniverse_runtime.memory_init import lazy_init_memory
        from cogniverse_runtime.routers.tenant import _require_config_manager

        # Read-only routes must not auto-create schemas: a deploy here
        # triggers a Vespa redeploy that can drop another process's
        # freshly-fed rows mid-read. Connect to the existing schema only.
        lazy_init_memory(mm, _tid, _require_config_manager(), auto_create_schema=False)
    if not mm.memory:
        raise HTTPException(
            status_code=503,
            detail=f"Memory backend not initialised for tenant {_tid}",
        )
    return mm


def _build_default_registry():
    from cogniverse_core.memory.schema import build_default_registry

    return build_default_registry()


def _get_config_manager() -> ConfigManager:
    """FastAPI dep, overridden in main.py — same pattern as other routers."""
    raise RuntimeError(
        "ConfigManager dependency not configured. "
        "Override this dependency in main.py using app.dependency_overrides."
    )


def _runtime_primary_llm_config(config_manager: ConfigManager):
    """Return the runtime's primary LLMEndpointConfig (api_base + model).

    Agents that take an optional ``llm_config`` (synthesis/RLM paths)
    must point at the in-cluster vLLM, not the upstream OpenAI default
    that ``LLMEndpointConfig.model_factory`` would synthesise. Without
    this, the RLM path 401s against api.openai.com.
    """
    from cogniverse_foundation.config.utils import get_config

    config = get_config(tenant_id="__system__", config_manager=config_manager)
    llm_cfg = config.get("llm_config", {}).get("primary")
    if not llm_cfg:
        return None
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    return LLMEndpointConfig(**llm_cfg)


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


def _bind_graph(agent, tenant_id: str) -> None:
    """Bind the tenant's shared Vespa knowledge-graph manager onto a
    graph-aware agent so its ``_process_impl`` can complement the mem0 answer
    with the provenance-rich KG (the ``kg_*`` output fields). No-op when the
    agent isn't graph-bindable or the graph backend isn't configured — the
    agent then falls back to its mem0-only path. Mirrors the generic dispatch
    binding in ``agent_dispatcher._bind_graph_manager``.
    """
    setter = getattr(agent, "set_graph_manager", None)
    if not callable(setter):
        return
    try:
        from cogniverse_runtime.routers.graph import get_graph_manager

        # Read-only routes: never deploy the KG schema here. A deploy
        # triggers a Vespa redeploy that drops rows the caller just fed.
        setter(get_graph_manager(tenant_id, deploy=False))
    except Exception as exc:  # graph backend not configured / unavailable
        logger.debug("Graph manager bind skipped for tenant %s: %s", tenant_id, exc)


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
    claim_id: Optional[str] = Field(
        None,
        description=(
            "Optional KG Edge id to ground via the bound graph, surfaced in "
            "``kg_primary_sources`` alongside the mem0 provenance walk."
        ),
    )
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
    await asyncio.to_thread(_inject_memory, agent, tenant_id, "citation_tracing_agent")
    await asyncio.to_thread(_bind_graph, agent, tenant_id)
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
    tenant_id: str,
    body: KnowledgeSummarizeRequest,
    config_manager: ConfigManager = Depends(_get_config_manager),
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
        # Resolved via Depends above — calling _get_config_manager()
        # directly bypasses app.dependency_overrides and always raises.
        config_manager=config_manager,
    )
    await asyncio.to_thread(_bind_graph, agent, tenant_id)
    out = await agent._process_impl(
        KnowledgeSummarizationInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()


# --- ContradictionReconciliationAgent ---------------------------------


class ContradictionReconcileRequest(BaseModel):
    target_kind: str = Field(..., min_length=1)
    conflict_member_ids: List[str] = Field(..., min_length=1)
    policy_override: Optional[str] = None
    subject_key: Optional[str] = Field(
        None,
        description=(
            "With ``predicate``, surface cross-document KG conflicts about "
            "``(subject_key, predicate)`` via the bound graph in "
            "``kg_conflict_entries``, alongside the mem0 member reconciliation."
        ),
    )
    predicate: Optional[str] = Field(None)


@router.post("/tenants/{tenant_id}/knowledge/contradictions/reconcile")
async def contradiction_reconcile(
    tenant_id: str, body: ContradictionReconcileRequest
) -> Dict[str, Any]:
    """apply schema policy to a conflict set (read-only — returns the resolved view)."""
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
    mm = await asyncio.to_thread(_build_factory, tenant_id)
    agent.memory_manager = mm
    agent._memory_initialized = True
    agent._memory_tenant_id = tenant_id
    agent._memory_agent_name = "contradiction_reconciliation_agent"
    await asyncio.to_thread(_bind_graph, agent, tenant_id)
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
    rlm: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional RLM options (RLMOptions schema). When provided with "
            "``enabled=true`` or ``auto_detect=true`` past the threshold, "
            "synthesis runs through RLMInference instead of the dspy.Predict "
            "fast path."
        ),
    )


@router.post("/tenants/{tenant_id}/knowledge/synthesis/multi_doc")
async def multi_doc_synthesize(
    tenant_id: str,
    body: MultiDocSynthesizeRequest,
    config_manager: ConfigManager = Depends(_get_config_manager),
) -> Dict[str, Any]:
    """synthesise an answer across N documents with citations."""
    from cogniverse_agents.multi_document_synthesis_agent import (
        MultiDocSynthesisDeps,
        MultiDocSynthesisInput,
        MultiDocumentSynthesisAgent,
    )

    agent = MultiDocumentSynthesisAgent(
        deps=MultiDocSynthesisDeps(tenant_id=tenant_id),
        llm_config=_runtime_primary_llm_config(config_manager),
    )
    await asyncio.to_thread(
        _inject_memory, agent, tenant_id, "multi_document_synthesis_agent"
    )
    await asyncio.to_thread(_bind_graph, agent, tenant_id)
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
    await asyncio.to_thread(_inject_memory, agent, tenant_id, "kg_traversal_agent")
    await asyncio.to_thread(_bind_graph, agent, tenant_id)
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
    # — the agent rejects everything else via ACLRejected. Default to
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
    from cogniverse_core.memory.federation import ACLRejected

    agent = CrossTenantComparisonAgent(
        deps=CrossTenantComparisonDeps(tenant_id=tenant_id),
        memory_manager_factory=_build_factory,
        registry=_build_default_registry(),
    )
    try:
        out = await agent._process_impl(
            CrossTenantComparisonInput(tenant_id=tenant_id, **body.model_dump())
        )
    except ACLRejected as exc:
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
    from cogniverse_core.memory.federation import ACLRejected

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
    except ACLRejected as exc:
        raise HTTPException(403, str(exc)) from exc
    return out.model_dump()


# --- TemporalReasoningAgent -------------------------------------------


class TemporalReasonRequest(BaseModel):
    subject_key: str = Field(..., min_length=1)
    # TemporalReasoningInput requires >= 2 windows (comparison is the
    # whole point of the agent). Mirror that floor here so the route
    # rejects single-window calls at validation rather than at the
    # agent which would 500 the request.
    windows: List[Dict[str, Any]] = Field(..., min_length=2)
    agent_name_filter: Optional[str] = Field(None)


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
    await asyncio.to_thread(_bind_graph, agent, tenant_id)
    out = await agent._process_impl(
        TemporalReasoningInput(tenant_id=tenant_id, **body.model_dump())
    )
    return out.model_dump()
