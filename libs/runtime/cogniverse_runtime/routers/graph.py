"""Knowledge graph endpoints — upsert, search, neighbors, path, stats.

Follows the same dependency-injection pattern as the wiki router:
a GraphManager factory is injected via set_graph_manager_factory() at
startup, and per-tenant managers are built from it. All operations are
tenant-scoped.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Bound a single upsert so one request can't materialize millions of nodes/edges
# and exhaust the pod's memory. Callers batch larger graphs across requests.
MAX_UPSERT_ITEMS = 10_000

logger = logging.getLogger(__name__)

router = APIRouter()

_graph_manager_factory = None


def set_graph_manager_factory(factory) -> None:
    """Inject a factory (tenant_id → GraphManager). Called from main.py."""
    global _graph_manager_factory
    _graph_manager_factory = factory
    logger.info("GraphManager factory injected into graph router")


def get_graph_manager(tenant_id: str, deploy: bool = True):
    """Return the tenant's GraphManager.

    ``deploy`` MUST be False on read-only paths — schema deploy triggers a
    Vespa redeploy that can drop another process's just-fed rows mid-read.
    """
    if _graph_manager_factory is None:
        raise HTTPException(
            status_code=503,
            detail="GraphManager not configured",
        )
    try:
        return _graph_manager_factory(tenant_id, deploy=deploy)
    except Exception as exc:
        logger.warning("Graph manager init failed for %s: %s", tenant_id, exc)
        raise HTTPException(status_code=503, detail="Graph backend unavailable")


class MentionDoc(BaseModel):
    source_doc_id: str
    segment_id: str
    ts_start: float
    ts_end: float
    modality: str
    evidence_span: str


class NodeDoc(BaseModel):
    name: str
    # Defaults to ``[]`` so admin / SDK callers that POST a bare
    # ``{"name": "Foo"}`` still validate. The internal ``Node``
    # dataclass keeps ``mentions`` required (it's the KG provenance
    # invariant); the REST DTO has a default so the route can do its
    # tenant + auth checks BEFORE rejecting the body shape. Routes
    # that need anchored mentions still verify mentions != [] before
    # writing to the KG.
    mentions: List[MentionDoc] = []
    description: str = ""
    kind: str = "concept"
    label: str = "Concept"  # GLiNER tag — gates same_as linking in CrossModalLinker


class EdgeDoc(BaseModel):
    source: str
    target: str
    relation: str
    evidence_span: str
    segment_id: str
    ts_start: float
    ts_end: float
    modality: str
    provenance: str = "EXTRACTED"
    confidence: float = 1.0


class UpsertRequest(BaseModel):
    tenant_id: str
    source_doc_id: str
    nodes: List[NodeDoc] = Field(default_factory=list, max_length=MAX_UPSERT_ITEMS)
    edges: List[EdgeDoc] = Field(default_factory=list, max_length=MAX_UPSERT_ITEMS)


class UpsertResponse(BaseModel):
    status: str
    nodes_upserted: int
    edges_upserted: int
    failed_ids: list[str] = []


class NodeSearchResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    count: int


class NeighborsResponse(BaseModel):
    node_id: str
    name: str
    out_edges: List[Dict[str, Any]]
    in_edges: List[Dict[str, Any]]


class PathResponse(BaseModel):
    source: str
    target: str
    path: Optional[List[str]]
    length: int


class StatsResponse(BaseModel):
    tenant_id: str
    node_count: int
    edge_count: int
    top_nodes: List[Dict[str, Any]]


@router.post("/upsert", response_model=UpsertResponse)
async def upsert(request: UpsertRequest) -> UpsertResponse:
    """Upsert a batch of nodes and edges for a tenant."""
    from cogniverse_agents.graph.graph_schema import (
        Edge,
        ExtractionResult,
        Mention,
        Node,
    )
    from cogniverse_core.common.tenant_utils import (
        assert_tenant_exists,
        canonical_tenant_id,
    )

    # Canonicalize before EVERY downstream use so the document's
    # tenant_id field matches the form queries (stats/search/neighbors)
    # filter by — without this, simple-form upsert and canonical-form
    # stats live in the same Vespa schema but never see each other.
    tenant_id = canonical_tenant_id(request.tenant_id)
    await assert_tenant_exists(tenant_id)
    mgr = get_graph_manager(tenant_id)

    nodes = [
        Node(
            tenant_id=tenant_id,
            name=n.name,
            description=n.description,
            kind=n.kind,
            label=n.label,
            mentions=[
                Mention(
                    source_doc_id=m.source_doc_id,
                    segment_id=m.segment_id,
                    ts_start=m.ts_start,
                    ts_end=m.ts_end,
                    modality=m.modality,
                    evidence_span=m.evidence_span,
                )
                for m in n.mentions
            ],
        )
        for n in request.nodes
    ]
    edges = [
        Edge(
            tenant_id=tenant_id,
            source=e.source,
            target=e.target,
            relation=e.relation,
            evidence_span=e.evidence_span,
            segment_id=e.segment_id,
            ts_start=e.ts_start,
            ts_end=e.ts_end,
            modality=e.modality,
            provenance=e.provenance,
            confidence=e.confidence,
            source_doc_id=request.source_doc_id,
        )
        for e in request.edges
    ]

    result = ExtractionResult(
        source_doc_id=request.source_doc_id,
        nodes=nodes,
        edges=edges,
    )
    counts = await asyncio.to_thread(mgr.upsert, result)
    failed_ids = counts.get("failed_ids", [])
    requested = len(nodes) + len(edges)
    upserted = counts["nodes_upserted"] + counts["edges_upserted"]

    if requested and upserted == 0:
        # Every feed failed — the backend rejected the whole batch (schema
        # missing/unconverged, or Vespa down). Returning 200 "upserted" with
        # zero counts hid a total data loss; surface it so the caller retries.
        raise HTTPException(
            status_code=502,
            detail=(
                f"Graph upsert persisted 0/{requested} documents; "
                f"failed ids: {failed_ids[:20]}"
            ),
        )

    status = "upserted" if not failed_ids else "partially_upserted"
    return UpsertResponse(status=status, **counts)


@router.get("/search", response_model=NodeSearchResponse)
async def search_nodes(
    tenant_id: str,
    q: str,
    top_k: int = Query(10, ge=1, le=100),
) -> NodeSearchResponse:
    """Semantic search over graph nodes."""
    from cogniverse_core.common.tenant_utils import (
        assert_tenant_exists,
        canonical_tenant_id,
    )

    tenant_id = canonical_tenant_id(tenant_id)
    await assert_tenant_exists(tenant_id)
    mgr = get_graph_manager(tenant_id)
    nodes = await asyncio.to_thread(mgr.search_nodes, q, top_k=top_k)
    return NodeSearchResponse(nodes=nodes, count=len(nodes))


@router.get("/neighbors", response_model=NeighborsResponse)
async def get_neighbors(
    tenant_id: str,
    node: str,
    depth: int = Query(1, ge=1, le=3),
) -> NeighborsResponse:
    """Return direct neighbors (out and in) of a node by name."""
    from cogniverse_core.common.tenant_utils import (
        assert_tenant_exists,
        canonical_tenant_id,
    )

    tenant_id = canonical_tenant_id(tenant_id)
    await assert_tenant_exists(tenant_id)
    mgr = get_graph_manager(tenant_id)
    result = await asyncio.to_thread(mgr.get_neighbors, node, depth=depth)
    return NeighborsResponse(**result)


@router.get("/path", response_model=PathResponse)
async def get_path(
    tenant_id: str,
    source: str,
    target: str,
    max_depth: int = Query(4, ge=1, le=6),
) -> PathResponse:
    """Shortest path between two nodes by name."""
    from cogniverse_core.common.tenant_utils import (
        assert_tenant_exists,
        canonical_tenant_id,
    )

    tenant_id = canonical_tenant_id(tenant_id)
    await assert_tenant_exists(tenant_id)
    mgr = get_graph_manager(tenant_id)
    path = await asyncio.to_thread(mgr.get_path, source, target, max_depth=max_depth)
    return PathResponse(
        source=source,
        target=target,
        path=path,
        length=len(path) - 1 if path else -1,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(tenant_id: str) -> StatsResponse:
    """Graph statistics: node and edge counts, top-degree nodes."""
    from cogniverse_core.common.tenant_utils import (
        assert_tenant_exists,
        canonical_tenant_id,
    )

    tenant_id = canonical_tenant_id(tenant_id)
    await assert_tenant_exists(tenant_id)
    mgr = get_graph_manager(tenant_id)
    stats = await asyncio.to_thread(mgr.get_stats)
    return StatsResponse(tenant_id=tenant_id, **stats)
