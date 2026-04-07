"""Knowledge graph endpoints — upsert, search, neighbors, path, stats.

Follows the same dependency-injection pattern as the wiki router:
GraphManager is injected via set_graph_manager() at startup. All
operations are tenant-scoped.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

_graph_manager_factory = None


def set_graph_manager_factory(factory) -> None:
    """Inject a factory (tenant_id → GraphManager). Called from main.py."""
    global _graph_manager_factory
    _graph_manager_factory = factory
    logger.info("GraphManager factory injected into graph router")


def get_graph_manager(tenant_id: str):
    if _graph_manager_factory is None:
        raise HTTPException(
            status_code=503,
            detail="GraphManager not configured",
        )
    try:
        return _graph_manager_factory(tenant_id)
    except Exception as exc:
        logger.warning("Graph manager init failed for %s: %s", tenant_id, exc)
        raise HTTPException(status_code=503, detail="Graph backend unavailable")


class NodeDoc(BaseModel):
    name: str
    description: str = ""
    kind: str = "concept"
    mentions: List[str] = []


class EdgeDoc(BaseModel):
    source: str
    target: str
    relation: str
    provenance: str = "INFERRED"
    confidence: float = 1.0


class UpsertRequest(BaseModel):
    tenant_id: str
    source_doc_id: str
    nodes: List[NodeDoc] = []
    edges: List[EdgeDoc] = []


class UpsertResponse(BaseModel):
    status: str
    nodes_upserted: int
    edges_upserted: int


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
    from cogniverse_agents.graph.graph_schema import Edge, ExtractionResult, Node

    mgr = get_graph_manager(request.tenant_id)

    nodes = [
        Node(
            tenant_id=request.tenant_id,
            name=n.name,
            description=n.description,
            kind=n.kind,
            mentions=n.mentions or [request.source_doc_id],
        )
        for n in request.nodes
    ]
    edges = [
        Edge(
            tenant_id=request.tenant_id,
            source=e.source,
            target=e.target,
            relation=e.relation,
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
    counts = mgr.upsert(result)
    return UpsertResponse(status="upserted", **counts)


@router.get("/search", response_model=NodeSearchResponse)
async def search_nodes(
    tenant_id: str,
    q: str,
    top_k: int = Query(10, ge=1, le=100),
) -> NodeSearchResponse:
    """Semantic search over graph nodes."""
    mgr = get_graph_manager(tenant_id)
    nodes = mgr.search_nodes(q, top_k=top_k)
    return NodeSearchResponse(nodes=nodes, count=len(nodes))


@router.get("/neighbors", response_model=NeighborsResponse)
async def get_neighbors(
    tenant_id: str,
    node: str,
    depth: int = Query(1, ge=1, le=3),
) -> NeighborsResponse:
    """Return direct neighbors (out and in) of a node by name."""
    mgr = get_graph_manager(tenant_id)
    result = mgr.get_neighbors(node, depth=depth)
    return NeighborsResponse(**result)


@router.get("/path", response_model=PathResponse)
async def get_path(
    tenant_id: str,
    source: str,
    target: str,
    max_depth: int = Query(4, ge=1, le=6),
) -> PathResponse:
    """Shortest path between two nodes by name."""
    mgr = get_graph_manager(tenant_id)
    path = mgr.get_path(source, target, max_depth=max_depth)
    return PathResponse(
        source=source,
        target=target,
        path=path,
        length=len(path) - 1 if path else -1,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(tenant_id: str) -> StatsResponse:
    """Graph statistics: node and edge counts, top-degree nodes."""
    mgr = get_graph_manager(tenant_id)
    stats = mgr.get_stats()
    return StatsResponse(tenant_id=tenant_id, **stats)
