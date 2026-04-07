"""GraphManager — persists nodes/edges to Vespa and supports queries.

Follows the same pattern as WikiManager: takes a VespaSearchBackend,
feeds documents via the Document v1 HTTP API, queries via YQL.

Upserts are idempotent per node_id / edge_id (deterministic from name
and relation), so re-indexing the same file is safe.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from cogniverse_agents.graph.code_extractor import CodeExtractor
from cogniverse_agents.graph.code_extractor import (
    supported_extensions as code_extensions,
)
from cogniverse_agents.graph.doc_extractor import DocExtractor
from cogniverse_agents.graph.doc_extractor import (
    supported_extensions as doc_extensions,
)
from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Node,
    normalize_name,
)

logger = logging.getLogger(__name__)


class GraphManager:
    """Manages the knowledge graph for a single tenant, backed by Vespa."""

    def __init__(self, backend: Any, tenant_id: str, schema_name: str) -> None:
        """
        Args:
            backend: VespaSearchBackend instance (provides _url, _port, search).
            tenant_id: Tenant identifier.
            schema_name: Tenant-specific schema name
                         (e.g. "knowledge_graph_acme_production").
        """
        self._backend = backend
        self._tenant_id = tenant_id
        self._schema_name = schema_name
        self._code_extractor = CodeExtractor()
        self._doc_extractor = DocExtractor()

    def extract_file(self, file_path: Path, source_doc_id: str) -> Optional[ExtractionResult]:
        """Run the appropriate extractor for a file's extension."""
        ext = file_path.suffix.lower()
        if ext in code_extensions():
            return self._code_extractor.extract(file_path, self._tenant_id, source_doc_id)
        if ext in doc_extensions():
            return self._doc_extractor.extract(file_path, self._tenant_id, source_doc_id)
        return None

    def upsert(self, result: ExtractionResult) -> Dict[str, int]:
        """Upsert extracted nodes and edges into Vespa.

        Returns a dict with counts: nodes_upserted, edges_upserted.
        """
        nodes_upserted = 0
        edges_upserted = 0

        merged_nodes = self._merge_duplicate_nodes(result.nodes)
        for node in merged_nodes:
            embedding = self._generate_embedding(f"{node.name}\n{node.description}")
            if self._feed_node(node, embedding):
                nodes_upserted += 1

        for edge in result.edges:
            if self._feed_edge(edge):
                edges_upserted += 1

        return {"nodes_upserted": nodes_upserted, "edges_upserted": edges_upserted}

    def search_nodes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic + BM25 search over nodes."""
        embedding = self._generate_embedding(query)
        yql = (
            f'select * from sources {self._schema_name} '
            f'where tenant_id contains "{self._tenant_id}" '
            f'and doc_type contains "node" '
            f'and userQuery() limit {top_k}'
        )
        try:
            resp = self._backend.search(
                query=query,
                yql=yql,
                hits=top_k,
                ranking="hybrid",
                query_embedding=embedding,
            )
            return self._extract_hits(resp)
        except Exception as exc:
            logger.warning("Node search failed, trying YQL-only fallback: %s", exc)
            return self._visit(doc_type="node", name_contains=query, top_k=top_k)

    def get_neighbors(self, node_name: str, depth: int = 1) -> Dict[str, Any]:
        """Get direct neighbors of a node (edges out and in)."""
        node_id = normalize_name(node_name)
        out_edges = self._visit_edges(source_node_id=node_id)
        in_edges = self._visit_edges(target_node_id=node_id)
        return {
            "node_id": node_id,
            "name": node_name,
            "out_edges": out_edges,
            "in_edges": in_edges,
            "depth": depth,
        }

    def get_path(self, source: str, target: str, max_depth: int = 4) -> Optional[List[str]]:
        """BFS shortest path between two node names."""
        source_id = normalize_name(source)
        target_id = normalize_name(target)
        if source_id == target_id:
            return [source_id]

        visited: set = {source_id}
        frontier: List[Tuple[str, List[str]]] = [(source_id, [source_id])]

        for _ in range(max_depth):
            next_frontier: List[Tuple[str, List[str]]] = []
            for current, path in frontier:
                out_edges = self._visit_edges(source_node_id=current)
                for edge in out_edges:
                    neighbor = edge.get("target_node_id", "")
                    if not neighbor or neighbor in visited:
                        continue
                    new_path = path + [neighbor]
                    if neighbor == target_id:
                        return new_path
                    visited.add(neighbor)
                    next_frontier.append((neighbor, new_path))
            frontier = next_frontier
            if not frontier:
                break

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Graph statistics: node count, edge count, top-degree nodes."""
        all_nodes = self._visit(doc_type="node", top_k=500)
        all_edges = self._visit(doc_type="edge", top_k=2000)

        degree: Dict[str, int] = defaultdict(int)
        for edge in all_edges:
            fields = edge.get("fields", edge)
            src = fields.get("source_node_id", "")
            tgt = fields.get("target_node_id", "")
            if src:
                degree[src] += 1
            if tgt:
                degree[tgt] += 1

        top_nodes = sorted(degree.items(), key=lambda kv: -kv[1])[:10]
        return {
            "node_count": len(all_nodes),
            "edge_count": len(all_edges),
            "top_nodes": [
                {"node_id": node_id, "degree": deg} for node_id, deg in top_nodes
            ],
        }

    # ----- Internal helpers ------------------------------------------------

    def _merge_duplicate_nodes(self, nodes: List[Node]) -> List[Node]:
        """Merge nodes with the same normalized name — union their mentions."""
        merged: Dict[str, Node] = {}
        for node in nodes:
            key = normalize_name(node.name)
            if key in merged:
                existing = merged[key]
                for m in node.mentions:
                    if m not in existing.mentions:
                        existing.mentions.append(m)
            else:
                merged[key] = node
        return list(merged.values())

    def _feed_node(self, node: Node, embedding: List[float]) -> bool:
        url = f"{self._backend._url}:{self._backend._port}"
        feed_url = (
            f"{url}/document/v1/graph_content/{self._schema_name}/docid/{node.doc_id}"
        )
        payload = node.to_vespa_document()
        payload["fields"]["embedding"] = embedding
        try:
            resp = requests.post(feed_url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(
                    "Feed for node %s returned %s: %s",
                    node.doc_id,
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception:
            logger.exception("Failed to feed graph node %s", node.doc_id)
            return False

    def _feed_edge(self, edge: Edge) -> bool:
        url = f"{self._backend._url}:{self._backend._port}"
        feed_url = (
            f"{url}/document/v1/graph_content/{self._schema_name}/docid/{edge.doc_id}"
        )
        payload = edge.to_vespa_document()
        payload["fields"]["embedding"] = [0.0] * 768
        try:
            resp = requests.post(feed_url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(
                    "Feed for edge %s returned %s: %s",
                    edge.doc_id,
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception:
            logger.exception("Failed to feed graph edge %s", edge.doc_id)
            return False

    def _visit(
        self,
        doc_type: str,
        top_k: int = 100,
        name_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Visit all documents of a given doc_type via Document v1 visit API."""
        url = f"{self._backend._url}:{self._backend._port}"
        visit_url = f"{url}/document/v1/graph_content/{self._schema_name}/docid"
        params = {"wantedDocumentCount": str(top_k), "selection": f'{self._schema_name}.doc_type=="{doc_type}" and {self._schema_name}.tenant_id=="{self._tenant_id}"'}
        try:
            resp = requests.get(visit_url, params=params, timeout=15)
            if not resp.ok:
                return []
            data = resp.json()
            hits = data.get("documents", [])
            return [h.get("fields", h) for h in hits]
        except Exception:
            logger.exception("Visit failed for doc_type=%s", doc_type)
            return []

    def _visit_edges(
        self,
        source_node_id: Optional[str] = None,
        target_node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Visit edges filtered by source or target node id."""
        url = f"{self._backend._url}:{self._backend._port}"
        visit_url = f"{url}/document/v1/graph_content/{self._schema_name}/docid"
        selection_parts = [
            f'{self._schema_name}.doc_type=="edge"',
            f'{self._schema_name}.tenant_id=="{self._tenant_id}"',
        ]
        if source_node_id:
            selection_parts.append(
                f'{self._schema_name}.source_node_id=="{source_node_id}"'
            )
        if target_node_id:
            selection_parts.append(
                f'{self._schema_name}.target_node_id=="{target_node_id}"'
            )
        params = {
            "wantedDocumentCount": "100",
            "selection": " and ".join(selection_parts),
        }
        try:
            resp = requests.get(visit_url, params=params, timeout=15)
            if not resp.ok:
                return []
            data = resp.json()
            return [d.get("fields", d) for d in data.get("documents", [])]
        except Exception:
            return []

    def _extract_hits(self, response) -> List[Dict[str, Any]]:
        """Extract hit fields from a search backend response."""
        if hasattr(response, "hits"):
            return [h.fields for h in response.hits]
        if isinstance(response, dict):
            return response.get("hits", [])
        return []

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a text embedding via Ollama nomic-embed-text."""
        try:
            import ollama

            result = ollama.embed(model="nomic-embed-text", input=text)
            embeddings = result.get("embeddings") or result.get("embedding") or []
            if embeddings and isinstance(embeddings[0], list):
                return embeddings[0]
            return list(embeddings)
        except Exception:
            logger.warning("Embedding generation failed; using zero vector")
            return [0.0] * 768
