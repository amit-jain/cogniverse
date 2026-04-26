"""GraphManager — persists nodes/edges to Vespa and supports queries.

Follows the same pattern as WikiManager: takes a VespaSearchBackend,
feeds documents via the Document v1 HTTP API, queries via YQL.

Embeddings are multi-vector ColBERT (LateOn 128-dim per token) served
by the ``colbert_pylate`` sidecar pod. Nodes ship two tensor fields to
Vespa — ``embedding`` (bfloat16 hex per token) and ``embedding_binary``
(1-bit packed) — so retrieval can use the standard MaxSim two-phase
rank: hamming on binary first, full bfloat16 MaxSim rerank.

Upserts are idempotent per node_id / edge_id (deterministic from name
and relation), so re-indexing the same file is safe.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor

logger = logging.getLogger(__name__)


class GraphManager:
    """Manages the knowledge graph for a single tenant, backed by Vespa."""

    def __init__(
        self,
        backend: Any,
        tenant_id: str,
        schema_name: str,
        colbert_endpoint_url: str,
        colbert_model: str = "lightonai/LateOn",
    ) -> None:
        """
        Args:
            backend: VespaSearchBackend instance (provides _url, _port).
            tenant_id: Tenant identifier.
            schema_name: Tenant-specific schema name
                         (e.g. "knowledge_graph_acme_production").
            colbert_endpoint_url: URL of the colbert_pylate sidecar that
                serves /pooling for ColBERT multi-vector encoding.
            colbert_model: HF model id the sidecar serves; passed
                through in /pooling requests for diagnostic logging.
        """
        if not colbert_endpoint_url:
            raise ValueError(
                "GraphManager requires a colbert_endpoint_url — KG node "
                "embeddings route through the colbert_pylate sidecar pod. "
                "Configure inference.colbert_pylate in the chart, or pass "
                "the URL explicitly."
            )

        self._backend = backend
        self._tenant_id = tenant_id
        self._schema_name = schema_name
        self._code_extractor = CodeExtractor()
        self._doc_extractor = DocExtractor()

        loader = RemoteColBERTLoader(
            model_name=colbert_model,
            config={"remote_inference_url": colbert_endpoint_url},
            logger=logger,
        )
        self._encoder, _ = loader.load_model()
        self._embedding_processor = VespaEmbeddingProcessor(
            logger=logger, model_name=colbert_model, schema_name=schema_name
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def extract_file(
        self, file_path: Path, source_doc_id: str
    ) -> Optional[ExtractionResult]:
        """Run the appropriate extractor for a file's extension."""
        ext = file_path.suffix.lower()
        if ext in code_extensions():
            return self._code_extractor.extract(
                file_path, self._tenant_id, source_doc_id
            )
        if ext in doc_extensions():
            return self._doc_extractor.extract(
                file_path, self._tenant_id, source_doc_id
            )
        return None

    def upsert(self, result: ExtractionResult) -> Dict[str, int]:
        """Upsert extracted nodes and edges into Vespa.

        Returns a dict with counts: nodes_upserted, edges_upserted.
        """
        nodes_upserted = 0
        edges_upserted = 0

        merged_nodes = self._merge_duplicate_nodes(result.nodes)
        if merged_nodes:
            texts = [f"{node.name}\n{node.description}" for node in merged_nodes]
            tensor_fields_per_node = self._encode_documents(texts)
            for node, tensor_fields in zip(merged_nodes, tensor_fields_per_node):
                if self._feed_node(node, tensor_fields):
                    nodes_upserted += 1

        for edge in result.edges:
            if self._feed_edge(edge):
                edges_upserted += 1

        return {"nodes_upserted": nodes_upserted, "edges_upserted": edges_upserted}

    def search_nodes(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """MaxSim multi-vector search over nodes with bm25 rerank."""
        try:
            qt_blocks, qtb_blocks = self._encode_query_blocks(query)
        except Exception as exc:
            logger.warning("Query encode failed, falling back to YQL visit: %s", exc)
            return self._visit(doc_type="node", name_contains=query, top_k=top_k)

        yql = (
            f"select * from sources {self._schema_name} "
            f'where tenant_id contains "{self._tenant_id}" '
            f'and doc_type contains "node" '
            f"and userQuery() limit {top_k}"
        )
        body = {
            "yql": yql,
            "query": query,
            "hits": top_k,
            "ranking.profile": "hybrid_binary_bm25",
            "input.query(qtb)": {"blocks": qtb_blocks},
            "input.query(qt)": {"blocks": qt_blocks},
        }
        url = f"{self._backend._url}:{self._backend._port}/search/"
        try:
            resp = requests.post(url, json=body, timeout=10)
            if not resp.ok:
                raise RuntimeError(f"search returned {resp.status_code}: {resp.text}")
            data = resp.json()
            return [
                h.get("fields", h) for h in data.get("root", {}).get("children", [])
            ]
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

    def get_path(
        self, source: str, target: str, max_depth: int = 4
    ) -> Optional[List[str]]:
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

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

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

    def _encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encode document texts via the ColBERT sidecar; return per-text
        ``{embedding, embedding_binary}`` tensor field dicts in Vespa wire
        format (bfloat16 hex per token, 1-bit packed binary per token)."""
        try:
            per_text = self._encoder.encode(texts, is_query=False)
        except Exception as exc:
            logger.warning(
                "ColBERT encode failed for %d texts; nodes will ship without "
                "embeddings (semantic recall will degrade until pod recovers): %s",
                len(texts),
                exc,
            )
            return [{} for _ in texts]

        out: List[Dict[str, Any]] = []
        for tokens in per_text:
            arr = np.asarray(tokens, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                out.append({})
                continue
            out.append(self._embedding_processor.process_embeddings(arr))
        return out

    def _encode_query_blocks(self, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Encode a query into the two block-format tensors Vespa expects:
        ``query(qt)`` (float per-token, 128-dim) and ``query(qtb)`` (1-bit
        packed binary per token, 16 bytes). Each block keyed by token index."""
        per_text = self._encoder.encode([query], is_query=True)
        if not per_text:
            raise RuntimeError("encoder returned no embeddings for query")
        arr = np.asarray(per_text[0], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise RuntimeError(
                f"encoder returned malformed query embedding shape {arr.shape}"
            )

        qt_blocks: Dict[str, Any] = {}
        for token_idx in range(arr.shape[0]):
            qt_blocks[str(token_idx)] = arr[token_idx].tolist()

        # 1-bit packed binary: pack the sign bits of each 128-d vector into
        # 16 bytes (int8). Matches Vespa's hamming distance over int8 tensors.
        signs = (arr > 0).astype(np.uint8)
        packed = np.packbits(signs, axis=-1).astype(np.int8)
        qtb_blocks: Dict[str, Any] = {}
        for token_idx in range(packed.shape[0]):
            # Vespa expects signed int8; numpy int8 is already correct.
            qtb_blocks[str(token_idx)] = packed[token_idx].tolist()

        return qt_blocks, qtb_blocks

    def _feed_with_retry(
        self, feed_url: str, payload: dict, doc_id: str, kind: str
    ) -> bool:
        """POST a graph node/edge to Vespa, retrying on schema-convergence races.

        Vespa's content-distributor convergence trails the config server by a
        few seconds after a fresh schema deploy. A feed during that window
        surfaces in two shapes depending on which layer rejects it:

        * 400 ``Document type ... does not exist`` — config layer hasn't
          accepted the schema yet.
        * 500 ``APP_FATAL_ERROR ... No handler for document type`` — config
          accepted it, but the content distributor at port 19115 hasn't
          loaded the new bucket handler yet.

        Both are transient post-deploy. Retry up to ~30s so the first
        request after a fresh tenant schema deploy doesn't fail the caller.
        """
        import time as _time

        backoff = 2.0
        for attempt in range(8):
            try:
                resp = requests.post(feed_url, json=payload, timeout=10)
            except Exception:
                logger.exception("Failed to feed graph %s %s", kind, doc_id)
                return False
            if resp.ok:
                return True
            body = resp.text
            transient = (resp.status_code == 400 and "does not exist" in body) or (
                resp.status_code == 500 and "No handler for document type" in body
            )
            if not transient or attempt == 7:
                logger.warning(
                    "Feed for %s %s returned %s: %s",
                    kind,
                    doc_id,
                    resp.status_code,
                    body[:2000],
                )
                return False
            _time.sleep(backoff)
            backoff = min(backoff * 1.5, 8.0)
        return False

    def _feed_node(self, node: Node, tensor_fields: Dict[str, Any]) -> bool:
        url = f"{self._backend._url}:{self._backend._port}"
        feed_url = (
            f"{url}/document/v1/graph_content/{self._schema_name}/docid/{node.doc_id}"
        )
        payload = node.to_vespa_document()
        # tensor_fields is empty when encoding failed — ship the node
        # without embeddings rather than blocking the upsert. Mapped
        # tensor attributes tolerate absent values.
        for k, v in tensor_fields.items():
            payload["fields"][k] = v
        return self._feed_with_retry(feed_url, payload, node.doc_id, "node")

    def _feed_edge(self, edge: Edge) -> bool:
        """Feed an edge document. Edges don't carry embeddings — semantic
        retrieval is over nodes only — so the mapped embedding fields are
        omitted entirely (Vespa attribute tensors handle absence)."""
        url = f"{self._backend._url}:{self._backend._port}"
        feed_url = (
            f"{url}/document/v1/graph_content/{self._schema_name}/docid/{edge.doc_id}"
        )
        payload = edge.to_vespa_document()
        return self._feed_with_retry(feed_url, payload, edge.doc_id, "edge")

    def _visit(
        self,
        doc_type: str,
        top_k: int = 100,
        name_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Visit all documents of a given doc_type via Document v1 visit API."""
        url = f"{self._backend._url}:{self._backend._port}"
        visit_url = f"{url}/document/v1/graph_content/{self._schema_name}/docid"
        params = {
            "wantedDocumentCount": str(top_k),
            "selection": (
                f'{self._schema_name}.doc_type=="{doc_type}" and '
                f'{self._schema_name}.tenant_id=="{self._tenant_id}"'
            ),
        }
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
