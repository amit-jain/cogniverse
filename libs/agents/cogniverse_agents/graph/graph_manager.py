"""GraphManager — persists nodes/edges to Vespa and supports queries.

Follows the same pattern as WikiManager: takes a VespaBackend,
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
from concurrent.futures import ThreadPoolExecutor
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
    _safe_tenant,
    normalize_name,
)
from cogniverse_agents.search.vespa_query import vespa_search_children
from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
from cogniverse_vespa._yql import yql_quote
from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor

logger = logging.getLogger(__name__)

# Vespa document namespace the knowledge-graph schema feeds under.
_GRAPH_NAMESPACE = "graph_content"


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
            backend: VespaBackend instance (provides the document API +
                _url/_port); node/edge upsert uses put/get_document_fields.
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
        # One keep-alive session for all graph HTTP — the module-level
        # requests helpers paid TCP setup per node/edge feed and per query.
        self._http = requests.Session()
        self._code_extractor = CodeExtractor()
        self._doc_extractor = DocExtractor()

        loader = RemoteColBERTLoader(
            model_name=colbert_model,
            config={"remote_inference_url": colbert_endpoint_url},
            logger=logger,
        )
        self._encoder, _ = loader.load_model()
        self._embedding_processor = VespaEmbeddingProcessor(
            logger=logger,
            model_name=colbert_model,
            schema_name=schema_name,
            # knowledge_graph embeddings are ColBERT multi-vector
            # (tensor(token{}, v[128])).
            single_vector=False,
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

    def search_nodes(
        self, query: str, top_k: int = 10, trace: str = ""
    ) -> List[Dict[str, Any]]:
        """MaxSim multi-vector search over nodes with bm25 rerank.

        When ``trace`` is non-empty, the encoder receives ``query + " " + trace``
        as joint input — the agentic CoT-rationale embedding pattern from
        Chen+Ma (AgentIR, arXiv 2410.09713) that exposes signal humans never
        give the retriever. ``trace=""`` falls back to the single-query path.
        """
        effective_query = f"{query} {trace}".strip() if trace else query
        try:
            qt_blocks, qtb_blocks = self._encode_query_blocks(effective_query)
        except Exception as exc:
            logger.warning("Query encode failed, falling back to YQL visit: %s", exc)
            return self._visit(doc_type="node", name_contains=query, top_k=top_k)

        yql = (
            f"select * from sources {self._schema_name} "
            f"where tenant_id contains {yql_quote(self._tenant_id)} "
            f'and doc_type contains "node" '
            f"and userQuery() limit {top_k}"
        )
        body = {
            "yql": yql,
            "query": effective_query,
            "hits": top_k,
            "ranking.profile": "hybrid_binary_bm25",
            "input.query(qtb)": {"blocks": qtb_blocks},
            "input.query(qt)": {"blocks": qt_blocks},
            # Scope rank-profile input validation to this schema — other
            # schemas in the content cluster define hybrid_binary_bm25 with
            # different query(qt) dims, and an unrestricted query 400s on
            # the conflicting declarations.
            "model.restrict": self._schema_name,
        }
        url = f"{self._backend._url}:{self._backend._port}/search/"
        # A Vespa-side failure (non-2xx, outage, soft-timeout) surfaces — the
        # YQL-visit fallback is reserved for encoder failures above; cascading
        # a failing Vespa into a second query only masked the degradation.
        resp = self._http.post(url, json=body, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"search returned {resp.status_code}: {resp.text}")
        return [h.get("fields", h) for h in vespa_search_children(resp.json())]

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
            # Visit every frontier node's edges concurrently — the GETs are
            # independent, so BFS latency scales with depth rather than total
            # frontier width. The bookkeeping below stays serial and in frontier
            # order, so the returned path is identical to fetching one at a time.
            with ThreadPoolExecutor(max_workers=min(8, len(frontier))) as executor:
                edge_lists = list(
                    executor.map(
                        lambda current: self._visit_edges(source_node_id=current),
                        (current for current, _ in frontier),
                    )
                )

            next_frontier: List[Tuple[str, List[str]]] = []
            for (current, path), out_edges in zip(frontier, edge_lists):
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
            src = edge.get("source_node_id", "")
            tgt = edge.get("target_node_id", "")
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
        """Merge nodes with the same normalized name — union Mentions by
        (source_doc_id, segment_id, ts_start, ts_end, modality)."""
        merged: Dict[str, Node] = {}
        for node in nodes:
            key = normalize_name(node.name)
            if key in merged:
                existing = merged[key]
                existing_keys = {
                    (m.source_doc_id, m.segment_id, m.ts_start, m.ts_end, m.modality)
                    for m in existing.mentions
                }
                for m in node.mentions:
                    m_key = (
                        m.source_doc_id,
                        m.segment_id,
                        m.ts_start,
                        m.ts_end,
                        m.modality,
                    )
                    if m_key not in existing_keys:
                        existing.mentions.append(m)
                        existing_keys.add(m_key)
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

    def _feed_with_retry(self, doc_id: str, fields: dict, kind: str) -> bool:
        """Put a graph node/edge through the backend document API, retrying on
        schema-convergence races.

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
                self._backend.put_document_fields(
                    doc_id,
                    fields,
                    schema_name=self._schema_name,
                    namespace=_GRAPH_NAMESPACE,
                )
                return True
            except Exception as exc:
                body = str(exc)
                transient = (
                    "does not exist" in body or "No handler for document type" in body
                )
                if not transient or attempt == 7:
                    logger.warning(
                        "Feed for %s %s failed: %s", kind, doc_id, body[:2000]
                    )
                    return False
                _time.sleep(backoff)
                backoff = min(backoff * 1.5, 8.0)
        return False

    def _feed_node(self, node: Node, tensor_fields: Dict[str, Any]) -> bool:
        payload = node.to_vespa_document()
        # tensor_fields is empty when encoding failed — ship the node
        # without embeddings rather than blocking the upsert. Mapped
        # tensor attributes tolerate absent values.
        for k, v in tensor_fields.items():
            payload["fields"][k] = v
        return self._feed_with_retry(node.doc_id, payload["fields"], "node")

    def _feed_edge(self, edge: Edge) -> bool:
        """Feed an edge document. Edges don't carry embeddings — semantic
        retrieval is over nodes only — so the mapped embedding fields are
        omitted entirely (Vespa attribute tensors handle absence)."""
        payload = edge.to_vespa_document()
        return self._feed_with_retry(edge.doc_id, payload["fields"], "edge")

    def _search_filtered(
        self, conditions: List[str], top_k: int
    ) -> List[Dict[str, Any]]:
        """Run an unranked indexed query and return hit fields.

        All filtered fields are fast-search attributes, so this is a
        dictionary lookup — the Document-v1 visit-with-selection this
        replaces scanned the tenant's whole graph corpus per call.
        """
        url = f"{self._backend._url}:{self._backend._port}"
        body = {
            "yql": (
                f"select * from {self._schema_name} where {' and '.join(conditions)}"
            ),
            "hits": top_k,
            # The default query profile caps hits at 400; graph fetches ask
            # for up to 2000, so raise the native limits per request.
            "maxHits": top_k,
            "maxOffset": top_k,
            "ranking": "unranked",
        }
        resp = self._http.post(f"{url}/search/", json=body, timeout=15)
        if not resp.ok:
            raise RuntimeError(
                f"Graph query failed ({resp.status_code}): {resp.text[:200]}"
            )
        return [c.get("fields", {}) for c in vespa_search_children(resp.json())]

    def _visit(
        self,
        doc_type: str,
        top_k: int = 100,
        name_contains: Optional[str] = None,
        source_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch documents of a given doc_type for this tenant, optionally
        scoped server-side to a name filter or one ``source_doc_id``."""
        conditions = [
            f"doc_type contains {yql_quote(doc_type)}",
            f"tenant_id contains {yql_quote(self._tenant_id)}",
        ]
        if name_contains:
            conditions.append(f"name contains {yql_quote(name_contains)}")
        if source_doc_id:
            conditions.append(f"source_doc_id contains {yql_quote(source_doc_id)}")
        return self._search_filtered(conditions, top_k)

    def _visit_edges(
        self,
        source_node_id: Optional[str] = None,
        target_node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch edges filtered by source or target node id."""
        conditions = [
            'doc_type contains "edge"',
            f"tenant_id contains {yql_quote(self._tenant_id)}",
        ]
        if source_node_id:
            conditions.append(f"source_node_id contains {yql_quote(source_node_id)}")
        if target_node_id:
            conditions.append(f"target_node_id contains {yql_quote(target_node_id)}")
        return self._search_filtered(conditions, top_k=100)

    def get_edge_by_id(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single edge's fields by its deterministic ``edge_id``.

        The edge doc_id is ``kg_edge_{safe_tenant}_{edge_id}`` (see
        ``Edge.doc_id``), so a direct by-id document GET resolves it without
        visiting and filtering the whole edge set. Returns ``None`` only when
        the edge does not exist; a backend failure raises so an outage is
        never mistaken for a missing edge.
        """
        doc_id = f"kg_edge_{_safe_tenant(self._tenant_id)}_{edge_id}"
        return self._backend.get_document_fields(
            doc_id, schema_name=self._schema_name, namespace=_GRAPH_NAMESPACE
        )
