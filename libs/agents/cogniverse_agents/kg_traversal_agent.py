"""KnowledgeGraphTraversalAgent.

Walks the entity-fact / kg_node / kg_edge memories starting from an entity
or specific node id, returning a structured graph view (nodes + edges)
plus a list of facts the walk surfaced. RLM-capable for very large
traversals — the deterministic walker calls an RLM summariser when the
collected output exceeds the RLM threshold and ``rlm`` options are set.

Edge memories are expected to carry metadata:

  * ``kind`` = ``kg_edge``
  * ``from_subject_key`` / ``to_subject_key`` (canonical entity ids)
  * ``relation`` (string, e.g. "located_in", "wrote", "depends_on")

Node memories are expected to carry:

  * ``kind`` = ``kg_node`` (or ``entity_fact`` — both treated as nodes)
  * ``subject_key`` (canonical entity id)

Memories without those tags are ignored — the walker is a structural
graph reader, not a free-text searcher. (MultiDocumentSynthesisAgent covers free-text synthesis.)
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from cogniverse_agents.graph.graph_schema import normalize_name
from cogniverse_agents.graph_bindable import GraphBindableMixin
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8022
_NODE_KINDS = frozenset({"kg_node", "entity_fact"})
_EDGE_KIND = "kg_edge"


class KGTraversalInput(AgentInput):
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    query: str = Field(
        "",
        description=(
            "Optional context label for telemetry; the actual traversal is "
            "driven by ``start_subject_key`` / ``start_memory_id``."
        ),
    )
    start_subject_key: Optional[str] = Field(
        None,
        description=(
            "Entity to start the walk from (matches a node's ``subject_key``)."
        ),
    )
    start_memory_id: Optional[str] = Field(
        None,
        description=(
            "Alternative starting point — the agent looks up the node by id "
            "and uses its ``subject_key`` as the seed."
        ),
    )
    max_depth: int = Field(
        3,
        ge=1,
        le=8,
        description="BFS depth cap (cycle / runaway protection)",
    )
    max_edges: int = Field(
        100,
        ge=1,
        le=2000,
        description="Stop after enumerating this many edges total",
    )
    relation_allowlist: Optional[List[str]] = Field(
        None,
        description=(
            "When provided, follow only edges whose ``relation`` matches "
            "(case-sensitive). When None, follow every relation."
        ),
    )
    rlm: Optional[RLMOptions] = Field(
        None,
        description=(
            "Optional RLM summariser. When set with ``enabled=True`` (or "
            "auto-detected past the context threshold) the agent runs a "
            "summarisation pass over the walked graph for caller convenience."
        ),
    )


class KGNodeOut(AgentInput):
    memory_id: str
    subject_key: str
    label: str = ""
    excerpt: str = ""


class KGEdgeOut(AgentInput):
    memory_id: str
    from_subject_key: str
    to_subject_key: str
    relation: str


class KGTraversalOutput(AgentOutput):
    start_subject_key: str = Field("", description="Resolved starting subject_key")
    nodes: List[KGNodeOut] = Field(default_factory=list)
    edges: List[KGEdgeOut] = Field(default_factory=list)
    summary: Optional[str] = Field(
        None,
        description=("LLM-summarised view when ``rlm`` was enabled or auto-detected"),
    )
    truncated: bool = Field(
        False,
        description="True when traversal hit max_depth or max_edges before exhausting",
    )
    used_rlm: bool = Field(False, description="Whether the summariser ran")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Telemetry: ``visited_nodes``, ``visited_edges``, ``relation_counts``"
        ),
    )


class KGTraversalDeps(AgentDeps):
    pass


def _read_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
    import json as _json

    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            return _json.loads(meta) or {}
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _is_node(meta: Dict[str, Any]) -> bool:
    return str(meta.get("kind") or "") in _NODE_KINDS


def _is_edge(meta: Dict[str, Any]) -> bool:
    return str(meta.get("kind") or "") == _EDGE_KIND


def _ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Closed-interval overlap on the time axis."""
    return a_start <= b_end and b_start <= a_end


def _node_passes_mention_filter(
    node_fields: Dict[str, Any],
    video_id: Optional[str],
    ts_range: Optional[Tuple[float, float]],
) -> bool:
    """Return True iff any Mention on the node matches the filter.

    Mentions are stored as a JSON string under ``fields["mentions"]`` by
    ``Node.to_vespa_document``. Each parsed dict carries ``source_doc_id``,
    ``segment_id``, ``ts_start``, ``ts_end``, ``modality``, ``evidence_span``.
    """
    raw = node_fields.get("mentions")
    if not raw:
        return False
    if isinstance(raw, str):
        try:
            mentions = json.loads(raw)
        except (ValueError, TypeError):
            return False
    elif isinstance(raw, list):
        mentions = raw
    else:
        return False
    for m in mentions:
        if not isinstance(m, dict):
            continue
        if video_id is not None and str(m.get("source_doc_id") or "") != str(video_id):
            continue
        if ts_range is not None:
            ms = float(m.get("ts_start") or 0.0)
            me = float(m.get("ts_end") or 0.0)
            if not _ranges_overlap(ms, me, ts_range[0], ts_range[1]):
                continue
        return True
    return False


class KnowledgeGraphTraversalAgent(
    GraphBindableMixin,
    MemoryAwareMixin,
    A2AAgent[KGTraversalInput, KGTraversalOutput, KGTraversalDeps],
):
    """A2A agent that walks the knowledge graph structurally."""

    def __init__(
        self,
        deps: KGTraversalDeps,
        llm_config=None,
        config_manager=None,
        port: int = _DEFAULT_PORT,
    ):
        config = A2AAgentConfig(
            agent_name="kg_traversal_agent",
            agent_description=(
                "Walks the entity / edge memories of the knowledge graph "
                "from a starting node, returning a structured node+edge "
                "view and an optional LLM summary."
            ),
            capabilities=[
                "kg_traversal",
                "graph_walk",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        from cogniverse_agents._llm_resolution import resolve_llm_config

        # Fall back to the system primary LM via config_manager when no
        # explicit llm_config was passed.
        self._llm_config = resolve_llm_config(llm_config, config_manager)
        self._config_manager = config_manager

    def traverse(
        self, node_name: str, filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Traverse outgoing edges from ``node_name`` with optional filters.

        ``filters`` keys:
          * ``video_id`` — keep only Mentions / Edges whose ``source_doc_id``
            matches.
          * ``ts_range`` — ``(start, end)`` tuple; keep only Edges whose
            ``[ts_start, ts_end]`` overlaps the requested window, AND only
            Mentions whose ``[ts_start, ts_end]`` overlaps.

        Returns ``{"nodes": [<target_node_id>, ...], "edges": [{"source":..,
        "relation":.., "target":..}, ...]}``.
        """
        graph_manager = self._require_graph_manager("traverse")
        filters = filters or {}
        video_id = filters.get("video_id")
        ts_range: Optional[Tuple[float, float]] = filters.get("ts_range")

        source_id = normalize_name(node_name)
        out_edges = graph_manager._visit_edges(source_node_id=source_id)

        kept_edges: List[Dict[str, str]] = []
        target_ids: List[str] = []
        for edge_fields in out_edges:
            if video_id is not None and str(
                edge_fields.get("source_doc_id") or ""
            ) != str(video_id):
                continue
            if ts_range is not None:
                e_start = float(edge_fields.get("ts_start") or 0.0)
                e_end = float(edge_fields.get("ts_end") or 0.0)
                if not _ranges_overlap(e_start, e_end, ts_range[0], ts_range[1]):
                    continue
            target = str(edge_fields.get("target_node_id") or "")
            relation = str(edge_fields.get("relation") or "")
            source = str(edge_fields.get("source_node_id") or "")
            if not target or not relation:
                continue
            kept_edges.append(
                {"source": source, "relation": relation, "target": target}
            )
            target_ids.append(target)

        # Filter the target node set further by Mention overlap when filters demand it.
        if video_id is not None or ts_range is not None:
            unique_targets = list(dict.fromkeys(target_ids))
            allowed: List[str] = []
            all_nodes = graph_manager._visit(doc_type="node", top_k=500)
            node_by_id: Dict[str, Dict[str, Any]] = {}
            for n in all_nodes:
                nid = normalize_name(str(n.get("name") or ""))
                node_by_id[nid] = n
            for tgt in unique_targets:
                node_fields = node_by_id.get(tgt)
                if node_fields is None:
                    continue
                if _node_passes_mention_filter(node_fields, video_id, ts_range):
                    allowed.append(tgt)
            nodes_out = sorted(allowed)
            kept_edges = [e for e in kept_edges if e["target"] in set(allowed)]
        else:
            nodes_out = sorted(set(target_ids))

        kept_edges.sort(key=lambda e: (e["source"], e["relation"], e["target"]))
        return {"nodes": nodes_out, "edges": kept_edges}

    async def _process_impl(self, input: KGTraversalInput) -> KGTraversalOutput:
        memory_available = self.is_memory_enabled() and self.memory_manager is not None
        graph_available = self._graph_manager is not None
        if not memory_available and not graph_available:
            logger.warning(
                "KGTraversal invoked without a memory manager or bound graph; "
                "returning empty graph"
            )
            return KGTraversalOutput(
                start_subject_key="",
                metadata={"reason": "no_backend_available"},
            )

        seed_subject = self._resolve_seed_subject(input)
        if not seed_subject:
            return KGTraversalOutput(
                start_subject_key="",
                metadata={"reason": "no_seed_resolved"},
            )

        # Walk the agent's own mem0 memory (when present). The shared Vespa
        # knowledge graph, bound at dispatch, is merged in below as a
        # complement — it carries cross-document nodes/edges the per-agent mem0
        # store lacks.
        nodes_by_subject: Dict[str, Any] = {}
        edges_by_from: Dict[str, List[Any]] = {}
        if memory_available:
            tenant_id = input.tenant_id or self._memory_tenant_id or ""
            snapshot = await asyncio.to_thread(
                self.memory_manager.get_all_memories,
                tenant_id=tenant_id,
                agent_name=getattr(self, "_memory_agent_name", ""),
            )
            nodes_by_subject = self._index_nodes(snapshot)
            edges_by_from = self._index_edges_by_from(snapshot)

        visited_nodes: List[KGNodeOut] = []
        visited_edges: List[KGEdgeOut] = []
        seen_subjects: set = set()
        relation_counts: Dict[str, int] = {}
        truncated = False

        frontier: deque[tuple[str, int]] = deque([(seed_subject, 0)])
        while frontier:
            if len(visited_edges) >= input.max_edges:
                truncated = True
                break
            subject, depth = frontier.popleft()
            if subject in seen_subjects:
                continue
            seen_subjects.add(subject)

            node_memory = nodes_by_subject.get(subject)
            if node_memory is not None:
                meta = _read_metadata(node_memory)
                visited_nodes.append(
                    KGNodeOut(
                        memory_id=str(node_memory.get("id") or ""),
                        subject_key=subject,
                        label=str(meta.get("label") or ""),
                        excerpt=str(
                            node_memory.get("memory")
                            or node_memory.get("content")
                            or ""
                        )[:200],
                    )
                )

            if depth + 1 > input.max_depth:
                continue

            for edge_memory in edges_by_from.get(subject, []):
                if len(visited_edges) >= input.max_edges:
                    truncated = True
                    break
                meta = _read_metadata(edge_memory)
                relation = str(meta.get("relation") or "")
                if (
                    input.relation_allowlist
                    and relation not in input.relation_allowlist
                ):
                    continue
                to_subject = str(meta.get("to_subject_key") or "")
                if not to_subject:
                    continue
                visited_edges.append(
                    KGEdgeOut(
                        memory_id=str(edge_memory.get("id") or ""),
                        from_subject_key=subject,
                        to_subject_key=to_subject,
                        relation=relation,
                    )
                )
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                if to_subject not in seen_subjects:
                    frontier.append((to_subject, depth + 1))

        # Complement the mem0 walk with the shared Vespa knowledge graph when a
        # GraphManager is bound (at dispatch). The KG carries cross-document
        # nodes/edges (with Mention provenance) the per-agent mem0 store lacks;
        # mem0 results are kept, KG results merged in deduplicated.
        if graph_available:
            try:
                kg = await asyncio.to_thread(self.traverse, seed_subject)
            except Exception as exc:
                logger.debug(
                    "Vespa-KG complement skipped for %s: %s", seed_subject, exc
                )
                kg = None
            if kg:
                seen_edges = {
                    (e.from_subject_key, e.relation, e.to_subject_key)
                    for e in visited_edges
                }
                for edge in kg.get("edges", []):
                    src = str(edge.get("source") or "")
                    rel = str(edge.get("relation") or "")
                    tgt = str(edge.get("target") or "")
                    if not (src and tgt) or (src, rel, tgt) in seen_edges:
                        continue
                    if input.relation_allowlist and rel not in input.relation_allowlist:
                        continue
                    seen_edges.add((src, rel, tgt))
                    visited_edges.append(
                        KGEdgeOut(
                            memory_id="",
                            from_subject_key=src,
                            to_subject_key=tgt,
                            relation=rel,
                        )
                    )
                    relation_counts[rel] = relation_counts.get(rel, 0) + 1
                known_subjects = {n.subject_key for n in visited_nodes}
                for node_id in kg.get("nodes", []):
                    nid = str(node_id)
                    if nid and nid not in known_subjects:
                        known_subjects.add(nid)
                        visited_nodes.append(
                            KGNodeOut(memory_id="", subject_key=nid, label="")
                        )

        # Optional RLM summarisation when enabled / auto-detected.
        used_rlm = False
        summary: Optional[str] = None
        rlm_options = input.rlm
        if rlm_options is not None:
            block = self._format_graph_for_summary(visited_nodes, visited_edges)
            if rlm_options.should_use_rlm(len(block)):
                summary = await self._summarise_with_rlm(
                    input.query or f"summarise the graph rooted at {seed_subject}",
                    block,
                    rlm_options,
                )
                used_rlm = True

        return KGTraversalOutput(
            start_subject_key=seed_subject,
            nodes=visited_nodes,
            edges=visited_edges,
            summary=summary,
            truncated=truncated,
            used_rlm=used_rlm,
            metadata={
                "visited_nodes": len(visited_nodes),
                "visited_edges": len(visited_edges),
                "relation_counts": relation_counts,
            },
        )

    # --- internals ---------------------------------------------------------

    def _resolve_seed_subject(self, input: KGTraversalInput) -> str:
        if input.start_subject_key:
            return input.start_subject_key
        if input.start_memory_id and self.memory_manager is not None:
            try:
                memory = self.memory_manager.memory.get(input.start_memory_id)
            except Exception as exc:
                logger.debug(
                    "KGTraversal: get(%s) failed: %s",
                    input.start_memory_id,
                    exc,
                )
                return ""
            if isinstance(memory, dict):
                meta = _read_metadata(memory)
                return str(meta.get("subject_key") or "")
        return ""

    @staticmethod
    def _index_nodes(snapshot: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for memory in snapshot:
            meta = _read_metadata(memory)
            if not _is_node(meta):
                continue
            subject = str(meta.get("subject_key") or "")
            if subject and subject not in out:
                out[subject] = memory
        return out

    @staticmethod
    def _index_edges_by_from(
        snapshot: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for memory in snapshot:
            meta = _read_metadata(memory)
            if not _is_edge(meta):
                continue
            from_subject = str(meta.get("from_subject_key") or "")
            if not from_subject:
                continue
            out.setdefault(from_subject, []).append(memory)
        return out

    @staticmethod
    def _format_graph_for_summary(
        nodes: List[KGNodeOut], edges: List[KGEdgeOut]
    ) -> str:
        lines: List[str] = []
        lines.append("== Nodes ==")
        for n in nodes:
            label = n.label or n.subject_key
            lines.append(f"- {n.subject_key} ({label}): {n.excerpt}")
        lines.append("\n== Edges ==")
        for e in edges:
            lines.append(f"- {e.from_subject_key} --{e.relation}--> {e.to_subject_key}")
        return "\n".join(lines)

    async def _summarise_with_rlm(
        self, query: str, block: str, rlm_options: RLMOptions
    ) -> str:
        from cogniverse_agents.inference.rlm_inference import build_rlm_from_options

        rlm = build_rlm_from_options(
            self._llm_config,
            rlm_options,
            config_manager=getattr(self, "_config_manager", None),
            tenant_id=getattr(self, "_memory_tenant_id", None) or "",
        )
        # Multi-call RLM LLM loop is synchronous — run it off the event loop
        # (same treatment as federated_query_agent / temporal_reasoning_agent).
        result = await asyncio.to_thread(rlm.process, query=query, context=block)
        return result.answer
