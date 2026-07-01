"""MultiDocumentSynthesisAgent.

Synthesises a coherent answer over N source documents (10–500), preserving
the citation graph that the provenance layer makes possible. RLM-capable: when the projected
document context exceeds the threshold, the agent runs through
``RLMInference`` to recursively decompose the input.

Each synthesis writes a new memory of kind ``synthesis_fact`` (registered
on the fly via the schema registry) carrying ``provenance.derivation_kind
= SYNTHESIS`` and ``derived_from`` referencing every document the LLM
actually used. Downstream tools (CitationTracingAgent) can walk the
chain back.

Inputs are flexible: a caller can either supply document dicts directly
(``{id, content}``) or a list of memory ids; the agent fetches memories
on demand and passes the joined context to the LLM. RLM-compatible
``rlm`` options are honoured.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import Field

from cogniverse_agents.graph_bindable import GraphBindableMixin
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)

logger = logging.getLogger(__name__)


_AGENT_PORT_DEFAULT = 8021
_SYNTHESIS_MEMORY_KIND = "synthesis_fact"


class DocumentRef(AgentInput):
    """Inline document supplied by the caller.

    Either ``content`` is provided directly (caller supplies the bytes) or
    ``memory_id`` references a previously-stored memory.
    """

    memory_id: Optional[str] = Field(
        None, description="Existing memory to load for synthesis"
    )
    content: Optional[str] = Field(
        None, description="Inline document content (alternative to memory_id)"
    )
    label: Optional[str] = Field(
        None, description="Human-readable name (rendered into the LLM prompt)"
    )


class MultiDocSynthesisInput(AgentInput):
    """Inputs for a multi-document synthesis."""

    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    query: str = Field(..., description="Synthesis prompt / user question")
    documents: List[DocumentRef] = Field(
        ..., description="Documents to synthesise across (10–500)", min_length=1
    )
    persist: bool = Field(
        True,
        description=(
            "When True, store the synthesised answer as a new memory of "
            "kind ``synthesis_fact`` with provenance pointing at the "
            "input documents. Set False for read-only / audit runs."
        ),
    )
    rlm: Optional[RLMOptions] = Field(
        None,
        description=(
            "Optional RLM configuration. When set with ``enabled=True`` or "
            "auto-detected past the context threshold, the agent runs the "
            "synthesis via RLMInference."
        ),
    )


class KGClaimGroupOut(AgentInput):
    """Claims grounded in one source video, from the shared Vespa knowledge
    graph (bound at dispatch). Complements the mem0 document synthesis — it
    surfaces the cross-document grounded claims the per-agent memory lacks."""

    video_id: str
    segment_ids: List[str] = Field(default_factory=list)
    claims: List[str] = Field(
        default_factory=list, description="``<relation>:<target>`` per claim"
    )


class MultiDocSynthesisOutput(AgentOutput):
    """Synthesised answer plus citation metadata."""

    answer: str = Field(..., description="The synthesised answer text")
    citation_refs: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Per-document citation references attached to the synthesis "
            "(``ref_kind``, ``ref_id``, optional ``label``)"
        ),
    )
    kg_claim_groups: List[KGClaimGroupOut] = Field(
        default_factory=list,
        description=(
            "Claims grouped by source video from the shared Vespa knowledge "
            "graph (bound at dispatch). Empty when no graph is bound. "
            "Complements the mem0 document synthesis."
        ),
    )
    persisted_memory_id: Optional[str] = Field(
        None,
        description=(
            "ID of the memory created for the synthesis when ``persist=True``"
        ),
    )
    used_rlm: bool = Field(
        False,
        description="True iff the answer was produced via RLMInference",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Telemetry: ``document_count``, ``context_chars``, ``derivation_kind``"
        ),
    )


class MultiDocSynthesisDeps(AgentDeps):
    """Tenant-agnostic deps."""

    pass


# --- DSPy signature for the no-RLM path ---


class _SynthesisSignature(dspy.Signature):
    """Synthesise a coherent answer that cites the relevant documents."""

    query: str = dspy.InputField(desc="The user's question / synthesis goal")
    documents: str = dspy.InputField(
        desc="Numbered, labelled documents to synthesise across"
    )
    answer: str = dspy.OutputField(
        desc="A coherent answer that cites documents by their numbered label"
    )


def _format_documents_for_prompt(refs: List[DocumentRef], contents: List[str]) -> str:
    lines: List[str] = []
    for i, (ref, content) in enumerate(zip(refs, contents), start=1):
        label = ref.label or ref.memory_id or f"doc_{i}"
        lines.append(f"=== Document {i} (label={label}) ===\n{content}")
    return "\n\n".join(lines)


class MultiDocumentSynthesisAgent(
    GraphBindableMixin,
    MemoryAwareMixin,
    A2AAgent[MultiDocSynthesisInput, MultiDocSynthesisOutput, MultiDocSynthesisDeps],
):
    """A2A agent that produces a synthesised answer across N documents."""

    def __init__(
        self,
        deps: MultiDocSynthesisDeps,
        llm_config=None,
        config_manager=None,
        port: int = _AGENT_PORT_DEFAULT,
    ):
        config = A2AAgentConfig(
            agent_name="multi_document_synthesis_agent",
            agent_description=(
                "Synthesises an answer across N source documents while "
                "preserving the citation graph for audit. RLM-capable for "
                "large document sets."
            ),
            capabilities=[
                "multi_document_synthesis",
                "citation_preservation",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        from cogniverse_agents._llm_resolution import resolve_llm_config

        # Fall back to the system primary LM via config_manager when no
        # explicit llm_config was passed.
        self._llm_config = resolve_llm_config(llm_config, config_manager)
        self._config_manager = config_manager
        self._dspy_module = dspy.ChainOfThought(_SynthesisSignature)

    def synthesize(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group all known claims by source video for the orchestrator to render.

        Walks every Edge in the graph, then partitions by ``source_doc_id``.
        Each group: ``{video_id, segment_ids (sorted, unique), claims
        (in ``"<relation>:<target>"`` form, in segment order)}``. Groups are
        emitted sorted by ``video_id``. ``query`` is accepted on the signature
        but not used for filtering at this layer — the orchestrator scopes
        which subjects the synthesis ranges over upstream.
        """
        graph_manager = self._require_graph_manager("synthesize")
        all_edges = graph_manager._visit(doc_type="edge", top_k=2000)
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for edge_fields in all_edges:
            video_id = str(edge_fields.get("source_doc_id") or "")
            if not video_id:
                continue
            by_video.setdefault(video_id, []).append(edge_fields)

        groups: List[Dict[str, Any]] = []
        for video_id in sorted(by_video):
            edges = by_video[video_id]
            edges_sorted = sorted(
                edges,
                key=lambda e: (
                    str(e.get("segment_id") or ""),
                    str(e.get("relation") or ""),
                    str(e.get("target_node_id") or ""),
                ),
            )
            segment_ids = sorted(
                {
                    str(e.get("segment_id") or "")
                    for e in edges_sorted
                    if e.get("segment_id")
                }
            )
            claims: List[str] = []
            for e in edges_sorted:
                rel = str(e.get("relation") or "")
                tgt = str(e.get("target_node_id") or "")
                if rel and tgt:
                    claims.append(f"{rel}:{tgt}")
            groups.append(
                {"video_id": video_id, "segment_ids": segment_ids, "claims": claims}
            )
        return {"groups": groups}

    def _kg_claim_groups(self) -> List[KGClaimGroupOut]:
        """Group the bound Vespa KG's claims by source video, complementary to
        the mem0 document synthesis. Empty when no graph is bound."""
        if self._graph_manager is None:
            return []
        try:
            kg = self.synthesize(query="")
        except Exception as exc:
            logger.debug("multidoc: Vespa-KG complement skipped: %s", exc)
            return []
        return [
            KGClaimGroupOut(
                video_id=str(g.get("video_id") or ""),
                segment_ids=[str(s) for s in g.get("segment_ids", [])],
                claims=[str(c) for c in g.get("claims", [])],
            )
            for g in kg.get("groups", [])
        ]

    async def _process_impl(
        self, input: MultiDocSynthesisInput
    ) -> MultiDocSynthesisOutput:
        kg_claim_groups = self._kg_claim_groups()

        # Resolve each DocumentRef to (citation_ref, content).
        refs_resolved: List[DocumentRef] = []
        contents: List[str] = []
        for ref in input.documents:
            content = await self._resolve_document(ref)
            if content is None:
                logger.debug(
                    "Skipping unresolved document ref: memory_id=%s label=%s",
                    ref.memory_id,
                    ref.label,
                )
                continue
            refs_resolved.append(ref)
            contents.append(content)

        if not contents:
            return MultiDocSynthesisOutput(
                answer="",
                citation_refs=[],
                kg_claim_groups=kg_claim_groups,
                persisted_memory_id=None,
                used_rlm=False,
                metadata={
                    "reason": "no_resolvable_documents",
                    "kg_claim_group_count": len(kg_claim_groups),
                },
            )

        documents_block = _format_documents_for_prompt(refs_resolved, contents)
        context_chars = len(documents_block)

        used_rlm = False
        rlm_options = input.rlm
        if rlm_options is not None and rlm_options.should_use_rlm(context_chars):
            answer = await self._synthesise_with_rlm(
                input.query, documents_block, rlm_options
            )
            used_rlm = True
        else:
            # Offload the blocking DSPy LM call (and its dspy.context, set
            # inside the sync method so it lands on the worker thread) so the
            # event loop isn't stalled for the whole synthesis round-trip.
            answer = await asyncio.to_thread(
                self._synthesise_without_rlm, input.query, documents_block
            )

        citation_refs = [
            CitationRef.memory(ref.memory_id, label=ref.label)
            if ref.memory_id is not None
            else CitationRef.external(
                ref.label or "inline:document",
                label=ref.label,
            )
            for ref in refs_resolved
        ]

        persisted_id: Optional[str] = None
        if input.persist:
            persisted_id = await self._persist_synthesis(
                tenant_id=input.tenant_id or self._memory_tenant_id,
                answer=answer,
                citation_refs=citation_refs,
            )

        return MultiDocSynthesisOutput(
            answer=answer,
            citation_refs=[
                {
                    "ref_kind": r.ref_kind,
                    "ref_id": r.ref_id,
                    **({"label": r.label} if r.label else {}),
                }
                for r in citation_refs
            ],
            kg_claim_groups=kg_claim_groups,
            persisted_memory_id=persisted_id,
            used_rlm=used_rlm,
            metadata={
                "document_count": len(refs_resolved),
                "context_chars": context_chars,
                "derivation_kind": DerivationKind.SYNTHESIS.value,
                "kg_claim_group_count": len(kg_claim_groups),
            },
        )

    # --- internals ---------------------------------------------------------

    async def _resolve_document(self, ref: DocumentRef) -> Optional[str]:
        """Return the document content, fetching memory by id if needed."""
        if ref.content:
            return ref.content
        if (
            ref.memory_id
            and self.is_memory_enabled()
            and self.memory_manager is not None
        ):
            try:
                memory = self.memory_manager.memory.get(ref.memory_id)
            except Exception as exc:
                logger.debug("MultiDocSynth: get(%s) failed: %s", ref.memory_id, exc)
                return None
            if isinstance(memory, dict):
                return memory.get("memory") or memory.get("content") or ""
            if isinstance(memory, list) and memory and isinstance(memory[0], dict):
                return memory[0].get("memory") or memory[0].get("content") or ""
        return None

    def _synthesise_without_rlm(self, query: str, documents_block: str) -> str:
        """Single LM call via DSPy ChainOfThought."""
        if self._llm_config is not None:
            from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
            from cogniverse_foundation.config.gateway_routing import (
                routed_lm_context_for,
            )

            tenant_id = getattr(self, "_memory_tenant_id", None) or SYSTEM_TENANT_ID
            with routed_lm_context_for(
                self._config_manager,
                tenant_id,
                "multi_document_synthesis_agent",
                endpoint=self._llm_config,
            ):
                result = self._dspy_module(query=query, documents=documents_block)
        else:
            # Use ambient dspy.settings.lm if no per-agent override.
            result = self._dspy_module(query=query, documents=documents_block)
        return getattr(result, "answer", "") or ""

    async def _synthesise_with_rlm(
        self,
        query: str,
        documents_block: str,
        rlm_options: RLMOptions,
    ) -> str:
        from cogniverse_agents.inference.rlm_inference import build_rlm_from_options

        rlm = build_rlm_from_options(self._llm_config, rlm_options)
        result = await asyncio.to_thread(
            rlm.process,
            query=query,
            context=documents_block,
            include_trajectory=rlm_options.include_trajectory,
            trajectory_max_entries=rlm_options.trajectory_max_entries,
        )
        return result.answer

    async def _persist_synthesis(
        self,
        *,
        tenant_id: Optional[str],
        answer: str,
        citation_refs: List[CitationRef],
    ) -> Optional[str]:
        """Write the synthesis to memory with full provenance."""
        if not (
            self.is_memory_enabled() and self.memory_manager is not None and tenant_id
        ):
            return None
        provenance = make_provenance(
            written_by="agent:multi_document_synthesis_agent",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=0.7,
            derived_from=citation_refs,
        )
        metadata = attach_to_metadata({"kind": _SYNTHESIS_MEMORY_KIND}, provenance)
        try:
            return self.memory_manager.add_memory(
                content=answer,
                tenant_id=tenant_id,
                agent_name="multi_document_synthesis_agent",
                metadata=metadata,
                infer=False,
            )
        except Exception as exc:
            logger.warning(
                "MultiDocSynth: persist failed for tenant=%s: %s", tenant_id, exc
            )
            return None
