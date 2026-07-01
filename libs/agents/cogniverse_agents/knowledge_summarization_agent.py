"""KnowledgeSummarizationAgent.

Distills a knowledge subgraph (a subject area, a kind, optionally a time
window) into a structured summary suitable for org-trunk promotion. This
is distinct from `SummarizerAgent` (which summarises retrieval results
in-flight): this agent summarises the knowledge layer *itself*, with optional
admin-gated promotion of the summary into the org trunk via the
federation read path.

The agent:

  1. Selects memories matching ``subject_keys`` (any-of) and/or ``kinds``
     (any-of), optionally within a `[since, until)` window.
  2. Synthesises a summary (DSPy ChainOfThought; RLM when the joined
     context exceeds the threshold).
  3. Returns a `KnowledgeSummary` with citation refs to every source.
  4. Optionally promotes the summary to the org trunk when
     ``promote=True`` AND ``actor_role`` is at-or-above the schema's
     pin floor for ``knowledge_summary``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import dspy
from pydantic import Field

from cogniverse_agents.graph.graph_schema import normalize_name
from cogniverse_agents.graph_bindable import GraphBindableMixin
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.temporal_reasoning_agent import _parse_iso
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_core.memory.federation import FederationService
from cogniverse_core.memory.provenance import CitationRef
from cogniverse_core.memory.schema import (
    SUMMARY_KIND,
    KnowledgeRegistry,
    Pinnable,
    SchemaViolationError,
    build_default_registry,
    register_summary_kind,
)

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8026


class KnowledgeSummarizationInput(AgentInput):
    tenant_id: Optional[str] = Field(None)
    subject_keys: Optional[List[str]] = Field(
        None,
        description="Match memories whose metadata.subject_key is any of these",
    )
    kinds: Optional[List[str]] = Field(
        None,
        description="Match memories whose metadata.kind is any of these",
    )
    since: Optional[str] = Field(
        None, description="ISO-8601 inclusive lower bound on written_at"
    )
    until: Optional[str] = Field(
        None, description="ISO-8601 exclusive upper bound on written_at"
    )
    agent_name_filter: Optional[str] = Field(
        None,
        description="Restricts the read to one agent_name namespace",
    )
    max_memories: int = Field(
        500,
        ge=1,
        le=5000,
        description="Cap the in-context evidence — protects token budget",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Caller-provided title for the summary (telemetry + audit)",
    )
    promote: bool = Field(
        False,
        description=(
            "When True, write the summary into the org trunk. Requires "
            "actor_role >= tenant_admin (gated by schema sensitivity)."
        ),
    )
    actor_role: str = Field(
        "user",
        description="Caller role; only ``tenant_admin``/``org_admin`` may promote",
    )
    actor_id: str = Field(..., description="Caller actor id (audit)")
    rlm: Optional[RLMOptions] = Field(None)


class KGVideoSummaryOut(AgentInput):
    """Per-segment claim summary for one video that mentions the requested
    subject(s), rendered from the shared Vespa knowledge graph (bound at
    dispatch). Complements the mem0 ``summary`` — it grounds the subjects'
    claims in segment-level provenance the per-agent memory store lacks."""

    video_id: str
    text: str


class KnowledgeSummarizationOutput(AgentOutput):
    title: str
    summary: str
    citation_refs: List[CitationRef] = Field(default_factory=list)
    source_count: int = 0
    used_rlm: bool = False
    promoted_to_org_trunk: bool = False
    promoted_memory_id: Optional[str] = None
    kg_video_summaries: List[KGVideoSummaryOut] = Field(
        default_factory=list,
        description=(
            "Per-video KG claim summaries for videos mentioning the requested "
            "subject_keys (bound graph). Empty when no graph is bound or no "
            "subject_keys are given. Complements the mem0 ``summary``."
        ),
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeSummarizationDeps(AgentDeps):
    pass


class _SummarizationSignature(dspy.Signature):
    """Summarise the supplied knowledge memories under the given title."""

    title: str = dspy.InputField(desc="The summary's caller-supplied title")
    memories: str = dspy.InputField(
        desc="Numbered, labelled memories to summarise across"
    )
    summary: str = dspy.OutputField(
        desc=(
            "A coherent, citation-aware summary that references memories "
            "by their numbered label. Don't fabricate facts not present."
        )
    )


def _matches_filters(
    row: Dict[str, Any],
    subject_keys: Optional[List[str]],
    kinds: Optional[List[str]],
    since_dt: Optional[datetime],
    until_dt: Optional[datetime],
) -> bool:
    meta = row.get("metadata") or {}
    if not isinstance(meta, dict):
        return False
    if subject_keys and meta.get("subject_key") not in subject_keys:
        return False
    if kinds and meta.get("kind") not in kinds:
        return False
    if since_dt is not None or until_dt is not None:
        ts = _parse_iso(meta.get("written_at"))
        if ts is None:
            return False
        if since_dt is not None and ts < since_dt:
            return False
        if until_dt is not None and ts >= until_dt:
            return False
    return True


def _format_memories_for_prompt(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, r in enumerate(rows, start=1):
        mid = str(r.get("id") or f"mem_{i}")
        content = (r.get("memory", "")).strip()
        meta = r.get("metadata") or {}
        subj = (meta.get("subject_key") if isinstance(meta, dict) else None) or "-"
        lines.append(f"=== Memory {i} (id={mid}, subject={subj}) ===\n{content}")
    return "\n\n".join(lines)


class KnowledgeSummarizationAgent(
    GraphBindableMixin,
    MemoryAwareMixin,
    A2AAgent[
        KnowledgeSummarizationInput,
        KnowledgeSummarizationOutput,
        KnowledgeSummarizationDeps,
    ],
):
    """A2A agent that summarises a slice of the knowledge layer."""

    def __init__(
        self,
        deps: KnowledgeSummarizationDeps,
        memory_manager_factory=None,
        registry: Optional[KnowledgeRegistry] = None,
        llm_config=None,
        config_manager=None,
        port: int = _DEFAULT_PORT,
    ):
        config = A2AAgentConfig(
            agent_name="knowledge_summarization_agent",
            agent_description=(
                "Distills a knowledge subgraph into a structured summary "
                "with citations; optionally promotes the summary to the "
                "org trunk for org-wide visibility."
            ),
            capabilities=[
                "knowledge_summarization",
                "audit",
                "federation_promoter",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        from cogniverse_agents._mm_factory import make_mm_factory

        self._mm_factory = make_mm_factory(memory_manager_factory)
        self._registry = registry or build_default_registry()
        self._llm_config = llm_config
        self._config_manager = config_manager
        self._dspy_module = dspy.ChainOfThought(_SummarizationSignature)
        self._ensure_summary_kind_registered()

    def summarize(self, video_id: str) -> Dict[str, str]:
        """One-line-per-segment summary for a single video.

        Returns ``{"text": <str>}``. Each line:
        ``"<entity_name> [<ts_start>s-<ts_end>s]: <claim_or_summary>"`` where
        ``entity_name`` is the Edge's ``source_node_id`` (the subject) and
        ``claim_or_summary`` is rendered as ``"<predicate> <object_summary>"``.
        Edges grouped by ``(subject, segment_id)``; multiple claims sharing a
        ``(subject, ts_range)`` collapse into a single line whose body joins
        the predicate/object pairs with ``"; "``.
        """
        graph_manager = self._require_graph_manager("summarize")
        all_edges = graph_manager._visit(doc_type="edge", top_k=2000)
        # Group by (subject, segment_id, ts_start, ts_end), preserving the
        # original subject name (from source_node_id) for readability.
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for edge_fields in all_edges:
            if str(edge_fields.get("source_doc_id") or "") != video_id:
                continue
            subject = str(edge_fields.get("source_node_id") or "")
            segment_id = str(edge_fields.get("segment_id") or "")
            ts_start = float(edge_fields.get("ts_start") or 0.0)
            ts_end = float(edge_fields.get("ts_end") or 0.0)
            key = (subject, segment_id, ts_start, ts_end)
            grouped.setdefault(key, []).append(edge_fields)

        lines: List[str] = []
        for key in sorted(grouped, key=lambda k: (k[2], k[3], k[0])):
            subject, _seg_id, ts_start, ts_end = key
            edges = sorted(
                grouped[key],
                key=lambda e: (
                    str(e.get("relation") or ""),
                    str(e.get("target_node_id") or ""),
                ),
            )
            body_parts: List[str] = []
            for e in edges:
                rel = str(e.get("relation") or "")
                tgt = str(e.get("target_node_id") or "")
                if rel and tgt:
                    body_parts.append(f"{rel} {tgt}")
            body = "; ".join(body_parts)
            lines.append(f"{subject} [{ts_start}s-{ts_end}s]: {body}")
        return {"text": "\n".join(lines)}

    def _ensure_summary_kind_registered(self) -> None:
        """Register the org-shared ``knowledge_summary`` kind if absent.

        The default registry marks every kind ``tenant_private``, which blocks
        promotion; this installs the same org-shared schema the admin promote
        endpoint expects (shared definition in ``schema.register_summary_kind``).
        """
        register_summary_kind(self._registry)

    def _kg_video_summaries(
        self, subject_keys: Optional[List[str]]
    ) -> List[KGVideoSummaryOut]:
        """Summarise the bound KG's claims for each video that mentions one of
        ``subject_keys``, complementary to the mem0 summary. Empty when no graph
        is bound or no subject_keys are given."""
        if self._graph_manager is None or not subject_keys:
            return []
        videos: set = set()
        for sk in subject_keys:
            try:
                edges = self._graph_manager._visit_edges(
                    source_node_id=normalize_name(sk)
                )
            except Exception as exc:
                logger.debug(
                    "summarization: KG video lookup failed for %s: %s", sk, exc
                )
                continue
            for e in edges:
                vid = str(e.get("source_doc_id") or "")
                if vid:
                    videos.add(vid)
        out: List[KGVideoSummaryOut] = []
        for vid in sorted(videos):
            try:
                text = self.summarize(vid).get("text", "")
            except Exception as exc:
                logger.debug("summarization: summarize(%s) failed: %s", vid, exc)
                continue
            out.append(KGVideoSummaryOut(video_id=vid, text=text))
        return out

    async def _process_impl(
        self, input: KnowledgeSummarizationInput
    ) -> KnowledgeSummarizationOutput:
        from cogniverse_agents._mm_factory import tenant_id_from_input_or_deps

        tenant_id = tenant_id_from_input_or_deps(
            input, self.deps, "KnowledgeSummarizationAgent"
        )

        kg_video_summaries = self._kg_video_summaries(input.subject_keys)

        rows = self._fetch_filtered(
            tenant_id,
            input.agent_name_filter or "_promoted",
            input.subject_keys,
            input.kinds,
            input.since,
            input.until,
        )
        rows = rows[: input.max_memories]

        if not rows:
            return KnowledgeSummarizationOutput(
                title=input.title,
                summary="",
                citation_refs=[],
                source_count=0,
                used_rlm=False,
                promoted_to_org_trunk=False,
                promoted_memory_id=None,
                kg_video_summaries=kg_video_summaries,
                metadata={
                    "reason": "no_matching_memories",
                    "kg_video_summary_count": len(kg_video_summaries),
                },
            )

        block = _format_memories_for_prompt(rows)
        used_rlm = False
        rlm_options = input.rlm
        if rlm_options is not None and rlm_options.should_use_rlm(len(block)):
            summary_text = await self._summarise_with_rlm(
                input.title, block, rlm_options
            )
            used_rlm = True
        else:
            summary_text = self._summarise_without_rlm(input.title, block)

        citation_refs = [
            CitationRef.memory(str(r.get("id") or ""), label=None)
            for r in rows
            if r.get("id")
        ]

        promoted = False
        promoted_id: Optional[str] = None
        if input.promote:
            try:
                promoted_id = self._promote_summary(
                    tenant_id=tenant_id,
                    title=input.title,
                    summary=summary_text,
                    citation_refs=citation_refs,
                    actor_role=input.actor_role,
                    actor_id=input.actor_id,
                )
                promoted = True
            except SchemaViolationError as exc:
                logger.warning(
                    "Promotion refused: %s (actor=%s role=%s)",
                    exc,
                    input.actor_id,
                    input.actor_role,
                )

        return KnowledgeSummarizationOutput(
            title=input.title,
            summary=summary_text,
            citation_refs=citation_refs,
            source_count=len(rows),
            used_rlm=used_rlm,
            promoted_to_org_trunk=promoted,
            promoted_memory_id=promoted_id,
            kg_video_summaries=kg_video_summaries,
            metadata={
                "subject_keys": input.subject_keys or [],
                "kinds": input.kinds or [],
                "since": input.since,
                "until": input.until,
                "actor_role": input.actor_role,
                "actor_id": input.actor_id,
                "kg_video_summary_count": len(kg_video_summaries),
            },
        )

    def _fetch_filtered(
        self,
        tenant_id: str,
        agent_name: str,
        subject_keys: Optional[List[str]],
        kinds: Optional[List[str]],
        since: Optional[str],
        until: Optional[str],
    ) -> List[Dict[str, Any]]:
        try:
            mm = self._mm_factory(tenant_id)
        except Exception as exc:
            logger.debug("ksum: factory(%s) failed: %s", tenant_id, exc)
            return []
        if mm is None or not getattr(mm, "memory", None):
            return []
        try:
            rows = list(mm.get_all_memories(tenant_id=tenant_id, agent_name=agent_name))
        except Exception as exc:
            logger.debug("ksum: get_all_memories failed: %s", exc)
            return []
        since_dt = _parse_iso(since)
        until_dt = _parse_iso(until)
        return [
            r
            for r in rows
            if _matches_filters(r, subject_keys, kinds, since_dt, until_dt)
        ]

    def _summarise_without_rlm(self, title: str, block: str) -> str:
        try:
            # Bind the per-tenant LM the same way multi_document_synthesis does
            # (see multi_document_synthesis_agent._synthesise_without_rlm). Without
            # this wrap the call silently falls back to dspy.settings.lm — the
            # global runtime LM or none at all on the standalone endpoint —
            # ignoring the tenant's configured llm_config.
            if self._llm_config is not None:
                from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
                from cogniverse_foundation.config.gateway_routing import (
                    routed_lm_context_for,
                )

                tenant_id = getattr(self, "_memory_tenant_id", None) or SYSTEM_TENANT_ID
                with routed_lm_context_for(
                    self._config_manager,
                    tenant_id,
                    "knowledge_summarization_agent",
                    endpoint=self._llm_config,
                ):
                    result = self._dspy_module(title=title, memories=block)
            else:
                # No per-agent LM override — use ambient dspy.settings.lm.
                result = self._dspy_module(title=title, memories=block)
            return getattr(result, "summary", "") or ""
        except Exception as exc:
            logger.warning("ksum: DSPy synth failed (%s); falling back to raw", exc)
            # Fallback: return the block itself, prefixed — caller still
            # gets *some* artefact, never a silent empty.
            return f"[FALLBACK: synthesis failed] {block[:1000]}"

    async def _summarise_with_rlm(
        self, title: str, block: str, rlm_options: RLMOptions
    ) -> str:
        from cogniverse_agents.inference.rlm_inference import build_rlm_from_options

        rlm = build_rlm_from_options(
            self._llm_config,
            rlm_options,
            config_manager=getattr(self, "_config_manager", None),
            tenant_id=getattr(self, "_memory_tenant_id", None) or "",
        )
        result = rlm.process(
            query=f"Summarise the following knowledge under the title: {title}",
            context=block,
        )
        return result.answer

    def _promote_summary(
        self,
        *,
        tenant_id: str,
        title: str,
        summary: str,
        citation_refs: List[CitationRef],
        actor_role: str,
        actor_id: str,
    ) -> str:
        try:
            role = Pinnable(actor_role.lower())
        except ValueError as exc:
            raise SchemaViolationError(
                f"unknown actor_role={actor_role!r}: {exc}"
            ) from exc
        if role not in (Pinnable.TENANT_ADMIN, Pinnable.ORG_ADMIN):
            raise SchemaViolationError(
                f"actor_role={role.value} cannot promote summaries; "
                "tenant_admin or org_admin required"
            )

        federation = FederationService(self._mm_factory, self._registry)
        synthetic_memory = {
            "memory": summary,
            "agent_name": "_promoted",
            "metadata": {
                "kind": SUMMARY_KIND,
                "subject_key": f"summary:{title}",
                "title": title,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "derived_from": [
                    cr.to_dict() if hasattr(cr, "to_dict") else dict(cr.__dict__)
                    for cr in citation_refs
                ],
            },
        }
        result = federation.promote_to_org_trunk(
            source_tenant_id=tenant_id,
            source_memory=synthetic_memory,
            actor_role=role,
            actor_id=actor_id,
        )
        return result.promoted_memory_id
