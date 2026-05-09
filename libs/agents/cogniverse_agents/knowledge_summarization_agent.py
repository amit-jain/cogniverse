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

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.temporal_reasoning_agent import _parse_iso
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_core.memory.federation import FederationService
from cogniverse_core.memory.provenance import CitationRef
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    Pinnable,
    SchemaViolationError,
    build_default_registry,
)

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8026
SUMMARY_KIND = "knowledge_summary"


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


class KnowledgeSummarizationOutput(AgentOutput):
    title: str
    summary: str
    citation_refs: List[CitationRef] = Field(default_factory=list)
    source_count: int = 0
    used_rlm: bool = False
    promoted_to_org_trunk: bool = False
    promoted_memory_id: Optional[str] = None
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
        content = (r.get("memory") or r.get("content") or "").strip()
        meta = r.get("metadata") or {}
        subj = (meta.get("subject_key") if isinstance(meta, dict) else None) or "-"
        lines.append(f"=== Memory {i} (id={mid}, subject={subj}) ===\n{content}")
    return "\n\n".join(lines)


class KnowledgeSummarizationAgent(
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
        self._mm_factory = memory_manager_factory
        self._registry = registry or build_default_registry()
        self._llm_config = llm_config
        self._dspy_module = dspy.ChainOfThought(_SummarizationSignature)
        self._ensure_summary_kind_registered()

    def _ensure_summary_kind_registered(self) -> None:
        """Register `knowledge_summary` when the registry doesn't already know it.

        ``KnowledgeRegistry.get`` returns a safe-default schema for unknown
        kinds (and that default is ``tenant_private`` — which would block
        promotion). We must check ``is_registered`` and explicitly install
        an ``org_shared`` schema for summaries.
        """
        if self._registry.is_registered(SUMMARY_KIND):
            return
        from cogniverse_core.memory.schema import (
            ContradictionPolicy,
            KnowledgeSchema,
            Retention,
            Sensitivity,
        )

        self._registry.register(
            KnowledgeSchema(
                kind=SUMMARY_KIND,
                retention=Retention.PERMANENT,
                sensitivity=Sensitivity.ORG_SHARED,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=True,
                contradiction_policy=ContradictionPolicy.LATEST_WINS,
                default_trust=0.7,
            )
        )

    async def _process_impl(
        self, input: KnowledgeSummarizationInput
    ) -> KnowledgeSummarizationOutput:
        if self._mm_factory is None:
            from cogniverse_core.memory.manager import Mem0MemoryManager

            self._mm_factory = lambda tid: Mem0MemoryManager(tenant_id=tid)

        tenant_id = input.tenant_id or getattr(self.deps, "tenant_id", None)
        if not tenant_id:
            raise ValueError(
                "KnowledgeSummarizationAgent: no tenant_id on input or deps"
            )

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
                metadata={"reason": "no_matching_memories"},
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
            metadata={
                "subject_keys": input.subject_keys or [],
                "kinds": input.kinds or [],
                "since": input.since,
                "until": input.until,
                "actor_role": input.actor_role,
                "actor_id": input.actor_id,
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
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        llm_config = self._llm_config or LLMEndpointConfig(
            model=(
                f"{rlm_options.backend}/{rlm_options.model}"
                if rlm_options.model
                else f"{rlm_options.backend}/gpt-4o"
            )
        )
        rlm = RLMInference(
            llm_config=llm_config,
            max_iterations=rlm_options.max_iterations,
            max_llm_calls=rlm_options.max_llm_calls,
            timeout_seconds=rlm_options.timeout_seconds,
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
