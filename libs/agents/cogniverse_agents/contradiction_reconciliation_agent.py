"""ContradictionReconciliationAgent.

Consumes ConflictSets produced by :class:`ContradictionDetector` and
resolves them per the target schema's ``contradiction_policy``. V1 is
deterministic — it does not call out to an LLM. The plan reserves an RLM
trajectory for fetching extra evidence per side; that enrichment is left
as a follow-up.

Workflow:
  1. Caller supplies a list of memory ids that share a ``subject_key``
     (the conflict set members) plus the kind they belong to.
  2. Agent fetches each memory by id, applies ``reconcile`` from
     ``cogniverse_core.memory.contradiction``, and returns the resolved
     view: which member(s) survive and the rationale.

The agent is read-only — it does not mutate the underlying memories. A
follow-up commit can wire it to write ``conflict_set`` records (the
sentinel kind) when policy = ``preserve_both`` so audit history persists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import Field

from cogniverse_agents.graph.graph_schema import normalize_name
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.memory.contradiction import reconcile
from cogniverse_core.memory.schema import (
    ContradictionPolicy,
    KnowledgeRegistry,
    build_default_registry,
)

if TYPE_CHECKING:
    from cogniverse_agents.graph.graph_manager import GraphManager

logger = logging.getLogger(__name__)


def _policy_to_contract_string(policy: ContradictionPolicy) -> str:
    """Map the registry's enum to the consumer-contract string vocabulary."""
    mapping = {
        ContradictionPolicy.PRESERVE_BOTH: "PRESERVE_BOTH",
        ContradictionPolicy.LATEST_WINS: "LATEST_WINS",
        ContradictionPolicy.TRUST_RANKED: "TRUST_RANKED",
    }
    return mapping.get(policy, "PRESERVE_BOTH")


class ContradictionReconciliationInput(AgentInput):
    """Inputs for resolving one conflict set."""

    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    query: str = Field(
        "",
        description=(
            "Optional context label for telemetry. The actual reconciliation "
            "is driven by ``conflict_member_ids`` and ``target_kind``."
        ),
    )
    target_kind: str = Field(
        ...,
        description=(
            "Knowledge schema kind whose ``contradiction_policy`` decides "
            "the resolution (e.g. ``entity_fact``, ``kg_edge``)."
        ),
    )
    conflict_member_ids: List[str] = Field(
        ...,
        description="Memory ids belonging to a single conflict set.",
        min_length=2,
    )
    policy_override: Optional[str] = Field(
        None,
        description=(
            "Optional override of the schema's ``contradiction_policy`` for "
            "this resolution. One of: latest_wins / trust_ranked / preserve_both."
        ),
    )


class ResolvedMemberOut(AgentInput):
    """Serialisable view of a member after reconciliation."""

    memory_id: str
    survived: bool
    disputed: bool = False
    excerpt: str = ""


class ContradictionReconciliationOutput(AgentOutput):
    """Resolution outcome plus survivor list."""

    target_kind: str = Field(..., description="The schema kind reconciled")
    policy_used: str = Field(..., description="The policy applied")
    resolved: List[ResolvedMemberOut] = Field(
        default_factory=list,
        description="Per-member outcome (one entry per input id)",
    )
    survivors: List[str] = Field(
        default_factory=list,
        description="Memory ids that the policy retained as the canonical view",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Telemetry: ``input_count``, ``survivor_count``, "
            "``policy_overridden`` (bool)"
        ),
    )


class ContradictionReconciliationDeps(AgentDeps):
    """Tenant-agnostic deps; everything else is per-request."""

    pass


class ContradictionReconciliationAgent(
    MemoryAwareMixin,
    A2AAgent[
        ContradictionReconciliationInput,
        ContradictionReconciliationOutput,
        ContradictionReconciliationDeps,
    ],
):
    """A2A agent that resolves a ConflictSet via the schema's policy."""

    def __init__(
        self,
        deps: ContradictionReconciliationDeps,
        registry: Optional[KnowledgeRegistry] = None,
        config_manager=None,
        port: int = 8020,
    ):
        config = A2AAgentConfig(
            agent_name="contradiction_reconciliation_agent",
            agent_description=(
                "Resolves conflict sets by applying the schema's "
                "contradiction policy (latest_wins / trust_ranked / "
                "preserve_both) over the member memories."
            ),
            capabilities=[
                "contradiction_reconciliation",
                "audit",
            ],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        self._config_manager = config_manager
        self._registry = registry or build_default_registry()
        self._graph_manager: Optional["GraphManager"] = None

    def set_graph_manager(self, graph_manager: "GraphManager") -> None:
        """Bind a GraphManager so ``.detect`` can read Edge rows for a predicate."""
        self._graph_manager = graph_manager

    def detect(self, node_name: str, predicate: str) -> Dict[str, Dict[str, Any]]:
        """Group Edges on ``(node_name, predicate)`` across videos into a conflict set.

        Returns ``{"conflict_set": {"policy": <str>, "entries": [<entry>,...]}}``
        where ``policy`` is one of ``"PRESERVE_BOTH"|"LATEST_WINS"|"TRUST_RANKED"``
        (mapped from the schema's ``contradiction_policy`` for ``kg_edge``;
        falls back to ``"PRESERVE_BOTH"`` when more than one distinct
        ``(video, value)`` is observed). Each entry: ``{video_id,
        segment_id, ts_start, ts_end, value, confidence}``.
        """
        if self._graph_manager is None:
            raise RuntimeError(
                "ContradictionReconciliationAgent.detect requires a "
                "GraphManager — call set_graph_manager(...) first."
            )
        source_id = normalize_name(node_name)
        edges = self._graph_manager._visit_edges(source_node_id=source_id)
        entries: List[Dict[str, Any]] = []
        for edge_fields in edges:
            if str(edge_fields.get("relation") or "") != predicate:
                continue
            entries.append(
                {
                    "video_id": str(edge_fields.get("source_doc_id") or ""),
                    "segment_id": str(edge_fields.get("segment_id") or ""),
                    "ts_start": float(edge_fields.get("ts_start") or 0.0),
                    "ts_end": float(edge_fields.get("ts_end") or 0.0),
                    "value": str(edge_fields.get("target_node_id") or ""),
                    "confidence": float(edge_fields.get("confidence") or 0.0),
                }
            )
        entries.sort(key=lambda e: (e["video_id"], e["ts_start"]))

        # Map the schema's enum into the contract's string vocabulary.
        try:
            schema_policy = self._registry.get("kg_edge").contradiction_policy
            policy_str = _policy_to_contract_string(schema_policy)
        except Exception:
            policy_str = "PRESERVE_BOTH"

        return {"conflict_set": {"policy": policy_str, "entries": entries}}

    async def _process_impl(
        self, input: ContradictionReconciliationInput
    ) -> ContradictionReconciliationOutput:
        if not self.is_memory_enabled() or self.memory_manager is None:
            logger.warning(
                "ContradictionReconciliationAgent invoked without memory; "
                "returning empty resolution"
            )
            return ContradictionReconciliationOutput(
                target_kind=input.target_kind,
                policy_used=ContradictionPolicy.LATEST_WINS.value,
                resolved=[],
                survivors=[],
                metadata={"reason": "memory_manager_unavailable"},
            )

        # Resolve target policy: explicit override > schema lookup.
        schema = self._registry.get(input.target_kind)
        if input.policy_override is not None:
            try:
                policy = ContradictionPolicy(input.policy_override.lower())
            except ValueError as exc:
                raise ValueError(
                    f"unknown policy_override={input.policy_override!r}; "
                    f"valid: {[p.value for p in ContradictionPolicy]}"
                ) from exc
            policy_overridden = True
        else:
            policy = schema.contradiction_policy
            policy_overridden = False

        # Fetch each member by id. Missing members are skipped — the
        # detector's source-of-truth was a snapshot; some members may have
        # been deleted in the meantime. We record this in metadata so the
        # caller knows.
        members: List[Dict[str, Any]] = []
        missing: List[str] = []
        for mid in input.conflict_member_ids:
            try:
                memory = self.memory_manager.memory.get(mid)
            except Exception as exc:
                logger.debug(
                    "ContradictionReconciliationAgent: get(%s) failed: %s",
                    mid,
                    exc,
                )
                missing.append(mid)
                continue
            if isinstance(memory, dict):
                members.append(memory)
            elif isinstance(memory, list) and memory and isinstance(memory[0], dict):
                members.append(memory[0])
            else:
                missing.append(mid)

        if not members:
            return ContradictionReconciliationOutput(
                target_kind=input.target_kind,
                policy_used=policy.value,
                resolved=[],
                survivors=[],
                metadata={
                    "input_count": len(input.conflict_member_ids),
                    "missing_count": len(missing),
                    "missing": missing,
                    "policy_overridden": policy_overridden,
                },
            )

        resolved_view = reconcile(members, policy)
        survivor_ids = {m.get("id") for m in resolved_view if m.get("id") is not None}

        per_member: List[ResolvedMemberOut] = []
        for m in members:
            mid = m.get("id") or ""
            survived = mid in survivor_ids
            # PRESERVE_BOTH path tags survivors with metadata.disputed=True;
            # surface that on the typed output.
            disputed = False
            if survived:
                # Find the corresponding resolved entry (it carries the flag).
                for r in resolved_view:
                    if r.get("id") == mid:
                        rmeta = r.get("metadata") or {}
                        if isinstance(rmeta, dict):
                            disputed = bool(rmeta.get("disputed", False))
                        break
            content = m.get("memory") or m.get("content") or ""
            per_member.append(
                ResolvedMemberOut(
                    memory_id=str(mid),
                    survived=survived,
                    disputed=disputed,
                    excerpt=str(content)[:200],
                )
            )

        return ContradictionReconciliationOutput(
            target_kind=input.target_kind,
            policy_used=policy.value,
            resolved=per_member,
            survivors=sorted(survivor_ids),
            metadata={
                "input_count": len(input.conflict_member_ids),
                "fetched_count": len(members),
                "missing_count": len(missing),
                "missing": missing,
                "survivor_count": len(survivor_ids),
                "policy_overridden": policy_overridden,
            },
        )
