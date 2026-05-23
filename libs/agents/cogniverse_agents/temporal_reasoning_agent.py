"""TemporalReasoningAgent.

Compares knowledge about one subject across multiple time windows.
Useful for "what did we know about X in Q2 vs Q4" or
"how has our refund policy evolved over the last year." Read-only.

The agent does not require a Vespa-side time-versioned read path: time
windows are applied client-side over the ``written_at`` field that the
provenance layer attaches to every memory's metadata. Memories without a
parseable ``written_at`` are placed in an ``"undated"`` bucket so the
caller can decide what to do with them.

Per-window content signatures (a stable hash of the matching memory
content) make it trivial to detect "did anything change between
windows?" without dragging the LLM in for trivial cases. RLM-capable
when the per-window content set is large.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator

from cogniverse_agents.graph.graph_schema import normalize_name
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.agents.rlm_options import RLMOptions

if TYPE_CHECKING:
    from cogniverse_agents.graph.graph_manager import GraphManager

logger = logging.getLogger(__name__)


_DEFAULT_PORT = 8025
_UNDATED_BUCKET = "undated"


def _parse_iso(value: Any) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. Returns None for malformed input."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        out = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out


def _content_signature(rows: List[Dict[str, Any]]) -> str:
    """Stable hash of the sorted contents — for cheap window-vs-window diff."""
    contents = sorted((r.get("memory", "")).strip() for r in rows)
    h = hashlib.sha256()
    for c in contents:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _read_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive metadata reader — handles dict / JSON-string / None."""
    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _matches_subject(memory: Dict[str, Any], subject_key: str) -> bool:
    meta = _read_metadata(memory)
    return meta.get("subject_key") == subject_key


class TimeWindow(AgentInput):
    """Half-open `[start, end)` window. End may be omitted for "open-ended"."""

    label: str = Field(..., description="Caller label, e.g. ``Q2_2026``")
    start: str = Field(..., description="ISO-8601 inclusive lower bound")
    end: Optional[str] = Field(
        None,
        description=(
            "ISO-8601 exclusive upper bound. None = open-ended (matches all "
            "memories at-or-after ``start``)."
        ),
    )

    @field_validator("start")
    @classmethod
    def _start_is_iso(cls, v: str) -> str:
        if _parse_iso(v) is None:
            raise ValueError(f"start={v!r} is not a parseable ISO-8601 timestamp")
        return v

    @field_validator("end")
    @classmethod
    def _end_is_iso_or_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None or _parse_iso(v) is not None:
            return v
        raise ValueError(f"end={v!r} is not a parseable ISO-8601 timestamp")


class TemporalReasoningInput(AgentInput):
    tenant_id: Optional[str] = Field(None)
    subject_key: str = Field(
        ...,
        min_length=1,
        description="Canonical subject identifier (e.g. ``policy:refunds``)",
    )
    windows: List[TimeWindow] = Field(
        ...,
        min_length=2,
        description="At least two windows — comparison is the point of the agent",
    )
    agent_name_filter: Optional[str] = Field(
        None,
        description="Restricts the read to one agent_name namespace",
    )
    rlm: Optional[RLMOptions] = Field(None)


class WindowViewOut(AgentInput):
    label: str
    start: str
    end: Optional[str]
    matching_memory_ids: List[str]
    content_signature: str
    excerpts: List[str]


class TemporalReasoningOutput(AgentOutput):
    subject_key: str
    window_views: List[WindowViewOut]
    distinct_signatures_count: int = Field(
        ...,
        description=(
            "Count of distinct content signatures across windows; 1 = "
            "knowledge unchanged, >1 = knowledge evolved."
        ),
    )
    undated_count: int = Field(
        0,
        description="Memories on the subject lacking a parseable written_at",
    )
    summary: Optional[str] = Field(None, description="Set when RLM ran")
    used_rlm: bool = Field(False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemporalReasoningDeps(AgentDeps):
    pass


class TemporalReasoningAgent(
    MemoryAwareMixin,
    A2AAgent[TemporalReasoningInput, TemporalReasoningOutput, TemporalReasoningDeps],
):
    """A2A agent that compares a subject's knowledge across time windows."""

    def __init__(
        self,
        deps: TemporalReasoningDeps,
        memory_manager_factory=None,
        llm_config=None,
        port: int = _DEFAULT_PORT,
    ):
        config = A2AAgentConfig(
            agent_name="temporal_reasoning_agent",
            agent_description=(
                "Compares knowledge about a subject across explicit time "
                "windows using provenance.written_at; surfaces signature "
                "deltas and an optional RLM-summarised narrative."
            ),
            capabilities=["temporal_reasoning", "audit"],
            port=port,
        )
        super().__init__(deps=deps, config=config)
        from cogniverse_agents._mm_factory import make_mm_factory

        self._mm_factory = make_mm_factory(memory_manager_factory)
        self._llm_config = llm_config
        self._graph_manager: Optional["GraphManager"] = None

    def set_graph_manager(self, graph_manager: "GraphManager") -> None:
        """Bind a GraphManager so ``.compare_over_time`` can read Edges."""
        self._graph_manager = graph_manager

    def compare_over_time(
        self, node_name: str, videos: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build a chronological timeline of claims about ``node_name``.

        Reads outgoing Edges whose ``source_node_id == normalize_name(node_name)``
        and whose ``source_doc_id`` is in ``videos``. Returns
        ``{"timeline": [<entry>, ...]}`` ordered by ``ts_start`` ascending.
        Each entry: ``{ts_start, ts_end, video_id, segment_id, claim,
        evidence_span}`` where ``claim`` is ``"<relation>:<target_node_id>"``.
        """
        if self._graph_manager is None:
            raise RuntimeError(
                "TemporalReasoningAgent.compare_over_time requires a "
                "GraphManager — call set_graph_manager(...) first."
            )
        source_id = normalize_name(node_name)
        all_edges = self._graph_manager._visit_edges(source_node_id=source_id)
        video_set = {str(v) for v in videos}

        timeline: List[Dict[str, Any]] = []
        for edge_fields in all_edges:
            video_id = str(edge_fields.get("source_doc_id") or "")
            if video_id not in video_set:
                continue
            relation = str(edge_fields.get("relation") or "")
            target = str(edge_fields.get("target_node_id") or "")
            if not relation or not target:
                continue
            timeline.append(
                {
                    "ts_start": float(edge_fields.get("ts_start") or 0.0),
                    "ts_end": float(edge_fields.get("ts_end") or 0.0),
                    "video_id": video_id,
                    "segment_id": str(edge_fields.get("segment_id") or ""),
                    "claim": f"{relation}:{target}",
                    "evidence_span": str(edge_fields.get("evidence_span") or ""),
                }
            )
        timeline.sort(key=lambda e: (e["ts_start"], e["video_id"], e["segment_id"]))
        return {"timeline": timeline}

    async def _process_impl(
        self, input: TemporalReasoningInput
    ) -> TemporalReasoningOutput:
        from cogniverse_agents._mm_factory import require_tenant_id

        tenant_id = require_tenant_id(input, self.deps, "TemporalReasoningAgent")

        agent_name = input.agent_name_filter or "_promoted"
        rows = self._fetch_subject_rows(tenant_id, agent_name, input.subject_key)

        # Bucket per window; track undated separately.
        bucketed, undated = self._bucket_rows_by_window(rows, input.windows)

        window_views: List[WindowViewOut] = []
        signatures: List[str] = []
        for w in input.windows:
            members = bucketed.get(w.label, [])
            sig = _content_signature(members)
            signatures.append(sig)
            window_views.append(
                WindowViewOut(
                    label=w.label,
                    start=w.start,
                    end=w.end,
                    matching_memory_ids=[str(r.get("id") or "") for r in members],
                    content_signature=sig,
                    excerpts=[str(r.get("memory", ""))[:200] for r in members],
                )
            )

        distinct = len(set(signatures))

        used_rlm = False
        summary: Optional[str] = None
        rlm_options = input.rlm
        if rlm_options is not None and any(v.matching_memory_ids for v in window_views):
            block = self._format_for_summary(input.subject_key, window_views)
            if rlm_options.should_use_rlm(len(block)):
                summary = await self._summarise_with_rlm(
                    input.subject_key, block, rlm_options
                )
                used_rlm = True

        return TemporalReasoningOutput(
            subject_key=input.subject_key,
            window_views=window_views,
            distinct_signatures_count=distinct,
            undated_count=len(undated),
            summary=summary,
            used_rlm=used_rlm,
            metadata={
                "windows_compared": len(input.windows),
                "total_subject_memories": len(rows),
                "agent_name_filter": agent_name,
            },
        )

    def _fetch_subject_rows(
        self, tenant_id: str, agent_name: str, subject_key: str
    ) -> List[Dict[str, Any]]:
        try:
            mm = self._mm_factory(tenant_id)
        except Exception as exc:
            logger.debug("temporal: factory(%s) failed: %s", tenant_id, exc)
            return []
        if mm is None or not getattr(mm, "memory", None):
            return []
        try:
            rows = list(mm.get_all_memories(tenant_id=tenant_id, agent_name=agent_name))
        except Exception as exc:
            logger.debug("temporal: get_all_memories failed: %s", exc)
            return []
        return [r for r in rows if _matches_subject(r, subject_key)]

    @staticmethod
    def _bucket_rows_by_window(
        rows: List[Dict[str, Any]],
        windows: List["TimeWindow"],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        bucketed: Dict[str, List[Dict[str, Any]]] = {w.label: [] for w in windows}
        undated: List[Dict[str, Any]] = []
        parsed_windows: List[Tuple[str, datetime, Optional[datetime]]] = []
        for w in windows:
            start = _parse_iso(w.start)
            end = _parse_iso(w.end) if w.end is not None else None
            assert start is not None  # field_validator guaranteed this
            parsed_windows.append((w.label, start, end))

        for r in rows:
            meta = _read_metadata(r)
            # attach_to_metadata stores provenance under
            # metadata["provenance"]["written_at"]; legacy callers may
            # also stamp it at the top level.
            written_at: Any = None
            prov = meta.get("provenance")
            if isinstance(prov, dict):
                written_at = prov.get("written_at")
            if not written_at:
                written_at = meta.get("written_at")
            ts = _parse_iso(written_at)
            if ts is None:
                undated.append(r)
                continue
            for label, start, end in parsed_windows:
                if ts < start:
                    continue
                if end is not None and ts >= end:
                    continue
                bucketed[label].append(r)
        return bucketed, undated

    @staticmethod
    def _format_for_summary(subject_key: str, views: List[WindowViewOut]) -> str:
        lines = [f"Subject: {subject_key}"]
        for v in views:
            header = f"--- window={v.label} [{v.start} → {v.end or 'open'}]"
            lines.append(header)
            if not v.matching_memory_ids:
                lines.append("(no memories in this window)")
                continue
            for mid, ex in zip(v.matching_memory_ids, v.excerpts):
                lines.append(f"  {mid}: {ex}")
        return "\n".join(lines)

    async def _summarise_with_rlm(
        self,
        subject_key: str,
        block: str,
        rlm_options: RLMOptions,
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
            query=(
                f"How has the knowledge about {subject_key} evolved across "
                "the time windows below? Highlight what changed and when."
            ),
            context=block,
        )
        return result.answer
