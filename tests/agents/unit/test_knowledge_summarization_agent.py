"""Unit tests for KnowledgeSummarizationAgent."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.knowledge_summarization_agent import (
    SUMMARY_KIND,
    KnowledgeSummarizationAgent,
    KnowledgeSummarizationDeps,
    KnowledgeSummarizationInput,
    _matches_filters,
)
from cogniverse_agents.temporal_reasoning_agent import _parse_iso
from cogniverse_core.memory.schema import build_default_registry


def _row(
    mid: str,
    content: str,
    *,
    subject_key: str = "policy:refunds",
    kind: str = "external_doc",
    written_at: str | None = "2026-01-15T00:00:00Z",
    nested_provenance: bool = False,
):
    meta: Dict[str, Any] = {"kind": kind, "subject_key": subject_key}
    if written_at is not None:
        if nested_provenance:
            # The shape attach_to_metadata writes for real memories:
            # metadata["provenance"]["written_at"].
            meta["provenance"] = {"written_at": written_at}
        else:
            meta["written_at"] = written_at
    return {"id": mid, "memory": content, "metadata": meta}


def _factory_with_capture(
    rows_by_tenant: Dict[str, List[Dict[str, Any]]],
    promoted_ids: Dict[str, List[str]],
):
    """Factory that returns rows AND captures any add_memory calls per tenant."""

    def _factory(tenant_id: str):
        mm = MagicMock()
        mm.memory = MagicMock()
        rows = list(rows_by_tenant.get(tenant_id, []))
        mm.get_all_memories = lambda *, tenant_id=tenant_id, agent_name: list(rows)

        def _add(*, content, tenant_id, agent_name, metadata, infer):
            new_id = f"promoted::{tenant_id}::{len(promoted_ids[tenant_id])}"
            promoted_ids[tenant_id].append(new_id)
            return new_id

        mm.add_memory = _add
        promoted_ids.setdefault(tenant_id, [])
        return mm

    return _factory


def _build(rows_by_tenant: Dict[str, List[Dict[str, Any]]]):
    promoted: Dict[str, List[str]] = {}
    factory = _factory_with_capture(rows_by_tenant, promoted)
    agent = KnowledgeSummarizationAgent(
        deps=KnowledgeSummarizationDeps(tenant_id="acme:production"),
        memory_manager_factory=factory,
        registry=build_default_registry(),
    )
    # Stub the DSPy module so tests don't make LLM calls.
    agent._dspy_module = MagicMock(return_value=MagicMock(summary="STUB_SUMMARY"))
    return agent, promoted


class TestMatchesFilters:
    def test_subject_key_filter(self):
        r = _row("a", "x", subject_key="policy:refunds")
        assert _matches_filters(r, ["policy:refunds"], None, None, None) is True
        assert _matches_filters(r, ["policy:returns"], None, None, None) is False

    def test_kind_filter(self):
        r = _row("a", "x", kind="external_doc")
        assert _matches_filters(r, None, ["external_doc"], None, None) is True
        assert _matches_filters(r, None, ["learned_strategy"], None, None) is False

    def test_time_filter_inclusive_lower(self):
        r = _row("a", "x", written_at="2026-01-15T00:00:00Z")
        since = _parse_iso("2026-01-01T00:00:00Z")
        assert _matches_filters(r, None, None, since, None) is True

    def test_time_filter_exclusive_upper(self):
        r = _row("a", "x", written_at="2026-04-01T00:00:00Z")
        until = _parse_iso("2026-04-01T00:00:00Z")
        assert _matches_filters(r, None, None, None, until) is False

    def test_undated_excluded_when_time_filter_set(self):
        r = _row("a", "x", written_at=None)
        since = _parse_iso("2026-01-01T00:00:00Z")
        assert _matches_filters(r, None, None, since, None) is False

    def test_no_filters_passes_everything(self):
        r = _row("a", "x", written_at=None)
        assert _matches_filters(r, None, None, None, None) is True

    def test_time_filter_reads_nested_provenance_written_at(self):
        """attach_to_metadata nests written_at under metadata["provenance"];
        a time window must match those rows, not silently drop them."""
        r = _row("a", "x", written_at="2026-01-15T00:00:00Z", nested_provenance=True)
        since = _parse_iso("2026-01-01T00:00:00Z")
        until = _parse_iso("2026-02-01T00:00:00Z")
        assert _matches_filters(r, None, None, since, until) is True

    def test_nested_provenance_respects_window_bounds(self):
        r = _row("a", "x", written_at="2026-04-15T00:00:00Z", nested_provenance=True)
        until = _parse_iso("2026-04-01T00:00:00Z")
        assert _matches_filters(r, None, None, None, until) is False

    def test_top_level_written_at_wins_over_nested(self):
        """Promoted summaries write top-level written_at; when both shapes are
        present the top-level stamp is authoritative."""
        r = _row("a", "x", written_at="2026-01-15T00:00:00Z")
        r["metadata"]["provenance"] = {"written_at": "2020-01-01T00:00:00Z"}
        since = _parse_iso("2026-01-01T00:00:00Z")
        assert _matches_filters(r, None, None, since, None) is True


@pytest.mark.asyncio
class TestSummarizationFlow:
    async def test_no_matching_memories_returns_empty_summary(self):
        agent, _ = _build({"acme:production": []})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["nothing"],
                title="Refunds Q1",
                actor_role="user",
                actor_id="alice",
            )
        )
        assert out.summary == ""
        assert out.source_count == 0
        assert out.metadata["reason"] == "no_matching_memories"
        assert out.promoted_to_org_trunk is False

    async def test_summary_aggregates_matching_rows(self):
        rows = [
            _row("m1", "Refunds within 30 days"),
            _row("m2", "EU buyers: 14-day return window"),
        ]
        agent, _ = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["policy:refunds"],
                title="Refunds policy",
                actor_role="user",
                actor_id="alice",
            )
        )
        # LM is stubbed in this fixture; assertions verify the non-LM branch
        # (filter, count, citation assembly). Summary must be the stubbed LM
        # output, not just "any string".
        assert out.summary == "STUB_SUMMARY"
        assert out.source_count == 2
        assert {ref.ref_id for ref in out.citation_refs} == {"m1", "m2"}

    async def test_max_memories_caps_evidence(self):
        rows = [
            _row(f"m{i}", f"fact {i}", written_at="2026-01-15T00:00:00Z")
            for i in range(50)
        ]
        agent, _ = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["policy:refunds"],
                title="capped",
                actor_role="user",
                actor_id="alice",
                max_memories=5,
            )
        )
        assert out.source_count == 5
        assert len(out.citation_refs) == 5

    async def test_kind_filter_applied(self):
        rows = [
            _row("a", "right", kind="external_doc"),
            _row("b", "wrong", kind="learned_strategy"),
        ]
        agent, _ = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                kinds=["external_doc"],
                title="docs only",
                actor_role="user",
                actor_id="alice",
            )
        )
        ids = {ref.ref_id for ref in out.citation_refs}
        assert ids == {"a"}

    async def test_time_window_applied(self):
        rows = [
            _row("a", "in", written_at="2026-02-01T00:00:00Z"),
            _row("b", "out_early", written_at="2025-12-01T00:00:00Z"),
            _row(
                "c",
                "out_late",
                written_at="2026-05-01T00:00:00Z",
                nested_provenance=True,
            ),
        ]
        agent, _ = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                title="windowed",
                since="2026-01-01T00:00:00Z",
                until="2026-04-01T00:00:00Z",
                actor_role="user",
                actor_id="alice",
            )
        )
        ids = {ref.ref_id for ref in out.citation_refs}
        assert ids == {"a"}


@pytest.mark.asyncio
class TestPromotion:
    async def test_user_role_cannot_promote(self):
        rows = [_row("m1", "fact")]
        agent, promoted = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["policy:refunds"],
                title="should_not_promote",
                promote=True,
                actor_role="user",
                actor_id="alice",
            )
        )
        # Promotion attempt was made but refused — summary still returned.
        # Summary must be the stubbed LM output, not just "any string".
        assert out.summary == "STUB_SUMMARY"
        assert out.promoted_to_org_trunk is False
        # Org trunk got nothing.
        assert promoted.get("acme:_org_trunk", []) == []

    async def test_tenant_admin_promotes_into_org_trunk(self):
        rows = [_row("m1", "fact")]
        agent, promoted = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["policy:refunds"],
                title="ok_promote",
                promote=True,
                actor_role="tenant_admin",
                actor_id="tadm",
            )
        )
        assert out.promoted_to_org_trunk is True
        assert out.promoted_memory_id is not None
        assert promoted["acme:_org_trunk"] == [out.promoted_memory_id]

    async def test_failed_synthesis_not_promoted_to_org_trunk(self):
        """An admin allowed to promote must NOT promote a failed synthesis: the
        fallback text would otherwise pollute the shared org trunk that every
        tenant federates against."""
        rows = [_row("m1", "fact")]
        agent, promoted = _build({"acme:production": rows})
        # Synthesis fails — _summarise_without_rlm returns the fallback marker
        # and flags synthesis_ok=False.
        agent._dspy_module = MagicMock(side_effect=RuntimeError("LM down"))

        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["policy:refunds"],
                title="should_not_promote_fallback",
                promote=True,
                actor_role="tenant_admin",
                actor_id="tadm",
            )
        )

        assert out.summary.startswith("[FALLBACK: synthesis failed]")
        assert out.metadata["synthesis_ok"] is False
        assert out.promoted_to_org_trunk is False
        assert out.promoted_memory_id is None
        # The org trunk got nothing — no pollution.
        assert promoted.get("acme:_org_trunk", []) == []

    async def test_promote_false_skips_promotion_entirely(self):
        rows = [_row("m1", "fact")]
        agent, promoted = _build({"acme:production": rows})
        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme:production",
                subject_keys=["policy:refunds"],
                title="no_promote",
                promote=False,
                actor_role="tenant_admin",
                actor_id="tadm",
            )
        )
        assert out.promoted_to_org_trunk is False
        assert out.promoted_memory_id is None
        assert promoted.get("acme:_org_trunk", []) == []


def test_summary_kind_auto_registered():
    agent = KnowledgeSummarizationAgent(
        deps=KnowledgeSummarizationDeps(tenant_id="acme")
    )
    # If the agent didn't auto-register, .get() raises SchemaViolationError.
    schema = agent._registry.get(SUMMARY_KIND)
    assert schema.kind == SUMMARY_KIND


def test_input_requires_title():
    with pytest.raises(Exception):
        KnowledgeSummarizationInput(
            tenant_id="acme",
            subject_keys=["x"],
            title="",
            actor_id="a",
        )


def test_max_memories_bounds_enforced():
    with pytest.raises(Exception):
        KnowledgeSummarizationInput(
            tenant_id="acme",
            subject_keys=["x"],
            title="t",
            actor_id="a",
            max_memories=0,
        )
    with pytest.raises(Exception):
        KnowledgeSummarizationInput(
            tenant_id="acme",
            subject_keys=["x"],
            title="t",
            actor_id="a",
            max_memories=10_000,
        )


def test_agent_capabilities_advertised():
    agent = KnowledgeSummarizationAgent(
        deps=KnowledgeSummarizationDeps(tenant_id="acme")
    )
    assert agent.agent_name == "knowledge_summarization_agent"
    assert "knowledge_summarization" in agent.capabilities
    assert agent.port == 8026


@pytest.mark.asyncio
class TestMemoryOutage:
    """A memory-backend outage is not "no memories" — it must surface, never
    become a confident empty summary."""

    async def test_get_all_memories_outage_propagates(self):
        def _factory(tenant_id):
            mm = MagicMock()
            mm.memory = MagicMock()

            def _raise(**kwargs):
                raise ConnectionError("vespa down")

            mm.get_all_memories = _raise
            return mm

        agent = KnowledgeSummarizationAgent(
            deps=KnowledgeSummarizationDeps(tenant_id="acme:production"),
            memory_manager_factory=_factory,
            registry=build_default_registry(),
        )
        agent._dspy_module = MagicMock(return_value=MagicMock(summary="STUB"))
        with pytest.raises(ConnectionError, match="vespa down"):
            await agent._process_impl(
                KnowledgeSummarizationInput(
                    tenant_id="acme:production",
                    subject_keys=["policy:refunds"],
                    title="Refunds Q1",
                    actor_role="user",
                    actor_id="alice",
                )
            )

    async def test_factory_outage_propagates(self):
        def _factory(tenant_id):
            raise ConnectionError("mem0 init failed")

        agent = KnowledgeSummarizationAgent(
            deps=KnowledgeSummarizationDeps(tenant_id="acme:production"),
            memory_manager_factory=_factory,
            registry=build_default_registry(),
        )
        agent._dspy_module = MagicMock(return_value=MagicMock(summary="STUB"))
        with pytest.raises(ConnectionError, match="mem0 init failed"):
            await agent._process_impl(
                KnowledgeSummarizationInput(
                    tenant_id="acme:production",
                    subject_keys=["policy:refunds"],
                    title="Refunds Q1",
                    actor_role="user",
                    actor_id="alice",
                )
            )
