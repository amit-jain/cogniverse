"""Unit tests for KnowledgeGraphTraversalAgent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.kg_traversal_agent import (
    KGTraversalDeps,
    KGTraversalInput,
    KnowledgeGraphTraversalAgent,
    _node_passes_mention_filter,
)
from cogniverse_core.agents.rlm_options import RLMOptions


def _node(
    mid: str, subject_key: str, label: str = "", excerpt: str = ""
) -> Dict[str, Any]:
    return {
        "id": mid,
        "memory": excerpt,
        "metadata": {
            "kind": "kg_node",
            "subject_key": subject_key,
            "label": label,
        },
    }


def _entity_fact(mid: str, subject_key: str, content: str = "") -> Dict[str, Any]:
    return {
        "id": mid,
        "memory": content,
        "metadata": {"kind": "entity_fact", "subject_key": subject_key},
    }


def _edge(mid: str, from_key: str, to_key: str, relation: str) -> Dict[str, Any]:
    return {
        "id": mid,
        "memory": f"{from_key} --{relation}--> {to_key}",
        "metadata": {
            "kind": "kg_edge",
            "from_subject_key": from_key,
            "to_subject_key": to_key,
            "relation": relation,
        },
    }


def _build_agent(snapshot: List[Dict[str, Any]]):
    """Wire a KGTraversalAgent against a fake memory manager that yields ``snapshot``."""
    agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id="acme"))
    fake_mm = MagicMock()
    fake_mm.memory = MagicMock()
    fake_mm.memory.get.side_effect = lambda mid: next(
        (m for m in snapshot if m["id"] == mid), None
    )
    fake_mm.get_all_memories = lambda *, tenant_id, agent_name: list(snapshot)
    agent.memory_manager = fake_mm
    agent._memory_tenant_id = "acme"
    agent._memory_agent_name = "kg_test"
    agent.is_memory_enabled = lambda: True  # type: ignore[assignment]
    return agent


@pytest.mark.asyncio
class TestSeedResolution:
    async def test_seed_via_subject_key_walks_graph(self):
        snapshot = [
            _node("n_a", "alice", label="Alice"),
            _node("n_b", "bob", label="Bob"),
            _node("n_c", "carol", label="Carol"),
            _edge("e_ab", "alice", "bob", "knows"),
            _edge("e_bc", "bob", "carol", "knows"),
        ]
        agent = _build_agent(snapshot)
        out = await agent._process_impl(
            KGTraversalInput(
                tenant_id="acme",
                start_subject_key="alice",
                max_depth=3,
            )
        )
        assert out.start_subject_key == "alice"
        # All three nodes reachable from alice within depth 3.
        keys = {n.subject_key for n in out.nodes}
        assert keys == {"alice", "bob", "carol"}
        # Two edges traversed.
        relations = {
            (e.from_subject_key, e.relation, e.to_subject_key) for e in out.edges
        }
        assert ("alice", "knows", "bob") in relations
        assert ("bob", "knows", "carol") in relations

    async def test_seed_via_memory_id(self):
        snapshot = [
            _node("n_alice", "alice"),
            _node("n_bob", "bob"),
            _edge("e_ab", "alice", "bob", "knows"),
        ]
        agent = _build_agent(snapshot)
        out = await agent._process_impl(
            KGTraversalInput(
                tenant_id="acme",
                start_memory_id="n_alice",
            )
        )
        assert out.start_subject_key == "alice"
        assert {n.subject_key for n in out.nodes} == {"alice", "bob"}

    async def test_no_seed_returns_empty_with_reason(self):
        agent = _build_agent([])
        out = await agent._process_impl(KGTraversalInput(tenant_id="acme"))
        assert out.metadata.get("reason") == "no_seed_resolved"
        assert out.nodes == []


@pytest.mark.asyncio
class TestDepthAndCapacity:
    async def test_max_depth_truncates_walk(self):
        snapshot = [
            _node("n1", "a"),
            _node("n2", "b"),
            _node("n3", "c"),
            _node("n4", "d"),
            _edge("e_ab", "a", "b", "k"),
            _edge("e_bc", "b", "c", "k"),
            _edge("e_cd", "c", "d", "k"),
        ]
        agent = _build_agent(snapshot)
        out = await agent._process_impl(
            KGTraversalInput(tenant_id="acme", start_subject_key="a", max_depth=1)
        )
        # Depth-1 walk visits a + b only; b->c stays unwalked.
        keys = {n.subject_key for n in out.nodes}
        assert keys == {"a", "b"}

    async def test_max_edges_truncates_walk(self):
        # Wide fan-out: a points to 10 leaves.
        snapshot = [_node("n_a", "a")]
        for i in range(10):
            snapshot.append(_node(f"n_{i}", f"leaf_{i}"))
            snapshot.append(_edge(f"e_{i}", "a", f"leaf_{i}", "has"))
        agent = _build_agent(snapshot)
        out = await agent._process_impl(
            KGTraversalInput(tenant_id="acme", start_subject_key="a", max_edges=5)
        )
        assert out.truncated is True
        assert len(out.edges) <= 5


@pytest.mark.asyncio
class TestRelationAllowlist:
    async def test_only_allowed_relations_followed(self):
        snapshot = [
            _node("n_a", "a"),
            _node("n_b", "b"),
            _node("n_c", "c"),
            _edge("e_knows", "a", "b", "knows"),
            _edge("e_owes", "a", "c", "owes"),
        ]
        agent = _build_agent(snapshot)
        out = await agent._process_impl(
            KGTraversalInput(
                tenant_id="acme",
                start_subject_key="a",
                relation_allowlist=["knows"],
            )
        )
        relations = {e.relation for e in out.edges}
        assert relations == {"knows"}
        # 'c' was reachable only via 'owes' which is not allowed.
        keys = {n.subject_key for n in out.nodes}
        assert "c" not in keys


@pytest.mark.asyncio
class TestEntityFactKindCounted:
    async def test_entity_fact_treated_as_node(self):
        snapshot = [
            _entity_fact("f_a", "a", content="Alice is a person"),
            _entity_fact("f_b", "b", content="Bob is a person"),
            _edge("e_ab", "a", "b", "knows"),
        ]
        agent = _build_agent(snapshot)
        out = await agent._process_impl(
            KGTraversalInput(tenant_id="acme", start_subject_key="a")
        )
        keys = {n.subject_key for n in out.nodes}
        assert keys == {"a", "b"}


@pytest.mark.asyncio
class TestRLMSummary:
    async def test_rlm_summariser_runs_when_enabled(self):
        snapshot = [
            _node("n_a", "a"),
            _node("n_b", "b"),
            _edge("e_ab", "a", "b", "knows"),
        ]
        agent = _build_agent(snapshot)
        # Stub the summariser so we don't need a live LM.
        called: dict = {}

        async def fake_summarise(query, block, options):
            called["query"] = query
            called["block_len"] = len(block)
            return "an llm summary of the graph"

        agent._summarise_with_rlm = fake_summarise  # type: ignore[assignment]

        out = await agent._process_impl(
            KGTraversalInput(
                tenant_id="acme",
                start_subject_key="a",
                rlm=RLMOptions(enabled=True),
            )
        )
        assert out.used_rlm is True
        assert out.summary == "an llm summary of the graph"
        assert called["query"]


@pytest.mark.asyncio
class TestNoMemoryManager:
    async def test_returns_empty_when_disabled(self):
        agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id="acme"))
        out = await agent._process_impl(
            KGTraversalInput(tenant_id="acme", start_subject_key="a")
        )
        assert out.metadata.get("reason") == "no_backend_available"


def test_input_validation_bounds():
    with pytest.raises(Exception):  # pydantic ValidationError
        KGTraversalInput(start_subject_key="a", max_depth=0)
    with pytest.raises(Exception):
        KGTraversalInput(start_subject_key="a", max_depth=99)
    with pytest.raises(Exception):
        KGTraversalInput(start_subject_key="a", max_edges=0)


def test_agent_capabilities_advertised():
    agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id="acme"))
    assert agent.agent_name == "kg_traversal_agent"
    assert "kg_traversal" in agent.capabilities
    assert agent.port == 8022


class TestMentionFilterCrashSafety:
    """A Mention blob stores ts_start/ts_end as arbitrary JSON — a real
    extractor can emit human-readable strings ("00:05"). The time-window
    filter must coerce them defensively, not crash on float()."""

    def test_non_numeric_mention_timestamp_does_not_crash(self):
        node_fields = {
            "mentions": json.dumps(
                [{"source_doc_id": "v1", "ts_start": "garbage", "ts_end": "bad"}]
            )
        }
        # garbage -> 0.0; the [50,100] window excludes a 0.0 point, so the
        # node is filtered out — and, crucially, no ValueError is raised.
        assert _node_passes_mention_filter(node_fields, "v1", (50.0, 100.0)) is False

    def test_numeric_string_mention_timestamp_overlaps(self):
        node_fields = {
            "mentions": json.dumps(
                [{"source_doc_id": "v1", "ts_start": "5", "ts_end": "9"}]
            )
        }
        assert _node_passes_mention_filter(node_fields, "v1", (0.0, 100.0)) is True


class TestEdgeFilterCrashSafety:
    """The traverse() time-window filter reads ts_start/ts_end off each edge
    dict. A non-numeric value must degrade to 0.0, not crash traverse()."""

    def test_non_numeric_edge_timestamp_does_not_crash(self):
        agent = KnowledgeGraphTraversalAgent(deps=KGTraversalDeps(tenant_id="acme"))
        fake_gm = SimpleNamespace(
            _visit_edges=lambda source_node_id: [
                {
                    "source_doc_id": "v1",
                    "ts_start": "garbage",
                    "ts_end": "bad",
                    "target_node_id": "t",
                    "relation": "knows",
                    "source_node_id": "s",
                }
            ],
            _visit=lambda doc_type, top_k: [],
        )
        agent.set_graph_manager(fake_gm)
        # ts_range forces the edge float() coercion path; garbage -> 0.0 is
        # excluded from [50,100] so the edge drops without a ValueError.
        out = agent.traverse("s", filters={"video_id": "v1", "ts_range": (50.0, 100.0)})
        assert out == {"nodes": [], "edges": []}
