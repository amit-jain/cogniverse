"""Unit tests for MultiDocumentSynthesisAgent (C3.1)."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.multi_document_synthesis_agent import (
    DocumentRef,
    MultiDocSynthesisDeps,
    MultiDocSynthesisInput,
    MultiDocumentSynthesisAgent,
    _format_documents_for_prompt,
)
from cogniverse_core.agents.rlm_options import RLMOptions


def _build_agent_with_stub_synth(answer: str = "synthesised answer"):
    """Construct an agent and stub the LLM path to return ``answer``."""
    agent = MultiDocumentSynthesisAgent(deps=MultiDocSynthesisDeps(tenant_id="acme"))
    agent._synthesise_without_rlm = lambda query, documents_block: answer  # type: ignore[assignment]

    async def fake_rlm(query, documents_block, options):
        return f"rlm:{answer}"

    agent._synthesise_with_rlm = fake_rlm  # type: ignore[assignment]
    # Disable persistence for clean unit tests; integration tests cover persist.
    agent.is_memory_enabled = lambda: False  # type: ignore[assignment]
    return agent


class TestPromptFormatting:
    def test_format_numbers_and_labels(self):
        refs = [
            DocumentRef(memory_id="m1", label="alpha"),
            DocumentRef(content="raw content", label=None),
        ]
        contents = ["doc one body", "doc two body"]
        block = _format_documents_for_prompt(refs, contents)
        assert "Document 1" in block
        assert "Document 2" in block
        assert "alpha" in block
        assert "doc one body" in block
        assert "doc two body" in block


@pytest.mark.asyncio
class TestSynthesisPath:
    async def test_inline_documents_synthesise_without_rlm(self):
        agent = _build_agent_with_stub_synth("the answer")
        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what is the answer?",
                documents=[
                    DocumentRef(content="doc 1 body", label="a"),
                    DocumentRef(content="doc 2 body", label="b"),
                ],
                persist=False,
            )
        )
        assert out.answer == "the answer"
        assert out.used_rlm is False
        assert out.persisted_memory_id is None
        # Citations recorded with kind=url (label-based) since no memory_id.
        kinds = {r["ref_kind"] for r in out.citation_refs}
        assert "url" in kinds

    async def test_memory_id_documents_resolved_via_manager(self):
        agent = _build_agent_with_stub_synth("answer")
        agent.is_memory_enabled = lambda: True  # type: ignore[assignment]
        fake_mm = MagicMock()
        fake_mm.memory = MagicMock()
        fake_mm.memory.get.side_effect = lambda mid: {
            "id": mid,
            "memory": f"content for {mid}",
            "metadata": {"kind": "external_doc"},
        }
        agent.memory_manager = fake_mm

        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[
                    DocumentRef(memory_id="m_alpha", label="alpha"),
                    DocumentRef(memory_id="m_beta", label="beta"),
                ],
                persist=False,
            )
        )
        assert out.metadata["document_count"] == 2
        # Each document becomes a memory citation.
        kinds = [r["ref_kind"] for r in out.citation_refs]
        assert kinds == ["memory", "memory"]
        ids = [r["ref_id"] for r in out.citation_refs]
        assert sorted(ids) == ["m_alpha", "m_beta"]

    async def test_unresolvable_document_skipped(self):
        agent = _build_agent_with_stub_synth("answer")
        agent.is_memory_enabled = lambda: True  # type: ignore[assignment]
        fake_mm = MagicMock()
        fake_mm.memory = MagicMock()
        # m_unknown returns None
        fake_mm.memory.get.side_effect = lambda mid: (
            {"id": mid, "memory": "x", "metadata": {}} if mid == "m_known" else None
        )
        agent.memory_manager = fake_mm

        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[
                    DocumentRef(memory_id="m_known"),
                    DocumentRef(memory_id="m_unknown"),
                ],
                persist=False,
            )
        )
        assert out.metadata["document_count"] == 1

    async def test_no_resolvable_documents_returns_empty(self):
        agent = _build_agent_with_stub_synth("answer")
        agent.is_memory_enabled = lambda: True  # type: ignore[assignment]
        fake_mm = MagicMock()
        fake_mm.memory = MagicMock()
        fake_mm.memory.get.side_effect = lambda mid: None
        agent.memory_manager = fake_mm

        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[DocumentRef(memory_id="m1"), DocumentRef(memory_id="m2")],
                persist=False,
            )
        )
        assert out.answer == ""
        assert out.metadata.get("reason") == "no_resolvable_documents"


@pytest.mark.asyncio
class TestRLMRouting:
    async def test_rlm_enabled_routes_through_rlm_path(self):
        agent = _build_agent_with_stub_synth("base")
        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[DocumentRef(content="x" * 100, label="big")],
                persist=False,
                rlm=RLMOptions(enabled=True, max_iterations=2),
            )
        )
        assert out.used_rlm is True
        assert out.answer.startswith("rlm:")

    async def test_rlm_auto_detect_below_threshold_no_rlm(self):
        agent = _build_agent_with_stub_synth("base")
        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[DocumentRef(content="x" * 100)],
                persist=False,
                rlm=RLMOptions(auto_detect=True, context_threshold=10_000),
            )
        )
        assert out.used_rlm is False

    async def test_rlm_auto_detect_above_threshold_uses_rlm(self):
        agent = _build_agent_with_stub_synth("base")
        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[DocumentRef(content="x" * 100)],
                persist=False,
                rlm=RLMOptions(auto_detect=True, context_threshold=50),
            )
        )
        assert out.used_rlm is True


@pytest.mark.asyncio
class TestPersistence:
    async def test_persist_writes_memory_with_synthesis_provenance(self):
        agent = _build_agent_with_stub_synth("synth")
        # Wire a stub memory manager that returns an id from add_memory.
        agent.is_memory_enabled = lambda: True  # type: ignore[assignment]
        captured: Dict[str, Any] = {}

        def fake_add(*, content, tenant_id, agent_name, metadata=None, infer=False):
            captured["content"] = content
            captured["agent_name"] = agent_name
            captured["metadata"] = metadata
            return "m_synth_1"

        fake_mm = MagicMock()
        fake_mm.memory = MagicMock()
        # _resolve_document calls memory.get(mid); return a real dict for m_a
        # so the memory_id path resolves and ends up in derived_from.
        fake_mm.memory.get.side_effect = lambda mid: (
            {"id": mid, "memory": "m_a body content", "metadata": {}}
            if mid == "m_a"
            else None
        )
        fake_mm.add_memory = fake_add
        agent.memory_manager = fake_mm

        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query="what?",
                documents=[
                    DocumentRef(memory_id="m_a", label="alpha"),
                    DocumentRef(content="inline doc", label="beta"),
                ],
                persist=True,
            )
        )

        assert out.persisted_memory_id == "m_synth_1"
        # Memory was tagged with synthesis_fact + provenance.derivation_kind=synthesis.
        meta = captured["metadata"]
        assert meta["kind"] == "synthesis_fact"
        assert meta["provenance"]["derivation_kind"] == "synthesis"
        # derived_from references the inputs.
        derived = meta["provenance"]["derived_from"]
        assert any(r["ref_id"] == "m_a" for r in derived)


def test_input_validation_requires_at_least_one_document():
    with pytest.raises(Exception):  # pydantic ValidationError
        MultiDocSynthesisInput(
            tenant_id="acme",
            query="x",
            documents=[],
        )


def test_agent_capabilities_advertised():
    agent = MultiDocumentSynthesisAgent(deps=MultiDocSynthesisDeps(tenant_id="acme"))
    assert agent.agent_name == "multi_document_synthesis_agent"
    assert "multi_document_synthesis" in agent.capabilities
    assert agent.port == 8021
