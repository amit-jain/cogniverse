"""Knowledge agents in RLM mode — exercises ``_synthesise_with_rlm`` /
``_summarise_with_rlm`` against the configured test LM via Deno-backed
RLM REPL.

Endpoint, model, provider and api key all come from
``tests/fixtures/llm.py`` (driven by ``TEST_LLM_API_BASE`` /
``TEST_LLM_MODEL`` / ``TEST_LLM_PROVIDER`` / ``TEST_LLM_API_KEY``) so
the same test runs against any OpenAI-compatible provider without a
code change. Deno is installed by the ``ensure_deno`` session fixture
when missing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_core.agents.rlm_options import RLMOptions
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from tests.fixtures.llm import (
    is_test_lm_available,
    resolve_api_key,
    resolve_base_url,
    resolve_prefixed_model,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not is_test_lm_available(),
        reason=f"Test LM endpoint not reachable at {resolve_base_url()}",
    ),
]


def _llm_config() -> LLMEndpointConfig:
    return LLMEndpointConfig(
        model=resolve_prefixed_model(),
        api_base=resolve_base_url(),
        api_key=resolve_api_key(),
        max_tokens=600,
        temperature=0.1,
    )


_FACTS = [
    (
        "Standard refund window: customers may request a refund within "
        "30 days of purchase, no questions asked.",
        "30",
    ),
    (
        "European Union customers receive an additional 14 days of return "
        "rights under EU consumer protection law.",
        "14",
    ),
    (
        "Digital downloads are non-refundable once the file has been "
        "accessed; refund requests on accessed downloads are denied.",
        "digital",
    ),
]


@pytest.mark.usefixtures("ensure_deno")
class TestMultiDocSynthesisRLMMode:
    @pytest.mark.asyncio
    async def test_synthesises_across_docs_through_rlm(self):
        from cogniverse_agents.multi_document_synthesis_agent import (
            DocumentRef,
            MultiDocSynthesisDeps,
            MultiDocSynthesisInput,
            MultiDocumentSynthesisAgent,
        )

        rows_by_id = {
            f"doc_{i}": {"id": f"doc_{i}", "memory": text}
            for i, (text, _) in enumerate(_FACTS)
        }
        fake_mm = MagicMock()
        fake_mm.memory = MagicMock()
        fake_mm.memory.get = lambda mid: rows_by_id.get(mid)

        agent = MultiDocumentSynthesisAgent(
            deps=MultiDocSynthesisDeps(tenant_id="acme"),
            llm_config=_llm_config(),
        )
        agent.memory_manager = fake_mm
        agent._memory_initialized = True
        agent._memory_tenant_id = "acme"
        agent._memory_agent_name = "rlm_synth_test"

        out = await agent._process_impl(
            MultiDocSynthesisInput(
                tenant_id="acme",
                query=(
                    "Summarise the refund policy: standard window, EU "
                    "extension, and digital downloads."
                ),
                documents=[DocumentRef(memory_id=mid) for mid in rows_by_id],
                rlm=RLMOptions(auto_detect=True, context_threshold=50),
                persist=False,
            )
        )

        assert out.used_rlm is True, (
            f"expected RLM path; out.used_rlm={out.used_rlm}, answer={out.answer!r}"
        )
        answer = (out.answer or "").strip()
        assert answer, "RLM path returned empty answer"
        assert not answer.startswith("[FALLBACK:"), answer[:300]
        lower = answer.lower()
        assert "30 days" in lower, answer
        assert "14 days" in lower or "14 additional" in lower, answer
        assert "digital" in lower, answer


@pytest.mark.usefixtures("ensure_deno")
class TestKnowledgeSummarizationRLMMode:
    @pytest.mark.asyncio
    async def test_summarises_subject_slice_through_rlm(self):
        from cogniverse_agents.knowledge_summarization_agent import (
            KnowledgeSummarizationAgent,
            KnowledgeSummarizationDeps,
            KnowledgeSummarizationInput,
        )
        from cogniverse_core.memory.schema import build_default_registry

        rows = [
            {
                "id": f"k{i}",
                "memory": text,
                "metadata": {
                    "kind": "external_doc",
                    "subject_key": "policy:refunds",
                },
            }
            for i, (text, _) in enumerate(_FACTS)
        ]

        def factory(_tenant_id: str):
            mm = MagicMock()
            mm.memory = MagicMock()
            mm.get_all_memories = lambda *, tenant_id, agent_name: list(rows)
            return mm

        agent = KnowledgeSummarizationAgent(
            deps=KnowledgeSummarizationDeps(tenant_id="acme"),
            memory_manager_factory=factory,
            registry=build_default_registry(),
            llm_config=_llm_config(),
        )

        out = await agent._process_impl(
            KnowledgeSummarizationInput(
                tenant_id="acme",
                subject_keys=["policy:refunds"],
                kinds=["external_doc"],
                title="Refunds policy",
                actor_role="user",
                actor_id="alice",
                rlm=RLMOptions(auto_detect=True, context_threshold=50),
                promote=False,
            )
        )

        assert out.used_rlm is True, (
            f"expected RLM path; out.used_rlm={out.used_rlm}, summary={out.summary!r}"
        )
        summary = (out.summary or "").strip()
        assert summary, "RLM path returned empty summary"
        assert not summary.startswith("[FALLBACK:"), summary[:300]
        lower = summary.lower()
        assert "30 days" in lower, summary
        assert "14 days" in lower or "14 additional" in lower, summary
        assert "digital" in lower, summary
