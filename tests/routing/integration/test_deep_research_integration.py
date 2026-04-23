"""
Integration test for DeepResearchAgent with real Ollama and Vespa.

Exercises the full research cycle: decompose → search → evaluate → synthesize.
Uses vespa_instance fixture from conftest to manage its own Docker container.
Requires Ollama (localhost:11434).
"""

import logging

import dspy
import pytest

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchDeps,
    DeepResearchInput,
    DeepResearchOutput,
)

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "ollama_chat/qwen3:4b"


def _ollama_available():
    import requests

    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _ollama_available(), reason="Ollama not running on :11434"),
]


@pytest.fixture(scope="module", autouse=True)
def configure_dspy():
    lm = dspy.LM(OLLAMA_MODEL, api_base=OLLAMA_URL, temperature=0.1, max_tokens=500)
    dspy.configure(lm=lm)
    yield
    dspy.configure(lm=None)


@pytest.fixture
def real_search_fn(vespa_instance):
    """Search function that hits the test Vespa Docker container.

    The vespa_instance fixture sets BACKEND_URL/BACKEND_PORT env vars,
    so create_default_config_manager() automatically connects to the
    test container.
    """

    async def search(query: str, tenant_id: str):
        from cogniverse_agents.search.service import SearchService
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
            get_config,
        )

        config_manager = create_default_config_manager()
        config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        schema_loader = FilesystemSchemaLoader("configs/schemas")

        service = SearchService(
            config=config,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        profile = config.get("default_profile", "video_colpali_smol500_mv_frame")
        results = service.search(
            query=query,
            profile=profile,
            tenant_id=tenant_id,
            top_k=5,
            ranking_strategy="float_float",
        )
        return [r.to_dict() for r in results]

    return search


class TestDeepResearchWithRealServices:
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_full_research_cycle(self, real_search_fn):
        """Decompose → search Vespa → evaluate → synthesize with real Ollama."""
        deps = DeepResearchDeps(tenant_id="test:unit")
        agent = DeepResearchAgent(deps=deps, search_fn=real_search_fn)

        result = await agent.process(
            DeepResearchInput(
                query="What visual content appears in outdoor scenes?",
                max_iterations=2,
                tenant_id="test:unit",
            )
        )

        assert isinstance(result, DeepResearchOutput)
        assert len(result.sub_questions) >= 2, (
            f"Decomposition should produce >=2 sub-questions, got {result.sub_questions}"
        )
        assert result.iterations_used >= 1
        assert len(result.evidence) >= 1, "Should collect evidence from search"
        assert len(result.summary) > 50, (
            f"Summary too short ({len(result.summary)} chars)"
        )
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_decomposition_produces_real_subquestions(self, real_search_fn):
        """DSPy decomposition with real Ollama produces meaningful sub-questions."""
        deps = DeepResearchDeps(tenant_id="test:unit")
        agent = DeepResearchAgent(deps=deps, search_fn=real_search_fn)

        sub_qs = await agent._decompose(
            "How do cooking tutorials differ from nature documentaries?"
        )

        assert len(sub_qs) >= 2, f"Expected >=2 sub-questions, got {sub_qs}"
        for q in sub_qs:
            assert len(q) > 10, f"Sub-question too short: '{q}'"
