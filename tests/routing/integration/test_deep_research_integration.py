"""
Integration test for DeepResearchAgent against the configured LM and a real
Vespa instance.

Exercises the full research cycle: decompose → search → evaluate → synthesize.
Uses vespa_instance fixture from conftest to manage its own Docker container.
Requires the configured test LM endpoint to be reachable (see
``tests/fixtures/llm.py``).
"""

import logging
import time

import dspy
import pytest

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchDeps,
    DeepResearchInput,
    DeepResearchOutput,
)
from tests.fixtures.llm import (
    is_test_lm_available,
    resolve_api_key,
    resolve_base_url,
    resolve_prefixed_model,
)

logger = logging.getLogger(__name__)


pytestmark = [
    pytest.mark.skipif(
        not is_test_lm_available(),
        reason=f"Test LM not reachable at {resolve_base_url()}",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def configure_dspy():
    lm = dspy.LM(
        resolve_prefixed_model(),
        api_base=resolve_base_url(),
        api_key=resolve_api_key(),
        temperature=0.1,
        max_tokens=500,
    )
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


@pytest.fixture
def seeded_outdoor_corpus(vespa_instance):
    """Deep research can only find evidence that exists — feed segments
    whose descriptions cover outdoor scenes so the search step returns
    real passages instead of an empty index."""
    import requests

    from tests.utils.vespa_test_helpers import schema_full_name

    schema = schema_full_name("video_colpali_smol500_mv_frame", "test_unit")
    port = vespa_instance["http_port"]

    # The doc type takes a few seconds to converge after deploy_schema
    # activates — poll GET liveness (404 = type known, doc absent) before
    # feeding, or the first POST 400s with "Document type does not exist".
    probe_url = (
        f"http://localhost:{port}/document/v1/video/{schema}/docid/liveness_probe"
    )
    for _ in range(120):
        if requests.get(probe_url, timeout=5).status_code in (200, 404):
            break
        time.sleep(1)
    else:
        pytest.fail(f"schema {schema} doc type never went live")

    segments = [
        (
            "outdoor_forest",
            "A hiker walks through a dense green forest with tall pine "
            "trees, ferns, and sunlight filtering through the canopy.",
        ),
        (
            "outdoor_park",
            "Children play on swings in an urban park with grass lawns, "
            "oak trees and a fountain on a sunny afternoon.",
        ),
        (
            "outdoor_mountain",
            "A wide mountain landscape with snow-capped peaks, a clear "
            "blue sky and a rocky hiking trail across a meadow.",
        ),
    ]
    for video_id, description in segments:
        fields = {
            "video_id": video_id,
            "video_title": video_id.replace("_", " "),
            "source_url": f"file:///{video_id}.mp4",
            "creation_timestamp": 1700000000,
            "segment_id": 0,
            "start_time": 0.0,
            "end_time": 1.0,
            "segment_description": description,
            "audio_transcript": description,
            "embedding": {"blocks": {"0": [0.1] * 128}},
            "embedding_binary": {"blocks": {"0": [1] * 16}},
        }
        resp = requests.post(
            f"http://localhost:{port}/document/v1/video/{schema}"
            f"/docid/{video_id}_seg_0",
            json={"fields": fields},
            timeout=15,
        )
        assert resp.status_code in (200, 201), resp.text[:300]
    time.sleep(2)
    return [v for v, _ in segments]


class TestDeepResearchWithRealServices:
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_full_research_cycle(self, real_search_fn, seeded_outdoor_corpus):
        """Decompose → search Vespa → evaluate → synthesize against the configured LM."""
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
        """DSPy decomposition against the configured LM produces meaningful sub-questions."""
        deps = DeepResearchDeps(tenant_id="test:unit")
        agent = DeepResearchAgent(deps=deps, search_fn=real_search_fn)

        sub_qs = await agent._decompose(
            "How do cooking tutorials differ from nature documentaries?"
        )

        assert len(sub_qs) >= 2, f"Expected >=2 sub-questions, got {sub_qs}"
        for q in sub_qs:
            assert len(q) > 10, f"Sub-question too short: '{q}'"
