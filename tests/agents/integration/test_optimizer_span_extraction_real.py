"""Real agent -> span -> optimizer training pair, per operation.

The query_enhancement and entity_extraction optimizers build their training
sets from the spans their agents emit. Each test drives the REAL agent (only
the LM call is mocked — it is not what's under test), lets it emit its real
span into real Phoenix, then asserts the optimizer's extraction helper reads
back the exact training pair.

This is what the old code could not do: the agents wrote flat
``attributes.query_enhancement.*`` keys while the optimizers read a nested
``attributes.query_enhancement`` dict, so every optimizer built zero examples
and returned ``no_data`` even on real traffic.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.span_contract import read_span_io
from cogniverse_runtime.optimization_cli import (
    _entity_extraction_pairs,
    _profile_selection_pairs,
    _query_enhancement_pairs,
)

pytestmark = pytest.mark.integration


async def _fetch_named_spans(real_telemetry, tenant_id, span_name):
    canonical = canonical_tenant_id(tenant_id)
    project = real_telemetry.config.get_project_name(canonical)
    provider = real_telemetry.get_provider(tenant_id=canonical, project_name=project)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        now = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=1000,
        )
        if spans is not None and not spans.empty and "name" in spans.columns:
            hit = spans[spans["name"] == span_name]
            if not hit.empty:
                return hit
        await asyncio.sleep(2)
    return None


@pytest.mark.asyncio
async def test_query_enhancement_span_yields_simba_training_pair(real_telemetry):
    from cogniverse_agents.query_enhancement_agent import (
        QueryEnhancementAgent,
        QueryEnhancementDeps,
        QueryEnhancementInput,
    )

    tenant_id = "qe-opt-real"
    agent = QueryEnhancementAgent(deps=QueryEnhancementDeps(), port=19112)
    agent.set_telemetry_manager(real_telemetry)

    mock_result = MagicMock()
    mock_result.enhanced_query = "machine learning tutorials and step by step guides"
    mock_result.expansion_terms = "deep learning, neural networks"
    mock_result.synonyms = "ML, AI"
    mock_result.context = "education"
    mock_result.confidence = "0.85"
    mock_result.reasoning = "Added related terms"

    with patch.object(agent, "call_dspy", return_value=mock_result):
        await agent.process(
            QueryEnhancementInput(query="ML tutorials", tenant_id=tenant_id)
        )

    spans = await _fetch_named_spans(
        real_telemetry, tenant_id, "cogniverse.query_enhancement"
    )
    assert spans is not None, "cogniverse.query_enhancement span not indexed"

    pairs = _query_enhancement_pairs(spans)
    assert len(pairs) == 1
    assert pairs[0]["query"] == "ML tutorials"
    assert (
        pairs[0]["enhanced_query"]
        == "machine learning tutorials and step by step guides"
    )
    assert pairs[0]["confidence"] == 0.85


@pytest.mark.asyncio
async def test_entity_extraction_span_yields_training_pair(real_telemetry):
    from cogniverse_agents.entity_extraction_agent import (
        EntityExtractionAgent,
        EntityExtractionDeps,
        EntityExtractionInput,
    )

    tenant_id = "ee-opt-real"
    with patch.object(EntityExtractionAgent, "_initialize_extractors"):
        agent = EntityExtractionAgent(deps=EntityExtractionDeps(), port=19110)
    agent._gliner_extractor = None
    agent._spacy_analyzer = None
    agent.set_telemetry_manager(real_telemetry)

    mock_prediction = MagicMock()
    mock_prediction.entities = "machine learning|CONCEPT|0.9"
    mock_prediction.entity_types = "CONCEPT"

    with patch.object(agent, "call_dspy", return_value=mock_prediction):
        await agent.process(
            EntityExtractionInput(
                query="machine learning tutorials", tenant_id=tenant_id
            )
        )

    spans = await _fetch_named_spans(
        real_telemetry, tenant_id, "cogniverse.entity_extraction"
    )
    assert spans is not None, "cogniverse.entity_extraction span not indexed"

    pairs = _entity_extraction_pairs(spans)
    assert len(pairs) == 1
    assert pairs[0]["query"] == "machine learning tutorials"
    entities = pairs[0]["entities"]
    assert isinstance(entities, list) and len(entities) >= 1
    assert entities[0]["text"] == "machine learning"
    assert entities[0]["type"] == "CONCEPT"


@pytest.mark.asyncio
async def test_profile_selection_span_yields_training_pair(real_telemetry):
    from cogniverse_agents.profile_selection_agent import (
        ProfileSelectionAgent,
        ProfileSelectionDeps,
        ProfileSelectionInput,
    )

    tenant_id = "ps-opt-real"
    agent = ProfileSelectionAgent(deps=ProfileSelectionDeps(), port=19111)
    agent.set_telemetry_manager(real_telemetry)

    mock_result = MagicMock()
    mock_result.selected_profile = "video_colpali_smol500_mv_frame"
    mock_result.confidence = "0.8"
    mock_result.reasoning = "Video query matched colpali profile"
    mock_result.query_intent = "video_search"
    mock_result.modality = "video"
    mock_result.complexity = "simple"

    with patch.object(agent, "call_dspy", return_value=mock_result):
        await agent.process(
            ProfileSelectionInput(query="show me cooking videos", tenant_id=tenant_id)
        )

    spans = await _fetch_named_spans(
        real_telemetry, tenant_id, "cogniverse.profile_selection"
    )
    assert spans is not None, "cogniverse.profile_selection span not indexed"

    pairs = _profile_selection_pairs(spans)
    assert len(pairs) == 1
    assert pairs[0]["query"] == "show me cooking videos"
    assert pairs[0]["selected_profile"] == "video_colpali_smol500_mv_frame"
    assert pairs[0]["modality"] == "video"
    assert pairs[0]["confidence"] == 0.8


@pytest.mark.asyncio
async def test_orchestration_span_carries_canonical_workflow(real_telemetry):
    from cogniverse_agents.orchestrator_agent import (
        AgentStep,
        OrchestrationPlan,
        OrchestratorAgent,
        OrchestratorDeps,
        OrchestratorInput,
    )
    from cogniverse_core.common.agent_models import AgentEndpoint
    from cogniverse_core.registries.agent_registry import AgentRegistry
    from cogniverse_foundation.config.unified_config import SystemConfig

    tenant_id = "orch-opt-real"

    with patch.object(AgentRegistry, "__init__", lambda self, **kw: None):
        registry = AgentRegistry.__new__(AgentRegistry)
    registry.agents = {}
    registry.capabilities = {}
    registry.tenant_id = tenant_id
    registry.config_manager = MagicMock()
    registry.config = {}
    registry._http_client = MagicMock()
    registry.register_agent(
        AgentEndpoint(
            name="search_agent",
            url="http://localhost:8002",
            capabilities=["search"],
            process_endpoint="/tasks/send",
        )
    )

    mock_cm = MagicMock()
    mock_cm.get_system_config.return_value = SystemConfig(
        backend_url="http://localhost",
        backend_port=8080,
        iter_retrieval_max_iter=3,
        iter_retrieval_token_budget=10000,
        iter_retrieval_wall_clock_ms=10000,
    )
    mock_cm.get_config.return_value = {}

    agent = OrchestratorAgent(
        deps=OrchestratorDeps(),
        registry=registry,
        config_manager=mock_cm,
        port=19113,
    )
    agent.set_telemetry_manager(real_telemetry)

    plan = OrchestrationPlan(
        query="find machine learning videos",
        steps=[
            AgentStep(
                agent_name="search_agent",
                input_data={"query": "find machine learning videos"},
                depends_on=[],
                reasoning="Search for ML videos",
            ),
        ],
        parallel_groups=[],
        reasoning="Single search step",
        unavailable_agents=[],
    )
    results = {"search_agent": {"status": "success", "results": []}}

    with (
        patch.object(agent, "_create_plan", return_value=plan),
        patch.object(agent, "_execute_plan", return_value=results),
    ):
        await agent.process(
            OrchestratorInput(query="find machine learning videos", tenant_id=tenant_id)
        )

    spans = await _fetch_named_spans(
        real_telemetry, tenant_id, "cogniverse.orchestration"
    )
    assert spans is not None, "cogniverse.orchestration span not indexed"

    span_io = read_span_io(spans.iloc[0])
    assert span_io["input"] == "find machine learning videos"
    output = span_io["output"]
    assert isinstance(output, dict)
    assert output["workflow_id"]
    assert output["agent_sequence"] == ["search_agent"]
