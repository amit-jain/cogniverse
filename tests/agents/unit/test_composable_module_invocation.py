import pytest
from dspy.utils.callback import BaseCallback

from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
    DSPyAdvancedRoutingModule,
)


class _EmptyExtractor:
    gliner_model = None

    def extract_entities(self, query):
        return []


class _NoopDependencyAnalyzer:
    def extract_semantic_relationships(self, query):
        return []


class _ModuleCallRecorder(BaseCallback):
    def __init__(self):
        self.inputs = []

    def on_module_start(self, call_id, instance, inputs):
        self.inputs.append(inputs)


def _analysis_module(recorder):
    module = ComposableQueryAnalysisModule(
        gliner_extractor=_EmptyExtractor(),
        spacy_analyzer=_NoopDependencyAnalyzer(),
    )
    module.callbacks = [recorder]
    return module


def test_advanced_router_uses_composable_module_call_protocol():
    recorder = _ModuleCallRecorder()
    router = DSPyAdvancedRoutingModule(analysis_module=_analysis_module(recorder))

    prediction = router(
        "Find the robot soccer clip",
        available_agents=["search_agent", "summarizer_agent"],
    )

    assert recorder.inputs == [
        {
            "args": (),
            "kwargs": {
                "query": "Find the robot soccer clip",
                "search_context": "general",
            },
        }
    ]
    assert prediction.enhanced_query == "Find the robot soccer clip"
    assert prediction.extracted_entities == []
    assert prediction.extracted_relationships == []


@pytest.mark.asyncio
async def test_iterative_reformulation_uses_composable_module_call_protocol():
    recorder = _ModuleCallRecorder()
    analysis_module = _analysis_module(recorder)

    class _ReformulationHarness:
        _reformulate_query = OrchestratorAgent._reformulate_query

        def _get_query_analysis_module(self):
            return analysis_module

    reformulated, rationale = await _ReformulationHarness()._reformulate_query(
        "Find the robot soccer clip",
        ["red jerseys"],
    )

    expected_query = "Find the robot soccer clip (focus on: red jerseys)"
    assert recorder.inputs == [
        {
            "args": (),
            "kwargs": {
                "query": expected_query,
                "search_context": "general",
            },
        }
    ]
    assert reformulated == expected_query
    assert rationale == "All analysis paths failed"
