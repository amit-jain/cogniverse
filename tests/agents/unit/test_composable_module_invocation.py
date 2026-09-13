"""Both consumers call the analysis module through the DSPy call protocol.

The module runs end to end over recorded services -- the production GLiNER
extractor and a real LM endpoint, each answering from a recording -- so the
call protocol and what each consumer reads back are both pinned exactly.
"""

import json

import dspy
import pytest
from dspy.utils.callback import BaseCallback

from cogniverse_agents.orchestrator_agent import OrchestratorAgent
from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
    DSPyAdvancedRoutingModule,
)
from cogniverse_agents.routing.relationship_extraction_tools import (
    SpaCyDependencyAnalyzer,
)
from tests.utils.recorded_endpoints import (
    recorded_completion_lm,
    recorded_gliner_extractor,
)

GLINER_TEST_MODEL = "urchade/gliner_small-v2.1"
RECORDED_ENHANCED_QUERY = "robot soccer clip with players in red jerseys"
RECORDED_REASONING = "The gap names a visual attribute of the players."
RECORDED_ANSWER = json.dumps(
    {
        "reasoning": RECORDED_REASONING,
        "entities": json.dumps([]),
        "relationships": json.dumps([]),
        "enhanced_query": RECORDED_ENHANCED_QUERY,
        "query_variants": json.dumps([]),
        "domain_classification": "sports",
        "confidence": "0.8",
    }
)


class _ModuleCallRecorder(BaseCallback):
    def __init__(self):
        self.inputs = []

    def on_module_start(self, call_id, instance, inputs):
        if isinstance(instance, ComposableQueryAnalysisModule):
            self.inputs.append(inputs)


@pytest.fixture()
def recorded_services():
    """The production extractor with no entities, and an LM answering once."""
    with (
        recorded_gliner_extractor([], model_name=GLINER_TEST_MODEL) as extractor,
        recorded_completion_lm(RECORDED_ANSWER) as lm,
    ):
        yield extractor, lm


def _analysis_module(extractor, recorder):
    module = ComposableQueryAnalysisModule(
        gliner_extractor=extractor,
        spacy_analyzer=SpaCyDependencyAnalyzer(),
    )
    module.callbacks = [recorder]
    return module


def test_advanced_router_uses_composable_module_call_protocol(recorded_services):
    extractor, lm = recorded_services
    recorder = _ModuleCallRecorder()
    router = DSPyAdvancedRoutingModule(
        analysis_module=_analysis_module(extractor, recorder)
    )

    with dspy.context(lm=lm):
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
    assert prediction.enhanced_query == RECORDED_ENHANCED_QUERY
    assert prediction.extracted_entities == []
    assert prediction.extracted_relationships == []


@pytest.mark.asyncio
async def test_iterative_reformulation_uses_composable_module_call_protocol(
    recorded_services,
):
    extractor, lm = recorded_services
    recorder = _ModuleCallRecorder()
    analysis_module = _analysis_module(extractor, recorder)

    class _ReformulationHarness:
        _reformulate_query = OrchestratorAgent._reformulate_query

        def _get_query_analysis_module(self):
            return analysis_module

    with dspy.context(lm=lm):
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
    assert reformulated == RECORDED_ENHANCED_QUERY
    assert rationale == RECORDED_REASONING
