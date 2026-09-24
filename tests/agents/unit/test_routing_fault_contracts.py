"""Each way relationship-aware routing can fail gets its own outcome.

An empty analysis reads exactly like a query holding no entities and no
relationships, so the consumers never produce one for a failure. The contract
separates failures by class:

  - a malformed LM answer is a named fallback, carrying the reason
  - a boundary that could not answer -- the spaCy pipeline, the LM -- raises
  - anything else raises as itself

Every boundary here is real: a spaCy pipeline that is genuinely not installed,
a recorded completion served over HTTP by a real server, and an LM pointed at
a port nothing listens on.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import dspy
import pytest

from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
    DSPyAdvancedRoutingModule,
)
from cogniverse_agents.routing.relationship_extraction_tools import (
    RelationshipExtractorTool,
    SpaCyDependencyAnalyzer,
    SpaCyModelUnavailableError,
)
from cogniverse_foundation.telemetry.span_contract import (
    ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED,
)
from tests.utils.recorded_endpoints import (
    RECORDED_REFUSAL,
    recorded_completion_lm,
    recorded_gliner_extractor,
)

pytestmark = pytest.mark.unit

QUERY = "Barack Obama visited Chicago in 2012"
# A port the test suite guarantees nothing listens on.
DEAD_PORT = 29071
DEAD_LM_BASE = f"http://127.0.0.1:{DEAD_PORT}/v1"
MISSING_PIPELINE = "xx_not_a_pipeline"
SHIPPED_PIPELINE = SpaCyDependencyAnalyzer().model_name
GLINER_TEST_MODEL = "urchade/gliner_small-v2.1"


def _dead_lm() -> dspy.LM:
    return dspy.LM(
        model="openai/dead",
        api_base=DEAD_LM_BASE,
        api_key="not-required",
        cache=False,
        num_retries=0,
        timeout=3,
    )


@contextmanager
def _no_entities_module(analyzer=None):
    """The real analysis module whose extractor answers with no entities.

    Both the extractor and its remote client are the classes the served agent
    builds; only the inference service's answer is recorded.
    """
    with recorded_gliner_extractor([], model_name=GLINER_TEST_MODEL) as extractor:
        yield ComposableQueryAnalysisModule(
            gliner_extractor=extractor,
            spacy_analyzer=analyzer or SpaCyDependencyAnalyzer(),
        )


def _outcome(call):
    """``("raised", exc)`` or ``("returned", value)`` -- which one happened."""
    try:
        return "returned", call()
    except BaseException as exc:  # noqa: BLE001 - the contract under test
        return "raised", exc


class TestAMissingSpaCyPipelineIsNamedNotEmpty:
    def test_relationship_extraction_raises_naming_the_pipeline(self):
        analyzer = SpaCyDependencyAnalyzer(model_name=MISSING_PIPELINE)

        with pytest.raises(SpaCyModelUnavailableError) as excinfo:
            analyzer.extract_semantic_relationships(QUERY)

        assert excinfo.value.model_name == MISSING_PIPELINE
        assert str(excinfo.value) == (
            f"spaCy model '{MISSING_PIPELINE}' could not be loaded: OSError: "
            f"[E050] Can't find model '{MISSING_PIPELINE}'. It doesn't seem to be a "
            "Python package or a valid path to a data directory. Install it as "
            "described under 'spaCy pipeline' in the cogniverse-agents README."
        )
        assert type(excinfo.value.__cause__) is OSError

    def test_dependency_analysis_raises_naming_the_pipeline(self):
        analyzer = SpaCyDependencyAnalyzer(model_name=MISSING_PIPELINE)

        with pytest.raises(SpaCyModelUnavailableError) as excinfo:
            analyzer.analyze_dependencies(QUERY)

        assert excinfo.value.model_name == MISSING_PIPELINE

    def test_availability_is_asked_for_not_inferred_from_an_empty_result(self):
        assert SpaCyDependencyAnalyzer(model_name=MISSING_PIPELINE).is_available() is (
            False
        )
        assert SpaCyDependencyAnalyzer().is_available() is True

    def test_the_shipped_pipeline_extracts_exactly_these_relationships(self):
        """The whole output for one sentence, so an empty one is unmissable."""
        assert SpaCyDependencyAnalyzer().extract_semantic_relationships(QUERY) == [
            {
                "subject": "Obama",
                "relation": "visit",
                "object": "Chicago",
                "confidence": 0.8,
                "grammatical_pattern": "nsubj-ROOT-dobj",
            },
            {
                "subject": "visited",
                "relation": "in",
                "object": "2012",
                "confidence": 0.7,
                "grammatical_pattern": "prep-in",
            },
        ]

    @pytest.mark.asyncio
    async def test_the_relationship_tool_does_not_absorb_the_missing_pipeline(self):
        tool = RelationshipExtractorTool(spacy_model=MISSING_PIPELINE)

        with recorded_gliner_extractor([], model_name=GLINER_TEST_MODEL) as extractor:
            tool.gliner_extractor = extractor
            with pytest.raises(SpaCyModelUnavailableError) as excinfo:
                await tool.extract_comprehensive_relationships(QUERY)

        assert excinfo.value.model_name == MISSING_PIPELINE


class TestAMalformedLMAnswerIsAMarkedFallback:
    def test_the_unified_path_marks_the_refusal_it_could_not_parse(self):
        with (
            _no_entities_module() as module,
            recorded_completion_lm(RECORDED_REFUSAL) as lm,
        ):
            with dspy.context(lm=lm):
                prediction = module(query=QUERY)

        assert prediction.fallback_reason == ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED
        assert prediction.path_used == "fallback"
        assert prediction.entities == []
        assert prediction.relationships == []
        assert prediction.enhanced_query == QUERY
        assert prediction.query_variants == []
        assert prediction.confidence == 0.0
        assert prediction.fallback_model is None
        assert prediction.fallback_inference_url is None

    def test_the_marker_is_the_shipped_vocabulary_value(self):
        assert ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED == "schema_refused"


class TestAnUnreachableLMRaises:
    def test_the_analysis_module_does_not_return_an_empty_analysis(self):
        with _no_entities_module() as module, dspy.context(lm=_dead_lm()):
            how, exc = _outcome(lambda: module(query=QUERY))

        assert how == "raised"
        assert type(exc).__module__ == "litellm.exceptions"
        assert type(exc).__name__ == "InternalServerError"
        assert exc.status_code == 500

    def test_the_advanced_module_does_not_absorb_it_one_layer_up(self):
        decision_calls = []

        def decision_predictor(**kwargs):
            decision_calls.append(kwargs)
            prediction = dspy.Prediction()
            prediction.routing_decision = {
                "search_modality": "video_only",
                "generation_type": "raw_results",
                "primary_agent": "search_agent",
                "secondary_agents": [],
                "execution_mode": "single",
                "confidence": "0.9",
                "reasoning": "recorded decision",
            }
            return prediction

        with _no_entities_module() as analysis, dspy.context(lm=_dead_lm()):
            module = DSPyAdvancedRoutingModule(analysis_module=analysis)
            module.router = decision_predictor
            how, exc = _outcome(
                lambda: module(query=QUERY, available_agents=["search_agent"])
            )

        assert how == "raised"
        assert type(exc).__name__ == "InternalServerError"
        # The raise is the analysis step's: the decision predictor, which
        # answers without an LM, is never reached.
        assert decision_calls == []


class TestTheIterativeLoopDoesNotReformulateAroundAnOutage:
    """The orchestrator's reformulation step is the analysis module's caller."""

    @pytest.mark.asyncio
    async def test_an_unreachable_lm_raises_out_of_the_reformulation_step(self):
        from cogniverse_agents.orchestrator_agent import OrchestratorAgent

        with _no_entities_module() as module:

            class _ReformulationHarness:
                _reformulate_query = OrchestratorAgent._reformulate_query

                def _get_query_analysis_module(self):
                    return module

            with dspy.context(lm=_dead_lm()):
                with pytest.raises(Exception) as excinfo:
                    await _ReformulationHarness()._reformulate_query(
                        QUERY, ["red jerseys"]
                    )

        assert type(excinfo.value).__name__ == "InternalServerError"
        assert excinfo.value.status_code == 500

    @pytest.mark.asyncio
    async def test_a_malformed_answer_reformulates_to_the_seeded_query(self):
        from cogniverse_agents.orchestrator_agent import OrchestratorAgent

        with (
            _no_entities_module() as module,
            recorded_completion_lm(RECORDED_REFUSAL) as lm,
        ):

            class _ReformulationHarness:
                _reformulate_query = OrchestratorAgent._reformulate_query

                def _get_query_analysis_module(self):
                    return module

            with dspy.context(lm=lm):
                (
                    reformulated,
                    rationale,
                ) = await _ReformulationHarness()._reformulate_query(
                    QUERY, ["red jerseys"]
                )

        assert reformulated == f"{QUERY} (focus on: red jerseys)"
        assert rationale == "All analysis paths failed"


class TestEveryConcurrentCallDuringAnOutageRaises:
    CALLS = 8

    def test_eight_concurrent_analyses_raise_eight_times(self):
        lm = _dead_lm()

        def one(_index):
            with _no_entities_module() as module, dspy.context(lm=lm):
                return _outcome(lambda: module(query=QUERY))

        with ThreadPoolExecutor(max_workers=self.CALLS) as pool:
            outcomes = list(pool.map(one, range(self.CALLS)))

        assert Counter(how for how, _ in outcomes) == {"raised": self.CALLS}
        assert Counter(type(exc).__name__ for _, exc in outcomes) == {
            "InternalServerError": self.CALLS
        }

    def test_eight_concurrent_analyses_with_a_missing_pipeline_all_name_it(self):
        def one(_index):
            analyzer = SpaCyDependencyAnalyzer(model_name=MISSING_PIPELINE)
            return _outcome(lambda: analyzer.extract_semantic_relationships(QUERY))

        with ThreadPoolExecutor(max_workers=self.CALLS) as pool:
            outcomes = list(pool.map(one, range(self.CALLS)))

        assert Counter(how for how, _ in outcomes) == {"raised": self.CALLS}
        assert {exc.model_name for _, exc in outcomes} == {MISSING_PIPELINE}
        assert Counter(type(exc).__name__ for _, exc in outcomes) == {
            "SpaCyModelUnavailableError": self.CALLS
        }


class TestTheShippedPipelineNameIsTheConfiguredOne:
    def test_the_default_analyzer_uses_the_shipped_pipeline(self):
        assert SHIPPED_PIPELINE == "en_core_web_sm"
