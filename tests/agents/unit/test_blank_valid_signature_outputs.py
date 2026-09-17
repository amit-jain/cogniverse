"""Outputs whose blank answer is valid parse as their default; the rest stay required."""

from __future__ import annotations

import json

import dspy
import pytest

from cogniverse_agents.coding_agent import OutputEvaluationSignature
from cogniverse_agents.orchestrator_agent import OrchestrationSignature
from cogniverse_agents.query_enhancement_agent import QueryEnhancementSignature
from cogniverse_foundation.dspy.lenient_json_adapter import (
    LenientJSONAdapter,
    LMOutputIncomplete,
)

FILLED = {
    OrchestrationSignature: {
        "reasoning": "Search, then summarize.",
        "agent_sequence": "search_agent,summarizer_agent",
        "parallel_steps": "",
    },
    QueryEnhancementSignature: {
        "reasoning": "Adds the full term.",
        "enhanced_query": "machine learning tutorials for beginners",
        "expansion_terms": "machine learning",
        "synonyms": "",
        "context": "",
        "confidence": "0.8",
    },
    OutputEvaluationSignature: {
        "reasoning": "The output matches the task.",
        "is_successful": True,
        "feedback": "",
    },
}
BLANK_VALID = {
    OrchestrationSignature: {"parallel_steps"},
    QueryEnhancementSignature: {"synonyms", "context"},
    OutputEvaluationSignature: {"feedback"},
}


def cot(signature):
    return dspy.ChainOfThought(signature).predict.signature


@pytest.mark.parametrize(
    "signature", list(FILLED), ids=lambda signature: signature.__name__
)
def test_declared_outputs_are_the_blank_valid_ones(signature):
    assert {
        name
        for name, field in cot(signature).output_fields.items()
        if not field.is_required()
    } == BLANK_VALID[signature]
    assert {
        name: cot(signature).output_fields[name].default
        for name in BLANK_VALID[signature]
    } == dict.fromkeys(BLANK_VALID[signature], "")


@pytest.mark.parametrize(
    "signature", list(FILLED), ids=lambda signature: signature.__name__
)
def test_blank_valid_outputs_parse_as_blank(signature):
    completion = dict(FILLED[signature])
    for name in BLANK_VALID[signature]:
        completion[name] = None
    assert LenientJSONAdapter().parse(cot(signature), json.dumps(completion)) == {
        **FILLED[signature],
        **dict.fromkeys(BLANK_VALID[signature], ""),
    }


@pytest.mark.parametrize(
    "signature", list(FILLED), ids=lambda signature: signature.__name__
)
def test_every_other_blank_output_is_incomplete(signature):
    required = set(FILLED[signature]) - BLANK_VALID[signature]
    completion = {name: "" for name in FILLED[signature]}
    with pytest.raises(LMOutputIncomplete) as error:
        LenientJSONAdapter().parse(cot(signature), json.dumps(completion))
    assert set(error.value.missing_fields) == required
    assert error.value.parsed_result == dict.fromkeys(BLANK_VALID[signature], "")
