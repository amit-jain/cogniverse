"""``LLMAutoAnnotator`` sends the real inference bearer to ``litellm.completion``.

It forwarded ``LLMEndpointConfig.api_key`` verbatim, and the shipped
``annotation`` config carries ``api_key: null``, so a Modal ``api_base`` got no
credential and every auto-annotation raised 401.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
)
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
    AutoAnnotation,
    LLMAutoAnnotator,
)
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
MODEL = "openai/google/gemma-4-e4b-it"


def _request() -> AnnotationRequest:
    return AnnotationRequest(
        span_id="span-1",
        timestamp=datetime.now(timezone.utc),
        query="show me the red car",
        chosen_agent="video_search",
        routing_confidence=0.5,
        outcome=RoutingOutcome.SUCCESS,
        priority=AnnotationPriority.MEDIUM,
        reason="low confidence",
        context={},
    )


def _completion_reply(payload: dict):
    content = json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_modal_endpoint_gets_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    annotator = LLMAutoAnnotator(LLMEndpointConfig(model=MODEL, api_base=MODAL))

    assert annotator.api_key == "real-bearer"


def test_without_an_api_base_the_configured_key_passes_through(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    annotator = LLMAutoAnnotator(
        LLMEndpointConfig(model="anthropic/claude-3-5-sonnet-20241022")
    )

    assert annotator.api_key is None


def test_modal_endpoint_without_a_bearer_fails_at_construction(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        LLMAutoAnnotator(LLMEndpointConfig(model=MODEL, api_base=MODAL))


def test_the_bearer_is_handed_to_litellm(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    annotator = LLMAutoAnnotator(LLMEndpointConfig(model=MODEL, api_base=MODAL))
    calls: list[dict] = []

    def _completion(**kwargs):
        calls.append(kwargs)
        return _completion_reply(
            {
                "label": "correct_routing",
                "confidence": 0.9,
                "reasoning": "agent matched the query intent",
                "suggested_correct_agent": None,
                "requires_human_review": False,
            }
        )

    with patch("cogniverse_agents.routing.llm_auto_annotator.completion", _completion):
        annotation = annotator.annotate(_request())

    assert annotation == AutoAnnotation(
        span_id="span-1",
        label=AnnotationLabel.CORRECT_ROUTING,
        confidence=0.9,
        reasoning="agent matched the query intent",
        suggested_correct_agent=None,
        requires_human_review=False,
    )
    [call] = calls
    messages = call.pop("messages")
    assert call == {
        "model": MODEL,
        "max_tokens": 1024,
        "temperature": 0.3,
        "api_key": "real-bearer",
        "api_base": MODAL,
    }
    assert [m["role"] for m in messages] == ["user"]
    assert "- **Query**: show me the red car" in messages[0]["content"]
