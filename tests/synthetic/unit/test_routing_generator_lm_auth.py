"""The routing generator's query LM is built through ``create_dspy_lm``.

``lm_config`` used to be splatted into ``dspy.LM`` directly, so a Modal
``api_base`` got the config's ``api_key`` (``None`` or the chart placeholder)
instead of the environment bearer, and every synthetic routing example failed
with 401 inside the generator's retry loop.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.unified_config import (
    DSPyModuleConfig,
    OptimizerGenerationConfig,
)
from cogniverse_synthetic.generators.routing import RoutingGenerator

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

PLACEHOLDER = "placeholder-no-auth-needed"
MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
MODEL = "openai/google/gemma-4-e4b-it"


async def _extract_entities(text: str, tenant_id: str):
    return {"query": text, "entities": [], "relationships": []}


async def _route_query(query: str, tenant_id: str):
    return {"query": query, "routed_to": "video_search_agent", "confidence": 0.7}


def _generator(lm_config: dict) -> RoutingGenerator:
    return RoutingGenerator(
        entity_extractor=_extract_entities,
        routing_decider=_route_query,
        optimizer_config=OptimizerGenerationConfig(
            optimizer_type="routing",
            dspy_modules={
                "query_generator": DSPyModuleConfig(
                    signature_class=(
                        "cogniverse_synthetic.dspy_signatures.GenerateEntityQuery"
                    ),
                    module_type="Predict",
                    lm_config=lm_config,
                    metadata={"max_retries": 3},
                )
            },
        ),
    )


def test_query_lm_for_a_modal_endpoint_carries_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")

    lm = (
        _generator({"model": MODEL, "api_base": MODAL, "api_key": PLACEHOLDER})
        ._get_query_generator()
        .lm
    )

    assert lm.model == MODEL
    assert lm.kwargs["api_base"] == MODAL
    assert lm.kwargs["api_key"] == "real-bearer"


def test_query_lm_for_a_modal_endpoint_without_a_bearer_fails_at_build(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    generator = _generator({"model": MODEL, "api_base": MODAL})

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        generator._get_query_generator()


def test_empty_lm_config_defers_to_the_process_lm():
    assert _generator({})._get_query_generator().lm is None


def test_unknown_lm_config_keys_are_rejected_not_dropped():
    generator = _generator({"model": MODEL, "cache": False, "model_type": "chat"})

    with pytest.raises(
        ValueError,
        match=r"^query_generator lm_config has unknown keys: \['cache', 'model_type'\]$",
    ):
        generator._get_query_generator()


def test_lm_config_without_a_model_is_rejected():
    generator = _generator({"api_base": MODAL})

    with pytest.raises(
        ValueError, match="^query_generator lm_config requires a non-empty model$"
    ):
        generator._get_query_generator()
