"""Profile selection query intent contract tests."""

from types import SimpleNamespace
from typing import Literal, get_args, get_origin
from unittest.mock import AsyncMock, Mock, patch

import dspy
import pytest
from pydantic import ValidationError

from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
    ProfileSelectionOutput,
    ProfileSelectionSignature,
)
from cogniverse_core.approval.training_schema import PROFILE_TRAINING_MODALITIES
from cogniverse_synthetic.schemas import ProfileSelectionExampleSchema

EXPECTED_PROFILE_QUERY_INTENTS = (
    "multi_modal_search",
    "video_search",
    "image_search",
    "text_search",
    "audio_search",
    "document_search",
    "relationship_aware_search",
    "ensemble_search",
    "code_search",
    "wiki_search",
)


def _base_profile_selection_available_profiles() -> str:
    from cogniverse_agents.profile_selection_agent import (
        tenant_usable_profile_names,
    )
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import (
        BackendProfileConfig,
        SystemConfig,
    )
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    config_manager = ConfigManager(store=store)
    config_manager.set_system_config(
        SystemConfig(
            inference_service_urls={
                "vllm_colpali": "http://localhost:8000",
                "vllm_colqwen": "http://localhost:8001",
            }
        )
    )
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(
            "video_colpali_smol500_mv_frame",
            {
                "type": "video",
                "schema_name": "video_colpali_smol500_mv_frame",
                "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                "inference_services": {"embedding": "vllm_colpali"},
            },
        ),
        tenant_id="acme:docs",
    )
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(
            "video_colqwen_omni_mv_chunk_30s",
            {
                "type": "video",
                "schema_name": "video_colqwen_omni_mv_chunk_30s",
                "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                "inference_services": {"embedding": "vllm_colqwen"},
            },
        ),
        tenant_id="acme:docs",
    )
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(
            "video_videoprism_base_mv_chunk_30s",
            {
                "type": "video",
                "schema_name": "video_videoprism_base_mv_chunk_30s",
                "embedding_model": "videoprism_public_v1_base_hf",
                "inference_services": {"embedding": "videoprism_jax"},
            },
        ),
        tenant_id="acme:docs",
    )

    return ", ".join(tenant_usable_profile_names(config_manager, "acme:docs"))


BASE_PROFILE_SELECTION_EXAMPLE = {
    "query": "find a clip about transformer architecture",
    "available_profiles": _base_profile_selection_available_profiles(),
    "selected_profile": "video_colqwen_omni_mv_chunk_30s",
    "reasoning": "Selected chunk-based profile for medium-complexity video search.",
    "modality": "video",
    "complexity": "medium",
}


def _build_agent(
    modality: str, *, model_query_intent: str = "text_search"
) -> ProfileSelectionAgent:
    """A real ProfileSelectionAgent whose tenant config declares ``modality``
    for every profile and whose LM answers ``model_query_intent``."""
    with patch("dspy.ChainOfThought"):
        agent = ProfileSelectionAgent(
            deps=ProfileSelectionDeps(
                tenant_id="acme:docs", available_profiles=["video_profile"]
            ),
            port=8011,
        )
    agent._config_manager = Mock()
    agent._config_manager.get_backend_profile.return_value = SimpleNamespace(
        type=modality
    )
    agent.call_dspy = AsyncMock(
        return_value=dspy.Prediction(
            selected_profile="video_profile",
            confidence="0.9",
            reasoning="selected",
            query_intent=model_query_intent,
            modality="text",
            complexity="medium",
        )
    )
    return agent


def test_profile_query_intent_literal_alias_is_closed():
    assert (
        get_origin(ProfileSelectionSignature.__annotations__["query_intent"]) is Literal
    )
    assert get_args(ProfileSelectionSignature.__annotations__["query_intent"]) == (
        EXPECTED_PROFILE_QUERY_INTENTS
    )
    assert (
        get_origin(
            ProfileSelectionExampleSchema.model_fields["query_intent"].annotation
        )
        is Literal
    )
    assert (
        get_args(ProfileSelectionExampleSchema.model_fields["query_intent"].annotation)
        == EXPECTED_PROFILE_QUERY_INTENTS
    )


def test_profile_selection_schema_rejects_out_of_vocab_query_intent():
    with pytest.raises(ValidationError) as exc_info:
        ProfileSelectionExampleSchema.model_validate(
            {
                **BASE_PROFILE_SELECTION_EXAMPLE,
                "query_intent": "cross_modal_search",
            }
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("query_intent",)
    assert errors[0]["type"] == "literal_error"
    assert errors[0]["input"] == "cross_modal_search"


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", sorted(PROFILE_TRAINING_MODALITIES))
async def test_profile_selection_derives_in_vocab_query_intent_for_every_modality(
    modality: str,
):
    agent = _build_agent(modality)
    output = await ProfileSelectionAgent._process_impl(
        agent,
        ProfileSelectionInput(
            query="find a clip about transformer architecture",
            available_profiles=["video_profile"],
            tenant_id="acme:docs",
        ),
    )

    expected_query_intent = (
        "text_search" if modality == "text" else f"{modality}_search"
    )
    assert output.query_intent == expected_query_intent


def _build_agent_with_empty_intent(modality: str) -> ProfileSelectionAgent:
    return _build_agent(modality, model_query_intent="")


def test_profile_selection_output_pins_the_same_query_intent_vocabulary():
    annotation = ProfileSelectionOutput.model_fields["query_intent"].annotation
    assert get_origin(annotation) is Literal
    assert get_args(annotation) == EXPECTED_PROFILE_QUERY_INTENTS


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", sorted(PROFILE_TRAINING_MODALITIES))
async def test_empty_model_intent_still_derives_in_vocab_query_intent(modality: str):
    output = await ProfileSelectionAgent._process_impl(
        _build_agent_with_empty_intent(modality),
        ProfileSelectionInput(
            query="find a clip about transformer architecture",
            available_profiles=["video_profile"],
            tenant_id="acme:docs",
        ),
    )

    expected = "text_search" if modality == "text" else f"{modality}_search"
    assert output.query_intent == expected
