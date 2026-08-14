"""Profile selection query intent contract tests."""

from types import SimpleNamespace
from typing import Literal, get_args, get_origin

import pytest
from pydantic import ValidationError

from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
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

BASE_PROFILE_SELECTION_EXAMPLE = {
    "query": "find a clip about transformer architecture",
    "available_profiles": (
        "video_colpali_smol500_mv_frame,"
        "video_colqwen_omni_mv_chunk_30s,"
        "video_videoprism_base_mv_chunk_30s,"
        "video_videoprism_large_mv_chunk_30s"
    ),
    "selected_profile": "video_colqwen_omni_mv_chunk_30s",
    "reasoning": "Selected chunk-based profile for medium-complexity video search.",
    "modality": "video",
    "complexity": "medium",
}


def _build_agent(modality: str) -> SimpleNamespace:
    async def call_dspy(*args, **kwargs):
        return SimpleNamespace(
            selected_profile="video_profile",
            confidence="0.9",
            reasoning="selected",
            query_intent="text_search",
            modality="text",
            complexity="medium",
        )

    return SimpleNamespace(
        deps=SimpleNamespace(available_profiles=["video_profile"]),
        dspy_module=SimpleNamespace(),
        call_dspy=call_dspy,
        emit_progress=lambda *args, **kwargs: None,
        _configured_profile_modality=lambda selected_profile, tenant_id: modality,
        _generate_alternatives=lambda *args, **kwargs: [],
        _emit_profile_span=lambda *args, **kwargs: None,
    )


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
            tenant_id=None,
        ),
    )

    expected_query_intent = (
        "text_search" if modality == "text" else f"{modality}_search"
    )
    assert output.query_intent == expected_query_intent


def _build_agent_with_empty_intent(modality: str) -> SimpleNamespace:
    agent = _build_agent(modality)

    async def call_dspy(*args, **kwargs):
        return SimpleNamespace(
            selected_profile="video_profile",
            confidence="0.9",
            reasoning="selected",
            query_intent="",
            modality="text",
            complexity="medium",
        )

    agent.call_dspy = call_dspy
    return agent


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
            tenant_id=None,
        ),
    )

    expected = "text_search" if modality == "text" else f"{modality}_search"
    assert output.query_intent == expected
