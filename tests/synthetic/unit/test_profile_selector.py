"""Profile selection contracts for diversity and configured LM behavior."""

import asyncio
import json
from pathlib import Path

import pytest

from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    ProfileScoringRule,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.profile_selector import ProfileSelector
from cogniverse_synthetic.registry import OPTIMIZER_REGISTRY

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _described_profiles(*names: str) -> dict[str, dict[str, str]]:
    return {
        name: {
            "description": f"Configured description for {name}",
            "embedding_type": "multi_vector",
            "schema_name": f"{name}_schema",
        }
        for name in names
    }


def _generator_config() -> SyntheticGeneratorConfig:
    scoring_configs = {
        name: OptimizerGenerationConfig(
            optimizer_type=name,
            profile_scoring_rules=[
                ProfileScoringRule(
                    condition={"field": "type", "equals": "document"},
                    score_adjustment=3.0,
                    reason=f"{name} document rule",
                )
            ],
        )
        for name in OPTIMIZER_REGISTRY
    }
    scoring_configs["routing"].dspy_modules = {
        "query_generator": DSPyModuleConfig(
            signature_class=("cogniverse_synthetic.dspy_signatures.GenerateEntityQuery")
        )
    }
    scoring_configs["modality"] = OptimizerGenerationConfig(
        optimizer_type="modality",
        agent_mappings=[
            AgentMappingRule(modality="DOCUMENT", agent_name="document_agent")
        ],
    )
    return SyntheticGeneratorConfig(
        tenant_id="test:synthetic",
        optimizer_configs=scoring_configs,
    )


async def test_rule_selection_uses_only_configured_optimizer_rules() -> None:
    selector = ProfileSelector(generator_config=_generator_config())
    profiles = {
        "alpha": {"type": "video"},
        "beta": {"type": "document"},
    }

    selected, _ = await selector.select_profiles(
        optimizer_name="profile",
        optimizer_task="choose a search profile",
        available_profiles=profiles,
        max_profiles=1,
    )

    assert selected == ["beta"]


def test_rule_selection_rejects_missing_generator_config() -> None:
    with pytest.raises(
        ValueError,
        match="^profile scoring requires SyntheticGeneratorConfig$",
    ):
        ProfileSelector()._score_profile(
            "profile",
            "alpha",
            {"type": "video"},
        )


def test_synthetic_config_requires_scoring_rules_for_every_exposed_generator() -> None:
    optimizer_configs = _generator_config().optimizer_configs
    optimizer_configs["query_enhancement"] = OptimizerGenerationConfig(
        optimizer_type="query_enhancement"
    )

    with pytest.raises(
        ValueError,
        match="^optimizer 'query_enhancement' requires profile_scoring_rules$",
    ):
        SyntheticGeneratorConfig(
            tenant_id="test:synthetic",
            optimizer_configs=optimizer_configs,
        )


def test_synthetic_config_requires_every_canonical_optimizer() -> None:
    optimizer_configs = _generator_config().optimizer_configs
    optimizer_configs.pop("workflow")

    with pytest.raises(
        ValueError,
        match="^optimizer_configs must contain exactly: ",
    ):
        SyntheticGeneratorConfig(
            tenant_id="test:synthetic",
            optimizer_configs=optimizer_configs,
        )


def test_synthetic_config_rejects_decorative_optimizer_fields() -> None:
    optimizer_configs = _generator_config().optimizer_configs
    optimizer_configs["profile"].dspy_modules = {
        "unused": DSPyModuleConfig(signature_class="unused.Signature")
    }

    with pytest.raises(
        ValueError,
        match="^optimizer 'profile' only accepts profile_scoring_rules$",
    ):
        SyntheticGeneratorConfig(
            tenant_id="test:synthetic",
            optimizer_configs=optimizer_configs,
        )


def test_synthetic_config_serializes_only_consumed_optimizer_fields() -> None:
    serialized = _generator_config().to_dict()["optimizer_configs"]

    assert set(serialized) == {
        "cross_modal",
        "entity_extraction",
        "modality",
        "profile",
        "query_enhancement",
        "routing",
        "unified",
        "workflow",
    }
    assert set(serialized["modality"]) == {"optimizer_type", "agent_mappings"}
    assert set(serialized["routing"]) == {
        "optimizer_type",
        "dspy_modules",
        "profile_scoring_rules",
    }
    assert set(serialized["profile"]) == {
        "optimizer_type",
        "profile_scoring_rules",
    }
    assert set(serialized["entity_extraction"]) == {
        "optimizer_type",
        "profile_scoring_rules",
    }
    for optimizer_name in (
        "cross_modal",
        "query_enhancement",
        "unified",
        "workflow",
    ):
        assert set(serialized[optimizer_name]) == {
            "optimizer_type",
            "profile_scoring_rules",
        }


def test_agent_mapping_rejects_obsolete_confidence_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="^AgentMappingRule contains unsupported fields: confidence_threshold$",
    ):
        AgentMappingRule.from_dict(
            {
                "modality": "VIDEO",
                "agent_name": "search_agent",
                "confidence_threshold": 0.7,
            }
        )


def test_selection_prompt_uses_the_configured_profile_description() -> None:
    prompt = ProfileSelector()._build_selection_prompt(
        optimizer_name="profile",
        optimizer_task="select a visual profile",
        available_profiles={
            "video_colpali_smol500_mv_frame": {
                "description": "Tomoro visual embeddings with 320 dimensions.",
                "embedding_type": "multi_vector",
                "schema_name": "video_colpali_smol500_mv_frame",
            }
        },
        max_profiles=1,
    )

    assert "Description: Tomoro visual embeddings with 320 dimensions." in prompt
    assert "128-dim" not in prompt


def test_selection_prompt_rejects_a_profile_without_description() -> None:
    with pytest.raises(
        ValueError,
        match="^Backend profile 'image_profile' requires a non-empty description$",
    ):
        ProfileSelector()._build_selection_prompt(
            optimizer_name="profile",
            optimizer_task="select an image profile",
            available_profiles={
                "image_profile": {
                    "embedding_type": "multi_vector",
                    "schema_name": "image_colpali_mv",
                }
            },
            max_profiles=1,
        )


@pytest.mark.parametrize(
    "config_path",
    [
        _REPO_ROOT / "configs/config.json",
        _REPO_ROOT / "configs/examples/config.example.json",
    ],
)
def test_tomoro_profiles_match_the_320_dimensional_schema(config_path: Path) -> None:
    profiles = json.loads(config_path.read_text())["backend"]["profiles"]
    for profile_name in (
        "video_colpali_smol500_mv_frame",
        "image_colpali_mv",
        "video_colqwen_omni_mv_chunk_30s",
    ):
        profile = profiles[profile_name]
        assert profile["embedding_model"] == "TomoroAI/tomoro-colqwen3-embed-4b"
        assert profile["schema_config"]["embedding_dim"] == 320
        assert profile["schema_config"]["binary_dim"] == 40
        assert profile["inference_services"]["embedding"] == "vllm_colpali"
        assert "320-dim" in profile["description"]
        assert "128-dim" not in profile["description"]

    code_profile = profiles["code_lateon_mv"]
    assert code_profile["schema_config"]["embedding_dim"] == 48
    assert "48-dim" in code_profile["description"]
    assert "128-dim" not in code_profile["description"]


def test_chart_profile_descriptions_match_their_schema_dimensions() -> None:
    chart_config = (_REPO_ROOT / "charts/cogniverse/files/config.json").read_text()
    for stale_description in (
        "generates 128-dim patch embeddings",
        "ColPali multi-vector embeddings (128-dim per patch)",
        "128-dim per-patch multi-vector embeddings",
        "LateOn-Code-edge multi-vector embeddings (128-dim)",
    ):
        assert stale_description not in chart_config


async def test_configured_lm_failure_raises_with_optimizer_context() -> None:
    class _UnavailableLM:
        async def generate(self, prompt):
            raise TimeoutError("profile teacher timed out")

    selector = ProfileSelector(llm_client=_UnavailableLM())

    with pytest.raises(
        RuntimeError,
        match="profile.*LM profile selection failed",
    ) as error:
        await selector.select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=_described_profiles("image_profile"),
            max_profiles=1,
        )

    assert isinstance(error.value.__cause__, TimeoutError)
    assert str(error.value.__cause__) == "profile teacher timed out"


async def test_configured_lm_must_return_at_least_one_available_profile() -> None:
    class _WrongProfileLM:
        async def generate(self, prompt):
            return '{"selected": ["missing_profile"], "reasoning": "wrong"}'

    selector = ProfileSelector(llm_client=_WrongProfileLM())

    with pytest.raises(
        ValueError,
        match="no available profiles",
    ):
        await selector.select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=_described_profiles("image_profile"),
            max_profiles=1,
        )


async def test_configured_lm_rejects_unknown_profile_in_mixed_selection() -> None:
    class _UnknownProfileLM:
        async def generate(self, prompt):
            return (
                '{"selected": ["image_profile", "missing_profile"], '
                '"reasoning": "mixed selection"}'
            )

    selector = ProfileSelector(llm_client=_UnknownProfileLM())

    with pytest.raises(
        ValueError,
        match=(
            "^profile LM profile selection contains unknown profiles: missing_profile$"
        ),
    ):
        await selector.select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=_described_profiles("image_profile", "audio_profile"),
            max_profiles=2,
        )


async def test_configured_lm_rejects_duplicate_profiles() -> None:
    class _DuplicateProfileLM:
        async def generate(self, prompt):
            return (
                '{"selected": ["image_profile", "image_profile"], '
                '"reasoning": "duplicate selection"}'
            )

    selector = ProfileSelector(llm_client=_DuplicateProfileLM())

    with pytest.raises(
        ValueError,
        match=(
            "^profile LM profile selection contains duplicate profiles: image_profile$"
        ),
    ):
        await selector.select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=_described_profiles("image_profile", "audio_profile"),
            max_profiles=2,
        )


async def test_configured_lm_rejects_selection_above_requested_limit() -> None:
    class _ExcessiveProfileLM:
        async def generate(self, prompt):
            return (
                '{"selected": ["image_profile", "audio_profile"], '
                '"reasoning": "too many profiles"}'
            )

    selector = ProfileSelector(llm_client=_ExcessiveProfileLM())

    with pytest.raises(
        ValueError,
        match=("^profile LM profile selection returned 2 profiles; maximum is 1$"),
    ):
        await selector.select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=_described_profiles("image_profile", "audio_profile"),
            max_profiles=1,
        )


async def test_configured_lm_non_text_response_is_rejected_as_invalid_output() -> None:
    class _NonTextLM:
        async def generate(self, prompt):
            return {"selected": ["image_profile"], "reasoning": "wrong transport type"}

    selector = ProfileSelector(llm_client=_NonTextLM())

    with pytest.raises(
        ValueError,
        match="LM profile selection response must be text",
    ):
        await selector.select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=_described_profiles("image_profile"),
            max_profiles=1,
        )


async def test_concurrent_lm_selections_keep_request_results_separate() -> None:
    request_barrier = asyncio.Barrier(2)

    class _ConcurrentLM:
        async def generate(self, prompt):
            await request_barrier.wait()
            selected = "image_profile" if "image task" in prompt else "audio_profile"
            return f'{{"selected": ["{selected}"], "reasoning": "selected {selected}"}}'

    selector = ProfileSelector(llm_client=_ConcurrentLM())
    image_result, audio_result = await asyncio.gather(
        selector.select_profiles(
            optimizer_name="image",
            optimizer_task="image task",
            available_profiles=_described_profiles("image_profile", "audio_profile"),
            max_profiles=1,
        ),
        selector.select_profiles(
            optimizer_name="audio",
            optimizer_task="audio task",
            available_profiles=_described_profiles("image_profile", "audio_profile"),
            max_profiles=1,
        ),
    )

    assert image_result == (["image_profile"], "selected image_profile")
    assert audio_result == (["audio_profile"], "selected audio_profile")


@pytest.mark.parametrize(
    ("available_profiles", "max_profiles", "message"),
    [
        ({}, 1, "available_profiles must not be empty"),
        ({"image_profile": {}}, 0, "max_profiles must be at least 1"),
    ],
)
async def test_selection_rejects_invalid_bounds(
    available_profiles,
    max_profiles,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        await ProfileSelector().select_profiles(
            optimizer_name="profile",
            optimizer_task="choose a search profile",
            available_profiles=available_profiles,
            max_profiles=max_profiles,
        )
