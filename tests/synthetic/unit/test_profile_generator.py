"""Unit tests for ProfileGenerator."""

import pytest

from cogniverse_synthetic.generators import ProfileGenerator
from cogniverse_synthetic.schemas import ProfileSelectionExampleSchema


class TestProfileGenerator:
    @pytest.mark.asyncio
    async def test_generates_target_count(self):
        gen = ProfileGenerator()
        examples = await gen.generate(sampled_content=[], target_count=5)
        assert len(examples) == 5
        assert all(isinstance(e, ProfileSelectionExampleSchema) for e in examples)

    @pytest.mark.asyncio
    async def test_selected_profile_is_in_available(self):
        gen = ProfileGenerator()
        examples = await gen.generate(sampled_content=[], target_count=20)
        for ex in examples:
            available = [p.strip() for p in ex.available_profiles.split(",")]
            assert ex.selected_profile in available

    @pytest.mark.asyncio
    async def test_default_profile_universe_matches_optimizer_trainset(self):
        """The default available_profiles must match the comma-separated
        list ``run_profile_optimization`` builds when constructing
        ``dspy.Example.available_profiles`` — otherwise synthetic demos
        encode a different profile universe than production training
        examples and the optimizer treats them as out-of-distribution."""
        gen = ProfileGenerator()
        examples = await gen.generate(sampled_content=[], target_count=1)
        expected = ",".join(
            [
                "video_colpali_smol500_mv_frame",
                "video_colqwen_omni_mv_chunk_30s",
                "video_videoprism_base_mv_chunk_30s",
                "video_videoprism_large_mv_chunk_30s",
            ]
        )
        assert examples[0].available_profiles == expected

    @pytest.mark.asyncio
    async def test_modality_complexity_intent_align_with_profile(self):
        gen = ProfileGenerator()
        examples = await gen.generate(sampled_content=[], target_count=50)
        for ex in examples:
            traits = ProfileGenerator.PROFILE_TRAITS.get(ex.selected_profile)
            if traits is None:
                traits = ProfileGenerator.DEFAULT_TRAITS
            assert ex.modality == traits["modality"]
            assert ex.complexity == traits["complexity"]
            assert ex.query_intent == traits["intent"]

    @pytest.mark.asyncio
    async def test_uses_topics_from_sampled_content(self):
        gen = ProfileGenerator()
        sampled = [{"title": "quantum computing applications"}]
        examples = await gen.generate(sampled_content=sampled, target_count=20)
        # With a single-topic sampled list the default seed list is
        # bypassed, so every query must reference the sampled topic.
        assert all("quantum computing" in ex.query.lower() for ex in examples)

    @pytest.mark.asyncio
    async def test_invalid_target_count_raises(self):
        gen = ProfileGenerator()
        with pytest.raises(ValueError):
            await gen.generate(sampled_content=[], target_count=0)

    @pytest.mark.asyncio
    async def test_accepts_custom_profile_list_via_kwargs(self):
        gen = ProfileGenerator()
        custom = ["custom_profile_a", "custom_profile_b"]
        examples = await gen.generate(
            sampled_content=[],
            target_count=10,
            available_profiles=custom,
        )
        for ex in examples:
            assert ex.selected_profile in custom
            assert ex.available_profiles == ",".join(custom)
            # Unknown profiles fall through to the default trait set.
            assert ex.modality == ProfileGenerator.DEFAULT_TRAITS["modality"]
            assert ex.complexity == ProfileGenerator.DEFAULT_TRAITS["complexity"]
            assert ex.query_intent == ProfileGenerator.DEFAULT_TRAITS["intent"]

    @pytest.mark.asyncio
    async def test_confidence_bounded(self):
        gen = ProfileGenerator()
        examples = await gen.generate(sampled_content=[], target_count=30)
        for ex in examples:
            assert 0.0 <= ex.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_reasoning_references_modality_and_complexity(self):
        gen = ProfileGenerator()
        examples = await gen.generate(sampled_content=[], target_count=10)
        for ex in examples:
            assert ex.modality in ex.reasoning
            assert ex.complexity in ex.reasoning
