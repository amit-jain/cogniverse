"""Profile generation labels come from the production selection agent."""

import asyncio

import dspy
import pytest

from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_synthetic.generators import ProfileGenerator
from cogniverse_synthetic.registry import get_optimizer_config
from tests.utils.memory_store import InMemoryConfigStore

PROFILE_CONFIGS = {
    "audio_semantic": {
        "type": "audio",
        "schema_name": "audio_segments",
        "embedding_type": "multi_vector",
        "pipeline_config": {"transcribe_audio": True},
    },
    "document_semantic": {
        "type": "document",
        "schema_name": "document_pages",
        "embedding_type": "single_vector",
        "pipeline_config": {},
    },
}
SAMPLED_CONTENT = [
    {"title": "quantum computing applications", "schema_name": "document_pages"}
]


async def _select_document(query: str, profiles: list[str], tenant_id: str):
    assert query == "find quantum computing applications in document content"
    assert profiles == ["audio_semantic", "document_semantic"]
    assert tenant_id == "test:unit"
    return {
        "query": query,
        "selected_profile": "document_semantic",
        "confidence": 0.97,
        "reasoning": "The production selector chose the document index.",
        "query_intent": "document_search",
        "modality": "document",
        "complexity": "medium",
        "alternatives": [],
    }


class TestProfileGenerator:
    @pytest.mark.asyncio
    async def test_default_request_count_uses_one_hundred_unique_source_queries(self):
        target_count = get_optimizer_config("profile").default_generation_count
        sampled_content = [
            {
                "title": f"canonical source query {index:03d}",
                "schema_name": "document_pages",
            }
            for index in range(target_count + 5)
        ]
        observed_queries = []

        async def label_document(query, profiles, tenant_id):
            observed_queries.append(query)
            return {
                "query": query,
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose document retrieval.",
                "query_intent": "document_search",
                "modality": "document",
                "complexity": "medium",
            }

        examples = await ProfileGenerator(profile_labeler=label_document).generate(
            sampled_content=sampled_content,
            target_count=target_count,
            profile_configs=PROFILE_CONFIGS,
            tenant_id="test:unit",
        )

        expected_queries = [
            f"find canonical source query {index:03d} in document content"
            for index in range(target_count)
        ]
        assert [example.query for example in examples] == expected_queries
        assert observed_queries == expected_queries
        assert len(set(observed_queries)) == target_count == 100

    @pytest.mark.asyncio
    async def test_returns_floor_sized_result_when_target_exceeds_capacity(self):
        generator = ProfileGenerator(profile_labeler=_select_document)

        examples = await generator.generate(
            sampled_content=SAMPLED_CONTENT,
            target_count=5,
            profile_configs=PROFILE_CONFIGS,
            tenant_id="test:unit",
        )

        assert [example.model_dump() for example in examples] == [
            {
                "query": "find quantum computing applications in document content",
                "available_profiles": "audio_semantic,document_semantic",
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose the document index.",
                "query_intent": "document_search",
                "modality": "document",
                "complexity": "medium",
            }
        ]

    @pytest.mark.asyncio
    async def test_cross_modal_rejects_target_larger_than_unique_combinations(self):
        calls = []

        async def label_cross_modal(query: str, profiles: list[str], tenant_id: str):
            calls.append(query)
            return {
                "query": query,
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose document retrieval.",
                "query_intent": "multi_modal_search",
                "modality": "document",
                "complexity": "complex",
            }

        examples = await ProfileGenerator(profile_labeler=label_cross_modal).generate(
            sampled_content=[
                {"topic": "Curie lecture", "schema_name": "audio_segments"},
                {"topic": "Radium notes", "schema_name": "document_pages"},
            ],
            target_count=3,
            profile_configs=PROFILE_CONFIGS,
            tenant_id="test:unit",
            cross_modal=True,
        )

        assert calls == [
            "find Curie lecture in audio content together with Radium notes in document content",
            "find Radium notes in document content together with Curie lecture in audio content",
        ]
        assert [example.model_dump() for example in examples] == [
            {
                "query": (
                    "find Curie lecture in audio content together with "
                    "Radium notes in document content"
                ),
                "available_profiles": "audio_semantic,document_semantic",
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose document retrieval.",
                "query_intent": "multi_modal_search",
                "modality": "document",
                "complexity": "complex",
            },
            {
                "query": (
                    "find Radium notes in document content together with "
                    "Curie lecture in audio content"
                ),
                "available_profiles": "audio_semantic,document_semantic",
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose document retrieval.",
                "query_intent": "multi_modal_search",
                "modality": "document",
                "complexity": "complex",
            },
        ]

    @pytest.mark.asyncio
    async def test_labels_through_profile_selection_agent_process(self):
        config_manager = ConfigManager(store=InMemoryConfigStore())
        for profile_name, profile_config in PROFILE_CONFIGS.items():
            config_manager.add_backend_profile(
                BackendProfileConfig.from_dict(profile_name, profile_config),
                tenant_id="test:unit",
            )
        agent = ProfileSelectionAgent(
            deps=ProfileSelectionDeps(
                available_profiles=["audio_semantic", "document_semantic"]
            )
        )
        agent._config_manager = config_manager
        agent.dspy_module.selector = lambda **_: dspy.Prediction(
            selected_profile="document_semantic",
            confidence="0.93",
            reasoning="The deployed selector chose document retrieval.",
            query_intent="document_search",
            modality="document",
            complexity="complex",
        )

        async def label_with_agent(query: str, profiles: list[str], tenant_id: str):
            return await agent.process(
                ProfileSelectionInput(
                    query=query,
                    available_profiles=profiles,
                    tenant_id=tenant_id,
                )
            )

        examples = await ProfileGenerator(profile_labeler=label_with_agent).generate(
            sampled_content=SAMPLED_CONTENT,
            target_count=1,
            profile_configs=PROFILE_CONFIGS,
            tenant_id="test:unit",
        )

        assert examples[0].model_dump() == {
            "query": "find quantum computing applications in document content",
            "available_profiles": "audio_semantic,document_semantic",
            "selected_profile": "document_semantic",
            "modality": "document",
            "complexity": "complex",
            "query_intent": "document_search",
            "reasoning": "The deployed selector chose document retrieval.",
        }

    @pytest.mark.asyncio
    async def test_generated_modality_matches_selected_wiki_profile_config(self):
        profile_configs = {
            "knowledge_search": {
                "type": "wiki",
                "schema_name": "wiki_pages",
                "embedding_type": "single_vector",
                "pipeline_config": {},
            }
        }

        async def select_wiki(query, profiles, tenant_id):
            assert profiles == ["knowledge_search"]
            return {
                "query": query,
                "selected_profile": "knowledge_search",
                "reasoning": "The production selector chose the configured wiki.",
                "query_intent": "wiki_search",
                "modality": "wiki",
                "complexity": "medium",
            }

        examples = await ProfileGenerator(profile_labeler=select_wiki).generate(
            sampled_content=[
                {
                    "title": "quantum computing applications",
                    "schema_name": "wiki_pages",
                }
            ],
            target_count=1,
            profile_configs=profile_configs,
            tenant_id="test:unit",
        )

        assert examples[0].selected_profile == "knowledge_search"
        assert examples[0].modality == "wiki"

    @pytest.mark.asyncio
    async def test_rejects_modality_that_disagrees_with_selected_profile_config(self):
        async def select_wrong_modality(query, profiles, tenant_id):
            return {
                "query": query,
                "selected_profile": "document_semantic",
                "reasoning": "The selector emitted a conflicting modality.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            }

        with pytest.raises(
            ValueError,
            match=(
                "ProfileGenerator generated 0 unique grounded examples "
                "but target_count=1; source_context=1 unique source topics"
            ),
        ):
            await ProfileGenerator(profile_labeler=select_wrong_modality).generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
            )

    @pytest.mark.asyncio
    async def test_concurrent_tenants_do_not_share_selection_labels(self):
        both_started = asyncio.Event()
        calls: list[str] = []

        async def label_by_tenant(query: str, profiles: list[str], tenant_id: str):
            calls.append(tenant_id)
            if len(calls) == 2:
                both_started.set()
            await both_started.wait()
            selected = {
                "tenant:audio": "audio_semantic",
                "tenant:document": "document_semantic",
            }[tenant_id]
            return {
                "query": query,
                "selected_profile": selected,
                "reasoning": f"Selected for {tenant_id}",
                "query_intent": f"{PROFILE_CONFIGS[selected]['type']}_search",
                "modality": PROFILE_CONFIGS[selected]["type"],
                "complexity": "simple",
            }

        generator = ProfileGenerator(profile_labeler=label_by_tenant)
        audio_examples, document_examples = await asyncio.gather(
            generator.generate(
                sampled_content=[
                    {"title": "shared topic", "schema_name": "document_pages"}
                ],
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="tenant:audio",
            ),
            generator.generate(
                sampled_content=[
                    {"title": "shared topic", "schema_name": "document_pages"}
                ],
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="tenant:document",
            ),
        )

        assert audio_examples[0].selected_profile == "audio_semantic"
        assert audio_examples[0].reasoning == "Selected for tenant:audio"
        assert document_examples[0].selected_profile == "document_semantic"
        assert document_examples[0].reasoning == "Selected for tenant:document"

    @pytest.mark.asyncio
    async def test_selection_failure_raises_with_tenant_and_query_context(self):
        async def fail_selection(query: str, profiles: list[str], tenant_id: str):
            raise ConnectionError("selector unavailable")

        generator = ProfileGenerator(profile_labeler=fail_selection)

        with pytest.raises(RuntimeError) as error:
            await generator.generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
            )

        assert str(error.value) == (
            "profile optimizer callback profile_labeler failed: "
            "Profile selection failed for tenant='test:unit' "
            "query='find quantum computing applications in document content': "
            "selector unavailable"
        )
        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "selector unavailable"

    @pytest.mark.asyncio
    async def test_cross_modal_query_uses_exact_production_selection(self):
        calls = []

        async def label_cross_modal(query: str, profiles: list[str], tenant_id: str):
            calls.append((query, profiles, tenant_id))
            return {
                "query": query,
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose document retrieval.",
                "query_intent": "multi_modal_search",
                "modality": "document",
                "complexity": "complex",
            }

        generator = ProfileGenerator(profile_labeler=label_cross_modal)
        examples = await generator.generate(
            sampled_content=[
                {"topic": "Curie lecture", "schema_name": "audio_segments"},
                {"topic": "Radium notes", "schema_name": "document_pages"},
            ],
            target_count=2,
            profile_configs=PROFILE_CONFIGS,
            tenant_id="test:unit",
            cross_modal=True,
        )

        assert [example.model_dump() for example in examples] == [
            {
                "query": (
                    "find Curie lecture in audio content together with "
                    "Radium notes in document content"
                ),
                "available_profiles": "audio_semantic,document_semantic",
                "selected_profile": "document_semantic",
                "modality": "document",
                "complexity": "complex",
                "query_intent": "multi_modal_search",
                "reasoning": "The production selector chose document retrieval.",
            },
            {
                "query": (
                    "find Radium notes in document content together with "
                    "Curie lecture in audio content"
                ),
                "available_profiles": "audio_semantic,document_semantic",
                "selected_profile": "document_semantic",
                "modality": "document",
                "complexity": "complex",
                "query_intent": "multi_modal_search",
                "reasoning": "The production selector chose document retrieval.",
            },
        ]
        assert calls == [
            (
                example.query,
                ["audio_semantic", "document_semantic"],
                "test:unit",
            )
            for example in examples
        ]

    @pytest.mark.asyncio
    async def test_requires_production_profile_labeler(self):
        generator = ProfileGenerator()

        with pytest.raises(
            ValueError, match="ProfileGenerator requires a production profile_labeler"
        ):
            await generator.generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
            )

    @pytest.mark.parametrize(
        ("field", "value", "_message"),
        [
            ("reasoning", " padded reasoning ", "reasoning must not contain"),
            ("query_intent", " video_search ", "query_intent must not contain"),
            ("modality", " video ", "modality must not contain"),
            ("complexity", " medium ", "complexity must not contain"),
            ("modality", "spatial", "has unsupported modality 'spatial'"),
            ("complexity", "extreme", "has unsupported complexity 'extreme'"),
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_noncanonical_production_selector_values(
        self, field, value, _message
    ):
        async def label_profile(query, profiles, tenant_id):
            output = {
                "query": query,
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose document retrieval.",
                "query_intent": "document_search",
                "modality": "document",
                "complexity": "medium",
            }
            output[field] = value
            return output

        with pytest.raises(
            ValueError,
            match=(
                "ProfileGenerator generated 0 unique grounded examples "
                "but target_count=1; source_context=1 unique source topics"
            ),
        ):
            await ProfileGenerator(profile_labeler=label_profile).generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
            )

    @pytest.mark.asyncio
    async def test_requires_deployed_profile_configs(self):
        generator = ProfileGenerator(profile_labeler=_select_document)

        with pytest.raises(
            ValueError, match="ProfileGenerator requires deployed profile_configs"
        ):
            await generator.generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                tenant_id="test:unit",
            )

    @pytest.mark.asyncio
    async def test_rejects_sample_without_profile_topic(self):
        generator = ProfileGenerator(profile_labeler=_select_document)

        with pytest.raises(
            ValueError, match="sampled_content contains no usable profile topic"
        ):
            await generator.generate(
                sampled_content=[{"schema_name": "audio_segments"}],
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
            )


VIDEO_PROFILE_CONFIGS = {
    "video_frames_mv": {
        "type": "video",
        "schema_name": "video_frames",
        "embedding_type": "multi_vector",
        "pipeline_config": {"extract_keyframes": True, "generate_descriptions": True},
    }
}
FRAME_DESCRIPTION_A = (
    "This video frame captures an outdoor rodeo arena with metal bleachers today"
)
FRAME_DESCRIPTION_B = (
    "This video frame captures an indoor stage with red curtains and lights"
)
SAME_VIDEO_FRAMES = [
    {
        "topic": "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691",
        "description": FRAME_DESCRIPTION_A,
        "schema_name": "video_frames",
    },
    {
        "topic": "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691",
        "description": FRAME_DESCRIPTION_B,
        "schema_name": "video_frames",
    },
]


class TestFrameLevelGrounding:
    """Frames of one video are distinct grounded sources, not one topic."""

    @pytest.mark.asyncio
    async def test_frames_of_one_video_yield_one_templated_example_each(self):
        observed = []

        async def label_video(query, profiles, tenant_id):
            observed.append(query)
            return {
                "query": query,
                "selected_profile": "video_frames_mv",
                "reasoning": "The production selector chose keyframe retrieval.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "complex",
            }

        examples = await ProfileGenerator(profile_labeler=label_video).generate(
            sampled_content=SAME_VIDEO_FRAMES,
            target_count=2,
            profile_configs=VIDEO_PROFILE_CONFIGS,
            tenant_id="flywheel_org:production",
        )

        expected_queries = [
            f"find a video frame showing {FRAME_DESCRIPTION_A}",
            f"find a video frame showing {FRAME_DESCRIPTION_B}",
        ]
        assert [example.query for example in examples] == expected_queries
        assert observed == expected_queries
        assert len(examples) == 2
        assert [example.selected_profile for example in examples] == [
            "video_frames_mv",
            "video_frames_mv",
        ]

    @pytest.mark.asyncio
    async def test_shared_description_prefix_does_not_collapse_distinct_frames(self):
        generator = ProfileGenerator(profile_labeler=_select_document)
        topics = generator._extract_topics(SAME_VIDEO_FRAMES)

        assert topics == [FRAME_DESCRIPTION_A, FRAME_DESCRIPTION_B]
