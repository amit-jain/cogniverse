"""Profile generation labels come from the production selection agent."""

import asyncio
import logging

import dspy
import pytest

from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
    tenant_usable_profile_names,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_synthetic.generators import ProfileGenerator
from cogniverse_synthetic.registry import get_optimizer_config
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
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
    {"title": "quantum computing applications", "schema_name": "document_pages"},
    {"title": "machine learning algorithms", "schema_name": "document_pages"},
]
TENANT_ID = "flywheel_org:production"
SERVING_PROFILE_CONFIGS = {
    "audio_clap_semantic": {
        "type": "audio",
        "schema_name": "audio_clap_semantic",
        "embedding_type": "single_vector",
        "pipeline_config": {"transcribe_audio": True},
        "inference_services": {"embedding": "audio_embedding"},
    },
    "document_text_semantic": {
        "type": "document",
        "schema_name": "document_text_semantic",
        "embedding_type": "single_vector",
        "pipeline_config": {},
        "inference_services": {"embedding": "document_text_embedding"},
    },
    "document_visual_colpali": {
        "type": "document",
        "schema_name": "document_visual_colpali",
        "embedding_type": "single_vector",
        "pipeline_config": {},
        "inference_services": {"embedding": "document_visual_embedding"},
    },
    "image_colpali_mv": {
        "type": "image",
        "schema_name": "image_colpali_mv",
        "embedding_type": "single_vector",
        "pipeline_config": {},
        "inference_services": {"embedding": "image_embedding"},
    },
    "video_colpali_smol500_mv_frame": {
        "type": "video",
        "schema_name": "video_colpali_smol500_mv_frame",
        "embedding_type": "single_vector",
        "pipeline_config": {"extract_keyframes": True},
        "inference_services": {"embedding": "video_embedding"},
    },
}
GROUNDABLE_PROFILE_CONFIGS = {
    "wiki_semantic": {
        "type": "wiki",
        "schema_name": "wiki_semantic",
        "embedding_type": "single_vector",
        "pipeline_config": {},
        "inference_services": {"embedding": "wiki_embedding"},
    },
    **SERVING_PROFILE_CONFIGS,
}


def _profile_generation_config_manager(
    profile_configs: dict[str, dict[str, object]],
    tenant_id: str = "test:unit",
) -> ConfigManager:
    config_manager = ConfigManager(store=InMemoryConfigStore())
    service_urls = {}
    for profile_name, profile_config in profile_configs.items():
        inference_services = profile_config.get("inference_services")
        if isinstance(inference_services, dict):
            embedding_service = inference_services.get("embedding")
            if isinstance(embedding_service, str) and embedding_service.strip():
                service_urls[embedding_service] = f"http://{embedding_service}.invalid"
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(profile_name, profile_config),
            tenant_id=tenant_id,
        )
    config_manager.set_system_config(SystemConfig(inference_service_urls=service_urls))
    return config_manager


PROFILE_GENERATION_CONFIG_MANAGER = _profile_generation_config_manager(PROFILE_CONFIGS)


def _profile_selection_example(
    query: str,
    available_profiles: str,
    selected_profile: str,
    reasoning: str,
) -> dict[str, str]:
    profile_type = SERVING_PROFILE_CONFIGS[selected_profile]["type"]
    return {
        "query": query,
        "available_profiles": available_profiles,
        "selected_profile": selected_profile,
        "reasoning": reasoning,
        "query_intent": (
            "document_search"
            if profile_type == "document"
            else f"{profile_type}_search"
        ),
        "modality": profile_type,
        "complexity": "simple",
    }


async def _select_document(query: str, profiles: list[str], tenant_id: str):
    assert query in (
        "find quantum computing applications in document content",
        "find machine learning algorithms in document content",
    )
    assert profiles == ["document_semantic", "audio_semantic"]
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


class TestProfileGeneratorTenantPool:
    @pytest.mark.asyncio
    async def test_profile_generator_stamps_serving_pool_and_skips_unregistered_wiki(
        self, caplog
    ):
        config_manager = _profile_generation_config_manager(
            SERVING_PROFILE_CONFIGS,
            tenant_id=TENANT_ID,
        )
        expected_pool = tenant_usable_profile_names(config_manager, TENANT_ID)
        expected_available_profiles = ",".join(expected_pool)
        assert expected_pool == [
            "video_colpali_smol500_mv_frame",
            "document_text_semantic",
            "document_visual_colpali",
            "image_colpali_mv",
            "audio_clap_semantic",
        ]

        async def label_profile(query: str, profiles: list[str], tenant_id: str):
            selected_profile = {
                "find curie lecture in an audio transcript": "audio_clap_semantic",
                "find radium notes in document content": "document_text_semantic",
                "find visual diagram in document content": "document_visual_colpali",
                "find image content about microscope image": "image_colpali_mv",
                "find a video frame showing launch footage": (
                    "video_colpali_smol500_mv_frame"
                ),
                "find wiki content about orphaned wiki": "audio_clap_semantic",
            }[query]
            profile_type = SERVING_PROFILE_CONFIGS[selected_profile]["type"]
            return {
                "query": query,
                "selected_profile": selected_profile,
                "reasoning": f"Selected {selected_profile}.",
                "query_intent": (
                    "document_search"
                    if profile_type == "document"
                    else f"{profile_type}_search"
                ),
                "modality": profile_type,
                "complexity": "simple",
            }

        generator = ProfileGenerator(profile_labeler=label_profile)
        with caplog.at_level(logging.WARNING):
            examples = await generator.generate(
                sampled_content=[
                    {"title": "orphaned wiki", "schema_name": "wiki_semantic"},
                    {
                        "title": "curie lecture",
                        "schema_name": "audio_clap_semantic",
                    },
                    {
                        "title": "radium notes",
                        "schema_name": "document_text_semantic",
                    },
                    {
                        "title": "visual diagram",
                        "schema_name": "document_visual_colpali",
                    },
                    {
                        "title": "microscope image",
                        "schema_name": "image_colpali_mv",
                    },
                    {
                        "title": "launch footage",
                        "schema_name": "video_colpali_smol500_mv_frame",
                    },
                ],
                target_count=5,
                profile_configs=GROUNDABLE_PROFILE_CONFIGS,
                tenant_id=TENANT_ID,
                config_manager=config_manager,
            )

        assert [example.model_dump() for example in examples] == [
            _profile_selection_example(
                "find curie lecture in an audio transcript",
                expected_available_profiles,
                "audio_clap_semantic",
                "Selected audio_clap_semantic.",
            ),
            _profile_selection_example(
                "find radium notes in document content",
                expected_available_profiles,
                "document_text_semantic",
                "Selected document_text_semantic.",
            ),
            _profile_selection_example(
                "find visual diagram in document content",
                expected_available_profiles,
                "document_visual_colpali",
                "Selected document_visual_colpali.",
            ),
            _profile_selection_example(
                "find image content about microscope image",
                expected_available_profiles,
                "image_colpali_mv",
                "Selected image_colpali_mv.",
            ),
            _profile_selection_example(
                "find a video frame showing launch footage",
                expected_available_profiles,
                "video_colpali_smol500_mv_frame",
                "Selected video_colpali_smol500_mv_frame.",
            ),
        ]
        assert [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ] == [
            "ProfileGenerator skips non-qualifying backend profiles for tenant "
            "'flywheel_org:production': wiki_semantic (not tenant-usable)"
        ]

    @pytest.mark.asyncio
    async def test_profile_generator_raises_when_intersection_is_empty(self, caplog):
        config_manager = _profile_generation_config_manager(
            {"audio_clap_semantic": SERVING_PROFILE_CONFIGS["audio_clap_semantic"]},
            tenant_id=TENANT_ID,
        )

        async def label_profile(*_args, **_kwargs):
            pytest.fail("label_profile should not run when no usable profiles remain")

        generator = ProfileGenerator(profile_labeler=label_profile)
        with caplog.at_level(logging.WARNING):
            with pytest.raises(
                ValueError,
                match=(
                    "^ProfileGenerator requires at least one qualifying backend "
                    "profile for tenant 'flywheel_org:production'; excluded "
                    "profiles: wiki_semantic \\(not tenant-usable\\)$"
                ),
            ):
                await generator.generate(
                    sampled_content=[
                        {"title": "orphaned wiki", "schema_name": "wiki_semantic"}
                    ],
                    target_count=1,
                    profile_configs={
                        "wiki_semantic": GROUNDABLE_PROFILE_CONFIGS["wiki_semantic"]
                    },
                    tenant_id=TENANT_ID,
                    config_manager=config_manager,
                )

        assert caplog.records == []


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
            config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
        )

        # Saliency selects the unique part of each title — the number
        expected_queries = [
            f"find {index:03d} in document content" for index in range(target_count)
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
            config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
        )

        assert [example.model_dump() for example in examples] == [
            {
                "query": "find quantum computing applications in document content",
                "available_profiles": "document_semantic,audio_semantic",
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose the document index.",
                "query_intent": "document_search",
                "modality": "document",
                "complexity": "medium",
            },
            {
                "query": "find machine learning algorithms in document content",
                "available_profiles": "document_semantic,audio_semantic",
                "selected_profile": "document_semantic",
                "reasoning": "The production selector chose the document index.",
                "query_intent": "document_search",
                "modality": "document",
                "complexity": "medium",
            },
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
            config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
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
                "available_profiles": "document_semantic,audio_semantic",
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
                "available_profiles": "document_semantic,audio_semantic",
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
        agent.telemetry_manager = RecordingTelemetryManager()
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
            config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
        )

        assert examples[0].model_dump() == {
            "query": "find quantum computing applications in document content",
            "available_profiles": "document_semantic,audio_semantic",
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

        config_manager = _profile_generation_config_manager(profile_configs)
        examples = await ProfileGenerator(profile_labeler=select_wiki).generate(
            sampled_content=[
                {
                    "title": "quantum computing applications",
                    "schema_name": "wiki_pages",
                },
                {
                    "title": "neural network architectures",
                    "schema_name": "wiki_pages",
                },
            ],
            target_count=1,
            profile_configs=profile_configs,
            tenant_id="test:unit",
            config_manager=config_manager,
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
                "but target_count=1; source_context=2 unique source topics"
            ),
        ):
            await ProfileGenerator(profile_labeler=select_wrong_modality).generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
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
        # Both batches need 2+ records for saliency
        sampled_batch = [
            {"title": "shared topic", "schema_name": "document_pages"},
            {"title": "other content", "schema_name": "document_pages"},
        ]
        audio_config_manager = _profile_generation_config_manager(
            PROFILE_CONFIGS,
            tenant_id="tenant:audio",
        )
        document_config_manager = _profile_generation_config_manager(
            PROFILE_CONFIGS,
            tenant_id="tenant:document",
        )
        audio_examples, document_examples = await asyncio.gather(
            generator.generate(
                sampled_content=sampled_batch,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="tenant:audio",
                config_manager=audio_config_manager,
            ),
            generator.generate(
                sampled_content=sampled_batch,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="tenant:document",
                config_manager=document_config_manager,
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
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
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
            config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
            cross_modal=True,
        )

        assert [example.model_dump() for example in examples] == [
            {
                "query": (
                    "find Curie lecture in audio content together with "
                    "Radium notes in document content"
                ),
                "available_profiles": "document_semantic,audio_semantic",
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
                "available_profiles": "document_semantic,audio_semantic",
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
                ["document_semantic", "audio_semantic"],
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
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
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
                "but target_count=1; source_context=2 unique source topics"
            ),
        ):
            await ProfileGenerator(profile_labeler=label_profile).generate(
                sampled_content=SAMPLED_CONTENT,
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
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

        # Both records lack topic text, so saliency raises with "got 0"
        with pytest.raises(
            ValueError,
            match="topic saliency requires at least 2 sampled records with topic text; got 0",
        ):
            await generator.generate(
                sampled_content=[
                    {"schema_name": "audio_segments"},
                    {"schema_name": "document_pages"},
                ],
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
            )


VIDEO_PROFILE_CONFIGS = {
    "video_frames_mv": {
        "type": "video",
        "schema_name": "video_frames",
        "embedding_type": "multi_vector",
        "pipeline_config": {"extract_keyframes": True, "generate_descriptions": True},
    }
}
VIDEO_PROFILE_GENERATION_CONFIG_MANAGER = _profile_generation_config_manager(
    VIDEO_PROFILE_CONFIGS,
    tenant_id=TENANT_ID,
)
FRAME_DESCRIPTION_A = (
    "This video frame captures an outdoor rodeo arena with metal bleachers today"
)
FRAME_DESCRIPTION_B = (
    "This video frame captures an indoor stage with red curtains and lights"
)
# Saliency extracts the distinctive span, excluding shared "This video frame captures"
FRAME_TOPIC_A = "outdoor rodeo arena with metal bleachers"
FRAME_TOPIC_B = "indoor stage with red curtains"
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
            config_manager=VIDEO_PROFILE_GENERATION_CONFIG_MANAGER,
        )

        expected_queries = [
            f"find a video frame showing {FRAME_TOPIC_A}",
            f"find a video frame showing {FRAME_TOPIC_B}",
        ]
        assert [example.query for example in examples] == expected_queries
        assert observed == expected_queries
        assert len(examples) == 2
        assert [example.selected_profile for example in examples] == [
            "video_frames_mv",
            "video_frames_mv",
        ]

    def test_shared_description_prefix_does_not_collapse_distinct_frames(self):
        from cogniverse_synthetic.topics import TopicSaliency, extract_topic

        saliency = TopicSaliency.from_records(SAME_VIDEO_FRAMES)
        topics = [extract_topic(r, saliency=saliency) for r in SAME_VIDEO_FRAMES]

        # Saliency extracts the distinctive span of each description
        assert topics == [FRAME_TOPIC_A, FRAME_TOPIC_B]


class TestCrossModalSelectedProfileBinding:
    @pytest.mark.asyncio
    async def test_cross_modal_labeler_choice_is_bound_to_selected_profiles(self):
        calls = []

        async def label_cross_modal(query: str, profiles: list[str], tenant_id: str):
            calls.append(list(profiles))
            return {
                "query": query,
                "selected_profile": "audio_semantic",
                "reasoning": "Audio retrieval grounds this cross-modal query.",
                "query_intent": "multi_modal_search",
                "modality": "audio",
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
            config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
            cross_modal=True,
            selected_profiles=["audio_semantic"],
        )

        assert calls == [["audio_semantic"], ["audio_semantic"]]
        assert [
            (example.selected_profile, example.available_profiles)
            for example in examples
        ] == [
            ("audio_semantic", "document_semantic,audio_semantic"),
            ("audio_semantic", "document_semantic,audio_semantic"),
        ]

    @pytest.mark.asyncio
    async def test_cross_modal_rejects_label_outside_selected_profiles(self):
        async def label_outside(query: str, profiles: list[str], tenant_id: str):
            return {
                "query": query,
                "selected_profile": "document_semantic",
                "reasoning": "The selector escaped its constrained vocabulary.",
                "query_intent": "multi_modal_search",
                "modality": "document",
                "complexity": "complex",
            }

        generator = ProfileGenerator(profile_labeler=label_outside)
        with pytest.raises(ValueError) as excinfo:
            await generator.generate(
                sampled_content=[
                    {"topic": "Curie lecture", "schema_name": "audio_segments"},
                    {"topic": "Radium notes", "schema_name": "document_pages"},
                ],
                target_count=2,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
                cross_modal=True,
                selected_profiles=["audio_semantic"],
            )

        assert (
            "ProfileGenerator generated 0 unique grounded examples "
            "but target_count=2" in str(excinfo.value)
        )
        cause_messages = []
        cause = excinfo.value.__cause__
        while cause is not None:
            cause_messages.append(str(cause))
            cause = cause.__cause__
        assert (
            "profile selection selected_profile must be one of the "
            "selected profiles offered to the labeler"
        ) in cause_messages

    @pytest.mark.asyncio
    async def test_selected_profiles_must_be_usable_tenant_profiles(self):
        generator = ProfileGenerator(profile_labeler=_select_document)
        with pytest.raises(ValueError, match="usable tenant profile"):
            await generator.generate(
                sampled_content=[
                    {"topic": "Curie lecture", "schema_name": "audio_segments"},
                    {"topic": "Radium notes", "schema_name": "document_pages"},
                ],
                target_count=1,
                profile_configs=PROFILE_CONFIGS,
                tenant_id="test:unit",
                config_manager=PROFILE_GENERATION_CONFIG_MANAGER,
                cross_modal=True,
                selected_profiles=["never_deployed_profile"],
            )
