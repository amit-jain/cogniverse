"""
Integration tests for all synthetic data generators
"""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    BackendProfileConfig,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
)
from cogniverse_sdk.interfaces.workflow_store import (
    WorkflowExecution,
    WorkflowLearningState,
    WorkflowTemplate,
)
from cogniverse_synthetic.generators import (
    ProfileGenerator,
    QueryEnhancementGenerator,
    RoutingGenerator,
    WorkflowGenerator,
)
from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    RoutingExperienceSchema,
    SyntheticDataResponse,
    WorkflowExecutionSchema,
)
from cogniverse_synthetic.utils import AgentInferrer, PatternExtractor

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SHIPPED_BACKEND_PROFILES = json.loads(
    (_REPO_ROOT / "configs" / "config.json").read_text()
)["backend"]["profiles"]


def _shipped_backend_profile(profile_name: str) -> BackendProfileConfig:
    return BackendProfileConfig.from_dict(
        profile_name, _SHIPPED_BACKEND_PROFILES[profile_name]
    )


def create_routing_config():
    """Create test configuration for routing generator with mock DSPy"""
    return OptimizerGenerationConfig(
        optimizer_type="routing",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                module_type="Predict",
            )
        },
    )


UNOBSERVED_WORKFLOW_SEMANTICS = {
    "execution_time": "unobserved_zero_sentinel",
    "success": "unobserved_false_sentinel",
    "parallel_efficiency": "unobserved_zero_sentinel",
    "confidence_score": "unobserved_zero_sentinel",
}

OBSERVED_WORKFLOW_SEMANTICS = {
    "execution_time": "observed_duration_seconds",
    "success": "observed_execution_outcome",
    "parallel_efficiency": "observed_parallel_efficiency",
    "confidence_score": "observed_confidence_score",
}

CONFIGURED_AGENTS = {
    "search_agent": {
        "enabled": True,
        "modalities": ["VIDEO"],
        "capabilities": ["video_search"],
    },
    "document_agent": {
        "enabled": True,
        "modalities": ["DOCUMENT"],
        "capabilities": ["document_analysis"],
    },
    "image_search_agent": {
        "enabled": True,
        "modalities": ["IMAGE"],
        "capabilities": ["image_search"],
    },
    "audio_analysis_agent": {
        "enabled": True,
        "modalities": ["AUDIO"],
        "capabilities": ["audio_analysis"],
    },
    "coding_agent": {
        "enabled": True,
        "modalities": ["CODE"],
        "capabilities": ["coding"],
    },
    "wiki_agent": {
        "enabled": True,
        "modalities": ["WIKI"],
        "capabilities": ["document_analysis"],
    },
    "summarizer_agent": {
        "enabled": True,
        "modalities": [],
        "capabilities": ["summarization"],
    },
    "detailed_report_agent": {
        "enabled": True,
        "modalities": [],
        "capabilities": ["detailed_report"],
    },
}


def configured_agent_inferrer() -> AgentInferrer:
    return AgentInferrer(
        agents_config=CONFIGURED_AGENTS,
        agent_mappings=[
            AgentMappingRule(modality="VIDEO", agent_name="search_agent"),
            AgentMappingRule(modality="DOCUMENT", agent_name="document_agent"),
            AgentMappingRule(modality="IMAGE", agent_name="image_search_agent"),
            AgentMappingRule(modality="AUDIO", agent_name="audio_analysis_agent"),
            AgentMappingRule(modality="CODE", agent_name="coding_agent"),
            AgentMappingRule(modality="WIKI", agent_name="wiki_agent"),
        ],
    )


def video_workflow_sample(topic: str) -> dict[str, str]:
    return {
        "topic": topic,
        "profile_type": "video",
        "modality": "VIDEO",
        "schema_name": "video_videoprism_large_mv_chunk_30s",
    }


async def extract_entities(text: str, tenant_id: str):
    entities = []
    if "TensorFlow" in text:
        entities.append({"text": "TensorFlow", "type": "TECHNOLOGY"})
    return {"query": text, "entities": entities, "relationships": []}


async def route_query(query: str, tenant_id: str):
    return {"query": query, "routed_to": "search_agent", "confidence": 0.73}


async def select_profile(query: str, profiles: list[str], tenant_id: str):
    selected_profile = profiles[0]
    modality = selected_profile.split("_", 1)[0]
    return {
        "query": query,
        "selected_profile": selected_profile,
        "reasoning": f"Production selector chose {selected_profile}.",
        "query_intent": f"{modality}_search",
        "modality": modality,
        "complexity": "medium",
    }


class _BoundaryPatternExtractor:
    def extract(self, records):
        return {"topics": [records[0]["title"]]}


class _BoundaryQueryGenerator:
    max_retries = 3

    def __call__(self, *, topics, entities, entity_types):
        result = SimpleNamespace(
            query=f"find {entities[0]}",
            reasoning=f"Use the grounded {entity_types[0]} entity from {topics}.",
        )
        result._retry_count = 0
        result._max_retries = self.max_retries
        return result


async def _extract_boundary_entity(text: str, tenant_id: str):
    assert tenant_id == "tenant-a"
    return {
        "query": text,
        "entities": [{"text": text, "type": "TOPIC"}],
        "relationships": [],
    }


def _label_callback_query(kind: str, topic: str) -> str:
    if kind == "routing":
        return f"find {topic}"
    if kind == "profile":
        return f"find {topic} in document content"
    return topic


def _label_callback_result(kind: str, query: str) -> dict[str, Any]:
    if kind == "routing":
        return {"routed_to": "video_search", "confidence": 0.91}
    if kind == "profile":
        return {
            "query": query,
            "selected_profile": "document_text_semantic",
            "modality": "document",
            "complexity": "simple",
            "query_intent": "document_search",
            "reasoning": "The configured document profile matches the request.",
        }
    return {
        "original_query": query,
        "enhanced_query": f"{query} magnetic",
        "expansion_terms": ["magnetic"],
        "synonyms": ["eruption"],
        "reasoning": "The production enhancer added a source-grounded term.",
    }


def _expected_label(kind: str, topic: str) -> dict[str, Any]:
    query = _label_callback_query(kind, topic)
    if kind == "routing":
        return {
            "query": query,
            "chosen_agent": "video_search",
            "routing_confidence": 0.91,
        }
    if kind == "profile":
        return {
            "query": query,
            "selected_profile": "document_text_semantic",
            "modality": "document",
            "complexity": "simple",
            "query_intent": "document_search",
            "reasoning": "The configured document profile matches the request.",
        }
    return {
        "query": query,
        "enhanced_query": f"{query} magnetic",
        "expansion_terms": ["magnetic"],
        "synonyms": ["eruption"],
        "reasoning": "The production enhancer added a source-grounded term.",
    }


def _build_label_invoker(kind: str, callback, timeout_seconds: float):
    if kind == "routing":
        generator = RoutingGenerator(
            entity_extractor=_extract_boundary_entity,
            routing_decider=callback,
            pattern_extractor=_BoundaryPatternExtractor(),
            optimizer_config=object(),
            production_label_timeout_seconds=timeout_seconds,
        )
        generator.query_generator = _BoundaryQueryGenerator()

        async def invoke(topic: str):
            examples = await generator.generate(
                [{"title": topic}],
                target_count=1,
                tenant_id="tenant-a",
            )
            example = examples[0]
            return {
                "query": example.query,
                "chosen_agent": example.chosen_agent,
                "routing_confidence": example.routing_confidence,
            }

        return invoke

    if kind == "profile":
        generator = ProfileGenerator(
            profile_labeler=callback,
            production_label_timeout_seconds=timeout_seconds,
        )

        async def invoke(topic: str):
            examples = await generator.generate(
                [{"title": topic}],
                target_count=1,
                tenant_id="tenant-a",
                profile_configs={
                    "document_text_semantic": _shipped_backend_profile(
                        "document_text_semantic"
                    ).to_dict(),
                },
            )
            example = examples[0]
            return {
                "query": example.query,
                "selected_profile": example.selected_profile,
                "modality": example.modality,
                "complexity": example.complexity,
                "query_intent": example.query_intent,
                "reasoning": example.reasoning,
            }

        return invoke

    generator = QueryEnhancementGenerator(
        query_enhancer=callback,
        production_label_timeout_seconds=timeout_seconds,
    )

    async def invoke(topic: str):
        examples = await generator.generate(
            [
                {
                    "topic": topic,
                    "title": f"{topic} magnetic eruptions",
                    "profile_name": "document_text_semantic",
                }
            ],
            target_count=1,
            tenant_id="tenant-a",
        )
        example = examples[0]
        return {
            "query": example.query,
            "enhanced_query": example.enhanced_query,
            "expansion_terms": example.expansion_terms,
            "synonyms": example.synonyms,
            "reasoning": example.reasoning,
        }

    return invoke


def workflow_execution(
    workflow_id: str,
    *,
    observed: bool,
    semantics,
    success: bool = False,
    query: str = "find source",
) -> WorkflowExecution:
    return WorkflowExecution(
        workflow_id=workflow_id,
        query=query,
        query_type="VIDEO",
        execution_time=1.25 if observed else 0.0,
        success=success,
        agent_sequence=["search_agent"],
        task_count=1,
        parallel_efficiency=0.75 if observed else 0.0,
        confidence_score=0.875 if observed else 0.0,
        metadata={
            "_outcome_metadata": {
                "observed": observed,
                "required_field_semantics": semantics,
            }
        },
    )


class TestProfileGeneratorIntegration:
    """Integration tests for production-labelled profile examples."""

    PROFILE_CONFIGS = {
        "audio_clap_semantic": _shipped_backend_profile(
            "audio_clap_semantic"
        ).to_dict(),
        "document_text_semantic": _shipped_backend_profile(
            "document_text_semantic"
        ).to_dict(),
    }

    @pytest.mark.asyncio
    async def test_profile_generator_with_mock_data(self):
        generator = ProfileGenerator(profile_labeler=select_profile)
        video_profile = _shipped_backend_profile("video_videoprism_large_mv_chunk_30s")

        mock_content = [
            {
                "video_title": "Machine Learning Tutorial",
                "schema_name": video_profile.schema_name,
            }
        ]

        examples = await generator.generate(
            sampled_content=mock_content,
            target_count=1,
            profile_configs={video_profile.profile_name: video_profile.to_dict()},
            tenant_id="acme:profiles",
        )

        assert len(examples) == 1
        assert all(isinstance(ex, ProfileSelectionExampleSchema) for ex in examples)
        for ex in examples:
            available = [p.strip() for p in ex.available_profiles.split(",")]
            assert ex.selected_profile in available
            assert ex.modality == "video"
            assert ex.query_intent == "video_search"
            assert ex.query == "find video content about Machine Learning Tutorial"
            assert ex.complexity == "medium"

    @pytest.mark.asyncio
    async def test_profile_generator_uses_configured_audio_traits(self):
        generator = ProfileGenerator(profile_labeler=select_profile)
        examples = await generator.generate(
            sampled_content=[
                {"topic": "Curie lecture", "schema_name": "audio_content"}
            ],
            target_count=1,
            profile_configs={
                "audio_clap_semantic": self.PROFILE_CONFIGS["audio_clap_semantic"]
            },
            tenant_id="acme:profiles",
        )

        assert examples[0].model_dump() == {
            "query": "find Curie lecture in an audio transcript",
            "available_profiles": "audio_clap_semantic",
            "selected_profile": "audio_clap_semantic",
            "reasoning": "Production selector chose audio_clap_semantic.",
            "query_intent": "audio_search",
            "modality": "audio",
            "complexity": "medium",
        }

    @pytest.mark.parametrize("modality", [None, "", "unknown"])
    @pytest.mark.asyncio
    async def test_profile_generator_rejects_missing_or_unknown_modality(
        self, modality
    ):
        generator = ProfileGenerator(profile_labeler=select_profile)

        with pytest.raises(
            ValueError,
            match="Backend profile 'broken' requires a supported non-empty type",
        ):
            await generator.generate(
                sampled_content=[{"topic": "configured profile validation"}],
                target_count=1,
                profile_configs={
                    "broken": {
                        **_shipped_backend_profile("document_text_semantic").to_dict(),
                        "type": modality,
                    }
                },
            )

    @pytest.mark.asyncio
    async def test_cross_modal_query_uses_two_configured_content_modalities(self):
        generator = ProfileGenerator(profile_labeler=select_profile)

        examples = await generator.generate(
            sampled_content=[
                {"topic": "Curie lecture", "schema_name": "audio_content"},
                {"topic": "Radium notes", "schema_name": "document_text"},
            ],
            target_count=1,
            profile_configs=self.PROFILE_CONFIGS,
            tenant_id="acme:profiles",
            cross_modal=True,
        )

        assert [example.model_dump() for example in examples] == [
            {
                "query": (
                    "find Curie lecture in audio content together with "
                    "Radium notes in document content"
                ),
                "available_profiles": "audio_clap_semantic,document_text_semantic",
                "selected_profile": "audio_clap_semantic",
                "reasoning": "Production selector chose audio_clap_semantic.",
                "query_intent": "audio_search",
                "modality": "audio",
                "complexity": "medium",
            }
        ]

    @pytest.mark.asyncio
    async def test_cross_modal_rejects_one_configured_modality(self):
        generator = ProfileGenerator(profile_labeler=select_profile)

        with pytest.raises(
            ValueError,
            match="cross_modal requires at least two configured modalities",
        ):
            await generator.generate(
                sampled_content=[
                    {"topic": "Curie lecture", "schema_name": "audio_content"}
                ],
                target_count=1,
                profile_configs={
                    "audio_clap_semantic": self.PROFILE_CONFIGS["audio_clap_semantic"]
                },
                tenant_id="acme:profiles",
                cross_modal=True,
            )

    @pytest.mark.asyncio
    async def test_cross_modal_rejects_samples_from_one_modality(self):
        generator = ProfileGenerator(profile_labeler=select_profile)

        with pytest.raises(
            ValueError,
            match="cross_modal requires sampled content from at least two modalities",
        ):
            await generator.generate(
                sampled_content=[
                    {"topic": "Curie lecture", "schema_name": "audio_content"}
                ],
                target_count=1,
                profile_configs=self.PROFILE_CONFIGS,
                tenant_id="acme:profiles",
                cross_modal=True,
            )


class TestRoutingGeneratorIntegration:
    """Integration tests for RoutingGenerator"""

    @pytest.mark.asyncio
    async def test_routing_generator(self):
        """Test RoutingGenerator generates valid routing experiences"""
        pattern_extractor = PatternExtractor()

        generator = RoutingGenerator(
            entity_extractor=extract_entities,
            routing_decider=route_query,
            pattern_extractor=pattern_extractor,
            optimizer_config=create_routing_config(),
        )

        mock_content = [
            {
                "title": "TensorFlow Neural Networks Tutorial",
                "video_title": "TensorFlow Neural Networks Tutorial",
                "segment_description": "Learn TensorFlow for deep learning",
                "schema_name": "video_videoprism_large_mv_chunk_30s",
            }
        ]

        examples = await generator.generate(
            sampled_content=mock_content,
            target_count=1,
            tenant_id="acme:routing",
        )

        assert len(examples) == 1
        assert all(isinstance(ex, RoutingExperienceSchema) for ex in examples)
        assert all(len(ex.entities) >= 1 for ex in examples)
        assert all(
            ex.enhanced_query != ex.query for ex in examples
        )  # Should have annotations
        for example in examples:
            assert example.routing_confidence == 0.73
            assert example.search_quality == 0.0
            assert example.agent_success is False
            assert example.user_satisfaction is None
            assert example.processing_time == 0.0
            assert example.reward is None
            assert all(set(entity) == {"text", "type"} for entity in example.entities)
            assert example.metadata["_outcome_metadata"] == {
                "observed": True,
                "required_field_semantics": {
                    "routing_confidence": "observed_gateway_confidence",
                    "search_quality": "unobserved_zero_sentinel",
                    "agent_success": "unobserved_false_sentinel",
                    "processing_time": "unobserved_zero_sentinel",
                },
            }

    @pytest.mark.asyncio
    async def test_nested_entity_extraction_uses_its_own_timeout(self):
        never_released = asyncio.Event()

        async def hung_entity_extractor(text, tenant_id):
            await never_released.wait()

        generator = RoutingGenerator(
            entity_extractor=hung_entity_extractor,
            routing_decider=route_query,
            pattern_extractor=_BoundaryPatternExtractor(),
            optimizer_config=create_routing_config(),
            production_label_timeout_seconds=0.5,
            entity_extraction_timeout_seconds=0.02,
        )
        generator.query_generator = _BoundaryQueryGenerator()

        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(
                generator.generate(
                    [{"title": "solar flares"}],
                    target_count=1,
                    tenant_id="tenant-a",
                ),
                timeout=1.0,
            )

        assert str(raised.value) == (
            "entity extraction timed out after 0.02 seconds for source text "
            "'solar flares'"
        )
        assert isinstance(raised.value.__cause__, TimeoutError)

    @pytest.mark.asyncio
    async def test_dspy_query_generation_uses_gateway_timeout(self):
        started = threading.Event()
        release = threading.Event()

        class _HungQueryGenerator:
            max_retries = 3

            def __call__(self, *, topics, entities, entity_types):
                started.set()
                release.wait(timeout=1.0)

        generator = RoutingGenerator(
            entity_extractor=_extract_boundary_entity,
            routing_decider=route_query,
            pattern_extractor=_BoundaryPatternExtractor(),
            optimizer_config=create_routing_config(),
            production_label_timeout_seconds=0.02,
            entity_extraction_timeout_seconds=0.5,
        )
        generator.query_generator = _HungQueryGenerator()

        try:
            with pytest.raises(TimeoutError) as raised:
                await asyncio.wait_for(
                    generator.generate(
                        [{"title": "solar flares"}],
                        target_count=1,
                        tenant_id="tenant-a",
                    ),
                    timeout=1.0,
                )
        finally:
            release.set()

        assert started.is_set()
        assert str(raised.value) == (
            "routing optimizer DSPy query_generator timed out after 0.02 seconds "
            "for entities: solar flares"
        )
        assert isinstance(raised.value.__cause__, TimeoutError)

    @pytest.mark.parametrize(
        "timeout_seconds",
        [True, 0, -1, float("nan"), float("inf")],
    )
    def test_entity_extraction_timeout_must_be_positive_and_finite(
        self,
        timeout_seconds,
    ):
        with pytest.raises(
            ValueError,
            match=("entity_extraction_timeout_seconds must be finite and positive"),
        ):
            RoutingGenerator(
                entity_extractor=_extract_boundary_entity,
                routing_decider=route_query,
                pattern_extractor=_BoundaryPatternExtractor(),
                optimizer_config=create_routing_config(),
                entity_extraction_timeout_seconds=timeout_seconds,
            )


class TestProductionLabelCallbackBoundary:
    CALLBACK_NAMES = {
        "routing": "routing_decider",
        "profile": "profile_labeler",
        "query_enhancement": "query_enhancer",
    }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", CALLBACK_NAMES)
    async def test_successful_callback_returns_exact_production_label(self, kind):
        calls = []

        async def callback(query, *args):
            calls.append((query, args))
            return _label_callback_result(kind, query)

        invoke = _build_label_invoker(kind, callback, timeout_seconds=0.5)

        result = await invoke("solar flares")

        assert result == _expected_label(kind, "solar flares")
        if kind == "profile":
            expected_args = (["document_text_semantic"], "tenant-a")
        elif kind == "query_enhancement":
            expected_args = (
                "tenant-a",
                "solar flares magnetic eruptions\nsolar flares",
            )
        else:
            expected_args = ("tenant-a",)
        assert calls == [(_label_callback_query(kind, "solar flares"), expected_args)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", CALLBACK_NAMES)
    async def test_hung_callback_raises_contextual_timeout(self, kind):
        never_released = asyncio.Event()

        async def callback(*args):
            await never_released.wait()

        invoke = _build_label_invoker(kind, callback, timeout_seconds=0.02)
        query = _label_callback_query(kind, "solar flares")

        with pytest.raises(TimeoutError) as raised:
            await asyncio.wait_for(invoke("solar flares"), timeout=1.0)

        assert str(raised.value) == (
            f"{kind} optimizer callback {self.CALLBACK_NAMES[kind]} timed out "
            "after 0.02 seconds for tenant='tenant-a' "
            f"query='{query}'"
        )
        assert isinstance(raised.value.__cause__, TimeoutError)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", CALLBACK_NAMES)
    async def test_failing_callback_raises_with_original_cause(self, kind):
        failure = LookupError("label service unavailable")

        async def callback(*args):
            raise failure

        invoke = _build_label_invoker(kind, callback, timeout_seconds=0.5)
        query = _label_callback_query(kind, "solar flares")

        with pytest.raises(RuntimeError) as raised:
            await invoke("solar flares")

        if kind == "profile":
            assert str(raised.value) == (
                "profile optimizer callback profile_labeler failed: "
                "Profile selection failed for tenant='tenant-a' "
                f"query='{query}': label service unavailable"
            )
        else:
            assert str(raised.value) == (
                f"{kind} optimizer callback {self.CALLBACK_NAMES[kind]} failed for "
                f"tenant='tenant-a' query='{query}'"
            )
        assert raised.value.__cause__ is failure

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", CALLBACK_NAMES)
    async def test_concurrent_callbacks_remain_request_local(self, kind):
        topics = ["solar flares", "lunar launch"]
        started = {
            _label_callback_query(kind, topic): asyncio.Event() for topic in topics
        }
        release = {
            _label_callback_query(kind, topic): asyncio.Event() for topic in topics
        }

        async def callback(query, *args):
            started[query].set()
            await release[query].wait()
            return _label_callback_result(kind, query)

        invoke = _build_label_invoker(kind, callback, timeout_seconds=2.0)
        tasks = [asyncio.create_task(invoke(topic)) for topic in topics]
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started.values())),
            timeout=1.0,
        )

        release[_label_callback_query(kind, topics[0])].set()
        assert await asyncio.wait_for(tasks[0], timeout=1.0) == _expected_label(
            kind, topics[0]
        )
        assert tasks[1].done() is False

        release[_label_callback_query(kind, topics[1])].set()
        assert await asyncio.wait_for(tasks[1], timeout=1.0) == _expected_label(
            kind, topics[1]
        )

    @pytest.mark.parametrize("kind", CALLBACK_NAMES)
    @pytest.mark.parametrize(
        "timeout_seconds",
        [True, 0, -1, float("nan"), float("inf")],
    )
    def test_timeout_must_be_positive_and_finite(self, kind, timeout_seconds):
        async def callback(*args):
            return None

        with pytest.raises(
            ValueError,
            match="production_label_timeout_seconds must be finite and positive",
        ):
            _build_label_invoker(kind, callback, timeout_seconds=timeout_seconds)

    @pytest.mark.asyncio
    async def test_query_enhancement_rejects_term_absent_from_sampled_source(self):
        async def unrelated_enhancement(query, tenant_id, source_text):
            assert tenant_id == "tenant-a"
            assert source_text == "solar flares magnetic eruptions\nsolar flares"
            result = _label_callback_result("query_enhancement", query)
            result["expansion_terms"] = ["volcano"]
            return result

        invoke = _build_label_invoker(
            "query_enhancement",
            unrelated_enhancement,
            timeout_seconds=0.5,
        )

        with pytest.raises(ValueError) as raised:
            await invoke("solar flares")

        assert str(raised.value) == (
            "QueryEnhancementGenerator generated 0 unique grounded examples "
            "but target_count=1; source_context=5 unique source-template queries"
        )
        assert str(raised.value.__cause__) == (
            "query_enhancement optimizer callback query_enhancer returned "
            "expansion_terms absent from sampled source for tenant='tenant-a' "
            "query='explain solar flares': ['volcano']"
        )


class TestWorkflowGeneratorIntegration:
    """Integration tests for WorkflowGenerator"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("profile_type", "modality", "expected_agent"),
        [
            ("video", "VIDEO", "search_agent"),
            ("document", "DOCUMENT", "document_agent"),
            ("image", "IMAGE", "image_search_agent"),
            ("audio", "AUDIO", "audio_analysis_agent"),
            ("code", "CODE", "coding_agent"),
            ("wiki", "WIKI", "wiki_agent"),
        ],
    )
    async def test_workflow_uses_canonical_sample_modality(
        self,
        profile_type,
        modality,
        expected_agent,
    ):
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        example = (
            await generator.generate(
                sampled_content=[
                    {
                        "topic": "Redis lease coordination",
                        "profile_type": profile_type,
                        "modality": modality,
                        "schema_name": "video_videoprism_large_mv_chunk_30s",
                        "embedding_type": "image",
                    }
                ],
                target_count=1,
            )
        )[0]

        assert example.query == "find Redis lease coordination"
        assert example.query_type == modality
        assert example.agent_sequence == [expected_agent]
        assert example.task_count == 1

    @pytest.mark.asyncio
    async def test_workflow_never_infers_modality_from_schema_name(self):
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        with pytest.raises(ValueError) as raised:
            await generator.generate(
                sampled_content=[
                    {
                        "topic": "Redis lease coordination",
                        "schema_name": "video_videoprism_large_mv_chunk_30s",
                        "embedding_type": "video",
                    }
                ],
                target_count=1,
            )

        assert str(raised.value) == (
            "sampled workflow content requires profile_type and modality"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("profile_type", "modality", "expected_error"),
        [
            (
                "Video",
                "VIDEO",
                "sampled workflow content profile_type must be canonical lowercase",
            ),
            (
                "video",
                "video",
                "sampled workflow content modality must be canonical uppercase",
            ),
            (
                "robot",
                "ROBOT",
                "sampled workflow content modality must be one of: "
                "AUDIO, CODE, DOCUMENT, IMAGE, VIDEO, WIKI; got 'ROBOT'",
            ),
        ],
    )
    async def test_workflow_rejects_noncanonical_or_unsupported_modality(
        self,
        profile_type,
        modality,
        expected_error,
    ):
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        with pytest.raises(ValueError) as raised:
            await generator.generate(
                sampled_content=[
                    {
                        "topic": "Redis lease coordination",
                        "profile_type": profile_type,
                        "modality": modality,
                    }
                ],
                target_count=1,
            )

        assert str(raised.value) == expected_error

    @pytest.mark.asyncio
    async def test_workflow_rejects_mismatched_profile_type_and_modality(self):
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        with pytest.raises(ValueError) as raised:
            await generator.generate(
                sampled_content=[
                    {
                        "topic": "Redis lease coordination",
                        "profile_type": "wiki",
                        "modality": "VIDEO",
                    }
                ],
                target_count=1,
            )

        assert str(raised.value) == (
            "sampled workflow content modality 'VIDEO' does not match "
            "profile_type 'wiki'"
        )

    @pytest.mark.asyncio
    async def test_workflow_generator(self):
        """Test WorkflowGenerator generates valid workflow executions"""
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        mock_content = [video_workflow_sample("Machine Learning Tutorial")]

        examples = await generator.generate(
            sampled_content=mock_content, target_count=3
        )

        assert len(examples) == 3
        assert all(isinstance(ex, WorkflowExecutionSchema) for ex in examples)
        assert all(len(ex.agent_sequence) >= 1 for ex in examples)
        assert all(ex.task_count == len(ex.agent_sequence) for ex in examples)
        for example in examples:
            assert {
                "execution_time": example.execution_time,
                "success": example.success,
                "parallel_efficiency": example.parallel_efficiency,
                "confidence_score": example.confidence_score,
                "user_satisfaction": example.user_satisfaction,
                "error_details": example.error_details,
                "metadata": example.metadata,
            } == {
                "execution_time": 0.0,
                "success": False,
                "parallel_efficiency": 0.0,
                "confidence_score": 0.0,
                "user_satisfaction": None,
                "error_details": None,
                "metadata": {
                    "_outcome_metadata": {
                        "observed": False,
                        "required_field_semantics": {
                            "execution_time": "unobserved_zero_sentinel",
                            "success": "unobserved_false_sentinel",
                            "parallel_efficiency": "unobserved_zero_sentinel",
                            "confidence_score": "unobserved_zero_sentinel",
                        },
                    }
                },
            }
        assert [example.agent_sequence for example in examples[:3]] == [
            ["search_agent"],
            ["search_agent", "summarizer_agent"],
            ["search_agent", "summarizer_agent", "detailed_report_agent"],
        ]
        assert [example.query for example in examples[:3]] == [
            "find Machine Learning Tutorial",
            "summarize Machine Learning Tutorial",
            "analyze Machine Learning Tutorial and generate report",
        ]

    @pytest.mark.asyncio
    async def test_workflow_generator_patterns(self):
        """Test WorkflowGenerator uses different workflow patterns"""
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        examples = await generator.generate(
            sampled_content=[video_workflow_sample("Marie Curie radium")],
            target_count=3,
        )

        # Check we get different workflow lengths (simple, moderate, complex)
        lengths = [len(ex.agent_sequence) for ex in examples]
        assert min(lengths) >= 1
        assert max(lengths) >= 2  # Should have at least some multi-agent workflows

    @pytest.mark.asyncio
    async def test_workflow_ids_remain_unique_across_generation_calls(self):
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        sampled_content = [video_workflow_sample("Marie Curie radium")]
        first = await generator.generate(
            sampled_content=sampled_content, target_count=3
        )
        second = await generator.generate(
            sampled_content=sampled_content, target_count=3
        )
        workflow_ids = [example.workflow_id for example in first + second]

        assert len(workflow_ids) == 6
        assert len(set(workflow_ids)) == 6
        assert all(
            len(workflow_id) == 51
            and workflow_id.startswith("synthetic_workflow_")
            and workflow_id.removeprefix("synthetic_workflow_").isalnum()
            for workflow_id in workflow_ids
        )

    @pytest.mark.asyncio
    async def test_workflow_generator_rejects_count_above_unique_query_capacity(self):
        generator = WorkflowGenerator(agent_inferrer=configured_agent_inferrer())

        examples = await generator.generate(
            sampled_content=[video_workflow_sample("Marie Curie radium")],
            target_count=4,
        )

        assert len(examples) == 3
        assert [example.query for example in examples] == [
            "find Marie Curie radium",
            "summarize Marie Curie radium",
            "analyze Marie Curie radium and generate report",
        ]

    @pytest.mark.asyncio
    async def test_workflow_intelligence_persists_generated_plan_for_serving(
        self, monkeypatch
    ):
        example = (
            await WorkflowGenerator(
                agent_inferrer=configured_agent_inferrer()
            ).generate(
                sampled_content=[video_workflow_sample("Marie Curie radium")],
                target_count=1,
            )
        )[0]
        response = SyntheticDataResponse(
            optimizer="workflow",
            schema_name="WorkflowExecutionSchema",
            count=1,
            selected_profiles=["video_videoprism_large_mv_chunk_30s"],
            profile_selection_reasoning="Selected source video profile",
            data=[example.model_dump(mode="python")],
            metadata={"sampled_content_count": 1},
        )

        class StaticSyntheticDataService:
            def __init__(self, **_kwargs):
                pass

            async def generate(self, _request):
                return response

        import cogniverse_synthetic

        monkeypatch.setattr(
            cogniverse_synthetic,
            "SyntheticDataService",
            StaticSyntheticDataService,
        )

        class TemplateStore:
            def __init__(self):
                self.templates = {}

            async def save_generated_templates(self, _tenant_id, templates):
                for template in templates:
                    self.templates[template.template_id] = template
                return [template.template_id for template in templates]

        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        store = TemplateStore()
        intelligence._store = store

        added = await intelligence.generate_synthetic_training_data(
            count=1,
            backend=object(),
            backend_config={"profiles": {"video_videoprism_large_mv_chunk_30s": {}}},
            generator_config=object(),
            agents_config=CONFIGURED_AGENTS,
        )

        assert added == 1
        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}
        assert len(store.templates) == 1
        template = next(iter(store.templates.values()))
        assert intelligence.workflow_templates == {template.template_id: template}
        assert template.query_patterns == ["find Marie Curie radium"]
        assert template.task_sequence == [
            {"agent": "search_agent", "task": "process", "dependencies": []}
        ]
        assert template.expected_execution_time is None
        assert template.success_rate is None
        assert (
            intelligence._find_matching_template("find Marie Curie radium") is template
        )

    @pytest.mark.asyncio
    async def test_workflow_intelligence_rejects_recycled_source_plans_before_save(
        self, monkeypatch
    ):
        examples = await WorkflowGenerator(
            agent_inferrer=configured_agent_inferrer()
        ).generate(
            sampled_content=[video_workflow_sample("Marie Curie radium")],
            target_count=3,
        )
        examples.append(
            examples[0].model_copy(
                update={"workflow_id": "synthetic_workflow_recycled_input"}
            )
        )
        response = SyntheticDataResponse(
            optimizer="workflow",
            schema_name="WorkflowExecutionSchema",
            count=4,
            selected_profiles=["video_videoprism_large_mv_chunk_30s"],
            profile_selection_reasoning="Selected source video profile",
            data=[example.model_dump(mode="python") for example in examples],
            metadata={"sampled_content_count": 1},
        )

        class StaticSyntheticDataService:
            def __init__(self, **_kwargs):
                pass

            async def generate(self, _request):
                return response

        class RecordingStore:
            def __init__(self):
                self.saved = []

            async def save_generated_templates(self, _tenant_id, templates):
                self.saved.extend(templates)
                return [template.template_id for template in templates]

        import cogniverse_synthetic

        monkeypatch.setattr(
            cogniverse_synthetic,
            "SyntheticDataService",
            StaticSyntheticDataService,
        )
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        store = RecordingStore()
        intelligence._store = store

        with pytest.raises(
            RuntimeError,
            match="Synthetic workflow response produced 3 unique grounded plans; expected 4",
        ):
            await intelligence.generate_synthetic_training_data(
                count=4,
                backend=object(),
                backend_config={
                    "profiles": {"video_videoprism_large_mv_chunk_30s": {}}
                },
                generator_config=object(),
                agents_config=CONFIGURED_AGENTS,
            )

        assert store.saved == []
        assert intelligence.workflow_templates == {}

    @pytest.mark.asyncio
    async def test_generated_plan_save_failure_restores_prior_templates(
        self, monkeypatch
    ):
        examples = await WorkflowGenerator(
            agent_inferrer=configured_agent_inferrer()
        ).generate(
            sampled_content=[video_workflow_sample("Marie Curie radium")],
            target_count=2,
        )
        response = SyntheticDataResponse(
            optimizer="workflow",
            schema_name="WorkflowExecutionSchema",
            count=2,
            selected_profiles=["video_videoprism_large_mv_chunk_30s"],
            profile_selection_reasoning="Selected source video profile",
            data=[example.model_dump(mode="python") for example in examples],
            metadata={"sampled_content_count": 1},
        )

        class StaticSyntheticDataService:
            def __init__(self, **_kwargs):
                pass

            async def generate(self, _request):
                return response

        prior = WorkflowTemplate(
            template_id="prior-template",
            name="prior workflow",
            description="Previously persisted observed workflow",
            query_patterns=["find prior source"],
            task_sequence=[
                {"agent": "search_agent", "task": "process", "dependencies": []}
            ],
            expected_execution_time=1.25,
            success_rate=0.9,
        )

        class FailingReplacementTemplateStore:
            def __init__(self):
                self.templates = {prior.template_id: prior}

            async def save_generated_templates(self, _tenant_id, _templates):
                raise TimeoutError("template blob store timed out")

        import cogniverse_synthetic

        monkeypatch.setattr(
            cogniverse_synthetic,
            "SyntheticDataService",
            StaticSyntheticDataService,
        )
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        store = FailingReplacementTemplateStore()
        intelligence._store = store
        intelligence.workflow_templates = {prior.template_id: prior}

        with pytest.raises(TimeoutError, match="template blob store timed out"):
            await intelligence.generate_synthetic_training_data(
                count=2,
                backend=object(),
                backend_config={
                    "profiles": {"video_videoprism_large_mv_chunk_30s": {}}
                },
                generator_config=object(),
                agents_config=CONFIGURED_AGENTS,
            )

        assert store.templates == {prior.template_id: prior}
        assert intelligence.workflow_templates == {prior.template_id: prior}

    @pytest.mark.asyncio
    async def test_concurrent_generated_plan_retries_serialize_template_writes(
        self, monkeypatch
    ):
        examples = await WorkflowGenerator(
            agent_inferrer=configured_agent_inferrer()
        ).generate(
            sampled_content=[video_workflow_sample("Marie Curie radium")],
            target_count=2,
        )
        response = SyntheticDataResponse(
            optimizer="workflow",
            schema_name="WorkflowExecutionSchema",
            count=2,
            selected_profiles=["video_videoprism_large_mv_chunk_30s"],
            profile_selection_reasoning="Selected source video profile",
            data=[example.model_dump(mode="python") for example in examples],
            metadata={"sampled_content_count": 1},
        )

        class StaticSyntheticDataService:
            def __init__(self, **_kwargs):
                pass

            async def generate(self, _request):
                await asyncio.sleep(0)
                return response

        class InterleavingStore:
            def __init__(self):
                self.templates = {}
                self.active_batches = 0
                self.max_active_batches = 0
                self.lock = asyncio.Lock()

            async def save_generated_templates(self, _tenant_id, templates):
                async with self.lock:
                    self.active_batches += 1
                    self.max_active_batches = max(
                        self.max_active_batches,
                        self.active_batches,
                    )
                    await asyncio.sleep(0.01)
                    for template in templates:
                        self.templates[template.template_id] = template
                    self.active_batches -= 1
                    return [template.template_id for template in templates]

        import cogniverse_synthetic

        monkeypatch.setattr(
            cogniverse_synthetic,
            "SyntheticDataService",
            StaticSyntheticDataService,
        )
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        store = InterleavingStore()
        intelligence._store = store
        call = {
            "count": 2,
            "backend": object(),
            "backend_config": {"profiles": {"video_videoprism_large_mv_chunk_30s": {}}},
            "generator_config": object(),
            "agents_config": CONFIGURED_AGENTS,
        }

        counts = await asyncio.gather(
            intelligence.generate_synthetic_training_data(**call),
            intelligence.generate_synthetic_training_data(**call),
        )

        assert counts == [2, 2]
        assert store.max_active_batches == 1
        assert len(store.templates) == 2
        assert set(store.templates) == set(intelligence.workflow_templates)

    @pytest.mark.asyncio
    async def test_concurrent_unobserved_workflows_never_enter_history(self):
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        executions = [
            workflow_execution(
                f"synthetic-{index}",
                observed=False,
                semantics=dict(UNOBSERVED_WORKFLOW_SEMANTICS),
                query=f"find source {index}",
            )
            for index in range(20)
        ]

        await asyncio.gather(
            *(intelligence.record_execution(execution) for execution in executions)
        )

        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}

    @pytest.mark.asyncio
    async def test_persisted_executions_use_the_validated_recording_path(self):
        unobserved = workflow_execution(
            "synthetic-persisted",
            observed=False,
            semantics=dict(UNOBSERVED_WORKFLOW_SEMANTICS),
        )
        observed = workflow_execution(
            "observed-persisted",
            observed=True,
            semantics=dict(OBSERVED_WORKFLOW_SEMANTICS),
            success=True,
            query="show me persisted source footage",
        )

        class StaticWorkflowStore:
            async def load_learning_state(self, _tenant_id):
                return WorkflowLearningState(
                    executions=[unobserved, observed],
                    profiles=[],
                    patterns={"video_search": ["show me persisted source footage"]},
                    templates=[],
                )

        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        intelligence._store = StaticWorkflowStore()

        await intelligence.load_historical_data()

        assert [
            execution.workflow_id for execution in intelligence.workflow_history
        ] == ["observed-persisted"]
        assert dict(intelligence.query_type_patterns) == {
            "video_search": ["show me persisted source footage"]
        }

    @pytest.mark.asyncio
    async def test_persisted_malformed_provenance_fails_without_partial_history(self):
        observed = workflow_execution(
            "observed-before-malformed",
            observed=True,
            semantics=dict(OBSERVED_WORKFLOW_SEMANTICS),
            success=True,
        )
        malformed = workflow_execution(
            "synthetic-malformed",
            observed=False,
            semantics=None,
        )

        class StaticWorkflowStore:
            async def load_learning_state(self, _tenant_id):
                return WorkflowLearningState(
                    executions=[observed, malformed],
                    profiles=[],
                    patterns={},
                    templates=[],
                )

        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        intelligence._store = StaticWorkflowStore()

        with pytest.raises(ValueError) as error:
            await intelligence.load_historical_data()

        assert str(error.value) == (
            "metadata._outcome_metadata.required_field_semantics must be a dict"
        )
        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("observed", "semantics", "expected_error"),
        [
            (
                False,
                None,
                "metadata._outcome_metadata.required_field_semantics must be a dict",
            ),
            (
                False,
                {},
                "metadata._outcome_metadata.required_field_semantics must exactly "
                "match the observed=false contract",
            ),
            (
                False,
                {
                    **UNOBSERVED_WORKFLOW_SEMANTICS,
                    "success": "observed_execution_outcome",
                },
                "metadata._outcome_metadata.required_field_semantics must exactly "
                "match the observed=false contract",
            ),
            (
                False,
                {**UNOBSERVED_WORKFLOW_SEMANTICS, "unexpected": "value"},
                "metadata._outcome_metadata.required_field_semantics must exactly "
                "match the observed=false contract",
            ),
            (
                True,
                None,
                "metadata._outcome_metadata.required_field_semantics must be a dict",
            ),
            (
                True,
                dict(UNOBSERVED_WORKFLOW_SEMANTICS),
                "metadata._outcome_metadata.required_field_semantics must exactly "
                "match the observed=true contract",
            ),
            (
                True,
                {**OBSERVED_WORKFLOW_SEMANTICS, "unexpected": "value"},
                "metadata._outcome_metadata.required_field_semantics must exactly "
                "match the observed=true contract",
            ),
        ],
    )
    async def test_required_field_semantics_are_exact(
        self, observed, semantics, expected_error
    ):
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        execution = workflow_execution(
            "semantic-contract",
            observed=observed,
            semantics=semantics,
            success=observed,
        )

        with pytest.raises(ValueError) as error:
            await intelligence.record_execution(execution)

        assert str(error.value) == expected_error
        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}

    @pytest.mark.asyncio
    async def test_observed_execution_accepts_explicit_unobserved_confidence(self):
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        semantics = {
            **OBSERVED_WORKFLOW_SEMANTICS,
            "confidence_score": "unobserved_zero_sentinel",
        }
        execution = workflow_execution(
            "observed-without-confidence",
            observed=True,
            semantics=semantics,
            success=True,
        )
        execution.confidence_score = 0.0

        await intelligence.record_execution(execution)

        assert [item.workflow_id for item in intelligence.workflow_history] == [
            "observed-without-confidence"
        ]

    @pytest.mark.asyncio
    async def test_malformed_outcome_provenance_fails_before_history_mutation(self):
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        execution = WorkflowExecution(
            workflow_id="synthetic-malformed",
            query="find source",
            query_type="VIDEO",
            execution_time=0.0,
            success=False,
            agent_sequence=["search_agent"],
            task_count=1,
            parallel_efficiency=0.0,
            confidence_score=0.0,
            metadata={"_outcome_metadata": {"observed": "false"}},
        )

        with pytest.raises(
            ValueError,
            match=r"^metadata\._outcome_metadata\.observed must be a bool$",
        ):
            await intelligence.record_execution(execution)

        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}

    @pytest.mark.asyncio
    async def test_missing_outcome_provenance_fails_before_history_mutation(self):
        intelligence = WorkflowIntelligence(tenant_id="test:unit")
        execution = WorkflowExecution(
            workflow_id="missing-provenance",
            query="find source footage",
            query_type="VIDEO",
            execution_time=1.25,
            success=True,
            agent_sequence=["search_agent"],
            task_count=1,
            parallel_efficiency=0.75,
            confidence_score=0.875,
        )

        with pytest.raises(
            ValueError,
            match=r"^metadata must contain _outcome_metadata$",
        ):
            await intelligence.record_execution(execution)

        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}

    def test_orchestration_span_marks_execution_outcomes_observed(self):
        from cogniverse_agents.routing.orchestration_evaluator import (
            OrchestrationEvaluator,
        )

        evaluator = object.__new__(OrchestrationEvaluator)
        evaluator.tenant_id = "acme:acme"
        execution = evaluator._extract_workflow_execution(
            {
                "attributes.input.value": "find the Curie laboratory footage",
                "attributes.output.value": {
                    "workflow_id": "workflow-observed",
                    "pattern": "sequential",
                    "agent_sequence": ["search_agent"],
                    "execution_order": ["search_agent"],
                    "execution_time": 1.25,
                    "success": True,
                    "tasks_completed": 1,
                    "confidence": 0.875,
                },
                "status_code": "OK",
                "context.span_id": "span-observed",
            }
        )

        assert execution is not None
        assert execution.metadata["_outcome_metadata"] == {
            "observed": True,
            "required_field_semantics": dict(OBSERVED_WORKFLOW_SEMANTICS),
        }


class TestAllGeneratorsTogether:
    """Test all generators can work together"""

    @pytest.mark.asyncio
    async def test_all_generators_produce_valid_output(self):
        """Test all generators can produce valid output"""
        pattern_extractor = PatternExtractor()
        agent_inferrer = configured_agent_inferrer()

        mock_content = [
            {
                "title": "Deep Learning with TensorFlow",
                "video_title": "Deep Learning with TensorFlow",
                "segment_description": "Tutorial on neural networks",
                "schema_name": "video_videoprism_large_mv_chunk_30s",
                "profile_type": "video",
                "modality": "VIDEO",
                "embedding_type": "video",
            }
        ]

        generators = [
            (
                RoutingGenerator(
                    entity_extractor=extract_entities,
                    routing_decider=route_query,
                    pattern_extractor=pattern_extractor,
                    optimizer_config=create_routing_config(),
                ),
                1,
            ),
            (WorkflowGenerator(agent_inferrer=agent_inferrer), 3),
        ]

        for generator, target_count in generators:
            examples = await generator.generate(
                sampled_content=mock_content,
                target_count=target_count,
                tenant_id="acme:routing",
            )

            assert len(examples) == target_count
            # All should return Pydantic models with model_dump
            assert all(hasattr(ex, "model_dump") for ex in examples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
