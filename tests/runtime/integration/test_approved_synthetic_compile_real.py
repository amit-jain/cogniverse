import uuid

import dspy
import pytest

from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
from cogniverse_agents.profile_selection_agent import ProfileSelectionModule
from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule
from cogniverse_core.approval.interfaces import ApprovalStatus, ReviewDecision
from cogniverse_runtime.optimization_cli import (
    _create_teleprompter,
    _load_approved_synthetic_data,
    _project_approved_optimizer_example,
)
from cogniverse_synthetic.approval.confidence_extractor import (
    SyntheticDataConfidenceExtractor,
)

pytestmark = pytest.mark.integration


async def _approve_example(agent, optimizer_type, agent_type, example):
    batch_id = f"compile-{optimizer_type}-{uuid.uuid4().hex}"
    approved_item_ids = await _approve_examples(
        agent,
        optimizer_type,
        agent_type,
        [example],
        batch_id=batch_id,
    )
    return approved_item_ids[0]


async def _approve_examples(
    agent,
    optimizer_type: str,
    agent_type: str,
    examples,
    *,
    batch_id: str | None = None,
):
    batch_id = batch_id or f"compile-{optimizer_type}-{uuid.uuid4().hex}"
    batch = await agent.process_batch(
        [
            example.model_dump() if hasattr(example, "model_dump") else example
            for example in examples
        ],
        batch_id,
        {
            "tenant_id": agent.storage.tenant_id,
            "optimizer": optimizer_type,
            "agent_type": agent_type,
            "purpose": "compile the exact approved DSPy example",
        },
    )
    expected_items = [
        (f"{batch_id}_{index}", ApprovalStatus.PENDING_REVIEW)
        for index in range(len(examples))
    ]
    actual_items = [(item.item_id, item.status) for item in batch.items]
    if actual_items != expected_items:
        raise AssertionError(
            f"{optimizer_type} batch items drifted: "
            f"expected={expected_items} actual={actual_items}"
        )

    approved_item_ids: list[str] = []
    for index in range(len(examples)):
        item_id = f"{batch_id}_{index}"
        approved = await agent.apply_decision(
            batch_id,
            ReviewDecision(
                item_id=item_id,
                approved=True,
                feedback="The labels exactly match the reviewed source content.",
                reviewer="optimizer-integration@example.com",
            ),
        )
        if approved is None:
            raise AssertionError(
                f"{optimizer_type} apply_decision returned None for {item_id}"
            )
        if approved.status is not ApprovalStatus.APPROVED:
            raise AssertionError(
                f"{optimizer_type} approval {item_id} did not persist as approved: "
                f"{approved.status!r}"
            )
        approved_item_ids.append(item_id)
    return approved_item_ids


def _profile_selection_generator_setup(tenant_id: str):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import (
        BackendProfileConfig,
        SystemConfig,
    )
    from tests.utils.memory_store import InMemoryConfigStore

    profile_name = "video_colpali_smol500_mv_frame"
    profile_config = {
        "type": "video",
        "schema_name": profile_name,
        "embedding_type": "single_vector",
        "pipeline_config": {"extract_keyframes": True},
        "inference_services": {"embedding": "video_embedding"},
    }
    config_manager = ConfigManager(store=InMemoryConfigStore())
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict(profile_name, profile_config),
        tenant_id=tenant_id,
    )
    config_manager.set_system_config(
        SystemConfig(
            inference_service_urls={"video_embedding": "http://video_embedding.invalid"}
        )
    )
    return profile_name, profile_config, config_manager


@pytest.mark.asyncio
async def test_real_phoenix_approved_examples_compile_into_actual_dspy_modules(
    phoenix_container,
    real_telemetry,
    workflow_state_redis_url,
    dspy_test_lm,
):
    tenant_id = f"approved-compile:{uuid.uuid4().hex[:12]}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=real_telemetry,
        redis_url=workflow_state_redis_url,
    )
    agent = HumanApprovalAgent(
        storage=storage,
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        confidence_threshold=1.0,
    )
    cases = [
        (
            "query_enhancement",
            "query_enhancement",
            QueryEnhancementModule,
            ("query",),
            {
                "query": "transformer architecture",
                "enhanced_query": "transformer architecture attention mechanism",
                "expansion_terms": ["attention mechanism"],
                "synonyms": ["neural network model"],
                "context": "machine learning",
                "reasoning": "Added the exact attention term from the source.",
            },
        ),
        (
            "profile",
            "profile_selection",
            ProfileSelectionModule,
            ("query", "available_profiles"),
            {
                "query": "find the product launch recording",
                "available_profiles": "video_colpali,document_colpali",
                "selected_profile": "video_colpali",
                "reasoning": "The requested recording is video content.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "simple",
            },
        ),
        (
            "entity_extraction",
            "entity_extraction",
            EntityExtractionModule,
            ("query",),
            {
                "query": "PyTorch was created by Meta AI in Menlo Park",
                "entities": [
                    {"text": "PyTorch", "type": "PRODUCT"},
                    {"text": "Meta AI", "type": "ORG"},
                    {"text": "Menlo Park", "type": "PLACE"},
                ],
                "relationships": [
                    {
                        "source": "Meta AI",
                        "target": "PyTorch",
                        "type": "created",
                    }
                ],
            },
        ),
    ]

    approved_item_ids = {}
    for optimizer_type, agent_type, _, _, source in cases:
        approved_item_ids[optimizer_type] = await _approve_example(
            agent, optimizer_type, agent_type, source
        )

    provider = real_telemetry.get_provider(tenant_id=tenant_id)
    dspy.configure(lm=dspy_test_lm)
    for optimizer_type, _, module_type, input_names, source in cases:
        loaded = await _load_approved_synthetic_data(
            provider, tenant_id, optimizer_type
        )
        assert loaded == [
            {**source, "example_id": f"approved:{approved_item_ids[optimizer_type]}"}
        ]
        projected = _project_approved_optimizer_example(optimizer_type, loaded[0])
        trainset = [dspy.Example(**projected).with_inputs(*input_names)]

        compiled = _create_teleprompter(
            1, teacher_settings={"lm": dspy_test_lm}
        ).compile(module_type(), trainset=trainset)

        assert compiled._compiled is True
        predictors = compiled.named_predictors()
        assert len(predictors) == 1
        demos = predictors[0][1].demos
        assert len(demos) == 1
        assert demos[0].toDict() == projected
        assert demos[0].inputs().toDict() == {
            name: projected[name] for name in input_names
        }


@pytest.mark.asyncio
async def test_generator_output_survives_validation_and_persists(
    phoenix_container,
    real_telemetry,
    workflow_state_redis_url,
):
    """Real generator output must satisfy the approval validator.

    This is the join coverage for the synthetic approval pipeline. It uses
    real generator output for the shipped minimum unique query counts
    (query_enhancement=3, profile_selection=6, entity_extraction=15), then
    drives each batch through validation and persistence and reads the exact
    approved rows back from Phoenix.
    """
    from cogniverse_synthetic.generators.entity_extraction import (
        EntityExtractionGenerator,
    )
    from cogniverse_synthetic.generators.profile import ProfileGenerator
    from cogniverse_synthetic.generators.query_enhancement import (
        QueryEnhancementGenerator,
    )

    tenant_id = f"generator-persist:{uuid.uuid4().hex[:12]}"
    storage = ApprovalStorageImpl(
        grpc_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        tenant_id=tenant_id,
        telemetry_manager=real_telemetry,
        redis_url=workflow_state_redis_url,
    )
    agent = HumanApprovalAgent(
        storage=storage,
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        confidence_threshold=1.0,
    )

    expected_tenant_id = tenant_id

    async def _query_enhancer(query: str, incoming_tenant_id: str, source_text: str):
        if incoming_tenant_id != expected_tenant_id:
            raise AssertionError(
                f"query_enhancer tenant drifted: {incoming_tenant_id!r}"
            )
        if "attention mechanism" not in source_text:
            raise AssertionError("query_enhancer source_text lost the expansion term")
        if "neural network model" not in source_text:
            raise AssertionError("query_enhancer source_text lost the synonym term")
        return {
            "original_query": query,
            "enhanced_query": f"{query} attention mechanism",
            "expansion_terms": ["attention mechanism"],
            "synonyms": ["neural network model"],
            "reasoning": "Added the grounded attention term from the source.",
        }

    query_enhancement_generator = QueryEnhancementGenerator(
        query_enhancer=_query_enhancer
    )
    query_enhancement_examples = await query_enhancement_generator.generate(
        sampled_content=[
            {
                "title": "transformer architecture",
                "description": "transformer architecture attention mechanism neural network model",
                "content_type": "video",
            },
            {
                "title": "compiler passes",
                "description": "compiler passes attention mechanism neural network model",
                "content_type": "video",
            },
            {
                "title": "search ranking",
                "description": "search ranking attention mechanism neural network model",
                "content_type": "video",
            },
        ],
        target_count=3,
        tenant_id=tenant_id,
    )
    assert len(query_enhancement_examples) == 3

    profile_name, profile_config, profile_config_manager = (
        _profile_selection_generator_setup(tenant_id)
    )

    async def _profile_labeler(
        query: str, profiles: list[str], incoming_tenant_id: str
    ):
        if incoming_tenant_id != expected_tenant_id:
            raise AssertionError(
                f"profile_labeler tenant drifted: {incoming_tenant_id!r}"
            )
        if profiles != [profile_name]:
            raise AssertionError(
                f"profile_labeler available_profiles drifted: {profiles!r}"
            )
        return {
            "query": query,
            "selected_profile": profile_name,
            "reasoning": "Selected the video profile for the query.",
            "query_intent": "video_search",
            "modality": "video",
            "complexity": "simple",
        }

    profile_generator = ProfileGenerator(profile_labeler=_profile_labeler)
    profile_examples = await profile_generator.generate(
        sampled_content=[
            {"title": f"launch footage {index}", "schema_name": profile_name}
            for index in range(6)
        ],
        target_count=6,
        tenant_id=tenant_id,
        config_manager=profile_config_manager,
        profile_configs={profile_name: profile_config},
    )
    assert len(profile_examples) == 6

    async def _entity_extractor(text: str, incoming_tenant_id: str):
        if incoming_tenant_id != expected_tenant_id:
            raise AssertionError(
                f"entity_extractor tenant drifted: {incoming_tenant_id!r}"
            )
        if "PyTorch example" not in text:
            raise AssertionError(f"entity_extractor source text drifted: {text!r}")
        return {
            "query": text,
            "entities": [
                {"text": "PyTorch", "type": "PRODUCT"},
                {"text": "Meta AI", "type": "ORG"},
                {"text": "Menlo Park", "type": "PLACE"},
            ],
            "relationships": [
                {"subject": "Meta AI", "relation": "created", "object": "PyTorch"}
            ],
        }

    entity_generator = EntityExtractionGenerator(entity_extractor=_entity_extractor)
    entity_examples = await entity_generator.generate(
        sampled_content=[
            {"topic": (f"PyTorch example {index} was created by Meta AI in Menlo Park")}
            for index in range(15)
        ],
        target_count=15,
        tenant_id=tenant_id,
    )
    assert len(entity_examples) == 15

    provider = real_telemetry.get_provider(tenant_id=tenant_id)

    query_enhancement_item_ids = await _approve_examples(
        agent,
        "query_enhancement",
        "query_enhancement",
        query_enhancement_examples,
    )
    loaded_query_enhancement = await _load_approved_synthetic_data(
        provider, tenant_id, "query_enhancement"
    )
    assert loaded_query_enhancement == [
        {
            **example.model_dump(),
            "example_id": f"approved:{query_enhancement_item_ids[index]}",
        }
        for index, example in enumerate(query_enhancement_examples)
    ]

    profile_item_ids = await _approve_examples(
        agent,
        "profile",
        "profile_selection",
        profile_examples,
    )
    loaded_profile = await _load_approved_synthetic_data(provider, tenant_id, "profile")
    assert loaded_profile == [
        {
            **example.model_dump(),
            "example_id": f"approved:{profile_item_ids[index]}",
        }
        for index, example in enumerate(profile_examples)
    ]

    entity_item_ids = await _approve_examples(
        agent,
        "entity_extraction",
        "entity_extraction",
        entity_examples,
    )
    loaded_entity = await _load_approved_synthetic_data(
        provider, tenant_id, "entity_extraction"
    )
    assert loaded_entity == [
        {
            **example.model_dump(),
            "example_id": f"approved:{entity_item_ids[index]}",
        }
        for index, example in enumerate(entity_examples)
    ]
