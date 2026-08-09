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
    batch = await agent.process_batch(
        [example],
        batch_id,
        {
            "tenant_id": agent.storage.tenant_id,
            "optimizer": optimizer_type,
            "agent_type": agent_type,
            "purpose": "compile the exact approved DSPy example",
        },
    )
    assert [(item.item_id, item.status) for item in batch.items] == [
        (f"{batch_id}_0", ApprovalStatus.PENDING_REVIEW)
    ]
    approved = await agent.apply_decision(
        batch_id,
        ReviewDecision(
            item_id=f"{batch_id}_0",
            approved=True,
            feedback="The labels exactly match the reviewed source content.",
            reviewer="optimizer-integration@example.com",
        ),
    )
    assert approved is not None
    assert approved.status is ApprovalStatus.APPROVED


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
                "entity_types": "PRODUCT,ORG,PLACE",
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

    for optimizer_type, agent_type, _, _, source in cases:
        await _approve_example(agent, optimizer_type, agent_type, source)

    provider = real_telemetry.get_provider(tenant_id=tenant_id)
    dspy.configure(lm=dspy_test_lm)
    for optimizer_type, _, module_type, input_names, source in cases:
        loaded = await _load_approved_synthetic_data(
            provider, tenant_id, optimizer_type
        )
        assert loaded == [source]
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
