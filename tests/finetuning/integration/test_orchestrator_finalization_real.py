import asyncio
import time
import uuid

import pytest

from cogniverse_finetuning.evaluation.adapter_evaluator import (
    ComparisonResult,
    EvaluationMetrics,
)
from cogniverse_finetuning.orchestrator import (
    FinetuningOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
)

pytestmark = pytest.mark.integration


def _metrics(*, accuracy, confidence, latency_ms, correctness):
    return EvaluationMetrics(
        accuracy=accuracy,
        top_k_accuracy=accuracy,
        avg_confidence=confidence,
        confidence_calibration=abs(confidence - accuracy),
        error_rate=1.0 - accuracy,
        hallucination_rate=0.0,
        avg_latency_ms=latency_ms,
        sample_count=len(correctness),
        correctness=correctness,
    )


@pytest.mark.asyncio
async def test_finalization_exports_linked_evaluation_and_experiment_to_real_phoenix(
    telemetry_manager_with_phoenix,
):
    tenant_id = f"finalization-{uuid.uuid4().hex}"
    manager = telemetry_manager_with_phoenix
    provider = manager.get_provider(
        tenant_id=tenant_id,
        project_name="experiments",
    )
    project = manager.config.get_project_name(tenant_id, "experiments")
    orchestrator = FinetuningOrchestrator(
        telemetry_provider=provider,
        telemetry_manager=manager,
    )
    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=f"cogniverse-{tenant_id}-finetuning",
        model_type="llm",
        agent_type="routing",
        base_model="google/gemma-4-e4b-it",
        evaluate_after_training=True,
        test_set_size=4,
        enable_registry=False,
    )
    result = OrchestrationResult(
        model_type="llm",
        training_method="sft",
        adapter_path="/models/routing-sft",
        metrics={"train_loss": 0.125, "train_samples": 4, "epoch": 2},
        base_model="google/gemma-4-e4b-it",
        lora_config={"use_lora": True},
        used_synthetic=False,
    )
    comparison = ComparisonResult(
        base_metrics=_metrics(
            accuracy=0.5,
            confidence=0.55,
            latency_ms=12.0,
            correctness=(True, False, True, False),
        ),
        adapter_metrics=_metrics(
            accuracy=0.75,
            confidence=0.8,
            latency_ms=15.0,
            correctness=(True, True, True, False),
        ),
        accuracy_improvement=0.25,
        confidence_improvement=0.25,
        error_reduction=0.25,
        latency_overhead=3.0,
        improvement_significant=False,
        p_value=1.0,
    )

    finalized = await orchestrator._finalize_training_result(
        config=config,
        result=result,
        analysis=None,
        approved_batch=None,
        formatted_dataset=[
            {
                "text": (
                    "Route the request: find the launch video -> "
                    '{"recommended_agent":"video_search"}'
                )
            }
        ],
        evaluation_result=comparison,
    )

    deadline = time.monotonic() + 60
    matched = None
    while time.monotonic() < deadline:
        spans = await provider.traces.get_all_spans(project=project)
        if spans is not None and not spans.empty and "name" in spans.columns:
            matched = spans[
                spans["name"].isin(["evaluation.routing", "experiment.routing.sft"])
            ]
            if len(matched) == 2:
                break
        await asyncio.sleep(0.5)

    assert finalized is result
    assert finalized.adapter_id is None
    assert finalized.evaluation_result is comparison
    assert matched is not None
    assert sorted(matched["name"].tolist()) == [
        "evaluation.routing",
        "experiment.routing.sft",
    ]
    evaluation = matched[matched["name"] == "evaluation.routing"].iloc[0]
    experiment = matched[matched["name"] == "experiment.routing.sft"].iloc[0]
    assert (
        evaluation["attributes.experiment"]["run_id"]
        == experiment["attributes.experiment"]["run_id"]
    )
    assert evaluation["attributes.evaluation"] == {
        "adapter_path": "/models/routing-sft",
        "agent_type": "routing",
        "test_size": 4,
    }
    assert evaluation["attributes.metrics"]["adapter"] == {
        "accuracy": 0.75,
        "confidence": 0.8,
        "error_rate": 0.25,
        "hallucination_rate": 0.0,
        "latency_ms": 15.0,
    }
    assert experiment["attributes.params"]["base_model"] == ("google/gemma-4-e4b-it")
    assert experiment["attributes.metrics"]["train_loss"] == 0.125
    assert experiment["attributes.data"]["dataset_size"] == 1
    assert experiment["attributes.output"]["adapter_path"] == ("/models/routing-sft")
    assert evaluation["attributes.tenant"]["id"] == tenant_id
    assert experiment["attributes.tenant"]["id"] == tenant_id
