"""Real-Phoenix: approved synthetic counts toward the SFT threshold.

analyze_data reads approved training examples from real telemetry annotations.
On a re-run, approved synthetic examples from a prior run must also count, so an
empty project + enough approved synthetic moves the recommendation off
"insufficient" to "sft". Drives the real TrainingMethodSelector against a real
(empty) Phoenix project.
"""

from __future__ import annotations

import pytest

from cogniverse_finetuning.dataset.method_selector import TrainingMethodSelector

pytestmark = pytest.mark.integration


@pytest.fixture
def telemetry_manager(phoenix_container):
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_approved_synthetic_moves_recommendation_to_sft(
    phoenix_container, telemetry_manager
):
    tenant_id = "count_syn"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )
    with telemetry_manager.span(
        name="entity_extraction_agent",
        tenant_id=tenant_id,
        project_name=project_name,
        attributes={"input.query": "seed"},
    ):
        pass
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(
        tenant_id=tenant_id, project_name=project_name
    )
    selector = TrainingMethodSelector()

    # No approved real data and no synthetic -> insufficient.
    without = await selector.analyze_data(
        provider, full_project, "entity_extraction", min_sft_examples=50
    )
    assert without.recommended_method == "insufficient"
    assert without.approved_count == 0
    assert without.needs_synthetic is True

    # 50 approved synthetic examples clear the SFT threshold.
    approved_synthetic = [
        {"query": f"text {i}", "entities": [{"text": "X", "type": "ORG"}]}
        for i in range(50)
    ]
    with_syn = await selector.analyze_data(
        provider,
        full_project,
        "entity_extraction",
        min_sft_examples=50,
        approved_synthetic=approved_synthetic,
    )
    assert with_syn.approved_count == 50
    assert with_syn.recommended_method == "sft"
    assert with_syn.needs_synthetic is False
