"""Real-Phoenix: the orchestrator reports pending approval, not failure.

When real telemetry is insufficient and synthetic data is generated but lands in
human-review (synthetic confidence 0.8 < the 0.85 auto-approve threshold), the
orchestrator must raise SyntheticApprovalPending — a recoverable state — instead
of the old misleading "synthetic generation failed" ValueError.

Drives the real FinetuningOrchestrator with a real SyntheticDataService (entity
generator) and a real HumanApprovalAgent against an empty real Phoenix project.
"""

from __future__ import annotations

import pytest

from cogniverse_finetuning.orchestrator import (
    FinetuningOrchestrator,
    OrchestrationConfig,
    SyntheticApprovalPending,
)

pytestmark = pytest.mark.integration


class _NoopExtractor:
    def extract(self, data) -> float:
        return 0.0


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
async def test_run_reports_pending_approval_not_failure(
    phoenix_container, telemetry_manager
):
    from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        SyntheticGeneratorConfig,
    )
    from cogniverse_synthetic.service import SyntheticDataService

    tenant_id = "orch_pend"
    project_name = "finetuning"
    full_project = f"cogniverse-{tenant_id}-{project_name}"

    telemetry_manager.register_project(
        tenant_id=tenant_id,
        project_name=project_name,
        otlp_endpoint=phoenix_container["grpc_endpoint"],
        http_endpoint=phoenix_container["http_endpoint"],
        use_sync_export=True,
    )
    # One span so the project exists; no approval annotations -> 0 approved
    # training examples -> analysis is "insufficient" -> synthetic is generated.
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
    synthetic_service = SyntheticDataService(
        generator_config=SyntheticGeneratorConfig(tenant_id=tenant_id),
        backend_config=BackendConfig(profiles={}, tenant_id=tenant_id),
    )
    approval_agent = HumanApprovalAgent(
        confidence_extractor=_NoopExtractor(), confidence_threshold=0.85
    )

    orchestrator = FinetuningOrchestrator(
        telemetry_provider=provider,
        synthetic_service=synthetic_service,
        approval_agent=approval_agent,
    )

    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=full_project,
        model_type="llm",
        agent_type="entity_extraction",
        min_sft_examples=50,
        min_dpo_pairs=20,
        generate_synthetic=True,
        backend="local",
        enable_registry=False,
        evaluate_after_training=False,
    )

    with pytest.raises(SyntheticApprovalPending) as exc_info:
        await orchestrator.run(config)

    pending = exc_info.value
    assert pending.agent_type == "entity_extraction"
    assert pending.pending_count > 0
    assert pending.approved_count == 0
    assert pending.batch_id
