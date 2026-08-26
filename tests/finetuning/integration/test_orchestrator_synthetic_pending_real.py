"""Real-Phoenix: the orchestrator reports pending approval, not failure.

When real telemetry is insufficient and synthetic data is generated but lands in
human-review (synthetic confidence 0.8 < the 0.85 auto-approve threshold), the
orchestrator must raise SyntheticApprovalPending — a recoverable state — instead
of the old misleading "synthetic generation failed" ValueError.

Drives the real FinetuningOrchestrator with a real SyntheticDataService (entity
generator) and a real HumanApprovalAgent against an empty real Phoenix project.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
    phoenix_container, telemetry_manager, shared_vespa
):
    from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent
    from cogniverse_agents.entity_extraction_agent import (
        EntityExtractionAgent,
        EntityExtractionDeps,
        EntityExtractionInput,
    )
    from cogniverse_core.registries.schema_registry import SchemaRegistry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        BackendProfileConfig,
    )
    from cogniverse_synthetic.service import SyntheticDataService
    from cogniverse_vespa._vespa_factory import make_vespa_app
    from cogniverse_vespa.backend import VespaBackend
    from tests.utils.synthetic_config import video_synthetic_generator_config
    from tests.utils.vespa_test_helpers import (
        deploy_tenant_schema,
        make_config_manager,
    )

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
    profile_name = "video_colpali_smol500_mv_frame"
    title = "PyTorch was released by Meta AI"
    description = "PyTorch was released by Meta AI."
    config_manager = make_config_manager(shared_vespa)
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    backend_config = BackendConfig(
        backend_type="vespa",
        url="http://localhost",
        port=shared_vespa["http_port"],
        tenant_id=tenant_id,
        profiles={
            profile_name: BackendProfileConfig(
                profile_name=profile_name,
                type="video",
                schema_name=profile_name,
                embedding_type="multi_vector",
                pipeline_config={"generate_descriptions": True},
            )
        },
    )
    schema = deploy_tenant_schema(
        shared_vespa,
        tenant_id=tenant_id,
        base_schema_name=profile_name,
        config_manager=config_manager,
    )
    backend = VespaBackend(
        backend_config=backend_config,
        schema_loader=schema_loader,
        config_manager=config_manager,
    )
    backend.initialize({"tenant_id": tenant_id})
    registry = SchemaRegistry(
        config_manager=config_manager,
        backend=backend,
        schema_loader=schema_loader,
    )
    backend.schema_registry = registry
    backend.schema_manager._schema_registry = registry
    feed = make_vespa_app(
        url="http://localhost",
        port=shared_vespa["http_port"],
    ).feed_data_point(
        schema=schema,
        data_id="pytorch-meta-segment",
        fields={
            "video_id": "pytorch-meta",
            "video_title": title,
            "source_url": "http://example.test/pytorch-meta",
            "segment_id": 0,
            "segment_description": description,
            "start_time": 0.0,
            "end_time": 9.0,
        },
    )
    assert feed.is_successful(), feed.json

    indexed = []
    for _ in range(20):
        indexed = backend.query_metadata_documents(
            schema=profile_name,
            yql=f"select * from sources {profile_name} where true limit 1",
            hits=1,
            tenant_id=tenant_id,
        )
        if indexed:
            break
        await asyncio.sleep(0.5)
    assert len(indexed) == 1
    assert indexed[0]["video_title"] == title
    assert indexed[0]["segment_description"] == description

    entity_agent = EntityExtractionAgent(deps=EntityExtractionDeps())
    entity_agent.telemetry_manager = telemetry_manager
    extraction_paths = []

    async def extract_entities(text: str, tenant_id: str):
        result = await entity_agent.process(
            EntityExtractionInput(query=text, tenant_id=tenant_id)
        )
        extraction_paths.append(result.path_used)
        if result.path_used != "fast":
            raise RuntimeError(
                "finetuning synthetic generation did not use the GLiNER fast path"
            )
        return result

    synthetic_service = SyntheticDataService(
        backend=backend,
        generator_config=video_synthetic_generator_config(tenant_id),
        backend_config=backend_config,
        agents_config=json.loads(Path("configs/config.json").read_text())["agents"],
        entity_extractor=extract_entities,
    )
    approval_agent = HumanApprovalAgent(
        confidence_extractor=_NoopExtractor(), confidence_threshold=0.85
    )

    orchestrator = FinetuningOrchestrator(
        telemetry_provider=provider,
        telemetry_manager=telemetry_manager,
        synthetic_service=synthetic_service,
        approval_agent=approval_agent,
    )

    config = OrchestrationConfig(
        tenant_id=tenant_id,
        project=full_project,
        model_type="llm",
        agent_type="entity_extraction",
        min_sft_examples=2,
        min_dpo_pairs=20,
        generate_synthetic=True,
        backend="local",
        enable_registry=False,
        evaluate_after_training=False,
    )

    try:
        with pytest.raises(SyntheticApprovalPending) as exc_info:
            await orchestrator.run(config)
    finally:
        backend.close()

    pending = exc_info.value
    assert pending.agent_type == "entity_extraction"
    assert pending.pending_count == 2
    assert pending.approved_count == 0
    assert pending.batch_id.startswith("synthetic_entity_extraction_")
    assert extraction_paths == ["fast", "fast"]
