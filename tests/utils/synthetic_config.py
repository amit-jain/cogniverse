"""Canonical synthetic-service configuration used by real-boundary tests."""

from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    ProfileScoringRule,
    SyntheticGeneratorConfig,
)


def video_synthetic_generator_config(tenant_id: str) -> SyntheticGeneratorConfig:
    """Return the canonical VIDEO-to-search-agent generator configuration."""
    scoring_configs = {
        optimizer_name: OptimizerGenerationConfig(
            optimizer_type=optimizer_name,
            profile_scoring_rules=[
                ProfileScoringRule(
                    condition={"field": "type", "equals": "video"},
                    score_adjustment=1.0,
                    reason=f"video source for {optimizer_name}",
                )
            ],
        )
        for optimizer_name in (
            "cross_modal",
            "entity_extraction",
            "profile",
            "query_enhancement",
            "routing",
            "unified",
            "workflow",
        )
    }
    scoring_configs["routing"].dspy_modules = {
        "query_generator": DSPyModuleConfig(
            signature_class=(
                "cogniverse_synthetic.dspy_signatures.GenerateEntityQuery"
            ),
            module_type="Predict",
        )
    }
    scoring_configs["modality"] = OptimizerGenerationConfig(
        optimizer_type="modality",
        agent_mappings=[
            AgentMappingRule(
                modality="VIDEO",
                agent_name="search_agent",
            )
        ],
    )
    return SyntheticGeneratorConfig(
        tenant_id=tenant_id,
        optimizer_configs=scoring_configs,
    )
