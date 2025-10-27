"""
Pytest configuration for synthetic optimizer integration tests
"""

import pytest
from cogniverse_core.config.unified_config import (
    AgentMappingRule,
    BackendConfig,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    SyntheticGeneratorConfig,
)


@pytest.fixture
def test_generator_config():
    """Create test generator configuration with all required optimizer configs"""
    return SyntheticGeneratorConfig(
        optimizer_configs={
            "modality": OptimizerGenerationConfig(
                optimizer_type="modality",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateModalityQuery",
                        module_type="Predict",
                    )
                },
                agent_mappings=[
                    AgentMappingRule(modality="VIDEO", agent_name="video_search_agent"),
                    AgentMappingRule(modality="DOCUMENT", agent_name="document_search_agent"),
                ],
            ),
            "routing": OptimizerGenerationConfig(
                optimizer_type="routing",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                        module_type="Predict",
                    )
                },
            ),
        }
    )


@pytest.fixture
def test_backend_config():
    """Create test backend configuration"""
    return BackendConfig(profiles={})
