"""
Pytest configuration for synthetic optimizer integration tests
"""

import subprocess

import dspy
import pytest
from cogniverse_foundation.config.unified_config import (
    AgentMappingRule,
    BackendConfig,
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    SyntheticGeneratorConfig,
)


def is_ollama_available():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(autouse=True)
def dspy_lm():
    """Configure DSPy with real Ollama LM for integration tests"""
    if not is_ollama_available():
        pytest.skip("Ollama not available - required for integration tests")

    lm = dspy.LM(
        model="ollama/gemma3:4b",
        api_base="http://localhost:11434",
    )
    dspy.configure(lm=lm)
    yield lm
    dspy.configure(lm=None)


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
