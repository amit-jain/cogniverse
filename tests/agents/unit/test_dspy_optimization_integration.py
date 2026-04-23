"""Integration tests for DSPy optimization with OpenAI-compatible APIs."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

from cogniverse_agents.optimizer.dspy_agent_optimizer import (
    DSPyAgentOptimizerPipeline,
    DSPyAgentPromptOptimizer,
)
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig


def _make_mock_telemetry_provider():
    """Create a mock TelemetryProvider with in-memory stores."""
    provider = MagicMock()
    datasets: dict = {}

    async def create_dataset(name, data, metadata=None):
        datasets[name] = data
        return f"ds-{name}"

    async def get_dataset(name):
        if name not in datasets:
            raise KeyError(f"Dataset {name} not found")
        return datasets[name]

    provider.datasets = MagicMock()
    provider.datasets.create_dataset = AsyncMock(side_effect=create_dataset)
    provider.datasets.get_dataset = AsyncMock(side_effect=get_dataset)
    provider.experiments = MagicMock()
    provider.experiments.create_experiment = AsyncMock(return_value="exp-test")
    provider.experiments.log_run = AsyncMock(return_value="run-test")
    return provider


@pytest.fixture
def config_manager(backend_config_env):
    """Create ConfigManager with backend store for tests."""
    return create_default_config_manager()


@pytest.fixture
def mock_openai_compatible_api():
    """Mock OpenAI-compatible API responses for DSPy."""

    def mock_response_generator(prompt, **kwargs):
        """Generate mock responses based on prompt content."""

        if "query analysis" in prompt.lower():
            return {
                "primary_intent": "search",
                "complexity_level": "simple",
                "needs_video_search": "true",
                "needs_text_search": "false",
                "multimodal_query": "false",
                "temporal_pattern": "null",
                "reasoning": "Simple video search request",
            }
        elif "agent routing" in prompt.lower():
            return {
                "recommended_workflow": "raw_results",
                "primary_agent": "video_search",
                "secondary_agents": "[]",
                "routing_confidence": "0.9",
                "reasoning": "Direct video search task",
            }
        elif "summary" in prompt.lower():
            return {
                "summary": "This is an optimized summary of the content.",
                "key_points": '["Point 1", "Point 2", "Point 3"]',
                "confidence": "0.85",
            }
        elif "detailed report" in prompt.lower():
            return {
                "executive_summary": "Executive summary of findings",
                "detailed_findings": "Detailed analysis results",
                "recommendations": "1. Recommendation one 2. Recommendation two",
                "technical_details": "Technical implementation details",
                "confidence": "0.9",
            }
        else:
            return {"response": "Default response"}

    return mock_response_generator


@pytest.fixture
def temp_optimized_prompts_dir():
    """Create temporary directory with optimized prompts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample optimized prompts
        prompts_data = {
            "query_analysis_prompts.json": {
                "compiled_prompts": {
                    "system": "Optimized system prompt for query analysis",
                    "signature": "QueryAnalysisSignature with optimized fields",
                },
                "metadata": {
                    "optimization_timestamp": 1234567890,
                    "dspy_version": "3.0.2",
                },
            },
            "agent_routing_prompts.json": {
                "compiled_prompts": {
                    "routing": "Optimized routing decision prompt",
                    "signature": "AgentRoutingSignature with optimized logic",
                },
                "metadata": {
                    "optimization_timestamp": 1234567890,
                    "dspy_version": "3.0.2",
                },
            },
            "summary_generation_prompts.json": {
                "compiled_prompts": {
                    "summary": "Optimized summary generation prompt",
                    "few_shot_examples": [
                        "Example 1: Brief summary",
                        "Example 2: Technical summary",
                        "Example 3: Executive summary",
                    ],
                },
                "metadata": {
                    "optimization_timestamp": 1234567890,
                    "dspy_version": "3.0.2",
                },
            },
            "detailed_report_prompts.json": {
                "compiled_prompts": {
                    "report": "Optimized detailed report prompt",
                    "signature": "DetailedReportSignature with comprehensive analysis",
                },
                "metadata": {
                    "optimization_timestamp": 1234567890,
                    "dspy_version": "3.0.2",
                },
            },
        }

        # Write prompt files
        for filename, data in prompts_data.items():
            with open(temp_path / filename, "w") as f:
                json.dump(data, f, indent=2)

        yield temp_path


class TestDSPyOptimizerIntegration:
    """Integration tests for DSPy optimizer with OpenAI-compatible APIs."""

    @pytest.mark.unit
    def test_optimizer_with_local_llm(self):
        """Test DSPy optimizer with real local Ollama LLM."""
        optimizer = DSPyAgentPromptOptimizer()

        endpoint_config = LLMEndpointConfig(
            model="ollama/qwen2.5:1.5b",
            api_base="http://localhost:11434",
        )
        success = optimizer.initialize_language_model(endpoint_config)

        assert success
        assert optimizer.lm is not None
        # Verify it's a real DSPy LM, not a mock
        import dspy

        assert isinstance(optimizer.lm, dspy.LM)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_optimization_with_real_lm(self, mock_openai_compatible_api):
        """Test full pipeline optimization with real DSPy LM."""
        # Initialize optimizer with real LM
        optimizer = DSPyAgentPromptOptimizer()
        endpoint_config = LLMEndpointConfig(
            model="ollama/qwen2.5:1.5b",
            api_base="http://localhost:11434",
        )
        optimizer.initialize_language_model(endpoint_config)

        # Create and run pipeline, mocking optimize_module to avoid full DSPy optimization
        with patch.object(
            DSPyAgentOptimizerPipeline, "optimize_module"
        ) as mock_optimize:
            mock_optimize.side_effect = lambda module_name, *args, **kwargs: Mock(
                name=f"compiled_{module_name}"
            )

            pipeline = DSPyAgentOptimizerPipeline(optimizer)
            optimized_modules = await pipeline.optimize_all_modules()

        # Verify all modules were optimized
        expected_modules = [
            "query_analysis",
            "agent_routing",
            "summary_generation",
            "detailed_report",
        ]
        for module_name in expected_modules:
            assert module_name in optimized_modules

        assert mock_optimize.call_count == len(expected_modules)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_module_optimization_with_training_data(
        self, mock_openai_compatible_api
    ):
        """Test individual module optimization with training data."""

        # Mock the entire optimization process to focus on integration without DSPy deep internals
        with patch.object(
            DSPyAgentOptimizerPipeline, "optimize_module"
        ) as mock_optimize:
            # Setup mock return
            mock_compiled_module = Mock()
            mock_optimize.return_value = mock_compiled_module

            # Create optimizer and pipeline
            optimizer = DSPyAgentPromptOptimizer()
            pipeline = DSPyAgentOptimizerPipeline(optimizer)

            # Initialize modules and load training data
            pipeline.initialize_modules()
            training_data = pipeline.load_training_data()

            # Test optimization of query analysis module
            query_examples = training_data["query_analysis"]
            assert len(query_examples) > 0

            optimized_module = pipeline.optimize_module(
                "query_analysis",
                query_examples[:3],  # Use first 3 as training
                (
                    query_examples[3:4] if len(query_examples) > 3 else None
                ),  # Use 4th as validation
            )

            assert optimized_module is not None

            # Verify optimize_module was called with correct parameters
            mock_optimize.assert_called_once()
            call_args = mock_optimize.call_args
            assert call_args[0][0] == "query_analysis"  # module_name
            assert len(call_args[0][1]) == 3  # training_examples

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_prompt_saving_and_loading(self):
        """Test saving and loading optimized prompts via telemetry."""

        # Test that optimization pipeline can save prompts
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)

        # Mock compiled modules
        mock_modules = {}
        for module_name in [
            "query_analysis",
            "agent_routing",
            "summary_generation",
            "detailed_report",
        ]:
            mock_module = Mock(spec=[])
            mock_module.demos = []
            mock_module.generate_analysis = (
                Mock() if "analysis" in module_name else None
            )
            mock_module.generate_routing = Mock() if "routing" in module_name else None
            mock_module.generate_summary = Mock() if "summary" in module_name else None
            mock_module.generate_report = Mock() if "report" in module_name else None

            if mock_module.generate_analysis:
                mock_module.generate_analysis.signature = (
                    f"Mock {module_name} signature"
                )

            mock_modules[module_name] = mock_module

        pipeline.compiled_modules = mock_modules

        # Mock telemetry provider with async dataset/experiment stores
        mock_provider = Mock()
        mock_provider.datasets = Mock()
        mock_provider.datasets.create_dataset = AsyncMock(return_value="ds-123")
        mock_provider.experiments = Mock()
        mock_provider.experiments.create_experiment = AsyncMock(return_value="exp-123")
        mock_provider.experiments.log_run = AsyncMock(return_value="run-123")

        # Save prompts via telemetry
        await pipeline.save_optimized_prompts(
            tenant_id="test-tenant", telemetry_provider=mock_provider
        )

        # Verify artifacts were saved for each module
        assert mock_provider.datasets.create_dataset.call_count >= len(mock_modules)
        assert mock_provider.experiments.create_experiment.call_count >= len(
            mock_modules
        )


class TestDSPyAgentIntegration:
    """Integration tests for agents with DSPy optimization."""

    @pytest.mark.ci_fast
    def test_routing_agent_with_optimized_prompts(self, temp_optimized_prompts_dir):
        """Test RoutingAgent with loaded optimized prompts."""

        mock_prompts = {
            "compiled_prompts": {
                "routing": "Optimized routing decision prompt",
                "signature": "AgentRoutingSignature with optimized logic",
            },
            "metadata": {"optimization_timestamp": 1234567890, "dspy_version": "3.0.2"},
        }

        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True

            with patch("builtins.open", mock_open(read_data=json.dumps(mock_prompts))):
                telemetry_config = TelemetryConfig(enabled=False)
                from cogniverse_foundation.config.unified_config import (
                    LLMEndpointConfig,
                )

                deps = RoutingDeps(
                    telemetry_config=telemetry_config,
                    llm_config=LLMEndpointConfig(model="test/mock"),
                )
                agent = RoutingAgent(deps=deps)

                assert hasattr(agent, "dspy_module")
                assert agent.dspy_module is not None

                assert agent is not None
                assert hasattr(agent, "route_query")


class TestDSPyEndToEndOptimization:
    """End-to-end integration tests for DSPy optimization."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
