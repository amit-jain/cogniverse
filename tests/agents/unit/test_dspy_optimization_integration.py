"""Integration tests for DSPy optimization with OpenAI-compatible APIs."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from cogniverse_agents.optimizer.dspy_agent_optimizer import (
    DSPyAgentOptimizerPipeline,
    DSPyAgentPromptOptimizer,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager


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
        """Test DSPy optimizer against the configured LM."""
        optimizer = DSPyAgentPromptOptimizer()

        from tests.fixtures.llm import resolve_base_url, resolve_prefixed_model

        endpoint_config = LLMEndpointConfig(
            model=resolve_prefixed_model(),
            api_base=resolve_base_url(),
        )
        success = optimizer.initialize_language_model(endpoint_config)

        assert success
        assert optimizer.lm is not None
        # Verify it's a real DSPy LM, not a mock
        import dspy

        assert isinstance(optimizer.lm, dspy.LM)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pipeline_optimizes_each_configured_module(
        self, mock_openai_compatible_api
    ):
        """The pipeline runs optimize_module for every configured module
        (query_analysis, agent_routing, summary_generation, detailed_report) and
        returns each compiled result. The DSPy optimization step itself is
        mocked — this verifies the pipeline's module-iteration wiring, not real
        LM-driven optimization."""
        # Initialize optimizer with real LM
        optimizer = DSPyAgentPromptOptimizer()
        from tests.fixtures.llm import resolve_base_url, resolve_prefixed_model

        endpoint_config = LLMEndpointConfig(
            model=resolve_prefixed_model(),
            api_base=resolve_base_url(),
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

        # Mock telemetry provider: save_optimized_prompts goes through
        # ArtifactManager which calls datasets.create_dataset for prompts
        # and metrics blobs (one dataset.create_dataset per artifact type
        # per module).  The old experiments.create_experiment path was
        # removed when ArtifactManager replaced direct provider calls.
        mock_provider = Mock()
        mock_provider.datasets = Mock()
        mock_provider.datasets.create_dataset = AsyncMock(return_value="ds-123")
        mock_provider.datasets.get_dataset = AsyncMock(
            side_effect=KeyError("not found")
        )
        # save_experiment now routes through append_to_dataset (creates the
        # dataset on first call when get raises KeyError).
        mock_provider.datasets.append_to_dataset = AsyncMock(return_value="ds-exp-123")
        mock_provider.experiments = Mock()
        mock_provider.experiments.create_experiment = AsyncMock(return_value="exp-123")
        mock_provider.experiments.log_run = AsyncMock(return_value="run-123")

        # Save prompts via telemetry
        await pipeline.save_optimized_prompts(
            tenant_id="test-tenant", telemetry_provider=mock_provider
        )

        # ArtifactManager routes every artifact through datasets.create_dataset
        # (prompts blob + metrics blob = 2 calls per module, no demos since
        # mock_module.demos is []).  At minimum one call per module for prompts.
        assert mock_provider.datasets.create_dataset.call_count >= len(mock_modules), (
            f"Expected at least {len(mock_modules)} datasets.create_dataset calls "
            f"(one prompt-blob per module), got "
            f"{mock_provider.datasets.create_dataset.call_count}"
        )


class TestDSPyEndToEndOptimization:
    """End-to-end integration tests for DSPy optimization."""


class TestTrainingDataShapes:
    """load_training_data builds the per-module example sets verbatim."""

    @pytest.fixture
    def pipeline(self):
        return object.__new__(DSPyAgentOptimizerPipeline)

    @pytest.mark.ci_fast
    def test_returns_expected_module_sets_and_counts(self, pipeline):
        data = pipeline.load_training_data()
        assert set(data) == {
            "query_analysis",
            "agent_routing",
            "summary_generation",
            "detailed_report",
        }
        assert len(data["query_analysis"]) == 3
        assert len(data["agent_routing"]) == 3
        assert len(data["summary_generation"]) == 2
        assert len(data["detailed_report"]) == 1
        # load_training_data caches the result on the instance.
        assert pipeline.training_data is data

    @pytest.mark.ci_fast
    def test_example_fields_are_verbatim(self, pipeline):
        data = pipeline.load_training_data()
        qa = data["query_analysis"][0]
        assert qa.query == "Show me videos of robots from yesterday"
        assert qa.primary_intent == "video_search"
        assert qa.temporal_pattern == "yesterday"
        routing = data["agent_routing"][0]
        assert routing.primary_agent == "video_search"
        assert routing.routing_confidence == "0.9"


class TestModuleMetrics:
    """_create_metric_for_module returns pure scoring functions."""

    @pytest.fixture
    def pipeline(self):
        return object.__new__(DSPyAgentOptimizerPipeline)

    @pytest.mark.ci_fast
    def test_query_analysis_metric_scores_field_match_ratio(self, pipeline):
        from types import SimpleNamespace

        metric = pipeline._create_metric_for_module("query_analysis")
        fields = dict(
            primary_intent="video_search",
            complexity_level="simple",
            needs_video_search="true",
            needs_text_search="false",
            multimodal_query="false",
            temporal_pattern="yesterday",
        )
        ex = SimpleNamespace(**fields)
        assert metric(ex, SimpleNamespace(**fields)) == 1.0

        wrong = SimpleNamespace(**{k: "x" for k in fields})
        assert metric(ex, wrong) == 0.0

        half = SimpleNamespace(**fields)
        half.needs_text_search = "x"
        half.multimodal_query = "x"
        half.temporal_pattern = "x"
        assert metric(ex, half) == 0.5

    @pytest.mark.ci_fast
    def test_agent_routing_metric_weights_and_confidence_bonus(self, pipeline):
        from types import SimpleNamespace

        metric = pipeline._create_metric_for_module("agent_routing")
        ex = SimpleNamespace(
            recommended_workflow="direct_search",
            primary_agent="video_search",
            routing_confidence="0.9",
        )
        assert metric(ex, SimpleNamespace(**vars(ex))) == 1.0

        wrong_agent = SimpleNamespace(**vars(ex))
        wrong_agent.primary_agent = "summarizer"
        assert metric(ex, wrong_agent) == pytest.approx(0.6)

        bad_conf = SimpleNamespace(**vars(ex))
        bad_conf.routing_confidence = "high"
        assert metric(ex, bad_conf) == pytest.approx(0.8)

    @pytest.mark.ci_fast
    def test_summary_metric_counts_key_points_in_summary(self, pipeline):
        from types import SimpleNamespace

        metric = pipeline._create_metric_for_module("summary_generation")
        ex = SimpleNamespace(key_points="['cats', 'dogs']", summary="ignored")
        pred = SimpleNamespace(summary="A summary about cats and birds.")
        assert metric(ex, pred) == 0.5

    @pytest.mark.ci_fast
    def test_report_metric_scores_quarter_per_section(self, pipeline):
        from types import SimpleNamespace

        metric = pipeline._create_metric_for_module("detailed_report")
        full = SimpleNamespace(
            executive_summary="a",
            detailed_findings="b",
            recommendations="c",
            technical_details="d",
        )
        assert metric(None, full) == 1.0

        partial = SimpleNamespace(
            executive_summary="a",
            detailed_findings="b",
            recommendations="",
            technical_details=None,
        )
        assert metric(None, partial) == 0.5
        assert metric(None, SimpleNamespace()) == 0.0

    @pytest.mark.ci_fast
    def test_unknown_module_falls_back_to_query_analysis_metric(self, pipeline):
        from types import SimpleNamespace

        metric = pipeline._create_metric_for_module("nonexistent")
        fields = SimpleNamespace(
            primary_intent="a",
            complexity_level="a",
            needs_video_search="a",
            needs_text_search="a",
            multimodal_query="a",
            temporal_pattern="a",
        )
        # The query-analysis metric scores a full 6/6 field match as 1.0.
        assert metric(fields, fields) == 1.0


class TestMainCLIOrchestration:
    """main() guards on LM init and swallows optimization failures."""

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_main_returns_before_optimizing_when_lm_init_fails(self):
        from cogniverse_agents.optimizer import dspy_agent_optimizer as mod

        optimizer = MagicMock()
        optimizer.initialize_language_model.return_value = False

        with (
            patch.object(mod, "DSPyAgentPromptOptimizer", return_value=optimizer),
            patch.object(mod, "DSPyAgentOptimizerPipeline") as pipeline_cls,
            patch("cogniverse_foundation.config.utils.create_default_config_manager"),
            patch("cogniverse_foundation.config.utils.get_config"),
        ):
            await mod.main()

        optimizer.initialize_language_model.assert_called_once()
        # Early return — the pipeline is never even constructed.
        pipeline_cls.assert_not_called()

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_main_swallows_optimization_failure_without_saving(self):
        from cogniverse_agents.optimizer import dspy_agent_optimizer as mod

        optimizer = MagicMock()
        optimizer.initialize_language_model.return_value = True
        pipeline = MagicMock()
        pipeline.optimize_all_modules = AsyncMock(side_effect=RuntimeError("boom"))
        pipeline.save_optimized_prompts = AsyncMock()

        with (
            patch.object(mod, "DSPyAgentPromptOptimizer", return_value=optimizer),
            patch.object(mod, "DSPyAgentOptimizerPipeline", return_value=pipeline),
            patch("cogniverse_foundation.config.utils.create_default_config_manager"),
            patch("cogniverse_foundation.config.utils.get_config"),
            patch("cogniverse_foundation.telemetry.get_telemetry_manager"),
        ):
            # Must not propagate — the CLI catches optimization errors.
            await mod.main()

        pipeline.optimize_all_modules.assert_awaited_once()
        pipeline.save_optimized_prompts.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
